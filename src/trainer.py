import logging
import os
from dataclasses import dataclass

import mlflow
import torch
from distilled_data import DistilledData
from evaluator import Evaluator
from model import SASRec
from recbole.data.dataloader import FullSortEvalDataLoader, TrainDataLoader
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from transformers import get_scheduler
from utils import real_batch_on_device

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    skip_train: bool = False
    inner_loop: int = 50
    epoch: int = 30
    lr_inputs_embeds: float = 1.0e-2
    lr_attention_labels: float = 1.0e-2
    lr_lr: float = 1.0e-2
    optimizer_type: str = "adamw"  # ["sgd", "adam"]
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 2.0
    val_interval: int = 1
    log_interval: int = -1  # if -1 -> len(dataloader)//10
    n_eval_model: int = 5
    save_ckpt_dir: str = "path/to/checkpoint_dir"
    fp16: bool = False
    bf16: bool = False


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def fit(
        self,
        distilled_data: DistilledData,
        model: SASRec,
        train_loader: TrainDataLoader,
        valid_loader: FullSortEvalDataLoader,
        evaluator: Evaluator,
    ):
        model.cuda()
        distilled_data.cuda()

        max_training_steps = self.config.epoch * len(train_loader)

        logger.info(f"Train loader length: (`{len(train_loader)}`)")
        logger.info(f"Valid loader length: (`{len(valid_loader)}`)")

        if self.config.log_interval == -1:
            self.config.log_interval = len(train_loader) // 2

        optimizer, scheduler = self.configure_optimizer(
            distilled_data, max_training_steps=max_training_steps
        )
        scaler = amp.GradScaler(enabled=self.use_amp)

        """ 
        Evaluate before training:
        train on distilled-data and evaluate on real-valid-data.
        """
        results = evaluator.evaluate_fast(
            distilled_data, valid_loader, n_eval_model=self.config.n_eval_model
        )
        mlflow.log_metrics(results, step=0)
        logger.info(
            "Validation [{:>{}}/{}]: {}".format(
                0, len(str(self.config.epoch)), self.config.epoch, results
            )
        )
        best_ckpt_path = os.path.join(self.config.save_ckpt_dir, "best-ckpt")
        distilled_data.save_pretrained(best_ckpt_path)
        mlflow.log_artifact(best_ckpt_path)

        best_val_loss = results["loss"]

        """
        Start training:
        """
        logger.info("Start training !!!")
        for i in range(self.config.epoch):
            log_train_loss = 0
            with tqdm(
                train_loader,
                dynamic_ncols=True,
                leave=False,
                desc=f"Train synthetic data (Epoch[{i+1:>2}/{self.config.epoch}])",
            ) as pbar:
                for outer_step, batch_real in enumerate(pbar):
                    # initialize model
                    model.train()
                    model.apply(model._init_weights)

                    params = dict(model.named_parameters())
                    buffers = dict(model.named_buffers())

                    def compute_loss(
                        params,
                        buffers,
                        interaction=None,
                        attention_labels=None,
                        **kwargs,
                    ):
                        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            outputs, attentions = torch.func.functional_call(
                                model, (params, buffers), args=interaction
                            )

                        loss_task = outputs

                        if attention_labels is not None:
                            attn_weights = attentions
                            attn_weights = attn_weights[
                                ..., : attention_labels.size(-2), :
                            ]
                            assert attn_weights.shape == attention_labels.shape
                            loss_attn = F.kl_div(
                                torch.log(attn_weights + 1e-12),
                                attention_labels,
                                reduction="none",
                            )
                            loss_attn = loss_attn.sum(-1).mean()
                        else:
                            loss_attn = 0.0

                        return (
                            loss_task + distilled_data.attention_loss_lambda * loss_attn
                        )

                    # m = {
                    #     name: torch.zeros_like(p) for name, p in params.items()
                    # }  # First moment estimates
                    # v = {
                    #     name: torch.zeros_like(p) for name, p in params.items()
                    # }  # Second moment estimates
                    # t = 0  # Time step

                    for inner_step in range(self.config.inner_loop):
                        batch_syn = distilled_data.get_batch(inner_step)

                        inputs_embeds = batch_syn.pop("inputs_embeds")

                        syn_lr = batch_syn.pop("lr")

                        # update model on distilled data
                        grads = torch.func.grad(compute_loss)(
                            params, buffers, interaction=inputs_embeds, **batch_syn
                        )

                        # t += 1  # Increment time step
                        # for name in params.keys():
                        #     # mt = beta1 * mt-1 + (1 - beta1) * gt
                        #     m[name] = (
                        #         self.config.beta1 * m[name]
                        #         + (1 - self.config.beta1) * grads[name]
                        #     )
                        #     # vt = beta2 * vt-1 + (1 - beta2) * g2t
                        #     v[name] = self.config.beta2 * v[name] + (
                        #         1 - self.config.beta2
                        #     ) * (grads[name] ** 2)
                        #     # m_hat = mt / (1 - beta1^t)
                        #     m_hat = m[name] / (1 - self.config.beta1**t + 1e-8)
                        #     # v_hat = vt / (1 - beta2^t)
                        #     v_hat = v[name] / (1 - self.config.beta2**t + 1e-8)
                        #     # wt = wt-1 - alpha * m_hat / (sqrt(v_hat) + epsilon)
                        #     params[name] = params[name] - syn_lr * m_hat / (
                        #         torch.sqrt(v_hat) + 1e-8
                        #     )
                        params = {
                            name: p - syn_lr * grads[name] for name, p in params.items()
                        }

                    # evaluate updated model on real data
                    batch_real = real_batch_on_device(batch_real)
                    loss_real = compute_loss(params, buffers, interaction=batch_real)

                    # compute gradient
                    optimizer.zero_grad()
                    scaler.scale(loss_real).backward()

                    # gradient clipping
                    if self.config.max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            distilled_data.data_dict().values(),
                            max_norm=self.config.max_grad_norm,
                        )

                    # update distilled data
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # detach params
                    for name, param in params.items():
                        params[name] = param.detach().requires_grad_()

                    log_train_loss += loss_real.item()

                    pbar.set_postfix({"train_loss": loss_real.item()})

                    if (outer_step + 1) % self.config.log_interval == 0:
                        log_train_loss /= self.config.log_interval

                        mlflow.log_metric(
                            "train_loss",
                            log_train_loss,
                            step=len(train_loader) * i + outer_step,
                        )
                        mlflow.log_metrics(
                            {
                                f"lr.{i}": distilled_data.lr[i].item()
                                for i in range(self.config.inner_loop)
                            },
                            step=len(train_loader) * i + outer_step,
                        )
                        mlflow.log_metric(
                            "optimizer_lr",
                            scheduler.get_last_lr()[0],
                            step=len(train_loader) * i + outer_step,
                        )
                        logger.info(
                            "TRAIN (Epoch[{:>4.1f}]): train_loss={}".format(
                                (outer_step + 1) / len(train_loader) + i,
                                log_train_loss,
                            )
                        )
                        log_train_loss = 0

            if (i + 1) % self.config.val_interval == 0:
                results = evaluator.evaluate_fast(
                    distilled_data, valid_loader, n_eval_model=self.config.n_eval_model
                )
                mlflow.log_metrics(results, step=len(train_loader) * (i + 1))
                logger.info(
                    "VALIDATION (Epoch[{:>2}/{}]): {}".format(
                        i + 1, self.config.epoch, results
                    )
                )

                if results["loss"] < best_val_loss:
                    best_val_loss = results["loss"]
                    distilled_data.save_pretrained(best_ckpt_path)
                    mlflow.log_artifact(best_ckpt_path)
                    logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

        logger.info("Finish training!!")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_ckpt_dir, "last-ckpt")
        distilled_data.save_pretrained(last_ckpt_path)
        mlflow.log_artifact(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # load best checkpoint
        best_checkpoint = torch.load(os.path.join(best_ckpt_path, "data_dict"))
        distilled_data.load_data_dict(best_checkpoint)

    def configure_optimizer(
        self,
        distilled_data: DistilledData,
        max_training_steps: int,
    ) -> tuple[Optimizer, _LRScheduler]:
        optimizer_class = {"sgd": SGD, "momentum": SGD, "adam": Adam, "adamw": AdamW}
        assert self.config.optimizer_type in optimizer_class

        data_dict = distilled_data.data_dict()
        assert data_dict.keys() >= {
            "inputs_embeds",
            "lr",
        }, f"{data_dict.keys()}"
        grouped_params = [
            {
                "params": data_dict["inputs_embeds"],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr_inputs_embeds,
            },
            {
                "params": data_dict["lr"],
                "lr": self.config.lr_lr,
            },
        ]
        if "attention_labels" in data_dict:
            grouped_params.append(
                {
                    "params": data_dict["attention_labels"],
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.lr_attention_labels,
                }
            )

        optimizer = optimizer_class[self.config.optimizer_type](
            grouped_params, lr=1.0
        )  # `lr=1.0` is not used (dummy)
        logger.info(f"Optimizer: {optimizer}")
        # learning rate scheduler
        scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=optimizer if optimizer is not None else optimizer,
            num_warmup_steps=max_training_steps * self.config.warmup_ratio,
            num_training_steps=max_training_steps,
        )

        return optimizer, scheduler

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
