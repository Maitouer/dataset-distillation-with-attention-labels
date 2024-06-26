import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from distilled_data import DistilledData
from model import SASRec
from recbole.config import Config
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Collector
from recbole.evaluator import Evaluator as RecboleEvaluator
from torch.cuda import amp
from tqdm import tqdm, trange
from utils import average, real_batch_on_device

logger = logging.getLogger(__name__)


@dataclass
class EvaluateConfig:
    task_name: str
    n_eval_model: int = 100
    fp16: bool = False
    bf16: bool = False

    def __post_init__(self):
        assert not (self.fp16 and self.bf16)


class Evaluator:
    def __init__(self, config: EvaluateConfig, recbole_config: Config, model: SASRec):
        self.config = config
        self.recbole_config = recbole_config
        self.model = model
        self.eval_collector = Collector(self.recbole_config)
        self.recbole_evaluator = RecboleEvaluator(self.recbole_config)

    def evaluate(
        self,
        distilled_data: DistilledData,
        eval_loader: FullSortEvalDataLoader,
        n_eval_model: Optional[int] = None,
        verbose: bool = False,
    ) -> dict[str, tuple[float]]:
        self.model.cuda()
        distilled_data.cuda()
        if n_eval_model is None:
            n_eval_model = self.config.n_eval_model

        all_results = []
        for i in trange(n_eval_model, dynamic_ncols=True, leave=False, desc="Evaluate"):
            # train model on distilled data
            self.model.apply(self.model._init_weights)
            self.train_model(self.model, distilled_data)

            # evaluate trained model
            results = self.evaluate_model(self.model, eval_loader)
            if verbose:
                logger.info(
                    "[{:>{}}/{}]: {}".format(
                        i,
                        len(str(self.config.n_eval_model)),
                        self.config.n_eval_model,
                        results,
                    )
                )

            all_results.append(results)

        average_results = average(all_results, std=True)
        avg = {k: v[0] for k, v in average_results.items()}
        if verbose:
            logger.info(f"Average results: {avg}")

        return average_results

    def train_model(self, model: SASRec, distilled_data: DistilledData):
        model.train()
        train_config = distilled_data.train_config

        for step in trange(
            train_config.train_step,
            leave=False,
            dynamic_ncols=True,
            desc="Train model",
        ):
            batch = distilled_data.get_batch(step)

            # compute loss
            outputs, attentions = model(batch["inputs_embeds"])
            loss_task = outputs
            attention_labels = batch["attention_labels"]
            if attention_labels is not None:
                attn_weights = attentions
                attn_weights = attn_weights[..., : attention_labels.size(-2), :]
                assert attn_weights.shape == attention_labels.shape
                loss_attn = F.kl_div(
                    torch.log(attn_weights + 1e-12),
                    attention_labels,
                    reduction="none",
                )
                loss_attn = loss_attn.sum(-1).mean()
            else:
                loss_attn = 0.0

            loss = loss_task + distilled_data.attention_loss_lambda * loss_attn

            # update model
            model.zero_grad()
            loss.backward()
            for params in model.parameters():
                if params.grad is not None:
                    with torch.no_grad():
                        params.sub_(batch["lr"] * params.grad)

    def evaluate_model(
        self, model: SASRec, data_loader: FullSortEvalDataLoader
    ) -> dict[str, float]:
        """Evaluate on synthetic-valid-data"""
        model.eval()

        total_loss, num_samples = 0, 0
        for i, batch in enumerate(
            tqdm(data_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner")
        ):
            # evaluate
            model.eval()

            interaction, history_index, positive_u, positive_i = batch
            interaction = real_batch_on_device(interaction)

            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs, _ = model(interaction)
                    scores = model.full_sort_predict(interaction)
            # Calculate loss
            total_loss += outputs.item()
            num_samples += len(interaction)
            # Calculate metrics
            tot_item_num = data_loader._dataset.item_num
            scores = scores.view(-1, tot_item_num)
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        self.eval_collector.model_collect(model)
        struct = self.eval_collector.get_data_struct()
        result = self.recbole_evaluator.evaluate(struct)
        result["loss"] = total_loss / num_samples

        dict_result = dict(result)
        dict_result = {k.replace("@", "_at_"): float(v) for k, v in dict_result.items()}

        return dict_result

    def evaluate_fast(
        self,
        distilled_data: DistilledData,
        eval_loader: FullSortEvalDataLoader,
        n_eval_model: Optional[int] = None,
    ) -> dict[str, float]:
        """Evaluate on real-valid-data"""
        model = self.model.cuda()
        distilled_data.cuda()

        if n_eval_model is None:
            n_eval_model = self.config.n_eval_model

        reset_model_interval = max(len(eval_loader) // n_eval_model, 1)

        total_loss, num_samples = 0, 0
        for i, batch in enumerate(
            tqdm(eval_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner")
        ):
            if i % reset_model_interval == 0:
                # train model
                model.apply(model._init_weights)
                self.train_model(model, distilled_data)

            # evaluate
            model.eval()

            interaction, history_index, positive_u, positive_i = batch
            interaction = real_batch_on_device(interaction)

            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs, _ = model(interaction)
                    scores = model.full_sort_predict(interaction)
            # Calculate loss
            total_loss += outputs.item()
            num_samples += len(interaction)
            # Calculate metrics
            tot_item_num = eval_loader._dataset.item_num
            scores = scores.view(-1, tot_item_num)
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        self.eval_collector.model_collect(model)
        struct = self.eval_collector.get_data_struct()
        result = self.recbole_evaluator.evaluate(struct)
        result["loss"] = total_loss / num_samples

        dict_result = dict(result)
        dict_result = {k.replace("@", "_at_"): float(v) for k, v in dict_result.items()}

        return dict_result

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
