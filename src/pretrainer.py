import os
from time import time

import torch
from recbole.trainer import Trainer
from recbole.utils import get_gpu_usage, set_color
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm


class PretrainTrainer(Trainer):
    def __init__(self, config, model):
        super(PretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.config["pretrain_epochs"]
        self.save_step = self.config["save_step"]
        self.model = self.model.to(self.device)

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"], str(epoch_idx + 1)
                    ),
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = (
                    set_color("Saving current", "blue") + ": %s" % saved_model_file
                )
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result
