# TODO: cnt workers should be put once to params instead of using hvd
import math
import os
from pathlib import Path

import lightning as pl
import torch
# from apex.optimizers import FusedLAMB
from protonn.utils import num_to_str_with_suffix, save_data_json
from torch.optim import AdamW

# from transformers.optimization import AdamW
from .utils.model_utils import zero_and_freeze_param_by_name


class PLBase(pl.LightningModule):
    def __init__(self, net=None, tokenizer=None, params=None):
        super().__init__()
        # these None-s are for loading from checkpoint
        if net is not None:
            self.net = net
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if params is not None:
            self.hparams.update(params)
        # TODO: check if this works on resume
        # self.hparams["cnt_samples_processed"] = 0
        # self.init_train_logs()

    def setup(self, stage):
        if self.global_rank == 0:
            os.makedirs(self.hparams["path_results"], exist_ok=True)
        self.logger.log_hyperparams(self.hparams)

        if "siamese" in self.hparams and self.hparams["siamese"]:
            return

        # set token_type_embeddings to zero and token_type_embeddings.requires_grad = False
        # if there is only one possible token_type_id
        try:
            if self.net.config.to_dict().get("type_vocab_size", 0) == 1:
                zero_and_freeze_param_by_name(self.net, "token_type_embeddings.weight")
        except:
            # TODO: proper warning with logger here
            print("SOMETHING WENT WRONG WITH WREEZE TOKEN TYPE EMBEDDINGS")

    def get_cnt_training_steps(self):
        params = self.hparams
        # batch_size = self.hparams["batch_size_effective"]
        if hasattr(self.trainer.datamodule, "cnt_train_samples"):
            self.hparams["cnt_samples_per_epoch"] = self.trainer.datamodule.cnt_train_samples
        samples_per_epoch = params["cnt_samples_per_epoch"]
        steps_total = 0
        schedule = [(0, 1)]
        for epoch in self.hparams["accumulate_batches"]:
            schedule.append((epoch, params["accumulate_batches"][epoch]))
        schedule.append((params["cnt_epochs"], schedule[-1][1]))
        for i in range(len(schedule) - 1):
            epochs_in_span = schedule[i + 1][0] - schedule[i][0]
            batch_in_span = params["batch_size"] * params["cnt_workers"] * schedule[i][1]
            steps_in_epoch = math.ceil(samples_per_epoch / batch_in_span)
            steps_total += steps_in_epoch * epochs_in_span
        return steps_total

    def configure_optimizers(self):
        # param_optimizer = list(self.net.named_parameters())
        param_optimizer = [param for param in self.net.named_parameters() if param[1].requires_grad]

        no_decay = self.hparams["params_without_weight_decay"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # no_decay = [n for n, p in param_optimizer if any(nd in n for nd in no_decay)]
        # print("no decay", no_decay)
        # print(optimizer_grouped_parameters)
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams["initial_lr"],
            eps=self.hparams["eps"],
            # TODO: double check if wd working when in grouped params
            # weight_decay=self.hparams["weight_decay"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
        )
        # optimizer = FusedLAMB(
        #     optimizer_grouped_parameters,
        #     lr=self.hparams["initial_lr"],
        #     eps=self.hparams["eps"],
        #     # weight_decay=self.hparams["weight_decay"],
        #     betas=(self.hparams["beta1"], self.hparams["beta2"]),
        # )
        # optimizer.clip_grad_norm(1.0)

        # print(f"!!!!!!!! expected steps: {training_steps}")
        # TODO: get rough estimation of training steps here
        # maybe after first epoch is trained - reset iterators?
        pct_start = self.hparams["percent_warmup"] / 100.0
        training_steps = self.get_cnt_training_steps()
        # print("setting training_steps as", training_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams["max_lr"],
            total_steps=training_steps,
            pct_start=pct_start,
            anneal_strategy="linear",
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [[optimizer], [scheduler]]

    def save_as_hf(self, path):
        self.net.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_metadata(self, path=None):
        if path is None:
            path = self.hparams["path_results"]
        path = Path(path) / "metadata.json"
        save_data_json(self.hparams, path)

    def _get_ckecpoint_folder(self):
        cnt_snapshost = len(self.hparams["train_logs"]) - 1
        dir_checkpoints = Path(self.hparams["path_results"]) / "checkpoints"
        n_smpl = num_to_str_with_suffix(self.hparams["cnt_samples_processed"])
        dir_current = f"snap_{cnt_snapshost:03d}_smpl_{n_smpl}"
        return dir_checkpoints / dir_current

    def init_train_logs(self):
        if "train_logs" not in self.hparams:
            self.hparams["train_logs"] = []
            self.hparams["cnt_samples_processed"] = 0
            self.hparams["train_logs"].append({})
            self.hparams["train_logs"][-1]["epoch"] = -1
            self.hparams["train_logs"][-1]["epoch_time"] = 0.0
            self.hparams["train_logs"][-1]["cnt_samples_processed"] = 0
