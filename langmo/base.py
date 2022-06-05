# TODO: cnt workers should be put once to params instead of using hvd
import os
from pathlib import Path

import pytorch_lightning as pl
import torch

# from apex.optimizers import FusedLAMB
from protonn.utils import save_data_json
from torch.optim import AdamW

# from transformers.optimization import AdamW
from .utils.model_utils import zero_param_and_grad_with_string

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

    def setup(self, stage):
        if self.global_rank == 0:
            os.makedirs(self.hparams["path_results"], exist_ok=True)
        self.hparams["cnt_workers"] = self.trainer.world_size
        self.hparams["batch_size_effective"] = (
            self.hparams["batch_size"]
            * self.hparams["cnt_workers"]
            * self.hparams["accumulate_batches"]
        )
        self.logger.log_hyperparams(self.hparams)

        # set token_type_embeddings to zero and token_type_embeddings.requires_grad = False
        # if there is only one possible token_type_id
        if self.net.config.to_dict().get("type_vocab_size", 0) == 1:
            zero_param_and_grad_with_string(self.net, "token_type_embeddings.weight")

    def configure_optimizers(self):
        # param_optimizer = list(self.net.named_parameters())
        param_optimizer = [
            param for param in self.net.named_parameters() if param[1].requires_grad
        ]

        no_decay = ["bias", "gamma", "beta", "LayerNorm", "layer_norm"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
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
        cnt_epochs = self.hparams["cnt_epochs"]
        batch_size = self.hparams["batch_size_effective"]
        print("BATCH EFFECTIVE", batch_size)
        if hasattr(self.trainer.datamodule, "cnt_train_samples"):
            self.hparams[
                "cnt_samples_per_epoch"
            ] = self.trainer.datamodule.cnt_train_samples
        samples_per_epoch = self.hparams["cnt_samples_per_epoch"]
        # print(f"!!!!!!!! samples per epoch: {samples_per_epoch}")
        training_steps = (
            int((batch_size + samples_per_epoch) * cnt_epochs / batch_size) + 1
        )
        # print(f"!!!!!!!! expected steps: {training_steps}")
        # TODO: get rough estimation of training steps here
        # maybe after first epoch is trained - reset iterators?
        pct_start = self.hparams["percent_warmup"] / 100.0
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

    def save_metrics_and_model(self, metrics):
        if self.global_rank == 0:
            print("TODO: save_metrics_and_model shoudl be done in callback")
            self.logger.log_metrics(metrics, step=self.global_step)
            # self.append_metrics_to_train_logs(metrics)
            self.save_metadata()
            if metrics["epoch"] >= 0:
                path_hf = Path(self.hparams["path_results"]) / f"ep{metrics['epoch']}"
                self.save_as_hf(path_hf)
