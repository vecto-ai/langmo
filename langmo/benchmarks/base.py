from pathlib import Path

import horovod.torch as hvd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from protonn.utils import get_time_str
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)
from transformers import logging as tr_logging

from langmo.base import PLBase
from langmo.benchmarks.NLI.model import (BertWithCLS, BertWithLSTM, Siamese,
                                         TopMLP2)
from langmo.nn.utils import reinit_model, reinit_tensor
from langmo.utils import load_config


class BaseClassificationModel(PLBase):
    def forward(self, inputs):
        return self.net(**inputs)["logits"]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0]
        # 0 is there seince PL returns tuple of batched from all dataloaders
        # not sure if this will be persisten behavior
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(F.softmax(logits, dim=1), targets)
        metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def save_metrics_and_model(self, metrics):
        if hvd.rank() == 0:
            self.logger.log_metrics(metrics, step=self.global_step)
            self.append_metrics_to_train_logs(metrics)
            self.save_metadata()
            if metrics["epoch"] >= 0:
                path_hf = Path(self.hparams["path_results"]) / f"ep{metrics['epoch']}"
                self.save_as_hf(path_hf)


class BaseFinetuner:
    def __init__(self, name_task, class_data_module, class_model):
        # TODO: refactor this into sub-methods
        hvd.init()
        if hvd.rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
        self.params = load_config(name_task)
        timestamp = get_time_str()
        name_model = self.params["model_name"]
        if self.params["siamese"]:
            name_run = "siam_" + self.params["encoder_wrapper"] + "_"
            if self.params["freeze_encoder"]:
                name_run += "fr_"
            name_run += name_model
            encoder = AutoModel.from_pretrained(name_model, num_labels=3)
            encoder = BertWithCLS(encoder, freeze=self.params["freeze_encoder"])
            net = Siamese(encoder, TopMLP2(in_size=encoder.get_output_size() * 4))
        else:
            net = AutoModelForSequenceClassification.from_pretrained(
                name_model, num_labels=3
            )
            name_run = name_model.split("pretrain")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)

        # wandb_logger.watch(net, log='gradients', log_freq=100)
        # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
        # bottom = AutoModel.from_pretrained(name_model)
        # net = Siamese(bottom, TopMLP2())
        if self.params["randomize"]:
            reinit_model(net)
            name_run += "_RND"
        name_run += f"_{'↓' if self.params['uncase'] else '◯'}_{timestamp[:-3]}"
        if "suffix" in self.params:
            name_wandb_project = (
                self.params["name_project"] + f"_{self.params['suffix']}"
            )
        else:
            name_wandb_project = self.params["name_project"]
        self.wandb_logger = WandbLogger(
            project=name_wandb_project,
            name=name_run,
            save_dir=self.params["path_results"],
        )
        self.net = net

        self.randomize_tokens()

        if self.params["test"]:
            self.params["cnt_epochs"] = 3
        self.data_module = class_data_module(
            # embs.vocabulary,
            self.tokenizer,
            params=self.params,
        )
        self.model = class_model(self.net, self.tokenizer, self.params)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        self.trainer = pl.Trainer(
            default_root_dir=self.params["path_results"],
            weights_save_path=self.params["path_results"],
            gpus=1,
            num_sanity_val_steps=-1,
            # num_sanity_val_steps=0,
            max_epochs=self.params["cnt_epochs"],
            distributed_backend="horovod",
            precision=self.params["precision"],
            replace_sampler_ddp=False,
            # early_stop_callback=early_stop_callback,
            # we probably don't need to checkpoint eval - but can make this optional
            callbacks=[lr_monitor],  # on_n_step_callback
            checkpoint_callback=False,
            logger=self.wandb_logger,
            progress_bar_refresh_rate=0,
            gradient_clip_val=0.5,
            track_grad_norm=2,
        )

    def randomize_tokens(self):
        if "rand_tok" in self.params:
            rand_tok = self.params["rand_tok"]
            id_dict = {
                "cls": self.tokenizer.cls_token_id,
                "sep": self.tokenizer.sep_token_id,
            }
            for tok in rand_tok:
                tok_id = id_dict[tok]
                # with torch.no_grad:
                tok_emb = self.net.get_input_embeddings().weight[tok_id]
                reinit_tensor(tok_emb)

    def run(self):
        self.trainer.fit(self.model, self.data_module)


def aggregate_batch_stats(batch_stats, key):
    if key in batch_stats[0]:
        value = torch.stack([x[key] for x in batch_stats]).sum()
    else:
        value = torch.tensor(0)
    # print("reducing", key, value)
    value = hvd.allreduce(value, average=False)
    return value
