from collections import defaultdict
from pathlib import Path

import horovod.torch as hvd
# from .model import Net
import pytorch_lightning as pl
import torch
# import vecto
# import vecto.embeddings
import torch.nn.functional as F
import transformers
# from protonn.utils import describe_var
from protonn.utils import get_time_str, save_data_json
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
from transformers import AutoModelForSequenceClassification
from transformers import logging as tr_logging
from transformers.optimization import get_linear_schedule_with_warmup

# from langmo.checkpoint import CheckpointEveryNSteps
from langmo.nn.utils import reinit_model
from langmo.utils import load_config

from .data import NLIDataModule

# import logging


class PLModel(pl.LightningModule):
    def __init__(self, net, tokenizer, params):
        super().__init__()
        self.net = net
        self.tokenizer = tokenizer
        self.hparams = params
        self.hparams["train_logs"] = []
        # self.example_input_array = ((
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        # ))
        self.ds_prefixes = {0: "matched", 1: "mismatched", 2: "hans"}

    def forward(self, inputs):
        # print(describe_var(inputs))
        return self.net(*inputs)[0]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(logits, targets)
        metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
        # print(f"worker {hvd.rank()} of {hvd.size()} doing train batch {batch_idx} of size {logits.size()}")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        inputs, targets = batch
        logits = self(inputs)
        if dataloader_idx == 2:
            entail = logits[:, :1]
            non_entail = logits[:, 1:]
            non_entail = non_entail.max(axis=1).values
            logits = torch.cat((entail, non_entail.unsqueeze(1)), 1)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(logits, targets)
        if self.hparams["test"] and dataloader_idx == 2:
            print(
                f"worker {hvd.rank()} of {hvd.size()}\n"
                f"\tval batch {batch_idx} ({logits.size()}) of dloader {dataloader_idx}\n"
                f"\ttargets: {targets.sum()}, acc is {acc}"
            )
        metrics = {
            f"val_loss": loss,
            f"val_acc": acc,
        }
        # self.log_dict(metrics)
        return metrics

    def _update_hparams_train_logs(self, metrics):
        entry = dict(epoch=metrics["epoch"])
        for k, v in metrics.items():
            val = v.item() if hasattr(v, "item") else v
            entry[k] = val
        self.hparams["train_logs"].append(entry)

    def validation_epoch_end(self, outputs):
        metrics = {}
        if self.trainer.running_sanity_check:
            metrics["epoch"] = -1
        else:
            metrics["epoch"] = self.current_epoch
        for i, lst_split in enumerate(outputs):
            pref = self.ds_prefixes[i]
            loss = torch.stack([x["val_loss"] for x in lst_split]).mean()  # .item()
            # TODO: refactor this reduction an logging in one helper function
            loss = hvd.allreduce(loss)
            acc = torch.stack([x["val_acc"] for x in lst_split]).mean()  # .item()
            acc = hvd.allreduce(acc)
            metrics[f"val_loss_{pref}"] = loss
            metrics[f"val_acc_{pref}"] = acc
            if self.hparams["test"] and i == 2:
                print(
                    f"worker {hvd.rank()} of {hvd.size()}\n"
                    f"\tvalidation end\n"
                    f"\tdl id is {i}, acc is {acc}"
                )
        if hvd.rank() == 0:
            self.logger.log_metrics(metrics, step=self.global_step)
            self._update_hparams_train_logs(metrics)
            path = Path(self.hparams["path_results"]).joinpath("train_logs.json")
            save_data_json(self.hparams["train_logs"], path)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(
            [param for param in self.net.parameters() if param.requires_grad],
            lr=3e-5,
        )
        # steps = self.dataset_size / effective_batch_size) * self.hparams.max_epochs
        # TODO: get real number of steps here
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=50000
        )
        return [[optimizer], [scheduler]]
        # return torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)


def main():
    hvd.init()
    if hvd.rank() != 0:
        tr_logging.set_verbosity_error()  # to reduce warning of unused weights
    name_task = "NLI"
    params = load_config(name_task)
    timestamp = get_time_str()

    # wandb_logger.log_hyperparams(config)
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=5,
    #     verbose=True,
    #     mode="min",
    # )
    # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
    name_model = params["model_name"]
    net = AutoModelForSequenceClassification.from_pretrained(name_model, num_labels=3)
    name_run = name_model
    if params["randomize"]:
        reinit_model(net)
        name_run += "_RND"
    # net = Net(embs)
    name_run += f"_{'↓' if params['uncase'] else '◯'}_{timestamp[:-3]}"
    wandb_logger = WandbLogger(
        project=params["name_project"],
        name=name_run,
        save_dir=params["path_results"],
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(name_model)
    model = PLModel(net, tokenizer, params)
    # n_step = 1000 if not params["test"] else 4
    # on_n_step_callback = CheckpointEveryNSteps(n_step)
    if params["test"]:
        params["cnt_epochs"] = 3
    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=1,
        num_sanity_val_steps=-1,
        max_epochs=params["cnt_epochs"],
        distributed_backend="horovod",
        precision=params["precision"],
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        # we probably don't need to checkpoint eval - but can make this optional
        # callbacks=[on_n_step_callback],
        checkpoint_callback=False,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
    )

    # wandb_logger.watch(net, log='gradients', log_freq=100)
    data_module = NLIDataModule(
        # embs.vocabulary,
        tokenizer,
        batch_size=params["batch_size"],
        params=params,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
