from pathlib import Path

import horovod.torch as hvd
from .model import Siamese, TopMLP2, get_hidden_size, BertWithLSTM
import pytorch_lightning as pl
import torch
# import vecto
# import vecto.embeddings
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
import transformers
from protonn.utils import describe_var, get_time_str, save_data_json
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics.functional import accuracy
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import logging as tr_logging

from langmo.base import PLBase
# from langmo.checkpoint import CheckpointEveryNSteps
from langmo.nn.utils import reinit_model
from langmo.utils import load_config

from .data import NLIDataModule


class PLModel(PLBase):
    def __init__(self, net, tokenizer, params):
        super().__init__(net, tokenizer, params)

        # self.example_input_array = ((
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        # ))
        self.ds_prefixes = {0: "matched", 1: "mismatched", 2: "hans"}

    def forward(self, inputs):
        # print(inputs)
        # exit(1)
        return self.net(**inputs)["logits"]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0]
        # this is to fix PL 1.2+ thinking that top level list is multiple iterators
        # should be address by returning proper dataloader
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(torch.nn.functional.softmax(logits, dim=1), targets)
        # acc = accuracy(logits, targets)
        metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        print("got val batch\n" + describe_var(batch))
        inputs, targets = batch
        logits = self(inputs)
        if dataloader_idx == 2:
            entail = logits[:, :1]
            non_entail = logits[:, 1:]
            non_entail = non_entail.max(axis=1).values
            logits = torch.cat((entail, non_entail.unsqueeze(1)), 1)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(torch.nn.functional.softmax(logits, dim=1), targets)
        cnt_correct = (torch.argmax(logits, axis=1) == targets).sum()
        # if self.hparams["test"] and dataloader_idx == 2:
        #     print(
        #         f"worker {hvd.rank()} of {hvd.size()}\n"
        #         f"\tval batch {batch_idx} ({logits.size()}) of dloader {dataloader_idx}\n"
        #         f"\ttargets: {targets.sum()}, acc is {acc}"
        #     )
        metrics = {
            f"val_loss": loss,
            f"val_acc": acc,
            f"cnt_correct": cnt_correct,
            f"cnt_questions": torch.tensor(targets.shape[0]),
        }
        if dataloader_idx == 2:
            metrics["cnt_correct_entail"] = torch.logical_and((torch.argmax(logits, axis=1) == 0), (targets == 0)).sum()
            metrics["cnt_questions_entail"] = (targets == 0).sum()
            metrics["cnt_correct_nonentail"] = torch.logical_and((torch.argmax(logits, axis=1) == 1), (targets == 1)).sum()
            metrics["cnt_questions_nonentail"] = (targets == 0).sum()
        # print(f"targets", targets)
        # print(f"logits", logits)
        # print(f"logging cnt_correct as {cnt_correct} of {targets.shape}")
        # self.log_dict(metrics)
        return metrics

    def append_metrics_to_train_logs(self, metrics):
        entry = dict(epoch=metrics["epoch"])
        for k, v in metrics.items():
            val = v.item() if hasattr(v, "item") else v
            entry[k] = val
        self.hparams["train_logs"].append(entry)

    def save_metadata(self):
        path = Path(self.hparams["path_results"]) / "metadata.json"
        save_data_json(self.hparams, path)

    def validation_epoch_end(self, outputs):
        metrics = {}
        if self.trainer.running_sanity_check:
            metrics["epoch"] = -1
        else:
            metrics["epoch"] = self.current_epoch
        for id_dataloader, lst_split in enumerate(outputs):
            name_dataset = self.ds_prefixes[id_dataloader]
            loss = torch.stack([x["val_loss"] for x in lst_split]).mean()  # .item()
            # TODO: refactor this reduction an logging in one helper function
            loss = hvd.allreduce(loss)
            acc = torch.stack([x["val_acc"] for x in lst_split]).mean()  # .item()
            acc = hvd.allreduce(acc)
            for category in ["", "_entail", "_nonentail"]:
                cnt_correct = aggregate_batch_stats(lst_split, f"cnt_correct{category}")
                cnt_questions = aggregate_batch_stats(lst_split, f"cnt_questions{category}")
                if cnt_questions > 0:
                    metrics[f"cnt_correct_{name_dataset}{category}"] = cnt_correct
                    metrics[f"val_acc_{name_dataset}{category}"] = cnt_correct / cnt_questions

            # cnt_correct = aggregate_batch_stats(lst_split, "cnt_correct")
            metrics[f"val_loss_{name_dataset}"] = loss
            # metrics[f"val_acc_{pref}_btch"] = acc
            #if id_dataloader == 2:
             #   metrics[f"cnt"]
            # if self.hparams["test"] and id_dataloader == 2:
            #     print(
            #         f"worker {hvd.rank()} of {hvd.size()}\n"
            #         f"\tvalidation end\n"
            #         f"\tdl id is {id_dataloader}, acc is {acc}"
            #     )
        if hvd.rank() == 0:
            self.logger.log_metrics(metrics, step=self.global_step)
            self.append_metrics_to_train_logs(metrics)
            self.save_metadata()
            if metrics["epoch"] >= 0:
                path_hf = Path(self.hparams["path_results"]) / f"ep{metrics['epoch']}"
                self.save_as_hf(path_hf)


def aggregate_batch_stats(batch_stats, key):
    if key in batch_stats[0]:
        value = torch.stack([x[key] for x in batch_stats]).sum()
    else:
        value = torch.tensor(0)
    # print("reducing", key, value)
    value = hvd.allreduce(value, average=False)
    return value


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
    name_model = params["model_name"]
    # params["siamese"] = True
    # params["freeze_encoder"] = True
    if params["siamese"]:
        name_run = f"siam_rnn_{'fr_' if params['freeze_encoder'] else ''}" + name_model
        encoder = AutoModel.from_pretrained(name_model, num_labels=3)
        hidden_size = get_hidden_size(encoder)
        encoder = BertWithLSTM(encoder, freeze=params["freeze_encoder"])
        net = Siamese(encoder, TopMLP2(in_size=hidden_size * 8))
        # print("$$$$$$ CREATING SIAMESE")
    else:
        net = AutoModelForSequenceClassification.from_pretrained(name_model, num_labels=3)
        name_run = name_model

    # wandb_logger.watch(net, log='gradients', log_freq=100)
    # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
    # bottom = AutoModel.from_pretrained(name_model)
    # net = Siamese(bottom, TopMLP2())
    if params["randomize"]:
        reinit_model(net)
        name_run += "_RND"
    name_run += f"_{'↓' if params['uncase'] else '◯'}_{timestamp[:-3]}"
    wandb_logger = WandbLogger(
        project=params["name_project"],
        name=name_run,
        save_dir=params["path_results"],
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(name_model)
    # n_step = 4000 if not params["test"] else 4
    # on_n_step_callback = CheckpointEveryNSteps(n_step)
    if params["test"]:
        params["cnt_epochs"] = 3

    data_module = NLIDataModule(
        # embs.vocabulary,
        tokenizer,
        batch_size=params["batch_size"],
        params=params,
    )

    model = PLModel(net, tokenizer, params)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=1,
        num_sanity_val_steps=-1,
        # num_sanity_val_steps=0,
        max_epochs=params["cnt_epochs"],
        distributed_backend="horovod",
        precision=params["precision"],
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        # we probably don't need to checkpoint eval - but can make this optional
        callbacks=[lr_monitor],  # on_n_step_callback
        checkpoint_callback=False,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
        gradient_clip_val=0.5,
        track_grad_norm=2,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
