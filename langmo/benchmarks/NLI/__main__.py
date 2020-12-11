import yaml
import sys
import torch
# import vecto
# import vecto.embeddings
# import platform
import torch.nn.functional as F
from langmo.utils import get_unique_results_path
from .data import NLIDataModule
# from .model import Net
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
import transformers
from transformers import AutoModelForSequenceClassification
import horovod.torch as hvd
# from protonn.utils import describe_var
from protonn.utils import get_time_str
from transformers import logging as tr_logging
from transformers.optimization import get_linear_schedule_with_warmup
# import logging


class PLModel(pl.LightningModule):
    def __init__(self, net, params):
        super().__init__()
        self.net = net
        self.hparams = params
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

    def validation_epoch_end(self, outputs):
        metrics = {}
        if self.trainer.running_sanity_check:
            metrics["epoch"] = -1
        else:
            metrics["epoch"] = self.current_epoch
            # self.trainer.running_sanity_check = False
        for i, lst_split in enumerate(outputs):
            pref = self.ds_prefixes[i]
            loss = torch.stack([x['val_loss'] for x in lst_split]).mean()  # .item()
            # print("LOSSS IS", type(loss), loss)
            loss = hvd.allreduce(loss)
            acc = torch.stack([x['val_acc'] for x in lst_split]).mean()  # .item()
            acc = hvd.allreduce(acc)
            metrics[f"val_loss_{pref}"] = loss
            metrics[f"val_acc_{pref}"] = acc
            if self.hparams["test"] and i == 2:
                print(
                    f"worker {hvd.rank()} of {hvd.size()}\n"
                    f"\tvalidation end\n"
                    f"\tdl id is {i}, acc is {acc}"
                )
        # reduce
        # log only from worker 0
        if hvd.rank() == 0:
            # print("saving metrics", metrics)
            self.logger.log_metrics(metrics, step=self.global_step)
        # if self.trainer.running_sanity_check:
        #     self.trainer.current_epoch = -1
        #     self.trainer.running_sanity_check = False
        # self.log_dict(metrics, sync_dist=True, on_epoch=True)
        #if self.trainer.current_epoch == -1:
        #    self.trainer.current_epoch = 0
        #    self.trainer.running_sanity_check = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [param for param in self.net.parameters() if param.requires_grad], lr=0.00001
        )
        # steps = self.dataset_size / effective_batch_size) * self.hparams.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=50000
        )
        return [[optimizer], [scheduler]]
        # return torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)


def main():
    # path_data = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small"
    # path_data = "/groups1/gac50489/datasets/cosmoflow_full/cosmoUniverse_2019_05_4parE_tf"
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        return
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    path_results_base = "./out/NLI"
    params["path_results"] = get_unique_results_path(path_results_base)
    hvd.init()
    if hvd.rank() != 0:
        tr_logging.set_verbosity_error()
    timestamp = get_time_str()
    #model_name = "prajjwal1/bert-mini"
    #model_name = "bert-base-uncased"
    #model_name = "albert-base-v2"
    model_name = params["model_name"]
    name_run = f"{model_name}_{'↓' if params['uncase'] else '◯'}_{timestamp[:-3]}"
    wandb_logger = WandbLogger(project=f"NLI{'_test' if params['test'] else ''}",
                               name=name_run)
    # wandb_logger.log_hyperparams(config)
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=5,
    #     verbose=True,
    #     mode="min",
    # )
    # print("create tainer")
    # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])

    net = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # net = Net(embs)
    model = PLModel(net, params)
    if params["test"]:
        params["cnt_epochs"] = 3
    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=-1,
        max_epochs=params["cnt_epochs"],
        distributed_backend="horovod",
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=wandb_logger,
        progress_bar_refresh_rate=0)

    # wandb_logger.watch(net, log='gradients', log_freq=100)
    data_module = NLIDataModule(
        # embs.vocabulary,
        transformers.AutoTokenizer.from_pretrained(model_name),
        batch_size=params["batch_size"],
        params=params)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
