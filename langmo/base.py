# TODO: cnt workers should be put once to params instead of using hvd
import horovod.torch as hvd
import pytorch_lightning as pl
import torch
from protonn.utils import save_data_json
from pathlib import Path
import transformers


class PLBase(pl.LightningModule):
    def __init__(self, net, tokenizer, params):
        super().__init__()
        self.net = net
        self.tokenizer = tokenizer
        params["cnt_workers"] = hvd.size()
        params["train_logs"] = []
        params["batch_size_effective"] = params["batch_size"] * params["cnt_workers"]
        self.save_hyperparameters(params)
        self.hparams.update(params)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(
            [param for param in self.net.parameters() if param.requires_grad],
            lr=self.hparams["initial_lr"],
            eps=self.hparams["eps"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
        )
        # optimizer.clip_grad_norm(1.0)
        cnt_epochs = self.hparams["cnt_epochs"]
        batch_size = self.hparams["batch_size"]
        if hasattr(self.trainer.datamodule, "cnt_train_samples"):
            self.hparams["cnt_train_samples"] = self.trainer.datamodule.cnt_train_samples
            num_samples = self.hparams["cnt_train_samples"]
            training_steps = int((10 + num_samples / batch_size) * cnt_epochs / hvd.size())
        else:
            training_steps = self.hparams["cnt_training_steps"]

        # TODO: get rough estimation of training steps here
        # maybe after first epoch is trained - reset iterators?
        pct_start = self.hparams["cnt_warmup_steps"] / training_steps
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

    def add_epoch_id_to_metrics(self, metrics):
        if self.trainer.running_sanity_check:
            metrics["epoch"] = -1
        else:
            metrics["epoch"] = self.current_epoch

    def append_metrics_to_train_logs(self, metrics):
        entry = dict(epoch=metrics["epoch"])
        for k, v in metrics.items():
            val = v.item() if hasattr(v, "item") else v
            entry[k] = val
        self.hparams["train_logs"].append(entry)
