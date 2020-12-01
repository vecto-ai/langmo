import yaml
import sys
import torch
import vecto
import vecto.embeddings
import torch.nn.functional as F
from langmo.utils import get_unique_results_path
from .data import NLIDataModule
from .model import Net
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
from transformers import BertModel
import horovod.torch as hvd


class PLModel(pl.LightningModule):
    def __init__(self, net, params):
        super().__init__()
        self.net = net
        self.params = params
        # self.example_input_array = ((
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        # ))

    def forward(self, inputs):
        # print(s1.shape)
        return self.net(inputs)

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
        # print(f"worker {hvd.rank()} of {hvd.size()} doing train batch {batch_idx} of size {s1.size()}")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        ds_prefixes = {0: "matched", 1: "mismatched", 2: "hans"}
        pref = ds_prefixes[dataloader_idx]
        inputs, targets = batch
        if self.params["test"]:
            print(
                f"worker {hvd.rank()} of {hvd.size()} doing val batch {batch_idx} of dataloader {dataloader_idx}, {pref}"
            )
        logits = self(inputs)
        if dataloader_idx == 2:
            entail = logits[:, :1]
            non_entail = logits[:, 1:]
            non_entail = non_entail.max(axis=1).values
            logits = torch.cat((entail, non_entail.unsqueeze(1)), 1)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(logits, targets)
        metrics = {
            f"val_loss_{pref}": loss,
            f"val_acc_{pref}": acc,
        }
        # self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        if not self.trainer.running_sanity_check:
            for metrics_dict in outputs:
                # print(f"worker {hvd.rank()}", metrics_dict)
                # self.logger.agg_and_log_metrics(metrics_dict, step=self.current_epoch)
                for md in metrics_dict:
                    self.log_dict(md)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [param for param in self.net.parameters() if param.requires_grad], lr=0.0001
        )
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
    wandb_logger = WandbLogger(project=f"NLI{'_test' if params['test'] else ''}")
    # wandb_logger.log_hyperparams(config)
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=5,
    #     verbose=True,
    #     mode="min",
    # )
    # print("create tainer")
    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        max_epochs=10,
        distributed_backend="horovod",
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
    )

    embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
    data_module = NLIDataModule(
        params["path_mnli"],
        embs.vocabulary,
        batch_size=params["batch_size"],
        test=params["test"],
    )
    net = BertModel.from_pretrained("prajjwal1/bert-mini")
    # net = Net(embs)
    model = PLModel(net, params)
    if params["test"]:
        print("fit")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
