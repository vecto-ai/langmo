import sys
import yaml
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


class PLModel(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.example_input_array = (torch.zeros((128, 32), dtype=torch.int64), torch.zeros((128, 32), dtype=torch.int64))

    def forward(self, s1, s2):
        # print(s1.shape)
        return self.net(s1, s2)

    def training_step(self, batch, batch_idx):
        s1, s2, target = batch
        logits = self(s1, s2)
        loss = F.cross_entropy(logits, target)
        acc = accuracy(logits, target)
        metrics = {
            'train_loss': loss,
            'train_acc': acc,
        }
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        s1, s2, target = batch
        logits = self(s1, s2)
        loss = F.cross_entropy(logits, target)
        acc = accuracy(logits, target)
        metrics = {
            'val_loss': loss,
            'val_acc': acc,
        }
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam([param for param in self.net.parameters() if param.requires_grad], lr=0.0001)
        # return torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)


def main():
    #path_data = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small"
    #path_data = "/groups1/gac50489/datasets/cosmoflow_full/cosmoUniverse_2019_05_4parE_tf"
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        return
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    path_results_base = "./out/NLI"
    params["path_results"] = get_unique_results_path(path_results_base)
    params["batch_size"] = 32
    wandb_logger = WandbLogger(project="NLI")
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
    data_module = NLIDataModule(params["path_mnli"], embs.vocabulary, batch_size=params["batch_size"], test=params["test"])
    # net = BertModel.from_pretrained("prajjwal1/bert-mini")
    net = Net(embs)
    model = PLModel(net)
    print("fit")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
