import pytorch_lightning as pl
import torch
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from .data import TextDataModule


# define PL model
class PLModel(pl.LightningModule):
    def __init__(self, net, params):
        super().__init__()
        self.net = net

    def forward(self, encoded):
        input_ids = encoded["input_ids"]
        token_type_ids = encoded["token_type_ids"]
        attention_mask = encoded["attention_mask"]
        labels = encoded["labels"]
        result = self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return result

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        loss = result["loss"]
        # loss_mlm = for MLM, with long ids self.fwd_mlm()
        # loss_nsp = for NSP self.fwd_nsp()
        # use different forwards
        # loss = loss_mlm + loss_nsp # + other aux tasks
        print(f"loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [param for param in self.net.parameters() if param.requires_grad],
            lr=0.00001,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=50000
        )
        return [[optimizer], [scheduler]]


# create trainer


def main():
    params = dict(
        cnt_epochs=3,
        uncase=True,
        test=True,
        batch_size=16,
        max_length=32,
        path_corpus="/home/blackbird/Projects_heavy/NLP/vecto/tests/data/corpora/multiple_files"
    )
    name_model = "prajjwal1/bert-mini"
    model = PLModel(
        net=AutoModelForMaskedLM.from_pretrained(name_model),
        params=params,
    )

    name_run = name_model
    # if params["randomize"]:
    #     reinit_model(net)
    #     name_run += "_RND"
    # name_run += f"_{'↓' if params['uncase'] else '◯'}_{timestamp[:-3]}"
    wandb_name = f"NLI{'_test' if params['test'] else ''}"

    trainer = pl.Trainer(
        # gpus=1,
        num_sanity_val_steps=0,
        max_epochs=params["cnt_epochs"],
        # distributed_backend="horovod",
        # replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        # wandb_logger = WandbLogger(project=wandb_name, name=name_run),
        progress_bar_refresh_rate=0,
    )
    data_module = TextDataModule(
        tokenizer=AutoTokenizer.from_pretrained(name_model),
        params=params,
        # embs.vocabulary,
        # batch_size=params["batch_size"],
    )

    trainer.fit(model, data_module)
    print("All done")


if __name__ == "__main__":
    main()
