from pathlib import Path

import pytorch_lightning as pl
import torch
import os

from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.metrics.functional import accuracy
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from transformers import logging as tr_logging
import horovod.torch as hvd
from langmo.checkpoint import CheckpointEveryNSteps
from langmo.utils import load_config
from langmo.utils import get_unique_results_path
from .data import TextDataModule


class PLModel(pl.LightningModule):
    def __init__(self, net, tokenizer, params):
        super().__init__()
        # TODO: read this from params
        self.net = net
        self.tokenizer = tokenizer
        self.hparams = params

    def forward(self, encoded):
        input_ids = encoded.input_ids
        token_type_ids = encoded.token_type_ids
        attention_mask = encoded.attention_mask
        labels = encoded.labels
        result = self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return result

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        # TODO: how about loss only / more loss for masked tokens?
        loss = result["loss"]
        # loss_mlm = for MLM, with long ids self.fwd_mlm()
        # loss_nsp = for NSP self.fwd_nsp()
        # use different forwards
        # loss = loss_mlm + loss_nsp # + other aux tasks
        # print(f"loss: {loss.item()}")
        self.log("loss", loss)
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

    def save_pretrained(self):
        file_name = "tmp"  # logdir + current snapshot name
        self.net.save_pretrained(file_name)


def main():
    params = load_config()
    name_run = params["name_model"]
    name_project = f"pretrain{'_test' if params['test'] else ''}"
    params["path_results"] = os.path.join(params["path_results"], name_project)
    if params["create_unique_path"]:
        params["path_results"] = get_unique_results_path(params["path_results"])

    tokenizer = AutoTokenizer.from_pretrained(params["name_model"])
    model = PLModel(
        net=AutoModelForMaskedLM.from_pretrained(params["name_model"]),
        tokenizer=tokenizer,
        params=params,
    )

    # if params["randomize"]:
    #     reinit_model(net)
    #     name_run += "_RND"
    # name_run += f"_{'↓' if params['uncase'] else '◯'}_{timestamp[:-3]}"
    hvd.init()
    if hvd.rank() == 0:
        (Path(params["path_results"]) / "wandb").mkdir(parents=True, exist_ok=True)
    else:
        tr_logging.set_verbosity_error()  # to reduce warning of unused weights

    n_step = 1000  # TODO: should this go to params?
    on_n_step_callback = CheckpointEveryNSteps(n_step)

    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=1,
        num_sanity_val_steps=0,
        max_epochs=params["cnt_epochs"],
        distributed_backend="horovod",
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=WandbLogger(project=name_project, name=name_run, save_dir=params["path_results"]),
        # TODO: is this ok?
        # theirs samples do like you did
        # but there is special checkpoint_callback param too....
        callbacks=[on_n_step_callback],
        checkpoint_callback=False,
        # TODO: figure out what is this
        progress_bar_refresh_rate=0,
    )

    data_module = TextDataModule(
        tokenizer=tokenizer,
        params=params,
        # embs.vocabulary,
        # batch_size=params["batch_size"],
    )

    trainer.fit(model, data_module)
    print("All done")


if __name__ == "__main__":
    main()
