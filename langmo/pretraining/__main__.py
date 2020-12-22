from pathlib import Path

import pytorch_lightning as pl
import torch
from langmo.utils import load_config
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.metrics.functional import accuracy
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from .data import TextDataModule
from transformers import logging as tr_logging
import horovod.torch as hvd


# define PL model
class PLModel(pl.LightningModule):
    def __init__(self, net, tokenizer, params):
        super().__init__()
        # TODO: read this from params
        self.net = net
        self.tokenizer = tokenizer

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


# From here:
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2534#issuecomment-674582085
class CheckpointEveryNSteps(pl.Callback):
    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            dirpath = trainer.checkpoint_callback.dirpath

            save_path = Path(dirpath).joinpath(filename)
            trainer.save_checkpoint(save_path)

            save_path = Path(dirpath).joinpath(filename + ".hfull")
            trainer.model.net.save_pretrained(save_path)
            trainer.model.tokenizer.save_pretrained(save_path)


def main():
    params = load_config()
    tokenizer = AutoTokenizer.from_pretrained(params["name_model"])
    model = PLModel(
        net=AutoModelForMaskedLM.from_pretrained(params["name_model"]),
        tokenizer=tokenizer,
        params=params,
    )

    name_run = params["name_model"]
    # if params["randomize"]:
    #     reinit_model(net)
    #     name_run += "_RND"
    # name_run += f"_{'↓' if params['uncase'] else '◯'}_{timestamp[:-3]}"
    wandb_name = f"pretrain{'_test' if params['test'] else ''}"
    hvd.init()
    if hvd.rank() != 0:
        tr_logging.set_verbosity_error()

    n_step = 1000  # TODO: should this go to params?
    on_n_step_callback = CheckpointEveryNSteps(n_step)

    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        max_epochs=params["cnt_epochs"],
        distributed_backend="horovod",
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=WandbLogger(project=wandb_name, name=name_run),
        callbacks=[on_n_step_callback],
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
