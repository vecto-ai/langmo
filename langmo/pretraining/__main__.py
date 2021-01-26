from pathlib import Path

import horovod.torch as hvd
import pytorch_lightning as pl
import torch
import transformers
from protonn.utils import save_data_json
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as tr_logging
from transformers.optimization import get_linear_schedule_with_warmup

from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
from langmo.nn.utils import reinit_model
from langmo.utils import load_config

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
        assert not torch.isnan(loss).item(), "loss is nan, can't train"
        # loss_mlm = for MLM, with long ids self.fwd_mlm()
        # loss_nsp = for NSP self.fwd_nsp()
        # use different forwards
        # loss = loss_mlm + loss_nsp # + other aux tasks
        # print(f"loss: {loss.item()}")
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(
            [param for param in self.net.parameters() if param.requires_grad],
            lr=0.00001,
        )
        # TODO: get rough estimation of training steps here
        # maybe after first epoch is trained - reset iterators?
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=500000
        )
        return [[optimizer], [scheduler]]

    # def save_pretrained(self):
    #     file_name = "tmp"  # logdir + current snapshot name
    #     self.net.save_pretrained(file_name)

    def train_epoch_end(self, *args, **kwargs):
        if hvd.rank() == 0:
            print("training epoch end")
            print("args:", args)
            print("kwargs:", kwargs)

    def save_metadata(self, corpus_metadata, path=None):
        # default `save_path` is `hparam["path_results"]`
        if path is None:
            path = self.hparams["path_results"]
        path = Path(path) / "metadata.json"
        if corpus_metadata is not None:
            self.hparams["corpus"] = corpus_metadata
        save_data_json(self.hparams, path)


def main():
    hvd.init()
    if hvd.rank() != 0:
        tr_logging.set_verbosity_error()  # to reduce warning of unused weights
    name_task = "pretrain"
    params = load_config(name_task=name_task)
    name_run = params["name_model"]
    tokenizer = AutoTokenizer.from_pretrained(params["name_model"])
    net = AutoModelForMaskedLM.from_pretrained(params["name_model"])
    reinit_model(net)
    model = PLModel(
        net=net,
        tokenizer=tokenizer,
        params=params,
    )

    n_steps_checkpoint = 5000  # TODO: should this go to params?
    on_n_step_checkpoint = CheckpointEveryNSteps(n_steps_checkpoint)
    # scheudle_eval_callback = ScheduleEval(n_step)
    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=1,
        num_sanity_val_steps=0,
        max_epochs=params["cnt_epochs"],
        distributed_backend="horovod",
        precision=params["precision"],
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=WandbLogger(
            project=params["name_project"],
            name=name_run,
            save_dir=params["path_results"],
        ),
        reload_dataloaders_every_epoch=True,
        # TODO: is this ok?
        # theirs samples do like you did
        # but there is special checkpoint_callback param too....
        callbacks=[on_n_step_checkpoint],
        checkpoint_callback=False,
        gradient_clip_val=0.6,
        # TODO: figure out what is this
        progress_bar_refresh_rate=0,
        track_grad_norm=2,
    )

    data_module = TextDataModule(
        tokenizer=tokenizer,
        params=params,
        # embs.vocabulary,
    )

    trainer.fit(model, data_module)
    print("All done")


if __name__ == "__main__":
    main()
