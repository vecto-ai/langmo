# from langmo.nn.utils import reinit_model
# import os
from logging import getLogger
from pathlib import Path
from time import sleep

import pytorch_lightning as pl
import torch
from langmo.base import PLBase
from langmo.callbacks.monitor import Monitor
from langmo.callbacks.layernorm import LayerNormCallback

# from langmo.cluster_mpi import MPIClusterEnvironment
# from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
# from langmo.nn.utils import reinit_model
# from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
# from langmo.nn.utils import reinit_model
from langmo.config import ConfigPretrain as Config
from langmo.log_helper import set_root_logger
from protonn.distributed import dist_adapter as da
from protonn.utils import get_time_str
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as tr_logging

from .data import TextDataModule


class PLModel(PLBase):
    def __init__(self, net, tokenizer, params):
        super().__init__(net, tokenizer, params)
        # TODO: add corpus metadata
        self.pylogger = getLogger(__name__)
        self.hparams["cnt_samples_processed"] = 0
        path_checkpoint = Path(params["path_results"]) / "checkpoints" / "ep_-1_smpl_000" / "hf"
        # TODO: we might not need this when we return validation
        if self.global_rank == 0:
            print("saving to ", path_checkpoint)
            self.save_as_hf(path_checkpoint)

    def forward(self, batch):
        result = self.net(**batch._asdict())
        return result

    def training_step(self, batch, batch_idx):
        # print("train step start")
        assert self.hparams["batch_size"] == len(batch.input_ids)
        if self.hparams["test"] and batch_idx < 5:
            print(
                f"proc {self.global_rank}/{self.local_rank}, model on {self.device}, batch on {batch[0].device}"
            )
            print("inpts", self.tokenizer.decode(batch.input_ids[0]))
            print()
            print("lbls", batch.labels[0])
            print("mask", batch.attention_mask[0])
            # print("type ids", batch.token_type_ids[0])
            # print("mask", batch.attention_mask[0])
            # print()
            print()
        result = self.forward(batch)
        # TODO: how about loss only / more loss for masked tokens?
        loss = result["loss"]
        assert not torch.isnan(loss).item(), "loss is nan, can't train"
        # if torch.isnan(loss):
        #     print(">> loss is NaN\n")
        #     return None
        # # loss_mlm = for MLM, with long ids self.fwd_mlm()
        # loss_nsp = for NSP self.fwd_nsp()
        # use different forwards
        # loss = loss_mlm + loss_nsp # + other aux tasks
        # lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # print(
        #     f"ep {self.current_epoch}, step {self.global_step}, loss: {loss.item()}, lr {lr}"
        # )
        # TODO: move this to train_epoch_end when it is fixed
        # self.log("epoch", self.current_epoch)
        cnt_epochs = float(self.trainer.train_dataloader.loaders.cnt_restarts)
        self.hparams["cnt_samples_processed"] += (
            self.hparams["batch_size"] * self.hparams["cnt_workers"]
        )
        self.log("loss", loss, sync_dist=True)
        self.log("true_epochs", float(cnt_epochs))
        # print("logging samples processed as", self.hparams["cnt_samples_processed"])
        self.log("samples_processed", float(self.hparams["cnt_samples_processed"]))
        # print("train step done")
        # print(loss.shape)
        if batch_idx % 10000 == 0:
            print(
                f"end train step {batch_idx} on worker {self.global_rank}, loss={loss.item()}, time={get_time_str()}"
            )
        return loss

    def training_epoch_end(self, *args, **kwargs):
        # if self.global_rank == 0:
        #     print("args:", args)
        #     print("kwargs:", kwargs)
        #     metrics = {}
        #     self.add_epoch_id_to_metrics(metrics)
        #     self.append_metrics_to_train_logs(metrics)
        self.pylogger.info(f"training epoch end")
        sleep(1)

    # def validation_step(self, batch, batch_idx):
    #     # print("val step start")
    #     result = self.forward(batch)
    #     loss = result["loss"]
    #     # self.log("val_loss", loss, sync_dist=True)
    #     # TODO: add MLM accuracy here
    #     # metrics = {
    #     #     f"val_loss": loss,
    #     # }
    #     # print("val step done")
    #     return loss

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            print(f"########### main: validation epoch end ###############")
        # self.trainer.datamodule.val_rng_reset()
        # loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # loss = da.allreduce(loss)
        # self.hparams["train_logs"][-1]["val_loss"] = loss.item()

    # def save_metadata(self, corpus_metadata, path=None):
    #     # default `save_path` is `hparam["path_results"]`
    #     if path is None:
    #         path = self.hparams["path_results"]
    #     path = Path(path) / "metadata.json"
    #     if corpus_metadata is not None:
    #         self.hparams["corpus"] = corpus_metadata
    #     save_data_json(self.hparams, path)


def build_model(params):
    if "resume" in params:
        resume = params["resume"]
        tokenizer = AutoTokenizer.from_pretrained(resume["hf"])
        net = AutoModelForMaskedLM.from_pretrained(resume["hf"])
        net.train()
        # TODO: hm... why this not overwhiting params set in __init__ :-\
        print("RESUMING FROM PARAMS")
        print(params["train_logs"])
        model = PLModel.load_from_checkpoint(
            resume["checkpoint"],
            net=net,
            tokenizer=tokenizer,
            params=params,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
        config = AutoConfig.from_pretrained(params["model_name"])
        net = AutoModelForMaskedLM.from_config(config)
        # reinit_model(net)
        net.train()
        model = PLModel(
            net=net,
            tokenizer=tokenizer,
            params=params,
        )
    return model


def get_run_name(params):
    name_run = params["model_name"]
    # TODO: revisit this when we have model parallel training
    name_run += f"_{params['timestamp']}"
    # name_run += f"_bs{params['batch_size'] * params['cnt_workers']}"
    name_run += f"_lr{params['max_lr']}"
    # TODO: add batch size
    # name_run += f"_wd{params['weight_decay']}"
    # name_run += f"_bs{params['batch_size_effective']}"
    return name_run


def main():
    set_root_logger()
    da.init("horovod")
    # if da.rank() != 0:
    #     tr_logging.set_verbosity_error()  # to reduce warning of unused weights
    name_task = "pretrain"
    params = Config(name_task=name_task)
    # params = Config(name_task=name_task, is_master=(da.rank() == 0))
    # TODO: make logging report rank and size and use logging
    name_run = get_run_name(params)
    if params["use_gpu"]:
        assert torch.cuda.device_count() > 0, "Asked for `use_gpu` but no gpu detected"
    # use 1 GPU with horovod and -1 with DDP
    checkpoint = params["resume"]["checkpoint"] if "resume" in params else None
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # if (da.rank() != 0):
    #     params["path_results"] = "/tmp"
    # cluster_env = MPIClusterEnvironment()
    # gpus = [cluster_env.local_rank()] if params["use_gpu"] else 0
    gpus = 1 if params["use_gpu"] else 0
    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=gpus,
        # num_nodes=cluster_env.cnt_nodes(),
        num_sanity_val_steps=0 if "resume" in params else -1,
        max_epochs=params["cnt_epochs"],
        strategy="horovod",
        precision=params["precision"],
        replace_sampler_ddp=False,
        logger=WandbLogger(
            project=params["name_project"],
            name=name_run,
            save_dir=params["path_results"],
        ),
        log_every_n_steps=params["log_every_n_steps"],
        reload_dataloaders_every_n_epochs=0,
        # TODO: is this ok?
        # theirs samples do like you did
        # but there is special checkpoint_callback param too....
        callbacks=[lr_monitor, LayerNormCallback(), Monitor()],
        gradient_clip_val=params["gradient_clip_val"],
        enable_progress_bar=False,
        enable_checkpointing=False,
        # TODO: figure out what is this
        track_grad_norm=1,
        # detect_anomaly=True, # This is very slow!
        # profiler="simple",
        resume_from_checkpoint=checkpoint,
        # plugins="deepspeed_stage_2",
        # plugins=[cluster_env],
        accumulate_grad_batches=params["accumulate_batches"],
    )
    if trainer.global_rank == 0:
        (Path(params["path_results"]) / "wandb").mkdir(parents=True, exist_ok=True)

    model = build_model(params)
    data_module = TextDataModule(
        tokenizer=model.tokenizer,
        params=params,
        # embs.vocabulary,
    )
    model.hparams["corpus"] = data_module.corpus.metadata

    # n_steps_checkpoint = 10000  # TODO: should this go to params?
    # on_n_step_checkpoint = CheckpointEveryNSteps(n_steps_checkpoint)
    # scheudle_eval_callback = ScheduleEval(n_step)
    # listen()
    model.pylogger.info("calling fit")
    trainer.fit(model, data_module)
    print("All done")


if __name__ == "__main__":
    main()
