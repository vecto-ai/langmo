# from langmo.nn.utils import reinit_model
from pathlib import Path
from time import sleep

import horovod.torch as hvd
import pytorch_lightning as pl
import torch
from langmo.base import PLBase
from langmo.callbacks.perf import PerfMonitor
# from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
# from langmo.nn.utils import reinit_model
from langmo.utils import load_config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as tr_logging

from .data import TextDataModule


class PLModel(PLBase):
    def __init__(self, net, tokenizer, params):
        super().__init__(net, tokenizer, params)
        # TODO: add corpus metadata
        print("")
        self.hparams.update(params)
        if "cnt_samples_processed" not in self.hparams:
            self.hparams["cnt_samples_processed"] = 0
        if len(self.hparams["train_logs"]) == 0:
            self.hparams["train_logs"].append({"epoch": -1, "epoch_time": 0.0})

    def forward(self, encoded):
        # print(encoded)
        # dic_encoded = dict(encoded)
        # print(dic_encoded)
        result = self.net(** encoded._asdict())
        return result

    def training_step(self, batch, batch_idx):
        if self.hparams["test"] and batch_idx < 5:
            print("inpts", self.tokenizer.decode(batch.input_ids[0]))
            print()
            print("lbls", batch.labels[0])
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
        self.log("loss", loss)
        # TODO: move this to train_epoch_end when it is fixed
        # self.log("epoch", self.current_epoch)
        cnt_epochs = self.trainer.train_dataloader.loaders.cnt_restarts
        self.hparams["cnt_samples_processed"] += self.hparams["batch_size"] * self.hparams["cnt_workers"]
        self.log("true_epochs", cnt_epochs)
        self.log("samples_processed", self.hparams["cnt_samples_processed"])
        return loss

    def training_epoch_end(self, *args, **kwargs):
        if hvd.rank() == 0:
            # print("args:", args)
            # print("kwargs:", kwargs)
            metrics = {}
            self.add_epoch_id_to_metrics(metrics)
            self.append_metrics_to_train_logs(metrics)
            print(f" ########### training epoch {metrics['epoch']} end ###############")
        # FIXIN double borrow for tokenizers which is currently not working in threads
        # assume it will be ok as long as we don't access tokenizer simultaneously
        # so let's wait here till the queue of training samples is repopulated
        # TODO: this .loaders is kinda strange
        dataloader = self.trainer.train_dataloader.loaders
        while dataloader._queue.qsize() < dataloader._queue.maxsize:
            sleep(1)
        # well this would not block us while queue is full but train loader still preparing the next batch
        # which will be blocked on queue.put() :-\
        # this is beyound ugly but shikata nai
        sleep(5)

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        loss = result["loss"]
        self.log("val_loss", loss)
        # TODO: add MLM accuracy here
        metrics = {
            f"val_loss": loss,
        }
        return metrics

    def validation_epoch_end(self, outputs):
        # TODO: aggregate validation loss to per epoch metric, icnl metadata
        self.trainer.datamodule.val_rng_reset()
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss = hvd.allreduce(loss)
        if hvd.rank() == 0:
            if self.hparams["cnt_samples_per_epoch"] >= 1000000:
                str_cnt_sampels = f"smpl_{self.hparams['cnt_samples_processed'] // 1000000}M"
            else:
                str_cnt_sampels = f"smpl_{self.hparams['cnt_samples_processed'] // 1000}K"
            path_checkpoint = (
                Path(self.hparams["path_results"])
                / "checkpoints"
                / str_cnt_sampels
            )
            print("saving to ", path_checkpoint)
            self.trainer.save_checkpoint(path_checkpoint / "PL_model.ckpt")
            path_hf = path_checkpoint / "hf"
            self.trainer.model.save_as_hf(path_hf)
            self.hparams["train_logs"][-1]["val_loss"] = loss.item()
            self.save_metadata(path_checkpoint)

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


def main():
    hvd.init()
    if hvd.rank() != 0:
        tr_logging.set_verbosity_error()  # to reduce warning of unused weights
    name_task = "pretrain"
    params = load_config(name_task=name_task)
    name_run = params["model_name"]
    # TODO: revisit this when we have model parallel training
    name_run += f"_{params['timestamp']}"
    name_run += f"_bs{params['batch_size'] * params['cnt_workers']}"
    name_run += f"_lr{params['max_lr']}"
    name_run += f"_wd{params['weight_decay']}"
    name_run += f"_stp{params['cnt_training_steps']}"
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
    lr_monitor = LearningRateMonitor(logging_interval="step")
    if params["use_gpu"]:
        assert torch.cuda.device_count() > 0, "Asked for `use_gpu` but no gpu detected"
    gpus = 1 if params["use_gpu"] else 0
    checkpoint = params["resume"]["checkpoint"] if "resume" in params else None
    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=gpus,
        num_sanity_val_steps=0 if "resume" in params else -1,
        max_epochs=params["cnt_epochs"],
        accelerator="horovod",
        precision=params["precision"],
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=WandbLogger(
            project=params["name_project"],
            name=name_run,
            save_dir=params["path_results"],
        ),
        reload_dataloaders_every_epoch=False,
        # TODO: is this ok?
        # theirs samples do like you did
        # but there is special checkpoint_callback param too....
        callbacks=[lr_monitor, PerfMonitor()],
        checkpoint_callback=False,
        gradient_clip_val=params["gradient_clip_val"],
        # TODO: figure out what is this
        progress_bar_refresh_rate=0,
        track_grad_norm=0,
        # profiler="simple",
        resume_from_checkpoint=checkpoint,
    )
    trainer.fit(model, data_module)
    print("All done")


if __name__ == "__main__":
    main()
