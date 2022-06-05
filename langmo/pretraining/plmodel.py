from logging import getLogger
from time import sleep

import torch
from langmo.base import PLBase
from protonn.utils import get_time_str
from torchmetrics import MeanMetric


class PLModel(PLBase):
    def __init__(self, net, tokenizer, params):
        super().__init__(net, tokenizer, params)
        # TODO: add corpus metadata
        print("%%%%%%%%%%%%% WE ARE IN INIT OF PL MODEL")
        self.pylogger = getLogger(__name__)
        self.metric_loss = MeanMetric()
        # self.hparams["cnt_samples_processed"] = 0

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
            print(f"end train step {batch_idx} on worker {self.global_rank}, loss={loss.item()}, time={get_time_str()}")
        self.metric_loss.update(loss)
        return loss

    def training_epoch_end(self, *args, **kwargs):
        # if self.global_rank == 0:
            # print("args:", args)
            # print("kwargs:", kwargs)
            # metrics = {}
            # self.add_epoch_id_to_metrics(metrics)
            # self.append_metrics_to_train_logs(metrics)
        self.pylogger.info(f"training epoch end")
        self.hparams["train_logs"][-1]["loss"] = self.metric_loss.compute().item()
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
