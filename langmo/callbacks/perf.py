from pathlib import Path
from timeit import default_timer as timer

import pytorch_lightning as pl
from protonn.utils import num_to_str_with_suffix


class PerfMonitor(pl.Callback):

    # def append_metrics_to_train_logs(self, metrics):
    #     entry = dict(epoch=metrics["epoch"])
    #     for k, v in metrics.items():
    #         val = v.item() if hasattr(v, "item") else v
    #         entry[k] = val
    #     self.hparams["train_logs"].append(entry)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank == 0:
            print(f"@@@@ perf callback: train epoch epoch {pl_module.current_epoch} started @@@@")
        self.time_start = timer()
        pl_module.hparams["train_logs"].append({})

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank == 0:
            epoch_time = self.time_end - self.time_start
            print(f"@@@@ perf callback: train epoch {pl_module.current_epoch} end, done in {epoch_time} sec")
            pl_module.hparams["train_logs"][-1]["epoch_time"] = epoch_time
            pl_module.hparams["train_logs"][-1]["samples_per_second"] = pl_module.hparams["cnt_samples_per_epoch"] / epoch_time
            pl_module.hparams["train_logs"][-1]["samples_per_second_worker"] = pl_module.hparams["train_logs"][-1]["samples_per_second"] / pl_module.hparams["cnt_workers"]
            # pl_module.save_metadata()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # in PL train epoch start hook activates, then all training batches are processed
        # then validation epoch starts, then all val batches,
        # then val epoch ends, then train epoch ends
        # so we are placing "end training epoch" hook at start of val epoch
        self.time_end = timer()
        self.epoch = -1 if trainer.sanity_checking else trainer.current_epoch
        pl_module.hparams["train_logs"][-1]["epoch"] = self.epoch
        if trainer.global_rank == 0:
            print(f"@@@@ perf callback: validation epoch {pl_module.current_epoch} started @@@@")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank == 0:
            print(f"@@@@ perf callback: validation epoch epoch {pl_module.current_epoch} ended @@@@")
            if hasattr(self, "time_start"):
                epoch_time = self.time_end - self.time_start
                pl_module.hparams["train_logs"][-1]["epoch_time"] = epoch_time
                pl_module.hparams["train_logs"][-1]["samples_per_second"] = pl_module.hparams["cnt_samples_per_epoch"] / epoch_time
                pl_module.hparams["train_logs"][-1]["samples_per_second_worker"] = pl_module.hparams["train_logs"][-1]["samples_per_second"] / pl_module.hparams["cnt_workers"]
            # pl_module.save_metadata()
            # str_cnt_sampels = f"smpl_{num_to_str_with_suffix(pl_module.hparams['cnt_samples_processed'])}"
            # path_checkpoint = (
            #     Path(pl_module.hparams["path_results"])
            #     / "checkpoints"
            #     / str_cnt_sampels
            # )
            path_checkpoint = Path(pl_module.hparams["path_results"]) / "checkpoints" / f"ep_{self.epoch:03d}"
            print("saving to ", path_checkpoint)
            # self.trainer.save_checkpoint(path_checkpoint / "PL_model.ckpt")
            path_hf = path_checkpoint / "hf"
            trainer.model.save_as_hf(path_hf)
            pl_module.save_metadata(path_checkpoint)
