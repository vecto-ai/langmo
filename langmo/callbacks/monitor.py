from pathlib import Path
from timeit import default_timer as timer

import pytorch_lightning as pl
from protonn.utils import num_to_str_with_suffix

# TODO: proper logging
# TODO: processed samples per epoch

class Monitor(pl.Callback):

    # def append_metrics_to_train_logs(self, metrics):
    #     entry = dict(epoch=metrics["epoch"])
    #     for k, v in metrics.items():
    #         val = v.item() if hasattr(v, "item") else v
    #         entry[k] = val
    #     self.hparams["train_logs"].append(entry)
    def setup(self, trainer, pl_module, stage=None):
        self.hparams = pl_module.hparams
        # TODO: check what happens on resume
        self.hparams["train_logs"] = []
        if len(self.hparams["train_logs"]) == 0:
            self.hparams["train_logs"].append({"epoch": -1, "epoch_time": 0.0})
        if "cnt_samples_processed" not in self.hparams:
            self.hparams["cnt_samples_processed"] = 0
        self.log = self.hparams["train_logs"]

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.hparams["train_logs"].append({})
        pl_module.hparams["train_logs"][-1]["epoch"] = trainer.current_epoch
        self.epoch = trainer.current_epoch
        if trainer.global_rank == 0:
            print(f"@@@@ perf callback: train epoch epoch {pl_module.current_epoch} started @@@@")
        self.time_start = timer()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # TODO: this is here since we are trhing to remove valiation epoch
        self.time_end = timer()
        epoch_time = self.time_end - self.time_start
        if trainer.global_rank == 0:

            print(f"@@@@ perf callback: train epoch {pl_module.current_epoch} end, done in {epoch_time} sec")
            self.log[-1]["epoch_time"] = epoch_time
            self.log[-1]["samples_per_second"] = self.hparams["cnt_samples_per_epoch"] / epoch_time
            self.log[-1]["samples_per_second_worker"] = self.hparams["train_logs"][-1]["samples_per_second"] / self.hparams["cnt_workers"]
            self.log[-1]["cnt_samples_processed"] = self.hparams["cnt_samples_processed"]
            # pl_module.save_metadata()
            path_checkpoint = Path(pl_module.hparams["path_results"]) / "checkpoints" / f"ep_{self.epoch:03d}_smpl_{num_to_str_with_suffix(self.hparams['cnt_samples_processed'])}"
            print("saving to ", path_checkpoint)
            # trainer.save_checkpoint(path_checkpoint / "PL_model.ckpt")
            path_hf = path_checkpoint / "hf"
            pl_module.save_as_hf(path_hf)
            pl_module.save_metadata(path_checkpoint)
            print("saving done")

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
        print(f"@@@@ perf callback: validation epoch epoch {pl_module.current_epoch} ended wrkr {trainer.global_rank} @@@@")
        # if trainer.global_rank == 0:
            # pl_module.save_metadata()
            # str_cnt_sampels = f"smpl_{num_to_str_with_suffix(pl_module.hparams['cnt_samples_processed'])}"
            # path_checkpoint = (
            #     Path(pl_module.hparams["path_results"])
            #     / "checkpoints"
            #     / str_cnt_sampels
            # )
