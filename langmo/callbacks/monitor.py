# pylint: disable=attribute-defined-outside-init
from pathlib import Path
from timeit import default_timer as timer

import lightning as pl

# from protonn.utils import num_to_str_with_suffix

# TODO: proper logging


class Monitor(pl.Callback):

    # def append_metrics_to_train_logs(self, metrics):
    #     entry = dict(epoch=metrics["epoch"])
    #     for k, v in metrics.items():
    #         val = v.item() if hasattr(v, "item") else v
    #         entry[k] = val
    #     self.hparams["train_logs"].append(entry)

    def _save_best_only(self, path_new_checkpoint, pl_module):
        raise NotImplementedError("Don't use Monitor class directly")

    def _save_hf_and_metadata(self, path_checkpoint, pl_module):
        pl_module.save_metadata(path_checkpoint)
        pl_module.save_as_hf(path_checkpoint / "hf")

    def maybe_save_metadata_and_hf(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        path_new_checkpoint = pl_module._get_ckecpoint_folder()
        path_new_checkpoint.mkdir(parents=True, exist_ok=True)

        if pl_module.hparams["snapshot_strategy"] == "none":
            return

        print("saving to ", path_new_checkpoint)
        if pl_module.hparams["snapshot_strategy"] == "per_epoch":
            self._save_hf_and_metadata(path_new_checkpoint, pl_module)
        elif pl_module.hparams["snapshot_strategy"] == "best_only":
            self._save_best_only(path_new_checkpoint, pl_module)
        print("saving done")

    def setup(self, trainer, pl_module, stage=None):
        self.pl_module = pl_module
        self.hparams = pl_module.hparams
        # TODO: check what happens on resume
        # if "train_logs" not in self.hparams:
        #     self.hparams["train_logs"] = []
        #     self.hparams["cnt_samples_processed"] = 0
        #     self.hparams["train_logs"].append({})
        #     self.hparams["train_logs"][-1]["epoch"] = -1
        #     self.hparams["train_logs"][-1]["epoch_time"] = 0.0
        #     self.hparams["train_logs"][-1]["cnt_samples_processed"] = 0
        #     self.epoch = -1
        #     self.maybe_save_metadata_and_hf(trainer, pl_module)
        # self.epoch = trainer.current_epoch
        self.time_last_checkpoint = timer()
        self.metric_to_monitor = pl_module.hparams["metric_to_monitor"]
        # check if we are not resuimg

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.hparams["train_logs"].append({})
        self.hparams["train_logs"][-1]["epoch"] = trainer.current_epoch
        self.epoch = trainer.current_epoch
        if trainer.global_rank == 0:
            print(
                f"@@@@ perf callback: train epoch epoch {pl_module.current_epoch} started @@@@"
            )
        self.time_start = timer()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # TODO: this is here since we are trhing to remove valiation epoch
        self.time_end = timer()
        epoch_time = self.time_end - self.time_start
        self.hparams["train_logs"][-1]["epoch_time"] = epoch_time
        self.hparams["train_logs"][-1]["samples_per_second"] = (
            self.hparams["cnt_samples_per_epoch"] / epoch_time
        )
        self.hparams["train_logs"][-1]["samples_per_second_worker"] = (
            self.hparams["train_logs"][-1]["samples_per_second"]
            / self.hparams["cnt_workers"]
        )
        self.hparams["train_logs"][-1]["cnt_samples_processed"] = self.hparams[
            "cnt_samples_processed"
        ]
        epoch_time = self.hparams["train_logs"][-1]["epoch_time"]
        if trainer.global_rank == 0:
            print(
                f"@@@@ perf callback: train epoch {pl_module.current_epoch} "
                f"end, done in {epoch_time} sec"
            )
            pl_module.save_metadata(pl_module.hparams["path_results"])
        self.maybe_save_metadata_and_hf(trainer, pl_module)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # in PL train epoch start hook activates, then all training batches are processed
        # then validation epoch starts, then all val batches,
        # then val epoch ends, then train epoch ends
        # so we are placing "end training epoch" hook at start of val epoch
        self.time_end = timer()
        self.epoch = -1 if trainer.sanity_checking else trainer.current_epoch
        self.hparams["train_logs"][-1]["epoch"] = self.epoch
        # this is kinda ugly. having separate callback would have been better,
        # but then we have to return predictions from each validation step

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        print(
            f"@@@@ perf callback: validation epoch epoch {pl_module.current_epoch}"
            f" ended wrkr {trainer.global_rank} @@@@"
        )
        if self.hparams["save_predictions"]:
            for f in pl_module.files_predictions:
                f.close()
        # pl_module.files_detail = [open(path_details / f"pred_{split}_w{pl_module.global_rank}.json", "w")
        #     for split in pl_module.validation_split_names]
        # if trainer.global_rank == 0:
        # pl_module.save_metadata()
        # n_smpl = num_to_str_with_suffix(pl_module.hparams['cnt_samples_processed'])
        # str_cnt_sampels = f"smpl_{n_smpl}"
        # path_checkpoint = (
        #     Path(pl_module.hparams["path_results"])
        #     / "checkpoints"
        #     / str_cnt_sampels
        # )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        unused=0,
    ):
        checkpoint_interval = pl_module.hparams["minutes_between_snapshots"]
        if timer() - self.time_last_checkpoint > checkpoint_interval * 60:
            self.time_last_checkpoint = timer()
            if trainer.global_rank == 0:
                path_for_resume = Path(pl_module.hparams["path_results"]) / "resume"
                trainer.save_checkpoint(path_for_resume / "PL_model.ckpt")
                pl_module.save_metadata(path_for_resume)
                if pl_module.hparams["overwrite_timer_snapshot"]:
                    path_last_hf = Path(pl_module.hparams["path_results"]) / "hf_last"
                else:
                    samples_processed = self.pl_module.hparams["cnt_samples_processed"]
                    folder_last_hf = str(samples_processed)
                    path_last_hf = Path(pl_module.hparams["path_results"]) / "hf_on_timer" / folder_last_hf
                pl_module.save_as_hf(path_last_hf)

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     print("VAL BATCH END CALLBACK:", outputs)
