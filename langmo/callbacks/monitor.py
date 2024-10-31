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
        if pl_module.hparams["snapshot_strategy"] == "none":
            return

        path_new_checkpoint = pl_module._get_ckecpoint_folder()
        path_new_checkpoint.mkdir(parents=True, exist_ok=True)
        print("saving to ", path_new_checkpoint)
        if pl_module.hparams["snapshot_strategy"] == "per_epoch":
            self._save_hf_and_metadata(path_new_checkpoint, pl_module)
        elif pl_module.hparams["snapshot_strategy"] == "best_only":
            self._save_best_only(path_new_checkpoint, pl_module)
        print("saving done")

    def setup(self, trainer, pl_module, stage=None):
        print("!!!!! WE ARE IN SETUP")
        self.pl_module = pl_module
        self.hparams = pl_module.hparams
        pl_module.init_train_logs()
        self.epoch = self.hparams["train_logs"][-1]["epoch"]
        # TODO: this runs on resume because model state
        # includingg logs gets restored after callback is set....
        # TODO: aren't we rewriting first snapshot on resume???
        if len(self.hparams["train_logs"]) == 1:
            print("@@@@@@@@@@@@ SAVING -1")
            self.maybe_save_metadata_and_hf(trainer, pl_module)
        self.epoch = trainer.current_epoch
        self.time_last_checkpoint = timer()
        self.time_start = timer()
        self.metric_to_monitor = pl_module.hparams["metric_to_monitor"]
        if self.hparams["snapshot_schedule"] is not None:
            self.hparams["train_logs"].append({})

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank == 0:
            print(f"@@@@ perf callback: train epoch epoch {pl_module.current_epoch} started @@@@")
        self.epoch = trainer.current_epoch
        if self.hparams["snapshot_schedule"] is None:
            self.time_start = timer()
            if "is_resume" in self.hparams:
                self.hparams.pop("is_resume")
            else:
                self.hparams["train_logs"].append({})
            self.hparams["train_logs"][-1]["epoch"] = trainer.current_epoch

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.hparams["snapshot_schedule"] is None:
            print(f"TRAIN ep={self.epoch} snap={len(self.hparams['train_logs'])} END saving")
            self.save_snapshot(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # in PL train epoch start hook activates, then all training batches are processed
        # then validation epoch starts, then all val batches,
        # then val epoch ends, then train epoch ends
        # so we are placing "end training epoch" hook at start of val epoch
        self.time_end = timer()
        self.epoch = -1 if trainer.sanity_checking else trainer.current_epoch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print(
            f"@@@@ perf callback: validation epoch {pl_module.current_epoch} ended" f"on wrkr {trainer.global_rank} @@@@"
        )
        if self.hparams["save_predictions"]:
            for f in pl_module.files_predictions:
                f.close()
        if self.epoch == -1:
            self.maybe_save_metadata_and_hf(trainer, pl_module)
            self.hparams["train_logs"].append({})
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
        print(self.hparams["train_logs"])

    def save_snapshot(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.update_epoch_time()
        epoch_log = self.hparams["train_logs"][-1]
        epoch_time = epoch_log["time_from_last_snapshot"]
        epoch_log["samples_per_second"] = self.hparams["cnt_samples_per_epoch"] / epoch_time
        epoch_log["samples_per_second_worker"] = epoch_log["samples_per_second"] / self.hparams["cnt_workers"]
        epoch_log["cnt_samples_processed"] = self.hparams["cnt_samples_processed"]
        self.maybe_save_metadata_and_hf(trainer, pl_module)

    def update_epoch_time(self):
        epoch_log = self.hparams["train_logs"][-1]
        if "epoch_time" in epoch_log:
            time_before_resume = epoch_log["epoch_time"]
        else:
            time_before_resume = 0
        time_end = timer()
        time_after_resume = time_end - self.time_start
        epoch_log["time_from_last_snapshot"] = time_after_resume + time_before_resume
        epoch_log["cnt_samples_processed"] = self.hparams["cnt_samples_processed"]
        self.time_start = timer()

    def save_resume_checkpoint(self, trainer, pl_module):
        # TODO: can't we just make it on init point to the same variable?
        self.update_epoch_time()
        print("SAVING RESUME")
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

    def is_time_to_save_on_cnt_samples(self):
        train_log = self.hparams["train_logs"]
        cnt_samples_prcocessed = self.hparams["cnt_samples_processed"]
        cnt_samples_from_last_snapshot = cnt_samples_prcocessed - train_log[-2]["cnt_samples_processed"]
        # cache this when we save snapshot
        # chech if dictionary is sorted
        for threshold, schedule in self.hparams["snapshot_schedule"].items():
            if threshold >= cnt_samples_prcocessed:
                break
            current_scedule = schedule
        return cnt_samples_from_last_snapshot >= current_scedule

    def maybe_save_on_cnt_samples(self, trainer, pl_module):
        if self.hparams["snapshot_schedule"] is None:
            return
        if self.is_time_to_save_on_cnt_samples():
            self.hparams["train_logs"][-1]["epoch"] = trainer.current_epoch
            self.save_snapshot(trainer, pl_module)
            self.hparams["train_logs"].append({})

    def maybe_save_on_timer(self, trainer, pl_module):
        checkpoint_interval = pl_module.hparams["minutes_between_snapshots"]
        if timer() - self.time_last_checkpoint > checkpoint_interval * 60:
            self.time_last_checkpoint = timer()
            self.save_resume_checkpoint(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        unused=0,
    ):
        # TODO: multiply current accumulation
        pl_module.hparams["cnt_samples_processed"] += self.hparams["batch_size"] * self.hparams["cnt_workers"]
        pl_module.hparams["cnt_tokens_processed"] += (
            self.hparams["batch_size"] * self.hparams["cnt_workers"] * self.hparams["max_length"]
        )
        pl_module.log("samples_processed", float(self.hparams["cnt_samples_processed"]))
        pl_module.log("tokens_processed", float(self.hparams["cnt_tokens_processed"]))
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            self.maybe_save_on_cnt_samples(trainer, pl_module)
            self.maybe_save_on_timer(trainer, pl_module)
            pl_module.hparams["cnt_steps"] += 1
