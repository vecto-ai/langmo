from timeit import default_timer as timer

import horovod.torch as hvd
import pytorch_lightning as pl


class PerfMonitor(pl.Callback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.time_start = timer()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.time_end = timer()
        if hvd.rank() == 0:
            epoch_time = self.time_end - self.time_start
            pl_module.hparams["train_logs"][-1]["epoch_time"] = epoch_time
            pl_module.hparams["train_logs"][-1]["samples_per_second"] = pl_module.hparams["cnt_samples_per_epoch"] / epoch_time
            pl_module.hparams["train_logs"][-1]["samples_per_second_worker"] = pl_module.hparams["train_logs"][-1]["samples_per_second"] / pl_module.hparams["cnt_workers"]

            pl_module.save_metadata()
            print(f"epoch end, done in {epoch_time} sec")
