import pytorch_lightning as pl
import horovod.torch as hvd
from timeit import default_timer as timer


class PerfMonitor(pl.Callback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.time_start = timer()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.time_end = timer()
        if hvd.rank() == 0:
            epoch_time = self.time_end - self.time_start
            pl_module.hparams["train_logs"][-1]["epoch_time"] = epoch_time
            pl_module.save_metadata()
            print(f"epoch end, done in {epoch_time} sec")
