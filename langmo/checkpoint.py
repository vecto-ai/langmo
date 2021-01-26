from pathlib import Path

import horovod.torch as hvd
import pytorch_lightning as pl


class BaseNStepCallback(pl.Callback):
    def __init__(
        self,
        save_step_frequency,
        prefix="checkpoints",
    ):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def get_path_destination(self, trainer):
        global_step = trainer.global_step
        epoch = trainer.current_epoch
        path_destination = (
            Path(trainer.model.hparams["path_results"])
            / self.prefix
            / f"epoch{epoch}_step{global_step}"
        )
        return path_destination


class CheckpointEveryNSteps(BaseNStepCallback):
    def on_batch_end(self, trainer: pl.Trainer, _):
        if hvd.rank() != 0:
            return
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            path_checkpoint = self.get_path_destination(trainer)
            trainer.save_checkpoint(path_checkpoint / "PL_model.ckpt")
            path_hf = path_checkpoint / "hf"
            trainer.model.net.save_pretrained(path_hf)
            trainer.model.tokenizer.save_pretrained(path_hf)
            metadata = trainer.datamodule.corpus.metadata
            trainer.model.save_metadata(metadata, path_checkpoint)


# TODO: this moves to CLI. but we can create something like .doit file here
class ScheduleEval(BaseNStepCallback):
    def on_batch_end(self, trainer: pl.Trainer, _):
        if hvd.rank() != 0:
            return
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            # path_dst = self.get_path_destination(trainer)
            print("DON'T SCHEDULE EVAL HERE, USE CLI")
