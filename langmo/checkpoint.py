import pytorch_lightning as pl
from pathlib import Path


class CheckpointEveryNSteps(pl.Callback):
    def __init__(
        self,
        save_step_frequency,
        prefix="checkpoints",
    ):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_batch_end(self, trainer: pl.Trainer, _):
        # TODO: move saving logic to a separate method to reuse in in save_at_epoch_end
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            path_checkpoint = Path(trainer.model.hparams["path_results"]) / \
                self.prefix / \
                f"epoch{epoch}_step{global_step}"
            trainer.save_checkpoint(path_checkpoint / "PL_model.ckpt")
            path_hf = path_checkpoint / "hf"
            trainer.model.net.save_pretrained(path_hf)
            trainer.model.tokenizer.save_pretrained(path_hf)
