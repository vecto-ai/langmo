# pylint: disable=attribute-defined-outside-init
from shutil import rmtree

from .monitor import Monitor


class FinetuneMonitor(Monitor):
    def _find_best_checkpoint(self):
        best_checkpoint_metadata = max(
            self.hparams["train_logs"][:-1],
            key=lambda x: x[self.metric_to_monitor],
        )
        return best_checkpoint_metadata[self.metric_to_monitor]

    def _save_best_only(self, path_new_checkpoint, pl_module):
        if self.epoch == -1:
            self._save_hf_and_metadata(path_new_checkpoint, pl_module)
            return
        current_best_metric = self._find_best_checkpoint()
        existing_checkpoints = list(path_new_checkpoint.parent.iterdir())
        if self.hparams["train_logs"][-1][self.metric_to_monitor] > current_best_metric:
            other_checkpoints = next(path_new_checkpoint.parent.iterdir())
            print("OLD CHECKPOINT", other_checkpoints)
            pl_module.save_as_hf(path_new_checkpoint / "hf")
            pl_module.save_metadata(path_new_checkpoint)
            for existing_checkpoint in existing_checkpoints:
                # TODO: don't remove details
                # print("want to remove ", existing_checkpoint)
                if (existing_checkpoint / "hf").exists():
                    rmtree(existing_checkpoint / "hf")
