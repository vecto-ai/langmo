import pytorch_lightning as pl
from pathlib import Path


# TODO: allow user to provide paltform-specific headers
# TODO: this will be probably done though protonn
header_ABCI = (
    "#!/bin/bash\n"
    "#$ -cwd\n"
    "#$ -l rt_F=1\n"
    "#$ -l h_rt=04:00:00\n"
    "#$ -N NLP\n"
    "#$ -j y\n"
    "#$ -o $JOB_NAME.o$JOB_ID\n"
)


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
        path_destination = Path(trainer.model.hparams["path_results"]) / \
            self.prefix / \
            f"epoch{epoch}_step{global_step}"
        return path_destination


class CheckpointEveryNSteps(BaseNStepCallback):
    def on_batch_end(self, trainer: pl.Trainer, _):
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            path_checkpoint = self.get_path_destination(trainer)
            trainer.save_checkpoint(path_checkpoint / "PL_model.ckpt")
            path_hf = path_checkpoint / "hf"
            trainer.model.net.save_pretrained(path_hf)
            trainer.model.tokenizer.save_pretrained(path_hf)


class ScheduleEval(BaseNStepCallback):
    def create_job_files(self, path):
        path_config = path / "evalconfig.yaml"
        # TODO: do we need to set execution rights?
        with open(path / "evaljobscript.sh", "w") as file_jobscript:
            # TODO: get platform-specific headers from config
            # e.g. for ABCI the nodes, the groups etc
            # python version
            # expect langmo to be installed by user
            # TODO: use "schedule all" API
            file_jobscript.write(header_ABCI)
            file_jobscript.write(f"python3 -m langmo.benchmarks.NLI {path_config}\n")
        # TODO: this is not gonna scale to multiple benchmarks - use some defaults
        with open(path_config, "w") as file_jobscript:
            file_jobscript.write(f"model_name: {path / 'hf'}\n")
            file_jobscript.write(f"path_results: {path / 'eval'}\n")
            file_jobscript.write(f"cnt_epochs: 10\n")
        # TODO: schedule N runs
        # TODO: make wandb offline when schedule massive runs

    def on_batch_end(self, trainer: pl.Trainer, _):
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            path_dst = self.get_path_destination(trainer)
            self.create_job_files(path_dst)
