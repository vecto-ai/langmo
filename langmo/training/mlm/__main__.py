import socket

from langmo.log_helper import set_root_logger
from langmo.trainer import get_trainer
from langmo.utils.resolve_callbacks import init_callbacks
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as tr_logging

from .config import ConfigPretrain as Config
from .data import TextDataModule
from .plmodel import PLModel
from protonn.experiment import Experiment

# from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
# from langmo.nn.utils import reinit_model


def build_model(params):
    # TODO: support loading weights for continuation of pretraining
    tokenizer = AutoTokenizer.from_pretrained(params["tokenizer_name"])
    if params["model_name"] == "cnet":
        from langmo.nn.cnet import get_mlmodel

        net = get_mlmodel(params)
    else:
        config = AutoConfig.from_pretrained(params["model_name"])
        config.update(params["replace_hf_config"])
        net = AutoModelForMaskedLM.from_config(config)
    net.train()

    model = PLModel(
        net=net,
        tokenizer=tokenizer,
        params=params,
    )
    return model


class MLMExperiment(Experiment):
    def __init__(self):
        super().__init__(name_task="pretrain")
        # TODO: move to parent class (in langmo)
        set_root_logger()
        if not self.is_master:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
        self.params = Config(
            name_task=self.name_task,
            is_master=self.is_master,
        )
        self.params["name_run"] = self.params.get_run_name()
        # want to do it in super init but this depends on custom config....
        self.maybe_create_unique_path()
        self.cluster_env.barrier()
        # print(f"!!! Starting on host {socket.gethostname()}, p {trainer.global_rank} of {trainer.world_size}")

    def run(self):
        # TODO: make logging report rank and size and use logging

        callbacks = init_callbacks(self.params["callbacks"])
        trainer = get_trainer(self.params, self.cluster_env, callbacks)
        # TODO: this is done in add_distributed info
        self.params["cnt_workers"] = trainer.world_size
        # TODO: get current accumulation of batched dynamically and log per epoch
        # params["batch_size_effective"] = (
        #     params["batch_size"] * params["cnt_workers"] * params["accumulate_batches"]
        # )
        model = build_model(self.params)

        data_module = TextDataModule(
            cluster_env=self.cluster_env,
            tokenizer=model.tokenizer,
            params=self.params,
        )
        model.hparams["corpus"] = data_module.corpus.metadata

        model.pylogger.info("calling fit")
        trainer.fit(model, data_module)
        print("Training done")


if __name__ == "__main__":
    experiment = MLMExperiment()
    experiment.run()
