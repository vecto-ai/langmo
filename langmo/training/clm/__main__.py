import socket

from protonn.pl.cluster_mpi import MPIClusterEnvironment
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as tr_logging

from .config import ConfigPretrain as Config
from langmo.log_helper import set_root_logger
from langmo.trainer import get_trainer
from langmo.utils.resolve_callbacks import init_callbacks

from ..base_data import TextDataModule
from ..mlm.plmodel import PLModel
from .data import BatchIter
from protonn.experiment import Experiment

# from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
# from langmo.nn.utils import reinit_model


def build_model(params):
    # TODO: support loading weights for continuation of pretraining
    tokenizer = AutoTokenizer.from_pretrained(params["tokenizer_name"])
    # tokenizer.pad_token = tokenizer.eos_token

    if params["model_name"] == "cnet":
        from langmo.nn.cnet import get_mlmodel

        net = get_mlmodel(params)
    else:
        config = AutoConfig.from_pretrained(params["model_name"])
        config.update(params["replace_hf_config"])
        config.is_decoder = True
        net = AutoModelForCausalLM.from_config(config)
    net.train()

    model = PLModel(
        net=net,
        tokenizer=tokenizer,
        params=params,
    )
    return model


def get_run_name(params):
    name_run = params["model_name"]
    # TODO: revisit this when we have model parallel training
    name_run += f"_{params['timestamp']}"
    # name_run += f"_bs{params['batch_size'] * params['cnt_workers']}"
    name_run += f"_lr{params['max_lr']}"
    # TODO: add batch size
    # name_run += f"_wd{params['weight_decay']}"
    # name_run += f"_bs{params['batch_size_effective']}"
    return name_run


class CLMExperiment(Experiment):
    def __init__(self):
        super().__init__(name_task="pretrain")
        set_root_logger()
        cluster_env = MPIClusterEnvironment()
        if cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
        self.params = Config(
            name_task=self.name_task,
            is_master=self.is_master,
        )
        # TODO: make logging report rank and size and use logging
        self.params["name_run"] = self.params.get_run_name()
        # want to do it in super init but this depends on custom config....
        self.maybe_create_unique_path()
        self.cluster_env.barrier()

    def run(self):

        callbacks = init_callbacks(self.params["callbacks"])
        trainer = get_trainer(self.params, self.cluster_env, callbacks)
        self.params["cnt_workers"] = trainer.world_size
        # TODO: get current accumulation of batched dynamically and log per epoch
        # params["batch_size_effective"] = (
        #     params["batch_size"] * params["cnt_workers"] * params["accumulate_batches"]
        # )
        print(f"!!! Starting on host {socket.gethostname()}, p {trainer.global_rank} of {trainer.world_size}")
        model = build_model(self.params)

        data_module = TextDataModule(
            batch_iterator_cls=BatchIter,
            cluster_env=self.cluster_env,
            tokenizer=model.tokenizer,
            params=self.params,
        )
        model.hparams["corpus"] = data_module.corpus.metadata

        model.pylogger.info("calling fit")
        trainer.fit(model, data_module)
        print("Training done")


if __name__ == "__main__":
    experiment = CLMExperiment()
    experiment.run()
