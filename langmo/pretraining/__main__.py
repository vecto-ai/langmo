import socket

from langmo.callbacks.model_snapshots_schedule import Monitor
from langmo.cluster_mpi import MPIClusterEnvironment
from langmo.config import ConfigPretrain as Config
from langmo.log_helper import set_root_logger
from langmo.trainer import get_trainer
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as tr_logging

from .data import TextDataModule
from .plmodel import PLModel

# from langmo.checkpoint import CheckpointEveryNSteps  # , ScheduleEval
# from langmo.nn.utils import reinit_model


def build_model(params):
    # support loading weights for continuation of pretraining
    tokenizer = AutoTokenizer.from_pretrained(params["tokenizer_name"])
    config = AutoConfig.from_pretrained(params["model_name"])
    # update config with replace_hf_configs provided in yaml file
    config.update(params["replace_hf_config"])

    net = AutoModelForMaskedLM.from_config(config)
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


def main():
    set_root_logger()
    cluster_env = MPIClusterEnvironment()
    if cluster_env.global_rank() != 0:
        tr_logging.set_verbosity_error()  # to reduce warning of unused weights
    name_task = "pretrain"
    params = Config(name_task=name_task, cluster_env=cluster_env)
    # TODO: make logging report rank and size and use logging
    params["name_run"] = get_run_name(params)
    cluster_env.barrier()

    trainer = get_trainer(params, cluster_env, [Monitor()])
    params["cnt_workers"] = trainer.world_size
    params["batch_size_effective"] = params["batch_size"] * params["cnt_workers"] * params["accumulate_batches"]
    print(f"!!! Starting on host {socket.gethostname()}, p {trainer.global_rank} of {trainer.world_size}")
    model = build_model(params)

    data_module = TextDataModule(
        tokenizer=model.tokenizer,
        params=params,
    )
    model.hparams["corpus"] = data_module.corpus.metadata

    model.pylogger.info("calling fit")
    trainer.fit(model, data_module)
    print("Training done")


if __name__ == "__main__":
    main()
