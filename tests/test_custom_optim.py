import os
import subprocess
import sys
import unittest
from shutil import copytree, rmtree

import nltk
from protonn.pl.cluster_mpi import MPIClusterEnvironment
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as tr_logging

from langmo.callbacks.model_snapshots_schedule import Monitor
from langmo.training.mlm.config import ConfigPretrain as Config
from langmo.config.base import LangmoConfig
from langmo.trainer import get_trainer
from langmo.training.mlm.data import TextDataModule
from langmo.training.mlm.plmodel import PLModel
from langmo.utils.resolve_callbacks import init_callbacks

nltk.download("punkt")


def build_model(params):
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


class Optimizer(unittest.TestCase):
    def setUp(self):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "langmo.utils.preprocess",
                "bert-base-uncased",
                "128",
                "./tests/test_data/sense",
                "./tests/test_data/tokenized/sense",
            ],
            check=True,
        )

        os.makedirs("../.optim_test/optim/", exist_ok=True)
        f = open("../.optim_test/optim/__init__.py", mode="w")
        f.write("from torch.optim import SGD")
        f.close()

        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PROTONN_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["WANDB_MODE"] = "disabled"
        self.cluster_env = MPIClusterEnvironment()
        if self.cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights

    def tearDown(self):
        rmtree("./tests/test_output", ignore_errors=True)
        rmtree("./tests/test_data/tokenized", ignore_errors=True)
        rmtree("../.optim_test/", ignore_errors=True)

    def test_custom_optim(self):
        params = Config(
            name_task="test_custom_optim",
            cluster_env=self.cluster_env,
            param_path="tests/test_params/custom_optimizer.yaml",
        )
        params["name_run"] = "test_custom_optim"
        self.cluster_env.barrier()

        callbacks = init_callbacks(params["callbacks"])
        trainer = get_trainer(params, self.cluster_env, callbacks)
        model = build_model(params)

        data_module = TextDataModule(
            cluster_env=self.cluster_env,
            tokenizer=model.tokenizer,
            params=params,
        )
        model.hparams["corpus"] = data_module.corpus.metadata

        model.pylogger.info("calling fit")
        trainer.fit(model, data_module)
        self.assertEqual(model.optimizers().__class__.__name__, "LightningSGD")


if __name__ == "__main__":
    unittest.main()
