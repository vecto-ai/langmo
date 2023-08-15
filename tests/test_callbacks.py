import os
import unittest
from shutil import copytree, rmtree

from protonn.pl.cluster_mpi import MPIClusterEnvironment
from transformers import logging as tr_logging

from langmo.callbacks.model_snapshots_schedule import Monitor
from langmo.config import ConfigPretrain as Config
from langmo.config.base import LangmoConfig
from langmo.trainer import get_trainer
from langmo.utils.resolve_callbacks import init_callbacks


class Callbacks(unittest.TestCase):
    def setUp(self):
        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PROTONN_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["WANDB_MODE"] = "disabled"
        self.cluster_env = MPIClusterEnvironment()
        if self.cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights

    def test_backward_compatibility(self):
        params = Config(
            name_task="test_callbacks_1",
            cluster_env=self.cluster_env,
            param_path="tests/test_params/pretraining_minimal_callback_test.yaml",
        )
        params["name_run"] = "test_callbacks_1"
        self.cluster_env.barrier()

        callbacks = init_callbacks(params["callbacks"])
        cb1 = get_trainer(params, self.cluster_env, callbacks).callbacks

        params = Config(
            name_task="test_callbacks_2",
            cluster_env=self.cluster_env,
            param_path="tests/test_params/pretraining_minimal_test.yaml",
        )
        params["name_run"] = "test_callbacks_2"
        self.cluster_env.barrier()

        cb2 = get_trainer(params, self.cluster_env, [Monitor()]).callbacks

        for a, b in zip(cb1, cb2):
            self.assertIsInstance(a, type(b))

    def test_langmo_base_config_dafault_callback(self):
        params = LangmoConfig(
            name_task="test_callbacks_2",
            cluster_env=self.cluster_env,
            param_path="tests/test_params/minimal_callback_test.yaml",
        )
        params["name_run"] = "test_callbacks_2"
        self.cluster_env.barrier()

        self.assertIs(params["callbacks"], None)


class ExternalCallbacks(unittest.TestCase):
    def setUp(self):
        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PROTONN_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["WANDB_MODE"] = "disabled"
        # emulate external module by moving monitor outside of the project
        # assuming that Monitor does not import any langmo modules
        self.temp_dir = "../_callback_test/extra_suffix/"
        if os.path.isdir("../_callback_test/"):
            rmtree("../_callback_test/", ignore_errors=True)

        os.makedirs(self.temp_dir, exist_ok=True)
        self.new_langmo_loc = os.path.join(self.temp_dir, "langmo_callback")
        copytree("langmo/callbacks/", self.new_langmo_loc)
        self.cluster_env = MPIClusterEnvironment()
        if self.cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights

    def tearDown(self):
        rmtree("../_callback_test/", ignore_errors=True)

    def test_external(self):
        params = Config(
            name_task="test_callbacks_1",
            cluster_env=self.cluster_env,
            param_path="tests/test_params/pretraining_minimal_external_callback_test.yaml",
        )
        params["name_run"] = "test_callbacks_1"
        self.cluster_env.barrier()

        callbacks = init_callbacks(params["callbacks"])
        cb1 = get_trainer(params, self.cluster_env, callbacks).callbacks
        self.assertEqual(cb1[0].__class__.__name__, "Monitor")


if __name__ == "__main__":
    unittest.main()
