import os
import unittest

from protonn.pl.cluster_mpi import MPIClusterEnvironment
from transformers import logging as tr_logging

from langmo.config.base import LangmoConfig


class DDPParams(unittest.TestCase):
    def setUp(self):
        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PROTONN_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["WANDB_MODE"] = "disabled"
        self.cluster_env = MPIClusterEnvironment()
        if self.cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights

    def test_langmo_base_config_dafault_callback(self):
        params = LangmoConfig(
            name_task="test",
            param_path="tests/test_params/ddp_strategy_params_test.yaml",
        )
        params["name_run"] = "test"
        self.cluster_env.barrier()

        self.assertEqual(params["ddp_strategy_params"]["process_group_backend"], "gloo")
        self.assertIn("find_unused_parameters", params["ddp_strategy_params"])
        self.assertTrue(params["ddp_strategy_params"]["find_unused_parameters"])


if __name__ == "__main__":
    unittest.main()
