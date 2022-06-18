import unittest
import subprocess
from shutil import rmtree

import sys
import os
from pathlib import Path


class RunTestCase:
    def does_run(self, path):
        py_exec = Path(sys.executable).name
        if "fine_tune" in path:
            subprocess.run([py_exec, "-m", "langmo.benchmarks.NLI", path], check=True)
        elif "pretraining" in path:
            subprocess.run([py_exec, "-m", "langmo.pretraining", path], check=True)


class Tests(unittest.TestCase):
    def setUp(self):
        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "GLOO"
        os.environ["WANDB_MODE"] = "disabled"
        self.tester_does_run = RunTestCase()

    def tearDown(self):
        rmtree("./tests/test_output", ignore_errors=True)

    def test_does_run_pretrain(self):
        self.tester_does_run.does_run("./tests/test_params/pretrain_minimal_test.yaml")

    def test_does_run_fine_tune(self):
        self.tester_does_run.does_run("./tests/test_params/fine_tune_minimal_test.yaml")

    def test_does_run_siamese(self):
        self.tester_does_run.does_run("tests/test_params/fine_tune_siamese_minimal_test.yaml")


if __name__ == "__main__":
    unittest.main()
