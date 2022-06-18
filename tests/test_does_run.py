import unittest
import subprocess
from shutil import rmtree

import sys
import os
from pathlib import Path


class DoesRunTestCase:
    def __init__(self):
        self.py_exec = Path(sys.executable).name

    def does_NLI_fine_tune_run(self, path):
        subprocess.run([self.py_exec, "-m", "langmo.benchmarks.NLI", path], check=True)

    def does_GLUE_fine_tune_run(self, path):
        subprocess.run([self.py_exec, "-m", "langmo.benchmarks.GLUE", path, "rte"], check=True)

    def does_pretrain_run(self, path):
        subprocess.run([self.py_exec, "-m", "langmo.pretraining", path], check=True)


class Tests(unittest.TestCase):
    def setUp(self):
        Path("./data/tokenized").mkdir(exist_ok=True, parents=True)
        self.tester_does_run = DoesRunTestCase()
        subprocess.run(
            [
                self.tester_does_run.py_exec,
                "-m",
                "langmo.utils.preprocess",
                "bert-base-uncased",
                "128",
                "./tests/test_data/sense",
                "./tests/test_data/tokenized/sense",
            ],
            check=True,
        )
        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "GLOO"
        os.environ["WANDB_MODE"] = "disabled"

    def tearDown(self):
        rmtree("./tests/test_output", ignore_errors=True)
        rmtree("./tests/test_data/tokenized", ignore_errors=True)

    def test_does_run_pretrain(self):
        self.tester_does_run.does_pretrain_run("./tests/test_params/pretraining_minimal_test.yaml")

    def test_does_run_fine_tune(self):
        self.tester_does_run.does_NLI_fine_tune_run(
            "./tests/test_params/fine_tune_minimal_test.yaml"
        )
        self.tester_does_run.does_GLUE_fine_tune_run(
            "./tests/test_params/fine_tune_minimal_test.yaml"
        )

    def test_does_run_siamese(self):
        self.tester_does_run.does_NLI_fine_tune_run(
            "tests/test_params/fine_tune_siamese_minimal_test.yaml"
        )
        self.tester_does_run.does_GLUE_fine_tune_run(
            "tests/test_params/fine_tune_siamese_minimal_test.yaml"
        )


if __name__ == "__main__":
    unittest.main()
