import os
import subprocess
import sys
import unittest
from pathlib import Path
from shutil import rmtree

import nltk

nltk.download("punkt")
nltk.download("stopwords")


class LangmoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Path("./data/tokenized").mkdir(exist_ok=True, parents=True)
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
        os.environ["NUM_GPUS_PER_NODE"] = "0"
        os.environ["PROTONN_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["WANDB_MODE"] = "disabled"

    @classmethod
    def tearDownClass(cls):
        rmtree("./tests/test_output", ignore_errors=True)
        rmtree("./tests/test_data/tokenized", ignore_errors=True)

    @staticmethod
    def run_langmo(module, *args):
        cmd = [sys.executable, "-m", f"langmo.{module}", *args]
        print(f"cmd: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def test_pretrain(self):
        self.run_langmo("training.mlm", "./tests/test_params/pretraining_minimal_test.yaml")

    def test_GLUE_MNLI(self):
        yaml = "./tests/test_params/fine_tune_minimal_test.yaml"
        self.run_langmo("training.glue", yaml, "mnli")

    def test_GLUE_MNLI_siamese(self):
        yaml = "tests/test_params/fine_tune_siamese_minimal_test.yaml"
        self.run_langmo("training.glue", yaml, "mnli")

    def test_GLUE_RTE(self):
        yaml = "tests/test_params/fine_tune_minimal_test.yaml"
        self.run_langmo("training.glue", yaml, "rte")

    def test_GLUE_RTE_siamese(self):
        yaml = "tests/test_params/fine_tune_siamese_minimal_test.yaml"
        self.run_langmo("training.glue", yaml, "rte")

    def test_QA_squad(self):
        yaml = "tests/test_params/fine_tune_minimal_test.yaml"
        self.run_langmo("training.qa", yaml, "squad")

    def test_QA_squad_v2(self):
        yaml = "tests/test_params/fine_tune_minimal_test.yaml"
        self.run_langmo("training.qa", yaml, "squad_v2")


if __name__ == "__main__":
    unittest.main()
