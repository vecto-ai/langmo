import logging
import os
import os.path
import sys
import time
from pathlib import Path

import yaml
from langmo.log_helper import set_root_logger
from langmo.utils import get_unique_results_path, parse_float
from protonn.utils import get_time_str, load_json
from transformers import set_seed


def is_yaml_config(path):
    return path.is_file() and path.suffix in {".yaml", ".yml"}


def is_resume_run(path):
    _logger = logging.getLogger(__name__)
    _logger.info("resuming the experiment")
    path = Path(path) / "metadata.json"
    return path.is_file() and path.suffix == ".json"


def load_resume_run_params(path):
    path = Path(path)
    params = load_json(path / "metadata.json")
    paths = dict(
        metadata=str(path),
        checkpoint=str(path / "PL_model.ckpt"),
        hf=str(path / "hf"),
    )
    params["resume"] = paths
    return params


def load_yaml_file(path):
    with open(path, "r") as cfg:
        data = yaml.load(cfg, Loader=yaml.SafeLoader)
    return data


def load_yaml_or_empty(path):
    path = Path(path)
    if path.exists():
        with open(path, "r") as cfg:
            data = yaml.load(cfg, Loader=yaml.SafeLoader)
    else:
        data = {}
    return data


# TODO: raname
def load_yaml_config(path_config):
    params_user = load_yaml_file(path_config)
    parse_float(params_user, "initial_lr")
    parse_float(params_user, "max_lr")
    parse_float(params_user, "eps")
    parse_float(params_user, "beta1")
    parse_float(params_user, "beta2")
    return params_user


class Config(dict):
    def __init__(self, name_task, is_master, param_path=None):
        self.name_task = name_task
        set_root_logger()
        _logger = logging.getLogger(__name__)
        if len(sys.argv) < 2 and param_path is None:
            print("run main.py config.yaml")
            print("or")
            print("run main.py logs/path/to/snapshot/epoc10_step42000")
            exit(-1)

        if param_path is None:
            path = Path(sys.argv[1])
        else:
            path = Path(param_path)
        self.set_defaults()
        self._is_master = is_master

        if is_yaml_config(path):
            _logger.info("starting new experiment")
            self.read_from_yaml_and_set_default(path, name_task)
            # self.add_distributes_info()

    # def add_distributed_info(self):
    #     batch_size = self["batch_size"]
    #     acc_batches = self["accumulate_batches"]
    #     self["cnt_workers"] = cnt_workers
    #     self["batch_size_effective"] = batch_size * cnt_workers * acc_batches

    def read_from_yaml_and_set_default(self, path, name_task):
        _logger = logging.getLogger(__name__)
        user_config = load_yaml_config(path)
        for key, value in user_config.items():
            if (
                key not in self.defaults
                and key not in self.required_options
                and key != "suffix"
            ):
                raise RuntimeError(f"got unexpected key in user config\t{key}: {value}")
            # print(key, value)
        for key in self.required_options:
            if key not in user_config:
                raise RuntimeError(f"required key not in config {key}")
        for key, value in self.defaults.items():
            if key not in user_config:
                # tokenizer defaults to model_name if absent
                if key == "tokenizer_name":
                    value = user_config.get(key, user_config["model_name"])
                if self._is_master:
                    _logger.warning(f"setting parameter {key} to default value {value}")
                self[key] = value

        self.update(user_config)
        name_project = name_task
        if "suffix" in user_config:
            name_project += f"_{user_config['suffix']}"
        else:
            if user_config["test"]:
                name_project += "_test"
        self["name_project"] = name_project
        self["path_results"] = os.path.join(self["path_results"], name_project)
        self["timestamp"] = get_time_str()
        if self["create_unique_path"]:
            self["path_results"] = get_unique_results_path(
                self["path_results"],
                self["model_name"],
                self["timestamp"],
            )
        # Convert to "FP16" to (int) 16
        if isinstance(self["precision"], str):
            self["precision"] = int(self["precision"].lower().replace("fp", ""))
        set_seed(self["seed"])
        # TODO: we put it here for now for simplicitly
        # this needs to be revisited when we do model parallel
        # TODO: also we whould think what we do when we resume with different number of workers

    def set_defaults(self):
        self.defaults = dict(
            name_task=self.name_task,
            cnt_gpus_per_node=int(os.environ["NUM_GPUS_PER_NODE"]),
            test=False,
            precision=32,
            batch_size=32,
            padding="max_length",
            max_length=128,
            randomize=False,
            path_results="./logs",
            create_unique_path=True,
            uncase=False,
            cnt_epochs=5,
            eps=1e-6,
            weight_decay=0,
            beta1=0.9,
            beta2=0.999,
            max_lr=5e-5,
            initial_lr=0.0,
            tokenizer_name=None,
            gradient_clip_val=0.0,
            accumulate_batches=1,
            percent_warmup=6.0,
            log_every_n_steps=50,
            seconds_between_snapshots=3600,
            per_epoch_snapshot=True,
            replace_hf_config=dict(),
            seed=int(time.time()),
        )
        self.required_options = set()
        self.required_options.add("model_name")


class ConfigPretrain(Config):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["proba_masking"] = 0.12
        self.defaults["proba_random"] = 0.015
        self.defaults["mask_special_tokens"] = True
        self.required_options.add("path_corpus")
        self.required_options.add("path_val_corpus")
        self.required_options.add("cnt_samples_per_epoch")


class ConfigResume(Config):
    def __init__(self, name_task, old_params, is_master=False, param_path=None):
        self.old_params = old_params
        super().__init__(name_task, is_master=is_master, param_path=param_path)

    def set_defaults(self):
        self.defaults = self.old_params
        self.required_options = set()


class ConfigFinetune(Config):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["siamese"] = False
        self.defaults["freeze_encoder"] = False
        self.defaults["encoder_wrapper"] = "pooler"
        self.defaults["shuffle"] = False
        self.defaults["cnt_seps"] = -1


GLUETASKTOKEYS = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
}

GLUETASKTOMETRIC = {
    "cola": "matthews_correlation",
    "stsb": "pearson",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "wnli": "accuracy",
    "mnli": "accuracy",
    "mnli-mm": "accuracy",
}

GLUETASKTONUMLABELS = {
    "stsb": 1,
    "cola": 2,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "wnli": 2,
    "mnli": 3,
    "mnli-mm": 3,
}


class GLUEConfig(ConfigFinetune):
    def set_defaults(self):
        super().set_defaults()

        try:
            assert self.name_task in GLUETASKTOKEYS
        except:
            glue_tasks = list(self.defaults["task_to_keys"].keys())
            raise AssertionError(f"Task must be one of {glue_tasks}")

        self.defaults["sent1"] = GLUETASKTOKEYS[self.name_task][0]
        self.defaults["sent2"] = GLUETASKTOKEYS[self.name_task][1]
        self.defaults["num_labels"] = GLUETASKTONUMLABELS[self.name_task]
        self.defaults["metric_name"] = GLUETASKTOMETRIC[self.name_task]
        self.defaults["validation_split"] = (
            "validation_mismatched"
            if self.name_task == "mnli-mm"
            else "validation_matched"
            if self.name_task == "mnli"
            else "validation"
        )
