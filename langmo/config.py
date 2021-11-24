import logging
import os
import sys
from pathlib import Path

import yaml
from langmo.utils import get_unique_results_path, parse_float
from protonn.utils import get_time_str, load_json


def is_yaml_config(path):
    return len(sys.argv) == 2 and path.is_file() and path.suffix in {".yaml", ".yml"}


def is_resume_run(path):
    path = path / "metadata.json"
    return path.is_file() and path.suffix == ".json"


def load_resume_run_params(path):
    params = load_json(path / "metadata.json")
    paths = dict(
        metadata=str(path),
        checkpoint=str(path / "PL_model.ckpt"),
        hf=str(path / "hf"),
    )
    params["resume"] = paths
    return params


def load_yaml_config(path_config):
    with open(path_config, "r") as cfg:
        params_user = yaml.load(cfg, Loader=yaml.SafeLoader)
    parse_float(params_user, "initial_lr")
    parse_float(params_user, "max_lr")
    parse_float(params_user, "eps")
    parse_float(params_user, "beta1")
    parse_float(params_user, "beta2")
    return params_user


class Config(dict):
    def __init__(self, name_task, is_master=False):
        if len(sys.argv) < 2:
            print("run main.py config.yaml")
            print("or")
            print("run main.py logs/path/to/snapshot/epoc10_step42000")
            exit(-1)

        path = Path(sys.argv[1])
        self.set_defaults()
        self._is_master = is_master

        if is_yaml_config(path):
            self.read_from_yaml_and_set_default(path, name_task)

        if is_resume_run(path):
            self.update(load_resume_run_params(path))

    def read_from_yaml_and_set_default(self, path, name_task):
        _logger = logging.getLogger(__name__)
        user_config = load_yaml_config(path)
        for key, value in user_config.items():
            if key not in self.defaults and key not in self.required_options and key != "suffix":
                raise RuntimeError(f"got unexpected key in user config {key}:{value}")
            # print(key, value)
        for key, value in self.defaults.items():
            if key not in user_config:
                if self._is_master:
                    _logger.warning(f"setting parameter {key} to default value {value}")
                self[key] = value
        for key in self.required_options:
            if key not in user_config:
                raise RuntimeError(f"required key not in config {key}")
        self.update(user_config)
        name_project = name_task
        # if "suffix" in params:
        #     name_project += f"_{params['suffix']}"
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
        # TODO: we put it here for now for simplicitly
        # this needs to be revisited when we do model parallel
        # TODO: also we whould think what we do when we resume with different number of workers

    def set_defaults(self):
        self.defaults = dict(
            test=False,
            use_gpu=True,
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
            cnt_warmup_steps=500,
            cnt_training_steps=500000,
            shuffle=False,
            gradient_clip_val=0.0,
            accumulate_batches=1,
        )
        self.required_options = set()
        self.required_options.add("model_name")


class ConfigPretrain(Config):

    def set_defaults(self):
        super().set_defaults()
        self.required_options.add("path_corpus")
        self.required_options.add("path_val_corpus")
        # TODO: remove this one
        self.required_options.add("cnt_samples_per_epoch")


class ConfigFinetune(Config):

    def set_defaults(self):
        super().set_defaults()
        self.defaults["siamese"] = False
        self.defaults["freeze_encoder"] = False
        self.defaults["encoder_wrapper"] = "pooler"
