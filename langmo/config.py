import logging
import os
import sys
from pathlib import Path

import yaml
from langmo.log_helper import set_root_logger
from langmo.utils import get_unique_results_path, parse_float
# from protonn.distributed import dist_adapter as da
from protonn.utils import get_time_str, load_json


def is_yaml_config(path):
    return path.is_file() and path.suffix in {".yaml", ".yml"}


def is_resume_run(path):
    _logger = logging.getLogger(__name__)
    _logger.info("starting new experiment")
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
    def __init__(self, name_task, is_master=False, param_path=None):
        set_root_logger()
        _logger = logging.getLogger(__name__)
        if len(sys.argv) < 2 and param_path == None:
            print("run main.py config.yaml")
            print("or")
            print("run main.py logs/path/to/snapshot/epoc10_step42000")
            exit(-1)

        if param_path == None:
            path = Path(sys.argv[1])
        else:
            path = Path(param_path)
        self.set_defaults()
        self._is_master = is_master

        if is_yaml_config(path):
            _logger.info("starting new experiment")
            self.read_from_yaml_and_set_default(path, name_task)
            # self.add_distributes_info()

        # TODO: this breaks finetuning!
        # TODO: not even makes sense, it's not about path corpus, it's about path model
        # path_model = self["path_corpus"]
        # if is_resume_run(path_model):
        #     # TODO: decide what to do when e.g. cnt_workers changed
        #     _logger.info("resuming from checkpoint")
        #     self.update(load_resume_run_params(path_model))
        # else:
        #     _logger.info("not resuming")

    def add_distributes_info(self):
        # cnt_workers = da.world_size()
        batch_size = self["batch_size"]
        acc_batches = self["accumulate_batches"]
        self["cnt_workers"] = cnt_workers
        self["batch_size_effective"] = batch_size * cnt_workers * acc_batches

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
            gradient_clip_val=0.0,
            accumulate_batches=1,
            percent_warmup=6.0,
            log_every_n_steps=50,
            seconds_between_snapshots=3600,
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
