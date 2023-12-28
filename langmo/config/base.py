import logging
import os

# import time
from pathlib import Path

import yaml
from langmo.log_helper import set_root_logger
from langmo.utils import parse_float
from protonn.experiment_config import BaseConfig
from protonn.utils import get_time_str
import platform

CONFIG_OPTIONS = {
    "snapshot_strategy": ["per_epoch", "best_only", "none"],
    "encoder_wrapper": ["cls", "pooler", "lstm"],
    # TODO: related to todo in langmo.benchmarks.base
    # to allow different heads
    # "model_head": ["topmlp2", "lstm"]
}

TASKTOMETRIC = {
    "cola": "val_matthews_corr",
    "stsb": "val_pearson_corr",
    "mrpc": "val_accuracy",
    "qnli": "val_accuracy",
    "qqp": "val_accuracy",
    "rte": "val_accuracy",
    "sst2": "val_accuracy",
    "wnli": "val_accuracy",
    "mnli": "val_accuracy",
    "mnli-mm": "val_accuracy",
    "boolq": "val_accuracy",
    # TODO: for future better integratio
    # between NLI/GLUE
    "NLI": "val_acc_matched",
    "squad": "exact_match",
    "squad_v2": "exact_match",
}


CALLBACK_DEFAULTS = {
    "mlm": {
        "monitor": {
            "working_directory": None,
            "module": "langmo.callbacks.monitor",
            "class_name": "Monitor",
        }
    }
}

DEFAULT_OPTIMIZER = {
    "class_name": "AdamW",
    "module": "torch.optim",
    "working_directory": None,
    "params": {
        "betas": (0.9, 0.999),
        "eps": 1e-6,
    },
}


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


class LangmoConfig(BaseConfig):
    def __init__(self, name_task, param_path=None, is_master=True):
        set_root_logger()
        super().__init__(name_task, param_path, is_master)

        if "working_directory" not in self["optimizer"]:
            self["optimizer"]["working_directory"] = None
        if "params" not in self["optimizer"]:
            self["optimizer"]["params"] = {}

    # TODO: This os overriding parents, think how to reuse
    # TODO: we have near identical method in PROTONN
    def read_from_yaml_and_set_default(self, path, name_task):
        _logger = logging.getLogger(__name__)
        user_config = load_yaml_config(path)
        for key, value in user_config.items():
            if key not in self.defaults and key not in self.required_options and key != "suffix":
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
                if key == "metric_to_monitor":
                    value = TASKTOMETRIC.get(self["name_task"], None)
                if self._is_master:
                    _logger.warning(f"setting parameter {key} to default value {value}")
                self[key] = value

        self.update(user_config)

        name_project = name_task
        if self["test"]:
            name_project += "_test"
            self["cnt_epochs"] = 3
        if "suffix" in user_config:
            name_project += f"_{user_config['suffix']}"
        self["name_project"] = name_project
        self["timestamp"] = get_time_str()
        # Convert to "FP16" to (int) 16
        if isinstance(self["precision"], str):
            self["precision"] = int(self["precision"].lower().replace("fp", ""))
        # TODO: we put it here for now for simplicitly
        # this needs to be revisited when we do model parallel
        # TODO: also we whould think what we do when we resume with different number of workers
        self._postprocess()
        self._validate()

    def _validate(self):
        for key, val in self.items():
            if key in CONFIG_OPTIONS and val not in CONFIG_OPTIONS[key]:
                raise Exception(
                    f"Option {key} does not allow value {val}."
                    f"One among {'|'.join(CONFIG_OPTIONS[key])} must be picked"
                )

    def _postprocess(self):
        pass

    def set_defaults(self):
        super().set_defaults()

        self.defaults["cnt_gpus_per_node"] = int(os.environ["NUM_GPUS_PER_NODE"])

        self.defaults["classifier"] = "huggingface"
        self.defaults["test"] = False
        self.defaults["precision"] = 32
        self.defaults["batch_size"] = 32
        self.defaults["padding"] = "max_length"
        self.defaults["max_length"] = 128
        self.defaults["randomize"] = False
        self.defaults["create_unique_path"] = True
        self.defaults["uncase"] = False
        self.defaults["cnt_epochs"] = 5
        self.defaults["weight_decay"] = 0
        self.defaults["max_lr"] = 5e-5
        self.defaults["initial_lr"] = 0.0
        self.defaults["tokenizer_name"] = None
        self.defaults["gradient_clip_val"] = 0.0
        self.defaults["accumulate_batches"] = {1: 1}
        self.defaults["percent_warmup"] = 6.0
        self.defaults["log_every_n_steps"] = 50
        self.defaults["minutes_between_snapshots"] = 60
        self.defaults["overwrite_timer_snapshot"] = True
        self.defaults["num_sanity_val_steps"] = -1
        self.defaults["metric_to_monitor"] = None
        self.defaults["snapshot_strategy"] = "per_epoch"
        self.defaults["replace_hf_config"] = {}
        self.defaults["seed"] = 0
        self.defaults["params_without_weight_decay"] = ["bias", "gamma", "beta", "LayerNorm", "layer_norm"]
        self.defaults["callbacks"] = None
        self.defaults["snapshot_schedule"] = None
        self.defaults["optimizer"] = DEFAULT_OPTIMIZER

        self.required_options = set()
        self.required_options.add("model_name")

    def get_run_folder(self):
        path_results = Path(self["path_base"])
        path_results /= self["name_task"]
        path_results /= self["model_name"]
        timestamp = self["timestamp"][:-3]
        hostname = platform.node().split(".")[0]
        # run_folder = f"{timestamp}_w{workers}_lr{lr:.4f}_s{seed}_{hostname}"
        # lr = self["max_lr"] * self["cnt_workers"]
        seed = self["seed"]
        run_folder = f"{timestamp}_s{seed}_{hostname}"
        path_results /= run_folder

        # self["path_results"] = os.path.join(self["path_results"], self["name_project"])
        # # TODO: it's hacky, but for the time being for langmo
        # # ideally should be a list of names to concat into path
        # if "model_name" in self:
        #     self["path_results"] = os.path.join(self["path_results"], self["model_name"])

        # workers = self["cnt_workers"]
        # TODO: this might be dynamic
        # lr = self["max_lr"] * self["cnt_workers"]
        # TODO: make this trully unique
        return path_results


# TODO: where is this coming from???
# here is a good place to check if BS changed etc
# class ConfigResume(Config):
#     def __init__(self, name_task, old_params, is_master=False, param_path=None):
#         self.old_params = old_params
#         super().__init__(name_task, is_master=is_master, param_path=param_path)

#     def set_defaults(self):
#         self.defaults = self.old_params
#         self.required_options = set()
