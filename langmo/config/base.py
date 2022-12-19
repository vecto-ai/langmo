import logging
import os
import time
from pathlib import Path

import yaml
from langmo.log_helper import set_root_logger
from langmo.utils import parse_float
from protonn.experiment_config import BaseConfig
from protonn.utils import get_time_str
from transformers import set_seed

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
    def __init__(self, name_task, cluster_env, param_path=None):
        set_root_logger()
        super().__init__(name_task, cluster_env, param_path)

    # TODO: This os overriding parents, think how to reuse
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
        if "suffix" in user_config:
            name_project += f"_{user_config['suffix']}"
        else:
            if self["test"]:
                name_project += "_test"
        self["name_project"] = name_project
        self["timestamp"] = get_time_str()
        # Convert to "FP16" to (int) 16
        if isinstance(self["precision"], str):
            self["precision"] = int(self["precision"].lower().replace("fp", ""))
        set_seed(self["seed"])
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
        self.defaults = dict(
            cnt_gpus_per_node=int(os.environ["NUM_GPUS_PER_NODE"]),
            distributed_backend="gloo",
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
            num_sanity_val_steps=-1,
            metric_to_monitor=None,
            snapshot_strategy="per_epoch",
            replace_hf_config={},
            seed=int(time.time()),
            params_without_weight_decay=["bias", "gamma", "beta", "LayerNorm", "layer_norm"],
        )
        self.required_options = set()
        self.required_options.add("model_name")


# TODO: this needs to be rewritten
# here is a good place to check if BS changed etc
# class ConfigResume(Config):
#     def __init__(self, name_task, old_params, is_master=False, param_path=None):
#         self.old_params = old_params
#         super().__init__(name_task, is_master=is_master, param_path=param_path)

#     def set_defaults(self):
#         self.defaults = self.old_params
#         self.required_options = set()


class ConfigFinetune(LangmoConfig):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["siamese"] = False
        self.defaults["freeze_encoder"] = False
        self.defaults["encoder_wrapper"] = "pooler"
        self.defaults["shuffle"] = False
        self.defaults["cnt_seps"] = -1
        self.defaults["save_predictions"] = False
