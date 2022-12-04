from collections import OrderedDict
from torch import nn
from langmo.config.pretrain import ConfigPretrain
from langmo.config.glue import _glue_postprocess


_CLASSIFICATION_HEAD_CONFIG_TEMPLATE = {
    "hidden_size": 100,
    "n_layers": 2,
    "activation": "tanh",
    "dropout_p": 0.1,
    "num_labels": 2,
    "loss": "cross_entropy",
}

_REGRESSION_HEAD_CONFIG_TEMPLATE = {
    "hidden_size": 100,
    "n_layers": 1,
    "activation": "tanh",
    "dropout_p": 0.1,
    "num_labels": 1,
    "loss": "mse_loss"
}

_HEAD_ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}

_HEAD_LOSS_MAP = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse_loss": nn.MSELoss,
}


class HeadConfig(dict):
    def __init__(self, conf_dict=None):
        if conf_dict is not None:
            for key, val in conf_dict.items():
                self[key] = val

        if self["num_labels"] == 1:
            for key, val in _REGRESSION_HEAD_CONFIG_TEMPLATE.items():
                if key not in self:
                    self[key] = val

        elif self["num_labels"] > 1:
            for key, val in _CLASSIFICATION_HEAD_CONFIG_TEMPLATE.items():
                if key not in self:
                    self[key] = val


    def get_key(self, key):
        val = self.get(key)
        if val is not None:
            if key == "activation":
                return _HEAD_ACTIVATION_MAP[val]
            elif key == "loss":
                return _HEAD_LOSS_MAP[val]
            else:
                return val
        return val


class TaskConfigs(OrderedDict):
    # name_task is needed for compatibility with GLUE CONFIG
    def __init__(self, params):
        for key, val in params.items():
            if key != "tasks":
                self[key] = val
        self["tasks"] = OrderedDict()
        for task_name in sorted(list(params["tasks"])):
            task = params["tasks"][task_name]
            # TODO: here we create a new config for each glue task
            # by adding to the single task config the whole config
            # could only add glue relevant parts
            self["tasks"][task_name] = {
                "name_task": task_name,
                "siamese": False,
                **task,
                **{i: j for i, j in params.items() if i not in ["tasks", "name_task"]},
            }

            if task_name == "mlm":
                continue
            # TODO: this I scorporated from the GLUEConfig logic to reuse here
            # though to do this the GLUEConfig logic is worse
            _glue_postprocess(self["tasks"][task_name])

            # get num_labels from glue defaults
            if not "head_config" in self["tasks"][task_name]:
                self["tasks"][task_name]["head_config"] = {}

            self["tasks"][task_name]["head_config"]["num_labels"] = self["tasks"][task_name]["num_labels"]

            self["tasks"][task_name]["head_config"] = HeadConfig(
                self["tasks"][task_name]["head_config"]
            )



class ConfigMultitask(ConfigPretrain):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["path_val_corpus"] = None
        self.defaults["shuffle"] = False
        self.defaults["multitask_strategy"] = "sequential"
        self.defaults["save_predictions"] = False
        self.defaults["continuous_finetune"] = False
        self.required_options.add("tasks")
