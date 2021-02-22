import os
import platform
import sys
from pathlib import Path

import horovod.torch as hvd
import yaml
from protonn.utils import get_time_str


def get_unique_results_path(base):
    hostname = platform.node().split(".")[0]
    new_path = os.path.join(base, f"{get_time_str()}_{hostname}")
    # TODO: make this trully unique
    return new_path


def parse_float(dic, key):
    if key in dic:
        if isinstance(dic[key], str):
            dic[key] = float(dic[key])


def load_config(name_task):
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        exit(-1)
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params_user = yaml.load(cfg, Loader=yaml.SafeLoader)
    parse_float(params_user, "initial_lr")
    parse_float(params_user, "max_lr")
    parse_float(params_user, "eps")
    parse_float(params_user, "beta1")
    parse_float(params_user, "beta2")
    # default params
    params = dict(
        test=False,
        precision=32,
        batch_size=32,
        randomize=False,
        path_results="./logs",
        create_unique_path=True,
        uncase=False,
        cnt_epochs=5,
        eps=1e-6,
        beta1=0.9,
        beta2=0.999,
        max_lr=5e-5,
        initial_lr=0.0,
        cnt_warmup_steps=500,
        cnt_training_steps=500000,
    )
    params.update(params_user)
    name_project = f"{name_task}{'_test' if params['test'] else ''}"
    params["name_project"] = name_project
    params["path_results"] = os.path.join(params["path_results"], name_project)
    if params["create_unique_path"]:
        params["path_results"] = get_unique_results_path(params["path_results"])
    if hvd.rank() == 0:
        (Path(params["path_results"]) / "wandb").mkdir(parents=True, exist_ok=True)
    # Convert to "FP16" to (int) 16
    if isinstance(params["precision"], str):
        params["precision"] = int(params["precision"].lower().replace("fp", ""))

    return params
