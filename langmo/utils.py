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


def load_config(name_task):
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        exit(-1)
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params_user = yaml.load(cfg, Loader=yaml.SafeLoader)
    # default params
    params = dict(
        test=False,
        precision=32,
        batch_size=32,
        randomize=False,
        path_results="./logs",
        create_unique_path=True,
        uncase=False,
    )
    params.update(params_user)
    params["name_project"] = f"{name_task}{'_test' if params['test'] else ''}"
    params["path_results"] = os.path.join(params["path_results"], params["name_project"])
    if params["create_unique_path"]:
        params["path_results"] = get_unique_results_path(params["path_results"])
    if hvd.rank() == 0:
        (Path(params["path_results"]) / "wandb").mkdir(parents=True, exist_ok=True)
    # Convert to "FP16" to (int) 16
    if isinstance(params["precision"], str):
        params["precision"] = int(params["precision"].lower().replace("fp", ""))

    return params
