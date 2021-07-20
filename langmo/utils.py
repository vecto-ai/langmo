import os
import platform
import sys
from pathlib import Path

import horovod.torch as hvd
import yaml
from protonn.utils import get_time_str, load_json


def get_unique_results_path(base, model_name, timestamp):
    hostname = platform.node().split(".")[0]
    short_name = model_name.split("/")[-1]
    new_path = os.path.join(base, f"{timestamp}_{short_name}_{hostname}")
    # TODO: make this trully unique
    return new_path


def parse_float(dic, key):
    if key in dic:
        if isinstance(dic[key], str):
            dic[key] = float(dic[key])


def load_yaml_config(path_config):
    with open(path_config, "r") as cfg:
        params_user = yaml.load(cfg, Loader=yaml.SafeLoader)
    parse_float(params_user, "initial_lr")
    parse_float(params_user, "max_lr")
    parse_float(params_user, "eps")
    parse_float(params_user, "beta1")
    parse_float(params_user, "beta2")
    return params_user


def apply_defaults_to_params(params_user):
    params = dict(
        test=False,
        use_gpu=True,
        precision=32,
        batch_size=32,
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
        siamese=False,
        shuffle=False,
        gradient_clip_val=0.0,
    )
    params.update(params_user)
    return params


def is_yaml_config(path):
    return len(sys.argv) == 2 and path.is_file() and path.suffix in {".yaml", ".yml"}


def load_yaml_config_with_defaults(path, name_task):
    params_user = load_yaml_config(path)
    params = apply_defaults_to_params(params_user)
    name_project = name_task
    # if "suffix" in params:
    #     name_project += f"_{params['suffix']}"
    if params['test']:
        name_project += "_test"
    params["name_project"] = name_project
    params["path_results"] = os.path.join(params["path_results"], name_project)
    params["timestamp"] = get_time_str()
    if params["create_unique_path"]:
        params["path_results"] = get_unique_results_path(
            params["path_results"],
            params["model_name"],
            params["timestamp"]
        )
    if hvd.rank() == 0:
        (Path(params["path_results"]) / "wandb").mkdir(parents=True, exist_ok=True)
    # Convert to "FP16" to (int) 16
    if isinstance(params["precision"], str):
        params["precision"] = int(params["precision"].lower().replace("fp", ""))
    # TODO: we put it here for now for simplicitly
    # this needs to be revisited when we do model parallel
    # TODO: also we whould think what we do when we resume with different number of workers
    params["cnt_workers"] = hvd.size()
    params["batch_size_effective"] = params["batch_size"] * params["cnt_workers"]
    return params


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


def load_config(name_task):
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        print("or")
        print("run main.py logs/path/to/snapshot/epoc10_step42000")
        exit(-1)

    path = Path(sys.argv[1])

    if is_yaml_config(path):
        return load_yaml_config_with_defaults(path, name_task)

    if is_resume_run(path):
        return load_resume_run_params(path)
