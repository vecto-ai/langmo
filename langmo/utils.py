import os
import sys
import yaml
import platform
from protonn.utils import get_time_str


def get_unique_results_path(base):
    hostname = platform.node()
    r = os.path.join(base, f"{get_time_str()}_{hostname}")
    return r


def load_config():
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        exit(-1)
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    return params
