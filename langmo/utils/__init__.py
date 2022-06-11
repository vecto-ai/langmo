import os
import platform


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
