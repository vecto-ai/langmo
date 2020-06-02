import yaml


def get_config(path_config="config.yaml"):
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    return params
