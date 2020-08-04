"""Logging helper for beholder

"""
import logging
import os


__all__ = ('set_root_logger')
LOGGER = logging.getLogger()
FORMAT = '%(asctime)s : %(levelname)s : %(processName)s : %(name)s :\
          %(funcName)s : %(message)s'


def set_root_logger(config):
    """sets up logging format according to config"""
    if ("debug_level" not in config or config["debug_level"] == 0):
        debug_level = logging.WARNING
    elif config["debug_level"] == 1:
        debug_level = logging.INFO
    elif config["debug_level"] == 2 or config["debug_level"] == 3:
        debug_level = logging.DEBUG

    logging.basicConfig(level=debug_level,
                        format=FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S')
    dir_log = config["results_full_path"]
    os.makedirs(dir_log, exist_ok=True)
    file_log = os.path.join(dir_log, "beholder.log")
    file_handler = logging.FileHandler(file_log)
    formatter = logging.Formatter(FORMAT)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
