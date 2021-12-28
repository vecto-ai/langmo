"""Logging helper for langmo

"""
import logging

# import os

__all__ = ('set_root_logger')
# LOGGER = logging.getLogger()
# TODO: add rank to log
# FORMAT = '%(asctime)s : %(rank)s : %(levelname)s : %(processName)s : %(name)s : %(funcName)s : %(message)s'
FORMAT = '%(asctime)s : %(levelname)s : %(processName)s : %(name)s : %(funcName)s : %(message)s'

# logging.basicConfig(format="%(custom_attribute)s - %(message)s")

# old_factory = logging.getLogRecordFactory()


# def record_factory(*args, **kwargs):
#     record = old_factory(*args, **kwargs)
#     record.custom_attribute = "my-attr"
#     return record

# TODO: automatically exrtact name of calling frame


# class MyLogger:
#     def __init__(self, name, rank):
#         self.logger = logging.getLogger(name)
#         self.rank = rank

#     def info(self, message, **kwargs):
#         self.logger.info(message, extra={"rank", self.rank}, **kwargs)


def set_root_logger(config=None, rank=-1):
    """sets up logging format according to config"""
    # if ("debug_level" not in config or config["debug_level"] == 0):
    #     debug_level = logging.WARNING
    # elif config["debug_level"] == 1:
    #     debug_level = logging.INFO
    # elif config["debug_level"] == 2 or config["debug_level"] == 3:
    #     debug_level = logging.DEBUG
    log_level = logging.INFO

    logging.basicConfig(level=log_level,
                        format=FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # dir_log = config["results_full_path"]
    # os.makedirs(dir_log, exist_ok=True)
    # file_log = os.path.join(dir_log, "beholder.log")
    # file_handler = logging.FileHandler(file_log)
    # formatter = logging.Formatter(FORMAT)
    # file_handler.setFormatter(formatter)
    # LOGGER.addHandler(file_handler)
