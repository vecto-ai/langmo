import logging
import os
from textwrap import dedent
from typing import Dict, List, Optional

from lightning import Callback

from .dynamic_import_module import load_class


def log_callback(name: str, params: Dict[str, str], init_params: Dict[str, str]):
    logger = logging.getLogger(__name__)
    logger.info(
        dedent(
            f"""
        Attached Callback | {name}: {{
            \tworking_directory: {params["working_directory"] if params["working_directory"] is not None else os.getcwd()},
            \tmodule: {params["module"]},
            \tclass_name: {params["class_name"]},
            \tparams: {init_params}
        }}
    """
        )
    )


def init_callbacks(callback_configs: Dict) -> List[Callback]:
    """
    Loads and Initializes callbacks from config
    :param callback_configs: A dictionary containing all of the required callbacks, i.e.: config["callbacks"]
    """
    # if the list of callbacks is empty and no fallback was provided then return an empty list
    if callback_configs is None:
        return []

    # list of callbacks
    callbacks = []
    for name, module_params in callback_configs.items():
        # set of parameters required to init the class
        init_params = {}
        if "params" in module_params:
            init_params = module_params.pop("params")
        # if "working_directory" is not set then set it to None which will default to the current working directory
        if "working_directory" not in module_params:
            module_params["working_directory"] = None
        # load and init the class
        klass = load_class(**module_params)(**init_params)
        callbacks.append(klass)
        log_callback(name, module_params, init_params)
    return callbacks
