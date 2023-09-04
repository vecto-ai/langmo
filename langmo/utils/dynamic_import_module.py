import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import Optional


def set_sys_path(working_directory: Optional[str]):
    """
    Sets working directory in sys.path, if `working_directory` is None then it sets it to the current working directory
    """
    if working_directory is None:
        working_directory = os.getcwd()

    abs_path = os.path.abspath(working_directory)
    if abs_path not in sys.path:
        sys.path.insert(1, abs_path)


def load_module(working_directory: Optional[str], module: str):
    """
    Loads an entire module
    :param working_directory: Root directory of the module, e.g, ./
    :param module: Module path, e.g, "example.module.path"
    """
    set_sys_path(working_directory)
    _lib = __import__(module)
    return _lib


def load_class(working_directory: Optional[str], module: str, class_name: str):
    """
    Loads a class from a module
    :param working_directory: Root directory of the module, e.g, ./
    :param module: Module path, e.g, "example.module.path"
    :param class_name: Name of the class to be imported, e.g., ExampleClass
    """
    set_sys_path(working_directory)
    _lib = __import__(module, fromlist=[class_name])
    return getattr(_lib, class_name)
