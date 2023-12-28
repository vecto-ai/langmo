# import logging
# import os
# import os.path
from pathlib import Path

from protonn.utils import load_json

from .glue import GLUEConfig


def is_yaml_config(path):
    return path.is_file() and path.suffix in {".yaml", ".yml"}


# def is_resume_run(path):
#     _logger = logging.getLogger(__name__)
#     _logger.info("resuming the experiment")
#     path = Path(path) / "metadata.json"
#     return path.is_file() and path.suffix == ".json"


def load_resume_run_params(path):
    path = Path(path)
    params = load_json(path / "metadata.json")
    paths = dict(
        metadata=str(path),
        checkpoint=str(path / "PL_model.ckpt"),
        hf=str(path / "hf"),
    )
    params["resume"] = paths
    return params
