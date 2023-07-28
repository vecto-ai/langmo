from typing import Dict, Optional, Union

import torch
from lightning.pytorch.loggers.logger import Logger


# TODO: consider making it actually doing something useful
class DummyLogger(Logger):
    def log_hyperparams(self, * args, **kwargs):
        pass

    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        pass

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "0"

    @property
    def experiment(self):
        return None
