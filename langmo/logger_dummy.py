from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import torch
from pytorch_lightning.loggers.base import LightningLoggerBase


# TODO: consider making it actually doing something useful
class DummyLogger(LightningLoggerBase):
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
