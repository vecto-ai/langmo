import os
import platform
from protonn.utils import get_time_str


def get_unique_results_path(base):
    hostname = platform.node()
    r = os.path.join(base, f"{get_time_str()}_{hostname}")
    return r
