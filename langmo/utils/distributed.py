
from typing import Optional

import torch
import torch.distributed as dist

def allreduce(tensor: torch.Tensor, op: Optional[int] = None) -> torch.Tensor:
    if op is None:
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    else:
        dist.all_reduce(tensor, op=op)
    return tensor


def aggregate_batch_stats(batch_stats, key):
    if key in batch_stats[0]:
        value = torch.stack([x[key] for x in batch_stats]).sum()
    else:
        value = torch.tensor(0)
    # print("reducing", key, value)
    if torch.cuda.is_available():
        value = value.cuda()
    value = allreduce(value, op=dist.ReduceOp.SUM)
    return value.item()
