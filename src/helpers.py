
from __future__ import annotations

from typing import Callable, Sequence
import torch


Tensor = torch.Tensor


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """
    Compute the masked mean of x.
    """
    assert mask is not None
    mask = mask.to(dtype=x.dtype, device=x.device)
    denom = mask.sum()  # torch.mean wouldn't do the job
    assert denom >= 1
    return (x * mask).sum() / denom
