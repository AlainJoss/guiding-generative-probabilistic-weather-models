import math

import torch
import numpy as np
import xarray as xr

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply


def avg_over_mask(slice_, mask):
    avg = torch.sum(mask * slice_) / torch.sum(mask)
    return avg

def get_mask_tensordict(example_tdict: TensorDict, partition: str, var_idx: int, level_idx: int, mask_2d: torch.Tensor):
    mask = tensordict_apply(lambda x: torch.zeros_like(x), example_tdict)
    mask[partition][var_idx, level_idx] = mask_2d
    return mask

def get_guidance(y_n: float, mask_avg: float):
    return mask_avg + y_n * np.abs(mask_avg)

def get_guidance_trajectory(y: list[float], mean_rollout: list[float]):
    return [get_guidance(y[idx], mean_rollout[idx])
        for idx, _ in enumerate(mean_rollout)
    ]

def N_schedule(
    N: int,
    flatness: float,
    peak: float,
    alpha: float = 0.0,
) -> list[torch.Tensor]:
    if N < 1:
        raise ValueError("N must be >= 1")

    values = [
        torch.tensor(
            alpha + peak * (math.sin(math.pi * n / (N + 1)) ** flatness),
            dtype=torch.float32,
        )
        for n in range(1, N + 1)
    ]

    return [torch.tensor(0.0, dtype=torch.float32)] + values

def T_schedule(T: int, flatness: float, peak: float): 
    if T == 1: 
        return [torch.tensor(0.0, dtype=torch.float32)] 
    return [
        torch.tensor( peak * (math.sin(math.pi * t / (T - 1)) ** flatness), dtype=torch.float32, ) for t in range(T)
           ] 

def compute_mean_rollout(rollout_trajectory: dict[str, list]) -> dict[str, float]:
    mean_trajectory = []

    for values in rollout_trajectory:
        mean_trajectory.append(sum(values) / len(values))

    return mean_trajectory