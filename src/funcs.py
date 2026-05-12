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


def get_inverse_guidance(guidance: float, mask_avg: float, eps: float = 1e-12):
    denom = np.abs(mask_avg)
    if denom < eps:
        raise ValueError(
            "Cannot invert guidance when mask_avg is zero or very close to zero."
        )

    return (guidance - mask_avg) / denom


def get_inverse_guidance_trajectory(
    planned_guidance: list[float],
    mean_rollout: list[float],
    eps: float = 1e-12,
):
    return [
        get_inverse_guidance(planned_guidance[idx], mean_rollout[idx], eps=eps)
        for idx, _ in enumerate(mean_rollout)
    ]



def N_schedule(
    N: int,
    flatness: float,
    peak: float,
    alpha: float = 0.0,
    peak_at_n: int | None = None,
) -> list[torch.Tensor]:
    if N < 1:
        raise ValueError("N must be >= 1")

    if flatness <= 0:
        raise ValueError("flatness must be > 0")

    if peak_at_n is None:
        peak_at_n = N // 2

    if not (1 <= peak_at_n <= N):
        raise ValueError(
            f"For N >= 1, peak_at_n must be in [1, {N}], got {peak_at_n}"
        )

    values = []

    for n in range(N + 1):
        if n == 0:
            value = 0.0

        elif n <= peak_at_n:
            # Smooth rise from 0 at index 0 to 1 at index peak_at_n
            x = n / peak_at_n
            bump = 0.5 * (1.0 - math.cos(math.pi * x))
            value = (alpha + peak) * (bump ** flatness)

        else:
            # Smooth fall from 1 at index peak_at_n to 0 at index N
            # Only reached when peak_at_n < N
            x = (n - peak_at_n) / (N - peak_at_n)
            bump = 0.5 * (1.0 + math.cos(math.pi * x))
            value = (alpha + peak) * (bump ** flatness)

        values.append(torch.tensor(value, dtype=torch.float32))

    return values


def T_schedule(alpha: float, w: float): 
    T=25
    return [torch.tensor( w * (math.sin(math.pi * t / (T - 1)) ** alpha), dtype=torch.float32, ) for t in range(T)] 


def compute_mean_rollout(rollout_trajectory: dict[str, list]) -> dict[str, float]:
    mean_trajectory = []

    for values in rollout_trajectory:
        mean_trajectory.append(sum(values) / len(values))

    return mean_trajectory