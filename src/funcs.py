import math
import json
import hashlib

import torch
import numpy as np


def make_hash(params):
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# def avg_over_mask(slice_, mask):
#     avg = torch.sum(mask * slice_) / torch.sum(mask)
#     return avg

# def avg_xr_over_mask(xr_ds, var, timestamp, mask, level=None, member=None):
#     da = xr_ds[var].sel(time=timestamp)

#     if member is not None and "member" in da.dims:
#         da = da.sel(member=member)

#     if "level" in da.dims and level is not None:
#         da = da.sel(level=int(level))

#     x = torch.tensor(da.values, dtype=mask.dtype)
#     return avg_over_mask(x, mask)


# def get_guidance_trajectory(y: list[float], mean_rollout: list[float]):
#     def get_guidance(y_n: float, mask_avg: float):
#         return mask_avg + y_n * np.abs(mask_avg)
    
#     return [get_guidance(y[idx], mean_rollout[idx])
#         for idx, _ in enumerate(mean_rollout)
#     ]


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


# def compute_mean_rollout(rollout_trajectory: dict[str, list]) -> dict[str, float]:
#     mean_trajectory = []

#     for values in rollout_trajectory:
#         mean_trajectory.append(sum(values) / len(values))

#     return mean_trajectory


# def safe_abs_limits(arrays):
#     vmin = min(float(np.nanmin(np.asarray(arr))) for arr in arrays)
#     vmax = max(float(np.nanmax(np.asarray(arr))) for arr in arrays)

#     if vmax <= vmin:
#         vmax = vmin + 1e-9

#     center = 0.5 * (vmin + vmax)
#     center = min(max(center, vmin + 1e-9), vmax - 1e-9)

#     return vmin, vmax, center


# def safe_diff_absmax(arrays):
#     absmax = max(
#         float(np.nanmax(np.abs(np.asarray(arr))))
#         for arr in arrays
#     )

#     if absmax <= 0:
#         absmax = 1e-8

#     return absmax