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
) -> list[torch.Tensor]:
    if N < 1:
        raise ValueError("N must be >= 1")

    zero = torch.tensor(0.0, dtype=torch.float32)

    if N == 1:
        return [
            zero,
            torch.tensor(alpha + peak, dtype=torch.float32),
        ]

    values = [
        torch.tensor(
            alpha + peak * (math.sin(math.pi * n / N) ** flatness),
            dtype=torch.float32,
        )
        for n in range(1, N)
    ]

    return [zero] + values + [zero]

def T_schedule(alpha: float, w: float): 
    T=25
    return [torch.tensor( w * (math.sin(math.pi * t / (T - 1)) ** alpha), dtype=torch.float32, ) for t in range(T)] 

def compute_mean_rollout(rollout_trajectory: dict[str, list]) -> dict[str, float]:
    mean_trajectory = []

    for values in rollout_trajectory:
        mean_trajectory.append(sum(values) / len(values))

    return mean_trajectory


# 
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Print lambda_ schedule from T_schedule(alpha, w)."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--w", type=float, required=True)
    parser.add_argument(
        "--format",
        choices=["json", "python", "plain"],
        default="json",
    )

    args = parser.parse_args()

    lambda_ = T_schedule(args.alpha, args.w)
    lambda_ = [float(x.item()) for x in lambda_]

    if args.format == "json":
        print(json.dumps(lambda_))
    elif args.format == "python":
        print(lambda_)
    else:
        print(" ".join(str(x) for x in lambda_))


if __name__ == "__main__":
    main()