from datetime import datetime

import xarray as xr

from src.utils import get_rollout
from src.utils import get_gt_rollout

from src.utils import get_slices
import numpy as np

def get_target_slices(
    guidance_reference,
    rollout_id,
    N,
    M,
    timestamp,
    partition,
    var,
    level,
    delta_trajectory,
    m,
):
    reference_rollout = get_reference_rollouts(guidance_reference, rollout_id, N, M, timestamp)
    # reference_rollout = reference_rollout.sel(m=m)

    target = reference_rollout.copy(deep=True)
    
    delta = np.array(delta_trajectory).reshape((-1, 1, 1))

    if partition == "level":
        target[var].loc[{"level": level}] = (
            target[var].sel(level=level)
            + delta * abs(target[var].sel(level=level))
        )
    else:
        target[var] = target[var] + delta * abs(target[var])

    slices = get_slices(target, partition, var, level)
    return slices


def get_reference_rollouts(
    guidance_reference: str,
    rollout_id: str,
    N: int | None = None,
    M: int | None = None, 
    timestamp: datetime | None = None
):
    match guidance_reference:
        case "GT":
            reference_rollout = get_gt_rollout(N+1, timestamp)
            # mock dimension for arrays to match shape on ui
            reference_rollout = reference_rollout.expand_dims(member=[m for m in range(M)])

        case "UNG":
            reference_rollout = get_rollout("ung", rollout_id)

        case _:
            raise ValueError(f"Invalid reference {guidance_reference}")

    return reference_rollout