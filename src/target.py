from datetime import datetime

import xarray as xr

from src.utils.read_write import get_rollout_files
from src.utils.dataset_utils import get_gt_rollout
from src.dimensions import LEVELS_DICT, VARIABLES_DICT


def get_target_rollout(
    partition,
    var_idx,
    level_idx,
    delta_trajectory,
    reference_rollout,
):
    target = reference_rollout.copy(deep=True)

    var_name = VARIABLES_DICT[partition][var_idx]
    da = target[var_name].load()

    delta = xr.DataArray(
        [float(d) for d in delta_trajectory],
        dims=["time"],
        coords={"time": da.time},
    )

    selector = {}
    if partition == "level":
        selector["level"] = LEVELS_DICT["level"][level_idx]

    current = da.sel(selector)
    da.loc[selector] = current + delta * abs(current)

    return target

# TODO: merge with other func
def get_reference_rollouts(
    guidance_reference: str,
    rollout_id: str,
    N: int | None = None,
    M: int | None = None, 
    timestamp: datetime | None = None
):
    match guidance_reference:
        case "ground_truth":
            reference_rollout = get_gt_rollout(N+1, timestamp)
            # mock dimension for arrays to match shape on ui
            reference_rollout = reference_rollout.expand_dims(member=[m for m in range(M)])

        case "unguided_members":
            reference_rollout, _ = get_rollout_files("unguided_rollout", rollout_id)

        case _:
            raise ValueError(f"Invalid reference {guidance_reference}")

    return reference_rollout


def get_reference_rollout(
    guidance_reference: str,
    rollout_id: str,
    m: int | None = None,
    N: int | None = None,
    timestamp: datetime | None = None
):
    match guidance_reference:
        case "ground_truth":
            reference_rollout = get_gt_rollout(N+1, timestamp)

        case "unguided_members":
            rollout, _ = get_rollout_files("unguided_rollout", rollout_id)
            reference_rollout = rollout.sel(member=m)

        # TODO: need to move this to get trajectory or move everything together, 
        #       --> because I need to know the mask for this
        # case "lower_boundary":
        #     rollout = get_rollout_xr(rollout_id, "unguided")
        #     reference_trajectory = rollout.min(dim="member")

        # case "upper_boundary":
        #     rollout = get_rollout_xr(rollout_id, "unguided")
        #     reference_trajectory = rollout.max(dim="member")

        case "sampled_trajectory":
            reference_rollout = None

        case _:
            raise ValueError(f"Invalid reference {guidance_reference}")

    return reference_rollout

