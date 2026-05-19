import xarray as xr

from src.utils.read_write import (
    get_rollout_xr,
)
from src.dimensions import LEVELS_DICT, VARIABLES_DICT

def get_target_trajectory(
    partition,
    var_idx,
    level_idx,
    delta_trajectory,
    reference_trajectory: xr.Dataset,
):  
    # TODO: do I really need this?
    target = reference_trajectory.copy()
    var_name = VARIABLES_DICT[partition][var_idx]

    for n, delta_n in enumerate(delta_trajectory):
        if partition == "surface":
            target[var_name].isel(time=n).values[:] *= 1.0 + delta_n
        else:
            level_val = LEVELS_DICT["level"][level_idx]
            target[var_name].sel(level=level_val).isel(time=n).values[:] *= 1.0 + delta_n

    return target


def get_reference_trajectory(
    guidance_reference: str,
    rollout_id: str,
    m: int | None = None,
):
    match guidance_reference:
        case "ground_truth":
            rollout = get_rollout_xr(rollout_id, "ground_truth")
            reference_trajectory = rollout

        case "unguided_members":
            rollout = get_rollout_xr(rollout_id, "unguided")
            reference_trajectory = rollout.sel(member=m)

        # TODO: need to move this to get trajectory or move everything together, 
        #       --> because I need to know the mask for this
        # case "lower_boundary":
        #     rollout = get_rollout_xr(rollout_id, "unguided")
        #     reference_trajectory = rollout.min(dim="member")

        # case "upper_boundary":
        #     rollout = get_rollout_xr(rollout_id, "unguided")
        #     reference_trajectory = rollout.max(dim="member")

        case "sampled_trajectory":
            reference_trajectory = None

        case _:
            raise ValueError(f"Invalid reference {guidance_reference}")

    return reference_trajectory

