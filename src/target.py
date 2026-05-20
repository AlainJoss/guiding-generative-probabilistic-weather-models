import xarray as xr

from src.utils.read_write import (
    get_rollout_files,
)
from src.dimensions import LEVELS_DICT, VARIABLES_DICT


def get_target_trajectory(
    partition,
    var_idx,
    level_idx,
    delta_trajectory,
    reference_trajectory,
):
    target = reference_trajectory.copy(deep=True)

    var_name = VARIABLES_DICT[partition][var_idx]
    target[var_name].load()
    arr = target[var_name].values

    if partition == "surface":
        for n, d in enumerate(delta_trajectory):
            arr[n] += float(d) * abs(arr[n])
    else:
        level_val = LEVELS_DICT["level"][level_idx]
        li = list(target.level.values).index(level_val)

        for n, d in enumerate(delta_trajectory):
            arr[n, li] += float(d) * abs(arr[n, li])

    return target


def get_reference_trajectory(
    guidance_reference: str,
    rollout_id: str,
    m: int | None = None,
):
    match guidance_reference:
        case "ground_truth":
            rollout, _ = get_rollout_files("ground_truth", rollout_id)
            reference_trajectory = rollout

        case "unguided_members":
            rollout, _ = get_rollout_files("unguided", rollout_id)
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

