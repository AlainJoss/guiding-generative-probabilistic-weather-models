import xarray as xr

from src.utils.read_write import (
    get_rollout_xr,
)
from src.dimensions import LEVELS_DICT, VARIABLES_DICT

def get_target_trajectory(
    config,
    reference_trajectory: xr.Dataset,
):
    target = reference_trajectory.copy(deep=True)
    var_name = VARIABLES_DICT[config.partition][config.var_idx]

    for n, delta_n in enumerate(config.delta_trajectory):
        if config.partition == "surface":
            target[var_name].isel(time=n).values[:] *= 1.0 + delta_n
        else:
            level_val = LEVELS_DICT["level"][config.level_idx]
            target[var_name].sel(level=level_val).isel(time=n).values[:] *= 1.0 + delta_n

    return target


def get_reference_trajectory(
    config,
    m: int | None = None,
):
    match config.reference:
        case "ground_truth":
            rollout = get_rollout_xr(config.rollout_id, "ground_truth")
            reference_trajectory = rollout

        case "unguided_m":
            rollout = get_rollout_xr(config.rollout_id, "unguided")
            reference_trajectory = rollout.sel(member=m)

        case "lower_boundary":
            rollout = get_rollout_xr(config.rollout_id, "unguided")
            reference_trajectory = rollout.min(dim="member")

        case "upper_boundary":
            rollout = get_rollout_xr(config.rollout_id, "unguided")
            reference_trajectory = rollout.max(dim="member")

        case "sampled_trajectory":
            reference_trajectory = None

        case _:
            raise ValueError(f"Invalid reference {config.reference}")

    return reference_trajectory

