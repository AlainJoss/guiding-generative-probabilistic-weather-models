from datetime import datetime, timezone

import xarray as xr
import torch

from tensordict.tensordict import TensorDict

from src.utils.setup import get_device
from src.dimensions import VARIABLES_DICT, LEVELS_DICT, PARTITIONS
from src.constants import DATETIME_STR_FORMAT


# TODO: bs
def xr_slice_to_torch(xr_ds, var, timestamp, level=None):
    da = xr_ds[var].sel(time=timestamp)

    if "level" in da.dims and level is not None:
        da = da.sel(level=int(level))

    return torch.tensor(da.values)


def get_var_idx(partition: str, var: str) -> int:
    match partition:
        case "surface":
            variables = VARIABLES_DICT["surface"]
        case "level":
            variables = VARIABLES_DICT["level"]
        case _:
            raise ValueError(
                f"Unknown partition '{partition}'. "
                f"Expected one of {PARTITIONS}."
            )
    return variables.index(var)


def get_level_idx(partition: str, level: int) -> int:
    match partition:
        case "surface":
            levels = LEVELS_DICT["surface"]
        case "level":
            levels = LEVELS_DICT["level"]
        case _:
            raise ValueError(
                f"Unknown partition '{partition}'. "
                f"Expected one of {PARTITIONS}."
            )
    return levels.index(level)

# TODO: bs
def xr_rollout_slice_to_tdict(xr_slice: xr.Dataset) -> TensorDict:
    """
    Convert a single-time xr slice from a saved rollout netcdf into a
    TensorDict matching the layout of x_cond["state"]. Saved rollouts are
    already in model orientation, so skip the Europe-roll / lat-flip that
    Era5Forecast.convert_to_tensordict would apply.
    """
    from rollout_config import VARIABLES_DICT

    xr_slice = xr_slice.transpose(..., "level", "latitude", "longitude")
    arrays = {
        key: xr_slice[list(vars_)].to_array().to_numpy()
        for key, vars_ in VARIABLES_DICT.items()
    }
    tdict = TensorDict(
        {k: torch.from_numpy(a).float() for k, a in arrays.items()}
    )
    tdict["surface"] = tdict["surface"].unsqueeze(-3)
    return tdict


def batchify_and_move(sample, device):
    return {
        k: v[None].to(device) if hasattr(v, "to") else v
        for k, v in sample.items()
    }

# TODO: clean this complete bs
def rollout_to_xarray(
    ds,
    sample_multistep,
    init_timestamp,
    member,
    lead_time_hours=24,
    include_init=False,
):
    sample_multistep = ds.denormalize(sample_multistep).detach().cpu()

    init_seconds = int(init_timestamp.detach().cpu().flatten()[0].item())
    step_iterations = sample_multistep.shape[1]

    xr_steps = []

    for i in range(step_iterations):
        offset = i if include_init else i + 1
        valid_seconds = init_seconds + lead_time_hours * 3600 * offset

        xr_step = ds.convert_to_xarray(
            sample_multistep[:, i],
            timestamp=torch.tensor([valid_seconds]),
        )

        xr_steps.append(xr_step)

    xr_rollout = xr.concat(xr_steps, dim="time")

    if member == -1:
        return xr_rollout

    return xr_rollout.expand_dims(member=[member])


def xr_to_torch(slice_: xr.DataArray):
    return torch.tensor(slice_.to_numpy())


def list_tensors_to_floats(list_):
    return [tensor.item() for tensor in list_]


def list_floats_to_tensors(list_):
    device = get_device()
    return [torch.tensor(float_).to(device) for float_ in list_]


def tensor_timestamp_to_string(
    timestamp: torch.Tensor,
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    ts = timestamp.item()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)


def datetime_to_string(timestamp: datetime):
    return datetime.strftime(timestamp, DATETIME_STR_FORMAT)


def string_to_datetime(timestamp: str):
    return datetime.strptime(timestamp, DATETIME_STR_FORMAT)