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


def xr_rollout_slice_to_tdict(xr_slice: xr.Dataset) -> TensorDict:
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

from geoarches.dataloaders.era5 import convert_to_xarray


def sampling_trace_to_xarray(trace: list[TensorDict], m, n):
    T = len(trace)
    dummy_ts = torch.tensor(0)
    xr_steps = [convert_to_xarray(s.unsqueeze(0), dummy_ts) for s in trace]
    xr_v = xr.concat(xr_steps, dim="time")
    xr_v = xr_v.rename({"time": "diffusion_step"}).assign_coords(
        diffusion_step=list(range(T))
    )
    xr_v = xr_v.expand_dims(member=[m], n=[n])
    return xr_v


def rollout_to_xarray(
    sample_multistep,
    start_timestamp,
    member,
) -> xr.Dataset:
    xr_rollout = []
    for i, sample in enumerate(sample_multistep):
        xr_sample = convert_to_xarray(sample.unsqueeze(0), start_timestamp+i*24*3600)
        xr_rollout.append(xr_sample)

    xr_rollout = xr.concat(xr_rollout, dim="time")

    xr_rollout = xr_rollout.expand_dims(member=[member])
    return xr_rollout


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