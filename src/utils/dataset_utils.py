from pathlib import Path
from datetime import datetime, timezone, timedelta

import xarray as xr
import numpy as np

from tensordict.tensordict import TensorDict

from src.paths import ERA5
from src.utils.converters import tensor_timestamp_to_string


def get_x_cond(ds, timestamp):
    timestamp = np.datetime64(timestamp, "ns")

    ds_timestamps = [
        np.datetime64(ts[2], "ns")
        for ts in ds.timestamps
    ]

    idx = ds_timestamps.index(timestamp) - 4  # everything is shifted because we have a "prev_state"
    x_cond = ds[idx]

    ts = tensor_timestamp_to_string(x_cond["timestamp"])
    print(f"get x_cond with timestamp {ts}")

    return x_cond


def get_xr_slice(state, partition, level, var, timestamp):
    if partition == "surface":
        return state[var].sel(time=timestamp, method='nearest')
    else: 
        return state[var].sel(time=timestamp, level=level, method='nearest')

# TODO: check this bs
def get_td_slice(
    state: TensorDict,
    partition: str,
    var_idx: int,
    level_idx: int | None = None,
):
    if partition == "surface":
        # expected shape maybe [B, V, 1, H, W] or [V, 1, H, W]
        return state["surface"][..., var_idx, 0, :, :]

    return state["level"][..., var_idx, level_idx, :, :]


def get_timestamps(ds: xr.Dataset):
    timestamps = []
    for i in range(len(ds.time)):
        ns_ts = ds.time[i].item()
        dt_utc = datetime.fromtimestamp(ns_ts / 10**9, tz=timezone.utc)
        dt_display = dt_utc.replace(tzinfo=None)
        dt = str(dt_display)
        timestamps.append(dt)

    return timestamps


def get_N_timestamps(timestamp: str, N: int) -> list[datetime]:
    return [timestamp + timedelta(days=n) for n in range(N)]


def get_N_states(ds: xr.Dataset, N: int, timestamp: datetime):
    assert timestamp.tzinfo is None 
    # alternatively
    # timestamp = timestamp.replace(tzinfo=None)
    timestamps = get_N_timestamps(timestamp, N)
    print(len(timestamps))
    return ds.sel(time=timestamps)


def get_slices(states: xr.Dataset, partition: str, var: str, level: str):
    slices = states[var]
    if partition == "level":
        slices = slices.sel(level=level)
    slices = slices.to_numpy()
    return slices


def get_N_slices(ds: xr.Dataset, N: int, timestamp: datetime, partition: str, var: str, level: str):
    N_states = get_N_states(ds, N, timestamp)
    N_slices = get_slices(N_states, partition, var, level)
    return N_slices