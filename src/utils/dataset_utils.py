import xarray as xr
import numpy as np

from tensordict.tensordict import TensorDict

from src.paths import ERA5


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


def get_xr_ds():
    timesteps = ["00", "06", "12", "18"]
    datasets = [
        xr.open_dataset(f"{ERA5}/{ts}h.nc", engine="netcdf4")
        for ts in timesteps
    ]
    return xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("time")


def get_x_cond_from_ts(ds, timestamp):
    """
    TODO: clean up this bs
    """
    target = np.datetime64(timestamp, "ns")

    offset = int(ds.load_prev) * int(ds.lead_time_hours) // int(ds.timedelta)

    ds_timestamps = [
        np.datetime64(ts[2], "ns")
        for ts in ds.timestamps
    ]

    if target not in ds_timestamps:
        raise ValueError(
            f"{target} not found in dataset timestamps. "
            f"First: {ds_timestamps[0]}, last: {ds_timestamps[-1]}"
        )

    raw_idx = ds_timestamps.index(target)
    dataset_idx = raw_idx - offset

    if dataset_idx < 0:
        raise ValueError(
            f"{target} exists at raw_idx={raw_idx}, "
            f"but ds[{dataset_idx}] would be needed because __getitem__ applies offset={offset}. "
            f"Pick a later timestamp."
        )

    x_cond = ds[dataset_idx]

    return x_cond