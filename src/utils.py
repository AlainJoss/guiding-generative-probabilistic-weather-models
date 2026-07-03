##########################
# setup utils
##########################

import json
import functools
from typing import Any
from pathlib import Path 
from datetime import datetime, timezone, timedelta

import xarray as xr
import torch
import numpy as np
import dask.array as da

from src.paths import ERA5, MODELSTORE, ROLLOUTS, GT
from src.dimensions import VARIABLES_DICT, LEVELS_DICT, PARTITIONS, SPATIAL_COORDS
from src.rollout_config import RolloutConfig

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

from tensordict.tensordict import TensorDict


def get_now_timestamp():
    date, time = str(datetime.now().replace(microsecond=0)).split(" ")
    return date + "_" + time


def ensure_rollout_dir(id_: str = None) -> Path:
    rollout_dir = Path(ROLLOUTS, f"{id_}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    return rollout_dir


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"running on device: {device}")
    return device


##########################
# read write utils
##########################

##### data and model #####

# @functools.lru_cache(maxsize=None)
def get_xr_dataset(year: int):
    if year == 2026:
        path = GT / "arches_era5_26.nc"
        ds = xr.open_dataset(path)
        return ds
    path = GT / "arches_era5.nc"
    ds = xr.open_dataset(path)
    return ds


def get_td_dataset(multistep:int=1, year:int=2020):
    return Era5Forecast(
        path=ERA5,  # default path
        domain="test",  # all files under ERA5; year-slicing happens on the time coord
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=24,
        timedelta_hours=6,
        multistep=multistep
    )


def get_model(device):
    gen_model, _ = load_module(  # _ := gen_config
        MODELSTORE / "archesweathergen",
        module_target="geoarches.lightning_modules.guided_diffusion.GuidedFlow",
    )
    gen_model = gen_model.to(device)
    gen_model.move_objects_to_device()
    return gen_model


##### data files #####

def get_rollout_dir(rollout_id:str):
    return ROLLOUTS / rollout_id


def get_dict_from_json(path: Path):
    with open(path, "r") as f:
        dict_ = json.load(f)
    return dict_


def get_sweep_dict(rollout_id):
    path = get_rollout_dir(rollout_id) / "sweep_params.json"
    return get_dict_from_json(path)


def append_w_star(rollout_dir, sweep_params: dict, m: int, n: int, w_star: float):
    """Persist an optimized guidance strength (FGW/FGWNOLR) for one (sweep point, m, n).

    Stored in <rollout_dir>/w_star.json as a list of records; an existing record for
    the same (sweep, m, n) is replaced (resume-safe). sweep values are the ACTUAL job
    params (GUIDANCE_DELTA as the vector)."""
    path = Path(rollout_dir) / "w_star.json"
    records = json.loads(path.read_text()) if path.exists() else []
    sweep = dict(sweep_params or {})
    records = [
        r for r in records
        if not (r["m"] == m and r["n"] == n and r["sweep"] == sweep)
    ]
    records.append({"sweep": sweep, "m": m, "n": n, "w_star": float(w_star)})
    path.write_text(json.dumps(records, indent=2))


def get_w_star(rollout_id, sweep_sel: dict | None = None, m: int | None = None, n: int | None = None):
    """w_star records matching a sweep selection (loose float compare; lists compared
    elementwise). sweep_sel uses ACTUAL values (GUIDANCE_DELTA as the vector, not its
    zarr index). Returns [] if no sidecar exists (pre-FGW rollouts)."""
    path = get_rollout_dir(rollout_id) / "w_star.json"
    if not path.exists():
        return []

    def _match(sel_v, rec_v):
        if isinstance(sel_v, (list, tuple)) or isinstance(rec_v, (list, tuple)):
            try:
                a, b = np.asarray(sel_v, float), np.asarray(rec_v, float)
                return a.shape == b.shape and bool(np.allclose(a, b))
            except (TypeError, ValueError):
                return sel_v == rec_v
        if isinstance(sel_v, (int, float)) and isinstance(rec_v, (int, float)):
            return bool(np.isclose(float(sel_v), float(rec_v)))
        return sel_v == rec_v

    out = []
    for r in json.loads(path.read_text()):
        if m is not None and r["m"] != m:
            continue
        if n is not None and r["n"] != n:
            continue
        if sweep_sel and not all(_match(v, r["sweep"].get(k)) for k, v in sweep_sel.items()):
            continue
        out.append(r)
    return out


# @functools.lru_cache(maxsize=None)
def get_rollout(rollout_type:str, rollout_id:str):
    path = get_rollout_dir(rollout_id) / f"{rollout_type}.zarr"
    rollout = xr.open_zarr(path)
    return rollout


def get_config(rollout_id: str) -> RolloutConfig:
    path = get_rollout_dir(rollout_id) / f"config.json"
    config_dict = get_dict_from_json(path)
    return RolloutConfig.from_dict(config_dict)


def get_rollout_ids(rollout_type: str):
    experiments = Path(ROLLOUTS).glob("2026*")

    def has_config(path: Path) -> bool:
        return (path / "config.json").exists()

    def has_file(path: Path, type_: str) -> bool:
        return any(path.glob(f"**/{type_}.zarr"))

    experiments = sorted(
        [
            p.name
            for p in experiments
            if has_config(p) and has_file(p, rollout_type)
        ],
        reverse=True,
    )
    return experiments


##### write files #####

def advance_x_cond(x_cond: dict, x_hat_curr: TensorDict):
    return {
        "prev_state": x_cond["state"],
        "state": x_hat_curr,
        "timestamp": x_cond["timestamp"] + x_cond["lead_time_hours"] * 3600, # converts to seconds the lead_time
        "lead_time_hours": x_cond["lead_time_hours"],
    }

def create_slice_zarr_container(
    m: int,
    n: int,
    t_dim: bool = False,
    T: int = 25,
    sweep_params: dict | None = None,
) -> xr.Dataset:
    ds = create_full_zarr_container(
        M=1,
        N=1,
        t_dim=t_dim,
        T=T,  
        sweep_params={
            k: [v]
            for k, v in (sweep_params or {}).items()
        },
        lazy=False,  # writable numpy so tdict_to_xr can assign into it
    )

    return ds.assign_coords(
        m=[m],
        n=[n],
    )


def _is_scalar_axis(values: list) -> bool:
    """A sweep axis is scalar-labelable iff every candidate is a single non-None
    value (float, str, bool). Non-scalar axes (e.g. delta_trajectory, a per-step
    vector) and None-bearing axes (None serializes to NaN, which never compares
    equal on region auto-detection) are indexed by integer position instead."""
    return all(v is not None and np.isscalar(v) for v in values)


def sweep_coord_label(key: str, value, sweep_params: dict[str, list]):
    """Map a raw sweep value to its zarr coordinate label: the value itself for a
    scalar axis, or its integer index along the sweep for a non-scalar axis. Keeps
    full-container coords and per-slice region writes aligned on the same labels."""
    if _is_scalar_axis(sweep_params[key]):
        return value
    return sweep_params[key].index(value)


def create_full_zarr_container(
    M:int, N:int, t_dim:bool, T:int=25,
    sweep_params:dict[str,list] | None = None,
    lazy:bool=True,
) -> xr.Dataset:
    m_steps = range(M)
    n_steps = range(N)
    t_steps = range(T)
    
    coords = {
        "m": m_steps,
        "n": n_steps,
    }

    surface_dims = ["m", "n"]
    level_dims = ["m", "n"]


    if t_dim:
        coords["t"] = t_steps
        surface_dims.append("t")
        level_dims.append("t")

    coords["level"] = LEVELS_DICT["level"]
    coords["latitude"] = SPATIAL_COORDS["latitude"]
    coords["longitude"] = SPATIAL_COORDS["longitude"]

    # spatial dims last so tdict_to_xr's trailing-axis assignment broadcasts correctly
    surface_dims += ["latitude", "longitude"]
    level_dims += ["level", "latitude", "longitude"]

    for key, values in sweep_params.items():
        if _is_scalar_axis(values):
            coords[key] = values
        else:
            # index an axis by integer position when its values can't be coord
            # labels (delta_trajectory vectors, or None which serializes to NaN);
            # keep the real values in a `<key>_value` coordinate. Coordinates are
            # always written by to_zarr (even compute=False), so they persist; a
            # dask data_var would be left as NaN metadata. (None -> NaN here.)
            coords[key] = list(range(len(values)))
            value_arr = np.asarray(values, dtype=np.float32)
            value_dims = (key,) if value_arr.ndim == 1 else (key, f"{key}_step")
            coords[f"{key}_value"] = (value_dims, value_arr)
        surface_dims.append(key)
        level_dims.append(key)

    data_vars = {}

    # chunk size 1 along m, n and sweep dims so each (m, n, sweep) region write
    # lands on exactly one chunk; full along t / level / spatial dims.
    full_chunk_dims = {"t", "level", "latitude", "longitude"}

    def _chunks_for(dims):
        return tuple(len(coords[d]) if d in full_chunk_dims else 1 for d in dims)

    def _data(dims, shape_):
        # lazy (dask) for the full store so the all-NaN skeleton is never held in RAM;
        # numpy for per-slice containers so tdict_to_xr can write into .values.
        if lazy:
            return da.full(shape_, np.nan, dtype=np.float32, chunks=_chunks_for(dims))
        return np.full(shape_, np.nan, dtype=np.float32)

    for var in VARIABLES_DICT["surface"]:
        shape_ = tuple(len(coords[d]) for d in surface_dims)
        data_vars[var] = (surface_dims, _data(surface_dims, shape_))

    for var in VARIABLES_DICT["level"]:
        shape_ = tuple(len(coords[d]) for d in level_dims)
        data_vars[var] = (level_dims, _data(level_dims, shape_))
    
    container_ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    return container_ds


def tdict_to_xr(container: xr.Dataset, tdict: TensorDict, t_dim: bool = False):
    if not t_dim:
        surface = tdict["surface"].squeeze(2).squeeze(0).detach().cpu().numpy()
        level = tdict["level"].squeeze(0).detach().cpu().numpy()

        for i, var in enumerate(VARIABLES_DICT["surface"]):
            container[var].values[...] = surface[i].reshape(container[var].shape)

        for i, var in enumerate(VARIABLES_DICT["level"]):
            container[var].values[...] = level[i].reshape(container[var].shape)

    else:
        surface = tdict["surface"].squeeze(3).squeeze(1).detach().cpu().numpy()
        level = tdict["level"].squeeze(1).detach().cpu().numpy()

        # surface: (t, 4, lat, lon)
        # level:   (t, 6, level, lat, lon)

        for i, var in enumerate(VARIABLES_DICT["surface"]):
            container[var].values[...] = surface[:, i].reshape(container[var].shape)

        for i, var in enumerate(VARIABLES_DICT["level"]):
            container[var].values[...] = level[:, i].reshape(container[var].shape)

    return container


def append_to_zarr(rollout_dir, name: str, ds_slice: xr.Dataset) -> None:
    ds_slice.to_zarr(
        rollout_dir / f"{name}.zarr",
        mode="r+",
        region="auto",
    )


def dump_json(dict_: dict, path: Path):
    with open(path, "w") as f:
        json.dump(dict_, f, indent=4)


##########################
# converters
##########################


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


def batchify_and_move(sample, device):
    return {
        k: v[None].to(device) if hasattr(v, "to") else v
        for k, v in sample.items()
    }


def tensor_timestamp_to_string(
    timestamp: torch.Tensor,
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    ts = timestamp.item()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)


##########################
# dataset utils
##########################

# TODO: this is not a good pattern
#       I should select lazyly and then compute() (or load() ) such that I don't have to load a 30GB file in memory
#       Need to convert the netcdf to zarr  
def get_gt_rollout(N, timestamp):
    ds = get_xr_dataset(timestamp.year)
    return get_N_states(ds, N, timestamp)


def gt_state_to_tdict(gt_ds: xr.Dataset, time_index: int, device) -> TensorDict:
    """One ground-truth state (at `time_index` of a gt-rollout Dataset) as a denormalized
    model TensorDict {surface:(1,4,1,lat,lon), level:(1,6,nlev,lat,lon)}, matching the
    structure/var-order (VARIABLES_DICT) the guidance loss consumes. Used as the GT data-
    loss reference (GUI_REF=GT). gt is already in physical units, so no normalization."""
    sfc = np.stack([gt_ds[v].isel(time=time_index).values for v in VARIABLES_DICT["surface"]])
    lvl = np.stack([gt_ds[v].isel(time=time_index).values for v in VARIABLES_DICT["level"]])
    sfc_t = torch.as_tensor(sfc, dtype=torch.float32, device=device)[None, :, None]
    lvl_t = torch.as_tensor(lvl, dtype=torch.float32, device=device)[None]
    return TensorDict({"surface": sfc_t, "level": lvl_t}, batch_size=[1], device=device)


def get_x_cond(timestamp: datetime, N):
    ds = get_td_dataset(multistep=N, year=timestamp.year)
    timestamp = np.datetime64(timestamp, "ns")

    ds_timestamps = [
        np.datetime64(ts[2], "ns")
        for ts in ds.timestamps
    ]

    idx = ds_timestamps.index(timestamp) - 4  # everything is shifted because we have a "prev_state"
    x_cond = ds[idx]
    # ts = tensor_timestamp_to_string(x_cond["timestamp"])
    # print(f"get x_cond with timestamp {ts} from timestamp {timestamp}")
    return x_cond


def get_timestamps(ds: xr.Dataset):
    timestamps = []
    for i in range(len(ds.time)):
        ns_ts = ds.time[i].item()
        dt_utc = datetime.fromtimestamp(ns_ts / 10**9, tz=timezone.utc)
        dt_display = dt_utc.replace(tzinfo=None)
        dt = str(dt_display)
        timestamps.append(dt)

    return timestamps


# TODO: should merge these in a lazyable func


def get_N_timestamps(timestamp: str, N: int) -> list[datetime]:
    return [timestamp + timedelta(days=n) for n in range(N)]


def get_N_states(ds: xr.Dataset, N: int, timestamp: datetime):
    # assert timestamp.tzinfo is None 
    # alternatively
    # timestamp = timestamp.replace(tzinfo=None)
    timestamps = get_N_timestamps(timestamp, N)
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