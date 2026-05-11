import json
import hashlib
import logging
import sys

from typing import Any
from pathlib import Path 
from datetime import datetime, timezone

import xarray as xr
import torch
import numpy as np

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

from src.constants import GUIDANCE_PARAM_KEYS
from src.paths import ERA5, MODELSTORE, ROLLOUTS, CONFIGS, LOGS


def setup_logging(log_prefix: str = "run") -> None:
    LOGS.mkdir(parents=True, exist_ok=True)

    log_file = LOGS / f"{log_prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, delay=False),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info(f"Logging to: {log_file}")


def make_hash(params):
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def get_x_cond(ds, timestamp):
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

    return x_cond, dataset_idx


def read_nc(rollout_dir: Path, type_: str):
    path = rollout_dir / f"{type_}.nc"
    return xr.open_dataset(path)

def get_experiment_ids(type_: str):
    experiments = Path(ROLLOUTS).glob("2026*")

    def has_config(path: Path) -> bool:
        return (path / "config.json").exists()

    def has_file(path: Path, type_: str) -> bool:
        return any(path.glob(f"**/{type_}.nc"))

    experiments = sorted(
        [
            p.name
            for p in experiments
            if has_config(p) and has_file(p, type_)
        ],
        reverse=True,
    )

    return experiments

def get_rollout_dir(id_: str):
    return ROLLOUTS / id_

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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"running on device: {device}")
    return device

def save_to_json(dict_: dict, rollout_dir: Path, name:str):
    path = rollout_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(dict_, f, indent=2)

def read_json(rollout_dir, name:str):
    path = Path(rollout_dir) / f"{name}.json"
    with open(path, "r") as f:
        dict_ = json.load(f)
    return dict_

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

def get_dataset(multistep:int=1):
    return Era5Forecast(
        path=ERA5,  # default path
        domain="all",  # all files under ERA5; year-slicing happens on the time coord
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
    return gen_model.to(device)

def get_now_timestamp():
    date, time = str(datetime.now().replace(microsecond=0)).split(" ")
    return date + "_" + time

def ensure_new_config_dir_path(sub_dir: str):
    experiment_id = get_now_timestamp()
    config_dir = Path(CONFIGS, sub_dir, f"{experiment_id}")
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def ensure_rollout_dir(experiment_id: str = None) -> Path:
    if experiment_id is None:
        experiment_id = get_now_timestamp()
    rollout_dir = Path(ROLLOUTS, f"{experiment_id}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    return rollout_dir

def get_last_experiment_dir():
    paths = Path(ROLLOUTS, "guided").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]

def batchify_and_move(sample, device):
    return {
        k: v[None].to(device) if hasattr(v, "to") else v
        for k, v in sample.items()
    }

def read_state(path: Path):
    return xr.open_dataset(path, engine="netcdf4")

def get_slice(state, partition, level, var, timestamp):
    if partition == "surface":
        return state[var].sel(time=timestamp, method='nearest')
    else: 
        return state[var].sel(time=timestamp, level=level, method='nearest')
    
def xr_to_torch(slice_: xr.DataArray):
    return torch.tensor(slice_.to_numpy())

def list_tens_to_floats(list_):
    return [tensor.item() for tensor in list_]

def tensor_timestamp_to_string(
    timestamp: torch.Tensor,
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    ts = timestamp.item()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)


##### experiments utils

def read_config(config_type: str, config_id: str) -> dict[str, Any]:
    return read_json(CONFIGS / config_type, config_id)

def update_experiment_params(
    rollout_dir: Path,
    config: dict[str, Any],
) -> dict[str, list[Any]]:
    try:
        experiment_params = read_json(rollout_dir, "experiment_params")
    except FileNotFoundError:
        experiment_params = {k: [] for k in GUIDANCE_PARAM_KEYS}

    for k in GUIDANCE_PARAM_KEYS:
        experiment_params.setdefault(k, [])

        value = config[k]
        if value not in experiment_params[k]:
            experiment_params[k].append(value)
        experiment_params[k] = sorted(
            experiment_params[k],
            key=lambda x: (str(type(x)), x),
        )

    save_to_json(experiment_params, rollout_dir, "experiment_params")
    return experiment_params
