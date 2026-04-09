import json
from pathlib import Path 
from datetime import datetime

import torch
import xarray as xr

import json
from pathlib import Path
from datetime import datetime
import torch

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

def save_to_json(dict_: dict, result_dir: Path, name:str):
    path = result_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(dict_, f, indent=2)

def read_json(result_dir, name:str):
    path = Path(result_dir) / f"{name}.json"
    with open(path, "r") as f:
        dict_ = json.load(f)
    return dict_

def get_dataset():
    return Era5Forecast(
        path="data/era5_240/full",  # default path
        domain="test", # domain to consider. domain = 'test' loads the 2020 period
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=6
    )

def get_model(device):
    gen_model, _ = load_module(  # _ := gen_config
        "archesweathergen",
        module_target="geoarches.lightning_modules.guided_diffusion.GuidedFlow",
    )
    return gen_model.to(device)

def get_timestamp():
    date, time = str(datetime.now().replace(microsecond=0)).split(" ")
    timestamp = date + "_" + time
    return timestamp

def ensure_ensemble_rollouts_dir(N: int):
    timestamp = get_timestamp()
    result_dir = Path("data", f"ensemble_rollouts", f"{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, N+1):
        path = Path(result_dir, f"{n}")
        path.mkdir(parents=True, exist_ok=True)
    return result_dir

def ensure_guided_rollouts_dir():
    timestamp = get_timestamp()
    result_dir = Path("data", f"guided_rollouts", f"{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    path = Path(result_dir, "guided")
    path.mkdir(parents=True, exist_ok=True)
    path = Path(result_dir, "unguided")
    path.mkdir(parents=True, exist_ok=True)
    return result_dir

def save_config(result_dir, config):
    path = result_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def read_config(result_dir):
    path = Path(result_dir) / "config.json"
    with open(path, "r") as f:
        config = json.load(f)
    return config

def get_last_experiment_dir():
    paths = Path("data", "results").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]

def get_experiment_dir():
    paths = Path("data", "results").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]

def state_to_device(state, device):
    return {k: v[None].to(device) for k, v in state.items()}

def save_state(result_dir: Path, array, state_type: str, step: int):
    """
    file: either "guided" or "unguided" in the current version.
    """
    path = result_dir / f"{state_type}" / f"{step}.nc"
    array.to_netcdf(path)

def read_state(result_dir, state_type, step: int):
    """
    file: either "guided" or "unguided" in the current version.
    """
    path = result_dir / f"{state_type}" / f"{step}.nc"
    return xr.open_dataset(path, engine="netcdf4")


def get_slice(state, partition, level, var, timestamp):
    if partition == "surface":
        return state[var].sel(time=timestamp, method='nearest')
    else: 
        return state[var].sel(time=timestamp, level=level, method='nearest')