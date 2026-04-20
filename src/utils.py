import json
import random
import string

from pathlib import Path 
from datetime import datetime

import xarray as xr
import torch

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast


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
    timesteps = ["6", "12", "18", "0"]
    datasets = [
        xr.open_dataset(f"data/era5_240/full/era5_240_2020_{ts}h.nc", engine="netcdf4")
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

def ensure_rollout_dir(sub_dir: Path, N):
    timestamp = get_timestamp()
    result_dir = Path("rollouts", sub_dir, f"{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, N+1):
        path = Path(result_dir, f"{n}")
        path.mkdir(parents=True, exist_ok=True)
    return result_dir

def get_last_experiment_dir():
    paths = Path("rollouts", "guided").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]

def state_to_device(state, device):
    return {k: v[None].to(device) for k, v in state.items()}

def save_state(rollout_dir: Path, array, n: int, m: int):
    path = rollout_dir / f"{n}" / f"{m}.nc"
    array.to_netcdf(path)

def read_state(path: Path):
    return xr.open_dataset(path, engine="netcdf4")

def read_states(rollout_dir: Path, n: int):
    paths = [path for path in rollout_dir.glob(f"{n}/*")]
    return [read_state(path) for path in paths]

def get_slice(state, partition, level, var, timestamp):
    if partition == "surface":
        return state[var].sel(time=timestamp, method='nearest')
    else: 
        return state[var].sel(time=timestamp, level=level, method='nearest')
    
def xr_to_torch(slice_: xr.DataArray):
    return torch.tensor(slice_.to_numpy())

def tensordict_to_xr(tensordict):
    return 

def list_tens_to_floats(list_):
    return [tensor.item() for tensor in list_]