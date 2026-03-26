import json
from pathlib import Path 
from datetime import datetime

import torch
import xarray as xr
import numpy as np

import json
from pathlib import Path
from datetime import datetime
import torch

def ensure_result_dir():
    def get_timestamp():
        date, time = str(datetime.now().replace(microsecond=0)).split(" ")
        timestamp = date + "_" + time
        return timestamp
    timestamp = get_timestamp()
    path = Path("data", "results", f"{timestamp}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_config(result_dir, config):
    path = result_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def read_config(result_dir):
    path = Path(result_dir) / "config.json"
    with open(path, "r") as f:
        config = json.load(f)

    config["timestamp"] = datetime.fromisoformat(config["timestamp"])
    config["y_t"] = torch.tensor(config["y_t"])
    return config

def get_last_experiment_dir():
    paths = Path("data", "results").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]


def state_to_device(state, device):
    return {k: v[None].to(device) for k, v in state.items()}


def save_state(result_dir: Path, array: xr.DataArray):
    path = result_dir / "state.nc"
    array.to_netcdf(path)

def read_state(result_dir):
    path = result_dir / "state.nc"
    return xr.open_dataset(path, engine="netcdf4")