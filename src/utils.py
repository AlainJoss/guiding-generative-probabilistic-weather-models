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


def get_mask_corners_from_widget(map_widget):
    x0, x1 = map_widget.value["x"]
    y0, y1 = map_widget.value["y"]

    lat_bottom, lat_top = sorted([y0, y1])

    # keep widget longitudes in plot coordinates [-180, 180]
    lon_left_plot = x0
    lon_right_plot = x1

    # convert each endpoint separately to ERA5 storage coords [0, 360]
    lon_left = lon_left_plot % 360.0
    lon_right = lon_right_plot % 360.0

    return lon_left, lon_right, lat_bottom, lat_top

def get_mask_from_corners(lon_left, lon_right, lat_bottom, lat_top):
    lon_e = np.linspace(0.0, 360.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    # handle seam-crossing selections, e.g. [350, 20]
    if lon_left <= lon_right:
        lon_mask = (lon_grid >= lon_left) & (lon_grid <= lon_right)
    else:
        lon_mask = (lon_grid >= lon_left) | (lon_grid <= lon_right)

    lat_mask = (lat_grid >= lat_bottom) & (lat_grid <= lat_top)

    mask = (lon_mask & lat_mask).astype(np.uint8)
    return torch.as_tensor(mask)