from pathlib import Path 
from datetime import datetime

import torch
import xarray as xr
import numpy as np



def get_last_experiment_id():
    paths = Path("data", "results").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]


def state_to_device(state, device):
    return {k: v[None].to(device) for k, v in state.items()}


def save_state(array: xr.DataArray):
    def get_timestamp():
        date, time = str(datetime.now().replace(microsecond=0)).split(" ")
        timestamp = date + "_" + time
        return timestamp
    timestamp = get_timestamp()
    path = Path("data", "results", f"{timestamp}")
    path.mkdir(parents=True, exist_ok=True)
    path = path / "state.nc"
    array.to_netcdf(path)


def get_mask_from_widget(map_widget):
    x0, x1 = map_widget.value["x"]
    y0, y1 = map_widget.value["y"]

    lon_left_plot, lon_right_plot = sorted([x0, x1])
    lat_bottom, lat_top = sorted([y0, y1])

    # widget is in [-180, 180], ERA5 storage grid is [0, 360]
    lon_left = lon_left_plot % 360.0
    lon_right = lon_right_plot % 360.0

    lon_e = np.linspace(0.0, 360.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    # handle seam-crossing selections, e.g. [-20, 20]
    if lon_left <= lon_right:
        lon_mask = (lon_grid >= lon_left) & (lon_grid <= lon_right)
    else:
        lon_mask = (lon_grid >= lon_left) | (lon_grid <= lon_right)

    lat_mask = (lat_grid >= lat_bottom) & (lat_grid <= lat_top)

    mask = (lon_mask & lat_mask).astype(np.uint8)
    return mask