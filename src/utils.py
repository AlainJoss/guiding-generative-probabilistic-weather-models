import json

from pathlib import Path 
from datetime import datetime, timezone

import xarray as xr
import torch

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

from src.paths import ERA5, MODELSTORE, ROLLOUTS, CONFIGS


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

def get_dataset():
    return Era5Forecast(
        path=ERA5,  # default path
        domain="all",  # all files under ERA5; year-slicing happens on the time coord
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=24,
        timedelta_hours=6
    )

def get_model(device):
    print(MODELSTORE)
    gen_model, _ = load_module(  # _ := gen_config
        MODELSTORE / "archesweathergen",
        module_target="geoarches.lightning_modules.guided_diffusion.GuidedFlow",
    )
    return gen_model.to(device)

def get_now_timestamp():
    date, time = str(datetime.now().replace(microsecond=0)).split(" ")
    return date + "_" + time

def get_new_rollout_dir_path(sub_dir: str):
    experiment_id = get_now_timestamp()
    return Path(ROLLOUTS, sub_dir, f"{experiment_id}")

def ensure_new_config_dir_path(sub_dir: str):
    experiment_id = get_now_timestamp()
    config_dir = Path(CONFIGS, sub_dir, f"{experiment_id}")
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def ensure_rollout_dir(sub_dir: Path, N) -> Path:
    rollout_dir = get_new_rollout_dir_path(sub_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, N+1):
        path = Path(rollout_dir, f"{n}")
        path.mkdir(parents=True, exist_ok=True)
    return rollout_dir

def get_last_experiment_dir():
    paths = Path(ROLLOUTS, "guided").glob("2026*")
    paths = sorted(paths)
    print(paths[-1])
    return paths[-1]

def state_to_device(state, device):
    return {k: v[None].to(device) for k, v in state.items()}

def save_state(rollout_dir: str, array, n: int, m: int):
    path = Path(rollout_dir, f"{n}", f"{m}.nc")
    array.to_netcdf(path)

def read_state(path: Path):
    return xr.open_dataset(path, engine="netcdf4")

def read_states(rollout_dir: Path, state_type: str, n: int):
    paths = list((rollout_dir / f"{n}").glob(f"{state_type}_[0-9]*.nc"))    
    paths.sort(key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    return [read_state(p) for p in paths]

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