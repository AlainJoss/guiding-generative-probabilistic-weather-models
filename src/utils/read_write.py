import json
from typing import Any
from pathlib import Path 

import xarray as xr
import numpy as np

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

from src.paths import ERA5, MODELSTORE, ROLLOUTS, RUN_CONFIGS
from src.dimensions import VARIABLES_DICT
from src.rollout_config import RolloutConfig


##### data and model #####


def get_xr_dataset():
    from src.paths import ERA5
    path = ERA5 / "arches_era5.nc"
    ds = xr.open_dataset(path)
    return ds


def get_td_dataset(multistep:int=1):
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
    return gen_model.to(device)


##### read files #####

def get_dict_from_json(path: Path):
    with open(path, "r") as f:
        dict_ = json.load(f)
    return dict_


def get_rollout_files(rollout_type: str, rollout_id: str, guided_id:str=None) -> RolloutConfig:
    match rollout_type:
        case "unguided" | "ground_truth":
            path = ROLLOUTS / rollout_id 
        case "guided":
            path = ROLLOUTS / rollout_id / rollout_type / guided_id
        case _:
            raise ValueError(f"Not a valid rollout_type: {rollout_type}")
    ds_path = path / f"{rollout_type}.nc"
    ds = xr.open_dataset(ds_path, engine="netcdf4")
    config_path = path / "config.json"
    config_dict = get_dict_from_json(config_path)
    config = RolloutConfig.from_dict(config_dict)
    return ds, config


def get_run_config(rollout_type: str, rollout_id: str) -> RolloutConfig:
    path = RUN_CONFIGS / rollout_type / f"{rollout_id}.json"
    config_dict = get_dict_from_json(path)
    return RolloutConfig.from_dict(config_dict)


def get_rollout_ids(type_: str):
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

##### write files #####

def save_to_json(dict_: dict, rollout_dir: Path, name:str):
    path = rollout_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(dict_, f, indent=2)

# TODO: pass dict instead of conf
def update_experiment_params(
    rollout_dir: Path,
    config: dict[str, Any],
    params: list["str"]
) -> dict[str, list[Any]]:
    try:
        path = rollout_dir / "experiment_params.json"
        experiment_params = get_dict_from_json(path)
    except FileNotFoundError:
        experiment_params = {k: [] for k in params}

    for param_key in params:
        experiment_params.setdefault(param_key, [])

        value = config[param_key]
        if value not in experiment_params[param_key]:
            experiment_params[param_key].append(value)
        experiment_params[param_key] = sorted(
            experiment_params[param_key],
            key=lambda x: (str(type(x)), x),
        )

    save_to_json(experiment_params, rollout_dir, "experiment_params")
    return experiment_params