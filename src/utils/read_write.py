import json
from typing import Any
from pathlib import Path 

import xarray as xr
import numpy as np

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

from src.paths import ERA5, MODELSTORE, ROLLOUTS, RUN_CONFIGS
from src.rollout_config import RolloutConfig


##### data and model #####


def get_xr_dataset():
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
        case "unguided_rollout":
            path = ROLLOUTS / rollout_id 
            config_path = path / "config.json"
            ds_path = path / f"unguided_rollout.nc"
        case "guided_rollout":
            path = ROLLOUTS / rollout_id / "guided_rollout" / guided_id
            config_path = path / "config.json"
            ds_path = path / f"guided_rollout.nc"
        case "clean_preds" | "grads" | "guided_vfs" | "vfs":
            path = ROLLOUTS / rollout_id / "guided_rollout" / guided_id
            config_path = path / "config.json"
            ds_path = path / f"{rollout_type}.nc"
        case _:
            raise ValueError(f"Not a valid rollout_type: {rollout_type}")
    ds = xr.open_dataset(ds_path, engine="netcdf4")
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

def dump_json(dict_: dict, rollout_dir: Path, name:str):
    path = rollout_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(dict_, f, indent=2)


def update_sweep_params(
    rollout_dir: Path,
    config: dict[str, Any],
    params: list["str"]
) -> dict[str, list[Any]]:
    """
    Pass config and update sweep_param.json with its sweep values.
    """
    # reads existing dict or create empty one
    try:
        path = rollout_dir / "experiment_params.json"
        sweep_params = get_dict_from_json(path)
    except FileNotFoundError:
        sweep_params = {k: [] for k in params}

    for param_key in params:
        # create default value
        sweep_params.setdefault(param_key, [])
        
        # list
        values = config[param_key]
    
        # update values list if value is not already there
        if values not in sweep_params[param_key]:
            sweep_params[param_key].append(values)

        # sort for later access 
        sweep_params[param_key] = sorted(
            sweep_params[param_key],
            key=lambda x: (str(type(x)), x),
        )

    dump_json(sweep_params, rollout_dir, "sweep_params")