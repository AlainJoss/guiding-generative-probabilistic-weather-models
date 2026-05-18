import json
from typing import Any
from pathlib import Path 

import xarray as xr

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

from src.paths import ERA5, MODELSTORE, ROLLOUTS, RUN_CONFIGS
from src.config import RolloutConfig


##### data and model #####

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


##### read files #####

def _get_dict_from_json(rollout_dir: Path, name:str):
    path = rollout_dir / f"{name}.json"
    with open(path, "r") as f:
        dict_ = json.load(f)
    return dict_


def get_rollout_config(rollout_type: str, rollout_id: str) -> dict[str, Any]:
    match rollout_type:
        case "guided":
            rollout_dir = _get_rollout_dir_path(rollout_id) / rollout_type
        case "unguided":
            rollout_dir = _get_rollout_dir_path(rollout_id)
        case _:
            raise ValueError(f"Not a valid rollout_type: {rollout_type}")
        
    config_dict = _get_dict_from_json(rollout_dir, "config")
    return RolloutConfig.from_dict(config_dict)


def get_run_config(config_type: str, config_id: str) -> dict[str, Any]:
    path = RUN_CONFIGS / config_type
    config_dict = _get_dict_from_json(path, config_id)
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


def _get_rollout_dir_path(id_: str):
    return ROLLOUTS / id_


def get_rollout_xr(rollout_id: str, type_: str):
    rollout_dir = _get_rollout_dir_path(rollout_id) / f"{type_}.nc"
    return xr.open_dataset(rollout_dir)


def read_json(rollout_id, name:str):
    rollout_dir = _get_rollout_dir_path(rollout_id)
    path = Path(rollout_dir) / f"{name}.json"
    with open(path, "r") as f:
        dict_ = json.load(f)
    return dict_


##### write files #####

def save_to_json(dict_: dict, rollout_dir: Path, name:str):
    path = rollout_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(dict_, f, indent=2)

def update_experiment_params(
    rollout_dir: Path,
    config: dict[str, Any],
    params: list["str"]
) -> dict[str, list[Any]]:
    try:
        experiment_params = _get_dict_from_json(rollout_dir, "experiment_params")
    except FileNotFoundError:
        experiment_params = {k: [] for k in params}

    for k in params:
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