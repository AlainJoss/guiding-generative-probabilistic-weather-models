import json

from pathlib import Path 
from datetime import datetime

import torch

from src.utils import (
    save_state, 
    ensure_result_dir,
    save_config
)
from src.funcs import get_mask_tensordict
from src.utils import get_mask_from_corners

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

# TODO: create run func and call from marimo after setup!

##### load #####

def rollout_step(
        ds, x_start, 
        gen_model, 
        mask_corners, y_t, N,
        timestamp, timestamp_idx, 
        partition, level, level_idx, var, var_idx
    ):
    device = gen_model.device
    mask = get_mask_from_corners(*mask_corners)
    mask = mask.to(device)
    y_t = y_t.to(device)

    # note: something is not unrolled here ...
    mask = get_mask_tensordict(x_start["state"][0], partition, var_idx, level_idx, mask)

    ##### run #####
    sampled_state = gen_model.rollout_step(
        x_cond=x_start, 
        y_t=y_t, 
        mask=mask
    )
    # sampled_state = x_start["state"]
    # TODO: this has later to be moved to rollout(N)
    sampled_state = gen_model.denormalize(sampled_state)
    sampled_state = sampled_state.cpu()

    ##### save #####

    result_dir = ensure_result_dir()

    sampled_state = ds.convert_to_xarray(sampled_state, x_start["timestamp"].cpu())
    save_state(result_dir, sampled_state)

    config = {
        "N": N,
        "mask_corners": mask_corners,
        "timestamp": str(timestamp),
        "timestamp_idx": int(timestamp_idx),
        "partition": partition,
        "level": None if level is None else str(level),
        "level_idx": None if level_idx is None else int(level_idx),
        "var": var,
        "var_idx": int(var_idx),
        "y_t": y_t.detach().cpu().tolist(),
    }
    save_config(result_dir, config)