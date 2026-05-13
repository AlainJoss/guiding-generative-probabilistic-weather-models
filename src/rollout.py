from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import xarray as xr
import torch

from src.utils import (
    save_to_json,
    rollout_to_xarray,
    batchify_and_move,
    get_dataset,
    get_x_cond,
    make_hash,
    list_floats_to_tensors,
    update_experiment_params,
    ensure_rollout_dir
)
from src.config import GUIDANCE_PARAMS

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict


def rollout(
    config: dict,
    flow_model: GuidedFlow | None = None,
    y: list[TensorDict] | None = None,
    test: bool = False,
):
    rollout_dir = ensure_rollout_dir(config.rollout_id)
    if config.guidance_flag:
        update_experiment_params(rollout_dir, config, GUIDANCE_PARAMS)

    ds = get_dataset(multistep=config.N)
    x_cond, _ = get_x_cond(ds, config.timestamp)

    device = flow_model.device
    x_cond = batchify_and_move(x_cond, device)
    lead_time_hours = int(x_cond["lead_time_hours"].cpu().flatten()[0].item())

    if config.guidance_flag:
        lambda_ = list_floats_to_tensors(config.lambda_, device)

    if not config.guidance_flag or test:
        ground_truth = torch.cat(
            [x_cond["state"].unsqueeze(1), x_cond["future_states"]],
            dim=1,
        )
        ground_truth = rollout_to_xarray(
            ds=ds,
            sample_multistep=ground_truth,
            init_timestamp=x_cond["timestamp"],
            member=-1,
            lead_time_hours=lead_time_hours,
            include_init=True,
        )
        y = None

    member_datasets = []
    for m in range(config.M):
        logger.info(f"sampling member={m}")
        if not test:
            sample_multistep = flow_model.sample_rollout(
                N=config.N,
                m=m, 
                x_cond=x_cond,
                y=y,
                lambda_=lambda_
            )
        else: 
            sample_multistep = x_cond["future_states"]

        xr_member = rollout_to_xarray(
            ds=ds,
            sample_multistep=sample_multistep,
            init_timestamp=x_cond["timestamp"], 
            member=m,
            lead_time_hours=lead_time_hours,
        )
        member_datasets.append(xr_member)

    xr_pred = xr.concat(member_datasets, dim="member")

    if config.guidance_flag:
        params = {
            "guidance_mode": config.guidance_mode,
            "alpha": config.alpha,
            "w": config.w,
        }

        guided_id = make_hash(params)
        guided_path = rollout_dir / "guided" / guided_id
        guided_path.mkdir(parents=True, exist_ok=True)

        xr_pred.to_netcdf(guided_path / "guided.nc")
        save_to_json(config, guided_path, "config")
    else:
        save_to_json(config, rollout_dir, "config")
        xr_pred.to_netcdf(rollout_dir / "unguided.nc")
        ground_truth.to_netcdf(rollout_dir / "ground_truth.nc")