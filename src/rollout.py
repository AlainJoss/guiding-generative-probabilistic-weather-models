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
    update_experiment_params,
    ensure_rollout_dir
)
from src.config import GUIDANCE_PARAMS
from src.target import get_reference_trajectory, get_target_trajectory, get_mask

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict


def rollout(
    config: dict,
    flow_model: GuidedFlow | None = None,
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
        target_trajectory = None

    member_datasets = []
    for m in range(config.M):
        logger.info(f"sampling member={m}")
        if not test:
            target_trajectory = get_reference_trajectory(config, m)  
            target_trajectory = get_target_trajectory(config, target_trajectory)
            masks = [get_mask(config, y_n) for y_n in target_trajectory]
            # TODO: save both target_trajectory and masks
            sample_multistep = flow_model.sample_rollout(
                config,
                m=m,
                x_cond=x_cond,
                y=target_trajectory, 
                masks=masks
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