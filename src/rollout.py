import logging

logger = logging.getLogger(__name__)

import xarray as xr
import torch

from src.utils.read_write import (
    dump_json,
    get_td_dataset,
    update_sweep_params,
)
from src.utils.dataset_utils import get_x_cond
from src.utils.converters import (
    rollout_to_xarray,
    batchify_and_move,
    xr_rollout_slice_to_tdict,
    get_var_idx, get_level_idx,
)
from src.paths import ROLLOUTS
from src.utils.setup import ensure_rollout_dir, get_device
from src.funcs import make_hash, T_schedule
from src.rollout_config import GUIDANCE_PARAMS, RolloutConfig
from src.target import get_reference_rollout, get_target_rollout
from src.mask import get_mask_tdict, get_mask_2d

from geoarches.lightning_modules.guided_diffusion import GuidedFlow

def rollout(
    config: RolloutConfig,
    flow_model: GuidedFlow | None = None,
    test: bool = False,
):  
    var_idx = get_var_idx(config.partition, config.var)
    level_idx = get_level_idx(config.partition, config.level)
    mask_2d = get_mask_2d(config.mask_mode, config.mask_corners)
    lambda_schedule = T_schedule(config.alpha, config.w)
    rollout_dir = ensure_rollout_dir(config.rollout_id)
    if config.guidance_flag:
        update_sweep_params(rollout_dir, config.to_dict(), GUIDANCE_PARAMS)

    ds = get_td_dataset(multistep=config.N)
    x_cond = get_x_cond(ds, config.timestamp)

    device = get_device()
    x_cond = batchify_and_move(x_cond, device)

    member_datasets = []
    for m in range(config.M):
        logger.info(f"sampling member={m}")

        if config.guidance_flag and not test and config.guidance_reference != "sampled_trajectory":
            reference_rollout = get_reference_rollout(config.guidance_reference, config.rollout_id, m)
            target_rollout = get_target_rollout(
                config.partition, 
                var_idx,
                level_idx,
                config.delta_trajectory,
                reference_rollout
            )
            target_tdicts = [
                xr_rollout_slice_to_tdict(target_rollout.isel(time=n)).unsqueeze(0).to(device)
                for n in range(config.N)
            ]
            masks_tdict = get_mask_tdict(x_cond["state"], config.partition, var_idx, level_idx, mask_2d)
            
            sweep_params = {
                "guidance_reference": config.guidance_reference,
                "alpha": config.alpha,
                "w": config.w,
            }
            guided_id = make_hash(sweep_params)
            sampling_trace_path = ROLLOUTS / config.rollout_id / "guided_rollout" / guided_id
            sampling_trace_path=None
        else:
            target_tdicts = None
            masks_tdict = None
            sampling_trace_path=None

        current_timestamp = x_cond["timestamp"]
        if not test:
            sample_multistep = flow_model.sample_rollout(
                config.N, 
                lambda_schedule,
                m=m,
                x_cond=x_cond,
                y=target_tdicts,
                mask=masks_tdict,
                sampling_trace_path=sampling_trace_path
            )
        else:
            sample_multistep = x_cond["future_states"]
        
        sample_multistep = torch.cat(
            [x_cond["state"].unsqueeze(1).cpu(), sample_multistep],  # unsqueeze adds batch dim
            dim=1,
        ).squeeze(0)
        sample_multistep = ds.denormalize(sample_multistep)
        xr_member = rollout_to_xarray(
            sample_multistep=sample_multistep,
            start_timestamp= current_timestamp,
            member=m,
        )
        member_datasets.append(xr_member)

    xr_pred = xr.concat(member_datasets, dim="member")

    config_dict = config.to_dict()

    if config.guidance_flag:
        guided_path = rollout_dir / "guided_rollout" / guided_id
        guided_path.mkdir(parents=True, exist_ok=True)

        xr_pred.to_netcdf(guided_path / "guided_rollout.nc")
        dump_json(config_dict, guided_path, "config")
    else:
        dump_json(config_dict, rollout_dir, "config")
        xr_pred.to_netcdf(rollout_dir / "unguided_rollout.nc")
