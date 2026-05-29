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
from src.rollout_config import SWEEP_PARAMS, RolloutConfig
from src.target import get_reference_rollout, get_target_rollout
from src.mask import get_mask_tdict, get_mask_2d

from geoarches.lightning_modules.guided_diffusion import GuidedFlow

def rollout(
    config: RolloutConfig,
    flow_model: GuidedFlow | None = None,
    test: bool = False,
):  
    flow_model.move_objects_to_device()
    rollout_dir = ensure_rollout_dir(config.rollout_id)
    sweep_params = {
        "guidance_reference": config.guidance_reference,
        "mask_mode": config.mask_mode,
        "alpha": config.alpha,
        "w": config.w,
    }
    guided_id = make_hash(sweep_params)
    
    if config.guidance_flag:
        guided_path = rollout_dir / "guided_rollout" / guided_id
        output_path = guided_path / "guided_rollout.nc"

        if output_path.exists() and not test:
            print(f"Skipping existing guided rollout: {guided_path}")
            return rollout_dir

        guided_path.mkdir(parents=True, exist_ok=True)
        update_sweep_params(rollout_dir, config.to_dict(), SWEEP_PARAMS)

    var_idx = get_var_idx(config.partition, config.var)
    level_idx = get_level_idx(config.partition, config.level)
    mask_2d = get_mask_2d(config.mask_mode, config.mask_corners)
    lambda_schedule = T_schedule(config.alpha, config.w)

    ds = get_td_dataset(multistep=config.N)
    x_cond = get_x_cond(ds, config.timestamp)

    device = get_device()
    x_cond = batchify_and_move(x_cond, device)
    current_timestamp = x_cond["timestamp"].clone()

    gui_member_datasets = []
    ung_member_datasets = []
    for m in range(config.M):
        logger.info(f"sampling member={m}")

        if config.guidance_flag and not test and config.guidance_reference != "sampled_trajectory":
            # reference_rollout = get_reference_rollout(config.guidance_reference, config.rollout_id, m, config.N, config.timestamp)
            # target_rollout = get_target_rollout(
            #     config.partition, 
            #     var_idx,
            #     level_idx,
            #     config.delta_trajectory,
            #     reference_rollout
            # )
            # target_tdicts = [
            #     xr_rollout_slice_to_tdict(target_rollout.isel(time=n)).unsqueeze(0).to(device)
            #     for n in range(config.N)
            # ]
            mask_tdict = get_mask_tdict(x_cond["state"], config.partition, var_idx, level_idx, mask_2d)
            
            sampling_trace_path = ROLLOUTS / config.rollout_id / "guided_rollout" / guided_id
            # sampling_trace_path=None
        else:
            target_tdicts = None
            mask_tdict = None
            sampling_trace_path=None

        if not test:
            gui_trajectory, ung_trajectory = flow_model.sample_rollout( 
                config.N, 
                lambda_schedule,
                m=m,
                x_cond=x_cond,
                # target_states=None, # before we had target_tdicts, but not anymore with new loss
                mask=mask_tdict,
                delta_trajectory=config.delta_trajectory[1:], 
                sampling_trace_path=sampling_trace_path,
                T=25
            )
        else:
            gui_trajectory = x_cond["future_states"].cpu()
            ung_trajectory = x_cond["future_states"].cpu()

        if config.guidance_flag:
            gui_trajectory = torch.cat(
                [x_cond["state"].unsqueeze(1).cpu(), gui_trajectory],  # unsqueeze adds batch dim
                dim=1,
            ).squeeze(0)
            gui_trajectory = ds.denormalize(gui_trajectory)
            gui_trajectory = rollout_to_xarray(
                sample_multistep=gui_trajectory,
                start_timestamp= current_timestamp,
                member=m,
            )
            gui_member_datasets.append(gui_trajectory)

        ung_trajectory = ung_trajectory.squeeze(0)
        ung_trajectory = ds.denormalize(ung_trajectory)
        ung_trajectory = rollout_to_xarray(
            sample_multistep=ung_trajectory,
            start_timestamp= current_timestamp,
            member=m,
        )
        ung_member_datasets.append(ung_trajectory)

    if config.guidance_flag:
        xr_pred_gui = xr.concat(gui_member_datasets, dim="member", join='exact')

    xr_pred_ung = xr.concat(ung_member_datasets, dim="member", join='exact')

    config_dict = config.to_dict()

    if config.guidance_flag:
        xr_pred_gui.to_netcdf(guided_path / "guided_rollout.nc")
        xr_pred_ung.to_netcdf(guided_path / "unguided_rollout.nc")
        dump_json(config_dict, guided_path, "config")
    else:
        xr_pred_ung.to_netcdf(rollout_dir / "unguided_rollout.nc")
        dump_json(config_dict, rollout_dir, "config")
    return rollout_dir