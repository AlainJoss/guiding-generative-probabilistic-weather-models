from pathlib import Path

import xarray as xr
import torch

from src.utils import get_x_cond
from src.utils import (
    batchify_and_move,
    get_var_idx, get_level_idx,
)
from src.funcs import T_schedule
from src.rollout_config import RolloutConfig
from src.mask import get_mask_tdict, get_mask_2d

from src.utils import create_slice_zarr_container, tdict_to_xr, append_to_zarr, advance_x_cond

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict


def rollout(
    rollout_dir: Path,
    guidance_flag: bool,
    config: RolloutConfig,
    flow_model: GuidedFlow,
    sweep_params: dict[str, any]
):  
    # create objects from config
    x_cond = get_x_cond(config.timestamp, config.N)
    x_cond = batchify_and_move(x_cond, flow_model.device)

    if guidance_flag and config.guidance_reference != "sampled_trajectory":
        var_idx = get_var_idx(config.partition, config.var)
        level_idx = get_level_idx(config.partition, config.level)
        mask_2d = get_mask_2d(config.mask_mode, config.mask_corners)

        delta_trajectory = config.delta_trajectory
        lambda_schedule = T_schedule(config.alpha, config.w) 
        mask_tdict = get_mask_tdict(x_cond["state"], config.partition, var_idx, level_idx, mask_2d)
    else:
        delta_trajectory=None
        lambda_schedule=None
        mask_tdict = None

    # run
    for m in range(config.M):
        # note how config gets not passed onwards
        # appends to zarr with m, n, sweep_params
        sample_rollout( 
            rollout_dir=rollout_dir,
            guidance_flag=guidance_flag,
            flow_model=flow_model,
            sweep_params=sweep_params,
            N=config.N, 
            m=m,
            x_cond=x_cond,
            mask=mask_tdict,
            delta_trajectory=delta_trajectory,
            lambda_schedule=lambda_schedule
        )

def sample_rollout(
    rollout_dir: Path,
    guidance_flag,
    flow_model: GuidedFlow,
    sweep_params: dict[str, any],
    m,
    N,
    x_cond, 
    mask: TensorDict | None = None,
    delta_trajectory: list[torch.Tensor] | None = None,
    lambda_schedule: list[torch.Tensor] | None = None
):  
    # flow_model.T=25
    for n in range(N):
        # runs both if guidance
        x_hat_ung, _ = flow_model.sample(
            guidance_flag=False,
            x_cond=x_cond,
            delta_t=None,
            mask=None,
            x_hat_ung=None, 
            lambda_schedule=None,
            seed= m + 1000 * n,  # + batch_nb * 10**6
        )
        x_hat_curr = x_hat_ung

        if guidance_flag:
            x_hat_gui, sampling_trace = flow_model.sample(
                guidance_flag=True,
                x_cond=x_cond,
                delta_t=delta_trajectory[n],
                mask=mask,
                x_hat_ung=flow_model.denormalize(x_hat_ung.detach().clone()), 
                lambda_schedule=lambda_schedule,
                seed = m + 1000 * n,  # + batch_nb * 10**6
            )
            x_hat_curr = x_hat_gui
            
        if guidance_flag:
            save_state = flow_model.denormalize(x_hat_gui).cpu()
            save_state = tdict_to_xr(
                create_slice_zarr_container(m, n, t_dim=False, sweep_params=sweep_params),
                save_state,
                t_dim=False
            )
            append_to_zarr(rollout_dir, "gui", save_state)

            save_state = flow_model.denormalize(x_hat_ung).cpu()
            save_state = tdict_to_xr(
                create_slice_zarr_container(m, n, t_dim=False, sweep_params=sweep_params),
                save_state,
                t_dim=False
            )
            append_to_zarr(rollout_dir, "ung_gui", save_state)

            for trace_type, trace in sampling_trace.items():
                save_trace = torch.stack(trace, dim=0)
                save_trace = tdict_to_xr(
                    create_slice_zarr_container(m, n, t_dim=True, sweep_params=sweep_params),
                    save_trace,
                    t_dim=True
                )
                append_to_zarr(rollout_dir, trace_type, save_trace)

        else:
            save_state = flow_model.denormalize(x_hat_ung).cpu()
            save_state = tdict_to_xr(
                create_slice_zarr_container(m, n, t_dim=False, sweep_params={}),
                save_state,
            )
            append_to_zarr(rollout_dir, "ung", save_state)

        # after the last iteration no need to set this again
        if n < N-1:
            x_cond = advance_x_cond(x_cond, x_hat_curr)