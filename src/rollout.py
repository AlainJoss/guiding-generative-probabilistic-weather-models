from pathlib import Path

import xarray as xr
import torch

from src.utils import get_x_cond
from src.utils import (
    batchify_and_move,
    get_var_idx, get_level_idx,
)
from src.schedules import T_schedule
from src.rollout_config import RolloutConfig
from src.mask import get_mask_tdict, get_mask_2d

from src.utils import create_slice_zarr_container, tdict_to_xr, append_to_zarr, advance_x_cond
from src.utils import append_diagnostics

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict


# t_dim=True trace keys that have a matching zarr container (see get_container_args
# in run.py). Other sampling_trace entries are method-specific diagnostics with no
# container and non-stackable shapes -- FlowGrad's "flowgrad" (per-iteration loss
# dict), "lambda_star" (T,), and "control_star" (list of TensorDicts) -- and are
# skipped from the zarr save so a FLOWGRAD/FLOWGRAD_FREE sweep runs end-to-end.
TRACE_CONTAINERS = ("clean_preds", "grads", "vfs", "gui_vfs")


def build_guidance_kwargs(config: RolloutConfig) -> dict[str, any]:
    """Select the guidance hyperparameters relevant to config.guidance_mode and
    drop unset (None) ones so the method defaults in GuidedFlow still apply."""
    shared = {
        "regularized": config.regularized,
        "beta": config.beta,
        "normalize": config.normalize,
        "eps": config.eps,
    }

    match config.guidance_mode:
        case "DPS":
            kwargs = shared
        case "LBG":
            kwargs = {**shared, "n_mc": config.lbg_n_mc, "r_t": config.lbg_r_t}
        case "UG":
            kwargs = {
                **shared,
                "S": config.ug_S,
                "m": config.ug_m,
                "delta_lr": config.ug_delta_lr,
            }
        case "FLOWGRAD" | "FLOWGRAD_FREE":
            kwargs = {
                **shared,
                "n_opt": config.fg_n_opt,
                "lr": config.fg_lr,
                "gamma": config.fg_gamma,
                "init_lambda": config.fg_init_lambda,
                "n_lambda": config.fg_n_lambda,
            }
        case _:
            kwargs = {}

    # drop Nones
    return {k: v for k, v in kwargs.items() if v is not None}


def rollout(
    rollout_dir: Path,
    guidance_flag: bool,
    config: RolloutConfig,
    flow_model: GuidedFlow,
    T: int = 25,
    sweep_params: dict[str, any] | None = None
):  
    # for testing purposes
    flow_model.T=T
    # create objects from config
    x_cond = get_x_cond(config.timestamp, config.N)
    x_cond = batchify_and_move(x_cond, flow_model.device)

    if guidance_flag and config.guidance_reference != "sampled_trajectory":
        var_idx = get_var_idx(config.partition, config.var)
        level_idx = get_level_idx(config.partition, config.level)
        mask_2d = get_mask_2d(config.mask_mode, config.mask_corners)

        delta_trajectory = config.delta_trajectory
        lambda_schedule = T_schedule(T, config.alpha, config.w) 
        mask_tdict = get_mask_tdict(x_cond["state"], config.partition, var_idx, level_idx, mask_2d)
        guidance_type = config.guidance_mode
    else:
        delta_trajectory=None
        lambda_schedule=None
        mask_tdict = None
        guidance_type = None

    guidance_kwargs = build_guidance_kwargs(config) if guidance_flag else {}

    # run
    for m in range(config.M):
        for n in range(config.N):
            print(f"m={m} - n={n}")
            # runs both if guidance
            x_hat_ung, _ = flow_model.sample(
                guidance_flag=False,
                guidance_type=None,
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
                    guidance_type=guidance_type,
                    x_cond=x_cond,
                    guidance_kwargs=guidance_kwargs,
                    delta_t=delta_trajectory[n],
                    mask=mask_tdict,
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
                    if trace_type not in TRACE_CONTAINERS:
                        continue  # diagnostics (flowgrad/lambda_star/control_star) -> no container
                    save_trace = torch.stack(trace, dim=0)
                    save_trace = tdict_to_xr(
                        create_slice_zarr_container(m, n, t_dim=True, T=T, sweep_params=sweep_params),
                        save_trace,
                        t_dim=True
                    )
                    append_to_zarr(rollout_dir, trace_type, save_trace)

                # FlowGrad learns its schedule/controls -> persist the small diagnostics
                # (no weather-shaped container) to a JSON sidecar for the notebook.
                if guidance_type in ("FLOWGRAD", "FLOWGRAD_FREE"):
                    fg = sampling_trace.get("flowgrad", {})
                    control_norm = fg.get("control_norm")
                    append_diagnostics(rollout_dir, {
                        "m": m,
                        "n": n,
                        "guidance_mode": guidance_type,
                        "sweep": {k: (v.item() if hasattr(v, "item") else v)
                                  for k, v in sweep_params.items()},
                        "opt_loss": [float(x) for x in fg.get("loss", [])],
                        "opt_target_loss": [float(x) for x in fg.get("target_loss", [])],
                        "opt_reg_loss": [float(x) for x in fg.get("reg_loss", [])],
                        "lambda_star": ([float(x) for x in sampling_trace["lambda_star"]]
                                        if "lambda_star" in sampling_trace else None),
                        "control_norm": ([float(x) for x in control_norm[-1]]
                                         if control_norm else None),
                    })

            else:
                save_state = flow_model.denormalize(x_hat_ung).cpu()
                save_state = tdict_to_xr(
                    create_slice_zarr_container(m, n, t_dim=False, sweep_params={}),
                    save_state,
                )
                append_to_zarr(rollout_dir, "ung", save_state)

            # after the last iteration no need to set this again
            if n < config.N-1:
                x_cond = advance_x_cond(x_cond, x_hat_curr)