from pathlib import Path

import xarray as xr
import torch

from src.utils import get_x_cond
from src.utils import (
    batchify_and_move,
    get_var_idx, get_level_idx,
)
from src.schedules import T_schedule
from src.rollout_config import RolloutConfig, MODE_HYPERS
from src.mask import get_mask_tdict, get_mask_2d

from src.utils import create_slice_zarr_container, tdict_to_xr, append_to_zarr, advance_x_cond
from src.utils import append_diagnostics, get_gt_rollout, gt_state_to_tdict

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict


# t_dim=True trace keys that have a matching zarr container (see get_container_args
# in run.py). Other sampling_trace entries are method-specific diagnostics with no
# container and non-stackable shapes -- FlowGrad's "flowgrad" (per-iteration loss
# dict), "lambda_star" (T,), and "control_star" (list of TensorDicts) -- and are
# skipped from the zarr save so a FLOWGRAD/FLOWGRAD_FREE sweep runs end-to-end.
TRACE_CONTAINERS = ("clean_preds", "grads", "vfs", "gui_vfs")


# config field (UPPER) -> guided_diffusion kwarg name
HYPER_TO_KWARG = {
    "M_MC": "n_mc",
    "SIGMA_MC": "r_t",
    "OPTIMIZE_K": "optimize_k",
    "OPTIMIZE_LR": "optimize_lr",
    "RENOISE_S": "renoise_s",
    "SHIFT_INIT": "shift_init",
    "CONTROL_GAMMA": "control_gamma",
    "LAMBDA_INIT": "lambda_init",
    "N_WINDOWS": "n_windows",
}


def build_guidance_kwargs(config: RolloutConfig) -> dict[str, any]:
    """Build the guidance kwargs for config.GUIDANCE_MODE: the shared loss controls
    (LOSS_TYPE derived from GUI_REF, REG_TYPE, BETA_REG) plus this mode's specific
    hypers from MODE_HYPERS. Drop None so GuidedFlow method defaults still apply."""
    loss_type = "STATE" if config.GUI_REF == "GT" else "MASKED_AVERAGE"
    kwargs = {
        "loss_type": loss_type,
        "reg_type": config.REG_TYPE,
        "beta_reg": config.BETA_REG,
    }
    for axis in MODE_HYPERS[config.GUIDANCE_MODE]:
        kwargs[HYPER_TO_KWARG[axis]] = getattr(config, axis)

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
    # the initial condition is shared by every member; advance_x_cond returns fresh
    # dicts and never mutates it, so we keep it and reset x_cond to it per member.
    x_cond_init = get_x_cond(config.START_TS, config.N)
    x_cond_init = batchify_and_move(x_cond_init, flow_model.device)

    if guidance_flag:
        var_idx = get_var_idx(config.PARTITION, config.VAR)
        level_idx = get_level_idx(config.PARTITION, config.LEVEL)
        mask_2d = get_mask_2d(config.MASK_MODE, config.MASK_CORNERS)

        delta_trajectory = config.GUIDANCE_DELTA
        lambda_schedule = T_schedule(T, config.ALPHA, config.W)
        mask_tdict = get_mask_tdict(x_cond_init["state"], config.PARTITION, var_idx, level_idx, mask_2d)
        guidance_type = config.GUIDANCE_MODE
        # GT reference: ground-truth field per step (member-independent). gt rollout holds
        # N+1 states (initial + N forecasts); guidance step n's valid time is index n+1.
        gt_ds = get_gt_rollout(config.N + 1, config.START_TS) if config.GUI_REF == "GT" else None
    else:
        delta_trajectory=None
        lambda_schedule=None
        mask_tdict = None
        guidance_type = None
        gt_ds = None

    guidance_kwargs = build_guidance_kwargs(config) if guidance_flag else {}

    # run
    for m in range(config.M):
        x_cond = x_cond_init  # reset to the initial condition at the start of every member
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
                seed= m + 1000 * n,  # + batch_nb * 10**6
            )
            x_hat_curr = x_hat_ung

            if guidance_flag:
                delta_n = delta_trajectory[n]
                if float(delta_n) == 0.0:
                    # δ=0 -> no guidance: don't produce a guided state, just copy the
                    # unguided online sample into gui (no second sampling pass, no traces)
                    x_hat_gui = x_hat_ung
                    sampling_trace = {}
                else:
                    x_ung_field = flow_model.denormalize(x_hat_ung.detach().clone())
                    # data-loss reference: GT field (step n) or the unguided member itself
                    x_ref = (
                        gt_state_to_tdict(gt_ds, n + 1, flow_model.device)
                        if config.GUI_REF == "GT" else x_ung_field
                    )
                    x_hat_gui, sampling_trace = flow_model.sample(
                        guidance_flag=True,
                        guidance_type=guidance_type,
                        x_cond=x_cond,
                        guidance_kwargs=guidance_kwargs,
                        delta_t=delta_n,
                        mask=mask_tdict,
                        x_hat_ung=x_ung_field,
                        x_ref=x_ref,
                        lambda_schedule=lambda_schedule,
                        seed = m + 1000 * n,  # + batch_nb * 10**6
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
                if guidance_type in ("FG", "FGF") and sampling_trace:
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