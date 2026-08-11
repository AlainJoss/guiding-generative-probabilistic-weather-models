from pathlib import Path

import xarray as xr
import torch

from src.utils import get_x_cond, find_era5_input
from src.utils import (
    batchify_and_move,
    get_var_idx, get_level_idx,
)
from src.rollout_config import RolloutConfig, GUIDANCE_METHOD_HYPERS
from src.mask import get_mask_tdict, get_mask_2d, get_masked_mean

from src.utils import create_slice_zarr_container, tdict_to_xr, append_to_zarr, advance_x_cond
from src.utils import get_gt_rollout, gt_state_to_tdict, append_w_star, append_guidance_schedule

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict

def build_guidance_kwargs(guidance_type, sweep_params):
    # Straight pass-through: hyper names already equal the guidance-fn kwarg names.
    return {hk: sweep_params[hk] for hk in GUIDANCE_METHOD_HYPERS[guidance_type]}


def rollout(
    rollout_dir: Path,
    guidance_flag: bool,
    config: RolloutConfig,
    flow_model: GuidedFlow,
    T: int = 25,
    sweep_params: dict[str, any] | None = None  # note: this are the actual params, value is not a list!
):  
    # for testing purposes
    flow_model.T=T
    # create objects from config. Prefer this experiment's downloaded era5_input.nc
    # (in the per-start subdir or the outer exp dir); else the global ERA5 store.
    input_path = find_era5_input(rollout_dir)
    x_cond_init = get_x_cond(config.START_TS, config.N, input_path=input_path)
    x_cond_init = batchify_and_move(x_cond_init, flow_model.device)

    if guidance_flag:
        var_idx = get_var_idx(config.PARTITION, config.VAR)
        level_idx = get_level_idx(config.PARTITION, config.LEVEL)
        mask_2d = get_mask_2d(
            config.MASK_MODE, config.MASK_CORNERS,
            sigma_div=config.sigma_div if config.sigma_div is not None else 2.0,
            mask_shift=config.mask_shift or "none",
        )

        # GUIDANCE_DELTA holds the target percentage profile p_n; the loss delta is
        # derived online per step so the target is ABSOLUTE (see the m/n loop).
        delta_trajectory = config.GUIDANCE_DELTA
        mask_tdict = get_mask_tdict(x_cond_init["state"], config.PARTITION, var_idx, level_idx, mask_2d)
        guidance_type = config.GUIDANCE_MODE
        # GT reference: ground-truth field per step (member-independent). gt rollout holds
        # N+1 states (initial + N forecasts); guidance step n's valid time is index n+1.
        gt_ds = get_gt_rollout(config.N + 1, config.START_TS) if config.GUI_REF == "GT" else None
        guidance_kwargs = build_guidance_kwargs(guidance_type, sweep_params)

        # baseline masked means anchoring the target trajectory A = (1 + p_n) * base:
        # GT -> the ground-truth rollout (member-independent, valid time n+1);
        # UNG -> the unguided baseline rollout of this experiment (per member)
        if config.GUI_REF == "GT":
            _base_da = gt_ds[config.VAR]
            if config.PARTITION == "level":
                _base_da = _base_da.sel(level=config.LEVEL)
            base_mm = get_masked_mean(_base_da.to_numpy(), mask_2d)  # (N+1,)
        else:
            _ung_path = rollout_dir / "ung.zarr"
            assert _ung_path.exists(), (
                "guided rollout needs the unguided baseline; run --rollout_type ung first"
            )
            _base_da = xr.open_zarr(_ung_path)[config.VAR]
            if "t" in _base_da.dims:  # ung stores the full flow trajectory now
                _base_da = _base_da.isel(t=-1)
            if config.PARTITION == "level":
                _base_da = _base_da.sel(level=config.LEVEL)
            base_mm = get_masked_mean(_base_da.to_numpy(), mask_2d)  # (M, N)
    else:
        delta_trajectory=None
        mask_tdict = None
        guidance_type = None
        gt_ds = None

    # run
    for m in range(config.M):
        x_cond = x_cond_init  # reset to the initial condition at the start of every member
        for n in range(config.N):
            print(f"m={m} - n={n}", flush=True)
            # runs both if guidance
            x_hat_ung, ung_trace = flow_model.sample(
                guidance_flag=False,
                guidance_type=None,
                x_cond=x_cond,
                delta_n=None,
                mask=None,
                x_ref=None,
                seed= m + 1000 * n,  # + batch_nb * 10**6
            )
            x_hat_curr = x_hat_ung

            if guidance_flag:
                p_n = delta_trajectory[n]
                if float(p_n) == 0.0:
                    # p=0 -> no guidance: don't produce a guided state, just copy the
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
                    # absolute target A = (1 + p_n) * baseline masked mean; the ONLINE
                    # delta makes the loss target (1 + delta_n) * S(x_ref) == A exactly,
                    # wherever the online reference drifted. s_ref ~ 0 would only
                    # inflate delta_n, never the realized target.
                    A_target = (1.0 + float(p_n)) * float(
                        base_mm[n + 1] if config.GUI_REF == "GT" else base_mm[m, n]
                    )
                    s_ref = float(
                        (x_ref[config.PARTITION][0, var_idx, level_idx]
                         .detach().cpu().numpy().astype("float64") * mask_2d).sum()
                    )
                    delta_n = A_target / s_ref - 1.0
                    x_hat_gui, sampling_trace = flow_model.sample(
                        guidance_flag=True,
                        guidance_type=guidance_type,
                        x_cond=x_cond,
                        guidance_kwargs=guidance_kwargs,
                        delta_n=delta_n,
                        mask=mask_tdict,
                        x_ref=x_ref,
                        seed = m + 1000 * n,  # + batch_nb * 10**6
                    )
                x_hat_curr = x_hat_gui

            if guidance_flag:
                # FGWNOLR attaches the optimized scalar w* to the trace; pop it before
                # the stacking loop (it is not a per-t tensor) and persist it as a sidecar.
                w_star = sampling_trace.pop("w_star", None)
                if w_star is not None:
                    append_w_star(rollout_dir, sweep_params, m, n, w_star)

                # applied weight schedule {w_t, a_t} (lambda_t = w_t * a_t) -> sidecar
                sched = sampling_trace.pop("guidance_schedule", None)
                if sched is not None:
                    append_guidance_schedule(rollout_dir, guidance_type, sweep_params, m, n, **sched)

                save_state = flow_model.denormalize(x_hat_gui).cpu()
                save_state = tdict_to_xr(
                    create_slice_zarr_container(m, n, t_dim=False, sweep_params=sweep_params),
                    save_state,
                    t_dim=False
                )
                append_to_zarr(rollout_dir, "gui", save_state)

                # deterministic core of this step. Same x_cond for the online unguided
                # call, so its trace supplies it when delta=0 skipped the guided pass.
                det_state = sampling_trace.pop("det_pred", None)
                if det_state is None:
                    det_state = ung_trace.get("det_pred")
                if det_state is not None:
                    save_state = flow_model.denormalize(det_state).cpu()
                    save_state = tdict_to_xr(
                        create_slice_zarr_container(m, n, t_dim=False, sweep_params=sweep_params),
                        save_state,
                        t_dim=False
                    )
                    append_to_zarr(rollout_dir, "gui_det", save_state)

                # gui_ung is the unguided clean-prediction trajectory over flow steps t
                # (last slice == final unguided state). Mirrors clean_preds for the guided
                # pass; clean_prediction already denormalizes, so this is in physical units.
                ung_traj = torch.stack(ung_trace["clean_preds"], dim=0)
                save_state = tdict_to_xr(
                    create_slice_zarr_container(m, n, t_dim=True, T=T, sweep_params=sweep_params),
                    ung_traj,
                    t_dim=True
                )
                append_to_zarr(rollout_dir, "gui_ung", save_state)

                for trace_type, trace in sampling_trace.items():
                    save_trace = torch.stack(trace, dim=0)
                    save_trace = tdict_to_xr(
                        create_slice_zarr_container(m, n, t_dim=True, T=T, sweep_params=sweep_params),
                        save_trace,
                        t_dim=True
                    )
                    append_to_zarr(rollout_dir, trace_type, save_trace)

            else:
                # ung stores the full clean-prediction trajectory over flow steps t
                # (physical units; last slice == final unguided state, like gui_ung)
                ung_traj = torch.stack(ung_trace["clean_preds"], dim=0)
                save_state = tdict_to_xr(
                    create_slice_zarr_container(m, n, t_dim=True, T=T, sweep_params={}),
                    ung_traj,
                    t_dim=True
                )
                append_to_zarr(rollout_dir, "ung", save_state)

            # after the last iteration no need to set this again
            if n < config.N-1:
                x_cond = advance_x_cond(x_cond, x_hat_curr)