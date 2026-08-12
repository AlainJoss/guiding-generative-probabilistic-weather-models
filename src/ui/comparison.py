"""Shared helpers for the cross-method comparison notebooks.

Each comparison notebook loads one rollout and compares its runs across the
sweep points (methods x hyperparameters) that actually ran, as enumerated from
the guidance_schedule.json sidecar.
"""

import json
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from geoarches.paths import STATS_PATH
from src.mask import get_mask_2d
from src.paths import ROLLOUTS
from src.rollout_config import GUIDANCE_METHOD_HYPERS
from src.utils import get_gt_rollout, get_level_idx, get_var_idx


def load_rollout(rollout_id):
    """(dir, config, sweep_values, schedule_records, mask) for one rollout."""
    rollout_dir = ROLLOUTS / rollout_id
    config = json.load(open(rollout_dir / "config.json"))
    sweep_values = json.load(open(rollout_dir / "sweep_params.json"))
    try:
        records = json.load(open(rollout_dir / "guidance_schedule.json"))
    except FileNotFoundError:
        records = []
    mask = get_mask_2d(sweep_values["MASK_MODE"][0], config["MASK_CORNERS"])
    return rollout_dir, config, sweep_values, records, mask


def sweep_points(sweep_values, records):
    """Label -> full coord selection, for every sweep combo that actually ran.

    Labels mention only the hypers that vary in this sweep and belong to the
    point's guidance mode (plus the delta index when several deltas ran).
    """
    varying = {k for k, v in sweep_values.items() if isinstance(v, list) and len(v) > 1}
    points = {}
    for r in records:
        sel = dict(r["sweep"])
        mode = sel["GUIDANCE_MODE"]
        bits = [mode] + [
            f"{k}={sel[k]}" for k in GUIDANCE_METHOD_HYPERS.get(mode, []) if k in varying
        ]
        if "GUIDANCE_DELTA" in varying:
            bits.append(f"δ#{sel['GUIDANCE_DELTA']}")
        points.setdefault(" ".join(bits), sel)
    return points


def open_store(rollout_dir, store, var):
    """One variable of a rollout store, lazily."""
    return xr.open_zarr(rollout_dir / f"{store}.zarr")[var]


def select_point(da, sel):
    """Select a sweep point, ignoring coords the store does not carry."""
    return da.sel({k: v for k, v in sel.items() if k in da.dims})


def masked_mean(field, mask):
    """Mask-weighted mean over the trailing (lat, lon) dims."""
    m = np.asarray(mask, dtype=float)
    return (np.asarray(field, dtype=float) * m).sum(axis=(-2, -1)) / m.sum()


def applied_lambda(records, sel, m, n):
    """lambda_t = w_t * a_t recorded for one sweep point at (m, n), or None."""
    for r in records:
        if r["m"] == m and r["n"] == n and all(
            r["sweep"].get(k) == v for k, v in sel.items() if k in r["sweep"]
        ):
            return np.asarray(r["w_t"], float) * np.asarray(r["a_t"], float)
    return None


def flow_grids(T):
    """Noise levels s_t and Euler step sizes h_t of the sampling grid."""
    s = np.linspace(1000, 1, T) / 1000
    h = np.empty_like(s)
    h[:-1] = s[:-1] - s[1:]
    h[-1] = s[-1]
    return s, h


def residual_scaler(partition, var, level=None):
    """Residual denorm scaler c of one channel (mirrors GuidedFlow)."""
    scaler = torch.load(STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False)
    if partition == "level":
        c = float(scaler["level"][get_var_idx("level", var), get_level_idx("level", level)].squeeze())
        return c * 3.0 if var == "vertical_velocity" else c
    return float(scaler["surface"][get_var_idx("surface", var)].squeeze())


def clean_pred_trajectory(rollout_dir, records, sel, m, n, var, c, level=None):
    """Guided clean-pred trajectory (T, lat, lon) reconstructed from raw traces:
    gui_vfs = vfs - lam*grads*s;  z_T = res[-1] + gui_vfs[-1]*(h/s)[-1];
    x_hat_t = gui_final + ((res_t + vfs_t) - z_T) * c.  None without a lambda record."""
    lam = applied_lambda(records, sel, m, n)
    if lam is None:
        return None
    def _load(store):
        da = select_point(open_store(rollout_dir, store, var), sel)
        da = da.sel(level=level) if "level" in da.dims and level is not None else da
        return np.asarray(da.isel(m=m, n=n), dtype=float)

    res, vfs, grads = _load("res"), _load("vfs"), _load("grads")
    gui = _load("gui")
    s, h = flow_grids(res.shape[0])
    z_final = res[-1] + (vfs[-1] - lam[-1] * grads[-1] * s[-1]) * (h[-1] / s[-1])
    return gui[None] + ((res + vfs) - z_final[None]) * c


def clean_pred_trajectory_primitive(rollout_dir, sel, m, n, var, c, level=None):
    """Guided clean-pred trajectory (T, lat, lon) from stored primitives, with NO lambda
    records -- faithful under the new w*a*c/g_norm guidance convention (the lambda-based
    `clean_pred_trajectory` above mis-scales it). Mirrors reductions.py:161-175:
    z_T = (gui - gui_det)/c;  x_hat_t = gui + ((res_t + vfs_t) - z_T) * c. The final flow
    step reproduces `gui` exactly (endpoint identity)."""
    def _load(store):
        da = select_point(open_store(rollout_dir, store, var), sel)
        da = da.sel(level=level) if "level" in da.dims and level is not None else da
        return np.asarray(da.isel(m=m, n=n), dtype=float)

    res, vfs = _load("res"), _load("vfs")
    gui, gui_det = _load("gui"), _load("gui_det")
    z_T = (gui - gui_det) / c
    return gui[None] + ((res + vfs) - z_T[None]) * c


def guided_velocity_primitive(rollout_dir, sel, m, n, var, c, level=None):
    """Per-flow-step velocities for one guided channel, reconstructed from stored
    primitives (no reductions store, no lambda records). Mirrors reductions.py:161-171:
    z_T=(gui-gui_det)/c; z_{t+1}=[res_{t+1}..,z_T]; gui_vfs=(z_{t+1}-res)*s/h;
    gui_vec=(vfs-gui_vfs)/s (the applied guidance kick; ==0 where guidance was off).
    Returns dict of (T,lat,lon) arrays vfs/gui_vfs/gui_vec and (T,) grids s/h."""
    def _load(store):
        da = select_point(open_store(rollout_dir, store, var), sel)
        da = da.sel(level=level) if "level" in da.dims and level is not None else da
        return np.asarray(da.isel(m=m, n=n), dtype=float)

    res, vfs = _load("res"), _load("vfs")
    gui, gui_det = _load("gui"), _load("gui_det")
    s, h = flow_grids(res.shape[0])
    z_T = (gui - gui_det) / c
    z_next = np.concatenate([res[1:], z_T[None]], axis=0)          # z_{t+1}
    gui_vfs = (z_next - res) * (s / h)[:, None, None]
    gui_vec = (vfs - gui_vfs) / s[:, None, None]
    return {"vfs": vfs, "gui_vfs": gui_vfs, "gui_vec": gui_vec, "s": s, "h": h}


def convergence_state_line(rollout_dir, sel, m, n, var, c, mask, A, level=None):
    """Guidance-convergence line (mirrors guidance.py:4506-4547): the realized guided STATE
    masked-mean minus the target ``A`` over flow steps ``t``, anchored at the pure-noise state,
    with the final Euler landing appended.

        state_t   = M(x_det + sigma_r z_t) - A       (t = 0..T-1; state_0 = noise)
        state_T   = state_{T-1} + c * (h/s)_{T-1} * M(gui_vfs_{T-1})   (final landing)
        land_ung_t= state_t + c * (h/s)_t * M(vfs_t)  (where the RAW flow step would land)

    Returns ``(states (T+1,), land_ung (T,))``. The purple-vs-yellow gap at t+1 is the
    guidance contribution; the state line's descent to 0 is the convergence to target."""
    mnp = np.asarray(mask, dtype=float)
    vel = guided_velocity_primitive(rollout_dir, sel, m, n, var, c, level=level)
    res = select_point(open_store(rollout_dir, "res", var), sel)
    res = res.sel(level=level) if "level" in res.dims and level is not None else res
    z_mm = (np.asarray(res.isel(m=m, n=n), dtype=float) * mnp).sum(axis=(-2, -1))   # M(z_t)
    u_mm = (vel["vfs"] * mnp).sum(axis=(-2, -1))                                     # M(s_t u_t)
    gu_mm = (vel["gui_vfs"] * mnp).sum(axis=(-2, -1))                                # M(gui_vfs_t)
    hs = vel["h"] / vel["s"]
    gd = select_point(open_store(rollout_dir, "gui_det", var), sel)
    gd = gd.sel(level=level) if "level" in gd.dims and level is not None else gd
    det_mm = float((np.asarray(gd.isel(m=m, n=n), dtype=float) * mnp).sum())         # M(x_det)
    T = z_mm.shape[0]
    states = np.empty(T + 1)
    states[:T] = det_mm + c * z_mm - float(A)
    states[T] = states[T - 1] + c * hs[-1] * gu_mm[-1]
    land_ung = states[:T] + c * hs * u_mm
    return states, land_ung


def clean_pred_branches_primitive(rollout_dir, sel, m, n, var, c, level=None):
    """(gui_ung_over_t, gui) clean-pred branch pair from stored primitives, NO lambda records
    -- convention-robust (works under the new w*a*c/g_norm guidance convention, unlike the
    lambda-based `clean_pred_branches`). base = the raw-velocity estimate
    (`clean_pred_trajectory_primitive`); kick = base minus the applied guidance
    `c * s_t * gui_vec_t`. Per step, base_t -> kick_t is the GUIDANCE step and
    kick_t -> base_{t+1} the PUSHBACK (pull-back) step -- the intensity_comparison pingpong."""
    base = clean_pred_trajectory_primitive(rollout_dir, sel, m, n, var, c, level=level)
    vel = guided_velocity_primitive(rollout_dir, sel, m, n, var, c, level=level)
    kick = base - float(c) * vel["s"][:, None, None] * vel["gui_vec"]
    return base, kick


def clean_pred_branches(rollout_dir, records, sel, m, n, var, c, level=None):
    """(gui_ung_over_t, gui) clean-pred trajectory pair, both (T, lat, lon).

    gui_ung_over_t: the raw-velocity estimate res_t + vfs_t (no step-t
    guidance kick); gui: the same estimate after subtracting the applied
    kick lam_t * grads_t * s_t. Their per-step difference is exactly the
    guidance contribution. None without a lambda record."""
    lam = applied_lambda(records, sel, m, n)
    if lam is None:
        return None
    def _load(store):
        da = select_point(open_store(rollout_dir, store, var), sel)
        da = da.sel(level=level) if "level" in da.dims and level is not None else da
        return np.asarray(da.isel(m=m, n=n), dtype=float)

    res, vfs, grads = _load("res"), _load("vfs"), _load("grads")
    gui = _load("gui")
    s, h = flow_grids(res.shape[0])
    z_final = res[-1] + (vfs[-1] - lam[-1] * grads[-1] * s[-1]) * (h[-1] / s[-1])
    base = gui[None] + ((res + vfs) - z_final[None]) * c
    kick = base - (lam[:, None, None] * grads * s[:, None, None]) * c
    return base, kick


def guidance_vector(rollout_dir, records, sel, m, n, var, level=None):
    """Applied guidance vector lambda_t * grads_t as (T, lat, lon), or None."""
    lam = applied_lambda(records, sel, m, n)
    if lam is None:
        return None
    grads = select_point(open_store(rollout_dir, "grads", var), sel)
    grads = grads.sel(level=level) if "level" in grads.dims and level is not None else grads
    return lam[:, None, None] * np.asarray(grads.isel(m=m, n=n), dtype=float)


def gt_states(config):
    """GT dataset for the rollout window: N+1 daily states from START_TS."""
    start = datetime.fromisoformat(config["START_TS"])
    return get_gt_rollout(config["N"] + 1, start)


def default_selection(points):
    """One label per guidance mode (the first seen) — the multiselect default."""
    out, seen = [], set()
    for label in points:
        mode = label.split()[0]
        if mode not in seen:
            seen.add(mode)
            out.append(label)
    return out


def channel(da, config):
    """Reduce a store variable to (..., lat, lon) at the config's guided channel."""
    return da.sel(level=config["LEVEL"]) if "level" in da.dims else da
