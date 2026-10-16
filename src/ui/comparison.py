"""Rollout loading and trajectory reconstruction for flow-time analysis.

Historical guidance-schedule records provide sweep selections for existing runs.
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
from src.utils import get_level_idx, get_var_idx, resolve_store_name
from src.utils import experiment_root, SUBDIR_TS_FMT


def load_rollout(rollout_id):
    """(dir, config, sweep_values, schedule_records, mask) for one rollout."""
    rollout_dir = ROLLOUTS / rollout_id
    # config + sweep are shared at the exp root (a multi-start member resolves UP); the
    # trace sidecars (guidance_schedule.json, ...) stay per-date in the rollout dir.
    root = experiment_root(rollout_dir)
    config = json.load(open(root / "config.json"))
    sweep_values = json.load(open(root / "sweep_params.json"))
    if root != rollout_dir:
        # shared config has START_TS=None -> recover this member's start from the subdir name
        # (ISO string, so consumers' datetime.fromisoformat keeps working).
        config["START_TS"] = datetime.strptime(rollout_dir.name, SUBDIR_TS_FMT).isoformat()
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
    """One variable of a rollout store, lazily (res/gui_res aliased to whichever exists)."""
    return xr.open_zarr(rollout_dir / f"{resolve_store_name(rollout_dir, store)}.zarr")[var]


def select_point(da, sel):
    """Select a sweep point, ignoring coords the store does not carry."""
    return da.sel({k: v for k, v in sel.items() if k in da.dims})


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


def clean_pred_trajectory_primitive(rollout_dir, sel, m, n, var, c, level=None):
    """Guided clean-pred trajectory (T, lat, lon) from stored primitives, no lambda records:
    z_T = res[-1] (res holds z_0..z_T, length T+1); x_hat_t = gui + ((res_t + vfs_t) - z_T) * c
    over the T Euler steps (res[:-1] aligned with the length-T vfs). The final flow step
    reproduces `gui` exactly (endpoint identity)."""
    def _load(store):
        da = select_point(open_store(rollout_dir, store, var), sel)
        da = da.sel(level=level) if "level" in da.dims and level is not None else da
        return np.asarray(da.isel(m=m, n=n), dtype=float)

    res, vfs = _load("gui_res"), _load("vfs")
    gui = _load("gui")
    z_T = res[-1]
    return gui[None] + ((res[:-1] + vfs) - z_T[None]) * c


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

    res, vfs = _load("gui_res"), _load("vfs")
    s, h = flow_grids(vfs.shape[0])                               # flow grids over the T Euler steps
    z_next = res[1:]                                              # z_{t+1} = z_1..z_T (length T)
    gui_vfs = (z_next - res[:-1]) * (s / h)[:, None, None]        # realized guided step velocity
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
    res = select_point(open_store(rollout_dir, "gui_res", var), sel)
    res = res.sel(level=level) if "level" in res.dims and level is not None else res
    z_mm = (np.asarray(res.isel(m=m, n=n), dtype=float) * mnp).sum(axis=(-2, -1))   # M(z_t)
    u_mm = (vel["vfs"] * mnp).sum(axis=(-2, -1))                                     # M(s_t u_t)
    gu_mm = (vel["gui_vfs"] * mnp).sum(axis=(-2, -1))                                # M(gui_vfs_t)
    hs = vel["h"] / vel["s"]
    gd = select_point(open_store(rollout_dir, "gui_det", var), sel)
    gd = gd.sel(level=level) if "level" in gd.dims and level is not None else gd
    det_mm = float((np.asarray(gd.isel(m=m, n=n), dtype=float) * mnp).sum())         # M(x_det)
    T = vel["vfs"].shape[0]                          # flow-step count (T); z_mm/states are T+1
    states = det_mm + c * z_mm - float(A)            # M(x_t) - A, t = 0..T (ends at x_T, from res[-1])
    land_ung = states[:T] + c * hs * u_mm            # where the raw flow step would land (T)
    return states, land_ung


def channel(da, config):
    """Reduce a store variable to (..., lat, lon) at the config's guided channel."""
    return da.sel(level=config["LEVEL"]) if "level" in da.dims else da
