"""Precomputed spatial reductions for the guidance-analysis notebook.

The cross-var / trajectory / convergence charts read a dict `red` of ~29
spatially-reduced cubes. Recomputing them from the 38 GB/mode raw traces on every
slider move is painfully slow, so we compute them ONCE for every sweep point and
persist a single `reductions.zarr` in the rollout folder; the notebook then just
reads it and selects the current sweep point in memory.

This module is the single source of truth for the reduction math: the notebook's
live path calls `compute_reductions_for_sweep` too, so the stored and on-the-fly
values can't drift. The stored convention is the notebook default
(`aggregate_spatially_dropdown` = None): the gated L2/mean terms are full-domain and
the `_wmn`/`_wnmn` family is mask-weighted with the experiment mask. The notebook
keeps a live fallback for the mask/!mask toggle and for legacy rollouts.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr
import dask
import torch

from geoarches.paths import STATS_PATH
from src.dimensions import VARIABLES_DICT
from src.normalization import XarrayNormalizer
from src.mask import get_mask_2d
from src.rollout_config import MASK_MODES
from src.utils import (
    get_rollout,
    get_rollout_dir,
    get_sweep_dict,
    get_guidance_schedule,
    get_config,
    get_gt_rollout,
    sweep_coord_label,
    append_to_zarr,
    ensure_rollout_dir,
)
from src.run import iter_sweeps

REDUCTIONS_NAME = "reductions"
_FLAT_SEP = "__"


def _res_scale_map(level_coord):
    """sigma_r denorm scaler per var (level vars: per-level vector), as in GuidedFlow."""
    _rsc = torch.load(STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False)
    out = {}
    for _vi, _v in enumerate(VARIABLES_DICT["surface"]):
        out[_v] = float(_rsc["surface"][_vi].squeeze())
    _lev = _rsc["level"].squeeze(-1).squeeze(-1).numpy()
    for _vi, _v in enumerate(VARIABLES_DICT["level"]):
        _arr = _lev[_vi] * (3.0 if _v == "vertical_velocity" else 1.0)
        out[_v] = xr.DataArray(_arr, dims=("level",), coords={"level": level_coord})
    return out


def _flow_grids(T):
    s = np.linspace(1000, 1, T) / 1000
    h = np.empty_like(s)
    h[:-1] = s[:-1] - s[1:]
    h[-1] = s[-1]
    return s, h


@lru_cache(maxsize=4)
def _rollout_ctx(rollout_id):
    """Per-rollout constants (lazy zarr handles + GT + normalizer + scaler), hoisted
    so a whole-store build doesn't reload the stat files / GT for every sweep point."""
    config = get_config(rollout_id)
    handles = {
        "grads": get_rollout("grads", rollout_id),
        "vfs": get_rollout("vfs", rollout_id),
        "gui": get_rollout("gui", rollout_id),
        "gui_ung": get_rollout("gui_ung", rollout_id),
    }
    try:
        handles["res"] = get_rollout("res", rollout_id)
    except FileNotFoundError:
        handles["res"] = None
    gui = handles["gui"]
    res_scale_map = _res_scale_map(handles["grads"].level)
    gt_rollout = get_gt_rollout(config.N + 1, config.START_TS)
    gt_n_xr = (
        gt_rollout.isel(time=slice(1, None))
        .rename({"time": "n"})
        .assign_coords(n=gui.n, latitude=gui.latitude,
                       longitude=gui.longitude, level=gui.level)
    )
    return dict(config=config, handles=handles, res_scale_map=res_scale_map,
                xnorm=XarrayNormalizer(), gt_n_xr=gt_n_xr)


def compute_reductions_for_sweep(rollout_id, sweep_point, *, spatial="full"):
    """The `red` dict of reduced cubes for ONE (coord-labeled) sweep point.

    Ports the notebook reconstruction + reduction verbatim. `spatial="full"` is the
    notebook default (dropdown None): the `_msk`-gated L2/mean terms are full-domain;
    `_wmn`/`_wnmn` are always mask-weighted with the experiment mask. Returns a dict
    of computed (dask-realized) xr.Dataset / DataArray, keyed exactly like the
    notebook's `red`.
    """
    _ctx = _rollout_ctx(rollout_id)
    config = _ctx["config"]
    res_scale_map = _ctx["res_scale_map"]
    xnorm = _ctx["xnorm"]
    gt_n_xr = _ctx["gt_n_xr"]
    _h = _ctx["handles"]

    # --- raw cubes at this sweep point (keep m, n, t, level, lat, lon) ---
    grads_xr = _h["grads"].sel(sweep_point)
    vfs_xr = _h["vfs"].sel(sweep_point)
    gui_ung_xr = _h["gui_ung"].sel(sweep_point)
    gui_ung_final_xr = gui_ung_xr.isel(t=-1) if "t" in gui_ung_xr.dims else gui_ung_xr
    guided_xr = _h["gui"].sel(sweep_point)
    res_xr = _h["res"].sel(sweep_point) if _h["res"] is not None else None

    # --- experiment mask at this sweep point (mask-weighted terms only) ---
    _mm = sweep_point.get("MASK_MODE")
    if _mm not in MASK_MODES:
        _mm = "ELLIPTICAL"
    _sigma_div = float(sweep_point.get("sigma_div", 2.0) or 2.0)
    mask = get_mask_2d(_mm, config.MASK_CORNERS, sigma_div=_sigma_div)

    # --- reconstruction (new-format identities) ---
    if res_xr is None:
        clean_preds_xr = get_rollout("clean_preds", rollout_id).sel(sweep_point)
        gui_vfs_xr = get_rollout("gui_vfs", rollout_id).sel(sweep_point)
        gui_vec_xr = None
    else:
        _recs = get_guidance_schedule(rollout_id, dict(sweep_point),
                                      method=sweep_point.get("GUIDANCE_MODE"))
        _M, _N, _T = grads_xr.sizes["m"], grads_xr.sizes["n"], grads_xr.sizes["t"]
        _lam = np.full((_M, _N, _T), np.nan)
        for _r in _recs:
            _wv, _av = np.asarray(_r["w_t"], float), np.asarray(_r["a_t"], float)
            _cv = np.asarray(_r.get("c_t", np.ones_like(_wv)), float)
            _gv = np.asarray(_r.get("g_norm_t", np.ones_like(_wv)), float)
            if len(_wv) == _T and _r["m"] < _M and _r["n"] < _N:
                _lam[_r["m"], _r["n"], :] = _wv * _av * _cv / np.where(_gv != 0, _gv, 1.0)
        lambda_t_xr = xr.DataArray(
            _lam, dims=("m", "n", "t"),
            coords={"m": grads_xr.m, "n": grads_xr.n, "t": grads_xr.t},
        )
        _s_np, _h_np = _flow_grids(_T)
        _s_da = xr.DataArray(_s_np, dims=("t",), coords={"t": grads_xr.t})
        _h_da = xr.DataArray(_h_np, dims=("t",), coords={"t": grads_xr.t})

        # realized guided step velocity, EXACT from stored primitives (no lambda
        # records needed): z_{t+1} = z_t + (h_t/s_t) * gui_vfs_t, closing the last
        # step with z_T recovered from the final guided state. The applied guidance
        # vector follows as gui_vec = (vfs - gui_vfs) / s_t, so unguided steps
        # (e.g. spike a_t past its index) give gui_vfs == vfs identically.
        _det_zT = get_rollout("gui_det", rollout_id).sel(sweep_point)
        _z_T = xr.Dataset(
            {_v: (guided_xr[_v] - _det_zT[_v]) / res_scale_map[_v] for _v in res_xr.data_vars}
        )
        _z_next = xr.concat(
            [res_xr.isel(t=slice(1, None)).assign_coords(t=res_xr.t.values[:-1]),
             _z_T.expand_dims(t=[int(res_xr.t.values[-1])])],
            dim="t",
        )
        gui_vfs_xr = (_z_next - res_xr) * _s_da / _h_da
        gui_vec_xr = (vfs_xr - gui_vfs_xr) / _s_da
        _dev = (res_xr + vfs_xr) - _z_T
        clean_preds_xr = xr.Dataset(
            {_v: guided_xr[_v] + _dev[_v] * res_scale_map[_v] for _v in _dev.data_vars}
        ).transpose("m", "n", "t", ...)

    # --- reduction helpers ---
    # spatial: "full" (persisted default) -> gated terms are full-domain; "mask"/
    # "!mask" (live fallback only) apply the half-maximum boolean region, matching
    # the notebook's maybe_mask, to the gated L2/mean terms. The _wmn/_wnmn
    # masked-average family is ALWAYS weighted with the experiment mask: those
    # charts are in-mask by definition, so the spatial toggle must not touch them.
    _sp = ("latitude", "longitude")
    if spatial == "full":
        _msk = lambda ds: ds
    else:
        _mbool = xr.DataArray(
            np.asarray(mask) >= 0.5 * np.asarray(mask).max(),
            dims=_sp, coords={"latitude": guided_xr.latitude, "longitude": guided_xr.longitude},
        )
        _msk = (lambda ds: ds.where(_mbool)) if spatial == "mask" else (lambda ds: ds.where(~_mbool))
    _sq = lambda ds: (_msk(ds) ** 2).sum(dim=_sp)
    _mn = lambda ds: _msk(ds).mean(dim=_sp)
    _nmn = lambda ds: xnorm.normalize(_mn(ds))
    _mask_w = xr.DataArray(np.asarray(mask, dtype=float), dims=_sp,
                           coords={"latitude": guided_xr.latitude, "longitude": guided_xr.longitude})
    _wmn = lambda ds: (ds * _mask_w).sum(dim=_sp) / float(_mask_w.sum())
    _wnmn = lambda ds: xnorm.normalize(_wmn(ds))

    _dvf = (vfs_xr - gui_vfs_xr.shift(t=1)).fillna(0.0)

    # Euler-landing masked-mean diff (first-row over-t chart)
    try:
        if res_xr is None:
            raise KeyError("no res trace")
        _det_up = get_rollout("gui_det", rollout_id).sel(sweep_point)
        _s_up, _h_up = _flow_grids(res_xr.sizes["t"])
        _hs_up = xr.DataArray(_h_up / _s_up, dims=("t",), coords={"t": res_xr.t})
        _land_up = xr.Dataset({
            _v: _det_up[_v] + res_scale_map[_v] * (res_xr[_v] + _hs_up * gui_vfs_xr[_v])
            for _v in res_xr.data_vars
        })
        _land_diff_up = _wnmn(_land_up) - _wnmn(gui_ung_xr)
    except (FileNotFoundError, KeyError):
        _land_diff_up = _wnmn(clean_preds_xr) - _wnmn(gui_ung_xr)

    # physical/latent guidance kick per step (latent units)
    if gui_vec_xr is not None:
        _s_k, _h_k = _flow_grids(grads_xr.sizes["t"])
        _h_k_da = xr.DataArray(_h_k, dims=("t",), coords={"t": grads_xr.t})
        _kick_lat_entry = _sq(gui_vec_xr * _h_k_da)
    else:
        _kick_lat_entry = None

    # unguided-trajectory angular deflection (reconstructed vf from clean preds + noise)
    def _defl_dot_nrm(_traj_phys):
        _det_l2 = get_rollout("gui_det", rollout_id).sel(sweep_point)
        _rlat = xr.Dataset({_v: (_traj_phys[_v] - _det_l2[_v]) / res_scale_map[_v]
                            for _v in _traj_phys.data_vars})
        _T2 = _rlat.sizes["t"]
        _s2, _h2 = _flow_grids(_T2)
        _z2 = res_xr.isel(t=0, drop=True)
        _vf_list = []
        for _i2 in range(_T2):
            _vf_i = _rlat.isel(t=_i2, drop=True) - _z2
            _vf_list.append(_vf_i)
            _z2 = _z2 + (_h2[_i2] / _s2[_i2]) * _vf_i
        _vf_tr = xr.concat(_vf_list, dim="t").assign_coords(t=_rlat.t)
        _a2 = _msk(_vf_tr); _b2 = _msk(_vf_tr.shift(t=1))
        return (_a2 * _b2).sum(dim=_sp), ((_a2 ** 2).sum(dim=_sp) * (_b2 ** 2).sum(dim=_sp)) ** 0.5

    try:
        if res_xr is None:
            raise KeyError("no res trace")
        _gud_dot, _gud_nrm = _defl_dot_nrm(gui_ung_xr)
        _ung_traj_ds = get_rollout("ung", rollout_id)
        _ung_traj_ds = _ung_traj_ds.sel({_k: _v for _k, _v in sweep_point.items()
                                         if _k in _ung_traj_ds.dims})
        _und_dot, _und_nrm = _defl_dot_nrm(_ung_traj_ds)
        _have_ung_defl = True
    except (FileNotFoundError, KeyError):
        _have_ung_defl = False

    # guided online STATE M(x_det + sigma_r z_t) (the convergence-plot 'state' line)
    # and the guided Euler-LANDING m(x_t^gui) = M(x_det + sigma_r(z_t + h_t u_t^gui))
    # (same object as the guided-levels chart, but along the flow t), both
    # masked-normalized per var
    try:
        if res_xr is None:
            raise KeyError("no res trace")
        _det_s = get_rollout("gui_det", rollout_id).sel(sweep_point)
        _s_st, _h_st = _flow_grids(res_xr.sizes["t"])
        _hs_st = xr.DataArray(_h_st / _s_st, dims=("t",), coords={"t": res_xr.t})
        _state_ds = xr.Dataset({_v: _det_s[_v] + res_scale_map[_v] * res_xr[_v]
                                for _v in res_xr.data_vars})
        _land_ds = xr.Dataset({_v: _det_s[_v] + res_scale_map[_v] * (res_xr[_v] + _hs_st * gui_vfs_xr[_v])
                               for _v in res_xr.data_vars})
        _state_entry = _wnmn(_state_ds)
        _land_entry = _wnmn(_land_ds)
    except (FileNotFoundError, KeyError):
        _state_entry = None
        _land_entry = None

    red = {
        **({"gui_ung_defl_dot": _gud_dot, "gui_ung_defl_nrm": _gud_nrm,
            "ung_defl_dot": _und_dot, "ung_defl_nrm": _und_nrm} if _have_ung_defl else {}),
        **({"kick_lat_l2": _kick_lat_entry} if _kick_lat_entry is not None else {}),
        **({"state_wnorm": _state_entry, "land_wnorm": _land_entry} if _state_entry is not None else {}),
        "grads_l2": _sq(grads_xr),
        "vfs_l2": _sq(vfs_xr),
        "gui_vfs_l2": _sq(gui_vfs_xr),
        "dvf_l2": _sq(_dvf),
        "dvf_mean": _mn(_dvf),
        "dvf_dot": (_msk(vfs_xr) * _msk(gui_vfs_xr.shift(t=1))).sum(dim=_sp),
        "gvf_prev_sq": _sq(gui_vfs_xr.shift(t=1)),
        "gui_gui_ung_mean": _nmn(guided_xr) - _nmn(gui_ung_final_xr),
        "gui_gt_mean": _nmn(guided_xr) - _nmn(gt_n_xr),
        "clean_gt_mean": _nmn(clean_preds_xr) - _nmn(gt_n_xr),
        "gui_ung_gt_mean": _nmn(gui_ung_final_xr) - _nmn(gt_n_xr),
        "gui_ung_gt_t_mean": _nmn(gui_ung_xr) - _nmn(gt_n_xr),
        "clean_gui_ung_mean": _nmn(clean_preds_xr) - _nmn(gui_ung_xr),
        "gui_gui_ung_denorm": _wmn(guided_xr) - _wmn(gui_ung_final_xr),
        "clean_gui_ung_denorm": _wmn(clean_preds_xr) - _wmn(gui_ung_xr),
        "gui_gui_ung_wnorm": _wnmn(guided_xr) - _wnmn(gui_ung_final_xr),
        "clean_gui_ung_wnorm": _wnmn(clean_preds_xr) - _wnmn(gui_ung_xr),
        "land_gui_ung_wnorm": _land_diff_up,
        "gui_wnorm": _wnmn(guided_xr),
        "ung_final_wnorm": _wnmn(gui_ung_final_xr),
        "grads_l2_full": sum(
            (grads_xr[_v] ** 2).sum(dim=[_d for _d in grads_xr[_v].dims if _d not in ("m", "n", "t")])
            for _v in grads_xr.data_vars
        ),
    }
    red = dict(zip(red, dask.compute(*red.values())))

    # angular deflection (post-compute, last-t gapped)
    _ang_denom = (red["vfs_l2"] * red["gvf_prev_sq"]) ** 0.5
    red["dvf_angle"] = np.degrees(np.arccos((red["dvf_dot"] / _ang_denom).clip(-1.0, 1.0)))
    red["dvf_angle"] = red["dvf_angle"].where(red["dvf_angle"]["t"] != red["dvf_angle"]["t"].values[-1])
    if _have_ung_defl:
        for _pref in ("gui_ung", "ung"):
            _ang = np.degrees(np.arccos(
                (red[f"{_pref}_defl_dot"] / red[f"{_pref}_defl_nrm"]).clip(-1.0, 1.0)))
            red[f"{_pref}_defl_angle"] = _ang.where(_ang["t"] != _ang["t"].values[-1])
    return red


# ----------------------------------------------------------------------------
# flatten / regroup: red (dict of Dataset|DataArray) <-> a single flat Dataset
# ----------------------------------------------------------------------------
def _flatten(red: dict) -> xr.Dataset:
    """{term: Dataset|DataArray} -> one Dataset with `term__var` data_vars
    (and bare `term` for DataArray terms). Each var keeps its own dims."""
    data = {}
    for term, obj in red.items():
        if isinstance(obj, xr.Dataset):
            for v in obj.data_vars:
                # drop non-dimension coords (e.g. GUIDANCE_DELTA_value on a stray
                # *_step dim) so the store carries only the reduction dims
                data[f"{term}{_FLAT_SEP}{v}"] = obj[v].reset_coords(drop=True)
        else:  # DataArray (e.g. grads_l2_full)
            data[term] = obj.reset_coords(drop=True)
    return xr.Dataset(data)


def _regroup(ds: xr.Dataset) -> dict:
    """Inverse of `_flatten`: flat Dataset -> {term: Dataset|DataArray}."""
    red = {}
    for name in ds.data_vars:
        if _FLAT_SEP in name:
            term, var = name.split(_FLAT_SEP, 1)
            red.setdefault(term, {})[var] = ds[name]
        else:
            red[name] = ds[name]
    return {k: (xr.Dataset(v) if isinstance(v, dict) else v) for k, v in red.items()}


# ----------------------------------------------------------------------------
# build / load the persisted store
# ----------------------------------------------------------------------------
def _sweep_axes(sweep_params: dict) -> dict:
    """Per-axis coord labels used by the zarr (scalar value, else integer index),
    mirroring sweep_coord_label but over the whole axis list."""
    from src.utils import _is_scalar_axis
    return {
        k: (list(vs) if _is_scalar_axis(vs) else list(range(len(vs))))
        for k, vs in sweep_params.items()
    }


def _empty_full_container(reference_flat: xr.Dataset, sweep_params: dict) -> xr.Dataset:
    """All-NaN dask container over the full sweep grid, dims = reference dims + sweep
    axes; chunk 1 on m/n/sweep, full on t/level. Mirrors create_full_zarr_container's
    chunking but for arbitrary (non-spatial) reduction dims."""
    axis_labels = _sweep_axes(sweep_params)
    full_chunk = {"t", "level"}
    data_vars = {}
    coords = {}
    for name, da in reference_flat.data_vars.items():
        base_dims = list(da.dims)
        dims = base_dims + list(sweep_params.keys())
        shape = [da.sizes[d] for d in base_dims] + [len(axis_labels[k]) for k in sweep_params]
        chunks = tuple(da.sizes[d] if d in full_chunk else 1 for d in base_dims) + \
                 tuple(1 for _ in sweep_params)
        data_vars[name] = (dims, dask_full(shape, chunks))
        for d in base_dims:
            if d in da.coords:
                coords[d] = da.coords[d].values
    for k in sweep_params:
        coords[k] = axis_labels[k]
    return xr.Dataset(data_vars=data_vars, coords=coords)


def dask_full(shape, chunks):
    import dask.array as da
    return da.full(tuple(shape), np.nan, dtype=np.float32, chunks=tuple(chunks))


def build_reductions_store(rollout_id, *, verbose=True):
    """Compute reductions for EVERY sweep point once and write one reductions.zarr in
    the rollout folder. Idempotent create: overwrites any existing store."""
    ensure_rollout_dir(rollout_id)
    rollout_dir = get_rollout_dir(rollout_id)
    sweep_params = get_sweep_dict(rollout_id)
    sweeps = list(iter_sweeps(sweep_params))
    if verbose:
        print(f"[reductions] building {len(sweeps)} sweep points for {rollout_id}", flush=True)

    # coord-labeled sweep dicts (as the zarr coords), matching build_jobs / the notebook
    coord_sweeps = [
        {k: sweep_coord_label(k, v, sweep_params) for k, v in sw.items()}
        for sw in sweeps
    ]

    # reference point -> discover per-term dims -> empty full container
    ref_red = compute_reductions_for_sweep(rollout_id, coord_sweeps[0])
    ref_flat = _flatten(ref_red)
    container = _empty_full_container(ref_flat, sweep_params)
    save_path = rollout_dir / f"{REDUCTIONS_NAME}.zarr"
    container.to_zarr(save_path, mode="w", compute=False)

    for i, cs in enumerate(coord_sweeps):
        red = ref_red if i == 0 else compute_reductions_for_sweep(rollout_id, cs)
        flat = _flatten(red)
        # add the sweep coords so region="auto" lands on the right chunk
        flat = flat.assign_coords({k: v for k, v in cs.items()})
        flat = flat.expand_dims({k: [cs[k]] for k in sweep_params})
        append_to_zarr(rollout_dir, REDUCTIONS_NAME, flat)
        if verbose:
            print(f"[reductions] {i + 1}/{len(sweeps)} {cs}", flush=True)
    if verbose:
        print(f"[reductions] done -> {save_path}", flush=True)
    return save_path


def reductions_exist(rollout_id) -> bool:
    return (get_rollout_dir(rollout_id) / f"{REDUCTIONS_NAME}.zarr").exists()


def reductions_grid_matches(rollout_id) -> bool:
    """True iff a stored reductions.zarr *covers* the current sweep grid, i.e. every
    current sweep-axis label is present in the store. A superset store (built for a
    wider grid, then the sweep narrowed) still counts as usable -- charts select the
    current point out of it via load_reductions' `.sel`. Exact equality is not
    required; only that nothing the current grid needs is missing."""
    if not reductions_exist(rollout_id):
        return False
    try:
        ds = get_rollout(REDUCTIONS_NAME, rollout_id)
        sweep_params = get_sweep_dict(rollout_id)
        axis_labels = _sweep_axes(sweep_params)
        for k, labels in axis_labels.items():
            if k not in ds.dims:
                return False
            store_labels = list(ds[k].values)
            if not set(labels).issubset(set(store_labels)):
                return False
        return True
    except Exception:
        return False


def reductions_complete(rollout_id) -> bool:
    """True iff every sweep point in the store is populated (not still building).
    A sweep point is 'unbuilt' iff a representative always-written term (grads L2)
    is all-NaN across its non-sweep dims -- the container's untouched skeleton."""
    if not reductions_grid_matches(rollout_id):
        return False
    try:
        ds = get_rollout(REDUCTIONS_NAME, rollout_id)
        _rep = next((v for v in ds.data_vars if v.startswith("grads_l2" + _FLAT_SEP)),
                    next(iter(ds.data_vars)))
        _da = ds[_rep]
        _sweep_dims = [d for d in _da.dims if d not in ("m", "n", "t", "level")]
        if not _sweep_dims:
            return bool(_da.notnull().any().compute())
        _allnan = _da.isnull().all(dim=[d for d in _da.dims if d not in _sweep_dims])
        return not bool(_allnan.any().compute())
    except Exception:
        return False


def load_reductions(rollout_id, sweep_point) -> dict:
    """Read the persisted reductions and select one sweep point -> the `red` dict,
    loaded into memory (the slice is ~1 MB) so downstream chart access is instant."""
    ds = get_rollout(REDUCTIONS_NAME, rollout_id)
    sel = {k: v for k, v in sweep_point.items() if k in ds.dims}
    ds = ds.sel(sel).load()
    return _regroup(ds)
