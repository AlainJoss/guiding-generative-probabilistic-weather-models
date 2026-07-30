import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    plt.style.use("petroff10")

    from src.utils import (
        get_config, get_sweep_dict, get_rollout, get_rollout_ids,
        sweep_coord_label, get_slices, get_gt_rollout,
    )
    from src.mask import get_mask_2d, get_masked_mean
    from src.dimensions import LEVELS_DICT
    from src.ui.map import visualize_map, visualize_mask_3d, to_display_units
    from src.ui.comparison import residual_scaler
    from src.ui.plot_trajectory import plot_trajectory
    from src.reductions import compute_reductions_for_sweep
    from src.normalization import XarrayNormalizer
    from src.run import iter_sweeps

    # variable universe, 1:1 with the guidance notebook's cross checks
    LEVEL_VARS = ["geopotential", "u_component_of_wind", "v_component_of_wind",
                  "temperature", "specific_humidity", "vertical_velocity"]
    SURFACE_PAIR = {"temperature": "2m_temperature",
                    "u_component_of_wind": "10m_u_component_of_wind",
                    "v_component_of_wind": "10m_v_component_of_wind"}

    # colormap conventions, 1:1 from the guidance notebook
    white_zero_cmap = plt.get_cmap("RdBu_r").copy()
    white_zero_cmap.set_bad("white")
    import matplotlib.colors as mcolors
    warm_half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_warm", white_zero_cmap(np.linspace(0.5, 1.0, 256)))
    cool_half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_cool", white_zero_cmap(np.linspace(0.0, 0.5, 256)))
    return (
        LEVELS_DICT,
        LEVEL_VARS,
        SURFACE_PAIR,
        XarrayNormalizer,
        compute_reductions_for_sweep,
        cool_half_cmap,
        get_config,
        get_gt_rollout,
        get_mask_2d,
        get_masked_mean,
        get_rollout,
        get_rollout_ids,
        get_slices,
        get_sweep_dict,
        iter_sweeps,
        mo,
        np,
        plot_trajectory,
        plt,
        residual_scaler,
        sweep_coord_label,
        to_display_units,
        visualize_map,
        visualize_mask_3d,
        warm_half_cmap,
        white_zero_cmap,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Intensity comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    refresh_button = mo.ui.run_button(label="refresh")
    return (refresh_button,)


@app.cell(hide_code=True)
def _(get_rollout_ids, mo, refresh_button):
    # experiment: the requested id when present, else pick from what exists
    if refresh_button.value:
        pass

    _ids = get_rollout_ids("gui")
    _default = "2026-07-28_00:00:07"
    rollout_dropdown = mo.ui.dropdown(
        _ids, value=(_default if _default in _ids else (_ids[0] if _ids else None)),
        label="experiment: ",
    )
    mo.hstack([rollout_dropdown, refresh_button], justify="start", align="center")
    return (rollout_dropdown,)


@app.cell(hide_code=True)
def _(get_config, get_sweep_dict, rollout_dropdown):
    ROLLOUT_ID = rollout_dropdown.value
    config = get_config(ROLLOUT_ID)
    SWEEP = get_sweep_dict(ROLLOUT_ID)
    N_STEPS = int(config.N)
    DELTAS = SWEEP["GUIDANCE_DELTA"]
    # intensity order: lowest peak first (labels carry the peak, values = zarr index)
    delta_order = sorted(range(len(DELTAS)), key=lambda _i: max(DELTAS[_i]))
    delta_labels = {_i: f"peak@{100 * max(DELTAS[_i]):+.3g}%" for _i in delta_order}
    return (
        DELTAS,
        N_STEPS,
        ROLLOUT_ID,
        SWEEP,
        config,
        delta_labels,
        delta_order,
    )


@app.cell(hide_code=True)
def _(SWEEP, mo):
    # sweep params: pin every swept axis EXCEPT the intensity axis (the old
    # "delta"); single-valued axes are pinned silently
    _pin = {}
    for _k, _vals in SWEEP.items():
        if _k == "GUIDANCE_DELTA" or not isinstance(_vals, list):
            continue
        if len(_vals) > 1:
            _sorted = sorted(_vals) if all(isinstance(_v, (int, float)) for _v in _vals) else _vals
            _pin[_k] = mo.ui.dropdown([str(_v) for _v in _sorted], value=str(_sorted[0]), label=f"{_k}: ")
    pin_dropdowns = mo.ui.dictionary(_pin)
    return (pin_dropdowns,)


@app.cell(hide_code=True)
def _(config, mo):
    m_slider = mo.ui.slider(0, max(int(config.M) - 1, 0), step=1, value=0, label="m: ", show_value=True, debounce=True)
    return (m_slider,)


@app.cell(hide_code=True)
def _(mo):
    dpi_slider = mo.ui.slider(steps=[80, 100, 130, 200, 300], value=100, label="dpi: ", show_value=True, debounce=True)
    return (dpi_slider,)


@app.cell(hide_code=True)
def _(sweep_params_row_ic):
    sweep_params_row_ic()
    return


@app.cell(hide_code=True)
def _(
    SWEEP,
    config,
    get_gt_rollout,
    get_mask_2d,
    iter_sweeps,
    mo,
    np,
    stores,
    sweep_coord_label,
    xr,
):
    # experiments tables (Experiment section, side by side):
    #   LEFT  -- run status per sweep point (zarrs are NaN-initialized, so
    #            all-finite = ran)
    #   RIGHT -- Δtarget per guided (n, m): SIGNED relative miss
    #            (M(x_gui) - A) / |A - base|, A = (1 + phi_n) * base (rollout.py);
    #            ❌ marks |Δtarget| > 5%; phi_n = 0 steps are not guided -> blank
    if stores["gui"] is not None:
        _probe_var = list(stores["gui"].data_vars)[0]
        _swept_axes = [_k for _k, _v in SWEEP.items() if isinstance(_v, list) and len(_v) > 1]
        _sweeps = list(iter_sweeps(SWEEP))

        # row order: a_t_mode first IN SWEEP ORDER (as authored in the selector),
        # then peak@ (GUIDANCE_DELTA), then sigma_div, then any remaining axes
        _am_order = {str(_v): _j for _j, _v in enumerate(SWEEP.get("a_t_mode") or [])}

        def _sort_key(_sw):
            _rest = [str(_sw[_k]) for _k in _swept_axes
                     if _k not in ("a_t_mode", "GUIDANCE_DELTA", "sigma_div")]
            return (_am_order.get(str(_sw.get("a_t_mode", "")), -1),
                    max(_sw["GUIDANCE_DELTA"]) if "GUIDANCE_DELTA" in _sw else 0.0,
                    float(_sw.get("sigma_div") or 0.0),
                    _rest)

        _sweeps.sort(key=_sort_key)

        _REL_TOL = 0.05  # achieved change may be within 5% of the requested change
        _tda = stores["gui"][config.VAR]
        if "level" in _tda.dims:
            _tda = _tda.sel(level=config.LEVEL)

        # each row is evaluated with ITS OWN mask (MASK_MODE / sigma_div can be
        # swept axes, and the rollout target uses that row's mask); baselines
        # anchoring A = (1 + phi_n) * base (rollout.py) are cached per mask
        _mask_cache, _base_cache = {}, {}

        def _row_mask(_sw):
            _mm2 = str(_sw.get("MASK_MODE") or config.MASK_MODE or "BBOX")
            _sg2 = float(_sw.get("sigma_div") or config.sigma_div or 2.0)
            _key = (_mm2, _sg2)
            if _key not in _mask_cache:
                _mask_cache[_key] = xr.DataArray(
                    np.asarray(get_mask_2d(_mm2, config.MASK_CORNERS, sigma_div=_sg2)),
                    dims=("latitude", "longitude"),
                    coords={"latitude": stores["gui"].latitude,
                            "longitude": stores["gui"].longitude})
            return _key, _mask_cache[_key]

        def _base_for(_key, _mda2):
            if _key in _base_cache:
                return _base_cache[_key]
            _b2 = None
            if config.GUI_REF == "GT":
                _gt_t = get_gt_rollout(config.N + 1, config.START_TS)[config.VAR]
                if "level" in _gt_t.dims:
                    _gt_t = _gt_t.sel(level=config.LEVEL)
                _b2 = np.asarray((_gt_t.astype("float64") * _mda2)
                                 .sum(("latitude", "longitude")).compute())[1:]  # (N,)
            elif stores["ung"] is not None and config.VAR in stores["ung"]:
                _uda = stores["ung"][config.VAR]
                if "t" in _uda.dims:
                    _uda = _uda.isel(t=-1)
                if "level" in _uda.dims:
                    _uda = _uda.sel(level=config.LEVEL)
                _b2 = np.asarray((_uda.astype("float64") * _mda2)
                                 .sum(("latitude", "longitude")).compute())  # (M, N)
            _base_cache[_key] = _b2
            return _b2


        def _fmt_sw(_k, _v):
            if _k == "GUIDANCE_DELTA":
                return f"peak@{100 * max(_v):+.3g}%"
            return str(_v)


        def _rel_miss(_sel3, _phis, _sw):
            # {(n, m): signed relative miss}; None when no baseline is available
            _key, _mda2 = _row_mask(_sw)
            _base = _base_for(_key, _mda2)
            if _base is None:
                return None
            _gm = np.asarray((_tda.sel(_sel3).astype("float64") * _mda2)
                             .sum(("latitude", "longitude")).compute())  # (M, N)
            _out = {}
            for _n2, _p2 in enumerate(_phis):
                if _n2 >= _gm.shape[1] or float(_p2) == 0.0:
                    continue  # phi_n = 0 -> step not guided (rollout.py copies ung)
                _b2 = _base[_n2] if _base.ndim == 1 else _base[:, _n2]
                _a2 = (1.0 + float(_p2)) * _b2
                _gap = np.maximum(np.abs(float(_p2) * _b2), 1e-12)  # |A - base|
                _r = np.atleast_1d((_gm[:, _n2] - _a2) / _gap)
                for _m2 in range(_gm.shape[0]):
                    if np.isfinite(_r[_m2]):
                        _out[(_n2, _m2)] = float(_r[_m2])
            return _out


        _rans, _rels = [], []
        for _sw in _sweeps:
            _sel2 = {_k: sweep_coord_label(_k, _v, SWEEP) for _k, _v in _sw.items()}
            try:
                _ran = bool(stores["gui"][_probe_var].sel(_sel2).notnull().all().compute())
            except (KeyError, ValueError):
                _ran = False
            _rans.append(_ran)
            try:
                _rels.append(_rel_miss({_k: _v for _k, _v in _sel2.items() if _k in _tda.dims},
                                       _sw.get("GUIDANCE_DELTA") or [], _sw) if _ran else None)
            except (KeyError, ValueError):
                _rels.append(None)

        _ax_cells = lambda _sw: " | ".join(_fmt_sw(_k, _sw[_k]) for _k in _swept_axes)

        # LEFT: run table
        _run_md = ["| run | " + " | ".join(_swept_axes) + " |",
                   "|" + "---|" * (len(_swept_axes) + 1)]
        for _ran, _sw in zip(_rans, _sweeps):
            _run_md.append("| " + ("✅" if _ran else "") + " | " + _ax_cells(_sw) + " |")

        # RIGHT: Δtarget table over the union of guided (n, m) columns
        _nm_cols = sorted({_nm for _r in _rels if _r for _nm in _r})


        def _cell(_r, _nm):
            if not _r or _nm not in _r:
                return ""
            _v = _r[_nm]
            return ("🎯" if abs(_v) <= _REL_TOL else "❌ ") + f"{100 * _v:+.1f}%"


        if _nm_cols:
            _tgt_md = ["| " + " | ".join(_swept_axes) + " | "
                       + " | ".join(f"n={_n2}, m={_m2}" for _n2, _m2 in _nm_cols) + " |",
                       "|" + "---|" * (len(_swept_axes) + len(_nm_cols))]
            for _r, _sw in zip(_rels, _sweeps):
                _tgt_md.append("| " + _ax_cells(_sw) + " | "
                               + " | ".join(_cell(_r, _nm) for _nm in _nm_cols) + " |")
            _tgt_view = mo.md("\n".join(_tgt_md))
        else:
            _tgt_view = mo.md(r"*(no baseline store -- $\Delta$target unavailable)*")

        experiments_table_ic = mo.hstack(
            [mo.vstack([mo.md("**runs**"), mo.md("\n".join(_run_md))], align="start"),
             mo.vstack([mo.md(r"**$\Delta$target**"),
                        _tgt_view], align="start")],
            justify="start", align="start", gap=2.0, wrap=True)
    else:
        experiments_table_ic = None
    experiments_table_ic
    return


@app.cell(hide_code=True)
def _(
    ROLLOUT_ID,
    SWEEP,
    config,
    get_mask_2d,
    get_rollout,
    np,
    pin_dropdowns,
    sweep_coord_label,
):
    # base selection (coord labels) for every axis except GUIDANCE_DELTA
    def _pinned(_k):
        _vals = SWEEP[_k]
        if _k in pin_dropdowns.value:
            _raw = pin_dropdowns.value[_k]
            for _v in _vals:
                if str(_v) == _raw:
                    return _v
        return _vals[0]

    base_sel = {}
    for _k, _vals in SWEEP.items():
        if _k == "GUIDANCE_DELTA" or not isinstance(_vals, list):
            continue
        base_sel[_k] = sweep_coord_label(_k, _pinned(_k), SWEEP)

    def delta_sel(_i):
        return {**base_sel, "GUIDANCE_DELTA": sweep_coord_label("GUIDANCE_DELTA", SWEEP["GUIDANCE_DELTA"][_i], SWEEP)}

    mask = get_mask_2d(str(base_sel.get("MASK_MODE", "BBOX")), config.MASK_CORNERS,
                       sigma_div=float(base_sel.get("sigma_div", 2.0)))
    mask_np = np.asarray(mask)

    # lazy store handles (full datasets; sliced per delta/var below)
    _stores = {}
    for _s in ("gui", "gui_ung", "gui_det", "ung_det", "ung", "res", "vfs", "grads"):
        try:
            _stores[_s] = get_rollout(_s, ROLLOUT_ID)
        except FileNotFoundError:
            _stores[_s] = None
    stores = _stores
    return base_sel, delta_sel, mask, mask_np, stores


@app.cell(hide_code=True)
def _(mo):
    differential_checkbox = mo.ui.checkbox(label=r"$\Delta$")
    return (differential_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    abs_checkbox = mo.ui.checkbox(label=r"$|\cdot|$")
    return (abs_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    top_k_slider = mo.ui.slider(steps=sorted({1, *range(5, 83, 5), 82}), value=5, label="top K: ", show_value=True, debounce=True)
    return (top_k_slider,)


@app.cell(hide_code=True)
def _(mo):
    rank_by_dropdown = mo.ui.dropdown(["final value", "initial value", "agg change"], value="agg change", label="rank by: ")
    return (rank_by_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    dist_bands_checkbox = mo.ui.checkbox(label="dist bands")
    return (dist_bands_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    cross_masked_checkbox = mo.ui.checkbox(value=True)
    return (cross_masked_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    cross_norms_checkbox = mo.ui.checkbox(value=True)
    return (cross_norms_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    masked_units_dropdown = mo.ui.dropdown(["normalized", "data"], value="normalized", label="units: ")
    return (masked_units_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    stats_scope_dropdown = mo.ui.dropdown(["mask", "zoom", "full"], value="mask", label="stats: ")
    return (stats_scope_dropdown,)


@app.cell(hide_code=True)
def _(LEVEL_VARS, mo):
    cross_var_dropdown = mo.ui.dropdown(LEVEL_VARS + ["mean_sea_level_pressure"], allow_select_none=True, label="variable: ")
    return (cross_var_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    ylim_mode_dropdown = mo.ui.dropdown(["shared", "own"], value="shared", label="ylim: ")
    return (ylim_mode_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    settings_bands_checkbox = mo.ui.checkbox(label="dist bands")
    return (settings_bands_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    intensity_bands_checkbox = mo.ui.checkbox(label="dist bands")
    return (intensity_bands_checkbox,)


@app.cell(hide_code=True)
def _(
    LEVEL_VARS,
    ROLLOUT_ID,
    SURFACE_PAIR,
    XarrayNormalizer,
    base_sel,
    compute_reductions_for_sweep,
    config,
    delta_sel,
    get_gt_rollout,
    mask_np,
    mo,
    np,
    pin_dropdowns,
    plt,
    red_cache_ic,
    stats_scope_dropdown,
    stores,
    xr,
    zoom_slider,
):
    # cross-var machinery, 1:1 with the guidance notebook (trimmed to what
    # this notebook needs: all vars by level, rank-by modes, min/max m-bands)
    _ALL_VARS = LEVEL_VARS + ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
    _SURF_TO_LVL = {s: l for l, s in SURFACE_PAIR.items()}
    _PALETTE = plt.get_cmap("tab10").colors
    import matplotlib.colors as _mc2


    def grouped_vars_ic(ds):
        out = {}
        for name in ds.data_vars:
            da = ds[name]
            if "level" in da.dims:
                for lv in da["level"].values:
                    out[f"{name} L{int(lv)}"] = da.sel(level=lv)
            else:
                out[name] = da
        return out


    def color_for_ic(labels):
        out = {}
        for label in labels:
            if " L" in label:
                name, lv = label.rsplit(" L", 1); lv = int(lv)
            else:
                name, lv = label, None
            base = _SURF_TO_LVL.get(name, name)
            hue = _mc2.to_rgb(_PALETTE[_ALL_VARS.index(base) % len(_PALETTE)])
            intensity = 1.0 if lv is None else 0.3 + 0.7 * lv / 1000
            out[label] = tuple(c + (1 - c) * (1 - intensity) for c in hue)
        return out


    def rank_score_ic(v, rank_by):
        _v = np.abs(np.asarray(v, dtype=float))
        if not np.isfinite(_v).any():
            return np.inf
        _fin = _v[np.isfinite(_v)]
        if rank_by == "final value":
            return -float(_fin[-1])
        if rank_by == "initial value":
            return -float(_fin[0])
        return -float(np.nanmean(_v))


    def cross_traces_ic(red_ds, agg, m_idx, *, k, diff, absv, bands, rank_by, var=None):
        _g = grouped_vars_ic(red_ds)
        if var is not None:
            _keep = {var} | ({SURFACE_PAIR[var]} if var in SURFACE_PAIR else set())
            _g = {k_: v for k_, v in _g.items() if k_.split(" L")[0] in _keep}

        def _reduce(da):
            _other = [d for d in da.dims if d not in ("n", "m")]
            out = np.sqrt((da ** 2).sum(dim=_other)) if agg == "l2" else da.mean(dim=_other)
            if "m" in out.dims:
                out = out.transpose("m", "n")
            arr = np.atleast_2d(np.asarray(out))
            if diff:
                arr = np.concatenate([np.zeros((arr.shape[0], 1)), np.diff(arr, axis=1)], axis=1)
            if absv:
                arr = np.abs(arr)
            return arr

        _full = {k_: _reduce(da) for k_, da in _g.items()}
        _member = {k_: v[m_idx if v.shape[0] > 1 else 0] for k_, v in _full.items()}
        traces = dict(sorted(_member.items(), key=lambda kv: rank_score_ic(kv[1], rank_by))[:k])
        bands_out = (
            {k_: (_full[k_].min(axis=0), _full[k_].max(axis=0)) for k_ in traces if _full[k_].shape[0] > 1}
            if bands else None
        )
        return traces, bands_out


    xnorm_ic = XarrayNormalizer()


    def red_for_ic(_i):
        _key = (ROLLOUT_ID, _i, tuple(sorted(base_sel.items())))
        if _key not in red_cache_ic:
            red_cache_ic[_key] = compute_reductions_for_sweep(ROLLOUT_ID, delta_sel(_i))
        return red_cache_ic[_key]


    # ground-truth initial condition (n=0) masked mean, normalized per variable --
    # the n=0 anchor of the absolute guided-levels chart (guidance convention)
    _gui_ds0 = stores["gui"]
    _mw_ic = xr.DataArray(mask_np, dims=("latitude", "longitude"),
                          coords={"latitude": _gui_ds0.latitude, "longitude": _gui_ds0.longitude})
    _ic0 = get_gt_rollout(config.N + 1, config.START_TS).isel(time=0).assign_coords(
        latitude=_gui_ds0.latitude, longitude=_gui_ds0.longitude, level=_gui_ds0.level)
    gt_ic_wnorm_ic = xnorm_ic.normalize((_ic0 * _mw_ic).sum(("latitude", "longitude")) / float(_mw_ic.sum()))


    def add_map_stats_ic(map_obj, stats_arr):
        """min/max (left) and mean/std (right) at title height; scope follows
        stats_scope_dropdown: mask core / current zoom window / full globe."""
        _fig = map_obj[0] if isinstance(map_obj, tuple) else map_obj
        if _fig is None or not hasattr(_fig, "axes") or not _fig.axes:
            return _fig
        _a = np.asarray(stats_arr, dtype=float)
        _scope = stats_scope_dropdown.value
        if _scope == "mask":
            _a = np.where(mask_np >= 0.5 * float(mask_np.max()), _a, np.nan)
        elif _scope == "zoom" and int(zoom_slider.value) > 1:
            _z = int(zoom_slider.value)
            _clon = 0.5 * (config.MASK_CORNERS[0] + config.MASK_CORNERS[1])
            _clat = 0.5 * (config.MASK_CORNERS[2] + config.MASK_CORNERS[3])
            if _clon > 180.0:
                _clon -= 360.0
            _lsp, _tsp = 360.0 / _z, 180.0 / _z
            _lat_e = np.linspace(90.0, -90.0, 122); _lat_c = 0.5 * (_lat_e[:-1] + _lat_e[1:])
            _lon_e = np.linspace(-180.0, 180.0, 241); _lon_c2 = 0.5 * (_lon_e[:-1] + _lon_e[1:])
            _zr = (((_lat_c >= _clat - _tsp / 2) & (_lat_c <= _clat + _tsp / 2))[:, None]
                   & ((_lon_c2 >= _clon - _lsp / 2) & (_lon_c2 <= _clon + _lsp / 2))[None, :])
            _a = np.where(_zr, _a, np.nan)
        _ax0 = _fig.axes[0]
        if np.isfinite(_a).any():
            _ax0.set_title(f"min = {np.nanmin(_a):.3g} | max = {np.nanmax(_a):.3g}", loc="left", fontsize=8)
            _ax0.set_title(f"mean = {np.nanmean(_a):.3g} | std = {np.nanstd(_a):.3g}", loc="right", fontsize=8)
        else:
            _ax0.set_title("all NaN", loc="left", fontsize=8)
        return _fig


    def sweep_params_row_ic():
        """First controls row of every section: sweep params — param1: ..."""
        _pins = [pin_dropdowns[_k] for _k in pin_dropdowns.value]
        return mo.hstack([mo.md("sweep params — ")] + (_pins or [mo.md("_(none)_")]),
                         justify="start", align="center")

    return (
        add_map_stats_ic,
        color_for_ic,
        cross_traces_ic,
        gt_ic_wnorm_ic,
        red_for_ic,
        sweep_params_row_ic,
        xnorm_ic,
    )


@app.cell(hide_code=True)
def _():
    red_cache_ic = {}  # (rollout, delta idx, pins) -> red dict
    return (red_cache_ic,)


@app.cell(hide_code=True)
def _(SWEEP, config, mo):
    mo.hstack([
        mo.accordion({"Config": mo.json(config.to_dict())}),
        mo.accordion({"Param grid": mo.json(SWEEP)}),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    zoom_slider = mo.ui.slider(1, 8, step=1, value=1, label="zoom: ", show_value=True, debounce=True)
    return (zoom_slider,)


@app.cell(hide_code=True)
def _(N_STEPS, mo):
    weather_n_slider = mo.ui.slider(0, max(N_STEPS - 1, 0), step=1, value=0, label="n: ", show_value=True, debounce=True)
    return (weather_n_slider,)


@app.cell(hide_code=True)
def _(mo):
    mask3d_elev_slider = mo.ui.slider(0, 90, step=5, value=45, label="elev: ", show_value=True, debounce=True)
    return (mask3d_elev_slider,)


@app.cell(hide_code=True)
def _(mo):
    mask3d_azim_slider = mo.ui.slider(-180, 180, step=5, value=-45, label="azim: ", show_value=True, debounce=True)
    return (mask3d_azim_slider,)


@app.cell(hide_code=True)
def _(
    contour_color_dropdown,
    dpi_slider,
    mo,
    sweep_params_row_ic,
    weather_n_slider,
    zoom_slider,
):
    mo.vstack([
        mo.md(r"""## Mask"""),
        sweep_params_row_ic(),
        mo.hstack([zoom_slider, weather_n_slider, dpi_slider, contour_color_dropdown], justify="start", align="center"),
    ], align="start")
    return


@app.cell(hide_code=True)
def _(
    add_map_stats_ic,
    base_sel,
    config,
    contour_color_dropdown,
    cool_half_cmap,
    dpi_slider,
    get_gt_rollout,
    get_slices,
    mask,
    mask3d_azim_slider,
    mask3d_elev_slider,
    mask_np,
    mo,
    np,
    to_display_units,
    visualize_map,
    visualize_mask_3d,
    warm_half_cmap,
    weather_n_slider,
    white_zero_cmap,
    zoom_slider,
):
    # mask-section charts, 1:1 style with the guidance notebook
    _lon_c = 0.5 * (config.MASK_CORNERS[0] + config.MASK_CORNERS[1])
    _lat_c = 0.5 * (config.MASK_CORNERS[2] + config.MASK_CORNERS[3])
    if _lon_c > 180.0:
        _lon_c -= 360.0
    _gt_roll = get_gt_rollout(config.N + 1, config.START_TS)
    _wsl = np.asarray(get_slices(_gt_roll, config.PARTITION, config.VAR, config.LEVEL), dtype=float)
    _wsl, _w_unit = to_display_units(_wsl, config.VAR)
    _wn = int(weather_n_slider.value)
    _w_d = _wsl[_wn]
    _wmin, _wmax = float(np.nanmin(_wsl)), float(np.nanmax(_wsl))
    if _wmin < 0.0 < _wmax:
        _w_cmap, _w_center = white_zero_cmap, 0.0
    elif _wmin >= 0.0:
        _w_cmap, _w_center = warm_half_cmap, None
    else:
        _w_cmap, _w_center = cool_half_cmap, None
    _weather = visualize_map(
        _w_d, cmap=_w_cmap, vmin=_wmin, vmax=_wmax, center=_w_center,
        title=f"{config.VAR} | level={config.LEVEL}" + (f" | [{_w_unit}]" if _w_unit else ""),
        mask_2d=mask, show_mask=True,
        contour_2d=None if str(base_sel.get("MASK_MODE", "BBOX")) == "BBOX" else mask,
        contour_levels=8, contour_color=contour_color_dropdown.value, contour_linewidth=0.5,
        zoom=zoom_slider.value, zoom_center_lon=_lon_c, zoom_center_lat=_lat_c,
        figsize=(10, 5.5), dpi=dpi_slider.value, interactive=False,
    )
    _weather = add_map_stats_ic(_weather, _w_d)
    _mmin, _mmax = float(np.min(mask_np)), float(np.max(mask_np))
    import matplotlib.colors as _mc
    _mask_cmap = (_mc.LinearSegmentedColormap.from_list(
        "mask_warm", white_zero_cmap(np.linspace(0.5, 1.0, 256))) if _mmin == 0.0 else white_zero_cmap)
    _mmap = visualize_map(
        mask_np, cmap=_mask_cmap,
        vmin=_mmin if _mmin < _mmax else -0.001, vmax=_mmax if _mmin < _mmax else 0.001,
        center=(None if _mmin == 0.0 else 0.5 * (_mmin + _mmax)) if _mmin < _mmax else 0.0,
        title="mask", contour_2d=None if str(base_sel.get("MASK_MODE", "BBOX")) == "BBOX" else mask_np,
        contour_levels=8, contour_color=contour_color_dropdown.value, contour_linewidth=0.5,
        zoom=zoom_slider.value, zoom_center_lon=_lon_c, zoom_center_lat=_lat_c,
        figsize=(10, 5.5), dpi=dpi_slider.value, interactive=False,
    )
    _mmap = add_map_stats_ic(_mmap, mask_np)
    _m3d = visualize_mask_3d(
        mask_np, cmap=white_zero_cmap, title="mask (3D)",
        elev=mask3d_elev_slider.value, azim=mask3d_azim_slider.value,
        zoom=zoom_slider.value, zoom_center_lon=_lon_c, zoom_center_lat=_lat_c,
        figsize=(10, 5.5), dpi=dpi_slider.value,
    )
    _m3d = _m3d[0] if isinstance(_m3d, tuple) else _m3d
    mo.hstack(
        [_weather, _mmap,
         mo.vstack([_m3d, mo.hstack([mask3d_elev_slider, mask3d_azim_slider], justify="start")], align="start")],
        justify="start", align="start",
    )
    return


@app.cell(hide_code=True)
def _(
    DELTAS,
    N_STEPS,
    config,
    delta_labels,
    delta_order,
    delta_sel,
    dpi_slider,
    get_gt_rollout,
    get_masked_mean,
    get_slices,
    mask,
    np,
    plt,
    settings_bands_checkbox,
    stores,
    to_display_units,
):
    # rollout trajectories per intensity + realized guidance schedules, side by side
    # (guidance-notebook grammar: n=0 anchor, dashed guides to the y-axis, Delta_cum
    # labels; dist bands shade the min/max across members)
    # sequential ramp: lightest = lowest peak, darkest = highest (monotone)
    ICOLORS = [plt.get_cmap("Reds")(_x) for _x in np.linspace(0.35, 0.95, max(len(delta_order), 2))]


    def _selp2(_ds, _sel):
        return _ds.sel({_k: _v for _k, _v in _sel.items() if _k in _ds.dims})


    def _mm_raw(_states, _n, _m):
        # store-unit masked mean: the rollout target A = (1+phi)*base lives in raw
        # units, so the residual must too (display offsets break the scaling)
        _a = np.asarray(get_slices(_states, config.PARTITION, config.VAR, config.LEVEL))
        _a = _a[_m][_n] if _a.ndim == 4 else _a[_n + 1]
        return float(get_masked_mean(_a[None, None], mask)[0, 0])


    def _mm_traj(_states, _n, _m):
        _a = np.asarray(get_slices(_states, config.PARTITION, config.VAR, config.LEVEL))
        _a = _a[_m][_n] if _a.ndim == 4 else _a[_n + 1]
        _a, _u = to_display_units(_a, config.VAR)
        return float(get_masked_mean(_a[None, None], mask)[0, 0])


    def traj_schedule_figs(_m, _sched_n):
        _fin = lambda _ds: _ds.isel(t=-1) if "t" in _ds.dims else _ds
        _fig, _ax = plt.subplots(figsize=(11, 5.2), dpi=dpi_slider.value)
        _xs = np.arange(0, N_STEPS + 1)
        _gt_roll2 = get_gt_rollout(config.N + 1, config.START_TS)
        _ic_val = _mm_traj(_gt_roll2, -1, _m)  # shared initial state = the n=0 anchor
        _x_pad = 0.06 * max(N_STEPS, 1)
        _guf0 = _fin(_selp2(stores["gui_ung"], delta_sel(delta_order[0])))
        _ungv = [_mm_traj(_guf0, _n, _m) for _n in range(N_STEPS)]
        for _j, _i in enumerate(delta_order):
            _gui_i = _selp2(stores["gui"], delta_sel(_i))
            _ys = [_ic_val] + [_mm_traj(_gui_i, _n, _m) for _n in range(N_STEPS)]
            _col = ICOLORS[_j % len(ICOLORS)]
            _ax.plot(_xs, _ys, "-o", color=_col, lw=2.0, ms=5, label=delta_labels[_i])
            for _k, (_xi, _yi) in enumerate(zip(_xs, _ys)):
                _ax.plot([-_x_pad, _xi], [_yi, _yi], linestyle="--", linewidth=0.8,
                         color=_col, alpha=0.35, zorder=1, label="_nolegend_")
                _cumtxt = "" if _k == 0 else rf", $\Delta_{{\mathrm{{cum}}}}${_yi - _ic_val:+.2f}"
                _ax.annotate(rf"{_yi:.2f}{_cumtxt}", xy=(0.0, _yi),
                             xycoords=("axes fraction", "data"), xytext=(3, 2),
                             textcoords="offset points", ha="left", va="bottom",
                             fontsize=7, color=_col, zorder=11)
            if settings_bands_checkbox.value and int(config.M) > 1:
                _ya = np.array([[_mm_traj(_gui_i, _n, _mm) for _n in range(N_STEPS)]
                                for _mm in range(int(config.M))])
                _ax.fill_between(_xs[1:], _ya.min(0), _ya.max(0), color=_col, alpha=0.14, linewidth=0)
            _tgt = [(1.0 + float(DELTAS[_i][_n])) * _ungv[_n] for _n in range(N_STEPS)]
            _ax.plot(_xs[1:], _tgt, ":", color=_col, lw=1.2, label="_nolegend_")
        _ax.plot(_xs, [_ic_val] + _ungv, "--", color="#800080", lw=1.8, label="gui_ung")
        _ax.plot(_xs, [_ic_val] + [_mm_traj(_gt_roll2, _n, _m) for _n in range(N_STEPS)],
                 "--", color="#009E73", lw=1.8, label="ground truth")
        _ax.set_xlim(-_x_pad, N_STEPS + _x_pad)
        _ax.set_xticks(_xs)
        _ax.set_xlabel("$n$")
        _ax.set_ylabel("Mask-averaged value" + " [°C]" if config.VAR in ("temperature", "2m_temperature") else "")
        _ax.set_title("rollout trajectories — per intensity (dotted = target)", loc="left", fontweight="bold")
        _ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
        _ax.legend(frameon=False, fontsize=8)
        _fig.tight_layout()
        plt.close(_fig)

        # realized schedules: lambda_hat_t = ||gui_vec_t|| for t < T-1 (last step never guided)
        def _lam_for(_ii, _mm):
            _res_i = _selp2(stores["res"], delta_sel(_ii)).isel(m=_mm, n=_sched_n)
            _vfs_i = _selp2(stores["vfs"], delta_sel(_ii)).isel(m=_mm, n=_sched_n)
            _Tl = int(_res_i.sizes["t"])
            _sg = np.linspace(1000, 1, _Tl) / 1000
            _hg = np.empty_like(_sg); _hg[:-1] = _sg[:-1] - _sg[1:]; _hg[-1] = _sg[-1]
            _shg = ((_sg / _hg)[:-1])
            _lam_sq = None
            for _v in _res_i.data_vars:
                _z = _res_i[_v]
                _dz = (_z.shift(t=-1) - _z).isel(t=slice(0, _Tl - 1))
                _gvf = _dz * _dz.t.copy(data=_shg)
                _gvec = (_vfs_i[_v].isel(t=slice(0, _Tl - 1)) - _gvf) / _dz.t.copy(data=_sg[:-1])
                _q = (_gvec ** 2).sum([d for d in _gvec.dims if d != "t"])
                _lam_sq = _q if _lam_sq is None else _lam_sq + _q
            return np.sqrt(np.asarray(_lam_sq.compute(), dtype=float))

        _fig2, _ax2 = plt.subplots(figsize=(9, 5.2), dpi=dpi_slider.value)
        for _j, _i in enumerate(delta_order):
            _T = int(_selp2(stores["res"], delta_sel(_i)).sizes["t"])
            _col = ICOLORS[_j % len(ICOLORS)]
            _lam = _lam_for(_i, _m)
            _ax2.plot(np.arange(1, _T), _lam, "-", color=_col, lw=1.8, label=delta_labels[_i])
            if settings_bands_checkbox.value and int(config.M) > 1:
                _la = np.stack([_lam_for(_i, _mm) for _mm in range(int(config.M))])
                _ax2.fill_between(np.arange(1, _T), _la.min(0), _la.max(0),
                                  color=_col, alpha=0.14, linewidth=0)
        _ax2.set_xlabel("$t$")
        _ax2.set_ylabel(r"$\hat\lambda_t$")
        _ax2.set_title(rf"Guidance schedule (realized, n={_sched_n})  $\hat\lambda_t = w\,a_t\,c_t$",
                       loc="left", fontweight="bold")
        _ax2.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
        for _sp in ("top", "right"):
            _ax2.spines[_sp].set_visible(False)
        _ax2.legend(frameon=False, fontsize=8)
        _fig2.tight_layout()
        plt.close(_fig2)
        return _fig, _fig2


    def target_diff_fig(_m):
        """Difference to the guidance target over n — same grammar as the
        trajectories chart (guide hlines, value labels, member bands)."""
        _fin = lambda _ds: _ds.isel(t=-1) if "t" in _ds.dims else _ds
        _fig, _ax = plt.subplots(figsize=(11, 5.2), dpi=dpi_slider.value)
        _xs = np.arange(0, N_STEPS + 1)
        _x_pad = 0.06 * max(N_STEPS, 1)
        # base anchoring A = (1+phi_n)*base -- SAME numbers as the experiments
        # table / rollout.py: the independent ung.zarr (per member) or gt (GT ref)
        _base_ds = (get_gt_rollout(config.N + 1, config.START_TS)
                    if config.GUI_REF == "GT" else _fin(stores["ung"]))

        def _gaps(_gui_i, _i, _mm):
            # phi_n = 0 -> the step is not guided (rollout copies ung): no target
            return [0.0] + [
                (_mm_raw(_gui_i, _n, _mm)
                 - (1.0 + float(DELTAS[_i][_n])) * _mm_raw(_base_ds, _n, _mm))
                if float(DELTAS[_i][_n]) != 0.0 else np.nan
                for _n in range(N_STEPS)
            ]

        for _j, _i in enumerate(delta_order):
            _gui_i = _selp2(stores["gui"], delta_sel(_i))
            _ys = _gaps(_gui_i, _i, _m)
            _col = ICOLORS[_j % len(ICOLORS)]
            _ax.plot(_xs, _ys, "-o", color=_col, lw=2.0, ms=5, label=delta_labels[_i])
            for _k, (_xi, _yi) in enumerate(zip(_xs, _ys)):
                if not np.isfinite(_yi):
                    continue
                _ax.plot([-_x_pad, _xi], [_yi, _yi], linestyle="--", linewidth=0.8,
                         color=_col, alpha=0.35, zorder=1, label="_nolegend_")
                _ax.annotate(rf"{_yi:+.2f}", xy=(0.0, _yi),
                             xycoords=("axes fraction", "data"), xytext=(3, 2),
                             textcoords="offset points", ha="left", va="bottom",
                             fontsize=7, color=_col, zorder=11)
            if settings_bands_checkbox.value and int(config.M) > 1:
                _ya = np.array([_gaps(_gui_i, _i, _mm) for _mm in range(int(config.M))])
                _ax.fill_between(_xs, np.nanmin(_ya, 0), np.nanmax(_ya, 0), color=_col, alpha=0.14, linewidth=0)
        _ax.axhline(0.0, color="#888888", linewidth=1.2, alpha=0.9, zorder=1)
        _ax.annotate("target", (_xs[-1], 0.0), textcoords="offset points", xytext=(10, 2),
                     ha="left", fontsize=8, color="#888888", annotation_clip=False)
        _ax.set_xlim(-_x_pad, N_STEPS + _x_pad)
        _ax.set_xticks(_xs)
        _ax.set_xlabel("$n$")
        _ax.set_ylabel("masked mean − target  $e = M(x^{gui}) - (1+\\varphi_n)\\,M(x^{ung})$" + (" [K]" if config.VAR in ("temperature", "2m_temperature") else ""))
        _ax.set_title("difference to the guidance target — per intensity", loc="left", fontweight="bold")
        _ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
        _ax.legend(frameon=False, fontsize=8)
        _fig.tight_layout()
        plt.close(_fig)
        return _fig

    return target_diff_fig, traj_schedule_figs


@app.cell(hide_code=True)
def _(
    dpi_slider,
    m_slider,
    mo,
    settings_bands_checkbox,
    sweep_params_row_ic,
    weather_n_slider,
):
    mo.vstack([
        mo.md(r"""## Settings"""),
        sweep_params_row_ic(),
        mo.hstack([m_slider, weather_n_slider, dpi_slider], justify="start", align="center"),
        mo.hstack([settings_bands_checkbox], justify="start", align="center"),
    ], align="start")
    return


@app.cell(hide_code=True)
def _(m_slider, mo, target_diff_fig, traj_schedule_figs, weather_n_slider):
    _tfig, _sfig = traj_schedule_figs(int(m_slider.value), int(weather_n_slider.value))
    _dfig = target_diff_fig(int(m_slider.value))
    mo.hstack([_tfig, _dfig, _sfig], justify="start", align="start", wrap=False)
    return


@app.cell(hide_code=True)
def _(
    dpi_slider,
    intensity_bands_checkbox,
    intensity_compare_by_dropdown,
    intensity_stat_dropdown,
    intensity_var_dropdown,
    m_slider,
    mo,
    sweep_params_row_ic,
):
    mo.vstack([
        mo.md(r"""## Guidance intensity"""),
        sweep_params_row_ic(),
        mo.hstack([intensity_var_dropdown, intensity_compare_by_dropdown, intensity_stat_dropdown, m_slider, dpi_slider], justify="start", align="center"),
        mo.hstack([intensity_bands_checkbox], justify="start", align="center"),
    ], align="start")
    return


@app.cell(hide_code=True)
def _(LEVEL_VARS, config, mo):
    intensity_var_dropdown = mo.ui.dropdown(LEVEL_VARS + ["mean_sea_level_pressure"], value=(config.VAR if config.VAR in LEVEL_VARS else "temperature"), label="variable: ")
    return (intensity_var_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    intensity_compare_by_dropdown = mo.ui.dropdown(["n", "levels"], value="n", label="compare by: ")
    return (intensity_compare_by_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    intensity_stat_dropdown = mo.ui.dropdown(["min", "mean", "max", "std"], value="mean", label="stat: ")
    return (intensity_stat_dropdown,)


@app.cell(hide_code=True)
def _(
    LEVELS_DICT,
    N_STEPS,
    SURFACE_PAIR,
    config,
    delta_labels,
    delta_order,
    delta_sel,
    dpi_slider,
    get_gt_rollout,
    get_masked_mean,
    get_slices,
    intensity_bands_checkbox,
    mask,
    mask_np,
    np,
    plt,
    stores,
    xr,
):
    # the "Guidance intensity" charts, parameterized by (variable, mask stat,
    # n or channel, member); one line per intensity. Channels follow the guided-
    # states convention: surface pair first (if any), then levels bottom -> top.
    # sequential ramp: lightest = lowest peak, darkest = highest (monotone)
    _ICOLORS = [plt.get_cmap("Reds")(_x) for _x in np.linspace(0.35, 0.95, max(len(delta_order), 2))]
    _gt = get_gt_rollout(config.N + 1, config.START_TS)
    from src.ui.map import to_display_units as _tdu
    _mask_core = mask_np >= 0.5 * float(mask_np.max())


    def _selp(_ds, _sel):
        # dim-safe .sel: stores from the unguided run carry no sweep dims
        return _ds.sel({_k: _v for _k, _v in _sel.items() if _k in _ds.dims})


    def intensity_channels(_var):
        if _var == "mean_sea_level_pressure":
            return [("sfc", "surface", "mean_sea_level_pressure", 0)]
        _sfc = [("sfc", "surface", SURFACE_PAIR[_var], 0)] if _var in SURFACE_PAIR else []
        return _sfc + [(f"L{_L}", "level", _var, _L) for _L in reversed(LEVELS_DICT["level"])]


    def _abs_stat(_states, _chan, _n, _m, _stat):
        _cl, _p, _v, _lv = _chan
        _a = np.asarray(get_slices(_states, _p, _v, _lv))
        _a = _a[_m][_n] if _a.ndim == 4 else _a[_n + 1]  # (m, n, ..) states; gt: (time, ..)
        _a, _u = _tdu(_a, _v, is_difference=False)
        if _stat == "mean":
            return float(get_masked_mean(_a[None, None], mask)[0, 0])
        _vals = np.asarray(_a)[_mask_core]
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size == 0:
            return float("nan")
        return float({"min": np.min, "max": np.max, "std": np.std}[_stat](_vals))


    def _band(_ax, _xs, _fn, _col):
        if intensity_bands_checkbox.value and int(config.M) > 1:
            _ya = np.array([_fn(_mm) for _mm in range(int(config.M))])
            _ax.fill_between(_xs, np.nanmin(_ya, 0), np.nanmax(_ya, 0), color=_col, alpha=0.14, linewidth=0)


    def _axes_style(_ax):
        _ax.yaxis.grid(True, color="#eeeeee")
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)


    def _lvl_axis(_ax, _chans):
        _xs = list(range(len(_chans)))
        _ax.set_xticks(_xs)
        _ax.set_xticklabels([_c[0] for _c in _chans], fontsize=8)
        _ax.set_xlabel("channel  (surface → top)")


    def _unit_tag(_var):
        return " [°C]" if _var in ("temperature", "2m_temperature") or _var in SURFACE_PAIR else ""


    # ---- x-axis = channels (one figure per n) ----
    def intensity_profile_abs(_n, _m, _stat, _var):
        _chans = intensity_channels(_var)
        _xs = list(range(len(_chans)))
        _fig, _ax = plt.subplots(figsize=(9.5, 4.2), dpi=dpi_slider.value)
        for _j, _i in enumerate(delta_order):
            _gui_i = _selp(stores["gui"], delta_sel(_i))
            _ys = [_abs_stat(_gui_i, _c, _n, _m, _stat) for _c in _chans]
            _col = _ICOLORS[_j % len(_ICOLORS)]
            _ax.plot(_xs, _ys, "-o", color=_col, lw=2.0, ms=4, label=delta_labels[_i])
            _band(_ax, _xs, lambda _mm: [_abs_stat(_gui_i, _c, _n, _mm, _stat) for _c in _chans], _col)
        _guf0 = _selp(stores["gui_ung"], delta_sel(delta_order[0]))
        _guf0 = _guf0.isel(t=-1) if "t" in _guf0.dims else _guf0
        _ax.plot(_xs, [_abs_stat(_guf0, _c, _n, _m, _stat) for _c in _chans], "--", color="#555555", lw=1.8, label="unguided (gui_ung)")
        _ax.plot(_xs, [_abs_stat(_gt, _c, _n, _m, _stat) for _c in _chans], "--", color="#009E73", lw=1.8, label="ground truth (gt)")
        _lvl_axis(_ax, _chans)
        _ax.set_ylabel(f"mask-{_stat} {_var}" + _unit_tag(_var))
        _ax.set_title(f"Absolute {_var} vs intensity (mask {_stat})", loc="left", fontweight="bold")
        _axes_style(_ax)
        _ax.legend(frameon=False, fontsize=8)
        _fig.tight_layout()
        plt.close(_fig)
        return _fig


    def intensity_profile_vs_gt(_n, _m, _stat, _var):
        _chans = intensity_channels(_var)
        _xs = list(range(len(_chans)))
        _fig, _ax = plt.subplots(figsize=(9.5, 4.6), dpi=dpi_slider.value)
        _gtv = [_abs_stat(_gt, _c, _n, _m, _stat) for _c in _chans]

        def _dev(_states, _mm=None):
            _use = _m if _mm is None else _mm
            return [_abs_stat(_states, _c, _n, _use, _stat) - _gtv[_k] for _k, _c in enumerate(_chans)]

        for _j, _i in enumerate(delta_order):
            _st_i = _selp(stores["gui"], delta_sel(_i))
            _col = _ICOLORS[_j % len(_ICOLORS)]
            _ax.plot(_xs, _dev(_st_i), "-o", color=_col, lw=2.0, ms=4, label=delta_labels[_i])
            _band(_ax, _xs, lambda _mm: _dev(_st_i, _mm), _col)
        _base0 = delta_sel(delta_order[0])
        for _lab, _st, _ls, _colb in [
            ("ung", stores["ung"].isel(t=-1) if stores["ung"] is not None and "t" in stores["ung"].dims else stores["ung"], "--", "#8B5A2B"),
            ("gui_ung", (lambda _d: _d.isel(t=-1) if "t" in _d.dims else _d)(_selp(stores["gui_ung"], _base0)), "--", "#800080"),
            ("gui_det", _selp(stores["gui_det"], _base0) if stores["gui_det"] is not None else None, "-.", "#555555"),
            ("ung_det", _selp(stores["ung_det"], _base0) if stores["ung_det"] is not None else None, ":", "#17becf"),
        ]:
            if _st is not None:
                _ax.plot(_xs, _dev(_st), _ls, color=_colb, lw=1.6, label=_lab)
        _ax.axhline(0.0, color="#009E73", lw=1.8, ls="--", label="ground truth (gt)")
        _lvl_axis(_ax, _chans)
        _ax.set_ylabel(f"mask-{_stat} {_var} − gt" + _unit_tag(_var))
        _ax.set_title(f"{_var} relative to ground truth (mask {_stat})", loc="left", fontweight="bold")
        _axes_style(_ax)
        _ax.legend(frameon=False, fontsize=8, ncol=2)
        _fig.tight_layout()
        plt.close(_fig)
        return _fig


    def intensity_kick_profile(_n, _m, _var):
        _chans = intensity_channels(_var)
        _xs = list(range(len(_chans)))
        _maskda = xr.DataArray(mask_np, dims=("latitude", "longitude"),
                               coords={"latitude": stores["grads"].latitude, "longitude": stores["grads"].longitude})
        _fig, _ax = plt.subplots(figsize=(9.5, 4.2), dpi=dpi_slider.value)
        for _j, _i in enumerate(delta_order):
            _gd = _selp(stores["grads"], delta_sel(_i))

            def _kick_ys(_mm):
                _out = []
                for _cl, _p, _v, _lv in _chans:
                    _da = _gd[_v].isel(m=_mm, n=_n)
                    if "level" in _da.dims:
                        _da = _da.sel(level=_lv)
                    _out.append(float(np.sqrt(((_da ** 2) * _maskda).sum(("latitude", "longitude")).sum("t").compute())))
                return _out
            _col = _ICOLORS[_j % len(_ICOLORS)]
            _ax.plot(_xs, _kick_ys(_m), "-o", color=_col, lw=2.0, ms=4, label=delta_labels[_i])
            _band(_ax, _xs, _kick_ys, _col)
        _lvl_axis(_ax, _chans)
        _ax.set_ylabel(r"guidance perturbation  $\|\nabla\mathcal{L}\|$  (mask, all steps)")
        _ax.set_title(r"Perturbation profile — guidance kick $\|\nabla\mathcal{L}\|$ by level", loc="left", fontweight="bold")
        _axes_style(_ax)
        _ax.legend(frameon=False, title="intensity")
        _fig.tight_layout()
        plt.close(_fig)
        return _fig


    # ---- x-axis = n (one figure per channel, compare-by = levels) ----
    def intensity_over_n_abs(_chan, _m, _stat, _var):
        _xs = list(range(N_STEPS))
        _fig, _ax = plt.subplots(figsize=(9.5, 4.2), dpi=dpi_slider.value)
        for _j, _i in enumerate(delta_order):
            _gui_i = _selp(stores["gui"], delta_sel(_i))
            _col = _ICOLORS[_j % len(_ICOLORS)]
            _ax.plot(_xs, [_abs_stat(_gui_i, _chan, _n, _m, _stat) for _n in _xs], "-o", color=_col, lw=2.0, ms=4, label=delta_labels[_i])
            _band(_ax, _xs, lambda _mm: [_abs_stat(_gui_i, _chan, _n, _mm, _stat) for _n in _xs], _col)
        _guf0 = _selp(stores["gui_ung"], delta_sel(delta_order[0]))
        _guf0 = _guf0.isel(t=-1) if "t" in _guf0.dims else _guf0
        _ax.plot(_xs, [_abs_stat(_guf0, _chan, _n, _m, _stat) for _n in _xs], "--", color="#555555", lw=1.8, label="unguided (gui_ung)")
        _ax.plot(_xs, [_abs_stat(_gt, _chan, _n, _m, _stat) for _n in _xs], "--", color="#009E73", lw=1.8, label="ground truth (gt)")
        _ax.set_xticks(_xs)
        _ax.set_xlabel("$n$")
        _ax.set_ylabel(f"mask-{_stat} {_var}" + _unit_tag(_var))
        _ax.set_title(f"Absolute {_var} over $n$ (mask {_stat}) — {_chan[0]}", loc="left", fontweight="bold")
        _axes_style(_ax)
        _ax.legend(frameon=False, fontsize=8)
        _fig.tight_layout()
        plt.close(_fig)
        return _fig


    def intensity_over_n_vs_gt(_chan, _m, _stat, _var):
        _xs = list(range(N_STEPS))
        _fig, _ax = plt.subplots(figsize=(9.5, 4.2), dpi=dpi_slider.value)
        _gtv = [_abs_stat(_gt, _chan, _n, _m, _stat) for _n in _xs]
        for _j, _i in enumerate(delta_order):
            _gui_i = _selp(stores["gui"], delta_sel(_i))
            _col = _ICOLORS[_j % len(_ICOLORS)]
            _ax.plot(_xs, [_abs_stat(_gui_i, _chan, _n, _m, _stat) - _gtv[_n] for _n in _xs], "-o", color=_col, lw=2.0, ms=4, label=delta_labels[_i])
            _band(_ax, _xs, lambda _mm: [_abs_stat(_gui_i, _chan, _n, _mm, _stat) - _gtv[_n] for _n in _xs], _col)
        _ax.axhline(0.0, color="#009E73", lw=1.8, ls="--", label="ground truth (gt)")
        _ax.set_xticks(_xs)
        _ax.set_xlabel("$n$")
        _ax.set_ylabel(f"mask-{_stat} {_var} − gt" + _unit_tag(_var))
        _ax.set_title(f"{_var} relative to gt over $n$ (mask {_stat}) — {_chan[0]}", loc="left", fontweight="bold")
        _axes_style(_ax)
        _ax.legend(frameon=False, fontsize=8)
        _fig.tight_layout()
        plt.close(_fig)
        return _fig

    return (
        intensity_channels,
        intensity_kick_profile,
        intensity_over_n_abs,
        intensity_over_n_vs_gt,
        intensity_profile_abs,
        intensity_profile_vs_gt,
    )


@app.cell(hide_code=True)
def _(
    N_STEPS,
    intensity_channels,
    intensity_compare_by_dropdown,
    intensity_kick_profile,
    intensity_over_n_abs,
    intensity_over_n_vs_gt,
    intensity_profile_abs,
    intensity_profile_vs_gt,
    intensity_stat_dropdown,
    intensity_var_dropdown,
    m_slider,
    mo,
):
    # Guidance-intensity grid: mask statistic from the selector; compare by n
    # (blocks per n, channel profiles) or by levels (blocks per channel, over n)
    _m = int(m_slider.value)
    _stat = intensity_stat_dropdown.value
    _var = intensity_var_dropdown.value
    _blocks = []
    if intensity_compare_by_dropdown.value == "n":
        for _n in range(N_STEPS):
            _blocks.append(mo.md(f"#### n = {_n}"))
            _blocks.append(mo.hstack(
                [intensity_profile_abs(_n, _m, _stat, _var),
                 intensity_profile_vs_gt(_n, _m, _stat, _var),
                 intensity_kick_profile(_n, _m, _var)],
                justify="start", align="start", wrap=False,
            ))
    else:
        for _chan in intensity_channels(_var):
            _blocks.append(mo.md(f"#### {_chan[0]}"))
            _blocks.append(mo.hstack(
                [intensity_over_n_abs(_chan, _m, _stat, _var),
                 intensity_over_n_vs_gt(_chan, _m, _stat, _var)],
                justify="start", align="start", wrap=False,
            ))
    mo.vstack(_blocks, align="start")
    return


@app.cell(hide_code=True)
def _(
    conv_compare_by_dropdown,
    conv_var_dropdown,
    dist_bands_checkbox,
    dpi_slider,
    m_slider,
    mo,
    sweep_params_row_ic,
):
    mo.vstack([
        mo.md(r"""## Guidance convergence — intensities × $n$"""),
        sweep_params_row_ic(),
        mo.hstack([conv_var_dropdown, conv_compare_by_dropdown, m_slider, dpi_slider], justify="start", align="center"),
        mo.hstack([dist_bands_checkbox], justify="start", align="center"),
    ], align="start")
    return


@app.cell(hide_code=True)
def _(LEVEL_VARS, config, mo):
    conv_var_dropdown = mo.ui.dropdown(LEVEL_VARS + ["mean_sea_level_pressure"], value=(config.VAR if config.VAR in LEVEL_VARS else "temperature"), label="variable: ")
    return (conv_var_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    conv_compare_by_dropdown = mo.ui.dropdown(["n", "levels"], value="n", label="compare by: ")
    return (conv_compare_by_dropdown,)


@app.cell(hide_code=True)
def _():
    conv_mm_cache = {}  # (rollout, pins, store, delta, var) -> masked series
    return (conv_mm_cache,)


@app.cell(hide_code=True)
def _(
    DELTAS,
    ROLLOUT_ID,
    base_sel,
    config,
    conv_mm_cache,
    delta_sel,
    dist_bands_checkbox,
    dpi_slider,
    mask_np,
    np,
    plt,
    stores,
    xr,
):
    # guidance-convergence chart, guidance-notebook grammar, per (intensity, n,
    # channel). Masked series are cached per (store, delta, variable) with ALL
    # levels reduced in one pass, so the level grids stay cheap.
    from src.ui.map import to_display_units as _tdu_c
    _maskda_c = None


    def _conv_series(_store_key, _i, _var):
        global _maskda_c
        _key = (ROLLOUT_ID, tuple(sorted(base_sel.items())), _store_key, _i, _var)
        if _key in conv_mm_cache:
            return conv_mm_cache[_key]
        _ds = stores[_store_key]
        if _ds is None or _var not in _ds:
            conv_mm_cache[_key] = None
            return None
        _da = _ds[_var]
        if _i is not None:
            _da = _da.sel({_k: _v for _k, _v in delta_sel(_i).items() if _k in _da.dims})
        if _store_key in ("ung", "gui_ung") and "t" in _da.dims:
            _da = _da.isel(t=-1)
        if _maskda_c is None:
            _maskda_c = xr.DataArray(mask_np, dims=("latitude", "longitude"),
                                     coords={"latitude": _ds.latitude, "longitude": _ds.longitude})
        conv_mm_cache[_key] = (_da * _maskda_c).sum(("latitude", "longitude")).compute()
        return conv_mm_cache[_key]


    def _conv_pick(_store_key, _i, _chan):
        _cl, _p, _v, _lv = _chan
        _s = _conv_series(_store_key, _i, _v)
        if _s is None:
            return None
        return np.asarray(_s.sel(level=_lv) if "level" in _s.dims else _s, dtype=float)


    def conv_fig(_i, n, m, _chan):
        _cl, _p, _v, _lv = _chan
        _z_all = _conv_pick("res", _i, _chan)
        _u_all = _conv_pick("vfs", _i, _chan)
        _g_all = _conv_pick("gui", _i, _chan)
        _d_all = _conv_pick("gui_det", _i, _chan)
        if _z_all is None or _u_all is None or _g_all is None or _d_all is None:
            return None
        _z_mm, _u_mm = _z_all[:, n], _u_all[:, n]        # (M, T)
        _gui_mm, _det_mm = _g_all[:, n], _d_all[:, n]     # (M,)
        _M_all, _T_len = _z_mm.shape
        _s_flow = np.linspace(1000, 1, _T_len) / 1000
        _h_flow = np.empty_like(_s_flow); _h_flow[:-1] = _s_flow[:-1] - _s_flow[1:]; _h_flow[-1] = _s_flow[-1]
        _hs = _h_flow / _s_flow
        from src.ui.comparison import residual_scaler as _rs
        _c_sc = _rs(_p, _v, _lv if _p == "level" else None)

        # realized guided step in masked-mean space (exact identity)
        _zT_mm = (_gui_mm - _det_mm) / _c_sc
        _z_next = np.concatenate([_z_mm[:, 1:], _zT_mm[:, None]], axis=1)
        _gu_mm = (_z_next - _z_mm) / _hs

        # target only at the TARGET channel: y = (1 + phi_n) * M(x_ung at n)
        _is_tgt = (_v == config.VAR) and (_p != "level" or _lv == config.LEVEL)
        _phi = float(DELTAS[_i][n]) if n < len(DELTAS[_i]) else 0.0
        if _is_tgt:
            _ung_all = _conv_pick("ung", None, _chan)
            _y = (1.0 + _phi) * _ung_all[:, n]
        else:
            _y = np.zeros(_M_all)

        _states = np.empty((_M_all, _T_len + 1))
        _states[:, :_T_len] = _det_mm[:, None] + _c_sc * _z_mm - _y[:, None]
        _states[:, _T_len] = _states[:, _T_len - 1] + _c_sc * _hs[-1] * _gu_mm[:, -1]
        _land_ung = _states[:, :_T_len] + _c_sc * _hs * _u_mm
        _pa_unit = ""
        if not _is_tgt:
            _states, _pa_unit = _tdu_c(_states, _v)
            _land_ung = _tdu_c(_land_ung, _v)[0]

        _xt = np.arange(_T_len + 1).astype(float)
        with plt.rc_context({"font.size": 8, "axes.titlesize": 10, "legend.fontsize": 7}):
            # not-yet-computed member / unrun point: render a flat-zero chart
            _not_computed = not np.isfinite(_states[m]).any()
            if _not_computed:
                _states = np.nan_to_num(_states, nan=0.0)
                _land_ung = np.nan_to_num(_land_ung, nan=0.0)
            _fig, _ax = plt.subplots(figsize=(8.6, 4.2), dpi=dpi_slider.value)
            if dist_bands_checkbox.value and _M_all > 1:
                _ax.fill_between(_xt, np.nanmin(_states, 0), np.nanmax(_states, 0),
                                 color="#B7950B", alpha=0.14, linewidth=0,
                                 label=f"state range, M={_M_all}")
            _bar_off = 0.16
            _flow_move = _land_ung[m] - _states[m, :_T_len]
            _gui_move = _states[m, 1:] - _land_ung[m]
            _ax.bar(_xt[1:] - _bar_off, _flow_move, bottom=_states[m, :_T_len], width=0.28,
                    color="#C0392B", alpha=0.35, edgecolor="#C0392B", linewidth=0.5, zorder=3,
                    label="flow move")
            _ax.bar(_xt[1:] + _bar_off, _gui_move, bottom=_land_ung[m], width=0.28,
                    color="#2E86C1", alpha=0.35, edgecolor="#2E86C1", linewidth=0.5, zorder=3,
                    label="guidance move")
            _ax.plot(_xt, _states[m], "-", color="#B7950B", alpha=0.5, linewidth=1.1, zorder=4)
            _ax.plot(_xt, _states[m], "o", color="#B7950B", markeredgecolor="white",
                     markeredgewidth=0.8, markersize=4.5, linestyle="none", zorder=7,
                     label=(r"state $-\,y_n$" if _is_tgt else "state"))
            for _ti in range(_T_len):
                _ax.plot([_xt[_ti], _xt[_ti + 1]], [_states[m, _ti], _land_ung[m, _ti]],
                         linestyle="--", linewidth=1.0, color="#800080", alpha=0.6,
                         zorder=5, label="_nolegend_")
            _ax.plot(_xt[1:], _land_ung[m], "o", color="#800080", markeredgecolor="white",
                     markeredgewidth=0.8, markersize=4.5, linestyle="none", zorder=6,
                     label="gui_ung landing")
            if _is_tgt:
                _ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
            _ax.set_xlim(-1.0, _T_len + 0.4)
            _ax.margins(y=0.06)
            _ax.annotate("noise", (_xt[0], _states[m, 0]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color="#666666")
            _ax.annotate("final", (_xt[-1], _states[m, -1]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color="#666666")
            if _is_tgt:
                _ax.annotate("target", (_xt[-1], 0.0), textcoords="offset points",
                             xytext=(8, 0), ha="left", va="center", fontsize=7,
                             color="#888888", annotation_clip=False)
            if float(np.nanmax(np.abs(_states[m]))) < 1e-9:
                _ax.set_ylim(-1.0, 1.0)
            if _not_computed:
                _ax.text(0.5, 0.5, "not computed yet", transform=_ax.transAxes,
                         ha="center", va="center", fontsize=11, color="#999999")
            _ax.set_xlabel("$t$")
            _ax.set_ylabel("masked mean − target" if _is_tgt else
                           (f"Mask-averaged value [{_pa_unit}]" if _pa_unit else "Mask-averaged value"))
            _ax.set_title(rf"$\phi$ peak {100 * max(DELTAS[_i]):+.3g}%  |  n={n}  |  {_cl}",
                          loc="left", fontweight="bold", fontsize=10, color="#222222")
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
            _ax.legend(loc="best", frameon=False)
            _ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
            _fig.tight_layout()
        plt.close(_fig)
        return _fig

    return (conv_fig,)


@app.cell(hide_code=True)
def _(
    N_STEPS,
    conv_compare_by_dropdown,
    conv_fig,
    conv_var_dropdown,
    delta_order,
    intensity_channels,
    m_slider,
    mo,
):
    # convergence grid: intensities horizontal; compare by n (blocks per n, one
    # row per channel of the selected variable) or by levels (blocks per channel)
    _cv = conv_var_dropdown.value
    _chans_c = intensity_channels(_cv)
    _blocks = []
    if conv_compare_by_dropdown.value == "n":
        for _n in range(N_STEPS):
            _blocks.append(mo.md(f"#### n = {_n}"))
            for _chan in _chans_c:
                _figs = [_f for _f in (conv_fig(_i, _n, m_slider.value, _chan) for _i in delta_order) if _f is not None]
                if _figs:
                    _blocks.append(mo.hstack(_figs, justify="start", align="start", wrap=False))
    else:
        for _chan in _chans_c:
            _blocks.append(mo.md(f"#### {_chan[0]}"))
            for _n in range(N_STEPS):
                _figs = [_f for _f in (conv_fig(_i, _n, m_slider.value, _chan) for _i in delta_order) if _f is not None]
                if _figs:
                    _blocks.append(mo.hstack(_figs, justify="start", align="start", wrap=False))
    mo.vstack(_blocks, align="start")
    return


@app.cell(hide_code=True)
def _(m_slider, mo):
    mo.vstack([mo.md(r"""## Guided states $x^{gui}$"""),
               mo.hstack([m_slider], justify="start")], align="start")
    return


@app.cell(hide_code=True)
def _(LEVEL_VARS, config, mo):
    # variable selector, 1:1 with the guidance cross checks (levels auto-selected)
    xgui_var_dropdown = mo.ui.dropdown(
        LEVEL_VARS + ["mean_sea_level_pressure"],
        value=(config.VAR if config.VAR in LEVEL_VARS + ["mean_sea_level_pressure"] else "temperature"),
        label="variable: ",
    )
    return (xgui_var_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    compare_by_dropdown = mo.ui.dropdown(["n", "levels"], value="n", label="compare by: ")
    return (compare_by_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    chart_mode_dropdown = mo.ui.dropdown(["abs", "diff"], value="abs", label="chart mode: ")
    return (chart_mode_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    # display labels -> internal mode values (guidance-notebook convention)
    xgui_norm_mode_dropdown = mo.ui.dropdown(
        {"own scale": "own_scale", "mask scale": "own_mask_scale", "same scale": "same_scale"},
        value="same scale", label="norm mode: ")
    return (xgui_norm_mode_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    xgui_zoom_scale_checkbox = mo.ui.checkbox(label="zoom scale")
    return (xgui_zoom_scale_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    charts_displayed_dropdown = mo.ui.dropdown(["guided-states", "guidance-effect", "guided-residual"], value="guided-states", label="charts displayed: ")
    return (charts_displayed_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    show_mask_switch = mo.ui.checkbox(label="show mask", value=True)
    return (show_mask_switch,)


@app.cell(hide_code=True)
def _(mo):
    contour_checkbox = mo.ui.checkbox(label="contours", value=True)
    return (contour_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    contour_levels_slider = mo.ui.slider(4, 30, step=2, value=24, label="levels: ", show_value=True, debounce=True)
    return (contour_levels_slider,)


@app.cell(hide_code=True)
def _(mo):
    contour_color_dropdown = mo.ui.dropdown(["dimgray", "white", "black"], value="black", label="contour color: ")
    return (contour_color_dropdown,)


@app.cell(hide_code=True)
def _(
    chart_mode_dropdown,
    charts_displayed_dropdown,
    compare_by_dropdown,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    dpi_slider,
    m_slider,
    mo,
    show_mask_switch,
    stats_scope_dropdown,
    sweep_params_row_ic,
    xgui_norm_mode_dropdown,
    xgui_var_dropdown,
    xgui_zoom_scale_checkbox,
    zoom_slider,
):
    mo.vstack([
        sweep_params_row_ic(),
        mo.hstack([xgui_var_dropdown, compare_by_dropdown, chart_mode_dropdown, m_slider], justify="start", align="center"),
        mo.hstack([xgui_norm_mode_dropdown, xgui_zoom_scale_checkbox, zoom_slider, dpi_slider, stats_scope_dropdown], justify="start", align="center"),
        mo.hstack([show_mask_switch, contour_checkbox, contour_levels_slider, contour_color_dropdown, charts_displayed_dropdown], justify="start", align="center"),
    ], align="start")
    return


@app.cell(hide_code=True)
def _(
    LEVELS_DICT,
    N_STEPS,
    SURFACE_PAIR,
    add_map_stats_ic,
    chart_mode_dropdown,
    charts_displayed_dropdown,
    compare_by_dropdown,
    config,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    cool_half_cmap,
    delta_labels,
    delta_order,
    delta_sel,
    dpi_slider,
    get_slices,
    m_slider,
    mask,
    mask_np,
    mo,
    np,
    show_mask_switch,
    stores,
    to_display_units,
    visualize_map,
    warm_half_cmap,
    white_zero_cmap,
    xgui_norm_mode_dropdown,
    xgui_var_dropdown,
    xgui_zoom_scale_checkbox,
    zoom_slider,
):
    # x^gui maps: intensities horizontal; vertical stacking per the compare-by
    # selector. Style 1:1 with the guidance notebook's absolute maps: white-zero
    # RdBu_r, zero-anchored (warm/cool half when single-signed), mask outline,
    # dimgray 0.4 contours, display units (K -> degC).
    _var = xgui_var_dropdown.value
    _m = int(m_slider.value)
    # channels: surface pair first (if any), then all pressure levels bottom -> top
    if _var == "mean_sea_level_pressure":
        _channels = [("sfc", "surface", "mean_sea_level_pressure", 0)]
    else:
        _channels = ([("surf", "surface", SURFACE_PAIR[_var], 0)] if _var in SURFACE_PAIR else []) \
            + [(f"L{_L}", "level", _var, _L) for _L in reversed(LEVELS_DICT["level"])]

    def _field(_i, _n, _p, _v, _lv):
        # _i = None -> the gui_ung reference column (equivalent to delta +0.0)
        if _i is None:
            _g0 = stores["gui_ung"].sel({_k: _v2 for _k, _v2 in delta_sel(delta_order[0]).items()
                                         if _k in stores["gui_ung"].dims})
            _g0 = _g0.isel(t=-1) if "t" in _g0.dims else _g0
            _a0 = np.asarray(get_slices(_g0, _p, _v, _lv))[_m][_n]
            if charts_displayed_dropdown.value == "guidance-effect":
                return np.zeros_like(np.asarray(_a0, dtype=float))  # gui_ung - gui_ung
            if charts_displayed_dropdown.value == "guided-residual":
                _dstore = "ung_det" if stores["ung_det"] is not None else "gui_det"
                _dd = stores[_dstore].sel({_k: _v2 for _k, _v2 in delta_sel(delta_order[0]).items()
                                           if _k in stores[_dstore].dims})
                _b0 = np.asarray(get_slices(_dd, _p, _v, _lv))[_m][_n]
                return to_display_units(np.asarray(_a0, dtype=float) - np.asarray(_b0, dtype=float),
                                        _v, is_difference=True)[0]
            return to_display_units(np.asarray(_a0, dtype=float), _v)[0]
        _a = np.asarray(get_slices(stores["gui"].sel(delta_sel(_i)), _p, _v, _lv))[_m][_n]
        def _other(_store):
            _o = stores[_store].sel({_k: _v2 for _k, _v2 in delta_sel(_i).items()
                                     if _k in stores[_store].dims})
            _o = _o.isel(t=-1) if "t" in _o.dims else _o
            return np.asarray(get_slices(_o, _p, _v, _lv))[_m][_n]

        if charts_displayed_dropdown.value == "guidance-effect":
            # guidance effect: r_n^gui - r_n^gui_ung (= x_n^gui - x_n^gui_ung)
            return to_display_units(np.asarray(_a, dtype=float) - np.asarray(_other("gui_ung"), dtype=float),
                                    _v, is_difference=True)[0]
        if charts_displayed_dropdown.value == "guided-residual":
            # r_n^gui = x_n^gui - x_n^gui_det (guidance inspect-states convention)
            return to_display_units(np.asarray(_a, dtype=float) - np.asarray(_other("gui_det"), dtype=float),
                                    _v, is_difference=True)[0]
        return to_display_units(np.asarray(_a, dtype=float), _v)[0]

    def _xgui_map(_d, _vmin, _vmax, _title):
        if _vmin < 0.0 < _vmax:
            _cmap, _center = white_zero_cmap, 0.0
        elif _vmin >= 0.0:
            _cmap, _center = warm_half_cmap, None
        else:
            _cmap, _center = cool_half_cmap, None
        _clon = 0.5 * (config.MASK_CORNERS[0] + config.MASK_CORNERS[1])
        _clat = 0.5 * (config.MASK_CORNERS[2] + config.MASK_CORNERS[3])
        if _clon > 180.0:
            _clon -= 360.0
        _res = visualize_map(
            _d, cmap=_cmap, vmin=_vmin, vmax=_vmax, center=_center,
            mask_2d=mask, show_mask=show_mask_switch.value, title=_title,
            contour_2d=_d if contour_checkbox.value else None,
            contour_levels=contour_levels_slider.value, contour_color=contour_color_dropdown.value,
            contour_linewidth=0.4, figsize=(7.0, 3.9), dpi=dpi_slider.value,
            zoom=zoom_slider.value, zoom_center_lon=_clon, zoom_center_lat=_clat,
            interactive=False,
        )
        return add_map_stats_ic(_res, _d)

    # chart mode: "abs" -> one row per channel; "diff" -> adjacent-level
    # differences (lower minus upper, story-notebook convention)
    if chart_mode_dropdown.value == "diff" and len(_channels) > 1:
        _specs = [
            ((lambda _i, _n, _a=_channels[_j], _b=_channels[_j + 1]:
              _field(_i, _n, _a[1], _a[2], _a[3]) - _field(_i, _n, _b[1], _b[2], _b[3])),
             f"{_channels[_j][0]} − {_channels[_j + 1][0]}")
            for _j in range(len(_channels) - 1)
        ]
    else:
        _specs = [
            ((lambda _i, _n, _c=_chan: _field(_i, _n, _c[1], _c[2], _c[3])), _chan[0])
            for _chan in _channels
        ]


    _lab_w = max((len(_l) for _f, _l in _specs), default=1)


    _core_reg = mask_np >= 0.5 * float(mask_np.max())


    def _zoom_lim_ic(_a):
        # zoom scale: restrict the limits computation to the visible zoom window
        if not xgui_zoom_scale_checkbox.value or int(zoom_slider.value) <= 1:
            return _a
        _z = int(zoom_slider.value)
        _clon = 0.5 * (config.MASK_CORNERS[0] + config.MASK_CORNERS[1])
        _clat = 0.5 * (config.MASK_CORNERS[2] + config.MASK_CORNERS[3])
        if _clon > 180.0:
            _clon -= 360.0
        _lat_e = np.linspace(90.0, -90.0, 122); _lat_cw = 0.5 * (_lat_e[:-1] + _lat_e[1:])
        _lon_e = np.linspace(-180.0, 180.0, 241); _lon_cw = 0.5 * (_lon_e[:-1] + _lon_e[1:])
        _zr = (((_lat_cw >= _clat - 90.0 / _z) & (_lat_cw <= _clat + 90.0 / _z))[:, None]
               & ((_lon_cw >= _clon - 180.0 / _z) & (_lon_cw <= _clon + 180.0 / _z))[None, :])
        return np.where(_zr, _a, np.nan)


    def _row(_n, _get, _lab):
        _mode = xgui_norm_mode_dropdown.value
        _cols = [None] + list(delta_order)  # gui_ung reference first (delta +0.0)
        _ds = [_get(_i, _n) for _i in _cols]
        # drop unrun sweep points (NaN-filled zarr regions); keep the zero-effect
        # gui_ung column even when it is identically zero
        _pairs = [(_d, _i) for _d, _i in zip(_ds, _cols)
                  if _i is None or np.isfinite(_d).any()]
        if not _pairs:
            return mo.hstack([mo.md(f"`{_lab:>{_lab_w}}`"), mo.md("_no data_")], justify="start")
        if _mode == "own_mask_scale":
            _pairs = [(np.where(_core_reg, _d, np.nan), _i) for _d, _i in _pairs]  # outside -> white
        _shared_lo = min(float(np.nanmin(_zoom_lim_ic(_d))) for _d, _i in _pairs)
        _shared_hi = max(float(np.nanmax(_zoom_lim_ic(_d))) for _d, _i in _pairs)
        _figs = [mo.md(f"`{_lab:>{_lab_w}}`")]
        for _d, _i in _pairs:
            if _mode == "same_scale":
                _vmin, _vmax = _shared_lo, _shared_hi
            else:
                _vmin = float(np.nanmin(_zoom_lim_ic(_d)))
                _vmax = float(np.nanmax(_zoom_lim_ic(_d)))
            _figs.append(_xgui_map(_d, _vmin, _vmax,
                                    "gui_ung (+0%)" if _i is None else delta_labels[_i]))
        return mo.hstack(_figs, justify="start", align="center")

    _blocks = []
    if compare_by_dropdown.value == "n":
        for _n in range(N_STEPS):
            _blocks.append(mo.md(f"### n = {_n}"))
            for _get, _lab in _specs:
                _blocks.append(_row(_n, _get, _lab))
    else:
        for _get, _lab in _specs:
            _blocks.append(mo.md(f"### {_lab}"))
            for _n in range(N_STEPS):
                _blocks.append(_row(_n, _get, f"n = {_n}"))
    mo.vstack(_blocks, align="start")
    return


@app.cell(hide_code=True)
def _(
    abs_checkbox,
    cross_var_dropdown,
    differential_checkbox,
    dist_bands_checkbox,
    dpi_slider,
    m_slider,
    mo,
    rank_by_dropdown,
    sweep_params_row_ic,
    top_k_slider,
    ylim_mode_dropdown,
):
    mo.vstack([
        mo.md(r"""## Cross checks — intensities × $n$"""),
        sweep_params_row_ic(),
        mo.hstack([cross_var_dropdown, top_k_slider, rank_by_dropdown], justify="start", align="center"),
        mo.hstack([differential_checkbox, abs_checkbox, ylim_mode_dropdown], justify="start", align="center"),
        mo.hstack([dist_bands_checkbox, m_slider, dpi_slider], justify="start", align="center"),
    ], align="start")
    return


@app.cell(hide_code=True)
def _(cross_masked_checkbox, masked_units_dropdown, mo):
    mo.hstack([cross_masked_checkbox, mo.md("### Masked averages"), masked_units_dropdown],
              justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(
    abs_checkbox,
    color_for_ic,
    cross_masked_checkbox,
    cross_traces_ic,
    cross_var_dropdown,
    delta_labels,
    delta_order,
    differential_checkbox,
    dist_bands_checkbox,
    dpi_slider,
    gt_ic_wnorm_ic,
    m_slider,
    masked_units_dropdown,
    mo,
    np,
    plot_trajectory,
    rank_by_dropdown,
    red_for_ic,
    top_k_slider,
    xnorm_ic,
    xr,
    ylim_mode_dropdown,
):
    # masked averages across intensities: guided − gui_ung (0-anchored) and the
    # ABSOLUTE guided levels (gt-anchored at n=0), with the units selector
    if cross_masked_checkbox.value:
        _ctl = dict(k=top_k_slider.value, diff=differential_checkbox.value,
                    absv=abs_checkbox.value, bands=dist_bands_checkbox.value,
                    rank_by=rank_by_dropdown.value, var=cross_var_dropdown.value)
        _mi = int(m_slider.value)

        def _conv_abs(_ds):
            if masked_units_dropdown.value != "data":
                return _ds
            _d = xnorm_ic.denormalize(_ds)
            return _d.map(lambda _da: _da - 273.15 if _da.name in ("temperature", "2m_temperature") else _da)

        def _conv_diff(_ds):
            if masked_units_dropdown.value != "data":
                return _ds
            return xnorm_ic.denormalize(_ds) - xnorm_ic.denormalize(xr.zeros_like(_ds))

        def _n0v(_k):
            _n0 = _conv_abs(gt_ic_wnorm_ic)
            if " L" in _k:
                _vv, _lv = _k.rsplit(" L", 1)
                return float(_n0[_vv].sel(level=int(_lv)))
            return float(_n0[_k])

        _blocks = []
        for _title, _cube, _mode in [
            ("Masked average: guided − gui_ung", "gui_gui_ung_wnorm", "diff"),
            ("Masked average: guided", "gui_wnorm", "abs"),
        ]:
            _figs = []
            for _i in delta_order:
                _red_i = red_for_ic(_i)
                if _cube not in _red_i:
                    continue
                _ds = _conv_diff(_red_i[_cube]) if _mode == "diff" else _conv_abs(_red_i[_cube])
                _tr, _bands = cross_traces_ic(_ds, "mean", _mi, **_ctl)
                _anchor = (lambda _k: 0.0) if _mode == "diff" else _n0v
                _tr = {_k: np.concatenate([[_anchor(_k)], np.asarray(_v, float)]) for _k, _v in _tr.items()}
                if _bands:
                    _bands = {_k: (np.concatenate([[_anchor(_k)], np.asarray(_lo, float)]),
                                   np.concatenate([[_anchor(_k)], np.asarray(_hi, float)]))
                              for _k, (_lo, _hi) in _bands.items()}
                _figp = plot_trajectory(
                    _tr, title=_title, subtitle=delta_labels[_i], xlabel="$n$",
                    step=None, color_map=color_for_ic(_tr), bands=_bands,
                    figsize=(7.6, 4.6), dpi=dpi_slider.value,
                    prepend_zero=False, start_index=0, mirror_right_axis=True,
                )
                if hasattr(_figp, "axes") and _figp.axes:
                    _axp = _figp.axes[0]
                    _nst = max((len(_v) for _v in _tr.values()), default=1)
                    _axp.set_xlim(-0.4, _nst - 1 + 0.4)
                    _axp.set_xticks(np.arange(0, _nst))
                _figs.append(_figp)
            if ylim_mode_dropdown.value == "shared" and _figs:
                _lo = min(_f.axes[0].get_ylim()[0] for _f in _figs if hasattr(_f, "axes") and _f.axes)
                _hi = max(_f.axes[0].get_ylim()[1] for _f in _figs if hasattr(_f, "axes") and _f.axes)
                for _f in _figs:
                    for _axx in getattr(_f, "axes", []):
                        _axx.set_ylim(_lo, _hi)
            _blocks.append(mo.hstack(_figs, justify="start", align="start"))
        masked_cross_block = mo.vstack(_blocks, align="start")
    else:
        masked_cross_block = None
    masked_cross_block
    return


@app.cell(hide_code=True)
def _(cross_norms_checkbox, mo):
    mo.hstack([cross_norms_checkbox, mo.md("### Norms")], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(
    abs_checkbox,
    color_for_ic,
    cross_norms_checkbox,
    cross_traces_ic,
    cross_var_dropdown,
    delta_labels,
    delta_order,
    differential_checkbox,
    dist_bands_checkbox,
    dpi_slider,
    m_slider,
    mo,
    np,
    plot_trajectory,
    rank_by_dropdown,
    red_for_ic,
    top_k_slider,
    ylim_mode_dropdown,
):
    # norm charts over n, one column per intensity
    if cross_norms_checkbox.value:
        _ctl = dict(k=top_k_slider.value, diff=differential_checkbox.value,
                    absv=abs_checkbox.value, bands=dist_bands_checkbox.value,
                    rank_by=rank_by_dropdown.value, var=cross_var_dropdown.value)
        _mi = int(m_slider.value)
        _ROWS = [
            ("Gradient norm", "grads_l2", "l2"),
            ("Vector field norm — ung step", "vfs_l2", "l2"),
            ("Vector field norm — gui step", "gui_vfs_l2", "l2"),
        ]
        _blocks = []
        for _title, _cube, _agg in _ROWS:
            _figs = []
            for _i in delta_order:
                _red_i = red_for_ic(_i)
                if _cube not in _red_i:
                    continue
                _tr, _bands = cross_traces_ic(_red_i[_cube], _agg, _mi, **_ctl)
                _figp = plot_trajectory(
                    _tr, title=f"{_title}", subtitle=delta_labels[_i], xlabel="$n$",
                    step=None, color_map=color_for_ic(_tr), bands=_bands,
                    figsize=(7.6, 4.6), dpi=dpi_slider.value,
                    prepend_zero=False, start_index=1, mirror_right_axis=True,
                )
                if hasattr(_figp, "axes") and _figp.axes:
                    _axp = _figp.axes[0]
                    _nst = max((len(_v) for _v in _tr.values()), default=1)
                    _axp.set_xlim(-0.4, _nst + 0.4)
                    _axp.set_xticks(np.arange(0, _nst + 1))
                _figs.append(_figp)
            if ylim_mode_dropdown.value == "shared" and _figs:
                _lo = min(_f.axes[0].get_ylim()[0] for _f in _figs if hasattr(_f, "axes") and _f.axes)
                _hi = max(_f.axes[0].get_ylim()[1] for _f in _figs if hasattr(_f, "axes") and _f.axes)
                for _f in _figs:
                    for _axx in getattr(_f, "axes", []):
                        _axx.set_ylim(_lo, _hi)
            _blocks.append(mo.hstack(_figs, justify="start", align="start"))
        norms_cross_block = mo.vstack(_blocks, align="start")
    else:
        norms_cross_block = None
    norms_cross_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Latent trajectories (PCA)

    Guided clean-pred trajectories $\hat{x}_t$ (mask-bbox pixels, flattened) in a fixed PCA
    frame of daily ERA5 states (gray cloud). **Solid**: guided; **dashed**: the run's same-seed
    `gui_ung` twin; **★** GT at the valid day; **○** the independent unguided rollout's final.
    Pick what to **compare** (intensity, $n$, …) — compared values fan out as fading greens;
    everything else is pinned.
    """)
    return


@app.cell(hide_code=True)
def _(N_STEPS, ROLLOUT_ID, get_gt_rollout, np, residual_scaler):
    # ===== latent trajectories: rollout records, bbox latents, PCA frame =====
    from datetime import datetime as _lt_datetime, timedelta as _lt_timedelta
    from src.ui.comparison import (
        channel as _lt_channel,
        clean_pred_branches as _lt_branches,
        clean_pred_trajectory as _lt_trajf,
        gt_states as _lt_gt_states,
        load_rollout as _lt_load_rollout,
        open_store as _lt_open_store,
        select_point as _lt_select_point,
        sweep_points as _lt_sweep_points,
    )

    _lt_dir, _lt_cfg, lt_sweep_values, _lt_records, _lt_mask = _lt_load_rollout(ROLLOUT_ID)
    lt_points = _lt_sweep_points(lt_sweep_values, _lt_records)
    _lt_VAR, _lt_LEVEL = _lt_cfg["VAR"], _lt_cfg["LEVEL"]
    _lt_start = _lt_datetime.fromisoformat(_lt_cfg["START_TS"])
    _lt_c = residual_scaler(_lt_cfg["PARTITION"], _lt_VAR, _lt_LEVEL)

    # bbox footprint of the mask at half max -> flattened feature vectors
    _lt_rows, _lt_cols = np.where(np.asarray(_lt_mask) >= 0.5 * float(np.asarray(_lt_mask).max()))
    _LT_BBOX = (slice(int(_lt_rows.min()), int(_lt_rows.max()) + 1),
                slice(int(_lt_cols.min()), int(_lt_cols.max()) + 1))


    def lt_bbox_latent(_field):
        _field = np.asarray(_field, dtype=float)
        return _field[..., _LT_BBOX[0], _LT_BBOX[1]].reshape(*_field.shape[:-2], -1)


    def _lt_gt_cloud(_days_back):
        # daily GT states ending just before the rollout start (shrinks if short)
        _err = None
        for _days in (_days_back, _days_back // 2, _days_back // 4, 7):
            try:
                _da = _lt_channel(get_gt_rollout(_days, _lt_start - _lt_timedelta(days=_days + 2))[_lt_VAR], _lt_cfg)
                return lt_bbox_latent(_da.values)
            except Exception as _e:
                _err = _e
        raise RuntimeError(f"no GT cloud loadable: {_err}")


    lt_pca_cloud = _lt_gt_cloud(60)
    _lt_mu = lt_pca_cloud.mean(axis=0)
    _lt_u, _lt_sv, _lt_vt = np.linalg.svd(lt_pca_cloud - _lt_mu, full_matrices=False)
    _lt_basis = _lt_vt[:3].T
    lt_pca_project = lambda _x: (_x - _lt_mu) @ _lt_basis
    lt_pca_evr = (_lt_sv ** 2 / (_lt_sv ** 2).sum())[:3]
    lt_pca_cloud_proj = lt_pca_project(lt_pca_cloud)
    lt_pca_targets = lt_bbox_latent(np.asarray(
        _lt_channel(_lt_gt_states(_lt_cfg)[_lt_VAR], _lt_cfg).isel(time=slice(1, None)), dtype=float))
    try:
        _lt_ung = _lt_channel(_lt_open_store(_lt_dir, "ung", _lt_VAR), _lt_cfg).isel(m=0)
        if "t" in _lt_ung.dims:
            _lt_ung = _lt_ung.isel(t=-1)  # independent unguided FINAL states
        lt_pca_ung_finals = lt_bbox_latent(np.asarray(_lt_ung, dtype=float))
    except FileNotFoundError:
        lt_pca_ung_finals = None


    def lt_guided_latents(_sel, _m, _n):
        _t = _lt_trajf(_lt_dir, _lt_records, _sel, _m, _n, _lt_VAR, _lt_c, level=_lt_LEVEL)
        return None if _t is None else lt_bbox_latent(_t)


    def lt_branch_latents(_sel, _m, _n):
        _p = _lt_branches(_lt_dir, _lt_records, _sel, _m, _n, _lt_VAR, _lt_c, level=_lt_LEVEL)
        return None if _p is None else (lt_bbox_latent(_p[0]), lt_bbox_latent(_p[1]))


    def lt_twin_latents(_sel, _m, _n):
        _tw = _lt_channel(_lt_select_point(_lt_open_store(_lt_dir, "gui_ung", _lt_VAR), _sel), _lt_cfg)
        return lt_bbox_latent(np.asarray(_tw.isel(m=_m, n=_n), dtype=float))


    def lt_delta_of(_sel):
        return np.asarray(lt_sweep_values["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], dtype=float)[:N_STEPS]


    print(f"cloud: {lt_pca_cloud.shape} | EVR(3): {float(lt_pca_evr.sum()):.3f} | runs: {len(lt_points)}")
    return (
        lt_branch_latents,
        lt_delta_of,
        lt_guided_latents,
        lt_pca_cloud_proj,
        lt_pca_evr,
        lt_pca_project,
        lt_pca_targets,
        lt_pca_ung_finals,
        lt_points,
        lt_twin_latents,
    )


@app.cell(hide_code=True)
def _(lt_points, mo):
    _axis_vals = {}
    for _sel in lt_points.values():
        for _k, _v in _sel.items():
            _axis_vals.setdefault(_k, {})[str(_v)] = _v
    lt_varying_axes = [_k for _k, _vs in _axis_vals.items() if len(_vs) > 1]
    lt_compare_dropdown = mo.ui.dropdown(
        lt_varying_axes + ["n", "gui_ung"],
        value=("GUIDANCE_DELTA" if "GUIDANCE_DELTA" in lt_varying_axes
               else (lt_varying_axes + ["n", "gui_ung"])[0]),
        label="compare: ",
    )
    return (lt_compare_dropdown,)


@app.cell(hide_code=True)
def _(lt_points, mo):
    _axis_vals = {}
    for _sel in lt_points.values():
        for _k, _v in _sel.items():
            _axis_vals.setdefault(_k, {})[str(_v)] = _v
    lt_pin_dropdowns = mo.ui.dictionary({
        _k: mo.ui.dropdown(list(_vs), value=list(_vs)[0], label=f"{_k}: ")
        for _k, _vs in _axis_vals.items() if len(_vs) > 1
    })
    return (lt_pin_dropdowns,)


@app.cell(hide_code=True)
def _(N_STEPS, mo):
    lt_n_slider = mo.ui.slider(1, N_STEPS, step=1, value=min(2, N_STEPS), label="n: ", show_value=True, debounce=True)
    return (lt_n_slider,)


@app.cell(hide_code=True)
def _(mo):
    lt_elev_slider = mo.ui.slider(0, 90, step=5, value=25, label="elev: ", show_value=True, debounce=True)
    return (lt_elev_slider,)


@app.cell(hide_code=True)
def _(mo):
    lt_azim_slider = mo.ui.slider(-180, 180, step=5, value=-60, label="azim: ", show_value=True, debounce=True)
    return (lt_azim_slider,)


@app.cell(hide_code=True)
def _(mo):
    lt_zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories")
    return (lt_zoom_traj_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    lt_zoom_pad_slider = mo.ui.slider(-0.4, 2.0, step=0.05, value=0.12, label="zoom pad: ", show_value=True, debounce=True)
    return (lt_zoom_pad_slider,)


@app.cell(hide_code=True)
def _(mo):
    lt_lw_plain_slider = mo.ui.slider(0.1, 5.0, step=0.1, value=1.5, label="solid lw: ", show_value=True, debounce=True)
    return (lt_lw_plain_slider,)


@app.cell(hide_code=True)
def _(mo):
    lt_lw_dash_slider = mo.ui.slider(0.1, 4.0, step=0.1, value=1.0, label="dashed lw: ", show_value=True, debounce=True)
    return (lt_lw_dash_slider,)


@app.cell(hide_code=True)
def _(
    N_STEPS,
    delta_labels,
    dpi_slider,
    lt_azim_slider,
    lt_branch_latents,
    lt_compare_dropdown,
    lt_delta_of,
    lt_elev_slider,
    lt_guided_latents,
    lt_lw_dash_slider,
    lt_lw_plain_slider,
    lt_n_slider,
    lt_pca_cloud_proj,
    lt_pca_evr,
    lt_pca_project,
    lt_pca_targets,
    lt_pca_ung_finals,
    lt_pin_dropdowns,
    lt_points,
    lt_twin_latents,
    lt_zoom_pad_slider,
    lt_zoom_traj_checkbox,
    m_slider,
    mo,
    np,
    plt,
    sweep_params_row_ic,
):
    # ===== 3D PCA trajectories (ported from latent_trajectories.py) =====
    _m = int(m_slider.value)
    _cmp = lt_compare_dropdown.value
    _pins = lt_pin_dropdowns.value


    def _fmt_val(_k, _v):
        if _k == "GUIDANCE_DELTA":
            try:
                return delta_labels.get(int(_v), f"δ#{_v}")
            except (TypeError, ValueError):
                return f"δ#{_v}"
        return f"{_k}={_v}"


    def _sort_key(_v):
        try:
            return (0, float(_v), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(_v))


    _gcmap = plt.get_cmap("Greens")


    def _fade(_i, _total):
        return _gcmap(0.7) if _total <= 1 else _gcmap(0.30 + 0.65 * _i / (_total - 1))


    def _match(_sel, _skip):
        return all(str(_sel[_k]) == _pins[_k] for _k in _pins if _k != _skip)


    # curves: (label, color, guided latents, twin latents, n index)
    _curves = []
    _branch_pair = None
    if _cmp == "n":
        _sel = next((_s for _l, _s in lt_points.items() if _match(_s, None)), None)
        if _sel is not None:
            _ns = [_i for _i in range(N_STEPS) if lt_delta_of(_sel)[_i] != 0]
            for _j, _ni in enumerate(_ns):
                _g = lt_guided_latents(_sel, _m, _ni)
                if _g is not None and np.isfinite(_g).any():
                    _curves.append((f"{_ni + 1}", _fade(_j, len(_ns)), _g, lt_twin_latents(_sel, _m, _ni), _ni))
    elif _cmp == "gui_ung":
        # one pinned run vs its unguided (gui_ung) counterpart, both first-class
        _n = lt_n_slider.value - 1
        _sel = next((_s for _l, _s in lt_points.items() if _match(_s, None)), None)
        if _sel is not None:
            _branch_pair = lt_branch_latents(_sel, _m, _n)
            _tw = lt_twin_latents(_sel, _m, _n)
            if _tw is not None and np.isfinite(_tw).any():
                _curves.append(("gui_ung", "#1F77B4", _tw, None, _n))
    else:
        _n = lt_n_slider.value - 1
        _matched = sorted(
            ((_l, _s) for _l, _s in lt_points.items() if _match(_s, _cmp)),
            key=lambda ls: _sort_key(ls[1].get(_cmp)),
        )
        for _j, (_l, _s) in enumerate(_matched):
            _g = lt_guided_latents(_s, _m, _n)
            if _g is not None and np.isfinite(_g).any():
                _curves.append((_fmt_val(_cmp, _s.get(_cmp)), _fade(_j, len(_matched)), _g, lt_twin_latents(_s, _m, _n), _n))


    def _make_fig(_branch_style):
        """One complete 3D figure; _branch_style: None | "chain" | "pingpong"."""
        _fig = plt.figure(figsize=(24, 16), dpi=dpi_slider.value)
        _ax3 = _fig.add_subplot(projection="3d")
        _ax3.scatter(*lt_pca_cloud_proj.T, color="#BBBBBB", s=14, alpha=0.5, depthshade=False,
                     label=f"ERA5 cloud ({lt_pca_cloud_proj.shape[0]} days)")
        _groups = []
        for _label, _color, _g, _tw, _ni in _curves:
            _gp = lt_pca_project(_g)
            _ax3.plot(*_gp.T, "-", color=_color, linewidth=1.2 * lt_lw_plain_slider.value, alpha=0.9, label=_label)
            _ax3.scatter(*_gp.T, s=np.linspace(8, 46, len(_gp)), color=_color, alpha=0.9, depthshade=False)
            _ax3.scatter(*_gp[-1], marker=("o" if _label == "gui_ung" else "D"), s=100, color="black", edgecolors="white", depthshade=False)
            _ax3.text(*_gp[-1], f"    {_label}", color="black", fontsize=9, fontweight="bold")
            if _tw is not None:
                _up = lt_pca_project(_tw)
                _ax3.plot(*_up.T, "--", color=_color, linewidth=lt_lw_dash_slider.value, alpha=0.6, label="_nolegend_")
                _ax3.scatter(*_up[-1], marker="o", s=100, color="black", alpha=0.85, edgecolors="white", depthshade=False)
                _ax3.text(*_up[-1], f"    {_label} (gui_ung)", color="black", fontsize=9, fontweight="bold")
                _groups.append(_up)
            _groups.append(_gp)
        if _cmp == "gui_ung" and _branch_pair is not None:
            _bp, _kp = lt_pca_project(_branch_pair[0]), lt_pca_project(_branch_pair[1])
            _gcol = "#B05C3A"  # terracotta for the gui branch elements
            if _branch_style == "pingpong":
                # TRUE event path: kick (base_t -> gui_t) solid green;
                # pull-back (gui_t -> base_{t+1}) dotted grey
                for _t in range(len(_bp)):
                    _seg = np.stack([_bp[_t], _kp[_t]])
                    _ax3.plot(*_seg.T, "-", color=_gcol, linewidth=lt_lw_plain_slider.value, alpha=0.9,
                              label=("gui" if _t == 0 else "_nolegend_"))
                    if _t + 1 < len(_bp):
                        _seg = np.stack([_kp[_t], _bp[_t + 1]])
                        _ax3.plot(*_seg.T, ":", color="#800080", linewidth=0.7 * lt_lw_dash_slider.value, alpha=0.8, label="_nolegend_")
                _gpath = np.vstack([_bp[:1], _kp])
                _ax3.plot(*_gpath.T, "-", color="#2CA02C", linewidth=lt_lw_plain_slider.value,
                          alpha=0.75, label="_nolegend_")
            else:
                # gui chain: start -> gui_0 -> gui_1 -> ...; dotted grey dead-end
                # branches gui_t -> base_{t+1} (counterfactual pull-backs)
                _gpath = np.vstack([_bp[:1], _kp])
                _ax3.plot(*_gpath.T, "-", color="#2CA02C", linewidth=lt_lw_plain_slider.value, alpha=0.9, label="gui")
                for _t in range(len(_bp) - 1):
                    _seg = np.stack([_kp[_t], _bp[_t + 1]])
                    _ax3.plot(*_seg.T, ":", color="#800080", linewidth=0.7 * lt_lw_dash_slider.value, alpha=0.8, label="_nolegend_")
            _ax3.scatter(*_kp.T, s=np.linspace(8, 46, len(_kp)), color=_gcol, alpha=0.9, depthshade=False)
            _ax3.scatter(*_bp.T, s=14, color="#800080", alpha=0.9, depthshade=False, label="gui_ung_over_t")
            _ax3.scatter(*_kp[-1], marker="D", s=100, color="black", edgecolors="white", depthshade=False)
            _ax3.text(*_kp[-1], "    gui", color="black", fontsize=9, fontweight="bold")
            _groups.extend([_bp, _kp])
        # shared starting latent (all curves at one n start from the same seed)
        if _curves:
            _sp = lt_pca_project(_curves[0][2])[0]
            _ax3.scatter(*_sp, marker="o", s=40, color="black", depthshade=False)
            _ax3.text(*_sp, "    start", color="black", fontsize=9, fontweight="bold")
        for _ni in sorted({_c[4] for _c in _curves}):
            _tp = lt_pca_project(lt_pca_targets[_ni])
            _ax3.scatter(*_tp, marker="*", s=100, color="black", depthshade=False)
            _ax3.text(*_tp, "    gt", color="black", fontsize=9, fontweight="bold")
            _groups.append(_tp[None, :])
            if lt_pca_ung_finals is not None:
                _op = lt_pca_project(lt_pca_ung_finals[_ni])
                _ax3.scatter(*_op, marker="o", s=100, facecolors="none", edgecolors="#111111",
                             linewidths=2.0, depthshade=False)
                _ax3.text(*_op, "    ung", color="#111111", fontsize=9, fontweight="bold")
                _groups.append(_op[None, :])
        if lt_zoom_traj_checkbox.value and _groups:
            _pts = np.vstack(_groups)
            _lo, _hi = _pts.min(axis=0), _pts.max(axis=0)
            _pad = lt_zoom_pad_slider.value * float((_hi - _lo).max())
            _ax3.set_xlim(_lo[0] - _pad, _hi[0] + _pad)
            _ax3.set_ylim(_lo[1] - _pad, _hi[1] + _pad)
            _ax3.set_zlim(_lo[2] - _pad, _hi[2] + _pad)
        _ax3.set_xlabel("PC1"); _ax3.set_ylabel("PC2"); _ax3.set_zlabel("PC3")
        _pin_desc = ", ".join(_fmt_val(_k, _pins[_k]) for _k in _pins if _k != _cmp)
        _style_tag = f"  |  {_branch_style}" if _branch_style else ""
        _ax3.set_title(
            f"compare {_cmp}{_style_tag}  |  {_pin_desc}  (m={_m}, EVR={lt_pca_evr.sum():.0%})",
            loc="left",
        )
        _ax3.view_init(elev=lt_elev_slider.value, azim=lt_azim_slider.value)
        _ax3.legend(loc="upper left", fontsize=8)
        plt.close(_fig)
        return _fig


    _pca_fig = _make_fig("chain" if _cmp == "gui_ung" else None)
    _pca_fig2 = _make_fig("pingpong") if (_cmp == "gui_ung" and _branch_pair is not None) else None

    mo.vstack(
        [
            sweep_params_row_ic(),
            mo.hstack(
                [lt_compare_dropdown] + [lt_pin_dropdowns[_k] for _k in lt_pin_dropdowns.value if _k != _cmp]
                + ([lt_n_slider] if _cmp != "n" else []) + [m_slider],
                justify="start", align="start",
            ),
            mo.hstack(
                [lt_elev_slider, lt_azim_slider, dpi_slider, lt_lw_plain_slider, lt_lw_dash_slider,
                 lt_zoom_traj_checkbox, lt_zoom_pad_slider],
                justify="start", align="start",
            ),
            mo.hstack([_pca_fig] + ([_pca_fig2] if _pca_fig2 is not None else []),
                      justify="start", align="start"),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
