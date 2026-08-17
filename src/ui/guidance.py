import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _(
    mo,
    notebook_mode,
    reductions_complete,
    reductions_grid_matches,
    rollout_id,
):
    # Reductions store: precomputed spatial reductions for the analysis charts.
    # Built once for the whole sweep grid -> every slide reads tiny in-memory arrays.
    if notebook_mode == "analyze_rollout":
        mo.stop(rollout_id is None)   # no rollout -> gated with a hint in the config cell
        build_reductions_button = mo.ui.run_button(label="build reductions")
        if not reductions_grid_matches(rollout_id):
            _rs_msg = "⚠️ to build"
        elif not reductions_complete(rollout_id):
            _rs_msg = "🟡 building"
        else:
            _rs_msg = "✅ built"
        reductions_widget = mo.hstack(
            [build_reductions_button, mo.md(_rs_msg)], justify="start", align="center"
        )
    else:
        build_reductions_button = None
        reductions_widget = None
    return build_reductions_button, reductions_widget


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch
    import numpy as np
    import xarray as xr
    import dask
    import matplotlib.pyplot as plt
    plt.style.use("petroff10")
    import matplotlib.colors as mcolors
    from datetime import datetime, timedelta, date, time
    import calendar
    import cv2
    from geoarches.paths import STATS_PATH


    return STATS_PATH, cv2, dask, mcolors, mo, np, plt, torch, xr


@app.cell
def _():
    from src.paths import ROLLOUTS
    from src.rollout_config import MASK_MODES, RolloutConfig, GUIDANCE_METHODS, GUI_REFS, GUIDANCE_METHOD_HYPERS
    from geoarches.lightning_modules.guided_diffusion import A_T_MODES, alpha_t_profile as a_t_profile
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.helpers import max_day, get_timestamp_from_sliders
    from src.ui.map import visualize_map, visualize_mask_3d, to_display_units
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_trajectories import plot_trajectories

    from src.utils import get_var_idx, get_level_idx
    from src.utils import get_now_timestamp, ensure_rollout_dir, get_rollout_dir, find_era5_input
    from src.utils import get_timestamps, get_N_timestamps, get_N_slices as base_get_N_slices, get_slices as base_get_slices, get_gt_rollout
    from src.utils import (
        dump_json, get_rollout_ids, get_rollout, get_sweep_dict, get_config, sweep_coord_label
    )
    from src.utils import get_w_star, get_guidance_schedule
    from src.run import iter_sweeps
    from src.reductions import (reductions_grid_matches, reductions_complete, load_reductions, build_reductions_store, compute_reductions_for_sweep, reductions_exist)
    from src.schedules import N_schedule, delta_schedule
    from src.normalization import XarrayNormalizer
    from src.spectrum import power_spectrum, log_spectral_distance, spectral_bias

    from src.mask import get_masked_mean, get_mask_2d, get_mask_center, get_great_circle_field, get_bbox_mask


    return (
        A_T_MODES,
        GUIDANCE_METHODS,
        GUIDANCE_METHOD_HYPERS,
        GUI_REFS,
        LEVELS_DICT,
        MASK_MODES,
        RolloutConfig,
        VARIABLES_DICT,
        XarrayNormalizer,
        a_t_profile,
        base_get_N_slices,
        base_get_slices,
        build_reductions_store,
        compute_reductions_for_sweep,
        dump_json,
        ensure_rollout_dir,
        find_era5_input,
        get_N_timestamps,
        get_config,
        get_gt_rollout,
        get_guidance_schedule,
        get_level_idx,
        get_mask_2d,
        get_mask_center,
        get_masked_mean,
        get_now_timestamp,
        get_rollout,
        get_rollout_dir,
        get_rollout_ids,
        get_sweep_dict,
        get_timestamp_from_sliders,
        get_var_idx,
        get_w_star,
        iter_sweeps,
        load_reductions,
        max_day,
        plot_trajectories,
        plot_trajectory,
        reductions_complete,
        reductions_grid_matches,
        sweep_coord_label,
        to_display_units,
        visualize_map,
    )


@app.cell
def _(mask_region, np, stats_scope_dropdown, zoom_centers, zoom_slider):
    def add_map_stats(map_obj, stats_arr):
        """min/max (left) and mean/std (right) stamped at title height.

        Scope follows stats_scope_dropdown: "mask" = mask core (mask_region),
        "zoom" = current zoom window (regardless of the zoom-scale checkbox),
        "full" = whole globe. Independent of the colorbar's norm mode."""
        if map_obj is None or not isinstance(map_obj, tuple):
            return map_obj  # interactive widget or skipped map
        _fig, _ax = map_obj
        _a = np.asarray(stats_arr, dtype=float)
        if stats_scope_dropdown.value == "mask":
            _a = np.where(mask_region, _a, np.nan)
        elif stats_scope_dropdown.value == "zoom" and int(zoom_slider.value) > 1:
            _z = int(zoom_slider.value)
            _lon_span, _lat_span = 360.0 / _z, 180.0 / _z
            _lo_min = max(-180.0, zoom_centers[0] - _lon_span / 2)
            _lo_max = min(180.0, zoom_centers[0] + _lon_span / 2)
            _la_min = max(-90.0, zoom_centers[1] - _lat_span / 2)
            _la_max = min(90.0, zoom_centers[1] + _lat_span / 2)
            _lat_e = np.linspace(90.0, -90.0, 122); _lat_c = 0.5 * (_lat_e[:-1] + _lat_e[1:])
            _lon_e = np.linspace(-180.0, 180.0, 241); _lon_c = 0.5 * (_lon_e[:-1] + _lon_e[1:])
            _zr = (((_lat_c >= _la_min) & (_lat_c <= _la_max))[:, None]
                   & ((_lon_c >= _lo_min) & (_lon_c <= _lo_max))[None, :])
            _a = np.where(_zr, _a, np.nan)
        # "full": unrestricted
        if np.isfinite(_a).any():
            _left = f"min = {np.nanmin(_a):.3g} | max = {np.nanmax(_a):.3g}"
            _right = f"mean = {np.nanmean(_a):.3g} | std = {np.nanstd(_a):.3g}"
        else:
            _left, _right = "all NaN", ""
        _ax.set_title(_left, loc="left", fontsize=8)
        _ax.set_title(_right, loc="right", fontsize=8)
        return map_obj

    return (add_map_stats,)


@app.function
def get_label(name: str, n_options: int) -> str:
    # sliders with a single option render without a draggable track; pad the
    # label with non-breaking spaces so rows of sliders keep their labels aligned
    return f"{name}:\u00A0\u00A0" if n_options <= 1 else f"{name}: "


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guidance Debugger
    """)
    return


@app.cell
def _(
    cross_section_checkbox,
    flow_section_checkbox,
    inspect_section_checkbox,
    mask_section_checkbox,
    mo,
    trajectories_section_checkbox,
):
    # section toggles (the SAME checkbox elements as in the section headers)
    mo.md(
        f"""
    - {mask_section_checkbox} Mask
    - {inspect_section_checkbox} Inspect states
    - {trajectories_section_checkbox} Trajectories
    - {flow_section_checkbox} Flow analysis
    - {cross_section_checkbox} Cross variable checks
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment
    """)
    return


@app.cell
def _(get_rollout_dir, mo, rollout_id):
    # experiment intent note: data/rollouts/<id>/what-it-tests.txt (hand-written)
    if rollout_id is not None and (get_rollout_dir(rollout_id) / "what-it-tests.txt").exists():
        _wit = (get_rollout_dir(rollout_id) / "what-it-tests.txt").read_text().strip()
    else:
        _wit = "not specified"
    mo.md(f"What it tests: *{_wit}*")
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.run_button(label="refresh")
    return (refresh_button,)


@app.cell
def _(mo, notebook_mode_dropdown, reductions_widget, refresh_button):
    mo.vstack(
        [
            *([reductions_widget] if reductions_widget is not None else []),
            mo.hstack([notebook_mode_dropdown, refresh_button], justify="start", align="start"),
        ],
        align="start",
    )
    return


@app.cell
def _(mo):
    NOTEBOOK_MODES = ["unguided_rollout", "guided_rollout", "analyze_rollout"]
    notebook_mode_dropdown = mo.ui.dropdown(
        options=NOTEBOOK_MODES,
        value=NOTEBOOK_MODES[0],
        label="notebook_mode: ",
    )
    return (notebook_mode_dropdown,)


@app.cell
def _(get_rollout_ids, mo, notebook_mode_dropdown, refresh_button):
    if refresh_button.value:
        pass

    notebook_mode = notebook_mode_dropdown.value

    save_config_button = mo.ui.run_button(label="save config")
    match notebook_mode:
        case "unguided_rollout":
            rollout_ids=[]
        case "guided_rollout":
            rollout_ids=get_rollout_ids("ung")
        case "analyze_rollout":
            rollout_ids=get_rollout_ids("gui")
            save_config_button=None
        case _:
            pass
    return notebook_mode, rollout_ids, save_config_button


@app.cell
def _(mo, rollout_ids):
    # experiment selector (outer folder). rollout_ids are "<exp>/<start_ts>" for a
    # multi-start experiment, or a bare "<exp>" for a legacy single rollout.
    _exp_ids = sorted({rid.split("/", 1)[0] for rid in rollout_ids})
    experiment_id_dropdown = mo.ui.dropdown(
        options=_exp_ids,
        value=_exp_ids[0] if _exp_ids else None,
        label="experiment: ",
        allow_select_none=True,
    )
    return (experiment_id_dropdown,)


@app.cell(hide_code=True)
def _(experiment_id_dropdown, mo, rollout_ids):
    # rollout selector: the start_ts subfolders under the selected experiment. Empty for a
    # legacy single rollout (there the experiment id IS the rollout id).
    _starts = [rid.split("/", 1)[1] for rid in rollout_ids
               if "/" in rid and rid.split("/", 1)[0] == experiment_id_dropdown.value]
    rollout_dropdown = mo.ui.dropdown(
        options=_starts,
        value=_starts[0] if _starts else None,
        label="rollout: ",
        allow_select_none=True,
    )
    return (rollout_dropdown,)


@app.cell
def _(
    M,
    N,
    NUMERIC_AXES,
    RolloutConfig,
    T,
    a_t_mode_select,
    compute_axis_values,
    delta_trajectories,
    dump_json,
    ensure_rollout_dir,
    get_now_timestamp,
    gui_ref_select,
    guidance_mode_select,
    level,
    mask_corners,
    mask_mode_select,
    mask_shift_px_slider,
    mask_shift_select,
    notebook_mode,
    partition,
    rollout_id,
    save_config_button,
    spread_modes,
    sweep_ranges,
    timestamp,
    var,
):
    if notebook_mode!="analyze_rollout":  # button is None  
        if save_config_button.value and notebook_mode == "unguided_rollout":
            save_id = get_now_timestamp()
            rollout_dir = ensure_rollout_dir(save_id)
            path = rollout_dir / "config.json"
            save_config = RolloutConfig(
                # common to guided and unguided
                M=M,
                N=N,
                T=T,
                START_TS=timestamp,
                # experiment level params
                LEVEL=level,
                PARTITION=partition,
                VAR=var,
                MASK_CORNERS=mask_corners,
            )
            dump_json(save_config.to_dict(), path)
        if save_config_button.value and notebook_mode == "guided_rollout":
            save_id = rollout_id
            rollout_dir = ensure_rollout_dir(save_id)
            # config.json already lives at the experiment root (shared, START_TS-agnostic) -> leave
            # it untouched; only the sweep is authored here, written once at the exp root so all
            # start dates share it (a flat rollout's root is its own dir).
            _root = rollout_dir.parent if (rollout_dir.parent / "starts.json").exists() else rollout_dir

            _rv = sweep_ranges.value
            _a_t_modes = list(dict.fromkeys(list(a_t_mode_select.value) + spread_modes))
            sweep = {
                "GUIDANCE_MODE": list(guidance_mode_select.value),
                "GUI_REF": list(gui_ref_select.value),
                "MASK_MODE": list(mask_mode_select.value),
                "a_t_mode": _a_t_modes or ["gap-closing"],
                "mask_shift": [_d if _d == "none" else f"{_d}@{int(mask_shift_px_slider.value)}"
                               for _d in (list(mask_shift_select.value) or ["none"])],
                "GUIDANCE_DELTA": delta_trajectories,
                **{ax: compute_axis_values(ax, _rv) for ax in NUMERIC_AXES},
            }
            if all(sweep[a] for a in ("GUIDANCE_MODE", "GUI_REF", "MASK_MODE")):
                dump_json(sweep, _root / "sweep_params.json")
            else:
                print("each categorical axis needs at least one value")
    return


@app.cell
def _(
    GUIDANCE_METHODS,
    GUI_REFS,
    MASK_MODES,
    NUMERIC_AXES,
    get_sweep_dict,
    mo,
    notebook_mode,
    rollout_id,
    sweep_coord_label,
):
    w_defaults = [0.1, 0.5, 1.0, 2.0, 5.0]
    match notebook_mode:
        case "unguided_rollout":
            guidance_reference_dropdown = mo.ui.dropdown(
                GUI_REFS, value=GUI_REFS[0], label="guidance reference: "
            )
            w_slider = mo.ui.slider(
                steps=w_defaults,
                value=w_defaults[0],
                label=get_label("w", len(w_defaults)),
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            guidance_mode_dropdown = mo.ui.dropdown(options=GUIDANCE_METHODS, value=GUIDANCE_METHODS[0], label="guidance_mode: ")
            delta_trajectory_dropdown = None
            sweep_extra_dropdowns = mo.ui.dictionary({})
            sweep_params_widget = None

        case "guided_rollout":
            guidance_reference_dropdown = mo.ui.dropdown(
                GUI_REFS, value=GUI_REFS[0], label="guidance reference: "
            )
            w_slider = mo.ui.slider(
                steps=w_defaults,
                value=w_defaults[0],
                label=get_label("w", len(w_defaults)),
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            guidance_mode_dropdown = mo.ui.dropdown(options=GUIDANCE_METHODS, value=GUIDANCE_METHODS[0], label="guidance_mode: ")
            delta_trajectory_dropdown = None
            sweep_extra_dropdowns = mo.ui.dictionary({})
            sweep_params_widget = None

        case "analyze_rollout":
            # gated in the config cell: no rollout -> stop the analyze sweep-control cascade
            mo.stop(rollout_id is None)
            experiment_params = get_sweep_dict(rollout_id)

            # Every swept axis is defined here from the rollout's own sweep dict (so a
            # control only ever offers values that exist in the zarr). Named widgets stay
            # defined regardless of how many candidates an axis has, because ~15 downstream
            # cells read their .value; single-candidate axes simply aren't displayed.
            guidance_reference_dropdown = mo.ui.dropdown(
                options=experiment_params["GUI_REF"],
                value=experiment_params["GUI_REF"][0],
                label="guidance_reference: ",
            )
            # "w" was the flow-level strength axis of legacy runs; current sweeps
            # (FGWNOLR/FGWNOGAP) don't have it
            if "w" in experiment_params:
                w_slider = mo.ui.slider(
                    steps=experiment_params["w"],
                    value=experiment_params["w"][0],
                    label=get_label("w", len(experiment_params["w"])),
                    debounce=True,
                    show_value=True,
                )
            else:
                w_slider = None
            mask_mode_dropdown = mo.ui.dropdown(
                options=experiment_params["MASK_MODE"],
                value=experiment_params["MASK_MODE"][0],
                label="mask_mode: ",
            )
            guidance_mode_dropdown = mo.ui.dropdown(
                options=experiment_params["GUIDANCE_MODE"],
                value=experiment_params["GUIDANCE_MODE"][0],
                label="guidance_mode: ",
            )
            # delta_trajectory is a swept axis stored by integer index in the zarr;
            # the dropdown maps each candidate vector's label -> its index.
            _dt_candidates = experiment_params["GUIDANCE_DELTA"]
            # candidates sorted by peak (lowest first), named by their peak value;
            # .value stays the vector's ORIGINAL zarr index
            _dt_order = sorted(range(len(_dt_candidates)), key=lambda _i: max(_dt_candidates[_i]))
            _dt_options = {
                f"peak@{100 * max(_dt_candidates[_i]):+.3g}%": _i for _i in _dt_order
            }
            delta_trajectory_dropdown = mo.ui.dropdown(
                options=_dt_options,
                value=next(iter(_dt_options)),
                label="intervention profile φ_n: ",
            )
            # axes with a dedicated named widget above; every other sweep key gets an
            # auto-generated dropdown. .value carries the zarr coord label directly
            # (the value itself, or its integer index for non-scalar / None-bearing axes).
            _NAMED_CONTROLS = {
                "GUIDANCE_MODE": guidance_mode_dropdown,
                "MASK_MODE": mask_mode_dropdown,
                "GUI_REF": guidance_reference_dropdown,
                "w": w_slider,
                "GUIDANCE_DELTA": delta_trajectory_dropdown,
            }
            _extra = {}
            for _k, _vals in experiment_params.items():
                if _k in _NAMED_CONTROLS:
                    continue
                if _k in NUMERIC_AXES:
                    # numeric axis -> slider over the swept values (.value is the zarr
                    # coord label, since sweep_coord_label is identity for scalar axes)
                    _sorted_vals = sorted(_vals)  # ascending regardless of authoring order
                    _extra[_k] = mo.ui.slider(
                        steps=_sorted_vals,
                        value=_sorted_vals[0],
                        label=get_label(_k, len(_vals)),
                        debounce=True,
                        show_value=True,
                    )
                else:
                    _opts = {str(_v): sweep_coord_label(_k, _v, experiment_params) for _v in _vals}
                    _extra[_k] = mo.ui.dropdown(
                        options=_opts,
                        value=next(iter(_opts)),
                        label=f"{_k}: ",
                    )
            # a plain dict of mo.ui elements is invisible to the reactive graph -> changing
            # one would never rerun anything. mo.ui.dictionary makes it one reactive element.
            sweep_extra_dropdowns = mo.ui.dictionary(_extra)

            # one unified sweep widget: show a control only for axes that are actually
            # swept (more than one candidate). single-value axes have nothing to pick.
            _sweep_controls = [
                _NAMED_CONTROLS[_k] if _k in _NAMED_CONTROLS else sweep_extra_dropdowns[_k]
                for _k, _vals in experiment_params.items()
                if len(_vals) > 1
            ]
            sweep_params_widget = (
                mo.md("\n".join(f"- {_c}" for _c in _sweep_controls)) if _sweep_controls
                else mo.md("_no swept hyperparameters_")
            )

        case _:
            pass
    return (
        delta_trajectory_dropdown,
        experiment_params,
        guidance_mode_dropdown,
        guidance_reference_dropdown,
        mask_mode_dropdown,
        sweep_extra_dropdowns,
        sweep_params_widget,
        w_slider,
    )


@app.cell
def experiment_configs(config, experiment_params, mo, notebook_mode):
    if notebook_mode == "analyze_rollout":
        experiment_configs_widget = mo.hstack([
            mo.accordion({f"Config": mo.json(config.to_dict())}),
            mo.accordion({"Param grid": mo.json(experiment_params)}),
        ], justify="start")
    else:
        experiment_configs_widget = None
    experiment_configs_widget
    return


@app.cell
def _(
    guidance_reference_dropdown,
    mask_mode_dropdown,
    notebook_mode,
    w_slider,
):
    match notebook_mode:
        case "unguided_rollout" | "guided_rollout":
            hash_params = None
        case "analyze_rollout":
            hash_params = {
                "guidance_reference": guidance_reference_dropdown.value,
                "mask_mode": mask_mode_dropdown.value,
                "w": w_slider.value if w_slider is not None else None,
            }
        case _:
            pass
    return


@app.cell
def _(experiment_id_dropdown, mo, rollout_dropdown, save_config_button):
    setup_widget = mo.hstack([
        experiment_id_dropdown,
        rollout_dropdown,
        *([save_config_button] if save_config_button is not None else []),
    ], justify="start", align="start")
    setup_widget
    return


@app.cell
def _(mo, sweep_params_widget):
    mo.vstack([mo.md("sweep params: "), sweep_params_widget], align="start") if sweep_params_widget is not None else None
    return


@app.cell
def _(
    config,
    experiment_params,
    find_era5_input,
    get_gt_rollout,
    get_mask_2d,
    get_rollout,
    get_rollout_dir,
    has_unguided,
    iter_sweeps,
    mo,
    notebook_mode,
    np,
    open_unguided_traj_ds,
    rollout_id,
    sweep_coord_label,
    unguided_final_state_ds,
    xr,
):
    # experiments tables (side by side):
    #   LEFT  -- run status per sweep point (zarrs are NaN-initialized, so
    #            all-finite = ran)
    #   RIGHT -- Δtarget per guided (n, m): SIGNED relative miss
    #            (M(x_gui) - A) / |A - base|, A = (1 + phi_n) * base (rollout.py),
    #            evaluated with EACH ROW'S OWN mask (MASK_MODE / sigma_div can be
    #            swept); ❌ marks |Δtarget| > 5%; phi_n = 0 steps not guided -> blank
    if notebook_mode == "analyze_rollout" and rollout_id is not None:
        try:
            _gui_ds = get_rollout("gui", rollout_id)
            _probe_var = list(_gui_ds.data_vars)[0]
        except (FileNotFoundError, KeyError):
            _gui_ds = None
        try:
            _ung_ds = open_unguided_traj_ds(rollout_id, "ung") if has_unguided(rollout_id, "ung") else None
        except (FileNotFoundError, KeyError):
            _ung_ds = None
        _swept_axes = [_k for _k, _v in experiment_params.items() if len(_v) > 1]
        _sweeps = list(iter_sweeps(experiment_params))

        # row order: a_t_mode first IN SWEEP ORDER, then peak@, then sigma_div
        _am_order = {str(_v): _j for _j, _v in enumerate(experiment_params.get("a_t_mode") or [])}

        def _sort_key(_sw):
            _rest = [str(_sw[_k]) for _k in _swept_axes
                     if _k not in ("a_t_mode", "GUIDANCE_DELTA", "sigma_div")]
            return (_am_order.get(str(_sw.get("a_t_mode", "")), -1),
                    max(_sw["GUIDANCE_DELTA"]) if "GUIDANCE_DELTA" in _sw else 0.0,
                    float(_sw.get("sigma_div") or 0.0),
                    _rest)

        _sweeps.sort(key=_sort_key)
        _REL_TOL = 0.05  # achieved change may be within 5% of the requested change

        def _fmt(_k, _v):
            if _k == "GUIDANCE_DELTA":
                return f"peak@{100 * max(_v):+.3g}%"
            return str(_v)

        _tda2 = None
        if _gui_ds is not None and config is not None and config.VAR in _gui_ds:
            _tda2 = _gui_ds[config.VAR]
            if "level" in _tda2.dims:
                _tda2 = _tda2.sel(level=config.LEVEL)

        # per-row mask + per-mask baselines anchoring A = (1 + phi_n) * base
        _mask_cache, _base_cache = {}, {}

        def _row_mask(_sw):
            _mm2 = str(_sw.get("MASK_MODE") or config.MASK_MODE or "BBOX")
            _sg2 = float(_sw.get("sigma_div") or config.sigma_div or 2.0)
            _ms2 = str(_sw.get("mask_shift") or getattr(config, "mask_shift", None) or "none")
            _key = (_mm2, _sg2, _ms2)
            if _key not in _mask_cache:
                _mask_cache[_key] = xr.DataArray(
                    np.asarray(get_mask_2d(_mm2, config.MASK_CORNERS, sigma_div=_sg2, mask_shift=_ms2)),
                    dims=("latitude", "longitude"),
                    coords={"latitude": _gui_ds.latitude, "longitude": _gui_ds.longitude})
            return _key, _mask_cache[_key]

        def _base_for(_key, _mda2):
            if _key in _base_cache:
                return _base_cache[_key]
            _b2 = None
            if config.GUI_REF == "GT":
                _gt2 = get_gt_rollout(config.N + 1, config.START_TS, input_path=find_era5_input(get_rollout_dir(rollout_id)))[config.VAR]
                if "level" in _gt2.dims:
                    _gt2 = _gt2.sel(level=config.LEVEL)
                _b2 = np.asarray((_gt2.astype("float64") * _mda2)
                                 .sum(("latitude", "longitude")).compute())[1:]  # (N,)
            elif _ung_ds is not None and config.VAR in _ung_ds:
                _u2 = unguided_final_state_ds(rollout_id, "ung")[config.VAR]
                if "level" in _u2.dims:
                    _u2 = _u2.sel(level=config.LEVEL)
                _b2 = np.asarray((_u2.astype("float64") * _mda2)
                                 .sum(("latitude", "longitude")).compute())  # (M, N)
            _base_cache[_key] = _b2
            return _b2

        _rans, _rels = [], []
        for _sw in _sweeps:
            _sel = {_k: sweep_coord_label(_k, _v, experiment_params) for _k, _v in _sw.items()}
            _ran = False
            if _gui_ds is not None:
                try:
                    _ran = bool(_gui_ds[_probe_var].sel(_sel).notnull().all().compute())
                except (KeyError, ValueError):
                    _ran = False
            _rans.append(_ran)
            _r = None
            if _ran and _tda2 is not None:
                try:
                    _mkey, _mda2 = _row_mask(_sw)
                    _b2 = _base_for(_mkey, _mda2)
                    if _b2 is not None:
                        _g2 = np.asarray((
                            _tda2.sel({_k: _v for _k, _v in _sel.items() if _k in _tda2.dims})
                            .astype("float64") * _mda2).sum(("latitude", "longitude")).compute())
                        _r = {}
                        for _n2, _p2 in enumerate(_sw.get("GUIDANCE_DELTA") or []):
                            if _n2 >= _g2.shape[1] or float(_p2) == 0.0:
                                continue  # phi_n = 0 -> step not guided
                            _bb = _b2[_n2] if _b2.ndim == 1 else _b2[:, _n2]
                            _aa = (1.0 + float(_p2)) * _bb
                            _gp = np.maximum(np.abs(float(_p2) * _bb), 1e-12)  # |A - base|
                            _rr = np.atleast_1d((_g2[:, _n2] - _aa) / _gp)
                            for _m2 in range(_g2.shape[0]):
                                if np.isfinite(_rr[_m2]):
                                    _r[(_n2, _m2)] = float(_rr[_m2])
                except (KeyError, ValueError):
                    _r = None
            _rels.append(_r)

        _ax_cells = lambda _sw: " | ".join(_fmt(_k, _sw[_k]) for _k in _swept_axes)

        _run_md = ["| run | " + " | ".join(_swept_axes) + " |",
                   "|" + "---|" * (len(_swept_axes) + 1)]
        for _ran, _sw in zip(_rans, _sweeps):
            _run_md.append("| " + ("✅" if _ran else "") + " | " + _ax_cells(_sw) + " |")

        _nm_cols = sorted({_nm for _r in _rels if _r for _nm in _r})

        def _cell(_r, _nm):
            if not _r or _nm not in _r:
                return ""
            _v = _r[_nm]
            return ("" if abs(_v) <= _REL_TOL else "❌ ") + f"{100 * _v:+.1f}%"

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

        experiments_table = mo.hstack(
            [mo.vstack([mo.md("**runs**"), mo.md("\n".join(_run_md))], align="start"),
             mo.vstack([mo.md(r"**$\Delta$target** -- signed relative miss "
                              r"$(M(x^{gui}) - A)\,/\,|A - \mathrm{base}|$; ❌ marks $|\cdot| > 5\%$"),
                        _tgt_view], align="start")],
            justify="start", align="start", gap=2.0, wrap=True)
    else:
        experiments_table = None
    experiments_table
    return


@app.cell
def _(
    experiment_id_dropdown,
    get_now_timestamp,
    notebook_mode,
    rollout_dropdown,
):
    # rollout
    # conf params 
    match notebook_mode:
        case "unguided_rollout":
            rollout_id=get_now_timestamp()
        case "guided_rollout" | "analyze_rollout":
            # combine the experiment + rollout selectors into "<exp>/<start_ts>"; a legacy
            # single rollout has no start subfolder, so the experiment id IS the rollout id
            _exp = experiment_id_dropdown.value
            rollout_id = (f"{_exp}/{rollout_dropdown.value}"
                          if (_exp is not None and rollout_dropdown.value is not None) else _exp)
        case _:
            pass
    return (rollout_id,)


@app.cell
def _(
    M_slider,
    N_slider,
    config,
    day_slider,
    get_N_timestamps,
    get_timestamp_from_sliders,
    hour_slider,
    month_slider,
    notebook_mode,
    year_dropdown,
):
    # conf params 
    match notebook_mode:
        case "unguided_rollout":
            guidance_flag=False
            M=M_slider.value
            N=N_slider.value
            year=year_dropdown.value
            month=month_slider.value
            day=day_slider.value
            hour=hour_slider.value
            timestamp = get_timestamp_from_sliders(year, month, day, hour)
            map_interactive = True
        case "guided_rollout":
            guidance_flag=True
            M=config.M
            N=config.N
            year=None
            month=None
            day=None
            hour=None
            timestamp=config.START_TS
            map_interactive = False
        case "analyze_rollout":
            guidance_flag=None
            M=config.M
            N=config.N
            year=None
            month=None
            day=None
            hour=None
            timestamp=config.START_TS
            map_interactive = False
        case _:
            pass

    timestamps=get_N_timestamps(timestamp, N+1)
    return M, N, map_interactive, timestamp, timestamps


@app.cell(hide_code=True)
def config_cell(get_config, mo, notebook_mode, rollout_id):
    # config = the loaded rollout's pinned settings, independent of the sweep selection.
    # Kept in its own cell (not bundled with ZHCJ's sweep-dependent data loads) so changing a
    # sweep axis (R, etc.) doesn't re-run config -> T/M/N -> reset the t/m/n sliders.
    match notebook_mode:
        case "unguided_rollout":
            config = None
        case "guided_rollout" | "analyze_rollout":
            # no rollout selected yet (e.g. analyze mode before any guided run exists) -> halt
            # this cell and every analysis cell that reads `config`, showing a hint instead of a
            # NoneType cascade. Generating a run + pressing refresh clears the gate.
            mo.stop(
                rollout_id is None,
                mo.md("⚠️ **No rollout available.** `analyze_rollout` needs a generated guided "
                      "run — switch to `guided_rollout` to create one, then press **refresh**."),
            )
            config = get_config(rollout_id)
        case _:
            config = None
    return (config,)


@app.cell
def _(
    get_rollout,
    has_unguided,
    notebook_mode,
    open_unguided_traj_ds,
    rollout_id,
    sweep_params,
    unguided_final_state_ds,
):
    # data objects (config lives in its own cell so sweep changes don't re-run it)
    # NEW ung stores hold the full flow-step trajectory; unguided_xr is the final-state
    # view (t=-1), ung_traj_xr the full trajectory (None on legacy stores without t).
    match notebook_mode:
        case "unguided_rollout":
            unguided_xr=None
            ung_traj_xr=None
            guided_xr=None
            # TODO: set everything to None
        case "guided_rollout":
            if has_unguided(rollout_id, "ung"):
                _ung = open_unguided_traj_ds(rollout_id, "ung")
                ung_traj_xr = _ung if "t" in _ung.dims else None
                unguided_xr = unguided_final_state_ds(rollout_id, "ung")
            else:
                ung_traj_xr = None; unguided_xr = None
            guided_xr = None
        case "analyze_rollout":
            if has_unguided(rollout_id, "ung"):
                _ung = open_unguided_traj_ds(rollout_id, "ung").compute()
                ung_traj_xr = _ung if "t" in _ung.dims else None
                unguided_xr = unguided_final_state_ds(rollout_id, "ung").compute()
            else:
                ung_traj_xr = None; unguided_xr = None
            guided_xr = get_rollout("gui", rollout_id).sel(sweep_params).compute()
        case _:
            pass
    return guided_xr, unguided_xr


@app.cell(hide_code=True)
def _(STATS_PATH, VARIABLES_DICT, get_rollout, get_rollout_dir, torch, xr):
    # --- unguided-state reconstruction (latent-format migration) --------------------------
    # ung / gui_ung were migrated from physical STATE zarrs to the latent format:
    #   x_t = x_det + sigma_r * z_t   (z_t in {prefix}_res, x_det in ung_det|gui_det).
    # Rebuild the physical state here so downstream cells keep their old ung.zarr/gui_ung.zarr
    # view. {prefix}_res holds z_0..z_T (length T+1), so the reconstructed trajectory already
    # reaches the true final x_T at t=-1 (no separate z_T store).
    def _res_scale_map_local(_level_coord):
        _rsc = torch.load(STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False)
        _out = {_v: float(_rsc["surface"][_vi].squeeze())
                for _vi, _v in enumerate(VARIABLES_DICT["surface"])}
        _lev = _rsc["level"].squeeze(-1).squeeze(-1).numpy()
        for _vi, _v in enumerate(VARIABLES_DICT["level"]):
            _out[_v] = xr.DataArray(_lev[_vi] * (3.0 if _v == "vertical_velocity" else 1.0),
                                    dims=("level",), coords={"level": _level_coord})
        return _out

    def _ung_pick(_ds, _prefix, _sweep):
        if _prefix == "gui_ung" and _sweep:
            return _ds.sel({_k: _v for _k, _v in _sweep.items() if _k in _ds.dims})
        return _ds

    def _ung_det_store(rollout_id, prefix):
        _d = "ung_det" if prefix == "ung" else "gui_det"
        if not (get_rollout_dir(rollout_id) / f"{_d}.zarr").exists():
            _d = "gui_det"
        return _d

    def has_unguided(rollout_id, prefix):
        _dir = get_rollout_dir(rollout_id)
        return (_dir / f"{prefix}_res.zarr").exists() or (_dir / f"{prefix}.zarr").exists()

    def open_unguided_traj_ds(rollout_id, prefix, sweep_params=None):
        """Full unguided STATE trajectory x_t = x_det + sigma_r*z_t (length T, ends x_{T-1}).
        Legacy {prefix}.zarr STATE store returned as-is when {prefix}_res is absent."""
        if not (get_rollout_dir(rollout_id) / f"{prefix}_res.zarr").exists():
            return _ung_pick(get_rollout(prefix, rollout_id), prefix, sweep_params)   # legacy STATE store
        _z = _ung_pick(get_rollout(f"{prefix}_res", rollout_id), prefix, sweep_params)
        _det = _ung_pick(get_rollout(_ung_det_store(rollout_id, prefix), rollout_id), prefix, sweep_params)
        _rsm = _res_scale_map_local(_z.level)
        # keep the res store's dim order (m,n,t,lat,lon,...): det+c*z broadcasts det over t and
        # otherwise appends t LAST, which mis-indexes get_slices[m][n][t] downstream
        return xr.Dataset({_v: (_det[_v] + _rsm[_v] * _z[_v]).transpose(*_z[_v].dims)
                           for _v in _z.data_vars})

    def unguided_final_state_ds(rollout_id, prefix, sweep_params=None):
        """Converged final unguided state x_T = x_det + sigma_r*z_T (no t). The res store holds
        z_0..z_T, so the reconstructed trajectory already ends at x_T -> just take its last t."""
        _traj = open_unguided_traj_ds(rollout_id, prefix, sweep_params)
        return _traj.isel(t=-1) if "t" in _traj.dims else _traj


    return has_unguided, open_unguided_traj_ds, unguided_final_state_ds


@app.cell
def _(
    MASK_MODES,
    config,
    get_mask_2d,
    get_mask_center_pt,
    mask_mode,
    mask_shift_preview_dropdown,
    mask_shift_px_slider,
    notebook_mode,
    np,
    side_lat_slider,
    side_lon_slider,
    sigma_div_slider,
    sweep_extra_dropdowns,
    view_mask_mode,
):
    # mask: rectangle = center puck +/- half side lengths
    match notebook_mode:
        case "unguided_rollout":
            _lon_c, _lat_c = get_mask_center_pt()
            # corners are NOT clamped: the box center must stay on the puck.
            # lon wraps at the dateline; lat may stick out over a pole -- the
            # mask functions handle both (elliptical: over-the-pole distances)
            mask_corners = (
                _lon_c - side_lon_slider.value / 2,
                _lon_c + side_lon_slider.value / 2,
                _lat_c - side_lat_slider.value / 2,
                _lat_c + side_lat_slider.value / 2,
            )
            mask = get_mask_2d(view_mask_mode, mask_corners,
                               sigma_div=sigma_div_slider.value)
        case "guided_rollout":
            # preview of the config-pinned corners under the selected mask mode;
            # local controls preview sigma_div and the mask shift (the swept
            # values are set per job)
            mask_corners = config.MASK_CORNERS
            _pv_shift = ("none" if mask_shift_preview_dropdown.value == "none"
                         else f"{mask_shift_preview_dropdown.value}@{int(mask_shift_px_slider.value)}")
            mask = get_mask_2d(view_mask_mode, mask_corners,
                               sigma_div=sigma_div_slider.value,
                               mask_shift=_pv_shift)
        case "analyze_rollout":
            # the EXPERIMENT mask at the selected sweep point: swept MASK_MODE +
            # swept sigma_div (default 2.0 for stores without that axis)
            mask_corners = config.MASK_CORNERS
            # legacy stores swept with removed modes (GAUSSIAN/SPHERICAL): render with
            # the closest current mode; the zarr selection still uses the stored label
            _mm = mask_mode if mask_mode in MASK_MODES else "ELLIPTICAL"
            mask = get_mask_2d(_mm, mask_corners,
                               sigma_div=float(sweep_extra_dropdowns.value.get("sigma_div", 2.0)),
                               mask_shift=str(sweep_extra_dropdowns.value.get("mask_shift") or "none"))
        case _:
            pass
    # own-mask-scale region at HALF-MAXIMUM (same convention as maybe_mask and the
    # state histogram): the box for BBOX, the blob core for ELLIPTICAL. The >1e-6
    # support is useless for wide elliptical masks -- their wrapped tails cover the
    # whole globe except the antipodal cap (which rendered as a white 'bubble')
    mask_region = np.asarray(mask) >= 0.5 * float(np.asarray(mask).max())
    return mask, mask_corners, mask_region


@app.cell
def _(
    delta_trajectories,
    delta_trajectory_dropdown,
    experiment_params,
    notebook_mode,
):
    # target percentage profile p_n (fractions). The absolute target trajectory is
    # A_n = (1 + p_n) * baseline masked mean; rollout.py derives the loss delta ONLINE
    # from x_ref so guidance lands exactly on A_n.
    match notebook_mode:
        case "unguided_rollout":
            delta_trajectory=None
        case "guided_rollout":
            # first authored profile (preview); the full set lives in delta_trajectories
            delta_trajectory = delta_trajectories[0]
        case "analyze_rollout":
            # config.GUIDANCE_DELTA is None under sweeping; the swept vectors live in
            # experiment_params["GUIDANCE_DELTA"], picked by the dropdown's index.
            delta_trajectory = experiment_params["GUIDANCE_DELTA"][delta_trajectory_dropdown.value]
        case _:
            pass
    return (delta_trajectory,)


@app.cell
def _(
    clean_preds_xr,
    config,
    delta_trajectory,
    get_masked_mean,
    get_slices,
    gt_rollout,
    guidance_reference,
    mask,
    notebook_mode,
    np,
    unguided_xr,
):
    # guidance-target at config coords (pinned -- independent of browsing sliders)
    if notebook_mode == "analyze_rollout":
        cfg_clean_preds_slices = get_slices(clean_preds_xr, config.PARTITION, config.VAR, config.LEVEL)
        # absolute target A = (1 + p_n) * BASELINE masked mean (GT means for GT reference)
        _p = np.asarray(delta_trajectory, dtype=float)
        if guidance_reference == "GT":
            _cfg_base = np.tile(
                get_masked_mean(get_slices(gt_rollout, config.PARTITION, config.VAR, config.LEVEL), mask)[1:len(_p) + 1],
                (config.M, 1),
            )
        else:
            _cfg_base = get_masked_mean(get_slices(unguided_xr, config.PARTITION, config.VAR, config.LEVEL), mask)
        cfg_target_guidance_M_N_trajectories = (1.0 + _p) * _cfg_base
    return (cfg_target_guidance_M_N_trajectories,)


@app.cell
def _(
    N,
    find_era5_input,
    get_gt_rollout,
    get_masked_mean,
    get_rollout,
    get_rollout_dir,
    get_slices,
    gui_ung_final_xr,
    guided_xr,
    level,
    m,
    mask,
    notebook_mode,
    partition,
    rollout_id,
    sweep_params,
    timestamp,
    unguided_xr,
    var,
):
    # data sub-objects
    match notebook_mode:
        case "unguided_rollout":
            det_m_trajectory = None
            ung_M_N_slices = None
            ung_M_N_trajectories = None
            ung_mean_trajectory = None
            ung_m_trajectory = None
            ung_lb_trajectory = None
            ung_ub_trajectory = None

            gui_M_N_slices = None
            gui_M_N_trajectories = None
            gui_mean_trajectory = None
            gui_m_trajectory = None

            gui_ung_m_trajectory = None
        case "guided_rollout":
            det_m_trajectory = None
            ung_M_N_slices = get_slices(unguided_xr, partition, var, level)
            ung_M_N_trajectories = get_masked_mean(ung_M_N_slices, mask)
            ung_mean_trajectory = ung_M_N_trajectories.mean(axis=0)
            ung_m_trajectory = ung_M_N_trajectories[m]
            ung_lb_trajectory = ung_M_N_trajectories.min(axis=0)
            ung_ub_trajectory = ung_M_N_trajectories.max(axis=0)

            gui_M_N_slices = None
            gui_M_N_trajectories = None
            gui_mean_trajectory = None
            gui_m_trajectory = None

            gui_ung_m_trajectory = None
        case "analyze_rollout":
            # deterministic masked-mean trajectory (branch overlay in the rollout chart)
            try:
                _det_MN = get_slices(get_rollout("gui_det", rollout_id).sel(sweep_params).compute(),
                                     partition, var, level)
                det_m_trajectory = get_masked_mean(_det_MN, mask)[m]
            except (FileNotFoundError, KeyError):
                det_m_trajectory = None
            ung_M_N_slices = get_slices(unguided_xr, partition, var, level)
            ung_M_N_trajectories = get_masked_mean(ung_M_N_slices, mask)
            ung_mean_trajectory = ung_M_N_trajectories.mean(axis=0)
            ung_m_trajectory = ung_M_N_trajectories[m]
            ung_lb_trajectory = ung_M_N_trajectories.min(axis=0)
            ung_ub_trajectory = ung_M_N_trajectories.max(axis=0)

            gui_M_N_slices = get_slices(guided_xr, partition, var, level)
            gui_M_N_trajectories = get_masked_mean(gui_M_N_slices, mask)
            gui_mean_trajectory = gui_M_N_trajectories.mean(axis=0)
            gui_m_trajectory = gui_M_N_trajectories[m]

            gui_ung_M_N_slices = get_slices(gui_ung_final_xr, partition, var, level)
            gui_ung_M_N_trajectories = get_masked_mean(gui_ung_M_N_slices, mask)
            gui_ung_m_trajectory = gui_ung_M_N_trajectories[m]
        case _:
            pass

    # gt (from this experiment's downloaded era5_input.nc, rolled to the mask
    # convention; falls back to the global arches store when no file is found)
    _gt_input = find_era5_input(get_rollout_dir(rollout_id))
    gt_rollout = get_gt_rollout(N + 1, timestamp, input_path=_gt_input)
    gt_N_slices = get_slices(gt_rollout, partition, var, level)
    gt_trajectory = get_masked_mean(gt_N_slices, mask)
    return (
        det_m_trajectory,
        gt_N_slices,
        gt_rollout,
        gt_trajectory,
        gui_M_N_slices,
        gui_M_N_trajectories,
        gui_m_trajectory,
        gui_ung_m_trajectory,
        ung_M_N_slices,
        ung_M_N_trajectories,
        ung_m_trajectory,
    )


@app.cell(hide_code=True)
def _(
    N,
    delta_trajectories,
    delta_trajectory,
    gt_trajectory,
    guidance_reference,
    m,
    notebook_mode,
    np,
    ung_M_N_trajectories,
):
    # target guidance: the absolute target trajectory A_n = (1 + p_n) * baseline masked
    # mean -- the single object formerly split into planned/target guidance. Baseline =
    # the unguided rollout per member (GT means for a GT reference, member-independent).
    if notebook_mode in ("guided_rollout", "analyze_rollout"):
        # guided mode previews EVERY authored profile; analyze mode has the selected one
        _profiles = delta_trajectories if notebook_mode == "guided_rollout" else [delta_trajectory]
        # clip each profile to the N guided weather steps: a profile longer than N broadcasts to
        # length len(profile) and desyncs from the N-step base / timestamps (delta-vs-N mismatch)
        _profiles = [np.asarray(_pp, dtype=float)[:N] for _pp in _profiles]
        _p0 = _profiles[0]
        if guidance_reference == "GT":
            _base = np.tile(gt_trajectory[1:len(_p0) + 1], (ung_M_N_trajectories.shape[0], 1))
        else:
            _base = ung_M_N_trajectories
        _targets_all = [(1.0 + _pp) * _base for _pp in _profiles]
        target_guidance_M_N_trajectories = _targets_all[0]
        target_guidance_trajectory = target_guidance_M_N_trajectories[m]
        # one member-m line per profile, drawn together on the trajectories chart
        target_guidance_trajectories_all = [_t[m] for _t in _targets_all]
    else:
        target_guidance_trajectory = None
        target_guidance_M_N_trajectories = None
        target_guidance_trajectories_all = None
    return target_guidance_M_N_trajectories, target_guidance_trajectories_all


@app.cell
def _(
    delta_trajectory_dropdown,
    experiment_params,
    guidance_mode_dropdown,
    guidance_reference_dropdown,
    mask_mode_dropdown,
    notebook_mode,
    sweep_coord_label,
    sweep_extra_dropdowns,
    w_slider,
):
    sweep_params = {
        "GUI_REF": guidance_reference_dropdown.value,
        "MASK_MODE": mask_mode_dropdown.value,
        "GUIDANCE_MODE": guidance_mode_dropdown.value,
    }
    # "w" axis only exists on legacy runs (see w_slider construction)
    if w_slider is not None:
        sweep_params["w"] = w_slider.value
    # delta_trajectory is an integer-indexed sweep axis in the zarr (real vectors live
    # in the delta_trajectory_value coord); the dropdown holds the chosen index.
    if delta_trajectory_dropdown is not None:
        sweep_params["GUIDANCE_DELTA"] = delta_trajectory_dropdown.value

    # Every sweep_params key is now a zarr dimension, so .sel must reduce all of them
    # or singleton dims linger and break the 2D plotters. The handful above are driven
    # by dedicated widgets; every other axis is driven by its auto-generated dropdown
    # (sweep_extra_dropdowns), whose .value is already the stored coord label.
    if notebook_mode == "analyze_rollout":
        for _key, _vals in experiment_params.items():
            if _key in sweep_params:
                continue
            if _key in sweep_extra_dropdowns:
                sweep_params[_key] = sweep_extra_dropdowns.value[_key]
            else:
                # single-value axis: no widget, take its sole stored coord label
                sweep_params[_key] = sweep_coord_label(_key, _vals[0], experiment_params)
    return (sweep_params,)


@app.cell
def _(gt_trajectory, ung_m_trajectory):
    # connect ui to data
    reference_trajectories = {
        "UNG": ung_m_trajectory,
        "GT": gt_trajectory,
    }
    return


@app.cell
def _():
    # all single ui elements
    return


@app.cell
def _(LEVELS_DICT, SURFACE_PAIR, level_slider, var_dropdown):
    _base_v = var_dropdown.value
    _lv = int(level_slider.value)  # 0 = surface tick, else the pressure level in hPa
    if _lv == 0:
        if _base_v in SURFACE_PAIR:
            var, partition, level, var_valid = SURFACE_PAIR[_base_v], "surface", 0, True
        elif _base_v == "mean_sea_level_pressure":
            var, partition, level, var_valid = _base_v, "surface", 0, True
        else:  # level-only variable: nothing lives at the surface tick
            var, partition, level, var_valid = _base_v, "level", LEVELS_DICT["level"][0], False
    else:
        if _base_v == "mean_sea_level_pressure":  # surface-only: nothing at pressure levels
            var, partition, level, var_valid = _base_v, "surface", 0, False
        else:
            var, partition, level, var_valid = _base_v, "level", _lv, True
    return level, partition, var, var_valid


@app.cell
def _(n_slider):
    # slider is 1-based (1..N); n is the 0-based index
    n=n_slider.value-1
    return (n,)


@app.cell
def _(m_slider):
    # slider is 1-based (1..M); m is the 0-based index
    m=m_slider.value-1
    return (m,)


@app.cell
def _(LEVELS_DICT, VARIABLES_DICT, mo):
    # variable selector: no partition dropdown -- the partition is DERIVED from
    # (variable, level index). Level 0 = surface (paired level vars map to their
    # surface twin, mslp to itself); 1..13 = the pressure levels in LEVELS_DICT
    # order. Invalid combinations (surface-only away from 0, level-only at 0)
    # render every chart of the selected variable empty via the gated slicers.
    _BASE_VARS = list(VARIABLES_DICT["level"]) + ["mean_sea_level_pressure"]
    var_dropdown = mo.ui.dropdown(_BASE_VARS, value="temperature", label="var: ")
    # level control: option slider, surface tick first then pressure levels ordered
    # surface-upward (1000 -> 50 hPa); .value IS the level (0 = surface)
    _LEVEL_STEPS = LEVELS_DICT["surface"] + LEVELS_DICT["level"][::-1]
    level_slider = mo.ui.slider(steps=_LEVEL_STEPS, value=0, label="level: ",
                                show_value=True, debounce=True)
    return level_slider, var_dropdown


@app.cell
def _(get_level_idx, get_var_idx, level, partition, var):
    var_idx = get_var_idx(partition, var)
    level_idx = get_level_idx(partition, level)
    return


@app.cell
def _(mo):
    year_options=[2020, 2026]
    year_dropdown=mo.ui.dropdown(options=year_options, value=year_options[0], label="year: ")
    return (year_dropdown,)


@app.cell
def _(mo):
    month_slider=mo.ui.slider(1, 12, value=2, step=1, label="month: ", show_value=True, debounce=True)
    return (month_slider,)


@app.cell
def _(max_day, mo, month_slider):
    day_slider=mo.ui.slider(start=1, stop=max_day(month_slider.value), value=1, label="day: ", show_value=True, debounce=True)
    hour_slider = mo.ui.slider(0, 18, value=0, step=6, label="hour: ", show_value=True, debounce=True)
    return day_slider, hour_slider


@app.cell
def _(mo):
    M_slider = mo.ui.slider(1, 20, value=1, label="M: ", show_value=True, debounce=True)
    N_slider = mo.ui.slider(1, 30, value=1, label="N: ", show_value=True, debounce=True)
    return M_slider, N_slider


@app.cell
def _(M, N, mo):
    n_slider = mo.ui.slider(
        start=1, 
        stop=N,
        step=1,
        label=get_label("n", N),
        value=1,
        debounce=True,
        show_value=True
    )

    m_slider = mo.ui.slider(
        start=1, 
        stop=M,
        step=1,
        label=get_label("m", M),
        value=1,
        debounce=True,
        show_value=True
    )
    return m_slider, n_slider


@app.cell
def _(mo):
    # number of target percentage profiles to author (each a sweep value of GUIDANCE_DELTA)
    n_deltas_slider = mo.ui.slider(1, 6, step=1, value=1, label="target percentage profiles: ", show_value=True, debounce=True)
    return (n_deltas_slider,)


@app.cell
def _(config, mo, n_deltas_slider, notebook_mode):
    # per-profile params: start% / peak% / start@n / stop@n (linear ramp)
    if notebook_mode == "guided_rollout":
        _dc = {}
        for _i in range(n_deltas_slider.value):
            _dc[f"{_i}.start"]    = mo.ui.number(value=0.0, label=f"profile {_i} — start %: ")
            _dc[f"{_i}.peak"]     = mo.ui.number(value=0.1, label="peak %: ")
            _dc[f"{_i}.start_at"] = mo.ui.slider(0, config.N, step=1, value=0, label="start@n: ", show_value=True, debounce=True)
            _dc[f"{_i}.stop_at"]  = mo.ui.slider(1, config.N, step=1, value=config.N, label=get_label("stop@n", config.N), show_value=True, debounce=True)
        delta_controls = mo.ui.dictionary(_dc)
    else:
        delta_controls = mo.ui.dictionary({})
    return (delta_controls,)


@app.cell
def _(N, delta_controls, mo, n_deltas_slider, notebook_mode):
    # target percentage profiles: 0 before start@n, linear ramp start%->peak% over
    # [start@n, stop@n], 0 after. Length-N lists of fractions indexed by rollout step n
    # (aligned with delta_trajectory[n] in rollout.py).
    def _linear_delta(N, start_pct, peak_pct, start_at, stop_at):
        start, peak = start_pct / 100, peak_pct / 100
        out = []
        for _n in range(1, N + 1):                      # rollout steps 1..N
            if start_at <= _n <= stop_at:
                frac = (_n - start_at) / (stop_at - start_at) if stop_at > start_at else 1.0
                out.append(start + (peak - start) * frac)
            else:
                out.append(0.0)
        return out

    if notebook_mode == "guided_rollout":
        _dv = delta_controls.value
        delta_trajectories = [
            _linear_delta(N, _dv[f"{_i}.start"], _dv[f"{_i}.peak"],
                          int(_dv[f"{_i}.start_at"]), int(_dv[f"{_i}.stop_at"]))
            for _i in range(n_deltas_slider.value)
        ]
        # intensity order: index 0 = lowest maximum intensity, everywhere downstream
        # (preview, sweep_params.json, zarr coord index)
        delta_trajectories = sorted(delta_trajectories, key=max)
        _rows = [
            mo.hstack([delta_controls[f"{_i}.start"], delta_controls[f"{_i}.peak"],
                       delta_controls[f"{_i}.start_at"], delta_controls[f"{_i}.stop_at"]],
                      justify="start", align="center")
            for _i in range(n_deltas_slider.value)
        ]
        # controls only; the profiles are drawn on the rollout-trajectories chart's right axis
        delta_widget = mo.vstack([n_deltas_slider, *_rows], align="start")
    else:
        delta_trajectories = []
        delta_widget = None
    return delta_trajectories, delta_widget


@app.cell
def _(mask_mode_dropdown):
    # mask_mode comes from the dropdown in every mode: authoring selection in
    # unguided/guided, and the sweep selector in analyze. (MASK_MODE is a swept axis,
    # so the scalar config.MASK_MODE is None -- don't read it here.)
    mask_mode = mask_mode_dropdown.value
    # single mask everywhere: the view mask always follows the selected mask mode
    view_mask_mode = mask_mode
    return mask_mode, view_mask_mode


@app.cell
def _(guidance_reference_dropdown):
    guidance_reference = guidance_reference_dropdown.value
    return (guidance_reference,)


@app.cell
def _():
    # presentation widgets
    return


@app.cell
def _(M_slider, N_slider, m_slider, mo, n_slider):
    M_N_widget = mo.hstack([M_slider, N_slider], justify="start")
    m_n_widget = mo.hstack([m_slider, n_slider], justify="start")
    return M_N_widget, m_n_widget


@app.cell
def _(mo):
    T_slider = mo.ui.slider(
        steps=range(1, 25+1),
        value=25,
        label="T: ",
        debounce=True,
        show_value=True
    )
    return (T_slider,)


@app.cell
def _(T_slider, config, notebook_mode):
    # T (flow/sampling steps): the slider drives NEW unguided rollouts; in guided/analyze
    # mode it must be the loaded rollout's T (config.T), which the data was generated with.
    T = T_slider.value if notebook_mode == "unguided_rollout" else config.T
    return (T,)


@app.cell
def _(day_slider, hour_slider, mo, month_slider, notebook_mode, year_dropdown):
    timestamp_widget=mo.hstack([year_dropdown, month_slider, day_slider, hour_slider], justify="start")
    match notebook_mode:
        case "unguided_rollout":
            pass
        case "guided_rollout":
            timestamp_widget=None
        case "analyze_rollout":   
            timestamp_widget=None
    return (timestamp_widget,)


@app.cell
def _(
    GUIDANCE_METHOD_HYPERS,
    NUMERIC_AXES,
    a_t_mode_select,
    compute_axis_values,
    gui_ref_select,
    guidance_mode_select,
    mask_mode_select,
    mask_shift_px_slider,
    mask_shift_select,
    mo,
    notebook_mode,
    spread_widget,
    sweep_ranges,
):
    # shown only for the modes that use them (see GUIDANCE_METHOD_HYPERS)
    _MODE_HYPERS = GUIDANCE_METHOD_HYPERS

    _rv = sweep_ranges.value
    def _sweep_row(ax):
        return mo.hstack(
            [mo.md(f"`{ax}`:"), sweep_ranges[f"{ax}.start"], sweep_ranges[f"{ax}.stop"],
             sweep_ranges[f"{ax}.n"], sweep_ranges[f"{ax}.log"], mo.md(f"-> `{compute_axis_values(ax, _rv)}`")],
            justify="start", align="center",
        )

    _selected_modes = guidance_mode_select.value
    _mode_num = list(dict.fromkeys(h for _m in _selected_modes for h in _MODE_HYPERS[_m] if h in NUMERIC_AXES))

    _ms_vals = [_d if _d == "none" else f"{_d}@{int(mask_shift_px_slider.value)}"
                for _d in (list(mask_shift_select.value) or ["none"])]

    # spread@ preview now lives per-row inside spread_widget (see the spread build cell)

    hypers_widget = mo.vstack([
        mo.md("## Sweep"),
        mo.md("**Common**:"),
        mo.hstack([guidance_mode_select, gui_ref_select, mask_mode_select, a_t_mode_select], justify="start"),
        *([spread_widget] if spread_widget is not None else []),
        _sweep_row("sigma_div"),
        mo.hstack([mask_shift_select, mask_shift_px_slider, mo.md(f"-> `{_ms_vals}`")],
                  justify="start", align="center"),
        mo.md("**Specific**:"),
        *([_sweep_row(ax) for ax in _mode_num] if _mode_num else [mo.md("_none_")]),
    ], align="start")

    hypers_widget if notebook_mode == "guided_rollout" else None
    return


@app.cell
def wa_schedule(
    T,
    a_t_mode_select,
    a_t_profile,
    cfg_target_guidance_M_N_trajectories,
    clean_preds_xr,
    compute_axis_values,
    config,
    dpi_slider,
    get_slices,
    get_w_star,
    gui_vec_xr,
    guidance_mode_dropdown,
    guidance_mode_select,
    guidance_schedules,
    lambda_t_by_method,
    m,
    mask,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    plt,
    rollout_id,
    sweep_params,
    sweep_ranges,
    trajectories_section_checkbox,
):
    # Guidance weight schedule over the flow, beside the rollout trajectories plot.
    # a_t = the guidance PROFILE (a_t_profile modes), dimensionless in [0, 1].
    # guided_rollout: preview at the swept w_init / eta (single-axes plot_trajectory).
    # analyze_rollout: recorded schedules as two stacked panels -- top: w_t (left) and the
    # applied kick norm lambda_hat = w*a*c (right twin); bottom: c_t (left, own scale) and
    # a_t on a fixed 0-1 right twin. The last flow step is never guided -> dropped from
    # all panels (points at t = 1..T-1; the axis still spans 0..T).
    if trajectories_section_checkbox.value and notebook_mode == "guided_rollout":
        selected_modes = list(guidance_mode_select.value)
        fgwnolr_w_choices = compute_axis_values("fgwnolr_w_init", sweep_ranges.value)
        eta_choices = compute_axis_values("eta", sweep_ranges.value)
        _am_choices = list(a_t_mode_select.value) or ["gap-closing"]
        w_from_star = False
    elif trajectories_section_checkbox.value and notebook_mode == "analyze_rollout":
        selected_modes = [guidance_mode_dropdown.value]
        _sweep_sel = dict(sweep_params)  # records store the coord-label sweep form
        _wstar_recs = get_w_star(rollout_id, _sweep_sel)
        w_from_star = bool(_wstar_recs)
        _wstar = sorted({r["w_star"] for r in _wstar_recs})
        fgwnolr_w_choices = _wstar or [sweep_params["fgwnolr_w_init"]]
        eta_choices = [sweep_params["eta"]]
        _am_choices = [sweep_params["a_t_mode"]]
    else:
        selected_modes = []
        fgwnolr_w_choices = []
        eta_choices = []
        _am_choices = []
    eta_modes = [mode for mode in selected_modes if mode in ("FGWNOLR", "FGWNOGAP")]
    _has_recorded = notebook_mode == "analyze_rollout" and bool(lambda_t_by_method)
    _AT_COLOR = "#7B2CBF"  # a_t purple
    if _has_recorded:
        # schedules actually applied by the run, one linestyle per method
        _LAM_COLOR, _W_COLOR, _C_COLOR = "#7B2CBF", "#444444", "#0B7285"
        _wa_styles = ["-", "--", "-.", ":"]
        # measured remaining gap eps_t = S(x_hat_t) - A on the guided channel,
        # normalized by the initial gap -> lives on the a_t axis for direct
        # comparison with the prescribed profile
        try:
            _cp_sched = (np.asarray(get_slices(clean_preds_xr, config.PARTITION, config.VAR, config.LEVEL))[:, n]
                         * np.asarray(mask)).sum(axis=(-1, -2))[m]
            _A_sched = float(np.asarray(cfg_target_guidance_M_N_trajectories)[m, n])
            _eps_sched = _cp_sched - _A_sched
            _eps_rel = _eps_sched / _eps_sched[0] if abs(float(_eps_sched[0])) > 1e-12 else None
        except Exception:
            _eps_rel = None
        _fig_wa, (_ax_wa_top, _ax_wa_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 7), dpi=dpi_slider.value, gridspec_kw={"hspace": 0.14}
        )
        _ax_wa_top_r = _ax_wa_top.twinx()
        _ax_wa_bot_r = _ax_wa_bot.twinx()
        for _mi, (_meth, _sd) in enumerate(guidance_schedules.items()):
            _ls = _wa_styles[_mi % len(_wa_styles)]
            _lam_v = np.asarray(_sd["lambda_hat"], dtype=float)[:-1]
            _w_v = np.asarray(_sd["w_t"], dtype=float)[:-1]
            _c_v = np.asarray(_sd["c_t"], dtype=float)[:-1]
            _a_v = np.asarray(_sd["a_t"], dtype=float)[:-1]
            _xs = np.arange(len(_lam_v)) + 1
            _wlab = f" (w={_w_v[0]:g})" if len(set(_w_v)) == 1 else ""
            _ax_wa_top.plot(_xs, _w_v, _ls, marker="s", markersize=3, color=_W_COLOR,
                            alpha=0.85, label=rf"$w_t$ — {_meth}{_wlab}")
            _ax_wa_top_r.plot(_xs, _lam_v, _ls, marker="o", markersize=3.5, color=_LAM_COLOR,
                              label=rf"$\hat\lambda_t$ — {_meth}")
            _ax_wa_bot.plot(_xs, _c_v, _ls, marker="o", markersize=3.5, color=_C_COLOR,
                            label=rf"$c_t$ — {_meth}")
            _am = str(_sd.get("a_t_mode") or "?")
            _ax_wa_bot_r.plot(_xs, _a_v, _ls, marker="o", markersize=3,
                              color=_AT_COLOR, alpha=0.9, label=rf"$a_t$ — {_meth} ({_am})")
        if _eps_rel is not None:
            # eps_t is the gap ENTERING step t (pre-kick clean prediction) -> the
            # line starts at t=0 with eps_0/eps_0 = 1
            _ax_wa_bot_r.plot(np.arange(0, len(_eps_rel)), _eps_rel, "-o",
                              markersize=3, linewidth=1.2, color="#E91E63", alpha=0.9,
                              label=r"$\varepsilon_t/\varepsilon_0$ — measured remaining gap")
        _ax_wa_top.set_ylabel(r"$w_t$", color=_W_COLOR)
        _ax_wa_top_r.set_ylabel(r"$\hat\lambda_t$", color=_LAM_COLOR)
        _ax_wa_bot.set_ylabel(r"$c_t$", color=_C_COLOR)
        _ax_wa_bot_r.set_ylabel(r"$a_t$", color=_AT_COLOR)
        _ax_wa_bot_r.set_ylim(0.0, 1.05)
        _ax_wa_bot.set_xlabel("$t$")
        _ax_wa_bot.set_xlim(-0.5, T + 0.5)
        _ax_wa_bot.set_xticks(np.arange(0, T + 1))
        for _axx in (_ax_wa_top, _ax_wa_top_r, _ax_wa_bot, _ax_wa_bot_r):
            _axx.spines["top"].set_visible(False)
        for _axx in (_ax_wa_top, _ax_wa_bot):
            _axx.yaxis.grid(True, color="#EAEAEA")
            _axx.set_axisbelow(True)
        _ht, _lt = _ax_wa_top.get_legend_handles_labels()
        _htr, _ltr = _ax_wa_top_r.get_legend_handles_labels()
        _ax_wa_top.legend(_ht + _htr, _lt + _ltr, frameon=False, fontsize=8)
        _hb, _lb = _ax_wa_bot.get_legend_handles_labels()
        _hbr, _lbr = _ax_wa_bot_r.get_legend_handles_labels()
        _ax_wa_bot.legend(_hb + _hbr, _lb + _lbr, frameon=False, fontsize=8)
        _fig_wa.suptitle(
            r"Guidance schedule  ($\hat\lambda_t = w\,a_t\,c_t$ — kick norm on the unit gradient)",
            x=0.06, y=0.985, ha="left", fontsize=15, fontweight="bold", color="#222222",
        )
        _fig_wa.text(0.06, 0.945, rf"recorded from run (m={m}, n={n})",
                     ha="left", va="top", fontsize=9.5, color="#555555")
        _fig_wa.subplots_adjust(top=0.9, left=0.09, right=0.91, bottom=0.08)
        wa_schedule_widget = _fig_wa
    elif notebook_mode == "analyze_rollout" and gui_vec_xr is not None:
        # no sidecar records -> REALIZED schedule from the exact reconstruction:
        # lambda_hat_t = ||gui_vec_t|| (kick norm on the unit gradient) over all
        # vars/levels/space at the current (m, n)
        _gv = gui_vec_xr.isel(m=m, n=n)
        _lam_sq = sum((_gv[_v] ** 2).sum(dim=[_d for _d in _gv[_v].dims if _d != "t"]) for _v in _gv.data_vars)
        _lam_real = np.sqrt(np.asarray(_lam_sq.compute() if hasattr(_lam_sq, "compute") else _lam_sq, dtype=float))
        # a_t of the selected sweep point on the right axis
        _am_real = str(sweep_params.get("a_t_mode", "gap-closing"))
        _eta_real = float(sweep_params.get("eta", 0.5))
        _a_real = a_t_profile(_am_real, _eta_real, T)
        wa_schedule_widget = plot_trajectory(
            {r"$\hat\lambda_t$ (realized)": _lam_real[:-1].tolist()},
            dpi=dpi_slider.value,
            var=r"$\hat\lambda_t$",
            right_trajectory={rf"$a_t$ ({_am_real}, $\eta$={_eta_real:g})": _a_real[:-1].tolist()},
            right_label=r"$a_t$",
            right_color=_AT_COLOR,
            title=r"Guidance schedule  ($\hat\lambda_t = w\,a_t\,c_t$ — kick norm on the unit gradient)",
            subtitle=f"realized from run (m={m}, n={n}) — reconstructed from the traces",
            xlabel="$t$",
            figsize=(10, 6),
            start_index=1,
        )
        _ax_wa = wa_schedule_widget.axes[0] if hasattr(wa_schedule_widget, "axes") else None
        if _ax_wa is not None:
            _ax_wa.set_xlim(-1.0, T + 0.4)
            _ax_wa.set_xticks(np.arange(0, T + 1))
    elif eta_modes:
        # preview: a_t on the RIGHT (shared by both methods). NOLR also previews
        # lambda_t = w_t * a_t on the LEFT; NOGAP's w_t / lambda_t are solved per
        # step at runtime, so only a_t is shown.
        _left_label = r"$\lambda_t,\ w_t$"
        _sched, _right = {}, {}
        _am_choices = _am_choices or ["gap-closing"]
        for _eta in (eta_choices or [0.5]):
            for _am in _am_choices:
                _a = a_t_profile(_am, _eta, T)
                _right[rf"$a_t$ ({_am}, $\eta$={_eta:g})"] = _a[:-1].tolist()
                if "FGWNOLR" in eta_modes:
                    for _w in (fgwnolr_w_choices or [250.0]):
                        _sched[rf"$\lambda_t$ ($w$={_w:g}, {_am}, $\eta$={_eta:g})"] = (_w * _a)[:-1].tolist()
        if not _sched:
            # NOGAP-only: no w/lambda -> put a_t on the left
            _sched, _right, _left_label = _right, None, r"$a_t$"
        _color_map = {_k: _AT_COLOR for _k in _sched} if _right is None else None
        wa_schedule_widget = plot_trajectory(
            _sched,
            dpi=dpi_slider.value,
            right_trajectory=_right,
            right_label=r"$a_t$",
            right_color=_AT_COLOR,
            color_map=_color_map,
            var=_left_label,
            title=r"Guidance schedule  ($\hat\lambda_t = w\,a_t\,c_t$ — kick norm on the unit gradient)",
            subtitle="preview — before guiding",
            xlabel="$t$",
            figsize=(10, 6),
            start_index=1,
        )
        # axis spans the full flow 0..T even though no point sits at 0 or T
        _ax_wa = wa_schedule_widget.axes[0] if hasattr(wa_schedule_widget, "axes") else None
        if _ax_wa is not None:
            _ax_wa.set_xlim(-0.5, T + 0.5)
            _ax_wa.set_xticks(np.arange(0, T + 1))
    else:
        wa_schedule_widget = None
    return (wa_schedule_widget,)


@app.cell
def _(
    M_N_widget,
    T_slider,
    climatology_rolling_slider,
    delta_widget,
    dpi_slider,
    experiment_params,
    guidance_convergence_t_plot,
    guidance_reference_dropdown,
    inspect_states_widget_make,
    level_slider,
    m_n_widget,
    mask_map,
    mask_mode_dropdown,
    mask_shift_preview_dropdown,
    mask_shift_px_slider,
    mo,
    notebook_mode,
    side_lat_slider,
    side_lon_slider,
    sigma_div_slider,
    state_hist_plot,
    sweep_extra_dropdowns,
    sweep_params_widget,
    traj_row_select,
    trajectories_plot,
    var_dropdown,
    wa_schedule_widget,
    weather_map,
    year_trajectory_plot,
    zoom_slider,
):
    mask_widget_controls = mo.hstack(
        [var_dropdown, level_slider],
        justify="start",
        align="start",
    )

    _mask_maps_row = mo.hstack(
        [_w_ for _w_ in [weather_map, mask_map] if _w_ is not None],
        justify="start",
        align="start",
    )
    mask_widget_maps = mo.vstack([
        mo.hstack([zoom_slider, dpi_slider, side_lon_slider, side_lat_slider, sigma_div_slider], justify="start"),
        _mask_maps_row,
    ], align="start")
    # authoring (unguided) exposes the mask coord controls; guided/analyze show only the
    # static, config-pinned mask map (the partition/var/level controls live in the
    # inspect-states / physical-realism sections for browsing the other maps).
    mask_widget = mo.vstack([
        mask_mode_dropdown,
        mask_widget_controls, mask_widget_maps], align="start")

    match notebook_mode:
        case "unguided_rollout":
            trajectory_widget=mo.vstack([
                T_slider,
                mask_mode_dropdown,
                mask_widget_controls,
                M_N_widget,
                m_n_widget,
                mo.hstack([traj_row_select, dpi_slider, climatology_rolling_slider], justify="start"),
                trajectories_plot,
                year_trajectory_plot,
            ])
            inspect_states_widget=None
        case "guided_rollout":
            trajectory_widget=mo.vstack([
                guidance_reference_dropdown,
                m_n_widget,
                mo.hstack(
                    [var_dropdown, level_slider],
                    justify="start",
                ),
                delta_widget,
                mo.vstack([
                    mo.hstack([traj_row_select, dpi_slider], justify="start"),
                    mo.hstack(
                        [trajectories_plot]
                        + ([wa_schedule_widget] if wa_schedule_widget is not None else []),
                        justify="start", align="start",
                    ),
                ], align="start")
            ], align="start")
            # config-pinned corners: no side sliders; mask mode, sigma_div and the
            # shift preview render the mask as a job would build it
            mask_widget = mo.vstack([
                mo.hstack([mask_mode_dropdown, zoom_slider, dpi_slider, sigma_div_slider,
                           mask_shift_preview_dropdown, mask_shift_px_slider], justify="start"),
                _mask_maps_row,
            ], align="start")
            inspect_states_widget=inspect_states_widget_make
        case "analyze_rollout":
            trajectory_widget=mo.vstack([
                *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                m_n_widget,
                mo.hstack(
                    [var_dropdown, level_slider],
                    justify="start",
                ),
                mo.vstack([
                    mo.hstack([traj_row_select, dpi_slider], justify="start", align="center"),
                    mo.hstack(
                        [trajectories_plot]
                        + ([state_hist_plot] if state_hist_plot is not None else []),
                        justify="start", align="start",
                    ),
                ], align="start"),
                mo.hstack(
                    [guidance_convergence_t_plot]
                    + ([wa_schedule_widget] if wa_schedule_widget is not None else []),
                    justify="start", align="start",
                ),
            ], align="start")
            # mask section: only the mask-related SWEEP selectors (when actually
            # swept) + the maps; no authoring commands in analyze mode
            _mask_sweep_controls = []
            if len(experiment_params["MASK_MODE"]) > 1:
                _mask_sweep_controls.append(mask_mode_dropdown)
            if len(experiment_params.get("sigma_div", [])) > 1:
                _mask_sweep_controls.append(sweep_extra_dropdowns["sigma_div"])
            if len(experiment_params.get("mask_shift", [])) > 1:
                _mask_sweep_controls.append(sweep_extra_dropdowns["mask_shift"])
            mask_widget = mo.vstack(
                ([mo.hstack(_mask_sweep_controls, justify="start")] if _mask_sweep_controls else [])
                + [mo.hstack([zoom_slider, dpi_slider], justify="start"), _mask_maps_row],
                align="start",
            )
            inspect_states_widget=inspect_states_widget_make
        case _:
            pass
    return inspect_states_widget, mask_widget, trajectory_widget


@app.cell(hide_code=True)
def _(mask_section_checkbox, mo):
    mo.hstack([mask_section_checkbox, mo.md("## Mask")], justify="start", align="center")
    return


@app.cell
def _(mask_section_checkbox, mask_widget):
    mask_widget if mask_section_checkbox.value else None
    return


@app.cell
def _(mo):
    get_mask_center_pt, set_mask_center_pt = mo.state((-4.0, 40.0))
    return get_mask_center_pt, set_mask_center_pt


@app.cell
def _(
    add_map_stats,
    config,
    cool_half_cmap,
    dpi_slider,
    get_slices,
    gt_rollout,
    level,
    map_interactive,
    mask,
    mask_corners,
    mask_section_checkbox,
    n,
    notebook_mode,
    np,
    partition,
    set_mask_center_pt,
    timestamps,
    to_display_units,
    var,
    view_mask_mode,
    visualize_map,
    warm_half_cmap,
    white_zero_cmap,
    zoom_centers,
    zoom_slider,
):
    # the mask section shows the guided variable: guided/analyze pin (partition, var,
    # level) to the loaded config; only the authoring (unguided) mode follows the
    # browsing widgets. The other maps stay widget-driven via the global partition/var/level.
    if notebook_mode == "unguided_rollout":
        mask_partition, mask_var, mask_level = partition, var, level
    else:
        mask_partition, mask_var, mask_level = config.PARTITION, config.VAR, config.LEVEL

    if mask_section_checkbox.value:
        weather_slices = get_slices(gt_rollout, mask_partition, mask_var, mask_level)
        # display units (K -> degC for temperature), like every other absolute map
        weather_slices, _weather_unit = to_display_units(weather_slices, mask_var)
        _wmin, _wmax = float(np.min(weather_slices)), float(np.max(weather_slices))
        if _wmin < 0.0 < _wmax:            # white anchored at 0 (guidance abs-field convention)
            _wcmap, _wcenter = white_zero_cmap, 0.0
        elif _wmin >= 0.0:
            _wcmap, _wcenter = warm_half_cmap, None
        else:
            _wcmap, _wcenter = cool_half_cmap, None
        weather_map = visualize_map(
            weather_slices[n],
            suptitle=f"{timestamps[n]}",
            title=f"partition={mask_partition} | var={mask_var} | level={mask_level}"
            + (f" | [{_weather_unit}]" if _weather_unit else ""),
            interactive=map_interactive,
            cmap=_wcmap,
            vmin=_wmin,
            vmax=_wmax,
            center=_wcenter,
            puck_center=(
                (mask_corners[0] + mask_corners[1]) / 2,
                (mask_corners[2] + mask_corners[3]) / 2,
            ),
            side_lon=mask_corners[1] - mask_corners[0],
            side_lat=mask_corners[3] - mask_corners[2],
            # mask level sets over the field (same rule/style as the mask map;
            # pointless on the flat-topped BBOX mask)
            contour_2d=None if view_mask_mode == "BBOX" else mask,
            contour_levels=8,
            contour_color="black",
            contour_linewidth=0.5,
            mask_2d=mask,
            show_mask=True,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
            figsize=(14, 8),
            dpi=dpi_slider.value,
        )
        weather_map = add_map_stats(weather_map, weather_slices[n])
        if map_interactive:
            weather_map.widget.observe(
                lambda _c: set_mask_center_pt(
                    (weather_map.widget.x[0], weather_map.widget.y[0])
                ),
                names=["x", "y"],
            )
    else:
        weather_map = None
    return mask_level, mask_partition, mask_var, weather_map


@app.cell
def _(
    add_map_stats,
    dpi_slider,
    mask,
    mask_level,
    mask_partition,
    mask_section_checkbox,
    mask_var,
    mcolors,
    np,
    view_mask_mode,
    visualize_map,
    white_zero_cmap,
    zoom_centers,
    zoom_slider,
):
    if mask_section_checkbox.value:
        _mmin, _mmax = float(np.min(mask)), float(np.max(mask))
        # min at exactly 0: linear scale on the WARM half of the colormap only, so the
        # colorbar starts white at 0 and saturates to red at the max (no blue band)
        _mask_cmap = (mcolors.LinearSegmentedColormap.from_list(
            "mask_warm", white_zero_cmap(np.linspace(0.5, 1.0, 256)))
            if _mmin == 0.0 else white_zero_cmap)
        mask_map = visualize_map(
            mask,
            suptitle="mask",
            title=f"partition={mask_partition} | var={mask_var} | level={mask_level}",
            interactive=False,
            cmap=_mask_cmap,
            vmin=_mmin if _mmin < _mmax else -0.001,
            vmax=_mmax if _mmin < _mmax else 0.001,
            # min at 0 -> plain linear norm on the warm half (white at 0); otherwise
            # diverging map centered on the range midpoint
            center=(None if _mmin == 0.0 else 0.5 * (_mmin + _mmax)) if _mmin < _mmax else 0.0,
            # level sets of the displayed mask (incl. cos weight); pointless on the
            # flat-topped BBOX mask, so contours only for the smooth modes
            contour_2d=None if view_mask_mode == "BBOX" else mask,
            contour_levels=8,
            contour_color="black",
            contour_linewidth=0.5,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
            figsize=(14, 8),
            dpi=dpi_slider.value
        )
        mask_map = add_map_stats(mask_map, mask)
    else:
        mask_map = None
    return (mask_map,)


@app.cell(hide_code=True)
def _(inspect_section_checkbox, mo):
    mo.hstack([inspect_section_checkbox, mo.md("## Inspect states")], justify="start", align="center")
    return


@app.cell
def _(inspect_states_widget):
    inspect_states_widget
    return


@app.cell
def _(mo):
    analysis_types = ["absolute", "difference", "sobel_grads", "sobel_diffs"]
    analysis_type_dropdown = mo.ui.dropdown(
        analysis_types,
        value=analysis_types[0],
        label="analysis type: ",
    )
    return (analysis_type_dropdown,)


@app.cell
def _(mo):
    show_mask_switch = mo.ui.checkbox(label="show mask", value=True)
    return (show_mask_switch,)


@app.cell
def _(mo):
    zoom_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=1,
        label="zoom: ",
        debounce=True,
        show_value=True
    )
    return (zoom_slider,)


@app.cell
def _(mo):
    _dpi_options = [20, 30, 40] + list(range(50, 1000, 50))
    dpi_slider = mo.ui.slider(steps=_dpi_options, debounce=True, show_value=True, label="dpi: ")
    return (dpi_slider,)


@app.cell
def _(mo):
    # display labels -> internal mode values (comparisons keep the old values)
    norm_modes = {"own scale": "own_scale", "mask scale": "own_mask_scale", "same scale": "same_scale"}
    norm_mode_dropdown = mo.ui.dropdown(
        norm_modes,
        value="own scale",
        label="norm mode: ",
    )
    zoom_scale_checkbox = mo.ui.checkbox(label="zoom scale")
    # scope of the per-map stats labels (min/max/mean/std)
    stats_scope_dropdown = mo.ui.dropdown(["mask", "zoom", "full"], value="mask", label="stats: ")
    return norm_mode_dropdown, stats_scope_dropdown, zoom_scale_checkbox


@app.cell
def _(get_mask_center, mask_corners):
    zoom_centers = get_mask_center(*mask_corners)
    return (zoom_centers,)


@app.cell
def _(
    get_rollout,
    get_slices,
    gt_N_slices,
    gui_M_N_slices,
    gui_ung_final_xr,
    level,
    m,
    n,
    notebook_mode,
    np,
    partition,
    rollout_id,
    sweep_params,
    ung_M_N_slices,
    var,
):
    if notebook_mode =="guided_rollout":
        ung_curr = ung_M_N_slices[m][n]
        # forecast step n is valid at day n+1; gt_N_slices holds days 0..N
        gt_curr = gt_N_slices[n+1]
        ung_prev = ung_M_N_slices[m][n-1] if n>0 else ung_M_N_slices[m][n]
        gt_prev = gt_N_slices[n]

        gt_gt = gt_curr - gt_prev
        gt_ung = gt_curr - ung_curr

    if notebook_mode =="analyze_rollout":
        ung_curr = ung_M_N_slices[m][n]
        gui_curr = gui_M_N_slices[m][n]
        # forecast step n is valid at day n+1; gt_N_slices holds days 0..N
        gt_curr = gt_N_slices[n+1]
        # ung_prev = ung_M_N_slices[m][n-1] if n>0 else ung_M_N_slices[m][n]
        gui_prev = gui_M_N_slices[m][n-1] if n>0 else gui_M_N_slices[m][n]
        gt_prev = gt_N_slices[n]

        gui_ung_slice = get_slices(gui_ung_final_xr, partition, var, level)
        gui_ung_curr = gui_ung_slice[m][n]
        gui_ung_prev = gui_ung_slice[m][n-1] if n>0 else gui_ung_slice[m][n]

        # r_n = x_n - x_n^det: realized generative residuals at forecast step n.
        # The unguided twin's det comes from ung_det.zarr when stored; this experiment
        # dir lacks it -> fall back to gui_det (exact at n=0, where both rollouts share
        # the initial state; approximate for n>0). Under the fallback,
        # r_n^gui - r_n^gui_ung degenerates to exactly x_n^gui - x_n^gui_ung.
        try:
            _det_n = get_slices(get_rollout("gui_det", rollout_id).sel(sweep_params).compute(),
                                partition, var, level)[m][n]
        except (FileNotFoundError, KeyError):
            _det_n = np.zeros_like(np.asarray(gui_curr))
        try:
            _und_ds = get_rollout("ung_det", rollout_id)
            _und_n = get_slices(_und_ds.sel({_k: _v for _k, _v in sweep_params.items()
                                             if _k in _und_ds.dims}).compute(),
                                partition, var, level)[m][n]
        except (FileNotFoundError, KeyError):
            _und_n = _det_n
        det_n_slice = _det_n  # public: shown as the x_det map in the third diff row
        ung_det_n_slice = _und_n  # public: unguided det, shown in the absolute dets row
        gui_det_res = gui_curr - _det_n
        gui_ung_det_res = gui_ung_curr - _und_n
        res_gui_minus_ung = gui_det_res - gui_ung_det_res

        gt_gt = gt_curr - gt_prev
        gui_gui = gui_curr - gui_prev
        gui_gui_ung = gui_curr - gui_ung_curr
        gui_gt = gui_curr - gt_curr
        gui_ung_gt = gui_ung_curr - gt_curr
        gui_ung_minus_ung = gui_ung_curr - ung_curr
        gui_det_ung = det_n_slice - ung_curr  # gui_det - ung difference row
        gui_det_ung_det = det_n_slice - ung_det_n_slice  # gui_det - ung_det difference row
        gui_det_gt = det_n_slice - gt_curr  # gui_det - gt difference row
        gui_minus_ung = gui_curr - ung_curr
        ung_gt = ung_curr - gt_curr
    return (
        det_n_slice,
        gt_curr,
        gt_gt,
        gt_prev,
        gt_ung,
        gui_curr,
        gui_det_gt,
        gui_det_res,
        gui_det_ung,
        gui_det_ung_det,
        gui_gt,
        gui_gui_ung,
        gui_minus_ung,
        gui_ung_curr,
        gui_ung_det_res,
        gui_ung_gt,
        gui_ung_minus_ung,
        ung_curr,
        ung_det_n_slice,
        ung_gt,
        ung_prev,
    )


@app.cell
def _(np):
    def safe_abs_limits(arrays):
        # keep only arrays that carry any finite value; an all-NaN selection (invalid
        # var/level gate) must not poison the limits with NaN -> TwoSlopeNorm crash
        _finite = [np.asarray(a) for a in arrays if np.isfinite(np.asarray(a)).any()]
        if not _finite:
            return -1.0, 1.0, 0.0
        vmin = min(float(np.nanmin(a)) for a in _finite)
        vmax = max(float(np.nanmax(a)) for a in _finite)
        if not np.isfinite(vmin):
            vmin = -1.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        center = 0.5 * (vmin + vmax)
        center = min(max(center, vmin + 1e-9), vmax - 1e-9)
        return vmin, vmax, center

    return (safe_abs_limits,)


@app.cell
def _(
    add_map_stats,
    analysis_type_dropdown,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    cool_half_cmap,
    cv2,
    det_n_slice,
    dpi_slider,
    gt_curr,
    gt_gt,
    gt_prev,
    gt_ung,
    gui_curr,
    gui_det_gt,
    gui_det_res,
    gui_det_ung,
    gui_det_ung_det,
    gui_gt,
    gui_gui_ung,
    gui_minus_ung,
    gui_ung_curr,
    gui_ung_det_res,
    gui_ung_gt,
    gui_ung_minus_ung,
    inspect_diff_rows,
    inspect_section_checkbox,
    map_interactive,
    mask,
    mask_region,
    mo,
    norm_mode_dropdown,
    notebook_mode,
    np,
    safe_abs_limits,
    show_mask_switch,
    to_display_units,
    ung_curr,
    ung_det_n_slice,
    ung_gt,
    ung_prev,
    var,
    visualize_map,
    warm_half_cmap,
    white_zero_cmap,
    zoom_centers,
    zoom_scale_checkbox,
    zoom_slider,
):
    # color-limit helpers: when 'zoom scale' is on, every limits computation is
    # restricted to the cells visible in the current zoom window (same window math
    # as apply_zoom in map.py, on the display grid of prepare_era5_plot_grid)
    _zoom_region = np.ones((121, 240), dtype=bool)
    if zoom_scale_checkbox.value and int(zoom_slider.value) > 1:
        _z = int(zoom_slider.value)
        _lon_span, _lat_span = 360.0 / _z, 180.0 / _z
        _lo_min = max(-180.0, zoom_centers[0] - _lon_span / 2)
        _lo_max = min(180.0, zoom_centers[0] + _lon_span / 2)
        _la_min = max(-90.0, zoom_centers[1] - _lat_span / 2)
        _la_max = min(90.0, zoom_centers[1] + _lat_span / 2)
        if _lo_max - _lo_min < _lon_span:
            if _lo_min <= -180.0:
                _lo_max = min(180.0, _lo_min + _lon_span)
            elif _lo_max >= 180.0:
                _lo_min = max(-180.0, _lo_max - _lon_span)
        if _la_max - _la_min < _lat_span:
            if _la_min <= -90.0:
                _la_max = min(90.0, _la_min + _lat_span)
            elif _la_max >= 90.0:
                _la_min = max(-90.0, _la_max - _lat_span)
        _lat_e_z = np.linspace(90.0, -90.0, 122); _lat_c_z = 0.5 * (_lat_e_z[:-1] + _lat_e_z[1:])
        _lon_e_z = np.linspace(-180.0, 180.0, 241); _lon_c_z = 0.5 * (_lon_e_z[:-1] + _lon_e_z[1:])
        _zoom_region = (((_lat_c_z >= _la_min) & (_lat_c_z <= _la_max))[:, None]
                        & ((_lon_c_z >= _lo_min) & (_lon_c_z <= _lo_max))[None, :])

    def _finite_or(_v, _d):
        return _v if np.isfinite(_v) else _d


    def _zoom_lim(_a):
        return np.where(_zoom_region, _a, np.nan) if zoom_scale_checkbox.value else _a

    def _lim_zoom(_arrs):
        return safe_abs_limits([_zoom_lim(_a) for _a in _arrs])

    if inspect_section_checkbox.value and notebook_mode =="guided_rollout":
        match analysis_type_dropdown.value:
            case "absolute":
                absolute_panels = [
                    ("$x_{n}^{\\text{gt}}$", gt_curr),
                    ("$x_{n-1}^{\\text{gt}}$", gt_prev),
                    ("$x_{n}^{ung}$", ung_curr),
                    ("$x_{n-1}^{ung}$", ung_prev),
                ]
                absolute_panels = [(_l, to_display_units(_a, var)[0]) for _l, _a in absolute_panels]

                abs_vmin, abs_vmax, abs_center = _lim_zoom(
                    [arr for _, arr in absolute_panels]
                )

                absolute_maps = {}

                for label, arr in absolute_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min, _v_max, _v_center = _lim_zoom([arr])
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # own limits, restricted to inside the mask (nonzero weights)
                        _v_min, _v_max, _v_center = _lim_zoom([np.where(mask_region, arr, np.nan)])
                        arr = np.where(mask_region, arr, np.nan)  # outside -> white
                    else:
                        _v_min, _v_max, _v_center = abs_vmin, abs_vmax, abs_center
                    _arr_wz = np.where(np.abs(arr) < 0.0 / 100.0 * float(np.nanmax(np.abs(arr))), np.nan, arr)
                    # white anchored at 0 (mask-map convention); single-signed ranges
                    # use only the warm/cool half of the colormap
                    _abs_cmap = white_zero_cmap
                    if _v_min < 0.0 < _v_max:
                        _v_center = 0.0
                    elif _v_min >= 0.0:
                        _abs_cmap, _v_center = warm_half_cmap, None
                    else:
                        _abs_cmap, _v_center = cool_half_cmap, None
                    absolute_maps[label] = visualize_map(
                        _arr_wz,
                        cmap=_abs_cmap,
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=0.4,
                        mask_2d=mask,
                        title=label,
                        interactive=map_interactive,
                        vmin=_v_min,
                        vmax=_v_max,
                        center=_v_center,
                        show_mask=show_mask_switch.value,
                        zoom=zoom_slider.value,
                        zoom_center_lon=zoom_centers[0],
                        zoom_center_lat=zoom_centers[1],
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )
                    absolute_maps[label] = add_map_stats(absolute_maps[label], arr)

                curr_map = absolute_maps["$x_{n}^{\\text{gt}}$"]
                prev_map = absolute_maps["$x_{n-1}^{\\text{gt}}$"]
                ung_map = absolute_maps["$x_{n}^{ung}$"]
                ung_prev_map = absolute_maps["$x_{n-1}^{ung}$"]
            case "difference":
                difference_panels = [
                    ("$x_{n}^{\\text{gt}} - x_{n-1}^{\\text{gt}}$", gt_gt),
                    ("$x_{n}^{\\text{gt}} - x_{n}^{\\text{ung}}$", gt_ung),
                ]

                # x_det is an ABSOLUTE field: keep it out of the shared limits, the
                # zero-anchored clamps and the white-below masking of the true diffs
                _ABS_PANELS = {"$x_{n}^{\\text{gui\\_det}}$", "$x_{n}^{\\text{ung|gui}}$", "$x_{n}^{gui}$", "$x_{n}^{ung}$", "$x_{n}^{\\text{gt}}$"}
                diff_vmin = min(float(np.nanmin(_zoom_lim(arr))) for _lbl, arr in difference_panels if _lbl not in _ABS_PANELS)
                diff_vmax = max(float(np.nanmax(_zoom_lim(arr))) for _lbl, arr in difference_panels if _lbl not in _ABS_PANELS)
                # same_scale for the ABSOLUTE panels: shared range across x_det /
                # x_gui_ung / x_gui (kept separate from the zero-centered diff limits)
                _abs_dm_arrs = [arr for _lbl, arr in difference_panels if _lbl in _ABS_PANELS]
                # guided mode has no absolute panels -> keep the shared-abs scale inert
                _abs_dm_vmin = min((float(np.nanmin(_zoom_lim(_a))) for _a in _abs_dm_arrs), default=0.0)
                _abs_dm_vmax = max((float(np.nanmax(_zoom_lim(_a))) for _a in _abs_dm_arrs), default=1.0)

                difference_maps = {}

                for label, arr in difference_panels:
                    _is_abs_panel = label in _ABS_PANELS

                    if _is_abs_panel:
                        if norm_mode_dropdown.value == "same_scale":
                            v_min, v_max = _abs_dm_vmin, max(_abs_dm_vmax, _abs_dm_vmin + 1e-12)
                        elif norm_mode_dropdown.value == "own_mask_scale":
                            _arr_in = np.where(mask_region, arr, np.nan)
                            _any_in = bool(np.isfinite(_arr_in).any())
                            v_min = _finite_or(float(np.nanmin(_zoom_lim(_arr_in))) if _any_in else np.nan, -1.0)
                            v_max = max(_finite_or(float(np.nanmax(_zoom_lim(_arr_in))) if _any_in else np.nan, 1.0), v_min + 1e-12)
                            arr = np.where(mask_region, arr, np.nan)  # outside -> white
                        else:
                            _any_a = bool(np.isfinite(arr).any())
                            v_min = _finite_or(float(np.nanmin(_zoom_lim(arr))) if _any_a else np.nan, -1.0)
                            v_max = max(_finite_or(float(np.nanmax(_zoom_lim(arr))) if _any_a else np.nan, 1.0), v_min + 1e-12)
                    elif norm_mode_dropdown.value == "own_scale":
                        v_min = min(float(np.nanmin(_zoom_lim(arr))), -1e-12)
                        v_max = max(float(np.nanmax(_zoom_lim(arr))), 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # EXACT in-mask min/max, regardless of larger values outside
                        _arr_in = np.where(mask_region, arr, np.nan)
                        v_min = float(np.nanmin(_zoom_lim(_arr_in)))
                        v_max = max(float(np.nanmax(_zoom_lim(_arr_in))), v_min + 1e-12)
                        arr = np.where(mask_region, arr, np.nan)  # outside -> white
                        # sign anchoring: a single-signed coloring area pins the scale
                        # to 0 so cool colors NEVER show positive values (and vice versa)
                        if v_min > 0.0:
                            v_min = 0.0
                        elif v_max < 0.0:
                            v_max = 0.0
                    else:
                        v_min, v_max = diff_vmin, diff_vmax
                    # zero-inclusion clamp only for the zero-centered modes; own-mask
                    # keeps its exact in-mask range. NaN guard applies to all modes.
                    if norm_mode_dropdown.value != "own_mask_scale" and not _is_abs_panel:
                        v_min = min(v_min, -1e-12) if np.isfinite(v_min) else -1.0
                        v_max = max(v_max, 1e-12) if np.isfinite(v_max) else 1.0
                    elif not (np.isfinite(v_min) and np.isfinite(v_max)):
                        v_min, v_max = -1.0, 1.0

                    # absolute panels: white anchored at 0 like the mask map --
                    # straddling ranges center the diverging map at 0; single-signed
                    # ranges use only the warm (all >= 0) or cool (all <= 0) half
                    _cmap_dm = white_zero_cmap
                    if _is_abs_panel:
                        if v_min < 0.0 < v_max:
                            _center_dm = 0.0
                        elif v_min >= 0.0:
                            _cmap_dm, _center_dm = warm_half_cmap, None
                        else:
                            _cmap_dm, _center_dm = cool_half_cmap, None
                    else:
                        # center at 0 (or 0 +/- eps when the range was sign-anchored so the
                        # divergent map keeps white at zero and one-signed shades beyond)
                        _center_dm = 0.0 if v_min < 0.0 < v_max else None
                        if v_min == 0.0 and v_max > 0.0:
                            _center_dm = min(1e-12, 0.5 * v_max)   # strictly inside even for degenerate ranges
                        elif v_max == 0.0 and v_min < 0.0:
                            _center_dm = max(-1e-12, 0.5 * v_min)
                    _absmax_dm = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
                    if (not _is_abs_panel) and _absmax_dm < 1e-9:
                        # (near-)identical fields, e.g. gui_ung - ung with a shared
                        # realization: render the zeros at the colormap's 0 color on a
                        # symmetric scale instead of letting the white-below mask (which
                        # is RELATIVE to the panel max) blank the whole panel
                        v_min, v_max, _center_dm = -1e-9, 1e-9, 0.0
                        _arr_wz = arr
                    elif _is_abs_panel:
                        _arr_wz = arr
                    else:
                        _arr_wz = np.where(np.abs(arr) < 0.0 / 100.0 * _absmax_dm, np.nan, arr)
                    difference_maps[label] = visualize_map(
                        _arr_wz,
                        cmap=_cmap_dm,
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=0.4,
                        mask_2d=mask,
                        title=label,
                        vmin=v_min,
                        vmax=v_max,
                        center=_center_dm,
                        show_mask=show_mask_switch.value,
                        zoom=zoom_slider.value,
                        zoom_center_lon=zoom_centers[0],
                        zoom_center_lat=zoom_centers[1],
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )
                    difference_maps[label] = add_map_stats(difference_maps[label], arr)

                gt_gt_map = difference_maps["$x_{n}^{\\text{gt}} - x_{n-1}^{\\text{gt}}$"]
                gt_ung_map = difference_maps["$x_{n}^{\\text{gt}} - x_{n}^{\\text{ung}}$"]
            case "sobel_grads":
                sobel_grad_widget = None
            case _:
                pass

    if inspect_section_checkbox.value and notebook_mode =="analyze_rollout":
        match analysis_type_dropdown.value:
            case "absolute":
                absolute_panels = [
                    ("$x_{n}^{\\text{gt}}$", gt_curr),
                    ("$x_{n}^{\\text{ung|gui}}$", gui_ung_curr),
                    ("$x_{n}^{gui}$", gui_curr),
                    ("$x_{n}^{ung}$", ung_curr),
                    ("$x_{n}^{\\text{ung\\_det}}$", ung_det_n_slice),
                    ("$x_{n}^{\\text{gui\\_det}}$", det_n_slice),
                ]
                absolute_panels = [(_l, to_display_units(_a, var)[0]) for _l, _a in absolute_panels]

                abs_vmin, abs_vmax, abs_center = _lim_zoom(
                    [arr for _, arr in absolute_panels]
                )

                absolute_maps = {}

                for label, arr in absolute_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min, _v_max, _v_center = _lim_zoom([arr])
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # own limits, restricted to inside the mask (nonzero weights)
                        _v_min, _v_max, _v_center = _lim_zoom([np.where(mask_region, arr, np.nan)])
                        arr = np.where(mask_region, arr, np.nan)  # outside -> white
                    else:
                        _v_min, _v_max, _v_center = abs_vmin, abs_vmax, abs_center
                    _arr_wz = np.where(np.abs(arr) < 0.0 / 100.0 * float(np.nanmax(np.abs(arr))), np.nan, arr)
                    # white anchored at 0 (mask-map convention); single-signed ranges
                    # use only the warm/cool half of the colormap
                    _abs_cmap = white_zero_cmap
                    if _v_min < 0.0 < _v_max:
                        _v_center = 0.0
                    elif _v_min >= 0.0:
                        _abs_cmap, _v_center = warm_half_cmap, None
                    else:
                        _abs_cmap, _v_center = cool_half_cmap, None
                    absolute_maps[label] = visualize_map(
                        _arr_wz,
                        cmap=_abs_cmap,
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=0.4,
                        mask_2d=mask,
                        title=label,
                        interactive=map_interactive,
                        vmin=_v_min,
                        vmax=_v_max,
                        center=_v_center,
                        show_mask=show_mask_switch.value,
                        zoom=zoom_slider.value,
                        zoom_center_lon=zoom_centers[0],
                        zoom_center_lat=zoom_centers[1],
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )
                    absolute_maps[label] = add_map_stats(absolute_maps[label], arr)

                curr_map = absolute_maps["$x_{n}^{\\text{gt}}$"]
                prev_map = absolute_maps["$x_{n}^{\\text{ung|gui}}$"]
                ung_map = absolute_maps["$x_{n}^{ung}$"]
                gui_map = absolute_maps["$x_{n}^{gui}$"]
                ung_det_abs_map = absolute_maps["$x_{n}^{\\text{ung\\_det}}$"]
                gui_det_abs_map = absolute_maps["$x_{n}^{\\text{gui\\_det}}$"]

            case "difference":
                difference_panels = [
                    ("$x_{n}^{ung} - x_{n}^{\\text{gt}}$", ung_gt),
                    ("$x_{n}^{\\text{ung|gui}} - x_{n}^{\\text{gt}}$", gui_ung_gt),
                    ("$x_{n}^{gui} - x_{n}^{\\text{gt}}$", gui_gt),
                    ("$x_{n}^{\\text{ung|gui}} - x_{n}^{\\text{ung}}$", gui_ung_minus_ung),
                    ("$x_{n}^{gui} - x_{n}^{\\text{ung}}$", gui_minus_ung),
                    ("$x_{n}^{\\text{gui_det}} - x_{n}^{\\text{ung}}$", gui_det_ung),
                    ("$x_{n}^{\\text{gui_det}} - x_{n}^{\\text{ung\\_det}}$", gui_det_ung_det),
                    ("$x_{n}^{\\text{gui_det}} - x_{n}^{\\text{gt}}$", gui_det_gt),
                    ("guidance effect: $r_n^{gui} - r_n^{\\text{ung|gui}}$  $(= x_{n}^{gui} - x_{n}^{\\text{ung|gui}})$", gui_gui_ung),
                    ("$r_n^{gui} = x_{n}^{gui} - x_{n}^{\\text{gui\\_det}}$", gui_det_res),
                    ("$r_n^{\\text{ung|gui}} = x_{n}^{\\text{ung|gui}} - x_{n}^{\\text{ung\\_det}}$", gui_ung_det_res),
                    ("$x_{n}^{\\text{gui\\_det}}$", to_display_units(det_n_slice, var)[0]),  # absolute field: K -> degC etc.
                    ("$x_{n}^{\\text{ung|gui}}$", to_display_units(gui_ung_curr, var)[0]),
                    ("$x_{n}^{gui}$", to_display_units(gui_curr, var)[0]),
                    ("$x_{n}^{ung}$", to_display_units(ung_curr, var)[0]),
                    ("$x_{n}^{\\text{gt}}$", to_display_units(gt_curr, var)[0]),
                ]
                # Each panel's inspect-row (aligned to difference_panels order); panels in
                # an unselected row are skipped so we never render maps nobody views.
                # "" = a panel not wired into any row (never displayed).
                _panel_row_keys = [
                    "",                        # x_ung - x_gt (not shown)
                    "gui_ung-gt / gui-gt",     # gui_ung - gt
                    "gui_ung-gt / gui-gt",     # gui - gt
                    "gui_ung-ung / gui-ung",   # gui_ung - ung
                    "gui_ung-ung / gui-ung",   # gui - ung
                    "gui_det-ung",             # gui_det - ung
                    "gui_det-ung",             # gui_det - ung_det
                    "",                        # gui_det - gt (not shown)
                    "x_det / gui_gui_ung",     # gui - gui_ung
                    "r_n^gui_ung / r_n^gui",   # r_n^gui
                    "r_n^gui_ung / r_n^gui",   # r_n^gui_ung
                    "x_det / gui_gui_ung",     # x_det
                    "x_gui_ung / x_gui",       # x_gui_ung
                    "x_gui_ung / x_gui",       # x_gui
                    "x_ung / x_gt",            # x_ung
                    "x_ung / x_gt",            # x_gt
                ]
                assert len(_panel_row_keys) == len(difference_panels)
                assert {_rk for _rk in _panel_row_keys if _rk} <= set(inspect_diff_rows.options)
                _diff_label_row = {_lbl: _rk for (_lbl, _a), _rk in zip(difference_panels, _panel_row_keys)}
                _diff_sel = set(inspect_diff_rows.value)

                # x_det is an ABSOLUTE field: keep it out of the shared limits, the
                # zero-anchored clamps and the white-below masking of the true diffs
                _ABS_PANELS = {"$x_{n}^{\\text{gui\\_det}}$", "$x_{n}^{\\text{ung|gui}}$", "$x_{n}^{gui}$", "$x_{n}^{ung}$", "$x_{n}^{\\text{gt}}$"}
                diff_vmin = min(float(np.nanmin(_zoom_lim(arr))) for _lbl, arr in difference_panels if _lbl not in _ABS_PANELS)
                diff_vmax = max(float(np.nanmax(_zoom_lim(arr))) for _lbl, arr in difference_panels if _lbl not in _ABS_PANELS)
                # same_scale for the ABSOLUTE panels: shared range across x_det /
                # x_gui_ung / x_gui (kept separate from the zero-centered diff limits)
                _abs_dm_arrs = [arr for _lbl, arr in difference_panels if _lbl in _ABS_PANELS]
                # guided mode has no absolute panels -> keep the shared-abs scale inert
                _abs_dm_vmin = min((float(np.nanmin(_zoom_lim(_a))) for _a in _abs_dm_arrs), default=0.0)
                _abs_dm_vmax = max((float(np.nanmax(_zoom_lim(_a))) for _a in _abs_dm_arrs), default=1.0)

                difference_maps = {}

                for label, arr in difference_panels:
                    if _diff_label_row.get(label) not in _diff_sel:
                        difference_maps[label] = None  # not shown -> skip the render
                        continue
                    _is_abs_panel = label in _ABS_PANELS

                    if _is_abs_panel:
                        if norm_mode_dropdown.value == "same_scale":
                            v_min, v_max = _abs_dm_vmin, max(_abs_dm_vmax, _abs_dm_vmin + 1e-12)
                        elif norm_mode_dropdown.value == "own_mask_scale":
                            _arr_in = np.where(mask_region, arr, np.nan)
                            _any_in = bool(np.isfinite(_arr_in).any())
                            v_min = _finite_or(float(np.nanmin(_zoom_lim(_arr_in))) if _any_in else np.nan, -1.0)
                            v_max = max(_finite_or(float(np.nanmax(_zoom_lim(_arr_in))) if _any_in else np.nan, 1.0), v_min + 1e-12)
                            arr = np.where(mask_region, arr, np.nan)  # outside -> white
                        else:
                            _any_a = bool(np.isfinite(arr).any())
                            v_min = _finite_or(float(np.nanmin(_zoom_lim(arr))) if _any_a else np.nan, -1.0)
                            v_max = max(_finite_or(float(np.nanmax(_zoom_lim(arr))) if _any_a else np.nan, 1.0), v_min + 1e-12)
                    elif norm_mode_dropdown.value == "own_scale":
                        v_min = min(float(np.nanmin(_zoom_lim(arr))), -1e-12)
                        v_max = max(float(np.nanmax(_zoom_lim(arr))), 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # EXACT in-mask min/max, regardless of larger values outside
                        _arr_in = np.where(mask_region, arr, np.nan)
                        v_min = float(np.nanmin(_zoom_lim(_arr_in)))
                        v_max = max(float(np.nanmax(_zoom_lim(_arr_in))), v_min + 1e-12)
                        arr = np.where(mask_region, arr, np.nan)  # outside -> white
                        # sign anchoring: a single-signed coloring area pins the scale
                        # to 0 so cool colors NEVER show positive values (and vice versa)
                        if v_min > 0.0:
                            v_min = 0.0
                        elif v_max < 0.0:
                            v_max = 0.0
                    else:
                        v_min, v_max = diff_vmin, diff_vmax
                    # zero-inclusion clamp only for the zero-centered modes; own-mask
                    # keeps its exact in-mask range. NaN guard applies to all modes.
                    if norm_mode_dropdown.value != "own_mask_scale" and not _is_abs_panel:
                        v_min = min(v_min, -1e-12) if np.isfinite(v_min) else -1.0
                        v_max = max(v_max, 1e-12) if np.isfinite(v_max) else 1.0
                    elif not (np.isfinite(v_min) and np.isfinite(v_max)):
                        v_min, v_max = -1.0, 1.0

                    # absolute panels: white anchored at 0 like the mask map --
                    # straddling ranges center the diverging map at 0; single-signed
                    # ranges use only the warm (all >= 0) or cool (all <= 0) half
                    _cmap_dm = white_zero_cmap
                    if _is_abs_panel:
                        if v_min < 0.0 < v_max:
                            _center_dm = 0.0
                        elif v_min >= 0.0:
                            _cmap_dm, _center_dm = warm_half_cmap, None
                        else:
                            _cmap_dm, _center_dm = cool_half_cmap, None
                    else:
                        # center at 0 (or 0 +/- eps when the range was sign-anchored so the
                        # divergent map keeps white at zero and one-signed shades beyond)
                        _center_dm = 0.0 if v_min < 0.0 < v_max else None
                        if v_min == 0.0 and v_max > 0.0:
                            _center_dm = min(1e-12, 0.5 * v_max)   # strictly inside even for degenerate ranges
                        elif v_max == 0.0 and v_min < 0.0:
                            _center_dm = max(-1e-12, 0.5 * v_min)
                    _absmax_dm = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
                    if (not _is_abs_panel) and _absmax_dm < 1e-9:
                        # (near-)identical fields, e.g. gui_ung - ung with a shared
                        # realization: render the zeros at the colormap's 0 color on a
                        # symmetric scale instead of letting the white-below mask (which
                        # is RELATIVE to the panel max) blank the whole panel
                        v_min, v_max, _center_dm = -1e-9, 1e-9, 0.0
                        _arr_wz = arr
                    elif _is_abs_panel:
                        _arr_wz = arr
                    else:
                        _arr_wz = np.where(np.abs(arr) < 0.0 / 100.0 * _absmax_dm, np.nan, arr)
                    difference_maps[label] = visualize_map(
                        _arr_wz,
                        cmap=_cmap_dm,
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=0.4,
                        mask_2d=mask,
                        title=label,
                        vmin=v_min,
                        vmax=v_max,
                        center=_center_dm,
                        show_mask=show_mask_switch.value,
                        zoom=zoom_slider.value,
                        zoom_center_lon=zoom_centers[0],
                        zoom_center_lat=zoom_centers[1],
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )
                    difference_maps[label] = add_map_stats(difference_maps[label], arr)

                ung_gt_map = difference_maps["$x_{n}^{ung} - x_{n}^{\\text{gt}}$"]
                gui_ung_gt_map = difference_maps["$x_{n}^{\\text{ung|gui}} - x_{n}^{\\text{gt}}$"]
                gui_gt_map = difference_maps["$x_{n}^{gui} - x_{n}^{\\text{gt}}$"]
                gui_ung_ung_map = difference_maps["$x_{n}^{\\text{ung|gui}} - x_{n}^{\\text{ung}}$"]
                gui_ung_map = difference_maps["guidance effect: $r_n^{gui} - r_n^{\\text{ung|gui}}$  $(= x_{n}^{gui} - x_{n}^{\\text{ung|gui}})$"]
                gui_minus_ung_map = difference_maps["$x_{n}^{gui} - x_{n}^{\\text{ung}}$"]
                gui_det_ung_map = difference_maps["$x_{n}^{\\text{gui_det}} - x_{n}^{\\text{ung}}$"]
                gui_det_ung_det_map = difference_maps["$x_{n}^{\\text{gui_det}} - x_{n}^{\\text{ung\\_det}}$"]
                gui_det_gt_map = difference_maps["$x_{n}^{\\text{gui_det}} - x_{n}^{\\text{gt}}$"]
                r_n_map = difference_maps["$r_n^{gui} = x_{n}^{gui} - x_{n}^{\\text{gui\\_det}}$"]
                r_n_gui_ung_map = difference_maps["$r_n^{\\text{ung|gui}} = x_{n}^{\\text{ung|gui}} - x_{n}^{\\text{ung\\_det}}$"]
                x_det_map = difference_maps["$x_{n}^{\\text{gui\\_det}}$"]
                x_gui_ung_abs_dmap = difference_maps["$x_{n}^{\\text{ung|gui}}$"]
                x_gui_abs_dmap = difference_maps["$x_{n}^{gui}$"]
                x_ung_abs_dmap = difference_maps["$x_{n}^{ung}$"]
                x_gt_abs_dmap = difference_maps["$x_{n}^{\\text{gt}}$"]
            case "sobel_grads":
                gradmap_gt_x = cv2.Sobel(gt_curr, cv2.CV_32F, 1, 0, ksize=3)
                gradmap_gt_y = cv2.Sobel(gt_curr, cv2.CV_32F, 0, 1, ksize=3)
                gradmap_gt_mag = np.sqrt(gradmap_gt_x**2 + gradmap_gt_y**2)

                gradmap_gui_x = cv2.Sobel(gui_curr, cv2.CV_32F, 1, 0, ksize=3)
                gradmap_gui_y = cv2.Sobel(gui_curr, cv2.CV_32F, 0, 1, ksize=3)
                gradmap_gui_mag = np.sqrt(gradmap_gui_x**2 + gradmap_gui_y**2)

                gradmap_ung_x = cv2.Sobel(ung_curr, cv2.CV_32F, 1, 0, ksize=3)
                gradmap_ung_y = cv2.Sobel(ung_curr, cv2.CV_32F, 0, 1, ksize=3)
                gradmap_ung_mag = np.sqrt(gradmap_ung_x**2 + gradmap_ung_y**2)

                gradmap_gui_ung_x = cv2.Sobel(gui_ung_curr, cv2.CV_32F, 1, 0, ksize=3)
                gradmap_gui_ung_y = cv2.Sobel(gui_ung_curr, cv2.CV_32F, 0, 1, ksize=3)
                gradmap_gui_ung_mag = np.sqrt(gradmap_gui_ung_x**2 + gradmap_gui_ung_y**2)

                gradmap_mag_panels = [
                    (r"$\|\nabla x_n^{\text{gt}}\|$", gradmap_gt_mag),
                    (r"$\|\nabla x_n^{\text{ung}}\|$", gradmap_ung_mag),
                    (r"$\|\nabla x_n^{\text{gui}}\|$", gradmap_gui_mag),
                    (r"$\|\nabla (x_n^{\text{ung|gui}})\|$", gradmap_gui_ung_mag),
                ]

                gradmap_mag_vmin = min(float(np.nanmin(_zoom_lim(gradmap_arr))) for _, gradmap_arr in gradmap_mag_panels)
                gradmap_mag_vmax = max(float(np.nanmax(_zoom_lim(gradmap_arr))) for _, gradmap_arr in gradmap_mag_panels)

                gradmap_figures = []

                for gradmap_title, gradmap_arr in gradmap_mag_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min = float(np.nanmin(_zoom_lim(gradmap_arr)))
                        _v_max = max(float(np.nanmax(_zoom_lim(gradmap_arr))), _v_min + 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        _arr_in = np.where(mask_region, gradmap_arr, np.nan)
                        _v_min = float(np.nanmin(_zoom_lim(_arr_in)))
                        _v_max = max(float(np.nanmax(_zoom_lim(_arr_in))), _v_min + 1e-12)
                        gradmap_arr = np.where(mask_region, gradmap_arr, np.nan)  # outside -> white
                    else:
                        _v_min, _v_max = gradmap_mag_vmin, gradmap_mag_vmax
                    print(_v_min, _v_max)
                    _gradmap_wz = np.where(np.abs(gradmap_arr) < 0.0 / 100.0 * float(np.nanmax(np.abs(gradmap_arr))), np.nan, gradmap_arr)
                    gradmap_figures.append(
                        visualize_map(
                            _gradmap_wz,
                            cmap=white_zero_cmap,
                            contour_2d=gradmap_arr if contour_checkbox.value else None,
                            contour_levels=contour_levels_slider.value,
                            contour_color=contour_color_dropdown.value,
                            contour_linewidth=0.4,
                            mask_2d=mask,
                            title=gradmap_title,
                            vmin=-1 if _v_min == _v_max else _v_min,
                            vmax=1 if _v_min == _v_max else _v_max,
                            center=0.0 if _v_min < 0.0 < _v_max else None,
                            show_mask=show_mask_switch.value,
                            zoom=zoom_slider.value,
                            zoom_center_lon=zoom_centers[0],
                            zoom_center_lat=zoom_centers[1],
                            dpi=dpi_slider.value,
                            figsize=(14, 8),
                        )
                    )
                    gradmap_figures[-1] = add_map_stats(gradmap_figures[-1], gradmap_arr)

                sobel_grad_widget = mo.vstack(
                    [
                        mo.hstack(gradmap_figures[:2], justify="start"),
                        mo.hstack(gradmap_figures[2:], justify="start"),
                    ]
                )
            case "sobel_diffs":
                sobel_gt_x = cv2.Sobel(gt_curr, cv2.CV_32F, 1, 0, ksize=3)
                sobel_gt_y = cv2.Sobel(gt_curr, cv2.CV_32F, 0, 1, ksize=3)
                sobel_gt_mag = np.sqrt(sobel_gt_x**2 + sobel_gt_y**2)

                sobel_gui_x = cv2.Sobel(gui_curr, cv2.CV_32F, 1, 0, ksize=3)
                sobel_gui_y = cv2.Sobel(gui_curr, cv2.CV_32F, 0, 1, ksize=3)
                sobel_gui_mag = np.sqrt(sobel_gui_x**2 + sobel_gui_y**2)

                sobel_ung_x = cv2.Sobel(ung_curr, cv2.CV_32F, 1, 0, ksize=3)
                sobel_ung_y = cv2.Sobel(ung_curr, cv2.CV_32F, 0, 1, ksize=3)
                sobel_ung_mag = np.sqrt(sobel_ung_x**2 + sobel_ung_y**2)

                sobel_gui_ung_x = cv2.Sobel(gui_ung_curr, cv2.CV_32F, 1, 0, ksize=3)
                sobel_gui_ung_y = cv2.Sobel(gui_ung_curr, cv2.CV_32F, 0, 1, ksize=3)
                sobel_gui_ung_mag = np.sqrt(sobel_gui_ung_x**2 + sobel_gui_ung_y**2)

                sobel_diff_panels = [
                    (r"$\|\nabla x_n^{\text{gt}}\| - \|\nabla x_n^{\text{ung}}\|$", sobel_gt_mag - sobel_ung_mag),
                    (r"$\|\nabla x_n^{\text{gui}}\| - \|\nabla x_n^{\text{ung}}\|$", sobel_gui_mag - sobel_ung_mag),
                    (r"$\|\nabla (x_n^{\text{ung|gui}})\| - \|\nabla x_n^{\text{ung}}\|$", sobel_gui_ung_mag - sobel_ung_mag),
                    (r"$\|\nabla x_n^{\text{gui}}\| - \|\nabla (x_n^{\text{ung|gui}})\|$", sobel_gui_mag - sobel_gui_ung_mag),
                ]

                sobel_diff_vmin = min(float(np.nanmin(_zoom_lim(arr))) for _, arr in sobel_diff_panels)
                sobel_diff_vmax = max(float(np.nanmax(_zoom_lim(arr))) for _, arr in sobel_diff_panels)

                sobel_diff_figures = []

                for sobel_diff_title, sobel_diff_arr in sobel_diff_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min = min(float(np.nanmin(_zoom_lim(sobel_diff_arr))), -1e-12)
                        _v_max = max(float(np.nanmax(_zoom_lim(sobel_diff_arr))), 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # EXACT in-mask min/max, regardless of larger values outside
                        _arr_in = np.where(mask_region, sobel_diff_arr, np.nan)
                        _v_min = float(np.nanmin(_zoom_lim(_arr_in)))
                        _v_max = max(float(np.nanmax(_zoom_lim(_arr_in))), _v_min + 1e-12)
                        sobel_diff_arr = np.where(mask_region, sobel_diff_arr, np.nan)  # outside -> white
                    else:
                        _v_min, _v_max = min(sobel_diff_vmin, -1e-12), max(sobel_diff_vmax, 1e-12)
                    _sobel_diff_wz = np.where(np.abs(sobel_diff_arr) < 0.0 / 100.0 * float(np.nanmax(np.abs(sobel_diff_arr))), np.nan, sobel_diff_arr)
                    sobel_diff_figures.append(
                        visualize_map(
                            _sobel_diff_wz,
                            cmap=white_zero_cmap,
                            contour_2d=sobel_diff_arr if contour_checkbox.value else None,
                            contour_levels=contour_levels_slider.value,
                            contour_color=contour_color_dropdown.value,
                            contour_linewidth=0.4,
                            mask_2d=mask,
                            title=sobel_diff_title,
                            vmin=-1 if _v_min == _v_max else _v_min,
                            vmax=1 if _v_min == _v_max else _v_max,
                            center=0.0 if _v_min < 0.0 < _v_max else None,
                            show_mask=show_mask_switch.value,
                            zoom=zoom_slider.value,
                            zoom_center_lon=zoom_centers[0],
                            zoom_center_lat=zoom_centers[1],
                            dpi=dpi_slider.value,
                            figsize=(14, 8),
                        )
                    )
                    sobel_diff_figures[-1] = add_map_stats(sobel_diff_figures[-1], sobel_diff_arr)

                sobel_diffs_widget = mo.vstack(
                    [
                        mo.hstack(sobel_diff_figures[:2], justify="start"),
                        mo.hstack(sobel_diff_figures[2:], justify="start"),
                    ]
                )
            case _:
                sobel_grad_widget = None
    return (
        curr_map,
        gt_gt_map,
        gt_ung_map,
        gui_det_abs_map,
        gui_det_ung_det_map,
        gui_det_ung_map,
        gui_gt_map,
        gui_map,
        gui_minus_ung_map,
        gui_ung_gt_map,
        gui_ung_map,
        gui_ung_ung_map,
        prev_map,
        r_n_gui_ung_map,
        r_n_map,
        sobel_diffs_widget,
        sobel_grad_widget,
        ung_det_abs_map,
        ung_map,
        ung_prev_map,
        x_det_map,
        x_gt_abs_dmap,
        x_gui_abs_dmap,
        x_gui_ung_abs_dmap,
        x_ung_abs_dmap,
    )


@app.cell
def _(
    analysis_type_dropdown,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    curr_map,
    dpi_slider,
    gt_gt_map,
    gt_ung_map,
    gui_det_abs_map,
    gui_det_ung_det_map,
    gui_det_ung_map,
    gui_gt_map,
    gui_map,
    gui_minus_ung_map,
    gui_ung_gt_map,
    gui_ung_map,
    gui_ung_ung_map,
    inspect_abs_rows,
    inspect_diff_rows,
    inspect_section_checkbox,
    level_slider,
    m_slider,
    mo,
    n_slider,
    norm_mode_dropdown,
    notebook_mode,
    prev_map,
    r_n_gui_ung_map,
    r_n_map,
    show_mask_switch,
    sobel_diffs_widget,
    sobel_grad_widget,
    stats_scope_dropdown,
    sweep_params_widget,
    ung_det_abs_map,
    ung_map,
    ung_prev_map,
    var_dropdown,
    x_det_map,
    x_gt_abs_dmap,
    x_gui_abs_dmap,
    x_gui_ung_abs_dmap,
    x_ung_abs_dmap,
    zoom_scale_checkbox,
    zoom_slider,
):
    # fallbacks when the section is unchecked
    inspect_states_widget_make = None
    common_controls = None
    if inspect_section_checkbox.value and notebook_mode == "guided_rollout":
        common_controls = [
            mo.hstack([analysis_type_dropdown, dpi_slider], justify="start"),
            mo.hstack([m_slider, n_slider], justify="start"),
            mo.hstack(
                [var_dropdown, level_slider],
                justify="start",
            ),
        ]

        match analysis_type_dropdown.value:
            case "absolute":
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, zoom_scale_checkbox, stats_scope_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_color_dropdown], justify="start", align="center"),
                        mo.hstack([curr_map, prev_map], justify="start"),
                        mo.hstack([ung_map, ung_prev_map], justify="start"),
                    ],
                    justify="start",
                )

            case "difference":
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, zoom_scale_checkbox, stats_scope_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_color_dropdown], justify="start", align="center"),
                        mo.hstack([gt_gt_map, gt_ung_map], justify="start")
                    ], justify="start",
                )
            case _:
                # sobel_grads (and any other type) has no guided-mode panel; show a
                # placeholder so inspect_states_widget_make is always defined here.
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        # mo.md(f"_'{analysis_type_dropdown.value}' analysis is not available in guided_rollout mode._"),
                    ],
                    justify="start",
                )

    if inspect_section_checkbox.value and notebook_mode == "analyze_rollout":
        common_controls = [
            mo.hstack([analysis_type_dropdown, dpi_slider], justify="start"),
            mo.hstack([n_slider, m_slider], justify="start"),
            mo.hstack(
                [var_dropdown, level_slider],
                justify="start",
            ),
        ]

        match analysis_type_dropdown.value:
            case "absolute":
                _rows = [
                    ("curr / prev", [curr_map, ung_map]),
                    ("gui / ung", [gui_map, prev_map]),
                    ("dets", [ung_det_abs_map, gui_det_abs_map]),
                ]
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, zoom_scale_checkbox, stats_scope_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_color_dropdown], justify="start", align="center"),
                        inspect_abs_rows,
                        mo.vstack(
                            [mo.hstack(_maps, justify="start") for _k, _maps in _rows if _k in inspect_abs_rows.value],
                            justify="start", align="start",
                        ),
                    ],
                    justify="start",
                )

            case "difference":
                _rows = [
                    ("gui_ung-gt / gui-gt", [gui_ung_gt_map, gui_gt_map]),
                    ("gui_det-ung", [gui_det_ung_det_map, gui_det_ung_map]),
                    ("gui_ung-ung / gui-ung", [gui_ung_ung_map, gui_minus_ung_map]),
                    ("r_n^gui_ung / r_n^gui", [r_n_gui_ung_map, r_n_map]),
                    ("x_det / gui_gui_ung", [x_det_map, gui_ung_map]),
                    ("x_gui_ung / x_gui", [x_gui_ung_abs_dmap, x_gui_abs_dmap]),
                    ("x_ung / x_gt", [x_ung_abs_dmap, x_gt_abs_dmap]),
                ]
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, zoom_scale_checkbox, stats_scope_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_color_dropdown], justify="start", align="center"),
                        inspect_diff_rows,
                        mo.vstack(
                            [mo.hstack(_maps, justify="start") for _k, _maps in _rows if _k in inspect_diff_rows.value],
                            justify="start", align="start",
                        ),
                    ], justify="start",
                )

            case "sobel_grads":
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, zoom_scale_checkbox, stats_scope_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_color_dropdown], justify="start", align="center"),
                        sobel_grad_widget
                    ], justify="start",
                )
            case "sobel_diffs":
                inspect_states_widget_make = mo.vstack(
                    [
                        *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, zoom_scale_checkbox, stats_scope_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_color_dropdown], justify="start", align="center"),
                        sobel_diffs_widget
                    ], justify="start",
                )
            case _:
                pass
    return (inspect_states_widget_make,)


@app.cell(hide_code=True)
def _(mo, trajectories_section_checkbox):
    mo.hstack([trajectories_section_checkbox, mo.md("## Trajectories")], justify="start", align="center")
    return


@app.cell
def _(timestamp_widget):
    timestamp_widget
    return


@app.cell
def _(trajectories_section_checkbox, trajectory_widget):
    trajectory_widget if trajectories_section_checkbox.value else None
    return


@app.cell
def _(mo):
    # chart/trace selector for the rollout-trajectories plot, mirroring the
    # cross-checks / flow-analysis / inspect-states multiselects.
    _traj_rows = ["unguided", "guided", "guided_unguided", "gui_det",
                  "target_guidance", "target_pct_profile", "dist_bands"]
    traj_row_select = mo.ui.multiselect(_traj_rows, value=_traj_rows, label="charts: ")
    return (traj_row_select,)


@app.cell
def _():
    # traj_checks
    return


@app.cell
def _(
    N,
    config,
    delta_trajectories,
    delta_trajectory,
    det_m_trajectory,
    dpi_slider,
    gt_trajectory,
    gui_M_N_trajectories,
    gui_m_trajectory,
    gui_ung_m_trajectory,
    m,
    notebook_mode,
    np,
    plot_trajectories,
    target_guidance_M_N_trajectories,
    target_guidance_trajectories_all,
    timestamps,
    to_display_units,
    traj_row_select,
    trajectories_section_checkbox,
    ung_M_N_trajectories,
    ung_m_trajectory,
    var,
    view_mask_mode,
):

    var_check = (var==config.VAR if notebook_mode in ("guided_rollout", "analyze_rollout") else False)

    # display units: K -> degC for temperature variables. Applies to every absolute
    # trace (members, ensembles, targets, ground truth); the percentage profiles are
    # relative and stay unchanged.
    def _disp(_a):
        return to_display_units(_a, var)[0] if _a is not None else None
    _unit = to_display_units(0.0, var)[1]

    trajectories_plot = plot_trajectories(
        timestamps=timestamps,
        var=var,
        m=m,
        n=None,  # trajectories plot the full n-axis; decoupled from the n-slider
        guided_member=_disp(gui_m_trajectory) if ("guided" in traj_row_select.value) else None,
        unguided_member=_disp(ung_m_trajectory) if ("unguided" in traj_row_select.value) else None,
        guided_unguided_member=_disp(gui_ung_m_trajectory) if ("guided_unguided" in traj_row_select.value) else None,
        det_member=_disp(det_m_trajectory) if ("gui_det" in traj_row_select.value and det_m_trajectory is not None) else None,
        guided_ensemble=_disp(gui_M_N_trajectories) if ("dist_bands" in traj_row_select.value) else None,
        unguided_ensemble=_disp(ung_M_N_trajectories) if ("dist_bands" in traj_row_select.value) else None,
        target_guidance_ensemble=_disp(target_guidance_M_N_trajectories) if (("dist_bands" in traj_row_select.value) and ("target_guidance" in traj_row_select.value) and var_check) else None,
        target_guidance_trajectory=[_disp(_t) for _t in target_guidance_trajectories_all] if (("target_guidance" in traj_row_select.value) and var_check) else None,
        ground_truth=_disp(gt_trajectory),
        ground_truth_label=f"Ground truth ({view_mask_mode})",
        delta_trajectories=(
            ([[0] + list(np.asarray(_t)[:N]) for _t in delta_trajectories] if notebook_mode == "guided_rollout"
             else [[0] + list(np.asarray(delta_trajectory)[:N])])
            if (("target_pct_profile" in traj_row_select.value) and notebook_mode in ("guided_rollout", "analyze_rollout")) else None
        ),
        annotate_target_guidance=(notebook_mode == "guided_rollout"),
        show_guided_mean=False,
        show_unguided_mean=False,
        title=f"rollout trajectories",
        subtitle=f"{var} | mask-averaged",
        ylabel=f"Mask-averaged value [{_unit}]" if _unit else "Mask-averaged value",
        figsize=(22, 6),
        dpi=dpi_slider.value
    ) if trajectories_section_checkbox.value else None
    return (trajectories_plot,)


@app.cell
def _(flow_section_checkbox, mo):
    mo.hstack([flow_section_checkbox, mo.md("## Flow analysis")], justify="start", align="center")
    return


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_norms_checkbox,
    cross_section_checkbox,
    cross_traces,
    dpi_slider,
    m,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    red,
    row_keys,
):
    if cross_section_checkbox.value and notebook_mode == "analyze_rollout" and cross_norms_checkbox.value:
        _vf_traces, _bands = cross_traces(red["vfs_l2"].isel(n=n), "t", "l2", m, **{**cross_ctl, "k": 10**9})
        _vf_traces = {_k: _vf_traces[_k] for _k in row_keys["vfs"] if _k in _vf_traces}
        _bands = {_k: _bands[_k] for _k in _vf_traces if _k in _bands} if _bands else None
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _vf_traces.values()), default=1)))
        vf_norms_plot = plot_trajectory(_vf_traces, title="Vector field norm — ung step",
            subtitle=r"$\|\mathrm{vf}_t\|$ (before kick)", step=None, color_map=color_for(_vf_traces), bands=_bands,
            figsize=(_w, 6), dpi=dpi_slider.value, prepend_zero=False, start_index=1, mirror_right_axis=True)
        _ax_fr = vf_norms_plot.axes[0] if hasattr(vf_norms_plot, "axes") and vf_norms_plot.axes else None
        if _ax_fr is not None:
            _n_fr = max((len(_v) for _v in _vf_traces.values()), default=1)
            _ax_fr.set_xlim(-1.0, _n_fr + 0.4)
            _ax_fr.set_xticks(np.arange(0, _n_fr + 1))
    else:
        vf_norms_plot = None
    return (vf_norms_plot,)


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_norms_checkbox,
    cross_section_checkbox,
    cross_traces,
    dpi_slider,
    m,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    red,
    row_keys,
):
    if cross_section_checkbox.value and notebook_mode == "analyze_rollout" and cross_norms_checkbox.value:
        _gvf_traces, _bands = cross_traces(red["gui_vfs_l2"].isel(n=n), "t", "l2", m, **{**cross_ctl, "k": 10**9})
        _gvf_traces = {_k: _gvf_traces[_k] for _k in row_keys["gui_vfs"] if _k in _gvf_traces}
        _bands = {_k: _bands[_k] for _k in _gvf_traces if _k in _bands} if _bands else None
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _gvf_traces.values()), default=1)))
        guided_vf_norms_plot = plot_trajectory(_gvf_traces, title="Vector field norm — gui step",
            subtitle=r"$\|\mathrm{vf}^{\mathrm{gui}}_t\|$", step=None, color_map=color_for(_gvf_traces), bands=_bands,
            figsize=(_w, 6), dpi=dpi_slider.value, prepend_zero=False, start_index=1, mirror_right_axis=True)
        _ax_fr = guided_vf_norms_plot.axes[0] if hasattr(guided_vf_norms_plot, "axes") and guided_vf_norms_plot.axes else None
        if _ax_fr is not None:
            _n_fr = max((len(_v) for _v in _gvf_traces.values()), default=1)
            _ax_fr.set_xlim(-1.0, _n_fr + 0.4)
            _ax_fr.set_xticks(np.arange(0, _n_fr + 1))
    else:
        guided_vf_norms_plot = None
    return (guided_vf_norms_plot,)


@app.cell
def _(T, mo, notebook_mode):
    t_slider = mo.ui.slider(
        # T (flow steps) is constant across sweep points; deriving the range from the
        # config-pinned T (not the reloaded data cube) keeps t_slider from resetting when a
        # sweep axis changes.
        steps=range(1, T + 1) if notebook_mode == "analyze_rollout" else range(1, 26),
        value=1,
        label=get_label("t", T),
        debounce=True,
        show_value=True
    )
    return (t_slider,)


@app.cell
def _(
    STATS_PATH,
    VARIABLES_DICT,
    get_guidance_schedule,
    get_rollout,
    guidance_mode_dropdown,
    guided_xr,
    notebook_mode,
    np,
    open_unguided_traj_ds,
    rollout_id,
    sweep_params,
    torch,
    unguided_final_state_ds,
    xr,
):
    # NOT gated on flow_section_checkbox: these are LAZY .sel handles consumed by
    # the data/trajectory/landing cells outside the flow section
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        # Keep the heavy per-flow-step cubes LAZY: only `.sel` the sweep point here.
        # NEW trace format (res.zarr present): stores hold raw primitives only --
        # grads = dL/dz, vfs = u*s_t, res = noisy state z_t. Everything derived
        # (gui_vec, gui_vf, gui_res, clean_preds) is reconstructed below via exact
        # affine identities; no model needed. LEGACY stores load directly.
        grads_xr = get_rollout("grads", rollout_id).sel(sweep_params)
        vfs_xr = get_rollout("vfs", rollout_id).sel(sweep_params)
        gui_ung_xr = open_unguided_traj_ds(rollout_id, "gui_ung", sweep_params)
        # ung/gui_ung migrated to the latent format: gui_ung_xr is the reconstructed physical
        # state trajectory x_t = x_det + sigma_r*z_t (length T); gui_ung_final_xr is the true
        # converged final state x_T = x_det + sigma_r*z_T (z_T-spliced, falls back to x_{T-1}).
        gui_ung_final_xr = unguided_final_state_ds(rollout_id, "gui_ung", sweep_params)

        try:
            res_xr = get_rollout("res", rollout_id).sel(sweep_params)
        except FileNotFoundError:
            res_xr = None  # legacy rollout (pre raw-primitives format)

        # residual denorm scaler c per var (level vars: per-level vector), as in GuidedFlow
        _rsc = torch.load(STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False)
        res_scale_map = {}
        for _vi, _v in enumerate(VARIABLES_DICT["surface"]):
            res_scale_map[_v] = float(_rsc["surface"][_vi].squeeze())
        _lev_np = _rsc["level"].squeeze(-1).squeeze(-1).numpy()
        for _vi, _v in enumerate(VARIABLES_DICT["level"]):
            _arr = _lev_np[_vi] * (3.0 if _v == "vertical_velocity" else 1.0)
            res_scale_map[_v] = xr.DataArray(_arr, dims=("level",), coords={"level": grads_xr.level})

        if res_xr is None:
            # ---- LEGACY: applied grads (x s_t), gui_vfs and clean_preds stored directly ----
            clean_preds_xr = get_rollout("clean_preds", rollout_id).sel(sweep_params)
            gui_vfs_xr = get_rollout("gui_vfs", rollout_id).sel(sweep_params)
            gui_vec_xr = None
            gui_res_xr = None
        else:
            # ---- NEW FORMAT: reconstruct in-memory (stays lazy in dask) ----

            # lambda_t[m, n, t] = w_t * a_t from the guidance_schedule sidecar
            _sched_sel = dict(sweep_params)  # records store the coord-label sweep form
            _recs = get_guidance_schedule(rollout_id, _sched_sel, method=guidance_mode_dropdown.value)
            _M, _N, _T = grads_xr.sizes["m"], grads_xr.sizes["n"], grads_xr.sizes["t"]
            _lam = np.full((_M, _N, _T), np.nan)
            for _r in _recs:
                _wv, _av = np.asarray(_r["w_t"], float), np.asarray(_r["a_t"], float)
                _cv = np.asarray(_r.get("c_t", np.ones_like(_wv)), float)
                _gv = np.asarray(_r.get("g_norm_t", np.ones_like(_wv)), float)
                if len(_wv) == _T and _r["m"] < _M and _r["n"] < _N:
                    # raw-gradient multiplier: (w*a*c)/g_norm; legacy records -> w*a
                    _lam[_r["m"], _r["n"], :] = _wv * _av * _cv / np.where(_gv != 0, _gv, 1.0)
            lambda_t_xr = xr.DataArray(
                _lam, dims=("m", "n", "t"),
                coords={"m": grads_xr.m, "n": grads_xr.n, "t": grads_xr.t},
            )

            # flow grids: s_t (noise level) and h_t (Euler step; last step h = s)
            _s_np = np.linspace(1000, 1, _T) / 1000
            _h_np = np.empty_like(_s_np); _h_np[:-1] = _s_np[:-1] - _s_np[1:]; _h_np[-1] = _s_np[-1]
            _s_da = xr.DataArray(_s_np, dims=("t",), coords={"t": grads_xr.t})
            _h_da = xr.DataArray(_h_np, dims=("t",), coords={"t": grads_xr.t})


            # exact identities from stored primitives (no lambda records needed): res holds z_0..z_T
            # (length T+1), so z_{t+1}=res[1:], z_t=res[:-1], z_T=res[-1] -- align these T-step slices
            # with vfs/grads (length T). gui_vec = (vfs - gui_vfs)/s_t; unguided steps give gui_vfs==vfs.
            _res_prev = res_xr.isel(t=slice(0, -1)).assign_coords(t=grads_xr.t)   # z_0..z_{T-1}
            _z_next = res_xr.isel(t=slice(1, None)).assign_coords(t=grads_xr.t)   # z_1..z_T
            _z_T = res_xr.isel(t=-1)                                              # the endpoint z_T
            gui_vfs_xr = (_z_next - _res_prev) * _s_da / _h_da  # guided vf, stored x s_t convention
            gui_vec_xr = (vfs_xr - gui_vfs_xr) / _s_da          # applied guidance vector per t
            gui_res_xr = -(gui_vec_xr * _h_da).sum("t")         # guidance contribution to z_T
            # clean prediction (physical): gui_final + ((z_t + s_t*u_t) - z_T) * c
            _dev = (_res_prev + vfs_xr) - _z_T
            clean_preds_xr = xr.Dataset(
                {_v: guided_xr[_v] + _dev[_v] * res_scale_map[_v] for _v in _dev.data_vars}
            ).transpose("m", "n", "t", ...)  # broadcast puts t last; restore trace order
    return (
        clean_preds_xr,
        grads_xr,
        gui_ung_final_xr,
        gui_ung_xr,
        gui_vec_xr,
        gui_vfs_xr,
        res_scale_map,
        res_xr,
        vfs_xr,
    )


@app.cell
def _(
    get_guidance_schedule,
    guidance_mode_dropdown,
    m,
    n,
    notebook_mode,
    np,
    rollout_id,
    sweep_params,
):
    # applied guidance weight schedules for the selected (sweep, m, n), recorded per
    # method into guidance_schedule.json. Convention: lambda_hat = w*a_t*c_t multiplies
    # the UNIT gradient (= the kick norm); the raw-gradient multiplier used by every
    # reconstruction is lambda_t = lambda_hat / g_norm. Legacy records (no c_t/g_norm_t)
    # default both to 1, under which w*a IS the raw multiplier as before.
    if notebook_mode == "analyze_rollout":
        # records store the sweep in coord-label form == the notebook sweep_params dict
        _sel = dict(sweep_params)
        _recs = get_guidance_schedule(rollout_id, _sel, m=m, n=n)
        guidance_schedules = {}
        for _r in _recs:
            _w = np.asarray(_r["w_t"], dtype=float)
            _a = np.asarray(_r["a_t"], dtype=float)
            _c = np.asarray(_r.get("c_t", np.ones_like(_w)), dtype=float)
            _gn = np.asarray(_r.get("g_norm_t", np.ones_like(_w)), dtype=float)
            _hat = _w * _a * _c
            guidance_schedules[_r["method"]] = {
                "w_t": _w, "a_t": _a, "c_t": _c, "g_norm_t": _gn,
                "a_t_mode": (_r.get("sweep") or {}).get("a_t_mode"),  # profile shape of this record
                "lambda_hat": _hat,                                # kick norm
                "lambda_t": _hat / np.where(_gn != 0, _gn, 1.0),   # raw-gradient multiplier
            }
        lambda_t_by_method = {_k: _v["lambda_t"] for _k, _v in guidance_schedules.items()}
        if guidance_mode_dropdown.value in guidance_schedules:
            _mine = guidance_schedules[guidance_mode_dropdown.value]
            w_t_schedule, a_t_schedule, lambda_t = _mine["w_t"], _mine["a_t"], _mine["lambda_t"]
            c_t_schedule = _mine["c_t"]
        else:
            w_t_schedule = a_t_schedule = lambda_t = c_t_schedule = None
    else:
        guidance_schedules = {}
        lambda_t_by_method = {}
        w_t_schedule = a_t_schedule = lambda_t = c_t_schedule = None
    return guidance_schedules, lambda_t_by_method


@app.cell
def _(
    clean_preds_xr,
    delta_trajectory,
    flow_section_checkbox,
    get_rollout,
    get_slices,
    grads_xr,
    gt_curr,
    gui_ung_curr,
    gui_ung_xr,
    gui_vec_xr,
    gui_vfs_xr,
    guided_xr,
    level,
    m,
    mask,
    n,
    notebook_mode,
    np,
    partition,
    res_scale_map,
    res_xr,
    rollout_id,
    sweep_params,
    t,
    ung_curr,
    var,
    vfs_xr,
):
    if flow_section_checkbox.value and notebook_mode not in ("unguided_rollout", "guided_rollout"):
        # changes over t
        clean_preds_slices = get_slices(clean_preds_xr, partition, var, level)
        grads_slices = get_slices(grads_xr, partition, var, level)
        vfs_slices = get_slices(vfs_xr, partition, var, level)
        guided_vfs_slices = get_slices(gui_vfs_xr, partition, var, level)

        # slices of interest
        # 1
        diff_gt_gui_ung_slice =  gui_ung_curr - gt_curr
        diff_gt_clean_pred_slice = clean_preds_slices[m][n][t] - gt_curr
        # 2
        # compare against the unguided clean prediction AT THE SAME flow step t (the
        # gui_ung trace carries the full t axis; older stores without it fall back to final)
        _gui_ung_t_slices = get_slices(gui_ung_xr, partition, var, level)
        _gui_ung_t = _gui_ung_t_slices[m][n][t] if _gui_ung_t_slices.ndim == 5 else _gui_ung_t_slices[m][n]
        gui_ung_clean_diff_slice = clean_preds_slices[m][n][t] - _gui_ung_t
        clean_preds_slice_prev = clean_preds_slices[m][n][t-1] if t>0 else clean_preds_slices[m][n][t]
        clean_preds_diff_slice = clean_preds_slices[m][n][t] - clean_preds_slice_prev
        # 3
        grads_slice = grads_slices[m][n][t]
        grads_slice_prev_slice = grads_slices[m][n][t-1] if t>0 else grads_slices[m][n][t]
        diff_grads_slice = grads_slice- grads_slice_prev_slice
        # noisy state z_t + applied guidance vector gui_vec = lambda_t * grad
        # (new-format rollouts only; zeros on legacy stores)
        if res_xr is not None:
            _res_mn = get_slices(res_xr, partition, var, level)[m][n]
            res_slice = _res_mn[t]
        else:
            res_slice = np.zeros_like(grads_slices[m][n][t])
        if gui_vec_xr is not None:
            gui_vec_slice = get_slices(gui_vec_xr, partition, var, level)[m][n][t]
        else:
            gui_vec_slice = np.zeros_like(grads_slices[m][n][t])

        # 4
        guided_vfs_slice = guided_vfs_slices[m][n][t]
        vfs_slice = vfs_slices[m][n][t]
        # residual integrand of the masked loss: mask * (x_hat - (1+delta_n)*x_ref) with
        # the ONLINE delta: (1+delta_n)*S(x_ref) == A = (1+p_n) * baseline masked mean,
        # mirroring rollout.py (GT reference: baseline == x_ref -> scale = 1+p_n exactly)
        _x_ref_slice = gt_curr if sweep_params.get("GUI_REF") == "GT" else gui_ung_curr
        _base_slice = gt_curr if sweep_params.get("GUI_REF") == "GT" else ung_curr
        _A = (1.0 + float(np.asarray(delta_trajectory, dtype=float)[n])) * float(
            (np.asarray(_base_slice, dtype=float) * np.asarray(mask)).sum())
        _scale = _A / float((np.asarray(_x_ref_slice, dtype=float) * np.asarray(mask)).sum())
        masked_residual_slice = (clean_preds_slices[m][n][t] - _scale * _x_ref_slice) * np.asarray(mask)
        # Model reaction to the guidance applied at t, in DISPLACEMENT units: each
        # vf scaled by its own step size h (stored vf = s*u -> h*u = vf*h/s), so the
        # map is the change in per-step motion h_{t+1} u_{t+1} - h_t u^gui_t -- did
        # the model continue the guided move or revert it. Last step: no t+1 -> zero.
        _s_sched = np.linspace(1000, 1, len(vfs_slices[m][n])) / 1000
        _h_sched = np.empty_like(_s_sched)
        _h_sched[:-1] = _s_sched[:-1] - _s_sched[1:]
        _h_sched[-1] = _s_sched[-1]
        vf_gui_next_diff_slice = (
            vfs_slices[m][n][t+1] * (_h_sched[t+1] / _s_sched[t+1])
            - guided_vfs_slice * (_h_sched[t] / _s_sched[t])
            if t + 1 < len(vfs_slices[m][n])
            else np.zeros_like(guided_vfs_slice)
        )
        # naive next-step disagreement in stored-vf units: vf_{t+1} - vf^gui_t
        # (last step has no t+1 -> all-NaN -> flat zero map downstream)
        vf_next_gui_diff_slice = (
            vfs_slices[m][n][t+1] - guided_vfs_slice
            if t + 1 < len(vfs_slices[m][n])
            else np.full_like(np.asarray(guided_vfs_slice, dtype=float), np.nan)
        )

        # ---- physical (denormalized) views: sigma_r = res_scale_map[var] ----
        _c_sel = res_scale_map[var]
        c_phys = float(_c_sel.sel(level=level)) if partition == "level" else float(_c_sel)
        # stored vf traces are s_t * u_t -> h_t * u_t = vfs * h_t/s_t: the maps show
        # the PHYSICAL PER-STEP DISPLACEMENT sigma_r * h_t * u_t (not the raw velocity)
        vfs_phys_slice = vfs_slice * (_h_sched[t] / _s_sched[t]) * c_phys
        guided_vfs_phys_slice = guided_vfs_slice * (_h_sched[t] / _s_sched[t]) * c_phys
        vf_gui_next_diff_slice = vf_gui_next_diff_slice * c_phys
        vf_next_gui_diff_slice = vf_next_gui_diff_slice * c_phys
        # r-hat_t = sigma_r * (z^t + s_t u_t): the physical clean-residual estimate
        clean_res_slice = (res_slice + vfs_slice) * c_phys
        # applied kick in physical units: sigma_r * lambda_t * h_t * grad (last h = s)
        gui_vec_phys_slice = gui_vec_slice * _h_sched[t] * c_phys
        # next-step unguided displacement sigma_r * h_{t+1} * u_{t+1};
        # last step has no t+1 -> all-NaN flat map
        u_next_slice = (
            vfs_slices[m][n][t+1] * (_h_sched[t+1] / _s_sched[t+1]) * c_phys
            if t + 1 < len(vfs_slices[m][n])
            else np.full_like(np.asarray(vfs_slice, dtype=float), np.nan)
        )
        # guided step increment: sigma_r * h_t * u_t^gui (stored gui_vf = s_t * u_t^gui,
        # so multiply by h_t/s_t); running sum over t builds the residual r-hat
        step_inc_slice = guided_vfs_slice * (_h_sched[t] / _s_sched[t]) * c_phys
        # masked unguided velocity: the integrand of the mask-weighted mean of u-hat_t
        masked_vf_slice = vfs_phys_slice * np.asarray(mask)
        # ---- x_t objects (Euler LANDING, only *h_t): x_t = x_det + sigma_r*(z_t +
        # h_t u_t) -- the partial step actually taken, vs x_hat_t = ... + s_t u_t
        # (complete denoise) used above. det field from the gui_det store; fallback
        # reconstructs it from the final guided state.
        try:
            _det2d = get_slices(get_rollout("gui_det", rollout_id).sel(sweep_params).compute(),
                                partition, var, level)[m][n]
        except (FileNotFoundError, KeyError):
            _gui2d = get_slices(guided_xr, partition, var, level)[m][n]
            _zT2d = _res_mn[-1]                             # res holds z_0..z_T; the endpoint is res[-1]
            _det2d = _gui2d - c_phys * _zT2d
        x_land_ung_slice = _det2d + c_phys * (np.asarray(res_slice) + (_h_sched[t] / _s_sched[t]) * np.asarray(vfs_slice))
        x_land_gui_slice = _det2d + c_phys * (np.asarray(res_slice) + (_h_sched[t] / _s_sched[t]) * np.asarray(guided_vfs_slice))
        # guidance effect: guided vs unguided-twin STATE at the SAME flow step, both res-based
        # (x_t = x_det + sigma_r*z_t, NO velocity term). At the noise step z_t is the shared init
        # noise of both passes (same seed) -> exactly 0 everywhere; the divergence grows with t.
        landing_diff_slice = (_det2d + c_phys * np.asarray(res_slice)) - np.asarray(_gui_ung_t)
        masked_residual_land_slice = (x_land_ung_slice - _scale * np.asarray(_x_ref_slice)) * np.asarray(mask)
        # row-1 left: realized guided landing vs the FINAL unguided state (the
        # reference the guidance target is built from)
        gui_land_vs_ung_final_slice = x_land_gui_slice - np.asarray(gui_ung_curr)
        # row-2 left: the GUIDED partial-step residual (== x_t^gui - x_det)
        r_land_gui_slice = (np.asarray(res_slice) + (_h_sched[t] / _s_sched[t]) * np.asarray(guided_vfs_slice)) * c_phys
        # r_t with h_t instead of s_t: the PARTIAL-step residual sigma_r*(z_t + h_t u_t)
        r_land_slice = (np.asarray(res_slice) + (_h_sched[t] / _s_sched[t]) * np.asarray(vfs_slice)) * c_phys
    return (
        clean_preds_diff_slice,
        clean_res_slice,
        diff_grads_slice,
        diff_gt_clean_pred_slice,
        diff_gt_gui_ung_slice,
        grads_slice,
        gui_land_vs_ung_final_slice,
        gui_ung_clean_diff_slice,
        gui_vec_phys_slice,
        guided_vfs_phys_slice,
        landing_diff_slice,
        masked_residual_land_slice,
        masked_residual_slice,
        masked_vf_slice,
        r_land_gui_slice,
        r_land_slice,
        step_inc_slice,
        u_next_slice,
        vf_gui_next_diff_slice,
        vfs_phys_slice,
    )


@app.cell
def _(t_slider):
    # slider is 1-based (1..T); t is the 0-based index
    t=t_slider.value-1
    return (t,)


@app.cell
def _(
    add_map_stats,
    clean_preds_diff_slice,
    clean_res_slice,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    diff_grads_slice,
    diff_gt_clean_pred_slice,
    diff_gt_gui_ung_slice,
    dpi_slider,
    flow_row_select,
    flow_section_checkbox,
    grads_slice,
    gui_land_vs_ung_final_slice,
    gui_ung_clean_diff_slice,
    gui_vec_phys_slice,
    guided_vfs_phys_slice,
    landing_diff_slice,
    mask,
    masked_residual_land_slice,
    masked_residual_slice,
    masked_vf_slice,
    notebook_mode,
    np,
    r_land_gui_slice,
    r_land_slice,
    show_mask_switch,
    step_inc_slice,
    u_next_slice,
    vf_gui_next_diff_slice,
    vfs_phys_slice,
    visualize_map,
    white_zero_cmap,
    zoom_centers,
    zoom_slider,
):
    if flow_section_checkbox.value and notebook_mode not in ("unguided_rollout", "guided_rollout"):
        diff_vfs_slice = guided_vfs_phys_slice - vfs_phys_slice

        # mask-weighted average (mask sums to 1) shown in the vf map titles
        def _mavg(_a):
            return float(np.nansum(np.asarray(_a) * np.asarray(mask)))

        map_specs = [
            ("diff_gt_gui_ung_map", diff_gt_gui_ung_slice, r"$x_{n}^{\text{ung|gui}} - x_{n}^{\text{gt}}$", -1, 1),
            ("diff_gt_clean_pred_map", diff_gt_clean_pred_slice, r"$\hat{x}_t^{\text{gui}} - x_{n}^{\text{gt}}$", -1, 1),
            ("gui_ung_clean_diff_map", gui_ung_clean_diff_slice, r"$\hat{x}_t^{\text{gui}} - \hat{x}_t^{\text{ung|gui}}$", -1, 1),
            ("clean_preds_diff_map", clean_preds_diff_slice, r"$\hat{x}_t^{\text{gui}} - \hat{x}_{t-1}^{\text{gui}}$", -1, 1),
            ("grads_map", grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t$", -1, 1),
            ("vfs_map", vfs_phys_slice, rf"$\sigma_r\, h_t u_t$ (mask avg {_mavg(vfs_phys_slice):+.3g})", -0.001, 0.001),
            ("masked_residual_map", masked_residual_slice, r"$(\hat{x}^{\text{gui}}_t - (1+\phi_n)\,x^{\text{ref}}) \cdot \text{mask}$", -1, 1),
            ("guided_vfs_map", guided_vfs_phys_slice, rf"$\sigma_r\, h_t u^{{\text{{gui}}}}_t$ (mask avg {_mavg(guided_vfs_phys_slice):+.3g})", -0.001, 0.001),
            ("diff_vfs_map", diff_vfs_slice, r"$\sigma_r h_t (u^{\text{gui}}_t - u_t)$", -0.001, 0.001),
            ("vf_gui_next_diff_map", vf_gui_next_diff_slice, r"$\sigma_r\,(h_{t+1} u_{t+1} - h_t u^{\text{gui}}_t)$", -0.001, 0.001),
            ("masked_vf_map", masked_vf_slice, r"$\sigma_r h_t u_t \cdot \text{mask}$", -0.001, 0.001),
            ("u_next_map", u_next_slice, rf"$\sigma_r\, h_{{t+1}} u_{{t+1}}$ (mask avg {_mavg(u_next_slice):+.3g})", -0.001, 0.001),
            ("step_inc_map", step_inc_slice, r"$\sigma_r\, h_t u^{\text{gui}}_t$", -0.001, 0.001),
            ("clean_res_map", clean_res_slice, r"$\hat{r}_t = \sigma_r\,(z^t + s_t u_t)$", -1, 1),
            ("gui_vec_map", gui_vec_phys_slice, r"$\sigma_r\,\lambda_t h_t\,\nabla_{z_t}\mathcal{L}_t$", -1, 1),
            ("landing_diff_map", landing_diff_slice, r"guidance effect  $x_t^{\text{gui}} - x_t^{\text{ung|gui}}$  ($x_t=\hat{x}^{\text{gui\_det}}+\sigma_r z_t$; $=0$ at the noise step)", -1, 1),
            ("masked_residual_land_map", masked_residual_land_slice, r"$(x_t - (1+\phi_n)\,x^{\text{ref}}) \cdot \text{mask}$", -1, 1),
            ("gui_land_vs_ung_final_map", gui_land_vs_ung_final_slice, r"$x_t^{\text{gui}} - x_T^{\text{ung|gui}}$  (landing vs final unguided state)", -1, 1),
            ("r_land_gui_map", r_land_gui_slice, r"$r_t^{\text{gui}} = \sigma_r\,(z_t + h_t u^{\text{gui}}_t)$", -1, 1),
            ("r_land_map", r_land_slice, r"$r_t = \sigma_r\,(z_t + h_t u_t)$", -1, 1),
            ("diff_grads_map", diff_grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t - \\nabla_{z_{t-1}} \\mathcal{L}_{t-1}$", -1, 1),
        ]

        # Only maps a flow row actually shows are worth rendering; the other
        # map_specs entries are never displayed (dead) or belong to an unselected
        # row. Gate the expensive visualize_map calls by the flow-row selection.
        _FLOW_MAP_ROW = {
            "landing_diff_map": "gui_t diffs", "masked_residual_land_map": "gui_t diffs",
            "r_land_gui_map": "x_t diffs", "r_land_map": "x_t diffs",
            "grads_map": "grads", "gui_vec_map": "grads",
            "vfs_map": "vfs", "guided_vfs_map": "vfs",
            "u_next_map": "u_next",
        }
        _flow_sel = set(flow_row_select.value)

        maps = {}

        for name, data, title, fallback_vmin, fallback_vmax in map_specs:
            if _FLOW_MAP_ROW.get(name) not in _flow_sel:
                maps[name] = None  # not shown -> skip the render
                continue
            data_min = np.nanmin(data) if np.isfinite(data).any() else np.nan
            data_max = np.nanmax(data) if np.isfinite(data).any() else np.nan
            data_mean = np.nanmean(data) if np.isfinite(data).any() else np.nan
            if not (np.isfinite(data_min) and np.isfinite(data_max)):
                # all-NaN slice (e.g. reconstruction without lambda records on mixed-run
                # dirs) -> render a flat zero map instead of crashing the colormap
                data = np.zeros_like(np.asarray(data))
                data_min = data_max = data_mean = 0.0
            print(name, data_min, data_mean, data_max)

            if data_min != data_max:
                data = np.where(np.abs(data) < 0.0 / 100.0 * float(np.nanmax(np.abs(data))), np.nan, data)
            maps[name] = visualize_map(
                data,
                cmap=white_zero_cmap,
                mask_2d=mask,
                show_mask=show_mask_switch.value,
                title=title,
                interactive=False,
                vmin=data_min if data_min != 0 else -1,
                vmax=data_max if data_max != 0 else 1,
                center=data_mean if data_mean != 0 else 0,
                figsize=(14, 8),
                dpi=dpi_slider.value,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
                contour_2d=data if contour_checkbox.value and data_min != data_max else None,
                contour_levels=contour_levels_slider.value,
                contour_color=contour_color_dropdown.value,
                contour_linewidth=0.4,
            )
            maps[name] = add_map_stats(maps[name], data)

        diff_gt_gui_ung_map = maps["diff_gt_gui_ung_map"]
        diff_gt_clean_pred_map = maps["diff_gt_clean_pred_map"]
        gui_ung_clean_diff_map = maps["gui_ung_clean_diff_map"]
        clean_preds_diff_map = maps["clean_preds_diff_map"]
        grads_map = maps["grads_map"]
        vfs_map = maps["vfs_map"]
        masked_residual_map = maps["masked_residual_map"]
        guided_vfs_map = maps["guided_vfs_map"]
        diff_vfs_map = maps["diff_vfs_map"]
        vf_gui_next_diff_map = maps["vf_gui_next_diff_map"]
        masked_vf_map = maps["masked_vf_map"]
        u_next_map = maps["u_next_map"]
        step_inc_map = maps["step_inc_map"]
        clean_res_map = maps["clean_res_map"]
        gui_vec_map = maps["gui_vec_map"]
        gui_land_vs_ung_final_map = maps["gui_land_vs_ung_final_map"]
        r_land_gui_map = maps["r_land_gui_map"]
        r_land_map = maps["r_land_map"]
        landing_diff_map = maps["landing_diff_map"]
        masked_residual_land_map = maps["masked_residual_land_map"]
        diff_grads_map = maps["diff_grads_map"]
    return (
        grads_map,
        gui_vec_map,
        guided_vfs_map,
        landing_diff_map,
        masked_residual_land_map,
        r_land_gui_map,
        r_land_map,
        u_next_map,
        vfs_map,
    )


@app.cell
def _(mo):
    _flow_rows = ["gui_t diffs", "x_t diffs", "grads", "vfs", "u_next"]
    flow_row_select = mo.ui.multiselect(_flow_rows, value=_flow_rows, label="charts: ")
    return (flow_row_select,)


@app.cell
def _(mo):
    # row selectors for the Inspect-states maps (analyze mode). Split by analysis
    # type so narrowing the rows in one mode never blanks the other (a single shared
    # selector could end up holding only the other mode's keys -> empty section).
    _ABS_ROWS = ["curr / prev", "gui / ung", "dets"]
    _DIFF_ROWS = ["gui_ung-gt / gui-gt", "gui_det-ung", "gui_ung-ung / gui-ung", "r_n^gui_ung / r_n^gui",
                  "x_det / gui_gui_ung", "x_gui_ung / x_gui", "x_ung / x_gt"]
    inspect_abs_rows = mo.ui.multiselect(_ABS_ROWS, value=_ABS_ROWS, label="charts: ")
    inspect_diff_rows = mo.ui.multiselect(_DIFF_ROWS, value=_DIFF_ROWS, label="charts: ")
    return inspect_abs_rows, inspect_diff_rows


@app.cell
def _(
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    dpi_slider,
    flow_row_select,
    flow_section_checkbox,
    grads_map,
    gui_vec_map,
    guided_vfs_map,
    landing_diff_map,
    level_slider,
    m_slider,
    masked_residual_land_map,
    mo,
    n_slider,
    notebook_mode,
    r_land_gui_map,
    r_land_map,
    show_mask_switch,
    stats_scope_dropdown,
    sweep_params_widget,
    t_slider,
    u_next_map,
    var_dropdown,
    vfs_map,
    zoom_slider,
):
    if flow_section_checkbox.value and notebook_mode not in ("unguided_rollout", "guided_rollout"):
        var_controls = mo.vstack(
            [
                mo.hstack(
                    [level_slider],
                    justify="start",
                    align="start",
                ),
                var_dropdown,
            ],
            align="start",
        )
        flow_controls = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                mo.hstack([t_slider, dpi_slider], justify="start", align="start"),
                                mo.hstack([m_slider, n_slider], justify="start", align="start"),
                            ],
                            align="start",
                        ),
                        var_controls,
                    ],
                    justify="start", align="start",
                ),
                mo.hstack([show_mask_switch, zoom_slider, stats_scope_dropdown], justify="start", align="start"),
                mo.hstack(
                    [contour_checkbox, contour_levels_slider, contour_color_dropdown],
                    justify="start", align="start",
                ),
            ],
            align="start",
        )

        map_rows = [
            ("gui_t diffs", [landing_diff_map, masked_residual_land_map]),
            ("x_t diffs", [r_land_gui_map, r_land_map]),
            ("grads", [grads_map, gui_vec_map]),
            ("vfs", [vfs_map, guided_vfs_map]),
            ("u_next", [u_next_map]),
        ]

        flow_widget_make = mo.vstack(
            [
                *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
                flow_controls,
                flow_row_select,
                mo.vstack(
                    [mo.hstack(_maps, justify="start", align="start") for _k, _maps in map_rows if _k in flow_row_select.value],
                    justify="start", align="start",
                ),
            ],
            justify="start", align="start",
        )
    return (flow_widget_make,)


@app.cell
def _(flow_section_checkbox, flow_widget_make, notebook_mode):
    if flow_section_checkbox.value and notebook_mode not in ("unguided_rollout", "guided_rollout"):
        flow_widget=flow_widget_make
    else:
        flow_widget=None
    flow_widget
    return


@app.cell(hide_code=True)
def _(cross_section_checkbox, mo):
    mo.hstack([cross_section_checkbox, mo.md("## Cross variable checks")], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    # per-subsection activation for the cross-variable charts: compute AND
    # display only when checked
    cross_masked_checkbox = mo.ui.checkbox(value=False)
    cross_norms_checkbox = mo.ui.checkbox(value=False)
    cross_angles_checkbox = mo.ui.checkbox(value=False)
    # units for the masked-averages charts: normalized (ranking space) or data (physical)
    masked_units_dropdown = mo.ui.dropdown(["normalized", "data"], value="normalized", label="units: ")
    return (
        cross_angles_checkbox,
        cross_masked_checkbox,
        cross_norms_checkbox,
        masked_units_dropdown,
    )


@app.cell(hide_code=True)
def _(mo):
    # per-section activation: compute AND display a section only when checked
    mask_section_checkbox = mo.ui.checkbox(value=True)
    inspect_section_checkbox = mo.ui.checkbox(value=True)
    flow_section_checkbox = mo.ui.checkbox(value=True)
    cross_section_checkbox = mo.ui.checkbox(value=True)
    trajectories_section_checkbox = mo.ui.checkbox(value=True)
    return (
        cross_section_checkbox,
        flow_section_checkbox,
        inspect_section_checkbox,
        mask_section_checkbox,
        trajectories_section_checkbox,
    )


@app.cell
def _(mask, mcolors, mo, np, plt, xr):

    LEVEL_VARS = ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "vertical_velocity"]
    SURFACE_PAIR = {"temperature": "2m_temperature", "u_component_of_wind": "10m_u_component_of_wind", "v_component_of_wind": "10m_v_component_of_wind"}
    _ALL_VARS = LEVEL_VARS + ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
    _SURF_TO_LVL = {s: l for l, s in SURFACE_PAIR.items()}
    _PALETTE = plt.get_cmap("tab10").colors
    _N_LEVELS = 13

    level_var_dropdown = mo.ui.dropdown(LEVEL_VARS + ["mean_sea_level_pressure"], label="variable: ")
    aggregate_by_level_checkbox = mo.ui.checkbox(label="aggregate by level")
    aggregate_spatially_dropdown = mo.ui.dropdown(["mask", "!mask"], allow_select_none=True, label="aggregate spatially: ")
    dist_bands_checkbox = mo.ui.checkbox(label="dist bands")
    _cross_rows = ["masked avg", "gui vs ung levels", "guidance kick (latent)", "grad_norms", "gui_vf_norms", "vf_norms", "angular deflection", "angular deflection (gui_ung)", "angular deflection (ung)"]
    cross_row_select = mo.ui.multiselect(_cross_rows, value=_cross_rows, label="charts: ")
    differential_checkbox = mo.ui.checkbox(label=r"$\Delta$")
    abs_checkbox = mo.ui.checkbox(label=r"$|\cdot|$")

    def n_traces(v, agg):
        if v is None:
            return len(_ALL_VARS) if agg else len(LEVEL_VARS) * _N_LEVELS + (len(_ALL_VARS) - len(LEVEL_VARS))
        if agg:
            return 1
        return (_N_LEVELS + (1 if v in SURFACE_PAIR else 0)) if v in LEVEL_VARS else 1

    def grouped_vars(ds, v, agg):
        vars_ = list(ds.data_vars) if v is None else [v] + ([SURFACE_PAIR[v]] if v in SURFACE_PAIR else [])
        if v is not None and agg:
            return {v: [ds[x] for x in vars_]}
        out = {}
        for name in vars_:
            da = ds[name]
            if "level" in da.dims and not agg:
                for lv in da["level"].values:
                    out[f"{name} L{int(lv)}"] = [da.sel(level=lv)]
            else:
                out[name] = [da]
        return out

    def color_for(labels):
        out = {}
        for label in labels:
            if " L" in label:
                name, lv = label.rsplit(" L", 1); lv = int(lv)
            else:
                name, lv = label, None
            base = _SURF_TO_LVL.get(name, name)
            hue = mcolors.to_rgb(_PALETTE[_ALL_VARS.index(base) % len(_PALETTE)])
            intensity = 1.0 if lv is None else 0.3 + 0.7 * lv / 1000
            out[label] = tuple(c + (1 - c) * (1 - intensity) for c in hue)
        return out

    def maybe_mask(ds, mode):
        if mode not in ("mask", "!mask"):
            return ds
        # boolean region at half-maximum: BBOX (0/1) is unchanged; ELLIPTICAL is
        # nonzero everywhere (astype(bool) would be all-True -> "!mask" all-NaN),
        # and its peak is ~1/sum, so the threshold must be relative to max.
        _m = xr.DataArray(
            mask >= 0.5 * mask.max(),
            dims=("latitude", "longitude"),
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
        )
        return ds.where(_m if mode == "mask" else ~_m)

    def cross_traces(red_ds, axis, agg, m_idx, *, k, var, by_level, diff, absv, bands, rank_by="agg change"):
        """Member trace + optional min/max band from spatially pre-reduced cubes (see `red`).
        Pure: all control values come in as args so callers carry the UI dependencies."""
        _g = grouped_vars(red_ds, var, by_level)

        def _reduce(das):
            _other = lambda da: [d for d in da.dims if d not in (axis, "m")]
            if agg == "l2":
                out = np.sqrt(sum(da.sum(dim=_other(da)) for da in das))
            else:
                out = sum(da.mean(dim=_other(da)) for da in das) / len(das)
            if "m" in out.dims:
                out = out.transpose("m", axis)
            arr = np.atleast_2d(out.values)  # (members, axis)
            if diff:
                arr = np.concatenate([np.zeros((arr.shape[0], 1)), np.diff(arr, axis=1)], axis=1)
            if absv:
                arr = np.abs(arr)
            return arr

        _full = {k_: _reduce(das) for k_, das in _g.items()}
        traces = top_k({k_: v[m_idx if v.shape[0] > 1 else 0] for k_, v in _full.items()}, k, rank_by)
        bands_out = (
            {k_: (_full[k_].min(axis=0), _full[k_].max(axis=0)) for k_ in traces if _full[k_].shape[0] > 1}
            if bands
            else None
        )
        return traces, bands_out

    def rank_score(v, rank_by):
        # nan-aware |trace| score per rank mode; all-NaN traces rank last (+inf key).
        _v = np.abs(np.asarray(v, dtype=float))
        if not np.isfinite(_v).any():
            return np.inf
        _fin = _v[np.isfinite(_v)]
        if rank_by == "final value":
            return -float(_fin[-1])
        if rank_by == "initial value":
            return -float(_fin[0])
        return -float(np.nanmean(_v))  # "agg change": aggregated influence over all steps

    def top_k(traces, k, rank_by="agg change"):
        return dict(sorted(traces.items(), key=lambda kv: rank_score(kv[1], rank_by))[:k])

    return (
        SURFACE_PAIR,
        abs_checkbox,
        aggregate_by_level_checkbox,
        aggregate_spatially_dropdown,
        color_for,
        cross_row_select,
        cross_traces,
        differential_checkbox,
        dist_bands_checkbox,
        level_var_dropdown,
        n_traces,
        rank_score,
    )


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    differential_checkbox,
    dist_bands_checkbox,
    level_var_dropdown,
    rank_by_dropdown,
    top_k_slider,
):
    cross_ctl = dict(
        k=top_k_slider.value,
        var=level_var_dropdown.value,
        by_level=aggregate_by_level_checkbox.value,
        diff=differential_checkbox.value,
        absv=abs_checkbox.value,
        bands=dist_bands_checkbox.value,
        rank_by=rank_by_dropdown.value,
    )
    return (cross_ctl,)


@app.cell
def _(aggregate_by_level_checkbox, level_var_dropdown, mo, n_traces):
    _kmax = n_traces(level_var_dropdown.value, aggregate_by_level_checkbox.value)
    _k_steps = sorted({1, *range(5, _kmax + 1, 5), _kmax})  # 1, 5, 10, ..., max
    top_k_slider = mo.ui.slider(steps=_k_steps, value=5 if 5 in _k_steps else _k_steps[0], label=get_label("top K", _kmax), show_value=True, debounce=True)
    rank_by_dropdown = mo.ui.dropdown(["final value", "initial value", "agg change"], value="final value", label="rank by: ")
    return rank_by_dropdown, top_k_slider


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    aggregate_spatially_dropdown,
    differential_checkbox,
    dpi_slider,
    level_var_dropdown,
    mo,
    rank_by_dropdown,
    top_k_slider,
):
    cross_check_controls = mo.vstack(
        [
            mo.hstack([differential_checkbox, abs_checkbox, top_k_slider, rank_by_dropdown, dpi_slider], justify="start", align="start"),
            mo.hstack([level_var_dropdown, aggregate_by_level_checkbox, aggregate_spatially_dropdown], justify="start", align="start"),
        ],
        align="start",
    )
    return (cross_check_controls,)


@app.cell
def _(XarrayNormalizer, gt_rollout, guided_xr, notebook_mode):
    if notebook_mode == "analyze_rollout":
        xnorm = XarrayNormalizer()
        # raw (un-normalized) gt aligned to rollout coords. Normalization is deferred
        # until AFTER the spatial reduction in the `red` cell: the affine, per-(var,level)
        # normalization commutes with the spatial mean, so we never build a full-cube copy.
        gt_n_xr = (
            gt_rollout.isel(time=slice(1, None))
            .rename({"time": "n"})
            # gt coords differ from rollout coords in float precision; assign exact
            # copies so xarray alignment keeps the full grid
            .assign_coords(
                n=guided_xr.n,
                latitude=guided_xr.latitude,
                longitude=guided_xr.longitude,
                level=guided_xr.level,
            )
        )
    return (xnorm,)


@app.cell(hide_code=True)
def _(gt_rollout, guided_xr, mask, notebook_mode, np, xnorm, xr):
    # ground-truth initial condition (n=0) masked mean, normalized per variable --
    # n=0 anchor for the guided-levels over-n chart (masked-averages row 2, left)
    if notebook_mode == "analyze_rollout":
        _mw_ic = xr.DataArray(np.asarray(mask, dtype=float), dims=("latitude", "longitude"),
                              coords={"latitude": guided_xr.latitude, "longitude": guided_xr.longitude})
        _ic = gt_rollout.isel(time=0).assign_coords(
            latitude=guided_xr.latitude, longitude=guided_xr.longitude, level=guided_xr.level)
        gt_ic_wnorm = xnorm.normalize((_ic * _mw_ic).sum(("latitude", "longitude")) / float(_mw_ic.sum()))
    else:
        gt_ic_wnorm = None
    return (gt_ic_wnorm,)


@app.cell
def _(
    aggregate_spatially_dropdown,
    compute_reductions_for_sweep,
    load_reductions,
    notebook_mode,
    red_cache,
    reductions_ready,
    rollout_id,
    sweep_params,
):
    # reduced cubes `red` for the analysis charts. Prefer the persisted reductions.zarr
    # (built once for the whole sweep grid -> instant slides); fall back to a live
    # per-point compute from the SHARED src.reductions module (so stored and live values
    # can't drift) for the mask/!mask spatial toggle or rollouts without a store. Both
    # paths are memoized per (rollout, sweep, spatial-agg) in red_cache.
    if notebook_mode == "analyze_rollout":
        _agg = aggregate_spatially_dropdown.value
        _red_key = (rollout_id, tuple(sorted(sweep_params.items())), _agg)
        if _red_key in red_cache:
            red = red_cache[_red_key]
        else:
            if _agg is None and reductions_ready:
                red = load_reductions(rollout_id, dict(sweep_params))
            else:
                red = compute_reductions_for_sweep(rollout_id, dict(sweep_params),
                                                   spatial=("full" if _agg is None else _agg))
            red_cache[_red_key] = red
    else:
        red = None
    return (red,)


@app.cell
def _():
    # final widget
    return


@app.cell
def _(
    cross_check_controls,
    cross_row_select,
    cross_section_checkbox,
    dist_bands_checkbox,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    sweep_params_widget,
    t_slider,
):
    if cross_section_checkbox.value and notebook_mode =="analyze_rollout":
        cross_checks_widget = mo.vstack([
            *([mo.md("sweep params: "), sweep_params_widget] if sweep_params_widget is not None else []),
            dist_bands_checkbox,
            cross_check_controls,
            mo.hstack([m_slider, n_slider, t_slider], justify="start"),
            cross_row_select,
        ], align="start")
    else:
        cross_checks_widget = None
    cross_checks_widget
    return


@app.cell
def _(
    cross_masked_checkbox,
    cross_section_checkbox,
    masked_units_dropdown,
    mo,
):
    mo.hstack([cross_masked_checkbox, mo.md("### Masked averages"), masked_units_dropdown], justify="start", align="center") if cross_section_checkbox.value else None
    return


@app.cell(hide_code=True)
def _(
    cross_masked_checkbox,
    cross_row_select,
    cross_section_checkbox,
    gui_land_t_plot,
    gui_mean_n_plot,
    gui_mean_t_plot,
    gui_ung_lvl_n_plot,
    mo,
    notebook_mode,
):
    if cross_section_checkbox.value and notebook_mode == "analyze_rollout" and cross_masked_checkbox.value:
        _w = mo.vstack(
            [
                mo.hstack(_row, justify="start")
                for _key, _row in [
                    ("masked avg", [gui_mean_n_plot, gui_mean_t_plot]),
                    ("gui vs ung levels", [gui_ung_lvl_n_plot, gui_land_t_plot]),
                ]
                if _key in cross_row_select.value
            ],
            justify="start", align="start",
        )
    else:
        _w = None
    _w
    return


@app.cell
def _(cross_norms_checkbox, cross_section_checkbox, mo):
    mo.hstack([cross_norms_checkbox, mo.md("### Norms")], justify="start", align="center") if cross_section_checkbox.value else None
    return


@app.cell(hide_code=True)
def _(
    cross_norms_checkbox,
    cross_row_select,
    cross_section_checkbox,
    grad_norms_n_plot,
    grad_norms_plot,
    gui_vf_norms_n_plot,
    guided_vf_norms_plot,
    kick_lat_n_plot,
    kick_lat_t_plot,
    mo,
    notebook_mode,
    vf_norms_n_plot,
    vf_norms_plot,
):
    if cross_section_checkbox.value and notebook_mode == "analyze_rollout" and cross_norms_checkbox.value:
        _w = mo.vstack(
            [
                mo.hstack(_row, justify="start")
                for _key, _row in [
                    ("grad_norms", [grad_norms_n_plot, grad_norms_plot]),
                    *([("guidance kick (latent)", [kick_lat_n_plot, kick_lat_t_plot])] if kick_lat_n_plot is not None else []),
                    ("gui_vf_norms", [gui_vf_norms_n_plot, guided_vf_norms_plot]),
                    ("vf_norms", [vf_norms_n_plot, vf_norms_plot]),
                ]
                if _key in cross_row_select.value
            ],
            justify="start", align="start",
        )
    else:
        _w = None
    _w
    return


@app.cell
def _(cross_angles_checkbox, cross_section_checkbox, mo):
    mo.hstack([cross_angles_checkbox, mo.md("### Angular deflections")], justify="start", align="center") if cross_section_checkbox.value else None
    return


@app.cell(hide_code=True)
def _(
    angdef_gu_n_plot,
    angdef_gu_t_plot,
    angdef_uo_n_plot,
    angdef_uo_t_plot,
    cross_angles_checkbox,
    cross_row_select,
    cross_section_checkbox,
    mo,
    notebook_mode,
    vf_angdef_n_plot,
    vf_angdef_t_plot,
):
    if cross_section_checkbox.value and notebook_mode == "analyze_rollout" and cross_angles_checkbox.value:
        _w = mo.vstack(
            [
                mo.hstack(_row, justify="start")
                for _key, _row in [
                    ("angular deflection", [vf_angdef_n_plot, vf_angdef_t_plot]),
                    *([("angular deflection (gui_ung)", [angdef_gu_n_plot, angdef_gu_t_plot])] if angdef_gu_n_plot is not None else []),
                    *([("angular deflection (ung)", [angdef_uo_n_plot, angdef_uo_t_plot])] if angdef_uo_n_plot is not None else []),
                ]
                if _key in cross_row_select.value
            ],
            justify="start", align="start",
        )
    else:
        _w = None
    _w
    return


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_norms_checkbox,
    cross_section_checkbox,
    cross_traces,
    dpi_slider,
    m,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    red,
    row_keys,
):
    if cross_section_checkbox.value and notebook_mode == "analyze_rollout" and cross_norms_checkbox.value:
        _traces, _bands = cross_traces(red["grads_l2"].isel(n=n), "t", "l2", m, **{**cross_ctl, "k": 10**9})
        _traces = {_k: _traces[_k] for _k in row_keys["grads"] if _k in _traces}
        _bands = {_k: _bands[_k] for _k in _traces if _k in _bands} if _bands else None
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _traces.values()), default=1)))
        grad_norms_plot = plot_trajectory(_traces, title="Gradient norm",
            subtitle=r"$\|\nabla_{z_t}\mathcal{L}_t\|$", step=None, color_map=color_for(_traces), bands=_bands,
            figsize=(_w, 6), dpi=dpi_slider.value, prepend_zero=False, start_index=1, mirror_right_axis=True,
        )
        # grad_norms_plot
        _ax_fr = grad_norms_plot.axes[0] if hasattr(grad_norms_plot, "axes") and grad_norms_plot.axes else None
        if _ax_fr is not None:
            _n_fr = max((len(_v) for _v in _traces.values()), default=1)
            _ax_fr.set_xlim(-1.0, _n_fr + 0.4)
            _ax_fr.set_xticks(np.arange(0, _n_fr + 1))
    else:
        grad_norms_plot = None
    return (grad_norms_plot,)


@app.cell
def _(
    color_for,
    cross_angles_checkbox,
    cross_ctl,
    cross_masked_checkbox,
    cross_norms_checkbox,
    cross_section_checkbox,
    cross_traces,
    dpi_slider,
    gt_ic_wnorm,
    m,
    masked_units_dropdown,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    rank_score,
    red,
    xnorm,
    xr,
):
    if notebook_mode == "analyze_rollout":

        def _plot(title, subtitle, ds, axis, agg, twin_ds=None, rank_ds=None, keys=None, twin_label="unguided (gui_ung)", prepend_zero=False, n0=None):
            if keys is not None:
                # lock the displayed variables to a caller-provided set (e.g. the matching
                # over-n chart) so both charts of a row show the SAME traces
                _all, _bands_all = cross_traces(ds, axis, agg, m, **{**cross_ctl, "k": 10**9})
                _tr = {_k: _all[_k] for _k in keys if _k in _all}
                _bands = {_k: _bands_all[_k] for _k in _tr if _k in _bands_all} if _bands_all else None
            elif rank_ds is not None:
                # rank variables by the NORMALIZED gui - gui_ung difference in rank_ds
                # (fair across variables) instead of their own magnitude; the DISPLAY (ds)
                # stays denormalized. Pull every variable's rank + display trace (k huge),
                # sort by max|rank|, keep the top-k.
                _ctl_all = {**cross_ctl, "k": 10**9}
                _rank_all, _ = cross_traces(rank_ds, axis, agg, m, **_ctl_all)
                _gui_all, _bands_all = cross_traces(ds, axis, agg, m, **_ctl_all)
                def _rank_key(_k):  # same rank-by scoring as top_k; all-NaN traces sort last
                    return rank_score(_rank_all[_k], cross_ctl["rank_by"])
                _keys = [_k for _k in sorted(_rank_all, key=_rank_key) if _k in _gui_all][: cross_ctl["k"]]
                _tr = {_k: _gui_all[_k] for _k in _keys}
                _bands = {_k: _bands_all[_k] for _k in _keys if _k in _bands_all} if _bands_all else None
            else:
                _tr, _bands = cross_traces(ds, axis, agg, m, **cross_ctl)
            _colors = color_for(_tr)
            _styles = None
            if twin_ds is not None:
                # gui_ung twin overlay for the displayed variables: same colors, dotted.
                # "_"-prefixed keys keep the per-variable twins OUT of the legend; one
                # generic entry is appended after plotting (matches the landing chart)
                _twin_src, _ = cross_traces(twin_ds, axis, agg, m, **{**cross_ctl, "k": 10**9})
                _twin = {f"_{_k} (gui_ung)": _twin_src[_k] for _k in _tr if _k in _twin_src}
                _colors |= {_k: _colors[_k.removeprefix("_").removesuffix(" (gui_ung)")] for _k in _twin}
                _styles = {_k: ":" for _k in _twin}
                _tr = _tr | _twin
            # optional n=0 anchors: zeros (difference charts start closed) or a
            # per-variable value from `n0` (e.g. the gt initial condition)
            _start = 1
            if axis == "n" and (prepend_zero or n0 is not None):
                def _n0v(_k):
                    if n0 is None:
                        return 0.0
                    _b = _k.removeprefix("_").removesuffix(" (gui_ung)")
                    if " L" in _b:
                        _vv, _lv = _b.rsplit(" L", 1)
                        return float(n0[_vv].sel(level=int(_lv)))
                    return float(n0[_b])
                _tr = {_k: np.concatenate([[_n0v(_k)], np.asarray(_v, dtype=float)]) for _k, _v in _tr.items()}
                if _bands:
                    _bands = {_k: (np.concatenate([[_n0v(_k)], np.asarray(_lo, float)]),
                                   np.concatenate([[_n0v(_k)], np.asarray(_hi, float)]))
                              for _k, (_lo, _hi) in _bands.items()}
                _start = 0
            # width scales with the number of steps so the sparse "over n" plots
            # (N points) are not stretched across the same width as "over t" (T points)
            _nsteps = max((len(_v) for _v in _tr.values()), default=1)
            _w = min(22.0, max(8.0, 3.4 + 0.78 * _nsteps))
            _figp = plot_trajectory(
                _tr, title=title, subtitle=subtitle, xlabel=f"${axis}$",
                step=None, color_map=_colors, bands=_bands,
                figsize=(_w, 6), dpi=dpi_slider.value, prepend_zero=False, start_index=_start,
                mirror_right_axis=True, linestyle_map=_styles,
            )
            # keep 0 on the x-axis (no data point there: traces start at 1)
            if hasattr(_figp, "axes") and _figp.axes:
                _axp = _figp.axes[0]
                _axp.set_xlim((-1.0 if axis == "t" else -0.4), _start + _nsteps - 1 + 0.4)
                _axp.set_xticks(np.arange(0, _start + _nsteps))
            if twin_ds is not None and hasattr(_figp, "axes") and _figp.axes:
                # single generic legend entry for the dotted gui_ung twins
                _axp0 = _figp.axes[0]
                _axp0.plot([], [], linestyle=":", color="#888888", linewidth=1.2,
                           label=twin_label)
                _hh, _ll = _axp0.get_legend_handles_labels()
                for _axr in _figp.axes[1:]:
                    _h2, _l2 = _axr.get_legend_handles_labels()
                    _hh, _ll = _hh + _h2, _ll + _l2
                _uni = dict(zip(_ll, _hh))
                _axp0.legend(_uni.values(), _uni.keys(), loc="center left",
                             bbox_to_anchor=(1.05, 0.5), frameon=False,
                             handlelength=2.4, borderaxespad=0.0)
            return _figp

        # units for the masked-averages charts: "data" converts the affine
        # normalization back per variable/level -- absolute levels get the full
        # inverse (std*x + mean); differences only the std scaling (means cancel).
        # Ranking (row_keys) stays in normalized space so it is fair across vars.
        if masked_units_dropdown.value == "data":
            # absolute levels: full inverse, with temperature shown in Celsius
            # (differences keep the std-only scaling; delta-K == delta-C anyway)
            def _u_abs(_ds):
                _d = xnorm.denormalize(_ds)
                return _d.map(lambda _da: _da - 273.15 if _da.name in ("temperature", "2m_temperature") else _da)
            _u_diff = lambda _ds: xnorm.denormalize(_ds) - xnorm.denormalize(xr.zeros_like(_ds))
            _n0_disp = _u_abs(gt_ic_wnorm.map(lambda _da: _da)) if gt_ic_wnorm is not None else None
        else:
            _u_abs = _u_diff = lambda _ds: _ds
            _n0_disp = gt_ic_wnorm
        grad_norms_n_plot = _plot(r"Gradient norm", r"$\|\nabla_{z_t}\mathcal{L}_t\|$", red["grads_l2"], "n", "l2") if (cross_section_checkbox.value and cross_norms_checkbox.value) else None
        # per-row variable sets, ranked on the over-n cubes (which fold ALL n and t):
        # the over-t charts display the SAME variables, so ranking always reflects the
        # aggregated influence of the variable over all steps, never a single-step peak
        row_keys = {
            "masked_avg": list(cross_traces(red["gui_gui_ung_wnorm"], "n", "mean", m, **cross_ctl)[0]),
            # row 2 (guided levels) ranks on ITS OWN data, not the gui−gui_ung diffs
            "gui_lvls": list(cross_traces(red["gui_wnorm"], "n", "mean", m, **cross_ctl)[0]),
            "grads": list(cross_traces(red["grads_l2"], "n", "l2", m, **cross_ctl)[0]),
            "gui_vfs": list(cross_traces(red["gui_vfs_l2"], "n", "l2", m, **cross_ctl)[0]),
            "vfs": list(cross_traces(red["vfs_l2"], "n", "l2", m, **cross_ctl)[0]),
            "angle": list(cross_traces(red["dvf_angle"], "n", "mean", m, **cross_ctl)[0]),
        }
        if "kick_lat_l2" in red:
            row_keys["kick_lat"] = list(cross_traces(red["kick_lat_l2"], "n", "l2", m, **cross_ctl)[0])
            kick_lat_n_plot = _plot(r"Guidance kick (latent units)", r"$\|h_t\,\lambda^{\mathrm{raw}}_t\,\nabla_{z_t}\mathcal{L}_t\|$", red["kick_lat_l2"], "n", "l2") if (cross_section_checkbox.value and cross_norms_checkbox.value) else None
            kick_lat_t_plot = _plot(r"Guidance kick (latent units)", r"$\|h_t\,\lambda^{\mathrm{raw}}_t\,\nabla_{z_t}\mathcal{L}_t\|$", red["kick_lat_l2"].isel(n=n), "t", "l2", keys=row_keys["kick_lat"]) if (cross_section_checkbox.value and cross_norms_checkbox.value) else None
        else:
            kick_lat_n_plot = kick_lat_t_plot = None
        if "gui_ung_defl_angle" in red:
            # one row per unguided trajectory, each [over n, over t]; deflection is
            # between the vf at t and the SAME trajectory's vf at t-1. Shared top-k
            # keys (ranked on gui_ung over n) so all four charts show the same set.
            row_keys["angdef_ung"] = list(cross_traces(red["gui_ung_defl_angle"], "n", "mean", m, **cross_ctl)[0])
            angdef_gu_n_plot = _plot(r"Angular deflection — gui_ung", r"$\angle(\mathrm{vf}^{\mathrm{gui\_ung}}_t,\ \mathrm{vf}^{\mathrm{gui\_ung}}_{t-1})$ [$^\circ$]", red["gui_ung_defl_angle"], "n", "mean", keys=row_keys["angdef_ung"]) if (cross_section_checkbox.value and cross_angles_checkbox.value) else None
            angdef_gu_t_plot = _plot(r"Angular deflection — gui_ung", r"$\angle(\mathrm{vf}^{\mathrm{gui\_ung}}_t,\ \mathrm{vf}^{\mathrm{gui\_ung}}_{t-1})$ [$^\circ$]", red["gui_ung_defl_angle"].isel(n=n), "t", "mean", keys=row_keys["angdef_ung"]) if (cross_section_checkbox.value and cross_angles_checkbox.value) else None
            angdef_uo_n_plot = _plot(r"Angular deflection — ung", r"$\angle(\mathrm{vf}^{\mathrm{ung}}_t,\ \mathrm{vf}^{\mathrm{ung}}_{t-1})$ [$^\circ$]", red["ung_defl_angle"], "n", "mean", keys=row_keys["angdef_ung"]) if (cross_section_checkbox.value and cross_angles_checkbox.value) else None
            angdef_uo_t_plot = _plot(r"Angular deflection — ung", r"$\angle(\mathrm{vf}^{\mathrm{ung}}_t,\ \mathrm{vf}^{\mathrm{ung}}_{t-1})$ [$^\circ$]", red["ung_defl_angle"].isel(n=n), "t", "mean", keys=row_keys["angdef_ung"]) if (cross_section_checkbox.value and cross_angles_checkbox.value) else None
        else:
            angdef_gu_n_plot = angdef_gu_t_plot = angdef_uo_n_plot = angdef_uo_t_plot = None
        vf_angdef_n_plot = _plot(r"Angular deflection — gui", r"$\angle(\mathrm{vf}_t,\ \mathrm{vf}^{\mathrm{gui}}_{t-1})$ [$^\circ$]", red["dvf_angle"], "n", "mean") if (cross_section_checkbox.value and cross_angles_checkbox.value) else None
        vf_angdef_t_plot = _plot(r"Angular deflection — gui", r"$\angle(\mathrm{vf}_t,\ \mathrm{vf}^{\mathrm{gui}}_{t-1})$ [$^\circ$]", red["dvf_angle"].isel(n=n), "t", "mean", keys=row_keys["angle"]) if (cross_section_checkbox.value and cross_angles_checkbox.value) else None
        gui_vf_norms_n_plot = _plot(r"Vector field norm — gui step", r"$\|\mathrm{vf}^{\mathrm{gui}}_t\|$", red["gui_vfs_l2"], "n", "l2") if (cross_section_checkbox.value and cross_norms_checkbox.value) else None
        vf_norms_n_plot = _plot(r"Vector field norm — ung step", r"$\|\mathrm{vf}_t\|$", red["vfs_l2"], "n", "l2") if (cross_section_checkbox.value and cross_norms_checkbox.value) else None
        diff_vfs_t_plot = _plot(r"Guided − gui_ung vf", r"$\mathrm{mean}_{\mathrm{mask}}(\mathrm{vf}^{\mathrm{gui}}_t - \mathrm{vf}_t)$", red["dvf_mean"].isel(n=n), "t", "mean") if (cross_section_checkbox.value and cross_norms_checkbox.value) else None
        gui_mean_n_plot = _plot(r"Masked average: guided − gui_ung", r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_n - x^{\mathrm{gui\_ung}}_n)$", _u_diff(red["gui_gui_ung_wnorm"]), "n", "mean", prepend_zero=True) if (cross_section_checkbox.value and cross_masked_checkbox.value) else None
        gui_mean_t_plot = _plot(r"Masked average: guided − gui_ung", r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_t - x^{\mathrm{gui\_ung}}_t)$", _u_diff(red["land_gui_ung_wnorm"]).isel(n=n), "t", "mean", keys=row_keys["masked_avg"]) if (cross_section_checkbox.value and cross_masked_checkbox.value) else None
        # absolute levels: gui (solid) vs gui_ung (dotted twin) for the SAME top-k
        # variables as the masked-avg row (ranked by aggregated |gui - gui_ung|)
        if "state_wnorm" in red and "land_wnorm" in red:
            # first right chart: the guidance-convergence STATE M(x_det + sigma_r z_t)
            # per top-k variable (online), cross-check style, normalized
            state_online_t_plot = _plot(r"Masked average: guided state (online)", r"$\mathrm{mean}_{\mathrm{mask}}\,M(\hat{x}^{\mathrm{det}}+\sigma_r z_t)$", _u_abs(red["state_wnorm"]).isel(n=n), "t", "mean", keys=row_keys["gui_lvls"]) if (cross_section_checkbox.value and cross_masked_checkbox.value) else None
            # second right chart: masked average of the flow residual r_t
            gui_land_t_plot = _plot(r"Masked average: guided", r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_t)$", _u_abs(red["land_wnorm"]).isel(n=n), "t", "mean", keys=row_keys["gui_lvls"]) if (cross_section_checkbox.value and cross_masked_checkbox.value) else None
        else:
            state_online_t_plot = gui_land_t_plot = None
        gui_ung_lvl_n_plot = _plot(r"Masked average: guided", r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_n)$", _u_abs(red["gui_wnorm"]), "n", "mean", keys=row_keys["gui_lvls"], n0=_n0_disp) if (cross_section_checkbox.value and cross_masked_checkbox.value) else None
    else:
        row_keys = None
        gui_ung_lvl_n_plot = None
        kick_lat_n_plot = kick_lat_t_plot = None
        angdef_gu_n_plot = angdef_gu_t_plot = angdef_uo_n_plot = angdef_uo_t_plot = None
        state_online_t_plot = gui_land_t_plot = None
        gui_mean_n_plot = gui_mean_t_plot = vf_angdef_n_plot = vf_angdef_t_plot = grad_norms_n_plot = diff_vfs_t_plot = gui_vf_norms_n_plot = vf_norms_n_plot = None
    return (
        angdef_gu_n_plot,
        angdef_gu_t_plot,
        angdef_uo_n_plot,
        angdef_uo_t_plot,
        grad_norms_n_plot,
        gui_land_t_plot,
        gui_mean_n_plot,
        gui_mean_t_plot,
        gui_ung_lvl_n_plot,
        gui_vf_norms_n_plot,
        kick_lat_n_plot,
        kick_lat_t_plot,
        row_keys,
        vf_angdef_n_plot,
        vf_angdef_t_plot,
        vf_norms_n_plot,
    )


@app.cell
def _(
    cfg_target_guidance_M_N_trajectories,
    config,
    dist_bands_checkbox,
    dpi_slider,
    get_rollout,
    get_slices,
    gui_vfs_xr,
    guided_xr,
    level,
    m,
    mask,
    n,
    notebook_mode,
    np,
    partition,
    plt,
    res_scale_map,
    res_xr,
    rollout_id,
    sweep_params,
    to_display_units,
    var,
    vfs_xr,
):
    if notebook_mode =="analyze_rollout":
        # Euler landings in masked-mean terms, anchored at the pure-noise state:
        # state_t = M(x_det + sigma_r z_t) - y_n (realized guided path; state_0 = noise;
        # res trace stores z_t so res[0] is the initial noise), and
        # land_ung_t = state_t + sigma_r (h_t/s_t) M(s_t u_t): where the RAW flow step
        # would land. The purple-vs-yellow gap at t+1 IS the guidance contribution.
        _mask_np = np.asarray(mask)
        if res_xr is None:
            with plt.rc_context({"font.size": 10}):
                _fig, _ax = plt.subplots(figsize=(22.0, 6), dpi=dpi_slider.value)
                _ax.text(0.5, 0.5, "no res trace on this store (legacy rollout)",
                         ha="center", va="center", transform=_ax.transAxes)
                _ax.set_axis_off()
            guidance_convergence_t_plot = _fig
        else:
            _z_mm  = (np.asarray(get_slices(res_xr, partition, var, level))[:, n] * _mask_np).sum(axis=(-1, -2))
            _u_mm  = (np.asarray(get_slices(vfs_xr, partition, var, level))[:, n] * _mask_np).sum(axis=(-1, -2))
            _gu_mm = (np.asarray(get_slices(gui_vfs_xr, partition, var, level))[:, n] * _mask_np).sum(axis=(-1, -2))
            _M_all = _z_mm.shape[0]
            _T_flow = _u_mm.shape[1]                        # Euler-step count (T); z_mm / states are T+1
            _s_flow = np.linspace(1000, 1, _T_flow) / 1000
            _h_flow = np.empty_like(_s_flow); _h_flow[:-1] = _s_flow[:-1] - _s_flow[1:]; _h_flow[-1] = _s_flow[-1]
            _hs = _h_flow / _s_flow
            _c_sc = res_scale_map[var]
            _c_sc = float(_c_sc.sel(level=level)) if partition == "level" else float(_c_sc)

            # M(x_det): from the gui_det store; older stores -> reconstruct from the
            # final guided state (M(x_det) = M(gui) - sigma_r M(z_T), z_T = z_mm[:, -1])
            try:
                _det_mm = (np.asarray(get_slices(get_rollout("gui_det", rollout_id).sel(sweep_params).compute(),
                                                 partition, var, level))[:, n] * _mask_np).sum(axis=(-1, -2))
            except (FileNotFoundError, KeyError):
                _gui_mm = (np.asarray(get_slices(guided_xr, partition, var, level))[:, n] * _mask_np).sum(axis=(-1, -2))
                _det_mm = _gui_mm - _c_sc * _z_mm[:, -1]

            _is_tgt = (var == config.VAR) and (partition != "level" or level == config.LEVEL)
            _msum_pa = float(np.asarray(_mask_np).sum())
            _y = (np.asarray(cfg_target_guidance_M_N_trajectories)[:, n] if _is_tgt
                  else np.zeros(_M_all))
            _states = _det_mm[:, None] + _c_sc * _z_mm - _y[:, None]   # M(x_t) - A, t=0..T (ends at x_T)
            _land_ung = _states[:, :_T_flow] + _c_sc * _hs * _u_mm     # where the raw flow step lands (T)
            _pa_unit = ""
            if not _is_tgt:
                # absolute mask-averaged values, display units (K -> degC etc.)
                _states, _pa_unit = to_display_units(_states / _msum_pa, var)
                _land_ung = to_display_units(_land_ung / _msum_pa, var)[0]

            _xt = np.arange(_states.shape[1]).astype(float)   # T+1 ticks (0..T)
            _wt = 22.0  # match the rollout trajectories figure width
            with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "legend.fontsize": 9}):
                _fig, _ax = plt.subplots(figsize=(_wt, 6), dpi=dpi_slider.value)
                if dist_bands_checkbox.value and _M_all > 1:
                    _ax.fill_between(_xt, _states.min(axis=0), _states.max(axis=0),
                                     color="#B7950B", alpha=0.14, linewidth=0, label=f"state range, M={_M_all}")
                # waterfall candles at every arrival tick t+1, in the old visual
                # grammar: red = the flow's OWN move (state_t -> unguided landing),
                # blue = the GUIDANCE move stacked on top (unguided -> guided landing).
                # The two stack to the full step; circles mark the realized states.
                _bar_off = 0.16
                _flow_move = _land_ung[m] - _states[m, :_T_flow]
                _gui_move = _states[m, 1:] - _land_ung[m]
                _ax.bar(_xt[1:] - _bar_off, _flow_move, bottom=_states[m, :_T_flow], width=0.28,
                        color="#C0392B", alpha=0.35, edgecolor="#C0392B", linewidth=0.5, zorder=3,
                        label=r"flow move  (state$_t$ $\to$ gui\_ung landing)")
                _ax.bar(_xt[1:] + _bar_off, _gui_move, bottom=_land_ung[m], width=0.28,
                        color="#2E86C1", alpha=0.35, edgecolor="#2E86C1", linewidth=0.5, zorder=3,
                        label=r"guidance move  (gui\_ung $\to$ guided landing)")
                # thin trajectory line + filled state circles / landing squares
                _ax.plot(_xt, _states[m], "-", color="#B7950B", alpha=0.5, linewidth=1.1, zorder=4)
                _ax.plot(_xt, _states[m], "o", color="#B7950B", markeredgecolor="white",
                         markeredgewidth=0.8, markersize=5.5, linestyle="none", zorder=7,
                         label=r"state  $M(\hat{x}^{\text{gui\_det}}+\sigma_r z_t) - y_n$")
                # branching sparks: from each realized state to where the SAVED unguided
                # vf would land (rollout-trajectories guided_unguided grammar), in the
                # landing marker's violet
                for _ti in range(_T_flow):
                    _ax.plot([_xt[_ti], _xt[_ti + 1]], [_states[m, _ti], _land_ung[m, _ti]],
                             linestyle="--", linewidth=1.2, color="#800080", alpha=0.6,
                             zorder=5, label="_nolegend_")
                _ax.plot(_xt[1:], _land_ung[m], "o", color="#800080", markeredgecolor="white",
                         markeredgewidth=0.8, markersize=5.5, linestyle="none", zorder=6,
                         label=r"gui\_ung landing  $M(\hat{x}^{\text{gui\_det}}+\sigma_r(z_t+h_t u_t)) - y_n$")
                if _is_tgt:
                    _ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
                _ax.set_xlim(-1.0, _T_flow + 0.4)
                _ax.set_xticks(_xt)
                # breathing room so the noise anchor doesn't sit on the axis, then a
                # direction-arrow strip inside the bottom margin (red row above blue)
                _ax.margins(y=0.06)
                _ylo, _yhi = _ax.get_ylim()
                _yr = _yhi - _ylo
                _ylo = _ylo - 0.09 * _yr  # dedicated band under the data for the arrows
                for _moves, _col, _off, _ystrip in (
                    (_flow_move, "#C0392B", -_bar_off, _ylo + 0.055 * _yr),
                    (_gui_move, "#2E86C1", +_bar_off, _ylo + 0.022 * _yr),
                ):
                    _up = _moves >= 0
                    _ax.scatter(_xt[1:][_up] + _off, np.full(int(_up.sum()), _ystrip),
                                marker="^", s=16, color=_col, alpha=0.9, zorder=5)
                    _ax.scatter(_xt[1:][~_up] + _off, np.full(int((~_up).sum()), _ystrip),
                                marker="v", s=16, color=_col, alpha=0.9, zorder=5)
                _ax.set_ylim(_ylo, _yhi)
                # anchor/end annotations + target label on the zero line
                _ax.annotate("noise", (_xt[0], _states[m, 0]), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=8, color="#666666")
                _ax.annotate("final", (_xt[-1], _states[m, -1]), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=8, color="#666666")
                if _is_tgt:
                    _ax.annotate("target", (_xt[-1], 0.0), textcoords="offset points",
                                 xytext=(30, 0), ha="left", va="center", fontsize=8,
                                 color="#888888", annotation_clip=False)
                _ax.set_xlabel("$t$")
                _ax.set_ylabel("masked mean − target" if _is_tgt else
                               (f"Mask-averaged value [{_pa_unit}]" if _pa_unit else "Mask-averaged value"))
                _ax.set_title("Guidance convergence — guided vs gui_ung landings", loc="left", fontweight="bold", fontsize=15, color="#222222", pad=24)
                _ax.text(0.0, 1.015, f"{var} | per flow step: red = flow move, blue = guidance move   (m={m}, n={n})",
                         transform=_ax.transAxes, fontsize=9.5, color="#555555", va="bottom")
                for _sp in ("top", "right"):
                    _ax.spines[_sp].set_visible(False)
                _ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
                _ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
                _fig.tight_layout(rect=(0, 0, 0.82, 1))
            guidance_convergence_t_plot = _fig
    else:
        guidance_convergence_t_plot = None
    return (guidance_convergence_t_plot,)


@app.cell
def _(mcolors, mo, np, plt):
    contour_checkbox = mo.ui.checkbox(label="contours", value=True)
    contour_levels_slider = mo.ui.slider(4, 30, step=2, value=24, label="levels: ", show_value=True, debounce=True)
    white_zero_cmap = plt.get_cmap("RdBu_r").copy()
    white_zero_cmap.set_bad("white")
    # half-range maps for single-signed absolute fields (white stays at the
    # zero end, like the mask map)
    warm_half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_warm", white_zero_cmap(np.linspace(0.5, 1.0, 256)))
    cool_half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_cool", white_zero_cmap(np.linspace(0.0, 0.5, 256)))
    return (
        contour_checkbox,
        contour_levels_slider,
        cool_half_cmap,
        warm_half_cmap,
        white_zero_cmap,
    )


@app.cell(hide_code=True)
def _(mo):
    contour_color_dropdown = mo.ui.dropdown(["dimgray", "white", "black"], value="black", label="contour color: ")
    return (contour_color_dropdown,)


@app.cell
def _(A_T_MODES, GUIDANCE_METHODS, GUI_REFS, MASK_MODES, mo, np):
    # ===== sweep authoring widgets (guided_rollout) =====
    guidance_mode_select = mo.ui.multiselect(GUIDANCE_METHODS, value=["FGWNOLR"], label="GUIDANCE_MODE: ")
    gui_ref_select = mo.ui.multiselect(GUI_REFS, value=["UNG"], label="GUI_REF: ")
    mask_mode_select = mo.ui.multiselect(MASK_MODES, value=["BBOX"], label="MASK_MODE: ")
    a_t_mode_select = mo.ui.multiselect(A_T_MODES, value=["gap-closing"], label="A_T_MODE: ")

    # spread@{start}-{end}: linear ramp 1->0 over the flow window [start,end], 0 outside (eta
    # unused). Author how MANY windows here (like the target-percentage profiles); each window's
    # [start,end] is a range slider in its own row (spread_range_controls) shown with a mini
    # profile, and appended to a_t_mode at save time.
    n_spreads_slider = mo.ui.slider(0, 8, step=1, value=0, label="spread@ windows: ", show_value=True, debounce=True)

    # numeric axes -> (start, stop, log_scale, integer)
    # keys equal the guidance-fn kwarg names (see GUIDANCE_METHOD_HYPERS)
    NUMERIC_AXES = {
        # FGWNOLR (secant on the exact scalar dL/dw; no lr, no iteration count --
        # optimizes until the hardcoded loss threshold in _fgwnolr_flow is reached)
        "fgwnolr_w_init": (5000.0, 10000.0, False, False),
        # eta: shared profile parameter (meaning depends on a_t_mode -- closure rate
        # for gap-closing, bell depth for gaussian, end level for linear/logistic)
        "eta":            (0.01,  1.0,   False, False),
        # sigma_div: mask hyper shared by ALL mask modes -- extent / sigma_div
        # (2.0 = base box, 4.0 = half, 1.0 = double)
        "sigma_div":      (2,   4.0,   False, False),
        # phi: FGWFREE kick-energy regularizer strength (log-scaled authoring range)
        "phi":            (0.01,  1.0,   True,  False),
    }

    _rc = {}
    for _ax, (_s, _e, _log, _int) in NUMERIC_AXES.items():
        _rc[f"{_ax}.start"] = mo.ui.number(value=_s, label="start: ")
        _rc[f"{_ax}.stop"]  = mo.ui.number(value=_e, label="stop: ")
        _rc[f"{_ax}.n"]     = mo.ui.slider(1, 20, step=1, value=1, label="n: ", show_value=True, debounce=True)
        _rc[f"{_ax}.log"]   = mo.ui.checkbox(value=_log, label="log")
    sweep_ranges = mo.ui.dictionary(_rc)


    def compute_axis_values(ax, rv):
        s, e = rv[f"{ax}.start"], rv[f"{ax}.stop"]
        nn = int(rv[f"{ax}.n"] or 1)
        log = rv[f"{ax}.log"]
        integer = NUMERIC_AXES[ax][3]
        if nn <= 1:
            vals = [s]
        elif log and s > 0 and e > 0:
            vals = list(np.logspace(np.log10(s), np.log10(e), nn))
        else:
            vals = list(np.linspace(s, e, nn))
        return sorted({int(round(v)) for v in vals}) if integer else [round(float(v), 8) for v in vals]


    return (
        NUMERIC_AXES,
        a_t_mode_select,
        compute_axis_values,
        gui_ref_select,
        guidance_mode_select,
        mask_mode_select,
        n_spreads_slider,
        sweep_ranges,
    )


@app.cell
def _(config, mo, n_spreads_slider, notebook_mode):
    # per-window range sliders: each picks a [start, end] flow-step window for one spread@ mode.
    # Recreated when the count changes (values reset to the full window), like the delta controls.
    if notebook_mode == "guided_rollout":
        _T = int(config.T or 25)
        spread_range_controls = mo.ui.dictionary({
            f"{_i}.range": mo.ui.range_slider(1, _T, step=1, value=[1, _T],
                                              label=f"window {_i} [t]: ", show_value=True, debounce=True)
            for _i in range(int(n_spreads_slider.value))
        })
    else:
        spread_range_controls = mo.ui.dictionary({})
    return (spread_range_controls,)


@app.cell
def _(
    a_t_profile,
    config,
    mo,
    n_spreads_slider,
    notebook_mode,
    plt,
    spread_range_controls,
):
    # build spread@ modes from the range sliders, each row shown with its own mini 1->0 profile
    if notebook_mode == "guided_rollout":
        _T = int(config.T or 25)
        _sv = spread_range_controls.value
        _modes = []
        _srows = []
        for _i in range(int(n_spreads_slider.value)):
            _lo, _hi = _sv[f"{_i}.range"]
            _lo, _hi = int(min(_lo, _hi)), int(max(_lo, _hi))
            _mode = f"spread@{_lo}-{_hi}"
            _prof = a_t_profile(_mode, 1.0, _T).copy()   # eta ignored for spread@
            _prof[-1] = 0.0                              # the last flow step is never guided
            _active = bool(_prof.any())                  # False -> window sits only on the last tick
            if _active:
                _modes.append(_mode)                     # else a silent no-op -> not swept (like spike@ cap)
            _col = "#e42536" if _active else "#bbbbbb"
            _pf, _pax = plt.subplots(figsize=(2.6, 0.8), dpi=100)
            _pax.plot(range(1, _T + 1), _prof, "-", color=_col, linewidth=1.2)
            _pax.fill_between(range(1, _T + 1), _prof, color=_col, alpha=0.15)
            _pax.set_ylim(-0.05, 1.05); _pax.set_xlim(1, _T)
            _pax.set_xticks([]); _pax.set_yticks([])
            _pax.set_title(_mode if _active else f"{_mode} (no guidance)", fontsize=7)
            for _s in _pax.spines.values():
                _s.set_visible(False)
            _pf.tight_layout(pad=0.1)
            _srows.append(mo.hstack([spread_range_controls[f"{_i}.range"], mo.as_html(_pf)],
                                    justify="start", align="center"))
            plt.close(_pf)
        # identical windows collapse to one sweep point (order preserved)
        spread_modes = list(dict.fromkeys(_modes))
        spread_widget = mo.vstack([n_spreads_slider, *_srows], align="start")
    else:
        spread_modes = []
        spread_widget = None
    return spread_modes, spread_widget


@app.cell(hide_code=True)
def _(mo):
    mask_shift_select = mo.ui.multiselect(["none", "right", "up", "down", "left"], value=["none"], label="MASK_SHIFT: ")
    return (mask_shift_select,)


@app.cell(hide_code=True)
def _(mo):
    mask_shift_px_slider = mo.ui.slider(1, 10, step=1, value=3, label="shift px: ", show_value=True, debounce=True)
    return (mask_shift_px_slider,)


@app.cell(hide_code=True)
def _(mo):
    mask_shift_preview_dropdown = mo.ui.dropdown(["none", "right", "up", "down", "left"], value="none", label="preview shift: ")
    return (mask_shift_preview_dropdown,)


@app.cell
def _(mo):
    side_lon_slider = mo.ui.slider(1.5, 90, step=1.5, value=12, label="lon side: ", show_value=True, debounce=True)
    side_lat_slider = mo.ui.slider(1.5, 60, step=1.5, value=10, label="lat side: ", show_value=True, debounce=True)
    sigma_div_slider = mo.ui.slider(steps=[0.25, 0.5, 1, 2, 4], value=2, label="sigma divisor: ", show_value=True, debounce=True)
    return side_lat_slider, side_lon_slider, sigma_div_slider


@app.cell
def _(
    cfg_target_guidance_M_N_trajectories,
    color_for,
    config,
    dask,
    dpi_slider,
    get_rollout,
    gui_ung_xr,
    gui_vfs_xr,
    guided_xr,
    m,
    mask,
    n,
    notebook_mode,
    np,
    plt,
    res_scale_map,
    res_xr,
    rollout_id,
    row_keys,
    sweep_params,
    vfs_xr,
    xnorm,
    xr,
):
    # Guidance convergence per variable: the convergence-waterfall grammar (Euler
    # landings, red flow move / blue guidance move candles) replicated for the same
    # top-k variable selection as the masked-avg / gui-vs-ung rows. One panel per
    # variable, absolute masked-mean units on each panel's own scale; the guided
    # variable's panel carries the target line. Level-aggregated keys use the plain
    # level mean (surface twin not folded in, unlike cross_traces).
    if notebook_mode == "analyze_rollout":
        if res_xr is None:
            with plt.rc_context({"font.size": 10}):
                _fig_cv, _ax_cv = plt.subplots(figsize=(22.0, 4), dpi=dpi_slider.value)
                _ax_cv.text(0.5, 0.5, "no res trace on this store (legacy rollout)",
                            ha="center", va="center", transform=_ax_cv.transAxes)
                _ax_cv.set_axis_off()
            conv_topk_t_plot = _fig_cv
            gui_ung_lvl_t_plot = _fig_cv
            gui_ung_lvl_step_t_plot = _fig_cv
        else:
            _mw_cv = xr.DataArray(np.asarray(mask, dtype=float), dims=("latitude", "longitude"),
                                  coords={"latitude": guided_xr.latitude, "longitude": guided_xr.longitude})
            try:
                _det_ds_cv = get_rollout("gui_det", rollout_id).sel(sweep_params)
            except (FileNotFoundError, KeyError):
                _det_ds_cv = None  # older store: det reconstructed from the final guided state
            _keys_cv = list(row_keys["masked_avg"])

            def _cv_parse(k_):
                if " L" in k_:
                    _v, _l = k_.rsplit(" L", 1)
                    return _v, int(_l)
                return k_, None

            def _cv_mm(da, lev, scale=None):
                # M operator of the convergence plot: mask-weighted spatial sum at the
                # selected n; sigma_r scaling applied BEFORE any level mean
                if scale is not None:
                    da = da * scale
                if lev is not None:
                    da = da.sel(level=lev)
                elif "level" in da.dims:
                    da = da.mean("level")
                return (da.isel(n=n) * _mw_cv).sum(("latitude", "longitude"))

            _lazy_cv = {}
            for _k in _keys_cv:
                _v, _lv = _cv_parse(_k)
                _sc = res_scale_map[_v]
                _e = dict(
                    z=_cv_mm(res_xr[_v], _lv, _sc),
                    u=_cv_mm(vfs_xr[_v], _lv, _sc),
                    gu=_cv_mm(gui_vfs_xr[_v], _lv, _sc),
                    ug=_cv_mm(gui_ung_xr[_v], _lv),
                )
                if _det_ds_cv is not None:
                    _e["det"] = _cv_mm(_det_ds_cv[_v], _lv)
                else:
                    _e["gui"] = _cv_mm(guided_xr[_v], _lv)
                _lazy_cv[_k] = _e
            _flat_cv = dask.compute(_lazy_cv)[0]

            _T_cv = vfs_xr.sizes["t"]                    # Euler-step count (T); res/states are T+1
            _s_cv = np.linspace(1000, 1, _T_cv) / 1000
            _h_cv = np.empty_like(_s_cv); _h_cv[:-1] = _s_cv[:-1] - _s_cv[1:]; _h_cv[-1] = _s_cv[-1]
            _hs_cv = _h_cv / _s_cv
            _xt_cv = np.arange(_T_cv + 1).astype(float)
            _tgt_key = config.VAR if config.PARTITION == "surface" else f"{config.VAR} L{config.LEVEL}"
            _y_cv = float(np.asarray(cfg_target_guidance_M_N_trajectories)[m, n])

            _kcv = len(_keys_cv)
            with plt.rc_context({"font.size": 9, "legend.fontsize": 8}):
                _fig_cv, _axs_cv = plt.subplots(_kcv, 1, figsize=(22.0, 2.4 * _kcv + 1.0),
                                                dpi=dpi_slider.value, sharex=True, squeeze=False)
                for _pi, _k in enumerate(_keys_cv):
                    _axc = _axs_cv[_pi, 0]
                    _d = _flat_cv[_k]
                    _zmm = np.atleast_2d(np.asarray(_d["z"]))    # (M, T), sigma_r-scaled
                    _umm = np.atleast_2d(np.asarray(_d["u"]))
                    _gumm = np.atleast_2d(np.asarray(_d["gu"]))
                    if "det" in _d:
                        _detv = float(np.atleast_1d(np.asarray(_d["det"]))[m])
                    else:
                        _detv = float(np.atleast_1d(np.asarray(_d["gui"]))[m]) - _zmm[m, -1]  # z_T = z[-1]
                    _st = _detv + _zmm[m]                # state trajectory (T+1), reaches x_T at index T
                    _lu = _st[:_T_cv] + _hs_cv * _umm[m] # where the raw flow step lands (T)
                    _fm = _lu - _st[:_T_cv]              # flow move (state_t -> ung landing)
                    _gm = _st[1:] - _lu                  # guidance move (ung -> gui landing)
                    _axc.bar(_xt_cv[1:] - 0.16, _fm, bottom=_st[:_T_cv], width=0.28, color="#C0392B",
                             alpha=0.35, edgecolor="#C0392B", linewidth=0.5, zorder=3, label="flow move")
                    _axc.bar(_xt_cv[1:] + 0.16, _gm, bottom=_lu, width=0.28, color="#2E86C1",
                             alpha=0.35, edgecolor="#2E86C1", linewidth=0.5, zorder=3, label="guidance move")
                    _axc.plot(_xt_cv, _st, "-", color="#B7950B", alpha=0.5, linewidth=1.0, zorder=4)
                    _axc.plot(_xt_cv, _st, "o", color="#B7950B", markeredgecolor="white",
                              markeredgewidth=0.7, markersize=4.5, linestyle="none", zorder=7, label="state")
                    _axc.plot(_xt_cv[1:], _lu, "s", color="#800080", markeredgecolor="white",
                              markeredgewidth=0.5, markersize=3.2, linestyle="none", zorder=6,
                              label="unguided landing")
                    if _k == _tgt_key:
                        _axc.axhline(_y_cv, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
                        _axc.annotate("target", (_xt_cv[-1], _y_cv), textcoords="offset points",
                                      xytext=(8, 0), ha="left", va="center", fontsize=8,
                                      color="#888888", annotation_clip=False)
                    _axc.margins(y=0.08)
                    _ylo_, _yhi_ = _axc.get_ylim(); _yr_ = _yhi_ - _ylo_
                    _ylo_ = _ylo_ - 0.14 * _yr_          # arrow strip band under the data
                    for _mv, _cl, _of, _ys in ((_fm, "#C0392B", -0.16, _ylo_ + 0.085 * _yr_),
                                               (_gm, "#2E86C1", +0.16, _ylo_ + 0.035 * _yr_)):
                        _up_ = _mv >= 0
                        _axc.scatter(_xt_cv[1:][_up_] + _of, np.full(int(_up_.sum()), _ys),
                                     marker="^", s=10, color=_cl, alpha=0.9, zorder=5)
                        _axc.scatter(_xt_cv[1:][~_up_] + _of, np.full(int((~_up_).sum()), _ys),
                                     marker="v", s=10, color=_cl, alpha=0.9, zorder=5)
                    _axc.set_ylim(_ylo_, _yhi_)
                    _axc.set_ylabel(_k, fontsize=8.5, color=color_for([_k])[_k])
                    for _sp_ in ("top", "right"):
                        _axc.spines[_sp_].set_visible(False)
                    _axc.yaxis.grid(True, color="#D7D7D7", linewidth=0.6, alpha=0.55)
                _axs_cv[0, 0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
                _axs_cv[-1, 0].set_xlabel("$t$")
                _axs_cv[-1, 0].set_xlim(-1.0, _T_cv + 0.4)
                _axs_cv[-1, 0].set_xticks(_xt_cv)
                _axs_cv[0, 0].set_title("Guidance convergence — top-k variables", loc="left",
                                        fontweight="bold", fontsize=15, color="#222222", pad=24)
                _axs_cv[0, 0].text(0.0, 1.05, f"per flow step: red = flow move, blue = guidance move   (m={m}, n={n})",
                                   transform=_axs_cv[0, 0].transAxes, fontsize=9.5, color="#555555", va="bottom")
                _fig_cv.tight_layout(rect=(0, 0, 0.87, 1))
            conv_topk_t_plot = _fig_cv

            # --- gui vs ung levels (over t): guided CLEAN PREDICTIONS per top-k
            # variable, based as the masked difference to the gui_ung reference run's
            # clean predictions at the same t (comparable across variables, scaled by
            # each variable's std -- the mean cancels). x_hat = det + sigma_r*(z + s*u):
            # solid = guided heading (z + s*u_gui, kick included); dashed spark = the
            # model's raw clean prediction from the same state (saved unguided vf),
            # branching off the previous guided point. Level-aggregated keys use
            # level-averaged stats (display approximation).
            _msum_cv = float(_mw_cv.sum())
            _lvl_vals = list(np.asarray(res_xr.level.values)) if "level" in res_xr.dims else []

            def _cv_zstats(v_, lv_):
                _mu, _sd = xnorm._mean[v_], xnorm._std[v_]
                if np.ndim(_mu) == 1:
                    if lv_ is None:
                        return float(np.mean(_mu)), float(np.mean(_sd))
                    _ix = _lvl_vals.index(lv_)
                    return float(_mu[_ix]), float(_sd[_ix])
                return float(_mu), float(_sd)

            _xs_sp = np.arange(1, _T_cv + 1).astype(float)  # step t evaluated -> tick t+1
            with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "legend.fontsize": 8}):
                _fig_sp, _ax_sp = plt.subplots(figsize=(22.0, 6), dpi=dpi_slider.value)
                _cols_sp = color_for(_keys_cv)
                for _k in _keys_cv:
                    _v, _lv = _cv_parse(_k)
                    _d = _flat_cv[_k]
                    _zmm = np.atleast_2d(np.asarray(_d["z"]))[m]    # sigma_r-scaled M-sums
                    _umm = np.atleast_2d(np.asarray(_d["u"]))[m]
                    _gumm = np.atleast_2d(np.asarray(_d["gu"]))[m]
                    _ugmm = np.atleast_2d(np.asarray(_d["ug"]))[m]  # gui_ung clean preds (physical)
                    if "det" in _d:
                        _detv = float(np.atleast_1d(np.asarray(_d["det"]))[m])
                    else:
                        _detv = float(np.atleast_1d(np.asarray(_d["gui"]))[m]) - _zmm[-1]  # z_T = z[-1]
                    _z_pre = _zmm[:_T_cv]           # z_0..z_{T-1} aligned with the length-T vfs/gui_vfs
                    _cpg = _detv + _z_pre + _gumm  # guided clean prediction (T)
                    _cpu = _detv + _z_pre + _umm   # model's raw (unguided-vf) clean prediction (T)
                    _sd_ = _cv_zstats(_v, _lv)[1]
                    _stn = (_cpg - _ugmm[:_T_cv]) / _msum_cv / _sd_
                    _tipn = (_cpu - _ugmm[:_T_cv]) / _msum_cv / _sd_
                    _cl = _cols_sp[_k]
                    _ax_sp.plot(_xs_sp, _stn, "-o", color=_cl, linewidth=1.4, markersize=4.5,
                                markeredgecolor="white", markeredgewidth=0.7, zorder=6, label=_k)
                    for _ti in range(1, _T_cv):
                        _ax_sp.plot([_xs_sp[_ti - 1], _xs_sp[_ti]], [_stn[_ti - 1], _tipn[_ti]],
                                    linestyle="--", linewidth=1.0, color=_cl, alpha=0.55,
                                    zorder=4, label="_nolegend_")
                    _ax_sp.scatter(_xs_sp, _tipn, s=16, color=_cl, alpha=0.75,
                                   edgecolors="white", linewidths=0.5, zorder=5, label="_nolegend_")
                _ax_sp.plot([], [], linestyle="--", color="#888888", linewidth=1.0,
                            label="unguided clean prediction (saved vf)")
                _ax_sp.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
                _ax_sp.set_xlim(0.0, _T_cv + 0.4)
                _ax_sp.set_xticks(np.arange(0, _T_cv + 1))
                _ax_sp.set_xlabel("$t$"); _ax_sp.set_ylabel("gui − gui_ung  (normalized units)")
                _ax_sp.set_title("Masked average: guided vs gui_ung clean predictions", loc="left",
                                 fontweight="bold", fontsize=15, color="#222222", pad=24)
                _ax_sp.text(0.0, 1.015,
                            f"solid = guided clean predictions minus the unguided reference; dashed spark = the model's raw clean prediction (saved unguided vf)   (m={m}, n={n})",
                            transform=_ax_sp.transAxes, fontsize=9.5, color="#555555", va="bottom")
                for _sp_ in ("top", "right"):
                    _ax_sp.spines[_sp_].set_visible(False)
                _ax_sp.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
                _ax_sp.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
                _fig_sp.tight_layout(rect=(0, 0, 0.87, 1))
            gui_ung_lvl_t_plot = _fig_sp

            # --- companion: per-step difference between the guided and unguided
            # ONLINE steps. Both steps start from the same guided state z_t; the
            # unguided one uses the vf saved before the kick, so the difference is
            # exactly the guidance move sigma_r*(h_t/s_t)*(vf^gui_t - vf_t) in
            # masked-mean terms, per variable, std-scaled. The step at t lands at
            # tick t+1; the last step is never kicked -> 0.
            _xs_st2 = np.arange(1, _T_cv + 1).astype(float)
            with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "legend.fontsize": 8}):
                _fig_sp2, _ax_sp2 = plt.subplots(figsize=(22.0, 6), dpi=dpi_slider.value)
                for _k in _keys_cv:
                    _v, _lv = _cv_parse(_k)
                    _d = _flat_cv[_k]
                    _umm = np.atleast_2d(np.asarray(_d["u"]))[m]
                    _gumm = np.atleast_2d(np.asarray(_d["gu"]))[m]
                    _sd_ = _cv_zstats(_v, _lv)[1]
                    _stepd = _hs_cv * (_gumm - _umm) / _msum_cv / _sd_
                    _ax_sp2.plot(_xs_st2, _stepd, "-o", color=_cols_sp[_k], linewidth=1.4,
                                 markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
                                 zorder=6, label=_k)
                _ax_sp2.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
                _ax_sp2.set_xlim(-1.0, _T_cv + 0.4)
                _ax_sp2.set_xticks(np.arange(0, _T_cv + 1))
                _ax_sp2.set_xlabel("$t$"); _ax_sp2.set_ylabel("gui − gui_ung step  (normalized units)")
                _ax_sp2.set_title(r"gui step − ung step (online)",
                                  loc="left", fontweight="bold", fontsize=15, color="#222222", pad=24)
                _ax_sp2.text(0.0, 1.015,
                             r"$\sigma_r(h_t/s_t)(\mathrm{vf}^{\mathrm{gui}}_t - \mathrm{vf}_t)$",
                             transform=_ax_sp2.transAxes, fontsize=9.5, color="#555555", va="bottom")
                for _sp_ in ("top", "right"):
                    _ax_sp2.spines[_sp_].set_visible(False)
                _ax_sp2.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
                _ax_sp2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
                _fig_sp2.tight_layout(rect=(0, 0, 0.87, 1))
            gui_ung_lvl_step_t_plot = _fig_sp2
    else:
        conv_topk_t_plot = None
        gui_ung_lvl_t_plot = None
        gui_ung_lvl_step_t_plot = None
    return


@app.cell
def _(
    det_n_slice,
    dpi_slider,
    gt_curr,
    gui_curr,
    gui_ung_curr,
    m,
    mask,
    n,
    notebook_mode,
    np,
    plt,
    to_display_units,
    ung_curr,
    var,
):
    # Distribution of the deterministic, guided and unguided states at the current
    # (m, n, var, level) slice, restricted to the mask CORE (half-maximum region --
    # the box for BBOX, the blob core for ELLIPTICAL; NOT mask_region, whose >1e-6
    # support spans half the globe for the elliptical tails).
    if notebook_mode == "analyze_rollout":
        _hist_specs = [
            (r"$x_n^{\text{gui\_det}}$", det_n_slice, "#AAAAAA", "-"),
            (r"$x_n^{\text{gui}}$", gui_curr, "#0072B2", "-"),
            (r"$x_n^{\text{gui_ung}}$", gui_ung_curr, "#800080", "-"),
            # ung: dashed grey-green; lies exactly on gui_ung when seeds are shared
            (r"$x_n^{\text{ung}}$", ung_curr, "#8B5A2B", "-"),
            (r"$x_n^{\text{gt}}$", gt_curr, "#009E73", "-"),
        ]
        _mreg_h = np.asarray(mask) >= 0.5 * float(np.asarray(mask).max())
        with plt.rc_context({"font.size": 10}):
            _fig_h, _ax_h = plt.subplots(figsize=(7.5, 6), dpi=dpi_slider.value)
            _unit_h = ""
            _any_finite = False
            for _lbl, _arr, _cl, _ls in _hist_specs:
                _v, _unit_h = to_display_units(np.asarray(_arr, dtype=float), var)
                _v = np.asarray(_v, dtype=float)[_mreg_h]
                _v = _v[np.isfinite(_v)]
                if _v.size == 0:
                    continue  # invalid var/level selection -> gated slices are all-NaN
                _any_finite = True
                _ax_h.hist(_v, bins=40, density=True, histtype="step",
                           color=_cl, linewidth=1.6, linestyle=_ls, label=_lbl)
            if not _any_finite:
                _ax_h.text(0.5, 0.5, "no data for this selection", ha="center", va="center",
                           fontsize=11, color="#888888", transform=_ax_h.transAxes)
            _ax_h.set_xlabel(f"{var} [{_unit_h}]" if _unit_h else var)
            _ax_h.set_ylabel("density")
            _ax_h.set_title("State distribution in the mask core", loc="left",
                            fontweight="bold", fontsize=15, color="#222222", pad=20)
            _ax_h.text(0.0, 1.01, f"current slice (m={m}, n={n})",
                       transform=_ax_h.transAxes, fontsize=9.5, color="#555555", va="bottom")
            for _sp_ in ("top", "right"):
                _ax_h.spines[_sp_].set_visible(False)
            _ax_h.yaxis.grid(True, color="#EEEEEE")
            _ax_h.set_axisbelow(True)
            _ax_h.legend(frameon=False, fontsize=9)
            _fig_h.tight_layout()
        state_hist_plot = _fig_h
    else:
        state_hist_plot = None
    return (state_hist_plot,)


@app.cell
def _(base_get_N_slices, base_get_slices, np, var, var_valid):
    def _gate_selected(_sl, _var):
        if not var_valid and _var == var:
            return np.full(np.asarray(_sl).shape, np.nan)
        return _sl

    def get_slices(states, partition_, var_, level_):
        return _gate_selected(base_get_slices(states, partition_, var_, level_), var_)

    def get_N_slices(states, partition_, var_, level_):
        return _gate_selected(base_get_N_slices(states, partition_, var_, level_), var_)

    return (get_slices,)


@app.cell
def _():
    red_cache = {}  # (rollout_id, sweep_params, spatial-agg) -> reduced `red` dict
    return (red_cache,)


@app.cell
def _(
    build_reductions_button,
    build_reductions_store,
    notebook_mode,
    reductions_grid_matches,
    rollout_id,
):
    # build on click if missing/stale; expose reductions_ready so the `red` cell
    # re-runs (switches to the disk-read path) once the store is available.
    if notebook_mode == "analyze_rollout":
        if build_reductions_button is not None and build_reductions_button.value and not reductions_grid_matches(rollout_id):
            build_reductions_store(rollout_id)
        reductions_ready = reductions_grid_matches(rollout_id)
    else:
        reductions_ready = False
    return (reductions_ready,)


@app.cell
def _(
    get_masked_mean,
    get_slices,
    hour_slider,
    level,
    mask,
    notebook_mode,
    partition,
    to_display_units,
    trajectories_section_checkbox,
    var,
    year_dropdown,
):
    from src.utils import get_xr_dataset

    # unguided-authoring only: mask-averaged <var> across the FULL YEAR at the selected
    # hour of day. 6-hourly ERA5 GT -> filter to the chosen hour -> one point per day, then
    # mask-average. Kept separate from the plot so the rolling-window slider re-plots
    # without reloading the year. Mirrors the gt_trajectory build (get_slices + masked_mean).
    if trajectories_section_checkbox.value and notebook_mode == "unguided_rollout":
        _ds = get_xr_dataset(year_dropdown.value)
        _ds_h = _ds.sel(time=_ds.time.dt.hour == hour_slider.value)
        year_dates = _ds_h["time"].values
        year_series, year_unit = to_display_units(
            get_masked_mean(get_slices(_ds_h, partition, var, level), mask), var
        )
    else:
        year_dates = year_series = year_unit = None
    return year_dates, year_series, year_unit


@app.cell
def _(mo):
    climatology_rolling_slider = mo.ui.slider(
        1, 31, value=7, step=1, label="rolling avg (days): ", show_value=True, debounce=True
    )
    return (climatology_rolling_slider,)


@app.cell
def _(
    climatology_rolling_slider,
    dpi_slider,
    hour_slider,
    var,
    year_dates,
    year_dropdown,
    year_series,
    year_unit,
):
    from src.ui.plot_climatology import plot_climatology

    # full-year mask climatology with a rolling-average overlay and hottest-window marker;
    # the rolling-window slider only re-runs this cell (the year load lives upstream).
    if year_series is not None:
        year_trajectory_plot = plot_climatology(
            year_dates,
            year_series,
            rolling_days=climatology_rolling_slider.value,
            var=var,
            unit=year_unit,
            title="Mask climatology (full year @ selected hour)",
            subtitle=f"{var} | mask-averaged | {hour_slider.value:02d}Z | {year_dropdown.value}",
            figsize=(22, 6),
            dpi=dpi_slider.value,
        )
    else:
        year_trajectory_plot = None
    return (year_trajectory_plot,)


if __name__ == "__main__":
    app.run()
