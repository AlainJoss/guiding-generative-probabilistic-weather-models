import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch
    import numpy as np
    import xarray as xr
    import dask
    import matplotlib.pyplot as plt
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
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.helpers import max_day, get_timestamp_from_sliders
    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_trajectories import plot_trajectories

    from src.utils import get_var_idx, get_level_idx
    from src.utils import get_now_timestamp, ensure_rollout_dir
    from src.utils import get_timestamps, get_N_timestamps, get_N_slices, get_slices, get_gt_rollout
    from src.utils import (
        dump_json, get_rollout_ids, get_rollout, get_sweep_dict, get_config, sweep_coord_label
    )
    from src.utils import get_w_star, get_guidance_schedule
    from src.schedules import N_schedule, delta_schedule
    from src.normalization import XarrayNormalizer
    from src.spectrum import power_spectrum, log_spectral_distance, spectral_bias

    from src.mask import get_masked_mean, get_mask_2d, get_mu_sigma, get_mask_center
    from src.target import get_target_slices


    return (
        GUIDANCE_METHODS,
        GUIDANCE_METHOD_HYPERS,
        GUI_REFS,
        LEVELS_DICT,
        MASK_MODES,
        PARTITIONS,
        RolloutConfig,
        VARIABLES_DICT,
        XarrayNormalizer,
        dump_json,
        ensure_rollout_dir,
        get_N_timestamps,
        get_config,
        get_gt_rollout,
        get_guidance_schedule,
        get_level_idx,
        get_mask_2d,
        get_mask_center,
        get_masked_mean,
        get_mu_sigma,
        get_now_timestamp,
        get_rollout,
        get_rollout_ids,
        get_slices,
        get_sweep_dict,
        get_target_slices,
        get_timestamp_from_sliders,
        get_var_idx,
        get_w_star,
        max_day,
        plot_trajectories,
        plot_trajectory,
        sweep_coord_label,
        visualize_map,
    )


@app.function
def get_label(name: str, n_options: int) -> str:
    # sliders with a single option render without a draggable track; pad the
    # label with non-breaking spaces so rows of sliders keep their labels aligned
    return f"{name}:\u00A0\u00A0" if n_options <= 1 else f"{name}: "


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guiding Generative Probabilistic Weather Models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment
    """)
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.run_button(label="refresh")
    return (refresh_button,)


@app.cell
def _(mo, notebook_mode_dropdown, refresh_button):
    mo.hstack([notebook_mode_dropdown, refresh_button], justify="start", align="start")
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
    rollout_id_dropdown = mo.ui.dropdown(
        options=rollout_ids,
        value=rollout_ids[0] if len(rollout_ids)>0 else None,
        label="rollout_id: ",
        allow_select_none=True
    )
    return (rollout_id_dropdown,)


@app.cell
def _(
    M,
    N,
    NUMERIC_AXES,
    RolloutConfig,
    T,
    compute_axis_values,
    config,
    delta_trajectories,
    dump_json,
    ensure_rollout_dir,
    get_now_timestamp,
    gui_ref_select,
    guidance_mode_select,
    level,
    mask_corners,
    mask_mode_select,
    notebook_mode,
    partition,
    rollout_id,
    save_config_button,
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
            # config.json already exists (created in unguided mode) -> preserve it as-is
            dump_json(config.to_dict(), rollout_dir / "config.json")

            _rv = sweep_ranges.value
            sweep = {
                "GUIDANCE_MODE": list(guidance_mode_select.value),
                "GUI_REF": list(gui_ref_select.value),
                "MASK_MODE": list(mask_mode_select.value),
                "GUIDANCE_DELTA": delta_trajectories,
                **{ax: compute_axis_values(ax, _rv) for ax in NUMERIC_AXES},
            }
            if all(sweep[a] for a in ("GUIDANCE_MODE", "GUI_REF", "MASK_MODE")):
                dump_json(sweep, rollout_dir / "sweep_params.json")
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
            _dt_options = {str(v): i for i, v in enumerate(_dt_candidates)}
            delta_trajectory_dropdown = mo.ui.dropdown(
                options=_dt_options,
                value=next(iter(_dt_options)),
                label="delta_trajectory: ",
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
                    _extra[_k] = mo.ui.slider(
                        steps=list(_vals),
                        value=_vals[0],
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
                mo.vstack(_sweep_controls) if _sweep_controls
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
def _(mo, rollout_id_dropdown, save_config_button, sweep_params_widget):
    setup_widget = mo.hstack([    
        rollout_id_dropdown, save_config_button if save_config_button is not None else sweep_params_widget
    ], justify="start", align="start")
    setup_widget
    return


@app.cell
def _(get_now_timestamp, notebook_mode, rollout_id_dropdown):
    # rollout
    # conf params 
    match notebook_mode:
        case "unguided_rollout":
            rollout_id=get_now_timestamp()
        case "guided_rollout":
            rollout_id = rollout_id_dropdown.value
        case "analyze_rollout":
            rollout_id = rollout_id_dropdown.value
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
def config_cell(get_config, notebook_mode, rollout_id):
    # config = the loaded rollout's pinned settings, independent of the sweep selection.
    # Kept in its own cell (not bundled with ZHCJ's sweep-dependent data loads) so changing a
    # sweep axis (R, etc.) doesn't re-run config -> T/M/N -> reset the t/m/n sliders.
    match notebook_mode:
        case "unguided_rollout":
            config = None
        case "guided_rollout" | "analyze_rollout":
            config = get_config(rollout_id)
        case _:
            config = None
    return (config,)


@app.cell
def _(get_rollout, notebook_mode, rollout_id, sweep_params):
    # data objects (config lives in its own cell so sweep changes don't re-run it)
    match notebook_mode:
        case "unguided_rollout":
            unguided_xr=None
            guided_xr=None
            # TODO: set everything to None
        case "guided_rollout":
            unguided_xr = get_rollout("ung", rollout_id)
            guided_xr = None
        case "analyze_rollout":
            unguided_xr = get_rollout("ung", rollout_id).compute()
            guided_xr = get_rollout("gui", rollout_id).sel(sweep_params).compute()
        case _:
            pass
    return guided_xr, unguided_xr


@app.cell
def _(
    config,
    get_corners,
    get_mask_2d,
    get_mu_sigma,
    mask_mode,
    notebook_mode,
):
    # mask
    match notebook_mode:
        case "unguided_rollout":
            mask_corners = get_corners()
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
        case "guided_rollout" | "analyze_rollout":
            mask_corners = config.MASK_CORNERS
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
        case _:
            pass
    return mask, mask_corners


@app.cell
def _(
    delta_trajectories,
    delta_trajectory_dropdown,
    experiment_params,
    notebook_mode,
    np,
):
    # delta schedule
    match notebook_mode:
        case "unguided_rollout":
            delta_trajectory=None
        case "guided_rollout":
            # first authored delta (preview); the full set lives in delta_trajectories
            delta_trajectory = delta_trajectories[0]
        case "analyze_rollout":
            # config.GUIDANCE_DELTA is None under sweeping; the swept vectors live in
            # experiment_params["GUIDANCE_DELTA"], picked by the dropdown's index.
            delta_trajectory = experiment_params["GUIDANCE_DELTA"][delta_trajectory_dropdown.value]
        case _:
            pass

    cumulative_delta_trajectory = (
        np.cumprod(1 + np.asarray(delta_trajectory, dtype=float)) - 1
        if delta_trajectory is not None
        else None
    )
    return cumulative_delta_trajectory, delta_trajectory


@app.cell
def _(
    M,
    N,
    config,
    cumulative_delta_trajectory,
    get_masked_mean,
    get_target_slices,
    guidance_reference,
    level,
    m,
    mask,
    notebook_mode,
    partition,
    rollout_id,
    timestamp,
    var,
):
    # planned guidance (depends on level/var/partition)
    match notebook_mode:
        case "unguided_rollout":
            planned_guidance_rollout=None
            planned_guidance_trajectories=None
            planned_guidance_trajectory=None
        case "guided_rollout" | "analyze_rollout":
            if (
                # GT reference uses the STATE loss (no delta-scaled masked target), so the
                # planned-guidance preview does not apply -- and GT has N+1 steps vs delta's N.
                guidance_reference != "GT"
                and partition == config.PARTITION
                and var == config.VAR
                and level == config.LEVEL
            ):
                planned_guidance_slices = get_target_slices(
                    guidance_reference,
                    rollout_id,
                    N,
                    M,
                    timestamp,
                    partition,
                    var,
                    level,
                    cumulative_delta_trajectory,
                    m,
                )
                planned_guidance_trajectories = get_masked_mean(planned_guidance_slices, mask)
                planned_guidance_trajectory = planned_guidance_trajectories[m]
            else:
                planned_guidance_rollout = None
                planned_guidance_slices = None
                planned_guidance_trajectories = None
                planned_guidance_trajectory = None
        case _:
            pass
    return planned_guidance_trajectories, planned_guidance_trajectory


@app.cell
def _(
    clean_preds_xr,
    config,
    delta_trajectory,
    get_masked_mean,
    get_slices,
    mask,
    notebook_mode,
    np,
    ung_gui_final_xr,
):
    # guidance-target at config coords (pinned — independent of browsing sliders)
    if notebook_mode == "analyze_rollout":
        cfg_clean_preds_slices = get_slices(clean_preds_xr, config.PARTITION, config.VAR, config.LEVEL)
        cfg_ung_gui_M_N_slices = get_slices(ung_gui_final_xr, config.PARTITION, config.VAR, config.LEVEL)
        cfg_ung_gui_M_N_trajectories = get_masked_mean(cfg_ung_gui_M_N_slices, mask)
        cfg_target_guidance_M_N_trajectories = (1 + np.asarray(delta_trajectory)) * cfg_ung_gui_M_N_trajectories
    return cfg_clean_preds_slices, cfg_target_guidance_M_N_trajectories


@app.cell
def _(
    N,
    delta_trajectory,
    get_gt_rollout,
    get_masked_mean,
    get_slices,
    guided_xr,
    level,
    m,
    mask,
    notebook_mode,
    np,
    partition,
    timestamp,
    ung_gui_final_xr,
    unguided_xr,
    var,
):
    # data sub-objects
    match notebook_mode:
        case "unguided_rollout":
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

            ung_gui_m_trajectory = None
            target_guidance_trajectory = None
            target_guidance_M_N_trajectories = None
        case "guided_rollout":
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

            ung_gui_m_trajectory = None
            target_guidance_trajectory = None
            target_guidance_M_N_trajectories = None
        case "analyze_rollout":
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

            ung_gui_M_N_slices = get_slices(ung_gui_final_xr, partition, var, level)
            ung_gui_M_N_trajectories = get_masked_mean(ung_gui_M_N_slices, mask)
            ung_gui_m_trajectory = ung_gui_M_N_trajectories[m]
            print(ung_gui_m_trajectory.shape)

            target_guidance_trajectory = (1 + np.asarray(delta_trajectory)) * ung_gui_m_trajectory
            target_guidance_M_N_trajectories = (1 + np.asarray(delta_trajectory)) * ung_gui_M_N_trajectories
        case _:
            pass

    # gt
    gt_rollout = get_gt_rollout(N+1, timestamp)
    gt_N_slices = get_slices(gt_rollout, partition, var, level)
    gt_trajectory = get_masked_mean(gt_N_slices, mask)
    return (
        gt_N_slices,
        gt_rollout,
        gt_trajectory,
        gui_M_N_slices,
        gui_M_N_trajectories,
        gui_m_trajectory,
        target_guidance_M_N_trajectories,
        target_guidance_trajectory,
        ung_M_N_slices,
        ung_M_N_trajectories,
        ung_gui_m_trajectory,
        ung_m_trajectory,
    )


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
def _(partition_dropdown):
    partition=partition_dropdown.value
    return (partition,)


@app.cell
def _(level_slider):
    level=level_slider.value
    return (level,)


@app.cell
def _(var_dropdown):
    var=var_dropdown.value
    return (var,)


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
def _(PARTITIONS, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(LEVELS_DICT, VARIABLES_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition]
    level_label = get_label("level", len(LEVELS))
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[-1], label=level_label, show_value=True, debounce=True)
    VARIABLES = VARIABLES_DICT[partition]
    if partition == "surface":
        DEFAULT_VAR_VALUE = VARIABLES[2]
    else:
        DEFAULT_VAR_VALUE = VARIABLES[3]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=DEFAULT_VAR_VALUE, label="var: ")
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
    # number of delta trajectories to author (each a sweep value of GUIDANCE_DELTA)
    n_deltas_slider = mo.ui.slider(1, 6, step=1, value=1, label="delta trajectories: ", show_value=True)
    # how each delta is shaped: hand-drawn linear ramp, or relative to the unguided ensemble
    # band  delta_n = k * sigma_band_n / |mean_n|,  sigma_band_n = (max_n - min_n)/2  (k per candidate;
    # k=1 pushes the masked mean to the top edge of the unguided ensemble band).
    delta_mode_dropdown = mo.ui.dropdown(["linear", "band-based"], value="linear", label="delta mode: ")
    return delta_mode_dropdown, n_deltas_slider


@app.cell
def _(N, delta_mode_dropdown, mo, n_deltas_slider, notebook_mode):
    # per-delta params. linear: start% / peak% / start@n / stop@n. band-based: one multiplier
    # k per candidate (delta_n = k * sigma_band_n/|mean_n|, sigma_band_n = (max-min)/2 of the ensemble).
    if notebook_mode == "guided_rollout":
        _dc = {}
        for _i in range(n_deltas_slider.value):
            if delta_mode_dropdown.value == "band-based":
                _dc[f"{_i}.k"] = mo.ui.number(value=1.0, label="k (× band): ")
            else:
                _dc[f"{_i}.start"]    = mo.ui.number(value=0.0, label="start %: ")
                _dc[f"{_i}.peak"]     = mo.ui.number(value=5.0, label="peak %: ")
                _dc[f"{_i}.start_at"] = mo.ui.slider(0, N, step=1, value=0, label="start@n: ", show_value=True)
                _dc[f"{_i}.stop_at"]  = mo.ui.slider(1, N, step=1, value=N, label=get_label("stop@n", N), show_value=True)
        delta_controls = mo.ui.dictionary(_dc)
    else:
        delta_controls = mo.ui.dictionary({})
    return (delta_controls,)


@app.cell(hide_code=True)
def delta_std_base(
    config,
    get_masked_mean,
    get_slices,
    mask,
    notebook_mode,
    np,
    unguided_xr,
):
    # unguided ensemble spread at the guidance coords (config var/level + authoring mask):
    # sigma_band_n = (max_n - min_n)/2 over the M members' masked-average; rel_band_n = sigma_band_n/|mean_n|
    # is the band-based delta base (k=1 -> target reaches the top edge of the unguided band).
    # NOTE: MASK_MODE is a separate swept axis but one delta vector is shared across mask modes;
    # the relative ratio is fairly mask-mode robust, so we use the authoring mask. Needs M > 1.
    if notebook_mode == "guided_rollout":
        _base = get_masked_mean(
            get_slices(unguided_xr, config.PARTITION, config.VAR, config.LEVEL), mask)  # (M, N)
        delta_band_n = (_base.max(axis=0) - _base.min(axis=0)) / 2.0            # (N,) half band-width
        delta_mean_n = _base.mean(axis=0)                                       # (N,)
        delta_rel_band_n = delta_band_n / np.maximum(np.abs(delta_mean_n), 1e-8)  # (N,); guards mean~0
    else:
        delta_rel_band_n = None
    return (delta_rel_band_n,)


@app.cell
def _(
    N,
    delta_controls,
    delta_mode_dropdown,
    delta_rel_band_n,
    mo,
    n_deltas_slider,
    notebook_mode,
):
    # delta trajectories. linear: 0 before start@n, ramp start%->peak% over [start@n, stop@n],
    # 0 after. band-based: delta_n = k * rel_band_n (k per candidate). Both return length-N lists
    # indexed by rollout step n (aligned with delta_trajectory[n] in rollout.py).
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
        if delta_mode_dropdown.value == "band-based":
            delta_trajectories = [
                [float(_dv[f"{_i}.k"] * delta_rel_band_n[_n]) for _n in range(N)]
                for _i in range(n_deltas_slider.value)
            ]
            _rows = [
                mo.hstack([mo.md(f"delta {_i}: "), delta_controls[f"{_i}.k"]], justify="start", align="center")
                for _i in range(n_deltas_slider.value)
            ]
        else:
            delta_trajectories = [
                _linear_delta(N, _dv[f"{_i}.start"], _dv[f"{_i}.peak"],
                              int(_dv[f"{_i}.start_at"]), int(_dv[f"{_i}.stop_at"]))
                for _i in range(n_deltas_slider.value)
            ]
            _rows = [
                mo.hstack([mo.md(f"delta {_i}: "), delta_controls[f"{_i}.start"], delta_controls[f"{_i}.peak"],
                           delta_controls[f"{_i}.start_at"], delta_controls[f"{_i}.stop_at"]],
                          justify="start", align="center")
                for _i in range(n_deltas_slider.value)
            ]
        # controls only; the trajectories are drawn on the rollout-trajectories chart's right axis
        delta_widget = mo.vstack([mo.hstack([n_deltas_slider, delta_mode_dropdown]), *_rows], align="start")
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
    return (mask_mode,)


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
    compute_axis_values,
    gui_ref_select,
    guidance_mode_select,
    mask_mode_select,
    mo,
    notebook_mode,
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

    hypers_widget = mo.vstack([
        mo.md("## Sweep"),
        mo.md("**Common categorical**:"),
        mo.hstack([guidance_mode_select, gui_ref_select, mask_mode_select], justify="start"),
        mo.md(f"**Specific**:"),
        *([_sweep_row(ax) for ax in _mode_num] if _mode_num else [mo.md("_none_")]),
    ], align="start")

    hypers_widget if notebook_mode == "guided_rollout" else None
    return


@app.cell
def wa_schedule(
    T,
    compute_axis_values,
    experiment_params,
    get_w_star,
    guidance_mode_dropdown,
    guidance_mode_select,
    guidance_schedules,
    lambda_t_by_method,
    m,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    rollout_id,
    sweep_params,
    sweep_ranges,
):
    # Guidance weight schedule over the flow, beside the rollout trajectories plot.
    # Separates a_t and the closed gap 1 - (1-eta)^t (right axis) from w_t and lambda_t (left).
    # a_t = (1 - eta)^(t+1) is the shared remaining-gap schedule of FGWNOLR and FGWNOGAP
    # (both depend on eta); 1 - (1-eta)^t is the fraction already closed ENTERING step t
    # (0 at t=0), matching the chart's 0-based t axis.
    # guided_rollout: preview at the swept w_init / eta. analyze_rollout: recorded schedules.
    if notebook_mode == "guided_rollout":
        selected_modes = list(guidance_mode_select.value)
        fgwnolr_w_choices = compute_axis_values("fgwnolr_w_init", sweep_ranges.value)
        eta_choices = compute_axis_values("eta", sweep_ranges.value)
        w_from_star = False
    elif notebook_mode == "analyze_rollout":
        selected_modes = [guidance_mode_dropdown.value]
        _sweep_sel = dict(sweep_params)  # records store the coord-label sweep form
        _wstar_recs = get_w_star(rollout_id, _sweep_sel)
        w_from_star = bool(_wstar_recs)
        _wstar = sorted({r["w_star"] for r in _wstar_recs})
        fgwnolr_w_choices = _wstar or list(experiment_params.get("fgwnolr_w_init") or [])
        eta_choices = list(experiment_params.get("eta") or [])
    else:
        selected_modes = []
        fgwnolr_w_choices = []
        eta_choices = []
    eta_modes = [mode for mode in selected_modes if mode in ("FGWNOLR", "FGWNOGAP")]
    _has_recorded = notebook_mode == "analyze_rollout" and bool(lambda_t_by_method)
    _AT_COLOR, _CLOSED_COLOR = "#7B2CBF", "#E67E22"  # a_t purple, closed-gap 1 - a_t orange
    if eta_modes or _has_recorded:
        _left_label = r"$\lambda_t,\ w_t$"
        if _has_recorded:
            # schedules actually applied by the run: lambda_t and w_t on the left axis,
            # a_t and the closed-gap 1 - a_t on the right axis
            _sched, _right = {}, {}
            for _meth, _sd in guidance_schedules.items():
                _sched[rf"$\lambda_t$ — {_meth}"] = list(_sd["lambda_t"])
                _sched[rf"$w_t$ — {_meth}"] = list(_sd["w_t"])
                _a = np.asarray(_sd["a_t"], dtype=float)
                _right[rf"$a_t$ — {_meth}"] = _a.tolist()
                # 1 - (1-eta)^t = 1 - a_{t-1} shifted right one step, 0 at t=0
                _right[rf"$1-(1-\eta)^t$ — {_meth}"] = np.concatenate([[0.0], (1.0 - _a)[:-1]]).tolist()
        else:
            # preview: a_t and 1 - a_t on the RIGHT (shared by both methods). NOLR also
            # previews w_t (constant w_init) and lambda_t = w_t * a_t on the LEFT; NOGAP's
            # w_t / lambda_t are solved per step at runtime, so only a_t / 1 - a_t are shown.
            _sched, _right = {}, {}
            _steps = np.arange(1, T + 1)
            for _eta in (eta_choices or [0.5]):
                _a = (1.0 - _eta) ** _steps
                _right[rf"$a_t$ ($\eta$={_eta:g})"] = _a.tolist()
                _right[rf"$1-(1-\eta)^t$ ($\eta$={_eta:g})"] = (1.0 - (1.0 - _eta) ** (_steps - 1)).tolist()
                if "FGWNOLR" in eta_modes:
                    for _w in (fgwnolr_w_choices or [250.0]):
                        _sched[rf"$w_t$ ($w$={_w:g})"] = [float(_w)] * T
                        _sched[rf"$\lambda_t$ ($w$={_w:g}, $\eta$={_eta:g})"] = (_w * _a).tolist()
            if not _sched:
                # NOGAP-only: no w/lambda -> put a_t and 1 - a_t on the left
                _sched, _right, _left_label = _right, None, r"$a_t,\ 1-(1-\eta)^t$"
        _gap_color = lambda _k: _CLOSED_COLOR if _k.startswith(r"$1-(") else _AT_COLOR
        _color_map = {_k: _gap_color(_k) for _k in _sched} if _right is None else None
        _right_color = {_k: _gap_color(_k) for _k in _right} if _right is not None else "#7B2CBF"
        wa_schedule_widget = plot_trajectory(
            _sched,
            right_trajectory=_right,
            right_label=r"$a_t$",
            right_color=_right_color,
            color_map=_color_map,
            var=_left_label,
            title=r"Guidance weight schedule  ($\lambda_t = w_t\,a_t$, $a_t=(1-\eta)^{t+1}$)",
            subtitle=(rf"recorded from run (m={m}, n={n})" if _has_recorded
                      else "preview — " + ", ".join(eta_modes)
                           + (" (NOLR w shown at w_init)" if "FGWNOLR" in eta_modes else "")),
            xlabel="$t$",
            figsize=(10, 6),
        )
    else:
        wa_schedule_widget = None
    return (wa_schedule_widget,)


@app.cell
def _(
    M_N_widget,
    T_slider,
    delta_widget,
    guidance_convergence_t_plot,
    guidance_reference_dropdown,
    inspect_states_widget_make,
    level_slider,
    m_n_widget,
    mask_map,
    mask_mode_dropdown,
    mo,
    notebook_mode,
    partition_dropdown,
    sweep_params_widget,
    traj_checks,
    trajectories_plot,
    var_dropdown,
    wa_schedule_widget,
    weather_map,
    zoom_slider,
):
    mask_widget_controls = mo.hstack(
        [partition_dropdown, var_dropdown, level_slider],
        justify="start",
        align="start",
    )

    mask_widget_maps = mo.vstack([
        mo.hstack([zoom_slider], justify="start"),
        mo.hstack(
            [weather_map, mask_map],
            justify="start",
            align="start",
        ),
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
                trajectories_plot
            ])
            inspect_states_widget=None
        case "guided_rollout":
            trajectory_widget=mo.vstack([
                guidance_reference_dropdown,
                m_n_widget,
                mo.hstack(
                    [partition_dropdown, var_dropdown, level_slider],
                    justify="start",
                ),
                delta_widget,
                mo.hstack(
                    [mo.vstack(list(traj_checks.values())), trajectories_plot]
                    + ([wa_schedule_widget] if wa_schedule_widget is not None else []),
                    justify="start", align="start",
                )
            ], align="start")
            mask_widget = mask_widget_maps
            inspect_states_widget=inspect_states_widget_make
        case "analyze_rollout":
            trajectory_widget=mo.vstack([
                sweep_params_widget,
                m_n_widget,
                mo.hstack(
                    [partition_dropdown, var_dropdown, level_slider],
                    justify="start",
                ),
                mo.hstack(
                    [mo.vstack(list(traj_checks.values())), trajectories_plot]
                    + ([wa_schedule_widget] if wa_schedule_widget is not None else []),
                    justify="start", align="start",
                ),
                guidance_convergence_t_plot,
            ], align="start")
            mask_widget = mask_widget_maps
            inspect_states_widget=inspect_states_widget_make
        case _:
            pass
    return inspect_states_widget, mask_widget, trajectory_widget


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mask
    """)
    return


@app.cell
def _(mask_widget):
    mask_widget
    return


@app.cell
def _(mo):
    get_corners, set_corners = mo.state((-10.0, 2.0, 35.0, 45.0))
    return get_corners, set_corners


@app.cell
def _(
    config,
    dpi_slider,
    get_slices,
    gt_rollout,
    level,
    map_interactive,
    mask_corners,
    n,
    notebook_mode,
    np,
    partition,
    set_corners,
    timestamps,
    var,
    visualize_map,
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

    weather_slices = get_slices(gt_rollout, mask_partition, mask_var, mask_level)
    weather_map = visualize_map(
        weather_slices[n],
        suptitle=f"{timestamps[n]}",
        title=f"partition={mask_partition} | var={mask_var} | level={mask_level}",
        interactive=map_interactive,
        vmin=np.min(weather_slices),
        vmax=np.max(weather_slices),
        center=np.mean(weather_slices),
        # mask_corners=mask_corners, # TODO: simplify to this
        rectangle_x=(mask_corners[0], mask_corners[1]),
        rectangle_y=(mask_corners[2], mask_corners[3]),
        zoom=zoom_slider.value,
        zoom_center_lon=zoom_centers[0],
        zoom_center_lat=zoom_centers[1],
        figsize=(14, 8),
        dpi=dpi_slider.value,
    )
    if map_interactive:
        weather_map.widget.observe(
            lambda _c: set_corners(
                (*sorted(weather_map.widget.x), *sorted(weather_map.widget.y))
            ),
            names=["x", "y"],
        )
    return mask_level, mask_partition, mask_var, weather_map


@app.cell
def _(
    dpi_slider,
    mask,
    mask_corners,
    mask_level,
    mask_partition,
    mask_var,
    np,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    mask_map = visualize_map(
        mask,
        suptitle="mask",
        title=f"partition={mask_partition} | var={mask_var} | level={mask_level}",
        interactive=False,
        vmin=np.min(mask) if np.min(mask) < np.max(mask) else -0.001,
        vmax=np.max(mask) if np.min(mask) < np.max(mask) else 0.001,
        center=np.mean(mask),
        rectangle_x=(mask_corners[0], mask_corners[1]),
        rectangle_y=(mask_corners[2], mask_corners[3]),
        zoom=zoom_slider.value,
        zoom_center_lon=zoom_centers[0],
        zoom_center_lat=zoom_centers[1],
        figsize=(14, 8),
        dpi=dpi_slider.value
    )
    return (mask_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect states
    """)
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
    dpi_slider = mo.ui.slider(start=50, stop=500, step=50, value=100, debounce=True, show_value=True, label="dpi: ")
    return (dpi_slider,)


@app.cell
def _(mo):
    norm_modes = ["own_scale", "own_mask_scale", "same_scale"]
    norm_mode_dropdown = mo.ui.dropdown(
        norm_modes,
        value=norm_modes[0],
        label="norm mode: ",
    )
    return (norm_mode_dropdown,)


@app.cell
def _(get_mask_center, mask_corners):
    zoom_centers = get_mask_center(*mask_corners)
    return (zoom_centers,)


@app.cell
def _(
    get_slices,
    gt_N_slices,
    gui_M_N_slices,
    level,
    m,
    n,
    notebook_mode,
    partition,
    ung_M_N_slices,
    ung_gui_final_xr,
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

        ung_onl_slice = get_slices(ung_gui_final_xr, partition, var, level)
        ung_onl_curr = ung_onl_slice[m][n]
        ung_onl_prev = ung_onl_slice[m][n-1] if n>0 else ung_onl_slice[m][n]

        gt_gt = gt_curr - gt_prev
        gui_gui = gui_curr - gui_prev
        gui_ung_gui = gui_curr - ung_onl_curr
        gui_gt = gui_curr - gt_curr
        ung_gui_gt = ung_onl_curr - gt_curr
        ung_gui_ung_gui = ung_onl_curr - ung_onl_prev
    return (
        gt_curr,
        gt_gt,
        gt_prev,
        gt_ung,
        gui_curr,
        gui_gt,
        gui_gui,
        gui_ung_gui,
        ung_curr,
        ung_onl_curr,
        ung_prev,
    )


@app.cell
def _(np):
    def safe_abs_limits(arrays):
        vmin = min(float(np.nanmin(np.asarray(arr))) for arr in arrays)
        vmax = max(float(np.nanmax(np.asarray(arr))) for arr in arrays)

        if vmax <= vmin:
            vmax = vmin + 1e-9

        center = 0.5 * (vmin + vmax)
        center = min(max(center, vmin + 1e-9), vmax - 1e-9)

        return vmin, vmax, center

    return (safe_abs_limits,)


@app.cell
def _(
    analysis_type_dropdown,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    contour_lw_slider,
    cv2,
    dpi_slider,
    gt_curr,
    gt_gt,
    gt_prev,
    gt_ung,
    gui_curr,
    gui_gt,
    gui_gui,
    gui_ung_gui,
    map_interactive,
    mask,
    mo,
    norm_mode_dropdown,
    notebook_mode,
    np,
    safe_abs_limits,
    show_mask_switch,
    ung_curr,
    ung_onl_curr,
    ung_prev,
    visualize_map,
    white_zero_checkbox,
    white_zero_cmap,
    white_zero_slider,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode =="guided_rollout":
        match analysis_type_dropdown.value:
            case "absolute":
                absolute_panels = [
                    ("$x_{n}^{\\text{gt}}$", gt_curr),
                    ("$x_{n-1}^{\\text{gt}}$", gt_prev),
                    ("$x_{n}^{ung}$", ung_curr),
                    ("$x_{n-1}^{ung}$", ung_prev),
                ]

                abs_vmin, abs_vmax, abs_center = safe_abs_limits(
                    [arr for _, arr in absolute_panels]
                )

                absolute_maps = {}

                for label, arr in absolute_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min, _v_max, _v_center = safe_abs_limits([arr])
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # own limits, restricted to inside the mask (nonzero weights)
                        _v_min, _v_max, _v_center = safe_abs_limits([np.where(np.asarray(mask) > 0, arr, np.nan)])
                    else:
                        _v_min, _v_max, _v_center = abs_vmin, abs_vmax, abs_center
                    absolute_maps[label] = visualize_map(
                        arr,
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=contour_lw_slider.value,
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

                curr_map = absolute_maps["$x_{n}^{\\text{gt}}$"]
                prev_map = absolute_maps["$x_{n-1}^{\\text{gt}}$"]
                ung_map = absolute_maps["$x_{n}^{ung}$"]
                ung_prev_map = absolute_maps["$x_{n-1}^{ung}$"]
            case "difference":
                difference_panels = [
                    ("$x_{n}^{\\text{gt}} - x_{n-1}^{\\text{gt}}$", gt_gt),
                    ("$x_{n}^{\\text{gt}} - x_{n}^{\\text{ung}}$", gt_ung),
                ]

                diff_vmin = min(float(np.nanmin(arr)) for _, arr in difference_panels)
                diff_vmax = max(float(np.nanmax(arr)) for _, arr in difference_panels)

                difference_maps = {}

                for label, arr in difference_panels:
                    # is_guided_unguided = label == "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"

                    if norm_mode_dropdown.value == "own_scale":
                        v_min = min(float(np.nanmin(arr)), -1e-12)
                        v_max = max(float(np.nanmax(arr)), 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        _arr_in = np.where(np.asarray(mask) > 0, arr, np.nan)
                        v_min = min(float(np.nanmin(_arr_in)), -1e-12)
                        v_max = max(float(np.nanmax(_arr_in)), 1e-12)
                    else:
                        v_min, v_max = diff_vmin, diff_vmax

                    _arr_wz = (
                        np.where(np.abs(arr) <= white_zero_slider.value / 100.0 * float(np.nanmax(np.abs(arr))), np.nan, arr)
                        if white_zero_checkbox.value else arr
                    )
                    difference_maps[label] = visualize_map(
                        _arr_wz,
                        cmap=white_zero_cmap if white_zero_checkbox.value else "coolwarm",
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=contour_lw_slider.value,
                        mask_2d=mask,
                        title=label,
                        vmin=v_min,
                        vmax=v_max,
                        center=0.0,
                        show_mask=show_mask_switch.value,
                        zoom=zoom_slider.value,
                        zoom_center_lon=zoom_centers[0],
                        zoom_center_lat=zoom_centers[1],
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )

                gt_gt_map = difference_maps["$x_{n}^{\\text{gt}} - x_{n-1}^{\\text{gt}}$"]
                gt_ung_map = difference_maps["$x_{n}^{\\text{gt}} - x_{n}^{\\text{ung}}$"]
            case "sobel_grads":
                sobel_grad_widget = None
            case _:
                pass

    if notebook_mode =="analyze_rollout":
        match analysis_type_dropdown.value:
            case "absolute":
                absolute_panels = [
                    ("$x_{n}^{\\text{gt}}$", gt_curr),
                    ("$x_{n}^{\\text{ung_gui}}$", ung_onl_curr),
                    ("$x_{n}^{gui}$", gui_curr),
                    ("$x_{n}^{ung}$", ung_curr),
                ]

                abs_vmin, abs_vmax, abs_center = safe_abs_limits(
                    [arr for _, arr in absolute_panels]
                )

                absolute_maps = {}

                for label, arr in absolute_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min, _v_max, _v_center = safe_abs_limits([arr])
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        # own limits, restricted to inside the mask (nonzero weights)
                        _v_min, _v_max, _v_center = safe_abs_limits([np.where(np.asarray(mask) > 0, arr, np.nan)])
                    else:
                        _v_min, _v_max, _v_center = abs_vmin, abs_vmax, abs_center
                    absolute_maps[label] = visualize_map(
                        arr,
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=contour_lw_slider.value,
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

                curr_map = absolute_maps["$x_{n}^{\\text{gt}}$"]
                prev_map = absolute_maps["$x_{n}^{\\text{ung_gui}}$"]
                ung_map = absolute_maps["$x_{n}^{ung}$"]
                gui_map = absolute_maps["$x_{n}^{gui}$"]

            case "difference":
                difference_panels = [
                    ("$x_{n}^{\\text{gt}} - x_{n-1}^{\\text{gt}}$", gt_gt),
                    ("$x_{n}^{gui} - x_{n}^{\\text{gt}}$", gui_gt),
                    ("$x_{n}^{gui} - x_{n}^{\\text{ung_gui}}$", gui_ung_gui),
                    ("$x_{n}^{gui} - x_{n-1}^{gui}$", gui_gui),
                ]

                diff_vmin = min(float(np.nanmin(arr)) for _, arr in difference_panels)
                diff_vmax = max(float(np.nanmax(arr)) for _, arr in difference_panels)

                difference_maps = {}

                for label, arr in difference_panels:
                    # is_guided_unguided = label == "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"

                    if norm_mode_dropdown.value == "own_scale":
                        v_min = min(float(np.nanmin(arr)), -1e-12)
                        v_max = max(float(np.nanmax(arr)), 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        _arr_in = np.where(np.asarray(mask) > 0, arr, np.nan)
                        v_min = min(float(np.nanmin(_arr_in)), -1e-12)
                        v_max = max(float(np.nanmax(_arr_in)), 1e-12)
                    else:
                        v_min, v_max = diff_vmin, diff_vmax

                    _arr_wz = (
                        np.where(np.abs(arr) <= white_zero_slider.value / 100.0 * float(np.nanmax(np.abs(arr))), np.nan, arr)
                        if white_zero_checkbox.value else arr
                    )
                    difference_maps[label] = visualize_map(
                        _arr_wz,
                        cmap=white_zero_cmap if white_zero_checkbox.value else "coolwarm",
                        contour_2d=arr if contour_checkbox.value else None,
                        contour_levels=contour_levels_slider.value,
                        contour_color=contour_color_dropdown.value,
                        contour_linewidth=contour_lw_slider.value,
                        mask_2d=mask,
                        title=label,
                        vmin=v_min,
                        vmax=v_max,
                        center=0.0,
                        show_mask=show_mask_switch.value,
                        zoom=zoom_slider.value,
                        zoom_center_lon=zoom_centers[0],
                        zoom_center_lat=zoom_centers[1],
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )

                gt_gt_map = difference_maps["$x_{n}^{\\text{gt}} - x_{n-1}^{\\text{gt}}$"]
                gui_gui_map = difference_maps["$x_{n}^{gui} - x_{n-1}^{gui}$"]
                gui_ung_map = difference_maps["$x_{n}^{gui} - x_{n}^{\\text{ung_gui}}$"]
                gui_gt_map = difference_maps["$x_{n}^{gui} - x_{n}^{\\text{gt}}$"]
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

                gradmap_gui_ung_x = cv2.Sobel(ung_onl_curr, cv2.CV_32F, 1, 0, ksize=3)
                gradmap_gui_ung_y = cv2.Sobel(ung_onl_curr, cv2.CV_32F, 0, 1, ksize=3)
                gradmap_gui_ung_mag = np.sqrt(gradmap_gui_ung_x**2 + gradmap_gui_ung_y**2)

                gradmap_mag_panels = [
                    (r"$\|\nabla x_n^{\text{gt}}\|$", gradmap_gt_mag),
                    (r"$\|\nabla x_n^{\text{ung}}\|$", gradmap_ung_mag),
                    (r"$\|\nabla x_n^{\text{gui}}\|$", gradmap_gui_mag),
                    (r"$\|\nabla (x_n^{\text{ung_gui}})\|$", gradmap_gui_ung_mag),
                ]

                gradmap_mag_vmin = min(float(np.nanmin(gradmap_arr)) for _, gradmap_arr in gradmap_mag_panels)
                gradmap_mag_vmax = max(float(np.nanmax(gradmap_arr)) for _, gradmap_arr in gradmap_mag_panels)

                gradmap_figures = []

                for gradmap_title, gradmap_arr in gradmap_mag_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min = float(np.nanmin(gradmap_arr))
                        _v_max = max(float(np.nanmax(gradmap_arr)), _v_min + 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        _arr_in = np.where(np.asarray(mask) > 0, gradmap_arr, np.nan)
                        _v_min = float(np.nanmin(_arr_in))
                        _v_max = max(float(np.nanmax(_arr_in)), _v_min + 1e-12)
                    else:
                        _v_min, _v_max = gradmap_mag_vmin, gradmap_mag_vmax
                    print(_v_min, _v_max)
                    gradmap_figures.append(
                        visualize_map(
                            gradmap_arr,
                            contour_2d=gradmap_arr if contour_checkbox.value else None,
                            contour_levels=contour_levels_slider.value,
                            contour_color=contour_color_dropdown.value,
                            contour_linewidth=contour_lw_slider.value,
                            mask_2d=mask,
                            title=gradmap_title,
                            vmin=-1 if _v_min == _v_max else _v_min,
                            vmax=1 if _v_min == _v_max else _v_max,
                            center=0.0 if (_v_max!=0.0 and _v_min!=0.0) else np.mean([_v_min,_v_max]),
                            show_mask=show_mask_switch.value,
                            zoom=zoom_slider.value,
                            zoom_center_lon=zoom_centers[0],
                            zoom_center_lat=zoom_centers[1],
                            dpi=dpi_slider.value,
                            figsize=(14, 8),
                        )
                    )

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

                sobel_ung_gui_x = cv2.Sobel(ung_onl_curr, cv2.CV_32F, 1, 0, ksize=3)
                sobel_ung_gui_y = cv2.Sobel(ung_onl_curr, cv2.CV_32F, 0, 1, ksize=3)
                sobel_ung_gui_mag = np.sqrt(sobel_ung_gui_x**2 + sobel_ung_gui_y**2)

                sobel_diff_panels = [
                    (r"$\|\nabla x_n^{\text{gt}}\| - \|\nabla x_n^{\text{ung}}\|$", sobel_gt_mag - sobel_ung_mag),
                    (r"$\|\nabla x_n^{\text{gt}}\| - \|\nabla x_n^{\text{gui}}\|$", sobel_gt_mag - sobel_gui_mag),
                    (r"$\|\nabla x_n^{\text{gt}}\| - \|\nabla (x_n^{\text{ung_gui}})\|$", sobel_gt_mag - sobel_ung_gui_mag),
                    (r"$\|\nabla x_n^{\text{gui}}\| - \|\nabla (x_n^{\text{ung_gui}})\|$", sobel_gui_mag - sobel_ung_gui_mag),
                ]

                sobel_diff_vmin = min(float(np.nanmin(arr)) for _, arr in sobel_diff_panels)
                sobel_diff_vmax = max(float(np.nanmax(arr)) for _, arr in sobel_diff_panels)

                sobel_diff_figures = []

                for sobel_diff_title, sobel_diff_arr in sobel_diff_panels:
                    if norm_mode_dropdown.value == "own_scale":
                        _v_min = min(float(np.nanmin(sobel_diff_arr)), -1e-12)
                        _v_max = max(float(np.nanmax(sobel_diff_arr)), 1e-12)
                    elif norm_mode_dropdown.value == "own_mask_scale":
                        _arr_in = np.where(np.asarray(mask) > 0, sobel_diff_arr, np.nan)
                        _v_min = min(float(np.nanmin(_arr_in)), -1e-12)
                        _v_max = max(float(np.nanmax(_arr_in)), 1e-12)
                    else:
                        _v_min, _v_max = min(sobel_diff_vmin, -1e-12), max(sobel_diff_vmax, 1e-12)
                    sobel_diff_figures.append(
                        visualize_map(
                            sobel_diff_arr,
                            contour_2d=sobel_diff_arr if contour_checkbox.value else None,
                            contour_levels=contour_levels_slider.value,
                            contour_color=contour_color_dropdown.value,
                            contour_linewidth=contour_lw_slider.value,
                            mask_2d=mask,
                            title=sobel_diff_title,
                            vmin=-1 if _v_min == _v_max else _v_min,
                            vmax=1 if _v_min == _v_max else _v_max,
                            center=0.0 if (_v_max!=0.0 and _v_min!=0.0) else np.mean([_v_min,_v_max]),
                            show_mask=show_mask_switch.value,
                            zoom=zoom_slider.value,
                            zoom_center_lon=zoom_centers[0],
                            zoom_center_lat=zoom_centers[1],
                            dpi=dpi_slider.value,
                            figsize=(14, 8),
                        )
                    )

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
        gui_gt_map,
        gui_gui_map,
        gui_map,
        gui_ung_map,
        prev_map,
        sobel_diffs_widget,
        sobel_grad_widget,
        ung_map,
        ung_prev_map,
    )


@app.cell
def _(
    analysis_type_dropdown,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    contour_lw_slider,
    curr_map,
    dpi_slider,
    gt_gt_map,
    gt_ung_map,
    gui_gt_map,
    gui_gui_map,
    gui_map,
    gui_ung_map,
    inspect_checks,
    level_slider,
    m_slider,
    mo,
    n_slider,
    norm_mode_dropdown,
    notebook_mode,
    partition_dropdown,
    prev_map,
    show_mask_switch,
    sobel_diffs_widget,
    sobel_grad_widget,
    sweep_params_widget,
    ung_map,
    ung_prev_map,
    var_dropdown,
    white_zero_checkbox,
    white_zero_slider,
    zoom_slider,
):
    if notebook_mode == "guided_rollout":
        common_controls = [
            mo.hstack([analysis_type_dropdown, dpi_slider], justify="start"),
            mo.hstack([m_slider, n_slider], justify="start"),
            mo.hstack(
                [partition_dropdown, var_dropdown, level_slider],
                justify="start",
            ),
        ]

        match analysis_type_dropdown.value:
            case "absolute":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider], justify="start", align="center"),
                        mo.hstack([curr_map, prev_map], justify="start"),
                        mo.hstack([ung_map, ung_prev_map], justify="start"),
                    ],
                    justify="start",
                )

            case "difference":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider], justify="start", align="center"),
                        mo.hstack([gt_gt_map, gt_ung_map], justify="start")
                    ], justify="start",
                )
            case _:
                # sobel_grads (and any other type) has no guided-mode panel; show a
                # placeholder so inspect_states_widget_make is always defined here.
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        # mo.md(f"_'{analysis_type_dropdown.value}' analysis is not available in guided_rollout mode._"),
                    ],
                    justify="start",
                )

    if notebook_mode == "analyze_rollout":
        common_controls = [
            mo.hstack([analysis_type_dropdown, dpi_slider], justify="start"),
            mo.hstack([n_slider, m_slider], justify="start"),
            mo.hstack(
                [partition_dropdown, var_dropdown, level_slider],
                justify="start",
            ),
        ]

        match analysis_type_dropdown.value:
            case "absolute":
                _rows = [
                    ("curr / prev", [curr_map, prev_map]),
                    ("gui / ung", [gui_map, ung_map]),
                ]
                inspect_states_widget_make = mo.vstack(
                    [
                        sweep_params_widget,
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider], justify="start", align="center"),
                        mo.hstack([
                            mo.vstack([inspect_checks[_k] for _k, _ in _rows], justify="start", align="start").style(width="fit-content"),
                            mo.vstack(
                                [mo.hstack(_maps, justify="start") for _k, _maps in _rows if inspect_checks[_k].value],
                                justify="start", align="start",
                            ).style(width="fit-content"),
                        ], justify="start", align="start"),
                    ],
                    justify="start",
                )

            case "difference":
                _rows = [
                    ("gt_gt / gui_gui", [gt_gt_map, gui_gui_map]),
                    ("gui_gt / gui_ung", [gui_gt_map, gui_ung_map]),
                ]
                inspect_states_widget_make = mo.vstack(
                    [
                        sweep_params_widget,
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider], justify="start", align="center"),
                        mo.hstack([
                            mo.vstack([inspect_checks[_k] for _k, _ in _rows], justify="start", align="start").style(width="fit-content"),
                            mo.vstack(
                                [mo.hstack(_maps, justify="start") for _k, _maps in _rows if inspect_checks[_k].value],
                                justify="start", align="start",
                            ).style(width="fit-content"),
                        ], justify="start", align="start"),
                    ], justify="start",
                )

            case "sobel_grads":
                inspect_states_widget_make = mo.vstack(
                    [
                        sweep_params_widget,
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider], justify="start", align="center"),
                        sobel_grad_widget
                    ], justify="start",
                )
            case "sobel_diffs":
                inspect_states_widget_make = mo.vstack(
                    [
                        sweep_params_widget,
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown], justify="start", align="center"),
                        mo.hstack([contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider], justify="start", align="center"),
                        sobel_diffs_widget
                    ], justify="start",
                )
            case _:
                pass
    return (inspect_states_widget_make,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trajectories
    """)
    return


@app.cell
def _(timestamp_widget):
    timestamp_widget
    return


@app.cell
def _(trajectory_widget):
    trajectory_widget
    return


@app.cell
def _(mo):
    # checkboxes compact
    traj_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "unguided", "guided", "unguided_guided", "planned_guidance", "target_guidance", "delta_trajectory", "dist_bands"
    )})
    return (traj_checks,)


@app.cell
def _():
    # traj_checks
    return


@app.cell
def _(
    config,
    cumulative_delta_trajectory,
    delta_trajectories,
    delta_trajectory,
    dpi_slider,
    gt_trajectory,
    gui_M_N_trajectories,
    gui_m_trajectory,
    m,
    n,
    notebook_mode,
    planned_guidance_trajectories,
    planned_guidance_trajectory,
    plot_trajectories,
    target_guidance_M_N_trajectories,
    target_guidance_trajectory,
    timestamps,
    traj_checks,
    ung_M_N_trajectories,
    ung_gui_m_trajectory,
    ung_m_trajectory,
    var,
):

    var_check = (var==config.VAR if notebook_mode in ("guided_rollout", "analyze_rollout") else False)

    trajectories_plot = plot_trajectories(
        timestamps=timestamps,
        var=var,
        m=m,
        n=n+1,
        guided_member=gui_m_trajectory if traj_checks["guided"].value else None,
        unguided_member=ung_m_trajectory if traj_checks["unguided"].value else None,
        unguided_guided_member=ung_gui_m_trajectory if traj_checks["unguided_guided"].value else None,
        guided_ensemble=gui_M_N_trajectories if traj_checks["dist_bands"].value else None,
        unguided_ensemble=ung_M_N_trajectories if traj_checks["dist_bands"].value else None,
        target_ensemble=planned_guidance_trajectories if (traj_checks["planned_guidance"].value and traj_checks["dist_bands"].value and var_check) else None,
        target_guidance_ensemble=target_guidance_M_N_trajectories if (traj_checks["dist_bands"].value and traj_checks["target_guidance"].value and var_check) else None,
        target_trajectory=planned_guidance_trajectory if (traj_checks["planned_guidance"].value and var_check) else None,
        target_guidance_trajectory=target_guidance_trajectory if (traj_checks["target_guidance"].value and var_check) else None,
        ground_truth=gt_trajectory,
        delta_trajectories=(
            ([[0] + list(_t) for _t in delta_trajectories] if notebook_mode == "guided_rollout"
             else [[0] + list(delta_trajectory)])
            if (traj_checks["delta_trajectory"].value and notebook_mode in ("guided_rollout", "analyze_rollout")) else None
        ),
        cumulative_delta_trajectory=[0] + list(cumulative_delta_trajectory) if (traj_checks["delta_trajectory"].value and notebook_mode in ("guided_rollout", "analyze_rollout")) else None,
        show_guided_mean=False,
        show_unguided_mean=False,
        title=f"rollout trajectories",
        subtitle=f"{var} | mask-averaged",
        ylabel="Mask-averaged value",
        figsize=(22, 6),
        dpi=dpi_slider.value
    )
    return (trajectories_plot,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Flow analysis
    """)
    return


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_traces,
    m,
    n,
    notebook_mode,
    plot_trajectory,
    red,
    row_keys,
    t,
):
    if notebook_mode == "analyze_rollout":
        _vf_traces, _bands = cross_traces(red["vfs_l2"].isel(n=n), "t", "l2", m, **{**cross_ctl, "k": 10**9})
        _vf_traces = {_k: _vf_traces[_k] for _k in row_keys["vfs"] if _k in _vf_traces}
        _bands = {_k: _bands[_k] for _k in _vf_traces if _k in _bands} if _bands else None
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _vf_traces.values()), default=1)))
        vf_norms_plot = plot_trajectory(_vf_traces, title="Vector field norm over $t$",
            subtitle=r"$\|\mathrm{vf}_t\|$ (model vf before the guidance kick, $\mathrm{vf} = s_t\,u_t$) — L2 over space, at the selected $n$", step=t + 1, color_map=color_for(_vf_traces), bands=_bands,
            figsize=(_w, 6), prepend_zero=False, start_index=1, mirror_right_axis=True)
    else:
        vf_norms_plot = None
    return (vf_norms_plot,)


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_traces,
    m,
    n,
    notebook_mode,
    plot_trajectory,
    red,
    row_keys,
    t,
):
    if notebook_mode == "analyze_rollout":
        _gvf_traces, _bands = cross_traces(red["gui_vfs_l2"].isel(n=n), "t", "l2", m, **{**cross_ctl, "k": 10**9})
        _gvf_traces = {_k: _gvf_traces[_k] for _k in row_keys["gui_vfs"] if _k in _gvf_traces}
        _bands = {_k: _bands[_k] for _k in _gvf_traces if _k in _bands} if _bands else None
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _gvf_traces.values()), default=1)))
        guided_vf_norms_plot = plot_trajectory(_gvf_traces, title="Guided vector field norm over $t$",
            subtitle=r"$\|\mathrm{vf}^{\mathrm{gui}}_t\|$ (after the guidance kick) — L2 over space, at the selected $n$", step=t + 1, color_map=color_for(_gvf_traces), bands=_bands,
            figsize=(_w, 6), prepend_zero=False, start_index=1, mirror_right_axis=True)
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
    rollout_id,
    sweep_params,
    torch,
    xr,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        # Keep the heavy per-flow-step cubes LAZY: only `.sel` the sweep point here.
        # NEW trace format (res.zarr present): stores hold raw primitives only --
        # grads = dL/dz, vfs = u*s_t, res = noisy state z_t. Everything derived
        # (gui_vec, gui_vf, gui_res, clean_preds) is reconstructed below via exact
        # affine identities; no model needed. LEGACY stores load directly.
        grads_xr = get_rollout("grads", rollout_id).sel(sweep_params)
        vfs_xr = get_rollout("vfs", rollout_id).sel(sweep_params)
        ung_gui_xr = get_rollout("ung_gui", rollout_id).sel(sweep_params)
        # ung_gui carries the full flow-step (t) axis; its last slice is the final
        # unguided state. Guard so older stores (no t axis) still work.
        ung_gui_final_xr = ung_gui_xr.isel(t=-1) if "t" in ung_gui_xr.dims else ung_gui_xr

        try:
            res_xr = get_rollout("res", rollout_id).sel(sweep_params)
        except FileNotFoundError:
            res_xr = None  # legacy rollout (pre raw-primitives format)

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
                if len(_wv) == _T and _r["m"] < _M and _r["n"] < _N:
                    _lam[_r["m"], _r["n"], :] = _wv * _av
            lambda_t_xr = xr.DataArray(
                _lam, dims=("m", "n", "t"),
                coords={"m": grads_xr.m, "n": grads_xr.n, "t": grads_xr.t},
            )

            # flow grids: s_t (noise level) and h_t (Euler step; last step h = s)
            _s_np = np.linspace(1000, 1, _T) / 1000
            _h_np = np.empty_like(_s_np); _h_np[:-1] = _s_np[:-1] - _s_np[1:]; _h_np[-1] = _s_np[-1]
            _s_da = xr.DataArray(_s_np, dims=("t",), coords={"t": grads_xr.t})
            _h_da = xr.DataArray(_h_np, dims=("t",), coords={"t": grads_xr.t})

            # residual denorm scaler c per var (level vars: per-level vector), as in GuidedFlow
            _rsc = torch.load(STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False)
            _c_map = {}
            for _vi, _v in enumerate(VARIABLES_DICT["surface"]):
                _c_map[_v] = float(_rsc["surface"][_vi].squeeze())
            _lev_np = _rsc["level"].squeeze(-1).squeeze(-1).numpy()
            for _vi, _v in enumerate(VARIABLES_DICT["level"]):
                _arr = _lev_np[_vi] * (3.0 if _v == "vertical_velocity" else 1.0)
                _c_map[_v] = xr.DataArray(_arr, dims=("level",), coords={"level": grads_xr.level})

            # exact identities
            gui_vec_xr = grads_xr * lambda_t_xr                 # applied guidance vector per t
            gui_vfs_xr = vfs_xr - gui_vec_xr * _s_da            # guided vf, stored x s_t convention
            gui_res_xr = -(gui_vec_xr * _h_da).sum("t")         # guidance contribution to z_T
            _z_T = res_xr.isel(t=-1, drop=True) + gui_vfs_xr.isel(t=-1, drop=True) * float(_h_np[-1] / _s_np[-1])
            # clean prediction (physical): gui_final + ((z_t + s_t*u_t) - z_T) * c
            _dev = (res_xr + vfs_xr) - _z_T
            clean_preds_xr = xr.Dataset(
                {_v: guided_xr[_v] + _dev[_v] * _c_map[_v] for _v in _dev.data_vars}
            ).transpose("m", "n", "t", ...)  # broadcast puts t last; restore trace order
    return (
        clean_preds_xr,
        grads_xr,
        gui_vec_xr,
        gui_vfs_xr,
        res_xr,
        ung_gui_final_xr,
        ung_gui_xr,
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
    # method into guidance_schedule.json. lambda_t = w_t * a_t is the per-step multiplier
    # on the raw gradient; for FGWNOGAP a_t is the ACHIEVED remaining-gap fraction and
    # w_t the implied factor. `guidance_schedules` holds all methods at this selection;
    # `lambda_t`/`w_t_schedule`/`a_t_schedule` are the selected method's (None for
    # rollouts predating the sidecar -> charts fall back to the reconstructed preview).
    if notebook_mode == "analyze_rollout":
        # records store the sweep in coord-label form == the notebook sweep_params dict
        _sel = dict(sweep_params)
        _recs = get_guidance_schedule(rollout_id, _sel, m=m, n=n)
        guidance_schedules = {
            _r["method"]: {
                "w_t": np.asarray(_r["w_t"], dtype=float),
                "a_t": np.asarray(_r["a_t"], dtype=float),
                "lambda_t": np.asarray(_r["w_t"], dtype=float) * np.asarray(_r["a_t"], dtype=float),
            }
            for _r in _recs
        }
        lambda_t_by_method = {_k: _v["lambda_t"] for _k, _v in guidance_schedules.items()}
        if guidance_mode_dropdown.value in guidance_schedules:
            _mine = guidance_schedules[guidance_mode_dropdown.value]
            w_t_schedule, a_t_schedule, lambda_t = _mine["w_t"], _mine["a_t"], _mine["lambda_t"]
        else:
            w_t_schedule = a_t_schedule = lambda_t = None
    else:
        guidance_schedules = {}
        lambda_t_by_method = {}
        w_t_schedule = a_t_schedule = lambda_t = None
    return a_t_schedule, guidance_schedules, lambda_t, lambda_t_by_method


@app.cell
def _(
    clean_preds_xr,
    delta_trajectory,
    get_slices,
    grads_xr,
    gt_curr,
    gui_vec_xr,
    gui_vfs_xr,
    level,
    m,
    mask,
    n,
    notebook_mode,
    np,
    partition,
    res_xr,
    sweep_params,
    t,
    ung_gui_xr,
    ung_onl_curr,
    var,
    vfs_xr,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        # changes over t
        clean_preds_slices = get_slices(clean_preds_xr, partition, var, level)
        grads_slices = get_slices(grads_xr, partition, var, level)
        vfs_slices = get_slices(vfs_xr, partition, var, level)
        guided_vfs_slices = get_slices(gui_vfs_xr, partition, var, level)

        # slices of interest
        # 1
        diff_gt_ung_onl_slice =  ung_onl_curr - gt_curr
        diff_gt_clean_pred_slice = clean_preds_slices[m][n][t] - gt_curr
        # 2
        # compare against the unguided clean prediction AT THE SAME flow step t (the
        # ung_gui trace carries the full t axis; older stores without it fall back to final)
        _ung_gui_t_slices = get_slices(ung_gui_xr, partition, var, level)
        _ung_gui_t = _ung_gui_t_slices[m][n][t] if _ung_gui_t_slices.ndim == 5 else _ung_gui_t_slices[m][n]
        ung_onl_clean_diff_slice = clean_preds_slices[m][n][t] - _ung_gui_t
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
            # z_{t+1}: the state entering the next flow step; all-NaN -> flat map at t=T-1
            res_next_slice = _res_mn[t+1] if t + 1 < len(_res_mn) else np.full_like(np.asarray(res_slice, dtype=float), np.nan)
        else:
            res_slice = np.zeros_like(grads_slices[m][n][t])
            res_next_slice = np.zeros_like(grads_slices[m][n][t])
        if gui_vec_xr is not None:
            gui_vec_slice = get_slices(gui_vec_xr, partition, var, level)[m][n][t]
        else:
            gui_vec_slice = np.zeros_like(grads_slices[m][n][t])

        # 4
        guided_vfs_slice = guided_vfs_slices[m][n][t]
        vfs_slice = vfs_slices[m][n][t]
        # residual integrand of the masked loss: mask * (x_hat - (1+delta)*x_ref),
        # whose spatial sum IS the signed residual r_t that the gradient differentiates
        _x_ref_slice = gt_curr if sweep_params.get("GUI_REF") == "GT" else ung_onl_curr
        _delta_n = 1.0 + float(np.asarray(delta_trajectory, dtype=float)[n])
        masked_residual_slice = (clean_preds_slices[m][n][t] - _delta_n * _x_ref_slice) * np.asarray(mask)
        # Model reaction to the guidance applied at t, in SAME units: the stored vf
        # traces are each weighted by their own s_t, so the naive vfs[t+1] - gui_vfs[t]
        # is dominated by the s-decay (at the last transition it renders ~= -gui_vfs and
        # fakes 'undoing'). Compare in u-space and scale by s_{t+1} -- the disagreement's
        # actual contribution to the next state (its masked mean = the staircase's
        # deflection segment). Last step: no t+1 -> zero map.
        _s_sched = np.linspace(1000, 1, len(vfs_slices[m][n])) / 1000
        vf_gui_next_diff_slice = (
            vfs_slices[m][n][t+1] - guided_vfs_slice * (_s_sched[t+1] / _s_sched[t])
            if t + 1 < len(vfs_slices[m][n])
            else np.zeros_like(guided_vfs_slice)
        )
    return (
        clean_preds_diff_slice,
        diff_grads_slice,
        diff_gt_clean_pred_slice,
        diff_gt_ung_onl_slice,
        grads_slice,
        gui_vec_slice,
        guided_vfs_slice,
        masked_residual_slice,
        res_next_slice,
        res_slice,
        ung_onl_clean_diff_slice,
        vf_gui_next_diff_slice,
        vfs_slice,
    )


@app.cell
def _(t_slider):
    # slider is 1-based (1..T); t is the 0-based index
    t=t_slider.value-1
    return (t,)


@app.cell
def _(
    clean_preds_diff_slice,
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    contour_lw_slider,
    diff_grads_slice,
    diff_gt_clean_pred_slice,
    diff_gt_ung_onl_slice,
    dpi_slider,
    grads_slice,
    gui_vec_slice,
    guided_vfs_slice,
    mask,
    masked_residual_slice,
    notebook_mode,
    np,
    res_next_slice,
    res_slice,
    show_mask_switch,
    ung_onl_clean_diff_slice,
    vf_gui_next_diff_slice,
    vfs_slice,
    visualize_map,
    white_zero_checkbox,
    white_zero_cmap,
    white_zero_slider,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        diff_vfs_slice = guided_vfs_slice - vfs_slice

        map_specs = [
            ("diff_gt_ung_onl_map", diff_gt_ung_onl_slice, r"$x_{n}^{\text{ung_gui}} - x_{n}^{\text{gt}}$", -1, 1),
            ("diff_gt_clean_pred_map", diff_gt_clean_pred_slice, r"$x_t^{\text{gui}} - x_{n}^{\text{gt}}$", -1, 1),
            ("ung_onl_clean_diff_map", ung_onl_clean_diff_slice, r"$x_t^{\text{gui}} - x_t^{\text{ung\_gui}}$", -1, 1),
            ("clean_preds_diff_map", clean_preds_diff_slice, r"$x_t^{\text{gui}} - x_{t-1}^{\text{gui}}$", -1, 1),
            ("grads_map", grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t$", -1, 1),
            ("vfs_map", vfs_slice, r"$\text{vf}_t$", -0.001, 0.001),
            ("masked_residual_map", masked_residual_slice, r"$(\hat{x}^{\text{gui}}_t - (1+\delta_n)\,x^{\text{ref}}) \cdot \text{mask}$", -1, 1),
            ("guided_vfs_map", guided_vfs_slice, r"$\text{vf}^{\text{gui}}_t$", -0.001, 0.001),
            ("diff_vfs_map", diff_vfs_slice, r"$\text{vf}^{\text{gui}}_t - \text{vf}_t$", -0.001, 0.001),
            ("vf_gui_next_diff_map", vf_gui_next_diff_slice, r"$s_{t+1}(u_{t+1} - u^{\text{gui}}_t)$", -0.001, 0.001),
            ("res_map", res_slice, r"$z_t$", -1, 1),
            ("res_next_map", res_next_slice, r"$z_{t+1}$", -1, 1),
            ("gui_vec_map", gui_vec_slice, r"$\lambda_t\,\nabla_{z_t}\mathcal{L}_t$", -1, 1),
            ("diff_grads_map", diff_grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t - \\nabla_{z_{t-1}} \\mathcal{L}_{t-1}$", -1, 1),
        ]

        maps = {}

        for name, data, title, fallback_vmin, fallback_vmax in map_specs:
            data_min = np.nanmin(data) if np.isfinite(data).any() else np.nan
            data_max = np.nanmax(data) if np.isfinite(data).any() else np.nan
            data_mean = np.nanmean(data) if np.isfinite(data).any() else np.nan
            if not (np.isfinite(data_min) and np.isfinite(data_max)):
                # all-NaN slice (e.g. reconstruction without lambda records on mixed-run
                # dirs) -> render a flat zero map instead of crashing the colormap
                data = np.zeros_like(np.asarray(data))
                data_min = data_max = data_mean = 0.0
            print(name, data_min, data_mean, data_max)

            if white_zero_checkbox.value and data_min != data_max:
                data = np.where(np.abs(data) <= white_zero_slider.value / 100.0 * float(np.nanmax(np.abs(data))), np.nan, data)
            maps[name] = visualize_map(
                data,
                cmap=white_zero_cmap if white_zero_checkbox.value else "coolwarm",
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
                contour_linewidth=contour_lw_slider.value,
            )

        diff_gt_ung_onl_map = maps["diff_gt_ung_onl_map"]
        diff_gt_clean_pred_map = maps["diff_gt_clean_pred_map"]
        ung_onl_clean_diff_map = maps["ung_onl_clean_diff_map"]
        clean_preds_diff_map = maps["clean_preds_diff_map"]
        grads_map = maps["grads_map"]
        vfs_map = maps["vfs_map"]
        masked_residual_map = maps["masked_residual_map"]
        guided_vfs_map = maps["guided_vfs_map"]
        diff_vfs_map = maps["diff_vfs_map"]
        vf_gui_next_diff_map = maps["vf_gui_next_diff_map"]
        res_map = maps["res_map"]
        res_next_map = maps["res_next_map"]
        gui_vec_map = maps["gui_vec_map"]
        diff_grads_map = maps["diff_grads_map"]
    return (
        grads_map,
        gui_vec_map,
        guided_vfs_map,
        masked_residual_map,
        res_map,
        res_next_map,
        ung_onl_clean_diff_map,
        vfs_map,
    )


@app.cell
def _(mo):
    flow_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "gui_t diffs", "grads", "vfs", "z_t",
    )})
    return (flow_checks,)


@app.cell
def _(mo):
    # row-toggle checkboxes for the Inspect states maps (analyze mode), mirroring
    # flow_checks / cross_row_checks. Keys cover the absolute + difference rows.
    inspect_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "curr / prev", "gui / ung",
        "gt_gt / gui_gui", "gui_gt / gui_ung",
    )})
    return (inspect_checks,)


@app.cell
def _(
    contour_checkbox,
    contour_color_dropdown,
    contour_levels_slider,
    contour_lw_slider,
    dpi_slider,
    flow_checks,
    grads_map,
    gui_vec_map,
    guided_vfs_map,
    level_slider,
    m_slider,
    masked_residual_map,
    mo,
    n_slider,
    notebook_mode,
    partition_dropdown,
    res_map,
    res_next_map,
    show_mask_switch,
    sweep_params_widget,
    t_slider,
    ung_onl_clean_diff_map,
    var_dropdown,
    vfs_map,
    white_zero_checkbox,
    white_zero_slider,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        var_controls = mo.vstack(
            [
                mo.hstack(
                    [partition_dropdown, level_slider],
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
                mo.hstack([show_mask_switch, zoom_slider], justify="start", align="start"),
                mo.hstack(
                    [contour_checkbox, contour_levels_slider, contour_lw_slider, contour_color_dropdown, white_zero_checkbox, white_zero_slider],
                    justify="start", align="start",
                ),
            ],
            align="start",
        )

        map_rows = [
            ("gui_t diffs", [ung_onl_clean_diff_map, masked_residual_map]),
            ("grads", [grads_map, gui_vec_map]),
            ("vfs", [vfs_map, guided_vfs_map]),
            ("z_t", [res_map, res_next_map]),
        ]

        flow_widget_make = mo.vstack(
            [
                sweep_params_widget,
                flow_controls,
                mo.hstack([
                    mo.vstack([flow_checks[_k] for _k, _ in map_rows], justify="start", align="start").style(width="fit-content"),
                    mo.vstack(
                        [mo.hstack(_maps, justify="start", align="start") for _k, _maps in map_rows if flow_checks[_k].value],
                        justify="start", align="start",
                    ).style(width="fit-content"),
                ], justify="start", align="start"),
            ],
            justify="start", align="start",
        )
    return (flow_widget_make,)


@app.cell
def _(flow_widget_make, notebook_mode):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        flow_widget=flow_widget_make
    else:
        flow_widget=None
    flow_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross variable checks
    """)
    return


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
    cross_row_checks = mo.ui.dictionary({k: mo.ui.checkbox(label=k, value=True) for k in (
        "masked avg", "grad_norms", "gui_vf_norms", "vf_norms", "angular deflection",
    )})
    differential_checkbox = mo.ui.checkbox(label=r"$\Delta$")
    abs_checkbox = mo.ui.checkbox(label=r"$|\cdot|$", value=True)

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
        # boolean region at half-maximum: BBOX (0/1) is unchanged; GAUSSIAN is
        # nonzero everywhere (astype(bool) would be all-True -> "!mask" all-NaN),
        # and its peak is ~1/sum, so the threshold must be relative to max.
        _m = xr.DataArray(
            mask >= 0.5 * mask.max(),
            dims=("latitude", "longitude"),
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
        )
        return ds.where(_m if mode == "mask" else ~_m)

    def cross_traces(red_ds, axis, agg, m_idx, *, k, var, by_level, diff, absv, bands):
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
        traces = top_k({k_: v[m_idx if v.shape[0] > 1 else 0] for k_, v in _full.items()}, k)
        bands_out = (
            {k_: (_full[k_].min(axis=0), _full[k_].max(axis=0)) for k_ in traces if _full[k_].shape[0] > 1}
            if bands
            else None
        )
        return traces, bands_out

    def top_k(traces, k):
        # rank by the variable's AGGREGATED influence over all steps (nan-aware mean of
        # |trace|), not by a single peak step. nan-safe: traces may carry NaN gaps (e.g.
        # dvf_angle at t=0); all-NaN traces rank last.
        def _key(kv):
            _v = np.abs(kv[1])
            return -float(np.nanmean(_v)) if np.isfinite(_v).any() else np.inf
        return dict(sorted(traces.items(), key=_key)[:k])

    return (
        abs_checkbox,
        aggregate_by_level_checkbox,
        aggregate_spatially_dropdown,
        color_for,
        cross_row_checks,
        cross_traces,
        differential_checkbox,
        dist_bands_checkbox,
        level_var_dropdown,
        maybe_mask,
        n_traces,
    )


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    differential_checkbox,
    dist_bands_checkbox,
    level_var_dropdown,
    top_k_slider,
):
    cross_ctl = dict(
        k=top_k_slider.value,
        var=level_var_dropdown.value,
        by_level=aggregate_by_level_checkbox.value,
        diff=differential_checkbox.value,
        absv=abs_checkbox.value,
        bands=dist_bands_checkbox.value,
    )
    return (cross_ctl,)


@app.cell
def _(aggregate_by_level_checkbox, level_var_dropdown, mo, n_traces):
    _kmax = n_traces(level_var_dropdown.value, aggregate_by_level_checkbox.value)
    top_k_slider = mo.ui.slider(1, _kmax, value=min(5,_kmax), label=get_label("top K", _kmax), show_value=True)
    return (top_k_slider,)


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    aggregate_spatially_dropdown,
    differential_checkbox,
    level_var_dropdown,
    mo,
    top_k_slider,
):
    cross_check_controls = mo.vstack(
        [
            mo.hstack([differential_checkbox, abs_checkbox, top_k_slider], justify="start", align="start"),
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
    return gt_n_xr, xnorm


@app.cell
def _(
    aggregate_spatially_dropdown,
    clean_preds_xr,
    dask,
    grads_xr,
    gt_n_xr,
    gui_vfs_xr,
    guided_xr,
    mask,
    maybe_mask,
    notebook_mode,
    np,
    ung_gui_final_xr,
    ung_gui_xr,
    vfs_xr,
    xnorm,
    xr,
):
    # spatially-reduced cubes: built LAZILY on the dask-backed cubes, then computed in
    # one fused pass -> dask streams chunks and never materializes a full cube. Sliders
    # (m, n, t, k, variable, Δ, abs, bands) then only index these tiny arrays.
    if notebook_mode == "analyze_rollout":

        _sp = ("latitude", "longitude")
        _msk = lambda ds: maybe_mask(ds, aggregate_spatially_dropdown.value)
        _sq = lambda ds: (_msk(ds) ** 2).sum(dim=_sp)        # squared sum; sqrt after folding
        _mn = lambda ds: _msk(ds).mean(dim=_sp)
        _nmn = lambda ds: xnorm.normalize(_mn(ds))           # normalize commutes with the spatial mean
        # mask-WEIGHTED mean (rollout-trajectories convention: sum(mask*x)/sum(mask)); the
        # masked-average charts use the mask weights, independent of the aggregate dropdown.
        _mask_w = xr.DataArray(np.asarray(mask, dtype=float), dims=_sp,
                               coords={"latitude": guided_xr.latitude, "longitude": guided_xr.longitude})
        _wmn = lambda ds: (ds * _mask_w).sum(dim=_sp) / float(_mask_w.sum())
        _wnmn = lambda ds: xnorm.normalize(_wmn(ds))
        # deflection accounts for the temporal ordering: within a step the unguided vf is
        # computed before the guided one, so the guided push at t is compared to the unguided
        # vf one step later -> vf_t - vf^gui_{t-1}. shift pads the initial step (t=0) -> 0,
        # like the other difference charts.
        _dvf = (vfs_xr - gui_vfs_xr.shift(t=1)).fillna(0.0)
        red = {
            "grads_l2": _sq(grads_xr),
            "vfs_l2": _sq(vfs_xr),
            "gui_vfs_l2": _sq(gui_vfs_xr),
            "dvf_l2": _sq(_dvf),
            "dvf_mean": _mn(_dvf),
            "dvf_dot": (_msk(vfs_xr) * _msk(gui_vfs_xr.shift(t=1))).sum(dim=_sp),
            "gvf_prev_sq": _sq(gui_vfs_xr.shift(t=1)),
            "gui_ung_gui_mean": _nmn(guided_xr) - _nmn(ung_gui_final_xr),
            "gui_gt_mean": _nmn(guided_xr) - _nmn(gt_n_xr),
            "clean_gt_mean": _nmn(clean_preds_xr) - _nmn(gt_n_xr),
            "ung_gui_gt_mean": _nmn(ung_gui_final_xr) - _nmn(gt_n_xr),
            "ung_gui_gt_t_mean": _nmn(ung_gui_xr) - _nmn(gt_n_xr),
            "clean_ung_gui_mean": _nmn(clean_preds_xr) - _nmn(ung_gui_xr),
            "gui_ung_gui_denorm": _wmn(guided_xr) - _wmn(ung_gui_final_xr),
            "clean_ung_gui_denorm": _wmn(clean_preds_xr) - _wmn(ung_gui_xr),
            "gui_ung_gui_wnorm": _wnmn(guided_xr) - _wnmn(ung_gui_final_xr),
            "clean_ung_gui_wnorm": _wnmn(clean_preds_xr) - _wnmn(ung_gui_xr),
            # full-domain ||dL/dz||^2 per (m, n, t) -- consumed by the convergence plot's
            # through-Jacobian claim; UNMASKED by design (the loss gradient lives everywhere)
            "grads_l2_full": sum(
                (grads_xr[_v] ** 2).sum(dim=[_d for _d in grads_xr[_v].dims if _d not in ("m", "n", "t")])
                for _v in grads_xr.data_vars
            ),
        }
        red = dict(zip(red, dask.compute(*red.values())))    # fused: each cube read once
        # angular deflection: angle between vf_t and vf^gui_{t-1} over the (masked) spatial
        # field, per (m, n, t[, level], var). NaN at t=0 (no previous guided vf) -> skipped
        # by the mean reductions / rendered as a gap.
        _ang_denom = (red["vfs_l2"] * red["gvf_prev_sq"]) ** 0.5
        red["dvf_angle"] = np.degrees(np.arccos((red["dvf_dot"] / _ang_denom).clip(-1.0, 1.0)))
        # gap the FINAL flow step: as s_t -> 0 the velocity (r_hat - z_t)/s_t is noise/
        # curvature-dominated, so the step-to-step rotation inflates regardless of guidance
        # (and the last step applies no kick anyway, a_T = 0) -> not a guidance signal
        red["dvf_angle"] = red["dvf_angle"].where(red["dvf_angle"]["t"] != red["dvf_angle"]["t"].values[-1])
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
    cross_row_checks,
    dist_bands_checkbox,
    grad_norms_n_plot,
    grad_norms_plot,
    gui_mean_n_plot,
    gui_mean_t_plot,
    gui_vf_norms_n_plot,
    guided_vf_norms_plot,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    sweep_params_widget,
    t_slider,
    vf_angdef_n_plot,
    vf_angdef_t_plot,
    vf_norms_n_plot,
    vf_norms_plot,
):
    if notebook_mode =="analyze_rollout":
        cross_checks_widget = mo.vstack([
            sweep_params_widget,
            dist_bands_checkbox,
            cross_check_controls,
            mo.hstack([m_slider, n_slider, t_slider], justify="start"),
            mo.hstack([
                mo.vstack(list(cross_row_checks.values()), justify="start", align="start").style(width="fit-content"),
                mo.vstack(
                    [
                        mo.hstack(_row, justify="start")
                        for _key, _row in [
                            ("masked avg", [gui_mean_n_plot, gui_mean_t_plot]),
                            ("grad_norms", [grad_norms_n_plot, grad_norms_plot]),
                            ("gui_vf_norms", [gui_vf_norms_n_plot, guided_vf_norms_plot]),
                            ("vf_norms", [vf_norms_n_plot, vf_norms_plot]),
                            ("angular deflection", [vf_angdef_n_plot, vf_angdef_t_plot]),
                        ]
                        if cross_row_checks[_key].value
                    ],
                    justify="start",
                    align="start",
                ).style(width="fit-content"),
            ],
            align="start",
            justify="start",),
        ], align="start")
    else:
        cross_checks_widget = None
    cross_checks_widget
    return


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_traces,
    m,
    n,
    notebook_mode,
    plot_trajectory,
    red,
    row_keys,
    t,
):
    if notebook_mode =="analyze_rollout":
        _traces, _bands = cross_traces(red["grads_l2"].isel(n=n), "t", "l2", m, **{**cross_ctl, "k": 10**9})
        _traces = {_k: _traces[_k] for _k in row_keys["grads"] if _k in _traces}
        _bands = {_k: _bands[_k] for _k in _traces if _k in _bands} if _bands else None
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _traces.values()), default=1)))
        grad_norms_plot = plot_trajectory(_traces, title="Gradient norm over $t$",
            subtitle=r"$\|\nabla_{z_t} \mathcal{L}_t\|$ — L2 over space, at the selected $n$", step=t + 1, color_map=color_for(_traces), bands=_bands,
            figsize=(_w, 6), prepend_zero=False, start_index=1, mirror_right_axis=True,
        )
        # grad_norms_plot
    return (grad_norms_plot,)


@app.cell
def _(
    color_for,
    cross_ctl,
    cross_traces,
    m,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    red,
    t,
):
    if notebook_mode == "analyze_rollout":

        def _plot(title, subtitle, ds, axis, agg, twin_ds=None, rank_ds=None, keys=None):
            if keys is not None:
                # lock the displayed variables to a caller-provided set (e.g. the matching
                # over-n chart) so both charts of a row show the SAME traces
                _all, _bands_all = cross_traces(ds, axis, agg, m, **{**cross_ctl, "k": 10**9})
                _tr = {_k: _all[_k] for _k in keys if _k in _all}
                _bands = {_k: _bands_all[_k] for _k in _tr if _k in _bands_all} if _bands_all else None
            elif rank_ds is not None:
                # rank variables by the NORMALIZED gui - ung_gui difference in rank_ds
                # (fair across variables) instead of their own magnitude; the DISPLAY (ds)
                # stays denormalized. Pull every variable's rank + display trace (k huge),
                # sort by max|rank|, keep the top-k.
                _ctl_all = {**cross_ctl, "k": 10**9}
                _rank_all, _ = cross_traces(rank_ds, axis, agg, m, **_ctl_all)
                _gui_all, _bands_all = cross_traces(ds, axis, agg, m, **_ctl_all)
                def _rank_key(_k):  # largest |rank| first; all-NaN traces sort last
                    _v = np.abs(_rank_all[_k])
                    return -float(np.nanmax(_v)) if np.isfinite(_v).any() else np.inf
                _keys = [_k for _k in sorted(_rank_all, key=_rank_key) if _k in _gui_all][: cross_ctl["k"]]
                _tr = {_k: _gui_all[_k] for _k in _keys}
                _bands = {_k: _bands_all[_k] for _k in _keys if _k in _bands_all} if _bands_all else None
            else:
                _tr, _bands = cross_traces(ds, axis, agg, m, **cross_ctl)
            _colors = color_for(_tr)
            _styles = None
            if twin_ds is not None:
                # ung_gui twin overlay for the displayed variables: same colors, dotted
                _twin_src, _ = cross_traces(twin_ds, axis, agg, m, **{**cross_ctl, "k": 10**9})
                _twin = {f"{_k} (ung_gui)": _twin_src[_k] for _k in _tr if _k in _twin_src}
                _colors |= {_k: _colors[_k.removesuffix(" (ung_gui)")] for _k in _twin}
                _styles = {_k: ":" for _k in _twin}
                _tr = _tr | _twin
            # width scales with the number of steps so the sparse "over n" plots
            # (N points) are not stretched across the same width as "over t" (T points)
            _nsteps = max((len(_v) for _v in _tr.values()), default=1)
            _w = min(22.0, max(8.0, 3.4 + 0.78 * _nsteps))
            return plot_trajectory(
                _tr, title=title, subtitle=subtitle, xlabel=f"${axis}$",
                step=(n + 1 if axis == "n" else t + 1), color_map=_colors, bands=_bands,
                figsize=(_w, 6), prepend_zero=False, start_index=1,
                mirror_right_axis=True, linestyle_map=_styles,
            )

        grad_norms_n_plot = _plot("Gradient norm over $n$", r"$\|\nabla_{z_t} \mathcal{L}_t\|$ — L2 over space and all flow steps $t$", red["grads_l2"], "n", "l2")
        # per-row variable sets, ranked on the over-n cubes (which fold ALL n and t):
        # the over-t charts display the SAME variables, so ranking always reflects the
        # aggregated influence of the variable over all steps, never a single-step peak
        row_keys = {
            "masked_avg": list(cross_traces(red["gui_ung_gui_wnorm"], "n", "mean", m, **cross_ctl)[0]),
            "grads": list(cross_traces(red["grads_l2"], "n", "l2", m, **cross_ctl)[0]),
            "gui_vfs": list(cross_traces(red["gui_vfs_l2"], "n", "l2", m, **cross_ctl)[0]),
            "vfs": list(cross_traces(red["vfs_l2"], "n", "l2", m, **cross_ctl)[0]),
            "angle": list(cross_traces(red["dvf_angle"], "n", "mean", m, **cross_ctl)[0]),
        }
        vf_angdef_n_plot = _plot("Angular deflection over $n$", r"$\angle(\mathrm{vf}_t,\ \mathrm{vf}^{\mathrm{gui}}_{t-1})$ [$^\circ$] — rotation of the model vf away from the previous guided vf, mean over $t$", red["dvf_angle"], "n", "mean")
        vf_angdef_t_plot = _plot("Angular deflection over $t$", r"$\angle(\mathrm{vf}_t,\ \mathrm{vf}^{\mathrm{gui}}_{t-1})$ [$^\circ$] — rotation of the model vf away from the previous guided vf, at the selected $n$; last step omitted ($s_t \to 0$ artifact)", red["dvf_angle"].isel(n=n), "t", "mean", keys=row_keys["angle"])
        gui_vf_norms_n_plot = _plot("Guided vector field norm over $n$", r"$\|\mathrm{vf}^{\mathrm{gui}}_t\|$ (after the guidance kick, $\mathrm{vf} = s_t\,u_t$) — L2 over space and all flow steps $t$", red["gui_vfs_l2"], "n", "l2")
        vf_norms_n_plot = _plot("Vector field norm over $n$", r"$\|\mathrm{vf}_t\|$ (model vf before the guidance kick, $\mathrm{vf} = s_t\,u_t$) — L2 over space and all flow steps $t$", red["vfs_l2"], "n", "l2")
        diff_vfs_t_plot = _plot("Diff (gui vf − ung vf) over $t$", r"$\mathrm{mean}_{\mathrm{spatial}}\,(\mathrm{vf}^{\mathrm{gui}}_t - \mathrm{vf}_t)$", red["dvf_mean"].isel(n=n), "t", "mean")
        gui_mean_n_plot = _plot("Masked average: guided − unguided over $n$", r"$\mathrm{mean}_{\mathrm{mask}}\,(\tilde{x}^{\,\mathrm{gui}}_{n}) - \mathrm{mean}_{\mathrm{mask}}\,(\tilde{x}^{\,\mathrm{ung\_gui}}_{n})$ — final states per forecast step, normalized units", red["gui_ung_gui_wnorm"], "n", "mean")
        gui_mean_t_plot = _plot("Masked average: guided − unguided over $t$", r"$\mathrm{mean}_{\mathrm{mask}}\,(\tilde{x}^{\,\mathrm{gui}}_{t}) - \mathrm{mean}_{\mathrm{mask}}\,(\tilde{x}^{\,\mathrm{ung\_gui}}_{t})$ — clean predictions along the flow at the selected $n$, normalized units", red["clean_ung_gui_wnorm"].isel(n=n), "t", "mean", keys=row_keys["masked_avg"])
    else:
        row_keys = None
        gui_mean_n_plot = gui_mean_t_plot = vf_angdef_n_plot = vf_angdef_t_plot = grad_norms_n_plot = diff_vfs_t_plot = gui_vf_norms_n_plot = vf_norms_n_plot = None
    return (
        grad_norms_n_plot,
        gui_mean_n_plot,
        gui_mean_t_plot,
        gui_vf_norms_n_plot,
        row_keys,
        vf_angdef_n_plot,
        vf_angdef_t_plot,
        vf_norms_n_plot,
    )


@app.cell
def _(
    a_t_schedule,
    cfg_clean_preds_slices,
    cfg_target_guidance_M_N_trajectories,
    dist_bands_checkbox,
    dpi_slider,
    get_masked_mean,
    guidance_mode_dropdown,
    lambda_t,
    m,
    mask,
    n,
    notebook_mode,
    np,
    plt,
    red,
    t,
):
    if notebook_mode =="analyze_rollout":
        _all_per_t = get_masked_mean(cfg_clean_preds_slices[:, n], mask).astype(float) - cfg_target_guidance_M_N_trajectories[:, n][:, None]
        _diff_per_t = _all_per_t[m]
        # ---- guidance claim: first-order effect THROUGH the model Jacobian ----
        # g = dL/dz = 2 r dS/dz, so one guided Euler step (dz = -h*lambda*g) changes the
        # masked-sum residual by  dS = <dS/dz, dz> = -h * lambda_t * ||g||^2 / (2 r).
        # This is the linearization the guidance itself is calibrated in: for FGWNOGAP,
        # post(t) lands EXACTLY on the prescribed schedule r_target = (1-eta)^(t+1) * r_0
        # (no overshoot by construction, drawn as the dashed reference); for FGW/FGWNOLR
        # it is the honest one-step claim. The identity-path object mean(gui_vfs-vfs)*c
        # is misleading here: the loss gradient acts through the network Jacobian, and
        # its z-space direction can even oppose its effect on the observable.
        if lambda_t is not None:
            # precomputed in the fused `red` pass -> indexing a tiny in-memory array
            _g2_t = np.asarray(red["grads_l2_full"].isel(m=m, n=n), dtype=float)
            _s_flow = np.linspace(1000, 1, len(_diff_per_t)) / 1000
            _h_flow = np.empty_like(_s_flow); _h_flow[:-1] = _s_flow[:-1] - _s_flow[1:]; _h_flow[-1] = _s_flow[-1]
            _r_sum = _diff_per_t * float(np.asarray(mask).sum())  # masked-SUM residual (mask sums to 1)
            _dS = np.where(np.abs(_r_sum) > 1e-12, -_h_flow * lambda_t * _g2_t / (2.0 * _r_sum), 0.0)
            _post_sel = _diff_per_t + _dS
        else:
            # rollout predates the guidance_schedule sidecar -> no recorded lambda_t
            _post_sel = np.full_like(_diff_per_t, np.nan)
        _guidance_move = _post_sel - _diff_per_t          # within-step: achieved(t) - measured(t)
        _deflection_move = _diff_per_t[1:] - _post_sel[:-1]  # between steps: measured(t+1) - achieved(t)

        _T_len = len(_diff_per_t)
        _xt = np.arange(1, _T_len + 1).astype(float)   # both points at the same integer t

        # between-step drift line: post(t) -> pre(t+1)
        _xz = np.empty(2 * _T_len); _yz = np.empty(2 * _T_len)
        _xz[0::2], _xz[1::2] = _xt, _xt
        _yz[0::2], _yz[1::2] = _diff_per_t, _post_sel

        _wt = 22.0  # match the rollout trajectories figure width so the stacked plots align
        with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "legend.fontsize": 9}):
            _fig, _ax = plt.subplots(figsize=(_wt, 6), dpi=dpi_slider.value)
            if dist_bands_checkbox.value:
                _ax.fill_between(_xt, _all_per_t.min(axis=0), _all_per_t.max(axis=0),
                                 color="#B7950B", alpha=0.14, linewidth=0, label=f"pre-step range, M={_all_per_t.shape[0]}")
            # waterfall candles anchored ON the trajectory (same units as the axis):
            # blue = the guidance move at t (ung_t -> gui_t);
            # red  = the deflection arriving at t (gui_{t-1} -> ung_t)
            # red bar sits just LEFT of the tick (the realization gap arriving at t),
            # blue bar just RIGHT of it (the new claim leaving t)
            _bar_off = 0.16
            _ax.bar(_xt + _bar_off, _guidance_move, bottom=_diff_per_t, width=0.28,
                    color="#2E86C1", alpha=0.35, zorder=3, label=r"guidance claim (1st order, via Jacobian)")
            if _T_len > 1:
                _ax.bar(_xt[1:] - _bar_off, _deflection_move, bottom=_post_sel[:-1], width=0.28,
                        color="#C0392B", alpha=0.35, zorder=3, label=r"realization gap  (pre$_{t+1}$ − post$_t$)")
            # thin drift lines keep the trajectory readable across steps
            for _i in range(_T_len - 1):
                _ax.plot([_xt[_i], _xt[_i + 1]], [_post_sel[_i], _diff_per_t[_i + 1]],
                         "-", color="#B7950B", alpha=0.5, linewidth=1.1, zorder=4,
                         label="model drift" if _i == 0 else "_nolegend_")
            # recorded a_t IS the theoretical remaining-gap schedule for NOGAP, so this
            # reference stays correct across rollouts regardless of the exponent used
            if guidance_mode_dropdown.value == "FGWNOGAP" and a_t_schedule is not None:
                _ax.plot(_xt, _diff_per_t[0] * np.asarray(a_t_schedule, dtype=float),
                         "--", color="#888888", linewidth=1.2, alpha=0.9, zorder=2,
                         label=r"NOGAP schedule  $r_0\,a_t$")
            _ax.plot(_xt, _post_sel, "D", color="#2E86C1", markersize=5, zorder=6,
                     label=r"claimed after step $t$")
            _ax.plot(_xt, _diff_per_t, "o", markerfacecolor="none", markeredgecolor="#B7950B",
                     markeredgewidth=1.8, markersize=8, linestyle="none", zorder=7,
                     label=r"measured before step $t$  (clean pred)")
            _ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
            _ax.axvline(t + 1, color="#222222", linestyle=(0, (4, 4)), linewidth=1.1, alpha=0.7, zorder=2)
            _ax.set_xlim(0.6, _T_len + 0.4)
            _ax.set_xticks(_xt)
            _ax.set_xlabel("$t$"); _ax.set_ylabel("masked mean − target")
            _ax.set_title("Guidance convergence over $t$", loc="left", fontweight="bold")
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
            _ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            _ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
            _fig.tight_layout(rect=(0, 0, 0.82, 1))
        guidance_convergence_t_plot = _fig
    return (guidance_convergence_t_plot,)


@app.cell
def _(mo, plt):
    contour_checkbox = mo.ui.checkbox(label="contours", value=True)
    contour_levels_slider = mo.ui.slider(4, 30, step=2, value=4, label="levels: ", show_value=True, debounce=True)
    contour_lw_slider = mo.ui.slider(start=0.4, stop=2.0, step=0.2, value=0.4, label="linewidth: ", show_value=True, debounce=True)
    contour_color_dropdown = mo.ui.dropdown(["black", "dimgray", "red", "white"], value="dimgray", label="contour color: ")
    white_zero_checkbox = mo.ui.checkbox(label="white zeros")
    white_zero_slider = mo.ui.slider(start=0.0, stop=100.0, step=0.5, value=50.0, label="white below (% of max |v|): ", show_value=True, debounce=True)
    white_zero_cmap = plt.get_cmap("coolwarm").copy()
    white_zero_cmap.set_bad("white")
    return (
        contour_checkbox,
        contour_color_dropdown,
        contour_levels_slider,
        contour_lw_slider,
        white_zero_checkbox,
        white_zero_cmap,
        white_zero_slider,
    )


@app.cell
def _(GUIDANCE_METHODS, GUI_REFS, MASK_MODES, mo, np):
    # ===== sweep authoring widgets (guided_rollout) =====
    guidance_mode_select = mo.ui.multiselect(GUIDANCE_METHODS, value=["FGWNOLR"], label="GUIDANCE_MODE: ")
    gui_ref_select = mo.ui.multiselect(GUI_REFS, value=["UNG"], label="GUI_REF: ")
    mask_mode_select = mo.ui.multiselect(MASK_MODES, value=["GAUSSIAN"], label="MASK_MODE: ")

    # numeric axes -> (start, stop, log_scale, integer)
    # keys equal the guidance-fn kwarg names (see GUIDANCE_METHOD_HYPERS)
    NUMERIC_AXES = {
        # FGWNOLR (secant on the exact scalar dL/dw; no lr, no iteration count --
        # optimizes until the hardcoded loss threshold in _fgwnolr_flow is reached)
        "fgwnolr_w_init": (1000.0, 5000.0, False, False),
        # eta: shared closure rate for FGWNOLR and FGWNOGAP; a_t = (1 - eta)^(t+1)
        # (FGWNOGAP: fraction of the remaining gap closed per step)
        "eta":            (0.01,  0.1,   False, False),
    }

    _rc = {}
    for _ax, (_s, _e, _log, _int) in NUMERIC_AXES.items():
        _rc[f"{_ax}.start"] = mo.ui.number(value=_s, label="start: ")
        _rc[f"{_ax}.stop"]  = mo.ui.number(value=_e, label="stop: ")
        _rc[f"{_ax}.n"]     = mo.ui.slider(1, 20, step=1, value=1, label="n: ", show_value=True)
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
        compute_axis_values,
        gui_ref_select,
        guidance_mode_select,
        mask_mode_select,
        sweep_ranges,
    )


if __name__ == "__main__":
    app.run()
