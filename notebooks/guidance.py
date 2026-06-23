import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch
    import numpy as np
    from datetime import datetime, timedelta, date, time
    import calendar
    import cv2

    return cv2, mo, np


@app.cell
def _():
    from src.paths import ROLLOUTS
    from src.rollout_config import MASK_MODES, RolloutConfig, GUIDANCE_MODES, GUI_REFS, REG_TYPES
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.helpers import max_day, get_timestamp_from_sliders
    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory

    from src.utils import get_var_idx, get_level_idx
    from src.utils import get_now_timestamp, ensure_rollout_dir
    from src.utils import get_timestamps, get_N_timestamps, get_N_slices, get_slices, get_gt_rollout
    from src.utils import (
        dump_json, get_rollout_ids, get_rollout, get_sweep_dict, get_config, sweep_coord_label
    )
    from src.schedules import N_schedule, delta_schedule

    from src.mask import get_masked_mean, get_mask_2d, get_mu_sigma, get_mask_center
    from src.target import get_target_slices

    return (
        GUIDANCE_MODES,
        GUI_REFS,
        LEVELS_DICT,
        MASK_MODES,
        PARTITIONS,
        REG_TYPES,
        RolloutConfig,
        VARIABLES_DICT,
        dump_json,
        ensure_rollout_dir,
        get_N_timestamps,
        get_config,
        get_gt_rollout,
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
        max_day,
        plot_trajectory,
        sweep_coord_label,
        visualize_map,
    )


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
    reg_type_select,
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
                "REG_TYPE": list(reg_type_select.value),
                "GUIDANCE_DELTA": delta_trajectories,
                **{ax: compute_axis_values(ax, _rv) for ax in NUMERIC_AXES},
            }
            if all(sweep[a] for a in ("GUIDANCE_MODE", "GUI_REF", "MASK_MODE", "REG_TYPE")):
                dump_json(sweep, rollout_dir / "sweep_params.json")
            else:
                print("each categorical axis needs at least one value")
    return


@app.cell
def _(
    GUIDANCE_MODES,
    GUI_REFS,
    MASK_MODES,
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
                label="w (guidance strength): ",
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            guidance_mode_dropdown = mo.ui.dropdown(options=GUIDANCE_MODES, value=GUIDANCE_MODES[0], label="guidance_mode: ")
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
                label="w (guidance strength): ",
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            guidance_mode_dropdown = mo.ui.dropdown(options=GUIDANCE_MODES, value=GUIDANCE_MODES[0], label="guidance_mode: ")
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
            w_slider = mo.ui.slider(
                steps=experiment_params["W"],
                value=experiment_params["W"][0],
                label="w (guidance strength): ",
                debounce=True,
                show_value=True,
            )
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
                "W": w_slider,
                "GUIDANCE_DELTA": delta_trajectory_dropdown,
            }
            _extra = {}
            for _k, _vals in experiment_params.items():
                if _k in _NAMED_CONTROLS:
                    continue
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
                "w": w_slider.value,
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


@app.cell
def _(get_config, get_rollout, notebook_mode, rollout_id, sweep_params):
    # data objects and config
    match notebook_mode:
        case "unguided_rollout":
            unguided_xr=None
            guided_xr=None
            config=None
            # TODO: set everything to None
        case "guided_rollout":
            unguided_xr = get_rollout("ung", rollout_id)
            guided_xr = None
            config = get_config(rollout_id)
        case "analyze_rollout":
            unguided_xr = get_rollout("ung", rollout_id).compute()
            guided_xr = get_rollout("gui", rollout_id).sel(sweep_params).compute()
            config = get_config(rollout_id)
        case _:
            pass
    return config, guided_xr, unguided_xr


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
    ung_gui_xr,
):
    # guidance-target at config coords (pinned — independent of browsing sliders)
    if notebook_mode == "analyze_rollout":
        cfg_clean_preds_slices = get_slices(clean_preds_xr, config.PARTITION, config.VAR, config.LEVEL)
        cfg_ung_gui_M_N_slices = get_slices(ung_gui_xr, config.PARTITION, config.VAR, config.LEVEL)
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
    ung_gui_xr,
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

            ung_gui_M_N_slices = get_slices(ung_gui_xr, partition, var, level)
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
        ung_gui_M_N_slices,
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
        "W": w_slider.value,
        "GUI_REF": guidance_reference_dropdown.value,
        "MASK_MODE": mask_mode_dropdown.value,
        "GUIDANCE_MODE": guidance_mode_dropdown.value,
    }
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
    n=n_slider.value-1
    return (n,)


@app.cell
def _(m_slider):
    m=m_slider.value
    return (m,)


@app.cell
def _(PARTITIONS, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(LEVELS_DICT, VARIABLES_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition]
    level_label = "level:\u00A0\u00A0" if partition == "surface" else "level: "
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[-1], label=level_label, show_value=True, debounce=True)
    VARIABLES = VARIABLES_DICT[partition]
    if partition == "surface":
        DEFAULT_VAR_VALUE = VARIABLES[2]
    else:
        DEFAULT_VAR_VALUE = VARIABLES[3]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=DEFAULT_VAR_VALUE, label="var : ")
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
        label="n: ",
        value=1,
        debounce=True,
        show_value=True
    )

    m_slider_label = "m:\u00A0\u00A0" if M==1 else "m: "
    m_slider = mo.ui.slider(
        start=0, 
        stop=M-1,
        step=1,
        label=m_slider_label,
        value=0,
        debounce=True,
        show_value=True
    )
    return m_slider, n_slider


@app.cell
def _(mo):
    # number of linear delta trajectories to author (each a sweep value of GUIDANCE_DELTA)
    n_deltas_slider = mo.ui.slider(1, 6, step=1, value=1, label="delta trajectories", show_value=True)
    return (n_deltas_slider,)


@app.cell
def _(N, mo, n_deltas_slider, notebook_mode):
    # per-delta linear params: start % , peak % , start@n , stop@n  (K rows)
    if notebook_mode == "guided_rollout":
        _dc = {}
        for _i in range(n_deltas_slider.value):
            _dc[f"{_i}.start"]    = mo.ui.number(value=0.0, label="start %")
            _dc[f"{_i}.peak"]     = mo.ui.number(value=5.0, label="peak %")
            _dc[f"{_i}.start_at"] = mo.ui.slider(1, N, step=1, value=1, label="start@n", show_value=True)
            _dc[f"{_i}.stop_at"]  = mo.ui.slider(1, N, step=1, value=N, label="stop@n", show_value=True)
        delta_controls = mo.ui.dictionary(_dc)
    else:
        delta_controls = mo.ui.dictionary({})
    return (delta_controls,)


@app.cell(hide_code=True)
def _(N, delta_controls, mo, n_deltas_slider, notebook_mode):
    # linear (ramp-only) delta trajectories: 0 before start@n, ramp start%->peak% over
    # [start@n, stop@n], 0 after stop@n.
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
        # controls only; the trajectories are drawn on the rollout-trajectories chart's right axis
        _rows = [
            mo.hstack([mo.md(f"delta {_i}: "), delta_controls[f"{_i}.start"], delta_controls[f"{_i}.peak"],
                       delta_controls[f"{_i}.start_at"], delta_controls[f"{_i}.stop_at"]],
                      justify="start", align="center")
            for _i in range(n_deltas_slider.value)
        ]
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
def _(GUIDANCE_MODES, GUI_REFS, MASK_MODES, REG_TYPES, mo, np):
    # ===== sweep authoring widgets (guided_rollout) =====
    guidance_mode_select = mo.ui.multiselect(GUIDANCE_MODES, value=["LBG"], label="GUIDANCE_MODE")
    gui_ref_select = mo.ui.multiselect(GUI_REFS, value=["UNG"], label="GUI_REF")
    mask_mode_select = mo.ui.multiselect(MASK_MODES, value=["BBOX"], label="MASK_MODE")
    reg_type_select = mo.ui.multiselect(REG_TYPES, value=["ID"], label="REG_TYPE")

    # numeric axes -> (start, stop, log_scale, integer)
    NUMERIC_AXES = {
        "W":             (0.1,  5.0,   False, False),
        "BETA_REG":      (1e-4, 1e-2,  True,  False),
        # LBG-MC
        "N_MC":          (4,    16,    False, True),
        "R":             (0.1,  1.0,   False, False),
        # UG
        "K":             (10,   50,    False, True),
        "ETA":           (1e-2, 1e-1,  True,  False),
        "S":             (4,    8,     False, True),
        # FG / FGF (legacy, not finalized)
        "OPTIMIZE_K":    (10,   50,    False, True),
        "OPTIMIZE_LR":   (1e-2, 1e-1,  True,  False),
        "SHIFT_INIT":    (0.0,  1.0,   False, False),
        "CONTROL_GAMMA": (1e-3, 1e-2,  True,  False),
        "LAMBDA_INIT":   (0.0,  2.0,   False, False),
        "N_WINDOWS":     (5,    25,    False, True),
    }

    _rc = {}
    for _ax, (_s, _e, _log, _int) in NUMERIC_AXES.items():
        _rc[f"{_ax}.start"] = mo.ui.number(value=_s, label="start")
        _rc[f"{_ax}.stop"]  = mo.ui.number(value=_e, label="stop")
        _rc[f"{_ax}.n"]     = mo.ui.slider(1, 20, step=1, value=1, label="n", show_value=True)
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
        reg_type_select,
        sweep_ranges,
    )


@app.cell
def _(
    NUMERIC_AXES,
    compute_axis_values,
    gui_ref_select,
    guidance_mode_select,
    mask_mode_select,
    mo,
    notebook_mode,
    reg_type_select,
    sweep_ranges,
):

    # ===== sweep authoring panel (guided_rollout) =====
    # reload so MODE_HYPERS reflects the latest src/rollout_config.py (W is now mode-specific,
    # shown only for the modes that use it: LBG / LBG-MC / UG).
    import importlib as _importlib
    import src.rollout_config as _rc_mod
    _importlib.reload(_rc_mod)
    _MODE_HYPERS = _rc_mod.MODE_HYPERS
    _COMMON_NUM = ["BETA_REG"]

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
        mo.hstack([guidance_mode_select, gui_ref_select, mask_mode_select, reg_type_select], justify="start"),
        mo.md("**Common numeric**:"),
        *[_sweep_row(ax) for ax in _COMMON_NUM],
        mo.md(f"**Specific**:"),
        *([_sweep_row(ax) for ax in _mode_num] if _mode_num else [mo.md("_none_")]),
    ], align="start")

    hypers_widget if notebook_mode == "guided_rollout" else None

    return


@app.cell(hide_code=True)
def wa_schedule(
    compute_axis_values,
    guidance_mode_select,
    mo,
    notebook_mode,
    np,
    plot_trajectory,
    sweep_ranges,
    w_slider,
):

    # Per-step guidance weight w * a_t across the flow (LBG / LBG-MC / UG), drawn with the
    # project's plot_trajectory styling (same look as the other trajectory charts).
    # a_t = (1 - t)/t is the CondOT score->vector-field conversion (MIT notes); mirrors
    # GuidedFlow.guidance_scale: s_t = 1 - t is the noise level, a_t = s_t / max(1 - s_t, ds).
    if notebook_mode == "guided_rollout":
        import matplotlib.pyplot as _plt

        _T, _N = 25, 1000                       # mirror GuidedFlow.T / num_train_timesteps
        _ds = (_N - 1) / (_N * (_T - 1))
        _s = np.linspace(_N, 1, _T) / _N        # code's s_t (1 = noise -> 0 = data)
        _a_t = _s / np.maximum(1 - _s, _ds)     # a_t = (1 - t)/t, clamped at the first step

        _ws = compute_axis_values("W", sweep_ranges.value) or [float(w_slider.value)]
        _traj = {f"w={_w:g}": (_w * _a_t).tolist() for _w in _ws}

        _modes = list(guidance_mode_select.value)
        _applies = [m for m in _modes if m in ("LBG", "LBG-MC", "UG")]
        _sub = (
            r"$a_t=(1-t)/t$, clamped at the first step — applies to "
            + (", ".join(_applies) if _applies else "LBG / LBG-MC / UG")
        )

        _fig = plot_trajectory(
            _traj,
            var=r"$w\,a_t$",
            title="Guidance weight schedule",
            subtitle=_sub,
            xlabel=r"$t$",
        )
        wa_schedule_widget = mo.as_html(_fig)
        _plt.close(_fig)
    else:
        wa_schedule_widget = mo.md("_guidance weight schedule is shown in guided_rollout mode_")
    wa_schedule_widget

    return


@app.cell
def _(
    M_N_widget,
    T_slider,
    delta_widget,
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
    weather_map,
):
    mask_widget_controls = mo.hstack(
        [partition_dropdown, var_dropdown, level_slider],
        justify="start",
        align="start",
    )

    mask_widget_maps = mo.hstack(
        [weather_map, mask_map],
        justify="start",
        align="start",
    )
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
                mo.hstack([mo.vstack(list(traj_checks.values())), trajectories_plot], justify="start", align="start")
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
                mo.hstack([mo.vstack(list(traj_checks.values())), trajectories_plot], justify="start", align="start")
            ], align="start")
            mask_widget = mo.vstack([sweep_params_widget, mask_widget_maps], align="start")
            inspect_states_widget=inspect_states_widget_make
        case _:
            pass
    return (
        inspect_states_widget,
        mask_widget,
        mask_widget_controls,
        trajectory_widget,
    )


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
    norm_modes = ["own_scale", "same_scale"]
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
    ung_gui_xr,
    var,
):
    if notebook_mode =="guided_rollout":
        ung_curr = ung_M_N_slices[m][n]
        gt_curr = gt_N_slices[n]
        ung_prev = ung_M_N_slices[m][n-1] if n>0 else ung_M_N_slices[m][n]
        gt_prev = gt_N_slices[n-1] if n>0 else gt_N_slices[n]

        gt_gt = gt_curr - gt_prev
        gt_ung = gt_curr - ung_curr

    if notebook_mode =="analyze_rollout":
        ung_curr = ung_M_N_slices[m][n]
        gui_curr = gui_M_N_slices[m][n]
        gt_curr = gt_N_slices[n]
        # ung_prev = ung_M_N_slices[m][n-1] if n>0 else ung_M_N_slices[m][n]
        gui_prev = gui_M_N_slices[m][n-1] if n>0 else gui_M_N_slices[m][n]
        gt_prev = gt_N_slices[n-1] if n>0 else gt_N_slices[n]

        ung_onl_slice = get_slices(ung_gui_xr, partition, var, level)
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
        ung_gui_gt,
        ung_gui_ung_gui,
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
    ung_gui_gt,
    ung_gui_ung_gui,
    ung_onl_curr,
    ung_prev,
    visualize_map,
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
                    absolute_maps[label] = visualize_map(
                        arr,
                        mask_2d=mask,
                        title=label,
                        interactive=map_interactive,
                        vmin=abs_vmin,
                        vmax=abs_vmax,
                        center=abs_center,
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
                    else:
                        v_min, v_max = diff_vmin, diff_vmax

                    difference_maps[label] = visualize_map(
                        arr,
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
                    absolute_maps[label] = visualize_map(
                        arr,
                        mask_2d=mask,
                        title=label,
                        interactive=map_interactive,
                        vmin=abs_vmin,
                        vmax=abs_vmax,
                        center=abs_center,
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
                    ("$x_{n}^{\\text{ung_gui}} - x_{n}^{\\text{gt}}$", ung_gui_gt),
                    ("$x_{n}^{\\text{ung_gui}} - x_{n-1}^{\\text{ung_gui}}$", ung_gui_ung_gui),
                ]

                diff_vmin = min(float(np.nanmin(arr)) for _, arr in difference_panels)
                diff_vmax = max(float(np.nanmax(arr)) for _, arr in difference_panels)

                difference_maps = {}

                for label, arr in difference_panels:
                    # is_guided_unguided = label == "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"

                    if norm_mode_dropdown.value == "own_scale":
                        v_min = min(float(np.nanmin(arr)), -1e-12)
                        v_max = max(float(np.nanmax(arr)), 1e-12)
                    else:
                        v_min, v_max = diff_vmin, diff_vmax

                    difference_maps[label] = visualize_map(
                        arr,
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
                ung_gui_gt_map = difference_maps["$x_{n}^{\\text{ung_gui}} - x_{n}^{\\text{gt}}$"]
                ung_gui_ung_gui_map = difference_maps["$x_{n}^{\\text{ung_gui}} - x_{n-1}^{\\text{ung_gui}}$"]
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
                    else:
                        _v_min, _v_max = gradmap_mag_vmin, gradmap_mag_vmax
                    print(_v_min, _v_max)
                    gradmap_figures.append(
                        visualize_map(
                            gradmap_arr,
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
                    else:
                        _v_min, _v_max = min(sobel_diff_vmin, -1e-12), max(sobel_diff_vmax, 1e-12)
                    sobel_diff_figures.append(
                        visualize_map(
                            sobel_diff_arr,
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
        ung_gui_gt_map,
        ung_gui_ung_gui_map,
        ung_map,
        ung_prev_map,
    )


@app.cell
def _(
    analysis_type_dropdown,
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
    ung_gui_gt_map,
    ung_gui_ung_gui_map,
    ung_map,
    ung_prev_map,
    var_dropdown,
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
                        mo.hstack([show_mask_switch, zoom_slider], justify="start"),
                        mo.hstack([curr_map, prev_map], justify="start"),
                        mo.hstack([ung_map, ung_prev_map], justify="start"),
                    ],
                    justify="start",
                )

            case "difference":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, mo.md(r"*own: $[\min, \max]$ per map · same: $[\min, \max]$ across maps · centered at 0*")], justify="start", align="center"),
                        mo.hstack([gt_gt_map, gt_ung_map], justify="start")
                    ], justify="start",
                )
            case _:
                # sobel_grads (and any other type) has no guided-mode panel; show a
                # placeholder so inspect_states_widget_make is always defined here.
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.md(f"_'{analysis_type_dropdown.value}' analysis is not available in guided_rollout mode._"),
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
                        mo.hstack([show_mask_switch, zoom_slider], justify="start"),
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
                    ("ung_gui diffs", [ung_gui_gt_map, ung_gui_ung_gui_map]),
                ]
                inspect_states_widget_make = mo.vstack(
                    [
                        sweep_params_widget,
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, mo.md(r"*own: $[\min, \max]$ per map · same: $[\min, \max]$ across maps · centered at 0*")], justify="start", align="center"),
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
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, mo.md(r"*own: $[\min, \max]$ per map · same: $[\min, \max]$ across maps · centered at 0*")], justify="start", align="center"),
                        sobel_grad_widget
                    ], justify="start",
                )
            case "sobel_diffs":
                inspect_states_widget_make = mo.vstack(
                    [
                        sweep_params_widget,
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, mo.md(r"*own: $[\min, \max]$ per map · same: $[\min, \max]$ across maps · centered at 0*")], justify="start", align="center"),
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
    target_guidance_M_N_trajectories,
    target_guidance_trajectory,
    timestamps,
    traj_checks,
    ung_M_N_trajectories,
    ung_gui_m_trajectory,
    ung_m_trajectory,
    var,
):
    from src.ui.plot_trajectories import plot_trajectories

    var_check = (var==config.VAR if notebook_mode in ("guided_rollout", "analyze_rollout") else False)

    import importlib as _importlib
    import src.ui.plot_trajectories as _ptm
    _importlib.reload(_ptm)
    trajectories_plot = _ptm.plot_trajectories(
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross variable checks
    """)
    return


@app.cell
def _(mask, mo, np):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

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
        "gt_diffs", "diffs", "grad_norms", "gui_vf_norms", "vf_norms", "deflections", "convergence",
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
        import xarray as _xr
        _m = _xr.DataArray(
            mask.astype(bool),
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
        return dict(sorted(traces.items(), key=lambda kv: -float(np.max(np.abs(kv[1]))))[:k])

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
        plt,
    )


@app.cell(hide_code=True)
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
    top_k_slider = mo.ui.slider(1, _kmax, value=min(5,_kmax), label="top K", show_value=True)
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
def _(gt_rollout, guided_xr, notebook_mode):
    from src.normalization import XarrayNormalizer
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
    grads_xr,
    gt_n_xr,
    gui_vfs_xr,
    guided_xr,
    maybe_mask,
    notebook_mode,
    ung_gui_xr,
    vfs_xr,
    xnorm,
):
    # spatially-reduced cubes: built LAZILY on the dask-backed cubes, then computed in
    # one fused pass -> dask streams chunks and never materializes a full cube. Sliders
    # (m, n, t, k, variable, Δ, abs, bands) then only index these tiny arrays.
    if notebook_mode == "analyze_rollout":
        import dask

        _sp = ("latitude", "longitude")
        _msk = lambda ds: maybe_mask(ds, aggregate_spatially_dropdown.value)
        _sq = lambda ds: (_msk(ds) ** 2).sum(dim=_sp)        # squared sum; sqrt after folding
        _mn = lambda ds: _msk(ds).mean(dim=_sp)
        _nmn = lambda ds: xnorm.normalize(_mn(ds))           # normalize commutes with the spatial mean
        _dvf = gui_vfs_xr - vfs_xr
        red = {
            "grads_l2": _sq(grads_xr),
            "vfs_l2": _sq(vfs_xr),
            "gui_vfs_l2": _sq(gui_vfs_xr),
            "dvf_l2": _sq(_dvf),
            "dvf_mean": _mn(_dvf),
            "gui_ung_gui_mean": _nmn(guided_xr) - _nmn(ung_gui_xr),
            "gui_gt_mean": _nmn(guided_xr) - _nmn(gt_n_xr),
            "clean_gt_mean": _nmn(clean_preds_xr) - _nmn(gt_n_xr),
        }
        red = dict(zip(red, dask.compute(*red.values())))    # fused: each cube read once
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
    diff_gui_ung_gui_plot,
    diff_vfs_t_plot,
    dist_bands_checkbox,
    grad_norms_n_plot,
    grad_norms_plot,
    gui_gt_diff_n_plot,
    gui_gt_diff_t_plot,
    gui_vf_norms_n_plot,
    guidance_convergence_plot,
    guidance_convergence_t_plot,
    guided_vf_norms_plot,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    sweep_params_widget,
    t_slider,
    vf_deflection_n_plot,
    vf_deflection_t_plot,
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
                            ("gt_diffs", [gui_gt_diff_n_plot, gui_gt_diff_t_plot]),
                            ("diffs", [diff_gui_ung_gui_plot, diff_vfs_t_plot]),
                            ("grad_norms", [grad_norms_n_plot, grad_norms_plot]),
                            ("gui_vf_norms", [gui_vf_norms_n_plot, guided_vf_norms_plot]),
                            ("vf_norms", [vf_norms_n_plot, vf_norms_plot]),
                            ("deflections", [vf_deflection_n_plot, vf_deflection_t_plot]),
                            ("convergence", [guidance_convergence_plot, guidance_convergence_t_plot]),
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
def _(color_for, cross_ctl, cross_traces, m, n, notebook_mode, red, t):
    if notebook_mode =="analyze_rollout":
        import importlib as _importlib
        import src.ui.plot_trajectory as _ptmod
        _importlib.reload(_ptmod)
        _traces, _bands = cross_traces(red["grads_l2"].isel(n=n-1), "t", "l2", m, **cross_ctl)
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _traces.values()), default=1)))
        grad_norms_plot = _ptmod.plot_trajectory(_traces, title="Grad norms over $t$",
            subtitle=r"$\|\nabla_{z_t} \mathcal{L}_t\|_{\mathrm{spatial}}$", step=t + 1, color_map=color_for(_traces), bands=_bands,
            figsize=(_w, 6), prepend_zero=True, mirror_right_axis=True,
        )
        # grad_norms_plot
    return (grad_norms_plot,)


@app.cell
def _(color_for, cross_ctl, cross_traces, m, n, notebook_mode, red):
    if notebook_mode =="analyze_rollout":
        import importlib as _importlib
        import src.ui.plot_trajectory as _ptmod
        _importlib.reload(_ptmod)
        _raw, _bands = cross_traces(red["gui_ung_gui_mean"], "n", "mean", m, **cross_ctl)
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _raw.values()), default=1)))
        diff_gui_ung_gui_plot = _ptmod.plot_trajectory(
            _raw,
            title="Diff (gui − ung_gui) over $n$",
            subtitle=r"$\mathrm{mean}_{\mathrm{spatial}}\,(\tilde{x}^{\,\mathrm{gui}}_{n} - \tilde{x}^{\,\mathrm{ung\_gui}}_{n})$  ($\tilde{x}$: normalized $x$)",
            xlabel="$n$",
            step=n+1,
            color_map=color_for(_raw),
            bands=_bands,
            figsize=(_w, 6),
            prepend_zero=True,
            mirror_right_axis=True,
        )
        # diff_gui_ung_gui_plot
    return (diff_gui_ung_gui_plot,)


@app.cell
def _(color_for, cross_ctl, cross_traces, m, n, notebook_mode, red, t):
    if notebook_mode == "analyze_rollout":
        # reload so the absolute-legend-margin change in plot_trajectory takes effect
        # live (the kernel imported the old module at startup); local alias keeps it
        # contained to these cross-check plots.
        import importlib as _importlib
        import src.ui.plot_trajectory as _ptmod
        _importlib.reload(_ptmod)
        _plot_trajectory = _ptmod.plot_trajectory

        def _plot(title, subtitle, ds, axis, agg):
            _tr, _bands = cross_traces(ds, axis, agg, m, **cross_ctl)
            # width scales with the number of steps so the sparse "over n" plots
            # (N points) are not stretched across the same width as "over t" (T points)
            _nsteps = max((len(_v) for _v in _tr.values()), default=1)
            _w = min(22.0, max(8.0, 3.4 + 0.78 * _nsteps))
            return _plot_trajectory(
                _tr, title=title, subtitle=subtitle, xlabel=f"${axis}$",
                step=(n + 1 if axis == "n" else t + 1), color_map=color_for(_tr), bands=_bands,
                figsize=(_w, 6), prepend_zero=True, mirror_right_axis=True,
            )

        gui_gt_diff_n_plot = _plot("Diff (gui − gt) over $n$", r"$\mathrm{mean}_{\mathrm{spatial}}\,(\tilde{x}^{\,\mathrm{gui}}_{n} - \tilde{x}^{\,\mathrm{gt}}_{n})$  ($\tilde{x}$: normalized $x$)", red["gui_gt_mean"], "n", "mean")
        gui_gt_diff_t_plot = _plot("Diff (gui − gt) over $t$", r"$\mathrm{mean}_{\mathrm{spatial}}\,(\tilde{x}^{\,\mathrm{gui}}_{t} - \tilde{x}^{\,\mathrm{gt}}_{n})$", red["clean_gt_mean"].isel(n=n-1), "t", "mean")
        grad_norms_n_plot = _plot("Grad norms over $n$", r"$\|\nabla_{z_t} \mathcal{L}_t\|_{\mathrm{spatial},\,t}$", red["grads_l2"], "n", "l2")
        vf_deflection_n_plot = _plot("Vf deflection (gui − ung) over $n$", r"$\|\mathrm{vf}^{\mathrm{gui}} - \mathrm{vf}\|_{\mathrm{spatial},\,t}$", red["dvf_l2"], "n", "l2")
        vf_deflection_t_plot = _plot("Vf deflection (gui − ung) over $t$", r"$\|\mathrm{vf}^{\mathrm{gui}}_t - \mathrm{vf}_t\|_{\mathrm{spatial}}$", red["dvf_l2"].isel(n=n-1), "t", "l2")
        gui_vf_norms_n_plot = _plot("Gui vf norms over $n$", r"$\|\mathrm{vf}^{\mathrm{gui}}\|_{\mathrm{spatial},\,t}$", red["gui_vfs_l2"], "n", "l2")
        vf_norms_n_plot = _plot("Vf norms over $n$", r"$\|\mathrm{vf}\|_{\mathrm{spatial},\,t}$", red["vfs_l2"], "n", "l2")
        diff_vfs_t_plot = _plot("Diff (gui vf − ung vf) over $t$", r"$\mathrm{mean}_{\mathrm{spatial}}\,(\mathrm{vf}^{\mathrm{gui}}_t - \mathrm{vf}_t)$", red["dvf_mean"].isel(n=n-1), "t", "mean")
    else:
        grad_norms_n_plot = vf_deflection_n_plot = vf_deflection_t_plot = diff_vfs_t_plot = gui_vf_norms_n_plot = vf_norms_n_plot = gui_gt_diff_n_plot = gui_gt_diff_t_plot = None
    return (
        diff_vfs_t_plot,
        grad_norms_n_plot,
        gui_gt_diff_n_plot,
        gui_gt_diff_t_plot,
        gui_vf_norms_n_plot,
        vf_deflection_n_plot,
        vf_deflection_t_plot,
        vf_norms_n_plot,
    )


@app.cell
def _(
    cfg_clean_preds_slices,
    cfg_target_guidance_M_N_trajectories,
    delta_trajectory,
    dist_bands_checkbox,
    get_masked_mean,
    m,
    mask,
    n,
    notebook_mode,
    np,
    t,
):
    if notebook_mode =="analyze_rollout":
        import importlib as _importlib
        import src.ui.plot_trajectory as _ptmod
        _importlib.reload(_ptmod)
        _all_per_n = get_masked_mean(cfg_clean_preds_slices[:, :, -1], mask).astype(float) - cfg_target_guidance_M_N_trajectories
        _all_per_n[:, 0] = 0.0
        _diff_per_n = _all_per_n[m]
        _wn = min(22.0, max(8.0, 3.4 + 0.78 * len(_diff_per_n)))
        guidance_convergence_plot = _ptmod.plot_trajectory(
            {"realized − target": _diff_per_n},
            bands={"realized − target": (_all_per_n.min(axis=0), _all_per_n.max(axis=0))} if dist_bands_checkbox.value else None,
            title="Guidance convergence over $n$",
            subtitle=r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_{n,\,t=T}) - \mathrm{target}_n$",
            xlabel="$n$",
            step=n+1,
            color_map={"realized − target": "#B7950B"},
            right_trajectory=delta_trajectory,
            right_label=r"$\delta_n$",
            right_color="#8A2BE2",
            right_percentage=True,
            figsize=(_wn, 6),
            prepend_zero=True,
        )
        _all_per_t = get_masked_mean(cfg_clean_preds_slices[:, n], mask).astype(float) - cfg_target_guidance_M_N_trajectories[:, n][:, None]
        _diff_per_t = _all_per_t[m]
        _delta_diff_t = np.concatenate([[0.0], np.diff(_diff_per_t)])
        _wt = min(22.0, max(8.0, 3.4 + 0.78 * len(_diff_per_t)))
        guidance_convergence_t_plot = _ptmod.plot_trajectory(
            {"realized − target": _diff_per_t},
            bands={"realized − target": (_all_per_t.min(axis=0), _all_per_t.max(axis=0))} if dist_bands_checkbox.value else None,
            title="Guidance convergence over $t$",
            subtitle=r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_{n,\,t}) - \mathrm{target}_n$",
            xlabel="$t$",
            step=t + 1,
            color_map={"realized − target": "#B7950B"},
            right_trajectory={
                r"$\Delta$(realized $-$ target)": _delta_diff_t,
            },
            right_color={
                r"$\Delta$(realized $-$ target)": "#2E86C1",
            },
            figsize=(_wt, 6),
            prepend_zero=True,
        )
    return guidance_convergence_plot, guidance_convergence_t_plot


@app.cell
def _(mo):
    mo.md(r"""
    ## Flow analysis
    """)
    return


@app.cell
def _(color_for, cross_ctl, cross_traces, m, n, notebook_mode, red, t):
    if notebook_mode == "analyze_rollout":
        import importlib as _importlib
        import src.ui.plot_trajectory as _ptmod
        _importlib.reload(_ptmod)
        _vf_traces, _bands = cross_traces(red["vfs_l2"].isel(n=n-1), "t", "l2", m, **cross_ctl)
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _vf_traces.values()), default=1)))
        vf_norms_plot = _ptmod.plot_trajectory(_vf_traces, title="Vf norms over $t$",
            subtitle=r"$\|\mathrm{vf}_t\|_{\mathrm{spatial}}$", step=t + 1, color_map=color_for(_vf_traces), bands=_bands,
            figsize=(_w, 6), prepend_zero=True, mirror_right_axis=True)
    else:
        vf_norms_plot = None
    return (vf_norms_plot,)


@app.cell
def _(color_for, cross_ctl, cross_traces, m, n, notebook_mode, red, t):
    if notebook_mode == "analyze_rollout":
        import importlib as _importlib
        import src.ui.plot_trajectory as _ptmod
        _importlib.reload(_ptmod)
        _gvf_traces, _bands = cross_traces(red["gui_vfs_l2"].isel(n=n-1), "t", "l2", m, **cross_ctl)
        _w = min(22.0, max(8.0, 3.4 + 0.78 * max((len(_v) for _v in _gvf_traces.values()), default=1)))
        guided_vf_norms_plot = _ptmod.plot_trajectory(_gvf_traces, title="Gui vf norms over $t$",
            subtitle=r"$\|\mathrm{vf}^{\mathrm{gui}}_t\|_{\mathrm{spatial}}$", step=t + 1, color_map=color_for(_gvf_traces), bands=_bands,
            figsize=(_w, 6), prepend_zero=True, mirror_right_axis=True)
    else:
        guided_vf_norms_plot = None
    return (guided_vf_norms_plot,)


@app.cell
def _(grads_xr, mo, notebook_mode):
    t_slider = mo.ui.slider(
        steps=range(len(grads_xr.t)) if notebook_mode == "analyze_rollout" else range(25),
        value=0,
        label="t: ",
        debounce=True,
        show_value=True
    )
    return (t_slider,)


@app.cell
def _(guidance_mode_dropdown, mo, notebook_mode, w_slider):
    match notebook_mode:
        case "unguided_rollout":
            flow_schedule_widget = None
        case "guided_rollout":
            if guidance_mode_dropdown.value in ("FG", "FGF"):
                # lambda is learned (FG) or unused (FGF) -> the guidance-% knob does not apply.
                flow_schedule_widget = mo.md(
                    "_$\\lambda_t$ is **learned** by FlowGrad (FG) or not used "
                    "(FGF); the guidance-% knob $W$ does not apply to these modes._"
                )
            else:
                flow_schedule_widget = w_slider
        case "analyze_rollout":
            flow_schedule_widget = None
    flow_schedule_widget
    return


@app.cell
def _(
    get_rollout,
    guidance_mode_dropdown,
    notebook_mode,
    rollout_id,
    sweep_params,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        # Keep the heavy per-flow-step cubes LAZY: only `.sel` the sweep point here.
        # The `red` cell reduces them and computes the tiny result, and the map cells
        # materialize a single var/level via get_slices -> neither ever holds the full
        # (m, n, t, level, lat, lon) cube in RAM.
        grads_xr = get_rollout("grads", rollout_id).sel(sweep_params)
        vfs_xr = get_rollout("vfs", rollout_id).sel(sweep_params)
        clean_preds_xr = get_rollout("clean_preds", rollout_id).sel(sweep_params)
        gui_vfs_xr = get_rollout("gui_vfs", rollout_id).sel(sweep_params)
        ung_gui_xr = get_rollout("ung_gui", rollout_id).sel(sweep_params)
        # FLOWGRAD_FREE has no guidance direction -> these trace containers are all-NaN.
        # Fill with 0 so the (not-applicable) trace plots render as flat zero instead of
        # crashing on NaN axis limits; the learned controls live in the FlowGrad
        # diagnostics panel. Other modes are untouched.
        if guidance_mode_dropdown.value == "FGF":
            grads_xr = grads_xr.fillna(0.0)
            vfs_xr = vfs_xr.fillna(0.0)
            clean_preds_xr = clean_preds_xr.fillna(0.0)
            gui_vfs_xr = gui_vfs_xr.fillna(0.0)
    return clean_preds_xr, grads_xr, gui_vfs_xr, ung_gui_xr, vfs_xr


@app.cell
def _(
    clean_preds_xr,
    get_slices,
    grads_xr,
    gt_curr,
    gui_vfs_xr,
    level,
    m,
    n,
    notebook_mode,
    partition,
    t,
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
        ung_onl_clean_diff_slice = clean_preds_slices[m][n][t]- ung_onl_curr
        clean_preds_slice_prev = clean_preds_slices[m][n][t-1] if t>0 else clean_preds_slices[m][n][t]
        clean_preds_diff_slice = clean_preds_slices[m][n][t] - clean_preds_slice_prev
        # 3
        grads_slice = grads_slices[m][n][t]
        grads_slice_prev_slice = grads_slices[m][n][t-1] if t>0 else grads_slices[m][n][t]
        diff_grads_slice = grads_slice- grads_slice_prev_slice
        # 4
        guided_vfs_slice = guided_vfs_slices[m][n][t]
        vfs_slice = vfs_slices[m][n][t]
        vf_gui_prev_diff_slice = vfs_slice - (guided_vfs_slices[m][n][t-1] if t>0 else guided_vfs_slices[m][n][t])
        # velocity-field evolution between consecutive flow steps (base vs guided)
        vf_prev_diff_slice = vfs_slice - (vfs_slices[m][n][t-1] if t>0 else vfs_slices[m][n][t])
        guided_vf_prev_diff_slice = guided_vfs_slice - (guided_vfs_slices[m][n][t-1] if t>0 else guided_vfs_slices[m][n][t])
    return (
        clean_preds_diff_slice,
        clean_preds_slices,
        diff_grads_slice,
        diff_gt_clean_pred_slice,
        diff_gt_ung_onl_slice,
        grads_slice,
        guided_vf_prev_diff_slice,
        guided_vfs_slice,
        ung_onl_clean_diff_slice,
        vf_gui_prev_diff_slice,
        vf_prev_diff_slice,
        vfs_slice,
    )


@app.cell
def _(t_slider):
    t=t_slider.value
    return (t,)


@app.cell
def _(
    clean_preds_diff_slice,
    clean_preds_slices,
    diff_grads_slice,
    diff_gt_clean_pred_slice,
    diff_gt_ung_onl_slice,
    dpi_slider,
    grads_slice,
    gt_curr,
    guided_vf_prev_diff_slice,
    guided_vfs_slice,
    m,
    mask,
    n,
    notebook_mode,
    np,
    show_mask_switch,
    t,
    ung_curr,
    ung_onl_clean_diff_slice,
    ung_onl_curr,
    vf_gui_prev_diff_slice,
    vf_prev_diff_slice,
    vfs_slice,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        diff_vfs_slice = guided_vfs_slice - vfs_slice

        map_specs = [
            ("gt_state_map", gt_curr, r"$x_{n}^{\text{gt}}$", -1, 1),
            ("ung_state_map", ung_curr, r"$x_{n}^{\text{ung}}$", -1, 1),
            ("ung_onl_map", ung_onl_curr, r"$x_{n}^{\text{ung_gui}}$", -1, 1),
            ("clean_pred_map", clean_preds_slices[m][n][t], r"$x_t^{\text{gui}}$", -1, 1),
            ("diff_gt_ung_onl_map", diff_gt_ung_onl_slice, r"$x_{n}^{\text{ung_gui}} - x_{n}^{\text{gt}}$", -1, 1),
            ("diff_gt_clean_pred_map", diff_gt_clean_pred_slice, r"$x_t^{\text{gui}} - x_{n}^{\text{gt}}$", -1, 1),
            ("ung_onl_clean_diff_map", ung_onl_clean_diff_slice, r"$x_t^{\text{gui}} - x_{n}^{\text{ung_gui}}$", -1, 1),
            ("clean_preds_diff_map", clean_preds_diff_slice, r"$x_t^{\text{gui}} - x_{t-1}^{\text{gui}}$", -1, 1),
            ("grads_map", grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t$", -1, 1),
            ("vfs_map", vfs_slice, r"$\text{vf}_t$", -0.001, 0.001),
            ("guided_vfs_map", guided_vfs_slice, r"$\text{vf}^{\text{gui}}_t$", -0.001, 0.001),
            ("diff_vfs_map", diff_vfs_slice, r"$\text{vf}^{\text{gui}}_t - \text{vf}_t$", -0.001, 0.001),
            ("vf_gui_prev_diff_map", vf_gui_prev_diff_slice, r"$\text{vf}_t - \text{vf}^{\text{gui}}_{t-1}$", -0.001, 0.001),
            ("vf_prev_diff_map", vf_prev_diff_slice, r"$\text{vf}_t - \text{vf}_{t-1}$", -0.001, 0.001),
            ("guided_vf_prev_diff_map", guided_vf_prev_diff_slice, r"$\text{vf}^{\text{gui}}_t - \text{vf}^{\text{gui}}_{t-1}$", -0.001, 0.001),
            ("diff_grads_map", diff_grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t - \\nabla_{z_{t-1}} \\mathcal{L}_{t-1}$", -1, 1),
        ]

        maps = {}

        for name, data, title, fallback_vmin, fallback_vmax in map_specs:
            data_min = np.min(data)
            data_max = np.max(data)
            data_mean = np.mean(data)
            print(name, data_min, data_mean, data_max)

            maps[name] = visualize_map(
                data,
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
            )

        gt_state_map = maps["gt_state_map"]
        ung_state_map = maps["ung_state_map"]
        ung_onl_map = maps["ung_onl_map"]
        clean_pred_map = maps["clean_pred_map"]
        diff_gt_ung_onl_map = maps["diff_gt_ung_onl_map"]
        diff_gt_clean_pred_map = maps["diff_gt_clean_pred_map"]
        ung_onl_clean_diff_map = maps["ung_onl_clean_diff_map"]
        clean_preds_diff_map = maps["clean_preds_diff_map"]
        grads_map = maps["grads_map"]
        vfs_map = maps["vfs_map"]
        guided_vfs_map = maps["guided_vfs_map"]
        diff_vfs_map = maps["diff_vfs_map"]
        vf_gui_prev_diff_map = maps["vf_gui_prev_diff_map"]
        diff_grads_map = maps["diff_grads_map"]
        vf_prev_diff_map = maps["vf_prev_diff_map"]
        guided_vf_prev_diff_map = maps["guided_vf_prev_diff_map"]
    return (
        clean_pred_map,
        diff_grads_map,
        diff_gt_clean_pred_map,
        diff_vfs_map,
        grads_map,
        gt_state_map,
        guided_vf_prev_diff_map,
        guided_vfs_map,
        ung_onl_clean_diff_map,
        ung_onl_map,
        ung_state_map,
        vf_gui_prev_diff_map,
        vf_prev_diff_map,
        vfs_map,
    )


@app.cell
def _(mo):
    flow_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "gt & ung states", "gui states", "gui_t diffs", "grads", "vfs", "vf diffs", "vf evolution",
    )})
    return (flow_checks,)


@app.cell(hide_code=True)
def _(mo):
    # row-toggle checkboxes for the Inspect states maps (analyze mode), mirroring
    # flow_checks / cross_row_checks. Keys cover the absolute + difference rows.
    inspect_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "curr / prev", "gui / ung",
        "gt_gt / gui_gui", "gui_gt / gui_ung", "ung_gui diffs",
    )})
    return (inspect_checks,)


@app.cell
def _(
    clean_pred_map,
    diff_grads_map,
    diff_gt_clean_pred_map,
    diff_vfs_map,
    dpi_slider,
    flow_checks,
    grads_map,
    gt_state_map,
    guided_vf_prev_diff_map,
    guided_vfs_map,
    level_slider,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    partition_dropdown,
    show_mask_switch,
    sweep_params_widget,
    t_slider,
    ung_onl_clean_diff_map,
    ung_onl_map,
    ung_state_map,
    var_dropdown,
    vf_gui_prev_diff_map,
    vf_prev_diff_map,
    vfs_map,
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
        flow_controls = mo.hstack(
            [
                mo.vstack(
                    [
                        mo.hstack([t_slider, dpi_slider], justify="start", align="start"),
                        mo.hstack([m_slider, n_slider], justify="start", align="start"),
                    ],
                    align="start",
                ),
                mo.vstack(
                    [
                        var_controls,
                        mo.hstack([show_mask_switch, zoom_slider], justify="start", align="start"),
                    ],
                    align="start",
                ),
            ],
            justify="start",
            align="start",
        )

        map_rows = [
            ("gt & ung states", [gt_state_map, ung_state_map]),
            ("gui states", [clean_pred_map, ung_onl_map]),
            ("gui_t diffs", [ung_onl_clean_diff_map, diff_gt_clean_pred_map]),
            ("grads", [grads_map, diff_grads_map]),
            ("vfs", [vfs_map, guided_vfs_map]),
            ("vf diffs", [vf_gui_prev_diff_map, diff_vfs_map]),
            ("vf evolution", [vf_prev_diff_map, guided_vf_prev_diff_map]),
        ]

        flow_widget_make = mo.vstack(
            [
                sweep_params_widget,
                mo.hstack(
                    [mo.vstack(list(flow_checks.values())), flow_controls],
                    justify="start", align="start",
                ),
                mo.vstack(
                    [mo.hstack(maps, justify="start", align="start") for n, maps in map_rows if flow_checks[n].value],
                    align="start",
                ),
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
def _(
    guidance_mode_dropdown,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    plt,
    rollout_id,
    sweep_params,
    sweep_params_widget,
):
    # FlowGrad learned diagnostics (analyze mode): learned lambda*_t / control-norm per
    # window + the optimization-loss curve. These come from the JSON sidecar written by
    # the rollout (FlowGrad learns its schedule/controls, so they have no zarr container).
    if notebook_mode == "analyze_rollout" and guidance_mode_dropdown.value in ("FG", "FGF"):
        # plt is a notebook-global (imported elsewhere); import get_diagnostics under a
        # private alias so this cell doesn't introduce a multiply-defined top-level name.
        from src.utils import get_diagnostics as _get_diagnostics

        _gm = guidance_mode_dropdown.value
        _mi, _ni = int(m_slider.value), int(n_slider.value)
        _records = _get_diagnostics(rollout_id)

        _recs = [
            r for r in _records
            if r["guidance_mode"] == _gm and r["m"] == _mi and r["n"] == _ni and r["sweep"] == sweep_params
        ]
        if not _recs:  # fall back: match on (mode, m, n) only if the sweep dict didn't line up
            _recs = [r for r in _records if r["guidance_mode"] == _gm and r["m"] == _mi and r["n"] == _ni]

        if not _recs:
            flowgrad_diag_widget = mo.md(
                f"_No FlowGrad diagnostics for {_gm}, m={_mi}, n={_ni}. "
                f"(Sidecar has {len(_records)} record(s).)_"
            )
        else:
            _r = _recs[0]
            _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
            _axes[0].plot(_r["opt_target_loss"], marker="o", label="target loss")
            _axes[0].plot(_r["opt_loss"], marker=".", label="total loss (+reg)")
            _axes[0].set_title(f"{_gm} optimization (m={_mi}, n={_ni})")
            _axes[0].set_xlabel("iteration"); _axes[0].set_ylabel("loss"); _axes[0].legend()

            if _r.get("lambda_star") is not None:
                _axes[1].plot(_r["lambda_star"], marker="o")
                _axes[1].set_title(r"learned $\lambda^\star_t$ (coarse, expanded to T)")
                _axes[1].set_xlabel("flow step $t$"); _axes[1].set_ylabel(r"$\lambda$")
            elif _r.get("control_norm") is not None:
                _axes[1].bar(range(len(_r["control_norm"])), _r["control_norm"])
                _axes[1].set_title("learned control norm per window")
                _axes[1].set_xlabel("control window"); _axes[1].set_ylabel(r"$\|u_j\|$")
            _fig.tight_layout()
            flowgrad_diag_widget = mo.as_html(_fig)
            plt.close(_fig)
    else:
        flowgrad_diag_widget = mo.md("_FlowGrad diagnostics apply only to FLOWGRAD / FLOWGRAD_FREE._")

    if notebook_mode == "analyze_rollout":
        flowgrad_diag_widget = mo.vstack([sweep_params_widget, flowgrad_diag_widget], align="start")
    flowgrad_diag_widget
    return


@app.cell(hide_code=True)
def _(
    N,
    T,
    clean_preds_slices,
    config,
    gt_N_slices,
    gt_curr,
    gt_rollout,
    gui_M_N_slices,
    gui_curr,
    level,
    m,
    m_slider,
    mask_widget_controls,
    mo,
    n,
    n_slider,
    notebook_mode,
    np,
    partition,
    plt,
    sweep_params_widget,
    t,
    t_slider,
    ung_M_N_slices,
    ung_curr,
    ung_gui_M_N_slices,
    ung_onl_curr,
    var,
):
    # ===== Physical realism: power spectra + spectral distance + activity, over n and over t =====
    if notebook_mode == "analyze_rollout":
        from src.spectrum import (
            power_spectrum as _ps, log_spectral_distance as _lsd,
            activity as _activity, climatology_slice as _climslice,
        )
        from pathlib import Path as _Path
        from datetime import timedelta as _timedelta
        from src.paths import CLIM
        import xarray as _xr

        _lat = gt_rollout.latitude.values
        _styles = {"gt": ("-", 2.4), "ung": ("--", 1.6), "ung_gui": (":", 2.0), "gui": ("-.", 1.6)}
        _hdr = f"{var} @ lvl {level} (m={m}, n={n})"

        _clim = _xr.open_dataset(CLIM) if CLIM.exists() else None
        _clim_level = level if partition == "level" else None
        def _clim_at(_ni):
            return (_climslice(_clim, var, config.START_TS + _timedelta(days=_ni), _lat, level=_clim_level)
                    if _clim is not None else None)
        _act_src = "ERA5 clim" if _clim is not None else "zonal proxy"

        _deg = _ps(np.asarray(gt_N_slices[0]), _lat)[0]
        _gt_spec = {_ni: _ps(np.asarray(gt_N_slices[_ni]), _lat)[1] for _ni in range(N)}
        _ns, _ts = list(range(N)), list(range(T))

        def _html(_fig):
            _fig.tight_layout(); _h = mo.as_html(_fig); plt.close(_fig); return _h

        # ---------- power spectrum ratio to gt: @ step n  |  @ flow step t ----------
        _fig, _ax = plt.subplots(figsize=(8, 4.5))
        _refn = _gt_spec[n]
        for _k, _f in {"gt": gt_curr, "ung": ung_curr, "ung_gui": ung_onl_curr, "gui": gui_curr}.items():
            _ls, _lw = _styles[_k]
            _ax.loglog(_deg[1:], _ps(np.asarray(_f), _lat)[1][1:] / _refn[1:], _ls, lw=_lw, alpha=0.8, label=_k)
        _ax.axhline(1.0, color="grey", lw=0.8, alpha=0.5)
        _ax.set_xlabel(r"degree $l$"); _ax.set_ylabel("power / gt"); _ax.set_title(f"Spectrum ratio @ step n - {_hdr}")
        _ax.legend(); _ax.grid(True, which="both", alpha=0.3)
        _spec_n = _html(_fig)

        _fig, _ax = plt.subplots(figsize=(8, 4.5))
        _ax.loglog(_deg[1:], _ps(np.asarray(gui_curr), _lat)[1][1:] / _refn[1:], "-.", lw=1.6, alpha=0.7, label="gui (final)")
        _ax.loglog(_deg[1:], _ps(np.asarray(clean_preds_slices[m][n][t]), _lat)[1][1:] / _refn[1:],
                   "-", lw=2.0, color="crimson", alpha=0.9, label=f"gui clean pred @ t={t}")
        _ax.axhline(1.0, color="grey", lw=0.8, alpha=0.5)
        _ax.set_xlabel(r"degree $l$"); _ax.set_ylabel("power / gt"); _ax.set_title(f"Spectrum ratio @ flow t={t} - {_hdr}")
        _ax.legend(); _ax.grid(True, which="both", alpha=0.3)
        _spec_t = _html(_fig)

        # ---------- spectral distance (LSD) to gt: over n  |  over t ----------
        _fig, _ax = plt.subplots(figsize=(8, 4))
        for _k, _sl in {"ung": ung_M_N_slices, "ung_gui": ung_gui_M_N_slices, "gui": gui_M_N_slices}.items():
            _ls, _lw = _styles[_k]
            _ax.plot(_ns, [_lsd(_ps(np.asarray(_sl[m][_ni]), _lat)[1], _gt_spec[_ni]) for _ni in _ns], _ls, lw=_lw, marker="o", label=_k)
        _ax.set_xlabel("rollout step $n$"); _ax.set_ylabel("LSD vs gt"); _ax.set_title(f"Spectral distance over n - {_hdr}")
        _ax.set_xticks(_ns); _ax.legend(); _ax.grid(True, alpha=0.3)
        _lsd_n = _html(_fig)

        _fig, _ax = plt.subplots(figsize=(8, 4))
        _ax.plot(_ts, [_lsd(_ps(np.asarray(clean_preds_slices[m][n][_ti]), _lat)[1], _gt_spec[n]) for _ti in _ts],
                 "-", color="crimson", marker=".", label="gui clean pred")
        _ax.set_xlabel("flow step $t$"); _ax.set_ylabel("LSD vs gt"); _ax.set_title(f"Spectral distance over t - {_hdr}")
        _ax.legend(); _ax.grid(True, alpha=0.3)
        _lsd_t = _html(_fig)

        # ---------- activity (eddy spatial std): over n  |  over t ----------
        _clim_n = _clim_at(n)
        _fig, _ax = plt.subplots(figsize=(8, 4))
        for _k in ["gt", "ung", "ung_gui", "gui"]:
            _ls, _lw = _styles[_k]
            _vals = []
            for _ni in _ns:
                _f = gt_N_slices[_ni] if _k == "gt" else {"ung": ung_M_N_slices, "ung_gui": ung_gui_M_N_slices, "gui": gui_M_N_slices}[_k][m][_ni]
                _vals.append(_activity(np.asarray(_f), _lat, climatology=_clim_at(_ni)))
            _ax.plot(_ns, _vals, _ls, lw=_lw, marker="o", label=_k)
        _ax.set_xlabel("rollout step $n$"); _ax.set_ylabel("activity"); _ax.set_title(f"Activity over n [{_act_src}] - {_hdr}")
        _ax.set_xticks(_ns); _ax.legend(); _ax.grid(True, alpha=0.3)
        _act_n = _html(_fig)

        _fig, _ax = plt.subplots(figsize=(8, 4))
        _ax.plot(_ts, [_activity(np.asarray(clean_preds_slices[m][n][_ti]), _lat, climatology=_clim_n) for _ti in _ts],
                 "-", color="crimson", marker=".", label="gui clean pred")
        _ax.axhline(_activity(np.asarray(gt_N_slices[n]), _lat, climatology=_clim_n), color="black", ls="--", lw=1.2, label="gt")
        _ax.set_xlabel("flow step $t$"); _ax.set_ylabel("activity"); _ax.set_title(f"Activity over t [{_act_src}] - {_hdr}")
        _ax.legend(); _ax.grid(True, alpha=0.3)
        _act_t = _html(_fig)

        power_spectrum_widget = mo.vstack([
            mo.md("## Physical realism"),
            sweep_params_widget,
            mask_widget_controls,
            mo.hstack([m_slider, n_slider, t_slider], justify="start"),
            mo.hstack([_spec_n, _spec_t], justify="start"),
            mo.hstack([_lsd_n, _lsd_t], justify="start"),
            mo.hstack([_act_n, _act_t], justify="start"),
        ], align="start")
    else:
        power_spectrum_widget = None

    power_spectrum_widget
    return


if __name__ == "__main__":
    app.run()
