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
    from src.rollout_config import GUIDANCE_REFERENCES, MASK_MODES, RolloutConfig, GUIDANCE_MODES
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
    from src.schedules import N_schedule, delta_schedule, T_schedule

    from src.mask import get_masked_mean, get_mask_2d, get_mu_sigma, get_mask_center
    from src.target import get_target_slices

    return (
        GUIDANCE_MODES,
        GUIDANCE_REFERENCES,
        LEVELS_DICT,
        MASK_MODES,
        PARTITIONS,
        RolloutConfig,
        T_schedule,
        VARIABLES_DICT,
        delta_schedule,
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
def _(save_config_button):
    save_config_button
    return


@app.cell
def _(
    M,
    N,
    RolloutConfig,
    alpha,
    beta_slider,
    delta_trajectory,
    dump_json,
    ensure_rollout_dir,
    fg_gamma_slider,
    fg_init_lambda_slider,
    fg_lr_slider,
    fg_n_lambda_slider,
    fg_n_opt_slider,
    get_now_timestamp,
    guidance_mode_dropdown,
    guidance_reference,
    lbg_n_mc_slider,
    lbg_r_t_slider,
    level,
    mask_corners,
    mask_mode,
    normalize_checkbox,
    notebook_mode,
    partition,
    regularized_checkbox,
    rollout_id,
    save_config_button,
    timestamp,
    ug_S_slider,
    ug_delta_lr_slider,
    ug_m_slider,
    var,
    w,
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
                timestamp=timestamp,  
                # experiment level params
                level=level,
                partition=partition,
                var=var,
                mask_corners=mask_corners
            )
            dump_json(save_config.to_dict(), path)
        if save_config_button.value and notebook_mode == "guided_rollout":
            save_id = rollout_id
            rollout_dir = ensure_rollout_dir(save_id)
            path = rollout_dir / "config.json"
            # TODO: these are only placeholders
            #       implement version with lists?
            save_config = RolloutConfig(
                # common to guided and unguided
                M=M,
                N=N,
                timestamp=timestamp,  
                # experiment level params
                level=level,
                partition=partition,
                var=var,
                mask_corners=mask_corners,
            )
            dump_json(save_config.to_dict(), path)
            print(save_config.to_dict_list())
            path = rollout_dir / "sweep_params.json"
            save_config = RolloutConfig(
                # guided rollout specific params -> can be swept
                delta_trajectory=delta_trajectory,
                mask_mode=mask_mode,
                guidance_mode=guidance_mode_dropdown.value,
                guidance_reference=guidance_reference,
                alpha=alpha,
                w=w,
                # guidance hyperparameters
                regularized=regularized_checkbox.value,
                normalize=normalize_checkbox.value,
                beta=beta_slider.value,
                lbg_n_mc=lbg_n_mc_slider.value,
                lbg_r_t=lbg_r_t_slider.value,
                ug_S=ug_S_slider.value,
                ug_m=ug_m_slider.value,
                ug_delta_lr=ug_delta_lr_slider.value,
                fg_n_opt=fg_n_opt_slider.value,
                fg_lr=fg_lr_slider.value,
                fg_gamma=fg_gamma_slider.value,
                fg_init_lambda=fg_init_lambda_slider.value,
                fg_n_lambda=fg_n_lambda_slider.value,
            )
            dump_json(save_config.to_dict_list(), path)
    return


@app.cell
def _(
    GUIDANCE_REFERENCES,
    MASK_MODES,
    get_sweep_dict,
    guidance_mode_dropdown,
    mo,
    notebook_mode,
    rollout_id,
):
    w_defaults = [5, 10, 20, 50, 100]
    match notebook_mode:
        case "unguided_rollout":
            guidance_reference_dropdown = mo.ui.dropdown(
                GUIDANCE_REFERENCES, value=GUIDANCE_REFERENCES[0], label="guidance reference: "
            )
            alpha_defaults = [0, 1, 2]
            alpha_slider = mo.ui.slider(
                steps=alpha_defaults,
                value=alpha_defaults[-1],
                label="alpha: ",
                debounce=True,
                show_value=True
            )
            w_slider = mo.ui.slider(
                steps=w_defaults,
                value=w_defaults[0],
                label="w: ",
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            delta_trajectory_dropdown = None
            sweep_params_widget = None

        case "guided_rollout":
            guidance_reference_dropdown = mo.ui.dropdown(
                GUIDANCE_REFERENCES, value=GUIDANCE_REFERENCES[0], label="guidance reference: "
            )
            alpha_defaults = [0, 1, 2]
            alpha_slider = mo.ui.slider(
                steps=alpha_defaults,
                value=alpha_defaults[-1],
                label="alpha: ",
                debounce=True,
                show_value=True
            )
            w_slider = mo.ui.slider(
                steps=w_defaults,
                value=w_defaults[0],
                label="w: ",
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            delta_trajectory_dropdown = None
            sweep_params_widget = None

        case "analyze_rollout":
            experiment_params = get_sweep_dict(rollout_id)

            # TODO: decide between the two depending on notebook mode
            guidance_reference_dropdown = mo.ui.dropdown(
                options=experiment_params["guidance_reference"],
                value=experiment_params["guidance_reference"][0],
                label="guidance reference",
            )

            alpha_label = "alpha: " if len(experiment_params["alpha"])>1 else "alpha:  "
            alpha_slider = mo.ui.slider(
                steps=experiment_params["alpha"],
                value=experiment_params["alpha"][0],
                label=alpha_label,
                debounce=True,
                show_value=True
            )
            w_label = "w: " if len(experiment_params["w"])>1 else "w:  "
            w_slider = mo.ui.slider(
                steps=experiment_params["w"],
                value=experiment_params["w"][0],
                label=w_label,
                debounce=True,
                show_value=True
            )

            mask_mode_dropdown = mo.ui.dropdown(options=experiment_params["mask_mode"], value=experiment_params["mask_mode"][0], label="mask_mode: ")

            # delta_trajectory is a swept axis stored by integer index in the zarr;
            # the dropdown maps each candidate vector's label -> its index.
            _dt_candidates = experiment_params["delta_trajectory"]
            _dt_options = {str(v): i for i, v in enumerate(_dt_candidates)}
            dt_label = "delta_trajectory: " if len(_dt_candidates) > 1 else "delta_trajectory:  "
            delta_trajectory_dropdown = mo.ui.dropdown(
                options=_dt_options,
                value=next(iter(_dt_options)),
                label=dt_label,
            )

            sweep_params_widget = mo.vstack([
                guidance_reference_dropdown, mask_mode_dropdown,
                alpha_slider,
                w_slider,
                guidance_mode_dropdown,
                delta_trajectory_dropdown,
            ])

        case _:
            pass
    return (
        alpha_slider,
        delta_trajectory_dropdown,
        experiment_params,
        guidance_reference_dropdown,
        mask_mode_dropdown,
        sweep_params_widget,
        w_slider,
    )


@app.cell
def _(
    alpha_slider,
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
                "alpha": alpha_slider.value,
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
            timestamp=config.timestamp
            map_interactive = False
        case "analyze_rollout":
            guidance_flag=None
            M=config.M
            N=config.N
            year=None
            month=None
            day=None
            hour=None
            timestamp=config.timestamp
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
            mask_corners = config.mask_corners
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
        case _:
            pass
    return mask, mask_corners


@app.cell
def _(
    N,
    delta_mode_dropdown,
    delta_peak,
    delta_peak_at_slider,
    delta_schedule,
    delta_shape_slider,
    delta_start_slider,
    delta_trajectory_dropdown,
    experiment_params,
    notebook_mode,
    np,
    stop_at_n_checkbox,
    stop_at_n_slider,
):
    # delta schedule
    match notebook_mode:
        case "unguided_rollout":
            delta_trajectory=None
        case "guided_rollout":
            stop_n = stop_at_n_slider.value if stop_at_n_checkbox.value else None
            delta_trajectory = delta_schedule(
                N, delta_mode_dropdown.value, delta_peak,
                peak_at_n=delta_peak_at_slider.value, stop_at_n=stop_n,
                flatness=delta_shape_slider.value,
                start_value=delta_start_slider.value / 100,
            )[1:]
        case "analyze_rollout":
            # config.delta_trajectory is None under sweeping; the swept vectors live in
            # experiment_params["delta_trajectory"], picked by the dropdown's index.
            delta_trajectory = experiment_params["delta_trajectory"][delta_trajectory_dropdown.value]
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
                partition == config.partition
                and var == config.var
                and level == config.level
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
def _():
    return


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
        cfg_clean_preds_slices = get_slices(clean_preds_xr, config.partition, config.var, config.level)
        cfg_ung_gui_M_N_slices = get_slices(ung_gui_xr, config.partition, config.var, config.level)
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
        ung_gui_m_trajectory,
        ung_lb_trajectory,
        ung_m_trajectory,
        ung_ub_trajectory,
    )


@app.cell
def _(
    alpha_slider,
    delta_trajectory_dropdown,
    experiment_params,
    guidance_mode_dropdown,
    guidance_reference_dropdown,
    mask_mode_dropdown,
    notebook_mode,
    sweep_coord_label,
    w_slider,
):
    sweep_params = {
        "w": w_slider.value,
        "alpha": alpha_slider.value,
        "guidance_reference": guidance_reference_dropdown.value,
        "mask_mode": mask_mode_dropdown.value,
        "guidance_mode": guidance_mode_dropdown.value,
    }
    # delta_trajectory is an integer-indexed sweep axis in the zarr (real vectors live
    # in the delta_trajectory_value coord); the dropdown holds the chosen index.
    if delta_trajectory_dropdown is not None:
        sweep_params["delta_trajectory"] = delta_trajectory_dropdown.value

    # Every sweep_params key is now a zarr dimension, so .sel must reduce all of them
    # or singleton dims linger and break the 2D plotters. Widgets drive a handful; the
    # rest take their sole stored value (sweep_coord_label -> index for eps/non-scalars).
    if notebook_mode == "analyze_rollout":
        for _key, _values in experiment_params.items():
            if _key not in sweep_params:
                sweep_params[_key] = sweep_coord_label(_key, _values[0], experiment_params)
    return (sweep_params,)


@app.cell
def _(gt_trajectory, ung_lb_trajectory, ung_m_trajectory, ung_ub_trajectory):
    # connect ui to data
    reference_trajectories = {
        "unguided_members": ung_m_trajectory,
        "ground_truth": gt_trajectory,
        "lower_boundary": ung_lb_trajectory,
        "upper_boundary": ung_ub_trajectory
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
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label=level_label, show_value=True, debounce=True)
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
    delta_bounds_slider = mo.ui.slider(
        steps=[5, 10, 25, 50, 100], value=10, label="bounds (%): ", show_value=True
    )
    delta_granularity_slider = mo.ui.slider(
        steps=[0.1, 0.5, 1, 2, 5],
        value=1,
        label="granularity (%): ",
        show_value=True,
        debounce=True,
    )
    return delta_bounds_slider, delta_granularity_slider


@app.cell
def _(N, delta_bounds_slider, delta_granularity_slider, mo):
    delta_shape_slider = mo.ui.slider(
        start=0.5,
        stop=10.0,
        step=0.5,
        value=0.5,
        label="shape: ",
        show_value=True,
        debounce=True,
    )
    delta_peak_at_slider = mo.ui.slider(
        start=1,
        stop=N,
        step=1,
        value=N // 2 if N >1 else 1,
        label="delta_peak @ n: ",
        show_value=True,
        debounce=True,
    )
    delta_peak_slider = mo.ui.slider(
        -delta_bounds_slider.value,
        delta_bounds_slider.value,
        value=0,
        step=delta_granularity_slider.value,
        label="peak (%): ",
        show_value=True,
        debounce=True,
    )

    delta_start_slider = mo.ui.slider(
        -delta_bounds_slider.value,
        delta_bounds_slider.value,
        value=0,
        step=delta_granularity_slider.value,
        label="start@ (%): ",
        show_value=True,
        debounce=True,
    )

    delta_mode_dropdown = mo.ui.dropdown(["linear", "sinusoidal"], value="sinusoidal", label="delta_mode: ")
    stop_at_n_slider = mo.ui.slider(1, N, step=1, value=N, label="stop @ n: ", show_value=True, debounce=True)
    stop_at_n_checkbox = mo.ui.checkbox(label="stop at n", value=False)
    return (
        delta_mode_dropdown,
        delta_peak_at_slider,
        delta_peak_slider,
        delta_shape_slider,
        delta_start_slider,
        stop_at_n_checkbox,
        stop_at_n_slider,
    )


@app.cell
def _(delta_peak_slider):
    delta_peak = delta_peak_slider.value / 100
    return (delta_peak,)


@app.cell
def _(mask_mode_dropdown):
    mask_mode=mask_mode_dropdown.value
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
def _(T_slider):
    T=T_slider.value
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
def _(GUIDANCE_MODES, mo):
    guidance_mode_dropdown = mo.ui.dropdown(options=GUIDANCE_MODES, value=GUIDANCE_MODES[0], label="guidance_mode: ")
    return (guidance_mode_dropdown,)


@app.cell
def _(mo, notebook_mode):
    # guidance hyperparameters (guided_rollout) -> saved into config.json
    regularized_checkbox = mo.ui.checkbox(label="regularized", value=False)
    normalize_checkbox = mo.ui.checkbox(label="normalize", value=True)
    beta_slider = mo.ui.slider(steps=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, label="beta: ", show_value=True)
    lbg_n_mc_slider = mo.ui.slider(1, 16, step=1, value=4, label="lbg n_mc: ", show_value=True)
    lbg_r_t_slider = mo.ui.slider(steps=[0.0, 0.5, 1.0, 2.0], value=1.0, label="lbg r_t: ", show_value=True)
    ug_S_slider = mo.ui.slider(1, 8, step=1, value=4, label="ug S: ", show_value=True)
    ug_m_slider = mo.ui.slider(1, 20, step=1, value=5, label="ug m: ", show_value=True)
    ug_delta_lr_slider = mo.ui.slider(steps=[1e-2, 5e-2, 1e-1, 5e-1], value=1e-1, label="ug delta_lr: ", show_value=True)
    fg_n_opt_slider = mo.ui.slider(1, 50, step=1, value=10, label="fg n_opt: ", show_value=True)
    fg_lr_slider = mo.ui.slider(steps=[1e-3, 1e-2, 5e-2], value=1e-2, label="fg lr: ", show_value=True)
    fg_gamma_slider = mo.ui.slider(steps=[1e-4, 1e-3, 1e-2], value=1e-3, label="fg gamma: ", show_value=True)
    fg_init_lambda_slider = mo.ui.slider(steps=[0.0, 0.1, 0.5, 1.0, 2.0], value=0.0, label="fg init_lambda (lambda_0): ", show_value=True)
    fg_n_lambda_slider = mo.ui.slider(1, 25, step=1, value=5, label="fg n_lambda (K): ", show_value=True)

    hypers_widget = mo.vstack([
        mo.hstack([regularized_checkbox, normalize_checkbox, beta_slider], justify="start"),
        mo.hstack([lbg_n_mc_slider, lbg_r_t_slider], justify="start"),
        mo.hstack([ug_S_slider, ug_m_slider, ug_delta_lr_slider], justify="start"),
        mo.hstack([fg_n_opt_slider, fg_lr_slider, fg_gamma_slider, fg_init_lambda_slider, fg_n_lambda_slider], justify="start"),
    ], align="start")

    hypers_widget if notebook_mode == "guided_rollout" else None
    return (
        beta_slider,
        fg_gamma_slider,
        fg_init_lambda_slider,
        fg_lr_slider,
        fg_n_lambda_slider,
        fg_n_opt_slider,
        lbg_n_mc_slider,
        lbg_r_t_slider,
        normalize_checkbox,
        regularized_checkbox,
        ug_S_slider,
        ug_delta_lr_slider,
        ug_m_slider,
    )


@app.cell
def _(
    M_N_widget,
    delta_bounds_slider,
    delta_granularity_slider,
    delta_mode_dropdown,
    delta_peak_at_slider,
    delta_peak_slider,
    delta_shape_slider,
    delta_start_slider,
    guidance_mode_dropdown,
    guidance_reference_dropdown,
    inspect_states_widget_make,
    lambda_trajectory_plot,
    level_slider,
    m_n_widget,
    mask_map,
    mask_mode_dropdown,
    mo,
    notebook_mode,
    partition_dropdown,
    stop_at_n_checkbox,
    stop_at_n_slider,
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
    mask_widget = mo.vstack([
        guidance_mode_dropdown,
        mask_mode_dropdown,
        mask_widget_controls, mask_widget_maps], align="start")
    delta_widget = mo.hstack([
        delta_mode_dropdown,
        delta_start_slider,
        delta_peak_slider,
        delta_granularity_slider,
        delta_bounds_slider,
        delta_shape_slider,
        delta_peak_at_slider,
        stop_at_n_slider,
        stop_at_n_checkbox,
    ], justify="start")

    match notebook_mode:
        case "unguided_rollout":
            trajectory_widget=mo.vstack([
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
                mask_mode_dropdown,
                mask_widget_controls,
                m_n_widget,
                delta_widget,
                mo.hstack([mo.vstack(list(traj_checks.values())), trajectories_plot], justify="start", align="start")
            ], align="start")
            inspect_states_widget=inspect_states_widget_make
        case "analyze_rollout":   
            trajectory_widget=mo.vstack([
                sweep_params_widget,
                mask_widget_controls,
                m_n_widget,
                mo.hstack([mo.vstack(list(traj_checks.values())), trajectories_plot, lambda_trajectory_plot], justify="start", align="start")
            ], align="start")
            mask_widget = mo.hstack([weather_map, mask_map], justify="start")
            inspect_states_widget=inspect_states_widget_make
        case _:
            pass
    return inspect_states_widget, mask_widget, trajectory_widget


@app.cell
def _(T_slider):
    T_slider
    return


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
    gt_N_slices,
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
    weather_partition = (
        partition
        if notebook_mode in ("guided_rollout", "unguided_rollout")
        else config.partition
    )

    weather_var = (
        var
        if notebook_mode in ("guided_rollout", "unguided_rollout")
        else config.var
    )

    weather_level = (
        level
        if notebook_mode in ("guided_rollout", "unguided_rollout")
        else config.level
    )

    weather_map = visualize_map(
        gt_N_slices[n],
        suptitle=f"{timestamps[n]}",
        title=f"partition={weather_partition} | var={weather_var} | level={weather_level}",
        interactive=map_interactive,
        vmin=np.min(gt_N_slices),
        vmax=np.max(gt_N_slices),
        center=np.mean(gt_N_slices),
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
    return (weather_map,)


@app.cell
def _(
    dpi_slider,
    level,
    mask,
    mask_corners,
    np,
    partition,
    var,
    visualize_map,
):
    mask_map = visualize_map(
        mask,
        suptitle="mask",
        title=f"partition={partition} | var={var} | level={level}",
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
            case "sobel_grads":
                pass

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
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider], justify="start"),
                        mo.hstack([curr_map, prev_map], justify="start"),
                        mo.hstack([gui_map, ung_map], justify="start"),
                    ],
                    justify="start",
                )

            case "difference":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, mo.md(r"*own: $[\min, \max]$ per map · same: $[\min, \max]$ across maps · centered at 0*")], justify="start", align="center"),
                        mo.hstack([gt_gt_map, gui_gui_map], justify="start"),
                        mo.hstack([gui_gt_map, gui_ung_map], justify="start"),
                        mo.hstack([ung_gui_gt_map, ung_gui_ung_gui_map], justify="start")
                    ], justify="start",
                )

            case "sobel_grads":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, mo.md(r"*own: $[\min, \max]$ per map · same: $[\min, \max]$ across maps · centered at 0*")], justify="start", align="center"),
                        sobel_grad_widget
                    ], justify="start",
                )
            case "sobel_diffs":
                inspect_states_widget_make = mo.vstack(
                    [
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

    var_check = (var==config.var if notebook_mode in ("guided_rollout", "analyze_rollout") else False)

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
        delta_trajectory=[0] +delta_trajectory if (traj_checks["delta_trajectory"].value and notebook_mode in ("guided_rollout", "analyze_rollout")) else None,
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
def _(gt_rollout, guided_xr, notebook_mode, ung_gui_xr):
    from src.normalization import XarrayNormalizer
    if notebook_mode == "analyze_rollout":
        xnorm = XarrayNormalizer()
        normalized_gui_xr = xnorm.normalize(guided_xr)
        normalized_ung_gui_xr = xnorm.normalize(ung_gui_xr)
        normalized_gt_xr = xnorm.normalize(
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
    return normalized_gt_xr, normalized_gui_xr, normalized_ung_gui_xr, xnorm


@app.cell(hide_code=True)
def _(
    aggregate_spatially_dropdown,
    clean_preds_xr,
    grads_xr,
    gui_vfs_xr,
    maybe_mask,
    normalized_gt_xr,
    normalized_gui_xr,
    normalized_ung_gui_xr,
    notebook_mode,
    vfs_xr,
    xnorm,
):
    # spatially-reduced cubes: heavy reductions happen here ONCE per data/mask change;
    # sliders (m, n, t, k, variable, Δ, abs, bands) then only index these tiny arrays
    if notebook_mode == "analyze_rollout":
        _sp = ("latitude", "longitude")
        _msk = lambda ds: maybe_mask(ds, aggregate_spatially_dropdown.value)
        _sq = lambda ds: (_msk(ds).astype(float) ** 2).sum(dim=_sp)   # squared sums; sqrt after folding
        _mn = lambda ds: _msk(ds).mean(dim=_sp)
        _dvf_full = gui_vfs_xr - vfs_xr
        red = {
            "grads_l2": _sq(grads_xr),
            "vfs_l2": _sq(vfs_xr),
            "gui_vfs_l2": _sq(gui_vfs_xr),
            "dvf_l2": _sq(_dvf_full),
            "dvf_mean": _mn(_dvf_full),
            "gui_ung_gui_mean": _mn(normalized_gui_xr - normalized_ung_gui_xr),
            "gui_gt_mean": _mn(normalized_gui_xr - normalized_gt_xr),
            # normalization is affine, so it commutes with the spatial mean
            "clean_gt_mean": xnorm.normalize(_mn(clean_preds_xr)) - _mn(normalized_gt_xr),
        }
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
    guidance_mode_dropdown,
    guided_vf_norms_plot,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    t_slider,
    vf_deflection_n_plot,
    vf_deflection_t_plot,
    vf_norms_n_plot,
    vf_norms_plot,
):
    if notebook_mode =="analyze_rollout":
        cross_checks_widget = mo.vstack([
            dist_bands_checkbox,
            guidance_mode_dropdown,
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
def _(
    color_for,
    cross_ctl,
    cross_traces,
    m,
    n,
    notebook_mode,
    plot_trajectory,
    red,
    t,
):
    if notebook_mode =="analyze_rollout":
        _traces, _bands = cross_traces(red["grads_l2"].isel(n=n-1), "t", "l2", m, **cross_ctl)
        grad_norms_plot = plot_trajectory(_traces, title="Grad norms over $t$",
            subtitle=r"$\|\nabla_{z_t} \mathcal{L}_t\|_{\mathrm{spatial}}$", step=t, color_map=color_for(_traces), bands=_bands,
            figsize=(22, 6)
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
    plot_trajectory,
    red,
):
    if notebook_mode =="analyze_rollout":
        _raw, _bands = cross_traces(red["gui_ung_gui_mean"], "n", "mean", m, **cross_ctl)
        diff_gui_ung_gui_plot = plot_trajectory(
            _raw,
            title="Diff (gui − ung_gui) over $n$",
            subtitle=r"$\mathrm{mean}_{\mathrm{spatial}}\,(\tilde{x}^{\,\mathrm{gui}}_{n} - \tilde{x}^{\,\mathrm{ung\_gui}}_{n})$  ($\tilde{x}$: normalized $x$)",
            xlabel="$n$",
            step=n+1,
            color_map=color_for(_raw),
            bands=_bands,
            figsize=(22, 6),
        )
        # diff_gui_ung_gui_plot
    return (diff_gui_ung_gui_plot,)


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
    t,
):
    if notebook_mode == "analyze_rollout":
        def _plot(title, subtitle, ds, axis, agg):
            _tr, _bands = cross_traces(ds, axis, agg, m, **cross_ctl)
            return plot_trajectory(
                _tr, title=title, subtitle=subtitle, xlabel=f"${axis}$",
                step=(n + 1 if axis == "n" else t), color_map=color_for(_tr), bands=_bands, figsize=(22, 6),
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
    lambda_trajectory,
    m,
    mask,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    t,
):
    if notebook_mode =="analyze_rollout":
        _all_per_n = get_masked_mean(cfg_clean_preds_slices[:, :, -1], mask).astype(float) - cfg_target_guidance_M_N_trajectories
        _all_per_n[:, 0] = 0.0
        _diff_per_n = _all_per_n[m]
        guidance_convergence_plot = plot_trajectory(
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
            figsize=(22, 6),
        )
        _all_per_t = get_masked_mean(cfg_clean_preds_slices[:, n], mask).astype(float) - cfg_target_guidance_M_N_trajectories[:, n][:, None]
        _diff_per_t = _all_per_t[m]
        _delta_diff_t = np.concatenate([[0.0], np.diff(_diff_per_t)])
        guidance_convergence_t_plot = plot_trajectory(
            {"realized − target": _diff_per_t},
            bands={"realized − target": (_all_per_t.min(axis=0), _all_per_t.max(axis=0))} if dist_bands_checkbox.value else None,
            title="Guidance convergence over $t$",
            subtitle=r"$\mathrm{mean}_{\mathrm{mask}}(x^{\mathrm{gui}}_{n,\,t}) - \mathrm{target}_n$",
            xlabel="$t$",
            step=t,
            color_map={"realized − target": "#B7950B"},
            right_trajectory={
                r"$\Delta$(realized $-$ target)": _delta_diff_t,
                r"$\lambda_t$ (scaled)": np.asarray(lambda_trajectory, dtype=float)
                    / max(float(np.max(np.abs(lambda_trajectory))), 1e-12)
                    * max(float(np.max(np.abs(_delta_diff_t))), 1e-12),
            },
            right_color={
                r"$\Delta$(realized $-$ target)": "#2E86C1",
                r"$\lambda_t$ (scaled)": "#8A2BE2",
            },
            figsize=(22, 6),
        )
    return guidance_convergence_plot, guidance_convergence_t_plot


@app.cell
def _(mo):
    mo.md(r"""
    ## Flow analysis
    """)
    return


@app.cell
def _(T, T_schedule, alpha_slider, plot_trajectory, t, w_slider):
    alpha = alpha_slider.value
    w=w_slider.value
    lambda_trajectory = T_schedule(T, alpha, w)
    lambda_trajectory_plot = plot_trajectory(lambda_trajectory, "$\\lambda_t$", title="$\\lambda_t$ schedule", step=t, figsize=(22, 6))
    return alpha, lambda_trajectory, lambda_trajectory_plot, w


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
    t,
):
    if notebook_mode == "analyze_rollout":
        _vf_traces, _bands = cross_traces(red["vfs_l2"].isel(n=n-1), "t", "l2", m, **cross_ctl)
        vf_norms_plot = plot_trajectory(_vf_traces, title="Vf norms over $t$",
            subtitle=r"$\|\mathrm{vf}_t\|_{\mathrm{spatial}}$", step=t, color_map=color_for(_vf_traces), bands=_bands,
            figsize=(22, 6))
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
    t,
):
    if notebook_mode == "analyze_rollout":
        _gvf_traces, _bands = cross_traces(red["gui_vfs_l2"].isel(n=n-1), "t", "l2", m, **cross_ctl)
        guided_vf_norms_plot = plot_trajectory(_gvf_traces, title="Gui vf norms over $t$",
            subtitle=r"$\|\mathrm{vf}^{\mathrm{gui}}_t\|_{\mathrm{spatial}}$", step=t, color_map=color_for(_gvf_traces), bands=_bands,
            figsize=(22, 6))
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
def _(
    alpha_slider,
    guidance_mode_dropdown,
    lambda_trajectory_plot,
    mo,
    notebook_mode,
    w_slider,
):
    match notebook_mode:
        case "unguided_rollout":
            flow_schedule_widget=None
        case "guided_rollout":
            if guidance_mode_dropdown.value in ("FLOWGRAD", "FLOWGRAD_FREE"):
                # lambda is learned (FLOWGRAD) or unused (FLOWGRAD_FREE) -> the fixed
                # alpha/w T_schedule preview does not apply to these modes.
                flow_schedule_widget = mo.md(
                    "_$\\lambda_t$ is **learned** by FlowGrad (FLOWGRAD) or not used "
                    "(FLOWGRAD_FREE); the fixed $\\alpha$/$w$ schedule does not apply._"
                )
            else:
                flow_schedule_widget = mo.vstack([
                    mo.hstack([
                        alpha_slider, w_slider
                    ], justify="start"),
                    lambda_trajectory_plot,
                ])
        case "analyze_rollout":
            flow_schedule_widget=None
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
        grads_xr = get_rollout("grads", rollout_id).sel(sweep_params).compute()
        vfs_xr = get_rollout("vfs", rollout_id).sel(sweep_params).compute()
        clean_preds_xr = get_rollout("clean_preds", rollout_id).sel(sweep_params).compute()
        gui_vfs_xr = get_rollout("gui_vfs", rollout_id).sel(sweep_params).compute()
        ung_gui_xr = get_rollout("ung_gui", rollout_id).sel(sweep_params).compute()
        # FLOWGRAD_FREE has no guidance direction -> these trace containers are all-NaN.
        # Fill with 0 so the (not-applicable) trace plots render as flat zero instead of
        # crashing on NaN axis limits; the learned controls live in the FlowGrad
        # diagnostics panel. Other modes are untouched.
        if guidance_mode_dropdown.value == "FLOWGRAD_FREE":
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
    return (
        clean_preds_diff_slice,
        clean_preds_slices,
        diff_grads_slice,
        diff_gt_clean_pred_slice,
        diff_gt_ung_onl_slice,
        grads_slice,
        guided_vfs_slice,
        ung_onl_clean_diff_slice,
        vf_gui_prev_diff_slice,
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
    return (
        clean_pred_map,
        diff_grads_map,
        diff_gt_clean_pred_map,
        diff_vfs_map,
        grads_map,
        gt_state_map,
        guided_vfs_map,
        ung_onl_clean_diff_map,
        ung_onl_map,
        ung_state_map,
        vf_gui_prev_diff_map,
        vfs_map,
    )


@app.cell
def _(mo):
    flow_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "gt & ung states", "gui states", "gui_t diffs", "grads", "vfs", "vf diffs",
    )})
    return (flow_checks,)


@app.cell
def _(
    alpha_slider,
    clean_pred_map,
    diff_grads_map,
    diff_gt_clean_pred_map,
    diff_vfs_map,
    dpi_slider,
    flow_checks,
    grads_map,
    gt_state_map,
    guidance_mode_dropdown,
    guided_vfs_map,
    level_slider,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    partition_dropdown,
    show_mask_switch,
    t_slider,
    ung_onl_clean_diff_map,
    ung_onl_map,
    ung_state_map,
    var_dropdown,
    vf_gui_prev_diff_map,
    vfs_map,
    w_slider,
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
                        mo.hstack([alpha_slider, w_slider], justify="start", align="start"),
                        mo.hstack([t_slider, dpi_slider], justify="start", align="start"),
                        mo.hstack([m_slider, n_slider], justify="start", align="start"),
                        guidance_mode_dropdown
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
            ("gui states", [ung_onl_map, clean_pred_map]),
            ("gui_t diffs", [ung_onl_clean_diff_map, diff_gt_clean_pred_map]),
            ("grads", [grads_map, diff_grads_map]),
            ("vfs", [vfs_map, guided_vfs_map]),
            ("vf diffs", [vf_gui_prev_diff_map, diff_vfs_map]),
        ]

        flow_widget_make = mo.vstack(
            [
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
):
    # FlowGrad learned diagnostics (analyze mode): learned lambda*_t / control-norm per
    # window + the optimization-loss curve. These come from the JSON sidecar written by
    # the rollout (FlowGrad learns its schedule/controls, so they have no zarr container).
    if notebook_mode == "analyze_rollout" and guidance_mode_dropdown.value in ("FLOWGRAD", "FLOWGRAD_FREE"):
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
        flowgrad_diag_widget = None

    flowgrad_diag_widget

    return


if __name__ == "__main__":
    app.run()
