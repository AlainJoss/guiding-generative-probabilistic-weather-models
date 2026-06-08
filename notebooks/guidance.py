import marimo

__generated_with = "0.23.6"
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
    from src.rollout_config import GUIDANCE_REFERENCES, MASK_MODES, RolloutConfig
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.helpers import max_day, get_timestamp_from_sliders
    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory

    from src.utils import get_var_idx, get_level_idx
    from src.utils import get_now_timestamp, ensure_rollout_dir
    from src.utils import get_timestamps, get_N_timestamps, get_N_slices, get_slices, get_gt_rollout
    from src.utils import (
        dump_json, get_rollout_ids, get_rollout, get_sweep_dict, get_config
    )
    from src.funcs import N_schedule, T_schedule, make_hash, safe_abs_limits

    from src.mask import get_masked_mean, get_mask_2d, get_mu_sigma, get_mask_center
    from src.target import get_target_slices

    return (
        GUIDANCE_REFERENCES,
        LEVELS_DICT,
        MASK_MODES,
        N_schedule,
        PARTITIONS,
        RolloutConfig,
        T_schedule,
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
        safe_abs_limits,
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
    delta_trajectory,
    dump_json,
    ensure_rollout_dir,
    get_now_timestamp,
    guidance_reference,
    level,
    mask_corners,
    mask_mode,
    notebook_mode,
    partition,
    rollout_id,
    save_config_button,
    timestamp,
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
            print(rollout_dir)
            path = rollout_dir / "sweep_params.json"
            # TODO: these are only placeholders
            #       implement version with lists?
            save_config = RolloutConfig(
                # guided rollout specific params -> can be swept
                mask_mode=mask_mode,
                guidance_reference=guidance_reference,
                delta_trajectory=delta_trajectory,
                alpha=alpha,
                w=w
            )
            dump_json(save_config.to_dict_list(), path)
    return


@app.cell
def _(
    GUIDANCE_REFERENCES,
    MASK_MODES,
    get_sweep_dict,
    mo,
    notebook_mode,
    rollout_id,
):
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
            w_defaults = [5, 10]
            w_slider = mo.ui.slider(
                steps=w_defaults,
                value=w_defaults[0],
                label="w: ",
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
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
            w_defaults = [5, 10]
            w_slider = mo.ui.slider(
                steps=w_defaults,
                value=w_defaults[0],
                label="w: ",
                debounce=True,
                show_value=True
            )
            mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
            sweep_params_widget = None

        case "analyze_rollout":
            experiment_params = get_sweep_dict(rollout_id)

            # TODO: decide between the two depending on notebook mode
            guidance_reference_dropdown = mo.ui.dropdown(
                options=experiment_params["guidance_reference"],
                value=experiment_params["guidance_reference"][0],
                label="guidance reference",
            )

            alpha_label = "alpha: " if len(experiment_params["alpha"])>1 else "alpha:\u00A0\u00A0"
            alpha_slider = mo.ui.slider(
                steps=experiment_params["alpha"],
                value=experiment_params["alpha"][0],
                label=alpha_label,
                debounce=True,
                show_value=True
            )
            w_label = "w: " if len(experiment_params["w"])>1 else "w:\u00A0\u00A0"
            w_slider = mo.ui.slider(
                steps=experiment_params["w"],
                value=experiment_params["w"][0],
                label=w_label,
                debounce=True,
                show_value=True
            )

            mask_mode_dropdown = mo.ui.dropdown(options=experiment_params["mask_mode"], value=experiment_params["mask_mode"][0], label="mask_mode: ")

            sweep_params_widget = mo.vstack([
                guidance_reference_dropdown, mask_mode_dropdown,
                alpha_slider,
                w_slider,
            ])

        case _:
            pass
    return (
        alpha_slider,
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
            unguided_xr = get_rollout("ung", rollout_id)
            guided_xr = get_rollout("gui", rollout_id)
            guided_xr = guided_xr.sel(sweep_params)
            config = get_config(rollout_id)
        case _:
            pass
    return config, guided_xr, unguided_xr


@app.cell
def _(
    M,
    N,
    N_schedule,
    config,
    delta_peak,
    delta_peak_at_slider,
    delta_shape_slider,
    get_corners,
    get_mask_2d,
    get_masked_mean,
    get_mu_sigma,
    get_target_slices,
    guidance_reference,
    level,
    m,
    mask_mode,
    notebook_mode,
    partition,
    rollout_id,
    timestamp,
    var,
):
    # sweep params 
    match notebook_mode:
        case "unguided_rollout":
            mask_corners = get_corners()
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory=None
            planned_guidance_rollout=None
            planned_guidance_trajectories=None
            planned_guidance_trajectory=None
        case "guided_rollout":
            mask_corners = config.mask_corners
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory = N_schedule(N, delta_shape_slider.value, delta_peak, peak_at_n=delta_peak_at_slider.value)[1:]
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
                    delta_trajectory,
                    m,
                )
                planned_guidance_trajectories = get_masked_mean(planned_guidance_slices, mask)
                planned_guidance_trajectory = planned_guidance_trajectories[m]
                pass
            else:
                planned_guidance_rollout = None
                planned_guidance_slices = None
                planned_guidance_trajectories = None
                planned_guidance_trajectory = None
        case "analyze_rollout":        
            mask_corners=config.mask_corners # should be a sweep param mask_corners_dropdown.value
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory=config.delta_trajectory
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
                    delta_trajectory,
                    m,
                )
                planned_guidance_trajectories = get_masked_mean(planned_guidance_slices, mask)
                planned_guidance_trajectory = planned_guidance_trajectories[m]
                pass
            else:
                planned_guidance_rollout = None
                planned_guidance_slices = None
                planned_guidance_trajectories = None
                planned_guidance_trajectory = None
        case _:
            pass
    return (
        delta_trajectory,
        mask,
        mask_corners,
        planned_guidance_trajectories,
        planned_guidance_trajectory,
    )


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
            target_guidance_trajectory = (1 + np.asarray(delta_trajectory)) * ung_m_trajectory
            target_guidance_M_N_trajectories = (1 + np.asarray(delta_trajectory)) * ung_M_N_trajectories
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
def _(alpha_slider, guidance_reference_dropdown, mask_mode_dropdown, w_slider):
    sweep_params = {
        "w": w_slider.value,
        "alpha": alpha_slider.value,
        "guidance_reference": guidance_reference_dropdown.value,
        "mask_mode": mask_mode_dropdown.value
    }
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
def _(level_slider, m_slider, n_slider, var_dropdown):
    level=level_slider.value
    var=var_dropdown.value
    n=n_slider.value-1
    m=m_slider.value
    return level, m, n, var


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
def _():
    return


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
    return delta_peak_at_slider, delta_peak_slider, delta_shape_slider


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
    M_N_widget,
    delta_bounds_slider,
    delta_granularity_slider,
    delta_peak_at_slider,
    delta_peak_slider,
    delta_shape_slider,
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
        mask_mode_dropdown,
        mask_widget_controls, mask_widget_maps], align="start")
    delta_widget = mo.hstack([
        delta_peak_slider,
        delta_granularity_slider,
        delta_bounds_slider,
        delta_shape_slider,
        delta_peak_at_slider,
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
                trajectories_plot
            ])
            inspect_states_widget=inspect_states_widget_make
        case "analyze_rollout":   
            trajectory_widget=mo.vstack([
                mask_widget_controls,
                m_n_widget,
                mo.hstack([mo.vstack(list(traj_checks.values())), trajectories_plot, lambda_trajectory_plot], justify="start", align="start")
            ], align="start")
            mask_widget = mo.hstack([weather_map, mask_map], justify="start")
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
        vmin=np.min(mask),
        vmax=np.max(mask),
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
    analysis_types = ["absolute", "difference", "sobel_grads"]
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
    dpi_slider = mo.ui.slider(start=50, stop=500, step=50, value=100, debounce=False, show_value=True, label="dpi: ")
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
def _(mo):
    show_values_checkbox = mo.ui.checkbox(label="show values")
    return (show_values_checkbox,)


@app.cell
def _(mo):
    # text thresh
    center_value_for_threshold = 0.0
    max_abs_diff_for_threshold =0.2
    # = float(
    #     max(
    #         np.nanmax(gui_ung),
    #         abs(np.nanmin(gui_ung)),
    #     )
    # )

    if max_abs_diff_for_threshold <= 0:
        max_abs_diff_for_threshold = 1e-8

    default_value_threshold = max_abs_diff_for_threshold * 0.9

    text_thresh_slider = mo.ui.slider(
        start=center_value_for_threshold,
        stop=max_abs_diff_for_threshold,
        step=max_abs_diff_for_threshold / 20,
        value=default_value_threshold,
        label="text thresh: ",
        debounce=True,
        show_value=True
    )
    return (text_thresh_slider,)


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
        ung_ung = ung_curr - ung_prev
        ung_gt = ung_curr - gt_curr
        ung_gt_prev = ung_curr - gt_prev

    if notebook_mode =="analyze_rollout":
        ung_curr = ung_M_N_slices[m][n]
        gui_curr = gui_M_N_slices[m][n]
        gt_curr = gt_N_slices[n]
        # ung_prev = ung_M_N_slices[m][n-1] if n>0 else ung_M_N_slices[m][n]
        gui_prev = gui_M_N_slices[m][n-1] if n>0 else gui_M_N_slices[m][n]
        gt_prev = gt_N_slices[n-1] if n>0 else gt_N_slices[n]

        ung_onl_slice = get_slices(ung_gui_xr, partition, var, level)
        ung_onl_curr = ung_onl_slice[m][n]

        gt_gt = gt_curr - gt_prev
        gui_gui = gui_curr - gui_prev
        gui_ung_gui = gui_curr - ung_onl_curr
        gui_gt = gui_curr - gt_curr
    return (
        gt_curr,
        gt_gt,
        gt_prev,
        gui_curr,
        gui_gt,
        gui_gui,
        gui_ung_gui,
        ung_curr,
        ung_gt,
        ung_gt_prev,
        ung_onl_curr,
        ung_prev,
        ung_ung,
    )


@app.cell
def _(
    analysis_type_dropdown,
    cv2,
    dpi_slider,
    gt_curr,
    gt_gt,
    gt_prev,
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
    show_values_checkbox,
    text_thresh_slider,
    ung_curr,
    ung_gt,
    ung_gt_prev,
    ung_onl_curr,
    ung_prev,
    ung_ung,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode =="guided_rollout":
        match analysis_type_dropdown.value:
            case "absolute":
                absolute_panels = [
                    ("$x_n$", gt_curr),
                    ("$x_{n-1}$", gt_prev),
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

                curr_map = absolute_maps["$x_n$"]
                prev_map = absolute_maps["$x_{n-1}$"]
                ung_map = absolute_maps["$x_{n}^{ung}$"]
                ung_prev_map = absolute_maps["$x_{n-1}^{ung}$"]
            case "difference":
                difference_panels = [
                    ("$x_{n} - x_{n-1}$", gt_gt),
                    ("$x_{n}^{ung} - x_{n}$", ung_gt),
                    ("$x_{n}^{ung} - x_{n-1}^{ung}$", ung_ung),
                    ("$x_{n}^{ung} - x_{n-1}$", ung_gt_prev),
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
                        show_values=show_values_checkbox.value,
                        value_threshold=text_thresh_slider.value,
                        value_fontsize=5,
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )

                gt_gt_map = difference_maps["$x_{n} - x_{n-1}$"]
                ung_ung_map = difference_maps["$x_{n}^{ung} - x_{n}$"]
                ung_ung_map = difference_maps["$x_{n}^{ung} - x_{n-1}^{ung}$"]
                ung_gt_prev_map = difference_maps["$x_{n}^{ung} - x_{n-1}$"]
            case "sobel_grads":
                sobel_grad_widget = None
            case _:
                pass

    if notebook_mode =="analyze_rollout":
        match analysis_type_dropdown.value:
            case "absolute":
                absolute_panels = [
                    ("$x_n$", gt_curr),
                    ("$x_{n}^{\\text{ung_gui}}$", ung_onl_curr),
                    ("$x_{n}^{ung}$", ung_curr),
                    ("$x_{n}^{gui}$", gui_curr),
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

                curr_map = absolute_maps["$x_n$"]
                prev_map = absolute_maps["$x_{n}^{\\text{ung_gui}}$"]
                ung_map = absolute_maps["$x_{n}^{ung}$"]
                gui_map = absolute_maps["$x_{n}^{gui}$"]

            case "difference":
                difference_panels = [
                    ("$x_{n} - x_{n-1}$", gt_gt),
                    ("$x_{n}^{gui} - x_{n}$", gui_gt),
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
                        show_values=show_values_checkbox.value,
                        value_threshold=text_thresh_slider.value,
                        value_fontsize=5,
                        dpi=dpi_slider.value,
                        figsize=(14, 8),
                    )

                gt_gt_map = difference_maps["$x_{n} - x_{n-1}$"]
                gui_gui_map = difference_maps["$x_{n}^{gui} - x_{n-1}^{gui}$"]
                gui_ung_map = difference_maps["$x_{n}^{gui} - x_{n}^{\\text{ung_gui}}$"]
                gui_gt_map = difference_maps["$x_{n}^{gui} - x_{n}$"]
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
                    (r"$\|\nabla x_n\|$", gradmap_gt_mag),
                    (r"$\|\nabla x_n^{\text{gui}}\|$", gradmap_gui_mag),
                    (r"$\|\nabla x_n^{\text{ung}}\|$", gradmap_ung_mag),
                    (r"$\|\nabla (x_n^{\text{ung_gui}})\|$", gradmap_gui_ung_mag),
                ]

                gradmap_mag_vmin = min(float(np.nanmin(gradmap_arr)) for _, gradmap_arr in gradmap_mag_panels)
                gradmap_mag_vmax = max(float(np.nanmax(gradmap_arr)) for _, gradmap_arr in gradmap_mag_panels)

                gradmap_figures = []

                for gradmap_title, gradmap_arr in gradmap_mag_panels:
                    gradmap_figures.append(
                        visualize_map(
                            gradmap_arr,
                            mask_2d=mask,
                            title=gradmap_title,
                            # vmin=gradmap_mag_vmin,
                            # vmax=gradmap_mag_vmax,
                            # center=0.0,
                            show_mask=show_mask_switch.value,
                            zoom=zoom_slider.value,
                            zoom_center_lon=zoom_centers[0],
                            zoom_center_lat=zoom_centers[1],
                            show_values=show_values_checkbox.value,
                            value_threshold=text_thresh_slider.value,
                            value_fontsize=5,
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
            case _:
                sobel_grad_widget = None
    return (
        curr_map,
        gt_gt_map,
        gui_gt_map,
        gui_gui_map,
        gui_map,
        gui_ung_map,
        prev_map,
        sobel_grad_widget,
        ung_gt_prev_map,
        ung_map,
        ung_prev_map,
        ung_ung_map,
    )


@app.cell
def _(
    analysis_type_dropdown,
    curr_map,
    dpi_slider,
    gt_gt_map,
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
    show_values_checkbox,
    sobel_grad_widget,
    text_thresh_slider,
    ung_gt_prev_map,
    ung_map,
    ung_prev_map,
    ung_ung_map,
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
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, show_values_checkbox, text_thresh_slider], justify="start"),
                        mo.hstack([gt_gt_map, ung_ung_map], justify="start"),
                        mo.hstack([ung_ung_map, ung_gt_prev_map], justify="start")
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
                        mo.hstack([ung_map, gui_map], justify="start"),
                    ],
                    justify="start",
                )

            case "difference":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, show_values_checkbox, text_thresh_slider], justify="start"),
                        mo.hstack([gt_gt_map, gui_ung_map], justify="start"),
                        mo.hstack([gui_gt_map, gui_gui_map], justify="start")
                    ], justify="start",
                )

            case "sobel_grads":
                inspect_states_widget_make = mo.vstack(
                    [
                        *common_controls,
                        mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, show_values_checkbox, text_thresh_slider], justify="start"),
                        sobel_grad_widget
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
def _(mo, np):
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
    show_delta_checkbox = mo.ui.checkbox(label="delta_trajectory")
    show_realized_diff_checkbox = mo.ui.checkbox(label="realized − target")
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

    def maybe_diff(traces, on):
        return {k: np.concatenate([[0.0], np.diff(v)]) for k, v in traces.items()} if on else traces

    def maybe_abs(traces, on):
        return {k: np.abs(v) for k, v in traces.items()} if on else traces

    def top_k(traces, k):
        return dict(sorted(traces.items(), key=lambda kv: -float(np.max(np.abs(kv[1]))))[:k]) if k < len(traces) else traces

    return (
        abs_checkbox,
        aggregate_by_level_checkbox,
        color_for,
        differential_checkbox,
        grouped_vars,
        level_var_dropdown,
        maybe_abs,
        maybe_diff,
        n_traces,
        show_delta_checkbox,
        show_realized_diff_checkbox,
        top_k,
    )


@app.cell
def _(aggregate_by_level_checkbox, level_var_dropdown, mo, n_traces):
    _kmax = n_traces(level_var_dropdown.value, aggregate_by_level_checkbox.value)
    top_k_slider = mo.ui.slider(1, _kmax, value=min(5,_kmax), label="top K", show_value=True)
    return (top_k_slider,)


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    differential_checkbox,
    level_var_dropdown,
    mo,
    top_k_slider,
):
    cross_check_controls = mo.hstack([level_var_dropdown, aggregate_by_level_checkbox, differential_checkbox, abs_checkbox, top_k_slider], justify="start", align="start")
    return (cross_check_controls,)


@app.cell
def _(guided_xr, notebook_mode, ung_gui_xr):
    from src.normalization import XarrayNormalizer
    if notebook_mode == "analyze_rollout":
        xnorm = XarrayNormalizer()
        normalized_gui_xr = xnorm.normalize(guided_xr)
        normalized_ung_gui_xr = xnorm.normalize(ung_gui_xr)
    return normalized_gui_xr, normalized_ung_gui_xr


@app.cell
def _():
    # final widget
    return


@app.cell
def _(
    cross_check_controls,
    diff_gui_ung_gui_plot,
    grad_norms_plot,
    guidance_convergence_plot,
    guided_vf_norms_plot,
    m_slider,
    mo,
    n_slider,
    notebook_mode,
    show_delta_checkbox,
    show_realized_diff_checkbox,
    t_slider,
    vf_norms_plot,
):
    if notebook_mode =="analyze_rollout":
        cross_checks_widget = mo.vstack([
            cross_check_controls,
            mo.hstack([m_slider, n_slider, t_slider], justify="start"),
            mo.hstack([
                mo.vstack([show_delta_checkbox, show_realized_diff_checkbox], justify="start", align="start").style(width="fit-content"),
                mo.vstack(
                    [diff_gui_ung_gui_plot, guidance_convergence_plot],
                    justify="start",
                    align="start",
                ).style(width="fit-content"),
                mo.vstack(
                    [grad_norms_plot, vf_norms_plot, guided_vf_norms_plot],
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
    abs_checkbox,
    aggregate_by_level_checkbox,
    clean_preds_slices,
    color_for,
    differential_checkbox,
    get_masked_mean,
    grads_xr,
    grouped_vars,
    level_var_dropdown,
    m,
    mask,
    maybe_abs,
    maybe_diff,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    show_realized_diff_checkbox,
    t,
    target_guidance_M_N_trajectories,
    top_k,
    top_k_slider,
):
    if notebook_mode =="analyze_rollout":
        _ds = grads_xr.isel(m=m, n=n-1)
        _traces = {k: np.sqrt(sum((da.astype(float)**2).sum(dim=[d for d in da.dims if d != "t"]).values for da in das))
                   for k, das in grouped_vars(_ds, level_var_dropdown.value, aggregate_by_level_checkbox.value).items()}
        _traces = top_k(maybe_abs(maybe_diff(_traces, differential_checkbox.value), abs_checkbox.value), top_k_slider.value)
        _realized_minus_target = (get_masked_mean(clean_preds_slices[m, :, n], mask) - target_guidance_M_N_trajectories[m, n]) if show_realized_diff_checkbox.value else None
        grad_norms_plot = plot_trajectory(_traces, title="Grad norms",
            subtitle=f"member={m} | rollout step n={n}", step=t, color_map=color_for(_traces),
            right_trajectory=_realized_minus_target,
            right_label=r"realized $-$ target",
            right_color="#8A2BE2",
            figsize=(22, 6)
        )
        # grad_norms_plot
    return (grad_norms_plot,)


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    color_for,
    delta_trajectory,
    differential_checkbox,
    grouped_vars,
    level_var_dropdown,
    m,
    maybe_abs,
    maybe_diff,
    n,
    normalized_gui_xr,
    normalized_ung_gui_xr,
    notebook_mode,
    plot_trajectory,
    show_delta_checkbox,
    top_k,
    top_k_slider,
):
    if notebook_mode =="analyze_rollout":
        _dds = (normalized_gui_xr - normalized_ung_gui_xr).isel(m=m)
        _raw = {k: sum(da.mean(dim=[d for d in da.dims if d != "n"]).values for da in das) / len(das)
                for k, das in grouped_vars(_dds, level_var_dropdown.value, aggregate_by_level_checkbox.value).items()}
        _raw = top_k(maybe_abs(maybe_diff(_raw, differential_checkbox.value), abs_checkbox.value), top_k_slider.value)
        diff_gui_ung_gui_plot = plot_trajectory(
            _raw,
            title="Diff (guided − unguided_guided)",
            subtitle=f"member={m} | normalized",
            xlabel="$n$",
            step=n,
            color_map=color_for(_raw),
            right_trajectory=delta_trajectory if show_delta_checkbox.value else None,
            right_label=r"$\delta_t$",
            right_color="#8A2BE2",
            right_percentage=True,
            figsize=(22, 6),
        )
        # diff_gui_ung_gui_plot
    return (diff_gui_ung_gui_plot,)


@app.cell
def _(
    abs_checkbox,
    clean_preds_slices,
    delta_trajectory,
    get_masked_mean,
    m,
    mask,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    target_guidance_M_N_trajectories,
):
    if notebook_mode =="analyze_rollout":
        _diff_per_n = get_masked_mean(clean_preds_slices[m, -1], mask).astype(float) - target_guidance_M_N_trajectories[m]
        _diff_per_n[0] = 0.0
        if abs_checkbox.value: _diff_per_n = np.abs(_diff_per_n)
        guidance_convergence_plot = plot_trajectory(
            {"realized − target": _diff_per_n},
            title="Convergence per rollout step",
            subtitle=f"member={m} | final clean pred (t={clean_preds_slices.shape[1]-1})",
            xlabel="$n$",
            step=n,
            color_map={"realized − target": "#B7950B"},
            right_trajectory=delta_trajectory,
            right_label=r"$\delta_n$",
            right_color="#8A2BE2",
            right_percentage=True,
            figsize=(22, 6),
        )
        # guidance_convergence_plot
    return (guidance_convergence_plot,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Flow analysis
    """)
    return


@app.cell
def _(T_schedule, alpha_slider, plot_trajectory, t, w_slider):
    alpha = alpha_slider.value
    w=w_slider.value
    lambda_trajectory = T_schedule(alpha, w)
    lambda_trajectory_plot = plot_trajectory(lambda_trajectory, "$\lambda_t$", title="$\lambda_t$ schedule", step=t, figsize=(22, 6))
    return alpha, lambda_trajectory_plot, w


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    color_for,
    differential_checkbox,
    grouped_vars,
    level_var_dropdown,
    m,
    maybe_abs,
    maybe_diff,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    t,
    top_k,
    top_k_slider,
    vfs_xr,
):
    if notebook_mode == "analyze_rollout":
        _ds_vf = vfs_xr.isel(m=m, n=n-1)
        _vf_traces = {k: np.sqrt(sum((da.astype(float)**2).sum(dim=[d for d in da.dims if d != "t"]).values for da in das))
                      for k, das in grouped_vars(_ds_vf, level_var_dropdown.value, aggregate_by_level_checkbox.value).items()}
        _vf_traces = top_k(maybe_abs(maybe_diff(_vf_traces, differential_checkbox.value), abs_checkbox.value), top_k_slider.value)
        vf_norms_plot = plot_trajectory(_vf_traces, title="VF norms",
            subtitle=f"member={m} | rollout step n={n}", step=t, color_map=color_for(_vf_traces),
            figsize=(22, 6))
    else:
        vf_norms_plot = None
    return (vf_norms_plot,)


@app.cell
def _(
    abs_checkbox,
    aggregate_by_level_checkbox,
    color_for,
    differential_checkbox,
    grouped_vars,
    gui_vfs_xr,
    level_var_dropdown,
    m,
    maybe_abs,
    maybe_diff,
    n,
    notebook_mode,
    np,
    plot_trajectory,
    t,
    top_k,
    top_k_slider,
):
    if notebook_mode == "analyze_rollout":
        _ds_gvf = gui_vfs_xr.isel(m=m, n=n-1)
        _gvf_traces = {k: np.sqrt(sum((da.astype(float)**2).sum(dim=[d for d in da.dims if d != "t"]).values for da in das))
                       for k, das in grouped_vars(_ds_gvf, level_var_dropdown.value, aggregate_by_level_checkbox.value).items()}
        _gvf_traces = top_k(maybe_abs(maybe_diff(_gvf_traces, differential_checkbox.value), abs_checkbox.value), top_k_slider.value)
        guided_vf_norms_plot = plot_trajectory(_gvf_traces, title="Guided VF norms",
            subtitle=f"member={m} | rollout step n={n}", step=t, color_map=color_for(_gvf_traces),
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
def _(alpha_slider, lambda_trajectory_plot, mo, notebook_mode, w_slider):
    match notebook_mode:
        case "unguided_rollout":
            flow_schedule_widget=None
        case "guided_rollout":
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
def _(get_rollout, notebook_mode, rollout_id, sweep_params):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        grads_xr = get_rollout("grads", rollout_id).sel(sweep_params)
        vfs_xr = get_rollout("vfs", rollout_id).sel(sweep_params)
        clean_preds_xr = get_rollout("clean_preds", rollout_id).sel(sweep_params)
        gui_vfs_xr = get_rollout("gui_vfs", rollout_id).sel(sweep_params)
        ung_gui_xr = get_rollout("ung_gui", rollout_id).sel(sweep_params)
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
        diff_gt_clean_pred_slice = clean_preds_slices[m][t][n] - gt_curr
        # 2
        ung_onl_clean_diff_slice = clean_preds_slices[m][t][n] - ung_onl_curr
        clean_preds_slice_prev = clean_preds_slices[m][t-1][n] if t>0 else clean_preds_slices[m][t][n]
        clean_preds_diff_slice = clean_preds_slices[m][t][n] - clean_preds_slice_prev
        # 3
        grads_slice = grads_slices[m][t][n]
        grads_slice_prev_slice = grads_slices[m][t-1][n] if t>0 else grads_slices[m][t][n]
        diff_grads_slice = grads_slice- grads_slice_prev_slice
        # 4
        guided_vfs_slice = guided_vfs_slices[m][t][n]
        vfs_slice = vfs_slices[m][t][n]
    return (
        clean_preds_diff_slice,
        clean_preds_slices,
        diff_grads_slice,
        diff_gt_clean_pred_slice,
        diff_gt_ung_onl_slice,
        grads_slice,
        guided_vfs_slice,
        ung_onl_clean_diff_slice,
        vfs_slice,
    )


@app.cell
def _(t_slider):
    t=t_slider.value
    return (t,)


@app.cell
def _(
    clean_preds_diff_slice,
    diff_grads_slice,
    diff_gt_clean_pred_slice,
    diff_gt_ung_onl_slice,
    dpi_slider,
    grads_slice,
    guided_vfs_slice,
    mask,
    notebook_mode,
    np,
    show_mask_switch,
    ung_onl_clean_diff_slice,
    vfs_slice,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        diff_vfs_slice = guided_vfs_slice - vfs_slice

        map_specs = [
            ("diff_gt_ung_onl_map", diff_gt_ung_onl_slice, r"$x_{\text{ung}} - x_n$", -1, 1),
            ("diff_gt_clean_pred_map", diff_gt_clean_pred_slice, r"$\hat{x}_t - x_n$", -1, 1),
            ("ung_onl_clean_diff_map", ung_onl_clean_diff_slice, r"$\hat{x}_t - x_{\text{ung}}$", -1, 1),
            ("clean_preds_diff_map", clean_preds_diff_slice, r"$\hat{x}_t - \hat{x}_{t-1}$", -1, 1),
            ("grads_map", grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t$", -1, 1),
            ("diff_grads_map", diff_grads_slice, "$\\nabla_{z_t} \\mathcal{L}_t - \\nabla_{z_{t-1}} \\mathcal{L}_{t-1}$", -1, 1),
            ("vfs_map", vfs_slice, r"$\text{vf}_t$", -0.001, 0.001),
            ("guided_vfs_map", guided_vfs_slice, r"$\text{vf}^{\text{gui}}_t$", -0.001, 0.001),
        ]

        maps = {}

        for name, data, title, fallback_vmin, fallback_vmax in map_specs:
            data_min = np.min(data)
            data_max = np.max(data)
            data_mean = np.mean(data)

            maps[name] = visualize_map(
                data,
                mask_2d=mask,
                show_mask=show_mask_switch.value,
                title=title,
                interactive=False,
                vmin=data_min if data_min != 0 else fallback_vmin,
                vmax=data_max if data_max != 0 else fallback_vmax,
                center=data_mean if data_mean != 0 else 0,
                figsize=(14, 8),
                dpi=dpi_slider.value,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
            )

        diff_gt_ung_onl_map = maps["diff_gt_ung_onl_map"]
        diff_gt_clean_pred_map = maps["diff_gt_clean_pred_map"]
        ung_onl_clean_diff_map = maps["ung_onl_clean_diff_map"]
        clean_preds_diff_map = maps["clean_preds_diff_map"]
        grads_map = maps["grads_map"]
        diff_grads_map = maps["diff_grads_map"]
        vfs_map = maps["vfs_map"]
        guided_vfs_map = maps["guided_vfs_map"]
    return (
        clean_preds_diff_map,
        diff_grads_map,
        diff_gt_clean_pred_map,
        diff_gt_ung_onl_map,
        grads_map,
        guided_vfs_map,
        ung_onl_clean_diff_map,
        vfs_map,
    )


@app.cell
def _(mo):
    flow_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "diffs_to_gt", "diffs_clean_preds", "grad_maps", "vf_maps",
    )})
    return (flow_checks,)


@app.cell
def _(
    alpha_slider,
    clean_preds_diff_map,
    diff_grads_map,
    diff_gt_clean_pred_map,
    diff_gt_ung_onl_map,
    dpi_slider,
    flow_checks,
    grads_map,
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
    var_dropdown,
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
            ("diffs_to_gt", [diff_gt_ung_onl_map, diff_gt_clean_pred_map]),
            ("diffs_clean_preds", [ung_onl_clean_diff_map, clean_preds_diff_map]),
            ("grad_maps", [grads_map, diff_grads_map]),
            ("vf_maps", [vfs_map, guided_vfs_map]),
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


if __name__ == "__main__":
    app.run()
