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

    return Path, mo, np


@app.cell
def _():
    from src.paths import ROLLOUTS, RUN_CONFIGS
    from src.rollout_config import GUIDANCE_REFERENCES, MASK_MODES, RolloutConfig
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.helpers import max_day, get_timestamp_from_sliders
    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory
    from src.ui.analysis.analysis_plots import plot_guidance_tracking


    from src.utils.dataset_utils import get_x_cond
    from src.utils.converters import list_tensors_to_floats, get_var_idx, get_level_idx
    from src.utils.setup import get_now_timestamp
    from src.utils.dataset_utils import get_timestamps, get_N_timestamps, get_N_slices, get_slices, get_gt_rollout
    from src.utils.read_write import (
        dump_json, get_dict_from_json, get_rollout_ids, get_rollout_files
    )

    from src.funcs import N_schedule, T_schedule, make_hash

    from src.mask import get_masked_mean, get_mask_2d, get_mu_sigma
    from src.target import get_reference_rollout, get_target_rollout

    return (
        GUIDANCE_REFERENCES,
        LEVELS_DICT,
        MASK_MODES,
        N_schedule,
        PARTITIONS,
        ROLLOUTS,
        RUN_CONFIGS,
        RolloutConfig,
        T_schedule,
        VARIABLES_DICT,
        dump_json,
        get_N_timestamps,
        get_dict_from_json,
        get_gt_rollout,
        get_level_idx,
        get_mask_2d,
        get_masked_mean,
        get_mu_sigma,
        get_now_timestamp,
        get_reference_rollout,
        get_rollout_files,
        get_rollout_ids,
        get_slices,
        get_target_rollout,
        get_timestamp_from_sliders,
        get_var_idx,
        make_hash,
        max_day,
        plot_dual_trajectory,
        plot_trajectory,
        visualize_map,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - shouldn't show interactivity elements when they are fixed parameters -> calendar, N, M, ... present them in a compact way
    - Use daterange calendar for presentation -> use in guided and analysis modes
    - I'm hiding the lambda_schedule from the ui for now to simplify things.
    """)
    return


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
    refresh_button
    return (refresh_button,)


@app.cell
def _(mo, refresh_button):
    if refresh_button.value:
        pass

    NOTEBOOK_MODES = ["unguided_rollout", "guided_rollout", "analyze_rollout"]
    notebook_mode_dropdown = mo.ui.dropdown(
        options=NOTEBOOK_MODES,
        value=NOTEBOOK_MODES[0],
        label="notebook_mode: ",
    )
    return (notebook_mode_dropdown,)


@app.cell
def _(get_rollout_ids, mo, notebook_mode_dropdown):
    notebook_mode = notebook_mode_dropdown.value

    save_config_button = mo.ui.run_button(label="save config")
    match notebook_mode:
        case "unguided_rollout":
            ROLLOUT_IDS=[]
        case "guided_rollout":
            ROLLOUT_IDS=get_rollout_ids("unguided_rollout")
        case "analyze_rollout":
            ROLLOUT_IDS=get_rollout_ids("guided_rollout")
            save_config_button=None
        case _:
            pass
    return ROLLOUT_IDS, notebook_mode, save_config_button


@app.cell
def _(ROLLOUT_IDS, mo):
    rollout_id_dropdown = mo.ui.dropdown(
        options=ROLLOUT_IDS,
        value=ROLLOUT_IDS[0] if len(ROLLOUT_IDS)>0 else None,
        label="rollout_id: ",
        allow_select_none=True
    )
    return (rollout_id_dropdown,)


@app.cell
def _(
    M,
    N,
    RUN_CONFIGS,
    RolloutConfig,
    alpha,
    delta_trajectory,
    dump_json,
    guidance_flag,
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
    if save_config_button is not None:
        if save_config_button.value and notebook_mode!="analyze_rollout":
            save_config = RolloutConfig(
                rollout_id=rollout_id,
                guidance_flag=guidance_flag,
                M=M,
                N=N,
                timestamp=timestamp,  
                level=level,
                partition=partition,
                var=var,
                mask_mode=mask_mode,
                mask_corners=mask_corners,
                guidance_reference=guidance_reference,
                delta_trajectory=delta_trajectory,
                alpha=alpha,
                w=w
            )
            run_dir = RUN_CONFIGS / notebook_mode
            dump_json(save_config.to_dict(), run_dir, f"{rollout_id}")
    return


@app.cell
def _(
    GUIDANCE_REFERENCES,
    Path,
    ROLLOUTS,
    get_dict_from_json,
    mo,
    notebook_mode,
    rollout_id,
):
    match notebook_mode:
        case "unguided_rollout" | "guided_rollout":
            guidance_reference_dropdown = mo.ui.dropdown(GUIDANCE_REFERENCES, value=GUIDANCE_REFERENCES[0], label="guidance reference: ")
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
            sweep_params_widget = None
        case "analyze_rollout":
            experiment_params = get_dict_from_json(Path(ROLLOUTS, rollout_id, "sweep_params.json"))

            # TODO: decide between the two depending on notebook mode
            guidance_reference_dropdown = mo.ui.dropdown(
                options=experiment_params["guidance_reference"],
                value=experiment_params["guidance_reference"][0],
                label="guidance reference",
            )

            alpha_slider = mo.ui.slider(
                steps=experiment_params["alpha"],
                value=experiment_params["alpha"][0],
                label="alpha: ",
                debounce=True,
                show_value=True
            )

            w_slider = mo.ui.slider(
                steps=experiment_params["w"],
                value=experiment_params["w"][0],
                label="w: ",
                debounce=True,
                show_value=True
            )

            sweep_params_widget = mo.vstack([
                guidance_reference_dropdown,
                alpha_slider,
                w_slider,
            ])

        case _:
            pass
    return (
        alpha_slider,
        guidance_reference_dropdown,
        sweep_params_widget,
        w_slider,
    )


@app.cell
def _(alpha_slider, guidance_reference_dropdown, notebook_mode, w_slider):
    match notebook_mode:
        case "unguided_rollout" | "guided_rollout":
            hash_params = None
        case "analyze_rollout":
            hash_params = {
                "guidance_reference": guidance_reference_dropdown.value,
                "alpha": alpha_slider.value,
                "w": w_slider.value,
            }
        case _:
            pass
    return (hash_params,)


@app.cell
def _(
    mo,
    notebook_mode_dropdown,
    rollout_id_dropdown,
    save_config_button,
    sweep_params_widget,
):
    setup_widget = mo.vstack([    
        notebook_mode_dropdown,
        rollout_id_dropdown,
        save_config_button if save_config_button is not None else sweep_params_widget
    ])
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
):
    # conf params 
    match notebook_mode:
        case "unguided_rollout":
            guidance_flag=False
            M=M_slider.value
            N=N_slider.value
            month=month_slider.value
            day=day_slider.value
            hour=hour_slider.value
            timestamp = get_timestamp_from_sliders(month, day, hour)
            map_interactive = True
        case "guided_rollout":
            guidance_flag=True
            M=config.M
            N=config.N
            month=None
            day=None
            hour=None
            timestamp=config.timestamp
            map_interactive = True
        case "analyze_rollout":
            guidance_flag=None
            M=config.M
            N=config.N
            month=None
            day=None
            hour=None
            timestamp=config.timestamp
            map_interactive = False
        case _:
            pass

    timestamps=get_N_timestamps(timestamp, N+1)
    return M, N, guidance_flag, map_interactive, timestamp, timestamps


@app.cell
def _(get_rollout_files, hash_params, make_hash, notebook_mode, rollout_id):
    # data objects and config
    match notebook_mode:
        case "unguided_rollout":
            unguided_xr=None
            unguided_xr=None
            config=None
            # TODO: set everything to None
        case "guided_rollout":
            unguided_xr, config = get_rollout_files("unguided_rollout", rollout_id)
        case "analyze_rollout":
            unguided_xr, _ = get_rollout_files("unguided_rollout", rollout_id)
            guided_id = make_hash(hash_params)
            unguided_xr, config = get_rollout_files("guided_rollout", rollout_id, guided_id)
        case _:
            pass
    return config, unguided_xr


@app.cell
def _(
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
    get_reference_rollout,
    get_slices,
    get_target_rollout,
    guidance_reference,
    level,
    level_idx,
    m,
    mask_mode,
    notebook_mode,
    partition,
    rollout_id,
    timestamp,
    var,
    var_idx,
):
    # sweep params 
    match notebook_mode:
        case "unguided_rollout":
            mask_corners = get_corners()
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory=None
            target_rollout=None
            target_trajectory=None
        case "guided_rollout":
            mask_corners = get_corners()
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory = N_schedule(N, delta_shape_slider.value, delta_peak, peak_at_n=delta_peak_at_slider.value)
            reference_rollout = get_reference_rollout(guidance_reference, rollout_id, m=m, N=N, timestamp=timestamp)
            # TODO: not sure if this good
            target_rollout = get_target_rollout(
                partition, var_idx, level_idx,
                delta_trajectory, reference_rollout
            )
            # TODO: check whether always M -> think not
            target_slices = get_slices(target_rollout, partition, var, level)
            target_trajectory = get_masked_mean(target_slices, mask)
        case "analyze_rollout":        
            mask_corners=config.mask_corners # should be a sweep param mask_corners_dropdown.value
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory=config.delta_trajectory
            reference_rollout = get_reference_rollout(guidance_reference, rollout_id, m=m, N=N, timestamp=timestamp)
            # TODO: not sure if this good
            target_rollout = get_target_rollout(
                partition, var_idx, level_idx,
                delta_trajectory, reference_rollout
            )
            # TODO: check whether always M -> think not
            target_slices = get_slices(target_rollout, partition, var, level)
            target_trajectory = get_masked_mean(target_slices, mask)
        case _:
            pass
    return delta_trajectory, mask, mask_corners, target_trajectory


@app.cell
def _(guidance_reference, reference_trajectories):
    reference_trajectory = reference_trajectories[guidance_reference]
    return (reference_trajectory,)


@app.cell
def _(
    N,
    get_gt_rollout,
    get_masked_mean,
    get_slices,
    level,
    m,
    mask,
    notebook_mode,
    partition,
    timestamp,
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
        case "analyze_rollout":
            ung_M_N_slices = get_slices(unguided_xr, partition, var, level)
            ung_M_N_trajectories = get_masked_mean(ung_M_N_slices, mask)
            ung_mean_trajectory = ung_M_N_trajectories.mean(axis=0)
            ung_m_trajectory = ung_M_N_trajectories[m]
            ung_lb_trajectory = ung_M_N_trajectories.min(axis=0)
            ung_ub_trajectory = ung_M_N_trajectories.max(axis=0)

            gui_M_N_slices = get_slices(unguided_xr, partition, var, level)
            gui_M_N_trajectories = get_masked_mean(gui_M_N_slices, mask)
            gui_mean_trajectory = gui_M_N_trajectories.mean(axis=0)
            gui_m_trajectory = gui_M_N_trajectories[m]
        case _:
            pass

    # gt
    gt_rollout = get_gt_rollout(N+1, timestamp)
    gt_N_slices = get_slices(gt_rollout, partition, var, level)
    gt_trajectory = get_masked_mean(gt_N_slices, mask)
    return (
        gt_N_slices,
        gt_trajectory,
        ung_M_N_trajectories,
        ung_lb_trajectory,
        ung_m_trajectory,
        ung_mean_trajectory,
        ung_ub_trajectory,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(gt_trajectory, ung_lb_trajectory, ung_m_trajectory, ung_ub_trajectory):
    # connect ui to data
    reference_trajectories = {
        "unguided_members": ung_m_trajectory,
        "ground_truth": gt_trajectory,
        "lower_boundary": ung_lb_trajectory,
        "upper_boundary": ung_ub_trajectory
    }
    return (reference_trajectories,)


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
    n=n_slider.value
    m=m_slider.value
    return level, m, n, var


@app.cell
def _(PARTITIONS, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(LEVELS_DICT, VARIABLES_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition]
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level: ", show_value=True, debounce=True)
    VARIABLES = VARIABLES_DICT[partition]
    if partition == "surface":
        DEFAULT_VAR_VALUE = VARIABLES[2]
    else:
        DEFAULT_VAR_VALUE = VARIABLES[3]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=DEFAULT_VAR_VALUE, label="variable : ")
    return level_slider, var_dropdown


@app.cell
def _(get_level_idx, get_var_idx, level, partition, var):
    var_idx = get_var_idx(partition, var)
    level_idx = get_level_idx(partition, level)
    return level_idx, var_idx


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
        start=0, 
        stop=N,
        step=1,
        label="n: ",
        value=0,
        debounce=True,
        show_value=True
    )

    m_slider = mo.ui.slider(
        start=0, 
        stop=M-1,
        step=1,
        label="m: ",
        value=0,
        debounce=True,
        show_value=True
    )
    return m_slider, n_slider


@app.cell
def _(MASK_MODES, mo):
    mask_mode_dropdown = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
    return (mask_mode_dropdown,)


@app.cell
def _(mask_mode_dropdown):
    mask_mode=mask_mode_dropdown.value
    return (mask_mode,)


@app.cell
def _():
    # lambda_w_slider = mo.ui.slider(1, 3, value=1.0, label="w: ", step=1, show_value=True, debounce=True)
    # lambda_shape_slider = mo.ui.slider(1.0, 3.0, step=1, value=1.0, label="$\\alpha$: ", show_value=True, debounce=True)
    return


@app.cell
def _():
    # lambda_schedule = T_schedule(lambda_shape_slider.value, lambda_w_slider.value) 
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
    )
    return delta_peak_at_slider, delta_peak_slider, delta_shape_slider


@app.cell
def _(delta_peak_slider):
    delta_peak = delta_peak_slider.value / 100
    return (delta_peak,)


@app.cell
def _(guidance_reference_dropdown):
    guidance_reference = guidance_reference_dropdown.value
    return (guidance_reference,)


@app.cell
def _():
    # presentation widgets
    return


@app.cell
def _(
    M_slider,
    N_slider,
    day_slider,
    delta_bounds_slider,
    delta_granularity_slider,
    delta_peak_at_slider,
    delta_peak_slider,
    delta_shape_slider,
    guidance_reference_dropdown,
    hour_slider,
    level_slider,
    m_slider,
    mask_map,
    mask_mode_dropdown,
    mo,
    month_slider,
    n_slider,
    notebook_mode,
    partition_dropdown,
    trajectories_plot,
    var_dropdown,
    weather_map,
):
    timestamp_widget=mo.hstack([month_slider, day_slider, hour_slider], justify="start")
    M_N_widget = mo.hstack([M_slider, N_slider], justify="start")
    m_n_widget = mo.hstack([m_slider, n_slider], justify="start")
    mask_widget = mo.vstack([
        mask_mode_dropdown, 
        mo.hstack([
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        mo.hstack([weather_map, mask_map], justify="start")
    ])
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
                timestamp_widget,
                M_N_widget,
                m_n_widget,
                trajectories_plot
            ])
            mask_widget=mask_widget
        case "guided_rollout":
            trajectory_widget=mo.vstack([
                guidance_reference_dropdown,
                m_n_widget,
                delta_widget,
                trajectories_plot
            ])
            mask_widget=mask_widget
        case "analyze_rollout":        
            trajectory_widget=mo.vstack([
                m_n_widget,
                trajectories_plot
            ])
            mask_widget=None
        case _:
            pass
    return mask_widget, trajectory_widget


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trajectories
    """)
    return


@app.cell
def _(trajectory_widget):
    trajectory_widget
    return


@app.cell
def _(
    delta_trajectory,
    gt_trajectory,
    m,
    n,
    plot_dual_trajectory,
    reference_trajectory,
    target_trajectory,
    timestamps,
    ung_M_N_trajectories,
    ung_m_trajectory,
    ung_mean_trajectory,
    var,
):
    """
    refactor to include all of this:
    ung_M_N_trajectories
    ung_mean_trajectory
    ung_m_trajectory

    gui_M_N_trajectories
    gui_mean_trajectory
    gui_m_trajectory

    gt_trajectory
    """
    trajectories_plot = plot_dual_trajectory(
        timestamps=timestamps,
        var=var,
        unguided_member=ung_m_trajectory,
        reference_trajectory=reference_trajectory,
        m=m,
        n=n,
        mean_rollout=ung_mean_trajectory,
        ground_truth=gt_trajectory,
        planned_guidance=target_trajectory,
        y_trajectory=delta_trajectory,
        ensemble_rollout=ung_M_N_trajectories,
        ymin_left=None,
        ymax_left=None, # ... ?
        figsize=(17.5, 5.5),
        title=f"{var} trajectory",
        subtitle="mask weighted average",
        dpi=500
    )
    return (trajectories_plot,)


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
    gt_N_slices,
    map_interactive,
    mask_corners,
    n,
    np,
    set_corners,
    visualize_map,
):
    weather_map = visualize_map(
        gt_N_slices[n],
        title="Mask region",
        interactive=map_interactive,
        vmin=np.min(gt_N_slices),
        vmax=np.max(gt_N_slices),
        center=np.mean(gt_N_slices),
        # mask_corners=mask_corners, # TODO: simplify to this
        rectangle_x=(mask_corners[0], mask_corners[1]),
        rectangle_y=(mask_corners[2], mask_corners[3]),
        figsize=(12, 5),
        dpi=200,
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
def _(mask, mask_corners, np, visualize_map):
    mask_map = visualize_map(
        mask,
        title="Mask",
        interactive=False,
        vmin=np.min(mask),
        vmax=np.max(mask),
        center=np.mean(mask),
        rectangle_x=(mask_corners[0], mask_corners[1]),
        rectangle_y=(mask_corners[2], mask_corners[3]),
        figsize=(14, 8),
        dpi=500,
    )
    return (mask_map,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Flow
    """)
    return


@app.cell
def _(T_schedule, alpha_slider, plot_trajectory, w_slider):
    alpha = alpha_slider.value
    w=w_slider.value
    lambda_trajectory = T_schedule(alpha, w)
    lambda_trajectory_plot = plot_trajectory(lambda_trajectory, "$\lambda_t$", title="$\lambda_t$ schedule")
    return alpha, lambda_trajectory_plot, w


@app.cell
def _(alpha_slider, lambda_trajectory_plot, mo, w_slider):
    mo.vstack([
        mo.hstack([
            alpha_slider, w_slider
        ], justify="start"),
        lambda_trajectory_plot,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - gradient and vector field plots
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
