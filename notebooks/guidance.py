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

    return Path, mo, np, timedelta


@app.cell
def _():
    from src.paths import ROLLOUTS, RUN_CONFIGS
    from src.rollout_config import GUIDANCE_REFERENCES, MASK_MODES, RolloutConfig
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.helpers import max_day, get_timestamp_from_sliders
    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory
    # from src.ui.plot_dual_trajectory import plot_dual_trajectory
    # from src.ui.analysis.analysis_plots import plot_guidance_tracking


    from src.utils.dataset_utils import get_x_cond
    from src.utils.converters import list_tensors_to_floats, get_var_idx, get_level_idx
    from src.utils.setup import get_now_timestamp
    from src.utils.dataset_utils import get_timestamps, get_N_timestamps, get_N_slices, get_slices, get_gt_rollout
    from src.utils.read_write import (
        dump_json, get_dict_from_json, get_rollout_ids, get_rollout_files
    )

    from src.funcs import N_schedule, T_schedule, make_hash

    from src.mask import get_masked_mean, get_mask_2d, get_mu_sigma
    from src.target import get_reference_rollout, get_target_rollout, get_reference_rollouts

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
        get_N_slices,
        get_N_timestamps,
        get_dict_from_json,
        get_gt_rollout,
        get_level_idx,
        get_mask_2d,
        get_masked_mean,
        get_mu_sigma,
        get_now_timestamp,
        get_reference_rollouts,
        get_rollout_files,
        get_rollout_ids,
        get_slices,
        get_target_rollout,
        get_timestamp_from_sliders,
        get_var_idx,
        make_hash,
        max_day,
        plot_trajectory,
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
    get_now_timestamp,
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
            save_id=get_now_timestamp() if notebook_mode=="unguided_rollout" else rollout_id
            save_config = RolloutConfig(
                rollout_id=save_id,
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
            dump_json(save_config.to_dict(), run_dir, f"{save_id}")
            print("saved config")
    return


@app.cell
def _(
    GUIDANCE_REFERENCES,
    Path,
    ROLLOUTS,
    get_dict_from_json,
    mask_mode_dropdown,
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
                guidance_reference_dropdown,mask_mode_dropdown,
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
                # "mask_mode": mask_mode_dropdown.value,
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
            guided_xr=None
            config=None
            # TODO: set everything to None
        case "guided_rollout":
            unguided_xr, config = get_rollout_files("unguided_rollout", rollout_id)
            guided_xr = None
        case "analyze_rollout":
            unguided_xr, _ = get_rollout_files("unguided_rollout", rollout_id)
            guided_id = make_hash(hash_params)
            # NOTE: enables to decrease N in guided_rollout config for testing purposes
            guided_xr, config = get_rollout_files("guided_rollout", rollout_id, guided_id)
        case _:
            pass
    return config, guided_id, guided_xr, unguided_xr


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
    get_reference_rollouts,
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
            target_trajectories=None
            target_trajectory=None
        case "guided_rollout":
            mask_corners = get_corners()
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory = N_schedule(N, delta_shape_slider.value, delta_peak, peak_at_n=delta_peak_at_slider.value)
            reference_rollout = get_reference_rollouts(guidance_reference, rollout_id, N, M, timestamp)
            target_rollout = get_target_rollout(
                partition, 
                var_idx,
                level_idx,
                delta_trajectory[:N+1],
                reference_rollout
            )
            target_slices = get_slices(target_rollout, partition, var, level)
            target_trajectories = get_masked_mean(target_slices, mask)
            target_trajectory = target_trajectories[m]
        case "analyze_rollout":        
            mask_corners=config.mask_corners # should be a sweep param mask_corners_dropdown.value
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory=config.delta_trajectory
            reference_rollout = get_reference_rollouts(guidance_reference, rollout_id, N, M, timestamp)
            target_rollout = get_target_rollout(
                partition, 
                var_idx,
                level_idx,
                delta_trajectory,
                reference_rollout
            )
            target_slices = get_slices(target_rollout, partition, var, level)
            target_trajectories = get_masked_mean(target_slices, mask)
            target_trajectory = target_trajectories[m]
        case _:
            pass
    return (
        delta_trajectory,
        mask,
        mask_corners,
        target_trajectories,
        target_trajectory,
    )


@app.cell
def _(
    N,
    get_gt_rollout,
    get_masked_mean,
    get_slices,
    guided_xr,
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

            gui_M_N_slices = get_slices(guided_xr, partition, var, level)
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
        gui_M_N_slices,
        gui_M_N_trajectories,
        gui_m_trajectory,
        ung_M_N_slices,
        ung_M_N_trajectories,
        ung_lb_trajectory,
        ung_m_trajectory,
        ung_ub_trajectory,
    )


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
    inspect_states_widget_make,
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
            inspect_states_widget=None
        case "guided_rollout":
            trajectory_widget=mo.vstack([
                guidance_reference_dropdown,
                m_n_widget,
                delta_widget,
                trajectories_plot
            ])
            mask_widget=mask_widget
            inspect_states_widget=None
        case "analyze_rollout":        
            trajectory_widget=mo.vstack([
                m_n_widget,
                trajectories_plot
            ])
            mask_widget=mask_widget
            inspect_states_widget=inspect_states_widget_make
        case _:
            pass
    return inspect_states_widget, mask_widget, trajectory_widget


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
    gt_trajectory,
    gui_M_N_trajectories,
    gui_m_trajectory,
    m,
    n,
    target_trajectories,
    target_trajectory,
    timestamps,
    ung_M_N_trajectories,
    ung_m_trajectory,
    var,
):
    from src.ui.plot_trajectories import plot_trajectories

    trajectories_plot = plot_trajectories(
        timestamps=timestamps,
        var=var,
        m=m,
        n=n,
        guided_member=gui_m_trajectory,
        unguided_member=ung_m_trajectory,
        guided_ensemble=gui_M_N_trajectories,
        unguided_ensemble=ung_M_N_trajectories,
        target_ensemble=target_trajectories,
        target_trajectory=target_trajectory,
        ground_truth=gt_trajectory,
        delta_trajectory=None, # delta_trajectory,
        show_guided_mean=False,
        show_unguided_mean=False,
        title=f"Realized guidance - {var}",
        subtitle="Guided vs unguided mask-averaged trajectory",
        ylabel="Mask-averaged value",
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
    dpi_slider,
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
        # figsize=(12, 5),
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
def _(dpi_slider, mask, mask_corners, np, visualize_map):
    mask_map = visualize_map(
        mask,
        title="Mask",
        interactive=False,
        vmin=np.min(mask),
        vmax=np.max(mask),
        center=np.mean(mask),
        rectangle_x=(mask_corners[0], mask_corners[1]),
        rectangle_y=(mask_corners[2], mask_corners[3]),
        # figsize=(14, 8),
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
    analysis_types = ["absolute", "difference"]
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
    )
    return (text_thresh_slider,)


@app.cell
def _(mask_corners):
    from src.funcs import safe_abs_limits
    from src.mask import get_mask_center
    zoom_centers = get_mask_center(*mask_corners)
    return safe_abs_limits, zoom_centers


@app.cell
def _(gt_N_slices, gui_M_N_slices, m, n, notebook_mode, ung_M_N_slices):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        ung_curr = ung_M_N_slices[m][n]
        gui_curr = gui_M_N_slices[m][n]
        gt_curr = gt_N_slices[n]
        # ung_prev = ung_M_N_slices[m][n-1] if n>0 else ung_M_N_slices[m][n]
        gui_prev = gui_M_N_slices[m][n-1] if n>0 else gui_M_N_slices[m][n]
        gt_prev = gt_N_slices[n-1] if n>0 else gt_N_slices[n]

        gt_gt = gt_curr - gt_prev
        gui_gui = gui_curr - gui_prev
        gui_ung = gui_curr - ung_curr
        gui_gt = gui_curr - gt_curr
    return (
        gt_curr,
        gt_gt,
        gt_prev,
        gui_curr,
        gui_gt,
        gui_gui,
        gui_ung,
        ung_curr,
    )


@app.cell
def _(
    analysis_type_dropdown,
    dpi_slider,
    gt_curr,
    gt_gt,
    gt_prev,
    gui_curr,
    gui_gt,
    gui_gui,
    gui_ung,
    map_interactive,
    mask,
    norm_mode_dropdown,
    notebook_mode,
    np,
    safe_abs_limits,
    show_mask_switch,
    show_values_checkbox,
    text_thresh_slider,
    ung_curr,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        if analysis_type_dropdown.value == "absolute":
            absolute_panels = [
                ("$x_t$", gt_curr),
                ("$x_{t+1}$", gt_prev),
                ("$x_{t+1}^{unguided}$", ung_curr),
                ("$x_{t+1}^{guided}$", gui_curr),
            ]
    
            abs_vmin, abs_vmax, abs_center = safe_abs_limits(
                [arr for _, arr in absolute_panels]
            )
    
            absolute_maps = {}
    
            for label, arr in absolute_panels:
                absolute_maps[label] = visualize_map(
                    arr,
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
                )
    
            curr_map = absolute_maps["$x_t$"]
            prev_map = absolute_maps["$x_{t+1}$"]
            ung_map = absolute_maps["$x_{t+1}^{unguided}$"]
            gui_map = absolute_maps["$x_{t+1}^{guided}$"]
    
            gt_gt_map = None
            gui_gui_map = None
            gui_ung_map = None
            gui_gt_map = None
    
        else:
            difference_panels = [
                ("$x_{t+1} - x_t$", gt_gt),
                ("$x_{t+1}^{guided} - x_{t+1}$", gui_gt),
                ("$x_{t+1}^{guided} - x_{t+1}^{unguided}$", gui_ung),
                ("$x_{t+1}^{guided} - x_{t}^{guided}$", gui_gui),
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
                )
    
            gt_gt_map = difference_maps["$x_{t+1} - x_t$"]
            gui_gui_map = difference_maps["$x_{t+1}^{guided} - x_{t}^{guided}$"]
            gui_ung_map = difference_maps["$x_{t+1}^{guided} - x_{t+1}^{unguided}$"]
            gui_gt_map = difference_maps["$x_{t+1}^{guided} - x_{t+1}$"]
    
            curr_map = None
            prev_map = None
            ung_map = None
            gui_map = None
    return (
        curr_map,
        gt_gt_map,
        gui_gt_map,
        gui_gui_map,
        gui_map,
        gui_ung_map,
        prev_map,
        ung_map,
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
    text_thresh_slider,
    ung_map,
    var_dropdown,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        common_controls = [
            mo.hstack([analysis_type_dropdown, dpi_slider], justify="start"),
            mo.hstack([n_slider, m_slider], justify="start"),
            mo.hstack(
                [partition_dropdown, var_dropdown, level_slider],
                justify="start",
            ),
        ]
    
        if analysis_type_dropdown.value == "absolute":
            inspect_states_widget_make = mo.vstack(
                [
                    *common_controls,
                    mo.hstack([show_mask_switch, zoom_slider], justify="start"),
                    mo.md("Absolute states:"),
                    mo.hstack([curr_map, prev_map], justify="start"),
                    mo.hstack([gui_map, ung_map], justify="start"),
                ],
                justify="start",
            )
    
        else:
            inspect_states_widget_make = mo.vstack(
                [
                    *common_controls,
                    mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, show_values_checkbox, text_thresh_slider], justify="start"),
                    mo.md("Difference over states:"),
                    mo.hstack([gt_gt_map, gui_ung_map], justify="start"),
                    mo.hstack([gui_gt_map, gui_gui_map], justify="start")
                ], justify="start",
            )
    return (inspect_states_widget_make,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Flow analysis
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
def _(mo):
    t_slider = mo.ui.slider(
        steps=range(25),
        value=0,
        label="t: ",
        debounce=True,
        show_value=True
    )
    return (t_slider,)


@app.cell
def _(alpha_slider, lambda_trajectory_plot, mo, t_slider, w_slider):
    mo.vstack([
        t_slider,
        mo.hstack([
            alpha_slider, w_slider
        ], justify="start"),
        lambda_trajectory_plot,
    ])
    return


@app.cell
def _(get_rollout_files, guided_id, notebook_mode, rollout_id):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        grads_xr, _ = get_rollout_files("grads", rollout_id, guided_id)
        vfs_xr, _ = get_rollout_files("vfs", rollout_id, guided_id)
        clean_preds_xr, _ = get_rollout_files("clean_preds", rollout_id, guided_id)
        guided_vfs_xr, _ = get_rollout_files("guided_vfs", rollout_id, guided_id)
    return clean_preds_xr, grads_xr, guided_vfs_xr, vfs_xr


@app.cell
def _(
    N,
    clean_preds_xr,
    get_N_slices,
    grads_xr,
    guided_vfs_xr,
    level,
    m,
    n,
    notebook_mode,
    np,
    partition,
    t,
    timedelta,
    timestamp,
    var,
    vfs_xr,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        def get_trace_slices(trace_xr):
            slices = get_N_slices(
                trace_xr,
                N,
                timestamp + timedelta(days=1),
                partition,
                var,
                level,
            )
            # print(slices.shape)
            zeros = np.zeros_like(slices[:, :, None, 0])
            # print(zeros.shape)
            return np.concatenate([zeros, slices], axis=2)
    
    
        clean_preds_slices = get_trace_slices(clean_preds_xr)
        grads_slices = get_trace_slices(grads_xr)
        vfs_slices = get_trace_slices(vfs_xr)
        guided_vfs_slices = get_trace_slices(guided_vfs_xr)
        diff_vfs_slices = guided_vfs_slices - vfs_slices
    
        clean_preds_slice = clean_preds_slices[m][t][n]
        grads_slice = grads_slices[m][t][n]
        grads_slice_prev_slice = grads_slices[m][t-1][n] if t>0 else grads_slices[m][t][n]
        vfs_slice = vfs_slices[m][t][n]
        guided_vfs_slice = guided_vfs_slices[m][t][n]
        diff_vfs_slice = diff_vfs_slices[m][t][n]
        diff_grads_slice = grads_slice- grads_slice_prev_slice
    return (
        clean_preds_slice,
        diff_grads_slice,
        diff_vfs_slice,
        grads_slice,
        guided_vfs_slice,
        vfs_slice,
    )


@app.cell
def _(t_slider):
    t=t_slider.value
    return (t,)


@app.cell
def _(
    clean_preds_slice,
    diff_grads_slice,
    diff_vfs_slice,
    dpi_slider,
    grads_slice,
    guided_vfs_slice,
    mask_corners,
    n,
    notebook_mode,
    np,
    vfs_slice,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        grads_map = visualize_map(
            grads_slice,
            title="grad_t",
            interactive=False,
            vmin=np.min(grads_slice) if n>0 else -1,
            vmax=np.max(grads_slice) if n>0 else 1,
            center=np.mean(grads_slice) if n>0 else 0,
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
        diff_vfs_map = visualize_map(
            diff_vfs_slice,
            title="vf^guided_t - vf_t",
            interactive=False,
            vmin=np.min(diff_vfs_slice) if np.min(diff_vfs_slice)<0 else -0.001,
            vmax=np.max(diff_vfs_slice) if np.max(diff_vfs_slice)>0 else 0.001,
            center=np.mean(diff_vfs_slice) if n>0 else 0,
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    
        vfs_map = visualize_map(
            vfs_slice,
            title="vf_t",
            interactive=False,
            vmin=np.min(vfs_slice) if n>0 else -0.001,
            vmax=np.max(vfs_slice) if n>0 else 0.001,
            center=np.mean(vfs_slice) if n>0 else 0,
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
        clean_preds_map = visualize_map(
            clean_preds_slice,
            title="x_hat_t",
            interactive=False,
            vmin=np.min(clean_preds_slice) if n>0 else -1,
            vmax=np.max(clean_preds_slice) if n>0 else 1,
            center=np.mean(clean_preds_slice) if n>0 else 0,
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
        guided_vfs_map = visualize_map(
            guided_vfs_slice,
            title="vf^guided_t",
            interactive=False,
            vmin=np.min(guided_vfs_slice) if n>0 else -1,
            vmax=np.max(guided_vfs_slice) if n>0 else 1,
            center=np.mean(guided_vfs_slice) if n>0 else 0,
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
        diff_grads_map = visualize_map(
            diff_grads_slice,
            title="grad_t - grad_{t-1}",
            interactive=False,
            vmin=np.min(diff_grads_slice) if np.min(diff_grads_slice)<0 else -1,
            vmax=np.max(diff_grads_slice) if np.max(diff_grads_slice)>0 else 1,
            center=np.mean(diff_grads_slice) if n>0 else 0,
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    return (
        clean_preds_map,
        diff_grads_map,
        diff_vfs_map,
        grads_map,
        guided_vfs_map,
        vfs_map,
    )


@app.cell
def _(
    dpi_slider,
    gt_N_slices,
    map_interactive,
    mask,
    mask_corners,
    n,
    notebook_mode,
    np,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        flow_weather_map = visualize_map(
            gt_N_slices[n],
            title="gt",
            interactive=map_interactive,
            vmin=np.min(gt_N_slices),
            vmax=np.max(gt_N_slices),
            center=np.mean(gt_N_slices),
            # mask_corners=mask_corners, # TODO: simplify to this
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
        flow_mask_map = visualize_map(
            mask,
            title="mask",
            interactive=False,
            vmin=np.min(mask),
            vmax=np.max(mask),
            center=np.mean(mask),
            rectangle_x=(mask_corners[0], mask_corners[1]),
            rectangle_y=(mask_corners[2], mask_corners[3]),
            figsize=(14, 8),
            dpi=dpi_slider.value,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    return flow_mask_map, flow_weather_map


@app.cell
def _(
    clean_preds_map,
    diff_grads_map,
    diff_vfs_map,
    dpi_slider,
    flow_mask_map,
    flow_weather_map,
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
    var_dropdown,
    vfs_map,
    zoom_slider,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        flow_controls = [
            mo.hstack([t_slider, dpi_slider], justify="start"),
            mo.hstack([n_slider, m_slider], justify="start"),
            mo.hstack(
                [partition_dropdown, var_dropdown, level_slider],
                justify="start",
            ),
        ]
    
        flow_widget_make = mo.vstack(
            [
                *flow_controls,
                mo.hstack([show_mask_switch, zoom_slider], justify="start"),
                # mo.md("Absolute states:"),
                mo.vstack([
                    mo.hstack([
                        flow_weather_map, flow_mask_map
                    ], justify="start"),
                    mo.hstack([
                        clean_preds_map, diff_grads_map
                    ], justify="start"),
                    mo.hstack([
                        diff_vfs_map, grads_map
                    ], justify="start"),
                    mo.hstack([
                        vfs_map, guided_vfs_map, 
                    ], justify="start")
                ])
            ],
            justify="start",
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


@app.cell
def _():
    # should check the shape of grads and the other guys and how to align them, because diff doesnpt make sense
    return


if __name__ == "__main__":
    app.run()
