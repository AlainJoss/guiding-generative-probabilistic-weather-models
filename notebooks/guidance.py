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
    - Really need to think about how to compare experiments by putting everything in a single netcdf or better zarr s.t. we can append online when sweeping experiments
    - What happens when you stop rolling out? ...
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
    MASK_MODES,
    Path,
    ROLLOUTS,
    get_dict_from_json,
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
            experiment_params = get_dict_from_json(Path(ROLLOUTS, rollout_id, "sweep_params.json"))

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
    return (hash_params,)


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
            map_interactive = False
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
            mask_corners = config.mask_corners
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory = N_schedule(N, delta_shape_slider.value, delta_peak, peak_at_n=delta_peak_at_slider.value)
            reference_rollout = get_reference_rollouts(guidance_reference, rollout_id, N, M, timestamp)
            if (
                partition == config.partition
                and var == config.var
                and level == config.level
            ):
                target_rollout = get_target_rollout(
                    partition,
                    var_idx,
                    level_idx,
                    delta_trajectory[: N + 1],
                    reference_rollout,
                )
                target_slices = get_slices(target_rollout, partition, var, level)
                target_trajectories = get_masked_mean(target_slices, mask)
                target_trajectory = target_trajectories[m]
            else:
                target_rollout = None
                target_slices = None
                target_trajectories = None
                target_trajectory = None
        case "analyze_rollout":        
            mask_corners=config.mask_corners # should be a sweep param mask_corners_dropdown.value
            mu, sigma = get_mu_sigma(*mask_corners)
            mask=get_mask_2d(mask_mode, mask_corners)
            delta_trajectory=config.delta_trajectory
            reference_rollout = get_reference_rollouts(guidance_reference, rollout_id, N, M, timestamp)
            if (
                partition == config.partition
                and var == config.var
                and level == config.level
            ):
                target_rollout = get_target_rollout(
                    partition,
                    var_idx,
                    level_idx,
                    delta_trajectory[: N + 1],
                    reference_rollout,
                )
                target_slices = get_slices(target_rollout, partition, var, level)
                target_trajectories = get_masked_mean(target_slices, mask)
                target_trajectory = target_trajectories[m]
            else:
                target_rollout = None
                target_slices = None
                target_trajectories = None
                target_trajectory = None
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
    unguided_guided_xr,
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

            und_gui_m_trajectory = None
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

            und_gui_m_trajectory = None
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

            und_gui_M_N_slices = get_slices(unguided_guided_xr, partition, var, level)
            und_gui_M_N_trajectories = get_masked_mean(und_gui_M_N_slices, mask)
            und_gui_m_trajectory = np.concatenate([[np.nan], und_gui_M_N_trajectories[m]]) if len(und_gui_M_N_trajectories[m]) == N+1-1 else und_gui_M_N_trajectories[m]

            target_guidance_trajectory = (1 + np.asarray(delta_trajectory)) * und_gui_m_trajectory
            target_guidance_M_N_trajectories = (1 + np.asarray(delta_trajectory)) * und_gui_M_N_trajectories
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
        und_gui_m_trajectory,
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
def _(mask_mode_dropdown):
    mask_mode=mask_mode_dropdown.value
    return (mask_mode,)


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
    lambda_trajectory_plot,
    level_slider,
    m_slider,
    mask_map,
    mask_mode_dropdown,
    mo,
    month_slider,
    n_slider,
    notebook_mode,
    partition_dropdown,
    traj_checks,
    trajectories_plot,
    var_dropdown,
    weather_map,
):
    timestamp_widget=mo.hstack([month_slider, day_slider, hour_slider], justify="start")
    M_N_widget = mo.hstack([M_slider, N_slider], justify="start")
    m_n_widget = mo.hstack([m_slider, n_slider], justify="start")
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
                timestamp_widget,
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
    dpi_slider,
    gt_N_slices,
    level,
    map_interactive,
    mask_corners,
    n,
    np,
    partition,
    set_corners,
    timestamps,
    var,
    visualize_map,
):
    weather_map = visualize_map(
        gt_N_slices[n],
        suptitle=f"{timestamps[n]}",
        title=f"partition={partition} | var={var} | level={level}",
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
        ung_gt,
        ung_gt_prev,
        ung_prev,
        ung_ung,
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
    ung_gt,
    ung_gt_prev,
    ung_prev,
    ung_ung,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    if notebook_mode =="guided_rollout":
        if analysis_type_dropdown.value == "absolute":
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

            gt_gt_map = None
            gui_gui_map = None
            gui_ung_map = None
            gui_gt_map = None

        else:
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
        
    if notebook_mode =="analyze_rollout":
        if analysis_type_dropdown.value == "absolute":
            absolute_panels = [
                ("$x_n$", gt_curr),
                ("$x_{n-1}$", gt_prev),
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
            prev_map = absolute_maps["$x_{n-1}$"]
            ung_map = absolute_maps["$x_{n}^{ung}$"]
            gui_map = absolute_maps["$x_{n}^{gui}$"]

        else:
            difference_panels = [
                ("$x_{n} - x_{n-1}$", gt_gt),
                ("$x_{n}^{gui} - x_{n}$", gui_gt),
                ("$x_{n}^{gui} - x_{n}^{ung}$", gui_ung),
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
            gui_ung_map = difference_maps["$x_{n}^{gui} - x_{n}^{ung}$"]
            gui_gt_map = difference_maps["$x_{n}^{gui} - x_{n}$"]
    return (
        curr_map,
        gt_gt_map,
        gui_gt_map,
        gui_gui_map,
        gui_map,
        gui_ung_map,
        prev_map,
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

        if analysis_type_dropdown.value == "absolute":
            inspect_states_widget_make = mo.vstack(
                [
                    *common_controls,
                    mo.hstack([show_mask_switch, zoom_slider], justify="start"),
                    mo.hstack([curr_map, prev_map], justify="start"),
                    mo.hstack([ung_map, ung_prev_map], justify="start"),
                ],
                justify="start",
            )

        else:
            inspect_states_widget_make = mo.vstack(
                [
                    *common_controls,
                    mo.hstack([show_mask_switch, zoom_slider, norm_mode_dropdown, show_values_checkbox, text_thresh_slider], justify="start"),
                    mo.hstack([gt_gt_map, ung_ung_map], justify="start"),
                    mo.hstack([ung_ung_map, ung_gt_prev_map], justify="start")
                ], justify="start",
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

        if analysis_type_dropdown.value == "absolute":
            inspect_states_widget_make = mo.vstack(
                [
                    *common_controls,
                    mo.hstack([show_mask_switch, zoom_slider], justify="start"),
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
                    mo.hstack([gt_gt_map, gui_ung_map], justify="start"),
                    mo.hstack([gui_gt_map, gui_gui_map], justify="start")
                ], justify="start",
            )
    return (inspect_states_widget_make,)


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
def _(mo):
    traj_checks = mo.ui.dictionary({n: mo.ui.checkbox(label=n, value=True) for n in (
        "unguided", "guided", "unguided_guided", "planned_guidance", "target_guidance",
    )})
    return (traj_checks,)


@app.cell
def _(
    delta_trajectory,
    dpi_slider,
    gt_trajectory,
    gui_M_N_trajectories,
    gui_m_trajectory,
    m,
    n,
    notebook_mode,
    target_guidance_M_N_trajectories,
    target_guidance_trajectory,
    target_trajectories,
    target_trajectory,
    timestamps,
    traj_checks,
    und_gui_m_trajectory,
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
        guided_member=gui_m_trajectory if traj_checks["guided"].value else None,
        unguided_member=ung_m_trajectory if traj_checks["unguided"].value else None,
        unguided_guided_member=und_gui_m_trajectory if traj_checks["unguided_guided"].value else None,
        guided_ensemble=gui_M_N_trajectories if traj_checks["guided"].value else None,
        unguided_ensemble=ung_M_N_trajectories if traj_checks["unguided"].value else None,
        target_ensemble=target_trajectories if traj_checks["planned_guidance"].value else None,
        target_guidance_ensemble=target_guidance_M_N_trajectories if traj_checks["target_guidance"].value else None,
        target_trajectory=target_trajectory if traj_checks["planned_guidance"].value else None,
        target_guidance_trajectory=target_guidance_trajectory if traj_checks["target_guidance"].value else None,
        ground_truth=gt_trajectory,
        delta_trajectory=delta_trajectory if notebook_mode == "guided_rollout" else None,
        show_guided_mean=False,
        show_unguided_mean=False,
        title=f"rollout trajectories",
        subtitle="{var} | mask-averaged",
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
def _(T_schedule, alpha_slider, plot_trajectory, t, w_slider):
    alpha = alpha_slider.value
    w=w_slider.value
    lambda_trajectory = T_schedule(alpha, w)
    lambda_trajectory_plot = plot_trajectory(lambda_trajectory, "$\lambda_t$", title="$\lambda_t$ schedule", t=t, figsize=(17, 6))
    return alpha, lambda_trajectory_plot, w


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
def _(get_rollout_files, guided_id, notebook_mode, rollout_id):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        grads_xr, _ = get_rollout_files("grad", rollout_id, guided_id)
        vfs_xr, _ = get_rollout_files("vf", rollout_id, guided_id)
        clean_preds_xr, _ = get_rollout_files("clean_pred", rollout_id, guided_id)
        guided_vfs_xr, _ = get_rollout_files("guided_vf", rollout_id, guided_id)
        unguided_guided_xr, _ = get_rollout_files("unguided_guided_rollout", rollout_id, guided_id)
    return clean_preds_xr, grads_xr, guided_vfs_xr, unguided_guided_xr, vfs_xr


@app.cell
def _(
    N,
    clean_preds_xr,
    get_N_slices,
    grads_xr,
    gt_curr,
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
    unguided_guided_xr,
    var,
    vfs_xr,
):
    if notebook_mode not in ("unguided_rollout", "guided_rollout"):
        def get_trace_slices(trace_xr, gt=False):
            if gt:
                slices = get_N_slices(
                    trace_xr,
                    N+1,
                    timestamp,
                    partition,
                    var,
                    level,
                )
                return slices
            slices = get_N_slices(
                trace_xr,
                N,
                timestamp + timedelta(days=1),
                partition,
                var,
                level,
            )
            zeros = np.zeros_like(slices[:, :, None, 0])
            return np.concatenate([zeros, slices], axis=2)

        # fix over t
        ung_onl_slice = get_trace_slices(unguided_guided_xr, gt=True)
        ung_onl_curr = ung_onl_slice[m][n]

        # changes over t
        clean_preds_slices = get_trace_slices(clean_preds_xr)
        grads_slices = get_trace_slices(grads_xr)
        vfs_slices = get_trace_slices(vfs_xr)
        guided_vfs_slices = get_trace_slices(guided_vfs_xr)

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
            ("diff_gt_ung_onl_map", diff_gt_ung_onl_slice, "x_ung - gt", -1, 1),
            ("diff_gt_clean_pred_map", diff_gt_clean_pred_slice, "x_hat_t - gt", -1, 1),
            ("ung_onl_clean_diff_map", ung_onl_clean_diff_slice, "x_hat_t - x_ung", -1, 1),
            ("clean_preds_diff_map", clean_preds_diff_slice, "x_hat_t - x_hat_{t-1}", -1, 1),
            ("grads_map", grads_slice, "grad_t", -1, 1),
            ("diff_grads_map", diff_grads_slice, "grad_t - grad_{t-1}", -1, 1),
            ("vfs_map", vfs_slice, "vf_t", -0.001, 0.001),
            ("guided_vfs_map", guided_vfs_slice, "vf^guided_t", -0.001, 0.001),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grad norms
    """)
    return


@app.cell
def _(grads_xr, m, mo, n, np, plot_trajectory, t, t_slider):
    grad_norms_plot = plot_trajectory(
        {v: np.sqrt(((g := grads_xr[v].isel(member=m, time=n-1).values) ** 2).sum(  # time=n-1 because sample traces start at day after n=0
            axis=tuple(range(1, g.ndim))))
         for v in grads_xr.data_vars},
        title="Grad norms per variable",
        subtitle=f"member={m} | rollout step n={n} | levels aggregated",
        # right_trajectory=lambda_trajectory,
        t=t,
    )
    mo.vstack([t_slider, grad_norms_plot])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Guided vs unguided-guided difference
    """)
    return


@app.cell
def _(guided_xr, m, plot_trajectory, unguided_guided_xr):
    diffs = {
        v: (guided_xr[v] - unguided_guided_xr[v]).isel(member=m)
           .mean(dim=[k for k in guided_xr[v].dims if k not in ("time", "member")]).values
        for v in guided_xr.data_vars
    }
    plot_trajectory(
        {v: (d - d.min()) / (d.max() - d.min() + 1e-12) for v, d in diffs.items()},
        title="Normalized (guided − unguided_guided) per variable",
        subtitle=f"member={m} | level/lat/lon aggregated | min-max normalized per variable",
        xlabel="$n$"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
