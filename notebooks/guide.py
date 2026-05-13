import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", css_file="")


@app.cell
def _():
    import marimo as mo
    import torch

    return mo, torch


@app.cell
def _():
    from src.ui.interaction import (
        visualize_map, 
        get_mask_from_corners,
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory
    from src.funcs import avg_over_mask, get_guidance_trajectory, N_schedule, T_schedule, compute_mean_rollout
    from src.utils import (
        read_nc,
        get_dataset, get_model,
        save_to_json, read_json,
        get_x_cond,
        get_device,
        list_tensors_to_floats
    )
    from src.config import PARTITIONS, LEVELS_DICT, VARIABLES_DICT, GUIDANCE_MODES
    from src.utils import get_rollout_dir, get_experiment_ids

    return (
        GUIDANCE_MODES,
        LEVELS_DICT,
        N_schedule,
        PARTITIONS,
        T_schedule,
        VARIABLES_DICT,
        avg_over_mask,
        compute_mean_rollout,
        get_dataset,
        get_device,
        get_experiment_ids,
        get_guidance_trajectory,
        get_mask_from_corners,
        get_model,
        get_rollout_dir,
        get_x_cond,
        list_tensors_to_floats,
        plot_dual_trajectory,
        plot_trajectory,
        read_json,
        read_nc,
        save_to_json,
        visualize_map,
    )


@app.cell
def _(get_dataset, get_device, get_model):
    device = get_device()
    ds = get_dataset()
    model = get_model(device)
    return (ds,)


@app.cell
def _(PARTITIONS, config, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=config["partition"], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(LEVELS_DICT, config, mo, partition):
    LEVELS = LEVELS_DICT[partition]
    level_slider = mo.ui.slider(steps=LEVELS, value=config["level"], label="level: ", show_value=True, debounce=True)
    return LEVELS, level_slider


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, config, mo, partition):
    VARIABLES = VARIABLES_DICT[partition]
    VARIABLE_DEFAULT = config["var"] if partition == config["partition"] else VARIABLES[0]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLE_DEFAULT, label="variable: ")
    return VARIABLES, var_dropdown


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(mo):
    w_slider = mo.ui.slider(1, 3, value=1.0, label="w: ", step=1, show_value=True, debounce=True)
    lambda_shape_slider = mo.ui.slider(1.0, 3.0, step=1, value=1.0, label="$\\alpha$: ", show_value=True, debounce=True)
    return lambda_shape_slider, w_slider


@app.cell
def _(T_schedule, lambda_shape_slider, w_slider):
    lambda_ = T_schedule(lambda_shape_slider.value, w_slider.value) 
    return (lambda_,)


@app.cell
def _(mo):
    min_max_lambda_slider = mo.ui.slider(
        steps=[5, 10, 25, 50, 100], value=10, label="bounds (%): ", show_value=True
    )
    return (min_max_lambda_slider,)


@app.cell
def _(mo):
    max_perc_granularity_slider = mo.ui.slider(
        steps=[0.1, 0.25, 0.5, 1, 2, 5],
        value=1,
        label="granularity (%): ",
        show_value=True,
    )
    return (max_perc_granularity_slider,)


@app.cell
def _(mo):
    peak_granularity_slider = mo.ui.slider(
        steps=[0.1, 0.5, 1, 2, 5],
        value=1,
        label="granularity (%): ",
        show_value=True,
    )
    return (peak_granularity_slider,)


@app.cell
def _(min_max_lambda, mo, peak_granularity_slider):
    peak_slider = mo.ui.slider(
        -min_max_lambda,
        min_max_lambda,
        value=0,
        step=peak_granularity_slider.value,
        label="peak (%): ",
        show_value=True,
    )
    return (peak_slider,)


@app.cell
def _(peak_slider):
    peak = peak_slider.value / 100
    return (peak,)


@app.cell
def _(mo):
    shape_slider = mo.ui.slider(
        start=0.5,
        stop=10.0,
        step=0.5,
        value=0.5,
        label="shape: ",
        show_value=True,
    )
    return (shape_slider,)


@app.cell
def _(N, mo):
    peak_at_slider = mo.ui.slider(
        start=1,
        stop=max(N, 1),
        step=1,
        value=N // 2,
        label="peak at n: ",
        show_value=True,
        debounce=True,
    )
    return (peak_at_slider,)


@app.cell
def _(min_max_lambda_slider):
    min_max_lambda = min_max_lambda_slider.value
    return (min_max_lambda,)


@app.cell
def _(max_perc_granularity_slider, min_max_lambda, mo):
    alpha_slider = mo.ui.slider(
        -min_max_lambda,
        min_max_lambda,
        value=0,
        step=max_perc_granularity_slider.value,
        label="max percentage change (%): ",
        show_value=True,
    )
    return (alpha_slider,)


@app.cell
def _(alpha_slider):
    alpha = alpha_slider.value / 100
    return (alpha,)


@app.cell
def _(GUIDANCE_MODES, mo):
    guidance_mode_dropdown = mo.ui.dropdown(GUIDANCE_MODES, value=GUIDANCE_MODES[0], label="guidance mode: ")
    return (guidance_mode_dropdown,)


@app.cell
def _(ground_truth, lower_boundary, mean_unguided_rollout, mo, upper_boundary):
    reference_rollouts = {
        "mean_unguided_rollout": mean_unguided_rollout,
        "ground_truth": ground_truth,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary
    }
    reference_rollout = mo.ui.dropdown(options=reference_rollouts.keys(), value=list(reference_rollouts.keys())[0], label="reference: ")
    return reference_rollout, reference_rollouts


@app.cell
def _(reference_rollout, reference_rollouts):
    reference = reference_rollouts[reference_rollout.value]
    return (reference,)


@app.cell
def _(get_ensemble_upper_bound, unguided_rollout):
    lower_boundary, upper_boundary = get_ensemble_upper_bound(unguided_rollout)
    return lower_boundary, upper_boundary


@app.cell
def _(
    GUIDANCE_MODES,
    N,
    N_schedule,
    alpha,
    get_guidance_trajectory,
    guidance_mode_dropdown,
    peak,
    peak_at_slider,
    reference,
    shape_slider,
):
    planned_guidance = reference  # overwritten if planned_trajectory
    y_trajectory = None
    if guidance_mode_dropdown.value == GUIDANCE_MODES[0]:
        y_trajectory = N_schedule(
            N, shape_slider.value, peak, alpha, peak_at_n=peak_at_slider.value
        )
        planned_guidance = get_guidance_trajectory(
            y_trajectory, reference
        )
    return planned_guidance, y_trajectory


@app.cell
def _(torch):
    def get_ensemble_upper_bound(unguided_rollout):
        rows = [[float(v) for v in row] for row in unguided_rollout]
        M = min(len(row) for row in rows)
        trimmed = [row[:M] for row in rows]
        lower_boundary = [torch.tensor(min(row)) for row in trimmed]
        upper_boundary = [torch.tensor(max(row)) for row in trimmed]
        return lower_boundary, upper_boundary

    return (get_ensemble_upper_bound,)


@app.cell
def _(config):
    timestamp = config["timestamp"]
    M = config["M"]
    N = config["N"]
    return M, N, timestamp


@app.cell
def _(LEVELS, VARIABLES, ds, get_x_cond, level, timestamp, var):
    var_idx = VARIABLES.index(var)
    level_idx = LEVELS.index(level)
    x_start, timestamp_idx = get_x_cond(ds, timestamp)
    return level_idx, timestamp_idx, var_idx, x_start


@app.cell
def _(config, get_mask_from_corners):
    mask_corners = tuple(config["mask_corners"])
    mask = get_mask_from_corners(*mask_corners)
    return mask, mask_corners


@app.cell
def _(ds, level_idx, partition, var_idx, x_start):
    slice = ds.denormalize(x_start["state"])[partition][var_idx, level_idx]
    return (slice,)


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="refresh")
    return (refresh_button,)


@app.cell
def _(get_experiment_ids, mo, refresh_button):
    if refresh_button.value:
        pass

    unguided_rollouts = get_experiment_ids("unguided")
    pick_unguided_rollout_dropdown = mo.ui.dropdown(
        label="Experiment: ", value=unguided_rollouts[0], options=unguided_rollouts
    )
    return (pick_unguided_rollout_dropdown,)


@app.cell
def _(get_rollout_dir, pick_unguided_rollout_dropdown, read_json):
    rollout_dir = get_rollout_dir(pick_unguided_rollout_dropdown.value)
    config = read_json(rollout_dir, "config")
    return config, rollout_dir


@app.cell
def _(read_nc, rollout_dir):
    ground_truth_xr = read_nc(rollout_dir, "ground_truth")
    unguided_xr = read_nc(rollout_dir ,"unguided")
    return ground_truth_xr, unguided_xr


@app.cell
def _(config, mo, pick_unguided_rollout_dropdown):
    experiment_dropdown = mo.vstack([
        mo.md("Pick unguided rollout experiment."),
        mo.hstack([
            pick_unguided_rollout_dropdown
        ], justify="start"),
        mo.accordion(
            {
                "Experiment params": mo.md("<br>".join(f"{k}: {v}" for k, v in config.items()))
            }
        )
    ])
    return (experiment_dropdown,)


@app.cell
def _(
    avg_over_mask,
    compute_mean_rollout,
    ground_truth_xr,
    level,
    mask,
    torch,
    unguided_xr,
    var,
):
    def xr_slice_to_torch(xr_ds, var, timestamp, level=None):
        da = xr_ds[var].sel(time=timestamp)

        if "level" in da.dims and level is not None:
            da = da.sel(level=int(level))

        return torch.tensor(da.values)


    def avg_xr_over_mask(xr_ds, var, timestamp, mask, level=None, member=None):
        da = xr_ds[var].sel(time=timestamp)

        if member is not None and "member" in da.dims:
            da = da.sel(member=member)

        if "level" in da.dims and level is not None:
            da = da.sel(level=int(level))

        x = torch.tensor(da.values, dtype=mask.dtype)
        return avg_over_mask(x, mask)


    # timestamps to compare:
    # ground_truth has init + N future times
    # unguided has only N future times
    # timestamps to compare:
    # ground_truth has init + N future times
    # unguided has only N future times
    members = list(unguided_xr.member.values)

    ground_truth = []
    unguided_rollout = []

    init_timestamp = ground_truth_xr.time.values[0]
    future_timestamps = list(unguided_xr.time.values)

    # Use this one for plotting
    timestamps = [init_timestamp] + future_timestamps

    # step 0: only ground truth exists
    init_avg = avg_xr_over_mask(
        ground_truth_xr,
        var=var,
        timestamp=init_timestamp,
        mask=mask,
        level=level,
    )

    ground_truth.append(init_avg)
    unguided_rollout.append([init_avg] * len(members))

    # steps 1..N: both ground_truth and unguided exist
    for timestamp_n in future_timestamps:
        gt_avg = avg_xr_over_mask(
            ground_truth_xr,
            var=var,
            timestamp=timestamp_n,
            mask=mask,
            level=level,
        )
        ground_truth.append(gt_avg)

        unguided_avgs = [
            avg_xr_over_mask(
                unguided_xr,
                var=var,
                timestamp=timestamp_n,
                mask=mask,
                level=level,
                member=m,
            )
            for m in members
        ]

        unguided_rollout.append(unguided_avgs)

    mean_unguided_rollout = compute_mean_rollout(unguided_rollout)
    return ground_truth, mean_unguided_rollout, timestamps, unguided_rollout


@app.cell
def _(
    ground_truth,
    mean_unguided_rollout,
    planned_guidance,
    plot_dual_trajectory,
    timestamps,
    unguided_rollout,
    var,
    y_trajectory,
):
    y_trajectory_plot = plot_dual_trajectory(
        timestamps=timestamps,
        mean_rollout=mean_unguided_rollout,
        ground_truth=ground_truth,
        planned_guidance=planned_guidance,
        y_trajectory=y_trajectory,
        var=var,
        ensemble_rollout=unguided_rollout,
        ymin_left=None,
        ymax_left=None,
        figsize=(17.5, 5.5),
        title=f"{var} trajectory",
        subtitle="Unguided rollout ensemble, planned guidance, ground truth, and percentage trajectory",
        dpi=500
    )
    return (y_trajectory_plot,)


@app.cell
def _():
    from src.ui.analysis.analysis_plots import plot_guidance_tracking

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PHASE 2 - guided rollout
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick unguided rollout
    """)
    return


@app.cell
def _(mo, refresh_button):
    mo.vstack([
        mo.md("Reload whole page if something fails."),
        refresh_button
    ])
    return


@app.cell
def _(experiment_dropdown):
    experiment_dropdown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure guidance experiment
    """)
    return


@app.cell
def _(mask, slice, visualize_map):
    map_widget = visualize_map(
        slice,
        title="Mask region",
        interactive=False,
        mask_2d=mask,
        show_mask=True,
        vmin=slice.min(),
        vmax=slice.max(),
        center=slice.mean(),
        figsize=(39,5.24),
        dpi=500
    )
    return (map_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Guidance @ weather time
    """)
    return


@app.cell
def _(map_widget):
    # mo.vstack([mo.md("Mask:"), map_widget])
    map_widget
    return


@app.cell
def _(
    GUIDANCE_MODES,
    alpha_slider,
    guidance_mode_dropdown,
    level_slider,
    max_perc_granularity_slider,
    min_max_lambda_slider,
    mo,
    partition_dropdown,
    peak_at_slider,
    peak_granularity_slider,
    peak_slider,
    reference_rollout,
    shape_slider,
    var_dropdown,
    y_trajectory_plot,
):
    if guidance_mode_dropdown.value == GUIDANCE_MODES[0]:
        weather_time_vstack = mo.vstack(
            [
                # mo.md("Note: selecting other variables will not affect the mask."),
                mo.hstack(
                    [
                        partition_dropdown,
                        var_dropdown,
                        level_slider,
                    ],
                    justify="start",
                ),
                mo.hstack([guidance_mode_dropdown, reference_rollout], justify="start"),
                mo.hstack(
                    [alpha_slider, max_perc_granularity_slider, min_max_lambda_slider],
                    justify="start",
                ),
                mo.hstack(
                    [
                        peak_slider,
                        peak_granularity_slider,
                        shape_slider,
                        peak_at_slider,
                    ],
                    justify="start",
                ),
                y_trajectory_plot,
            ]
        )
    else:
        weather_time_vstack = mo.vstack(
            [
                mo.hstack(
                    [
                        partition_dropdown,
                        var_dropdown,
                        level_slider,
                    ],
                    justify="start",
                ),
                guidance_mode_dropdown,
                y_trajectory_plot,
            ]
        )
    weather_time_vstack
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Guidance @ diffusion time
    """)
    return


@app.cell
def _(lambda_, plot_trajectory):
    # lambda_trajectory_plot = plot_trajectory(
    #     lambda_, "$\lambda_t$", ymax=3, ymin=0, 
    #     title="Guidance strength-schedule $\{\lambda_t\}_{t=0}^{T-1}$",
    #     figsize=(10.85, 2.5)
    # )
    lambda_trajectory_plot = plot_trajectory(
        lambda_,
        r"$\lambda_t$",
        ymin=0,
        ymax=3,
        title=r"Guidance strength schedule $\{\lambda_t\}_{t=0}^{T-1}$",
        figsize=(17.5, 4.0),
    )
    return (lambda_trajectory_plot,)


@app.cell
def _(lambda_shape_slider, lambda_trajectory_plot, mo, w_slider):
    mo.vstack([
        mo.md("The guidance strength schedule \(\{\lambda_t\}_{t=0}^{T-1}\) determines the strength with which the guidance vector modifies the vector field \(u_t^{\\theta}\) at diffusion timestep \(t\)."),
        mo.vstack([
            mo.hstack([
                lambda_shape_slider,
                mo.md("controls the schedule's shape")
            ], justify="start"),
            mo.hstack([
                w_slider,
                mo.md("controls the max value of lambda_t (always @ step 12)")
            ], justify="start"),
            lambda_trajectory_plot,
        ])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save config
    Run from terminal.
    """)
    return


@app.cell
def _(mo):
    test_flag_checkbox = mo.ui.checkbox(value=False, label="test")
    return


@app.cell
def _(
    M,
    N,
    alpha,
    config,
    ground_truth,
    guidance_mode_dropdown,
    lambda_,
    lambda_shape_slider,
    level,
    level_idx,
    list_tensors_to_floats,
    lower_boundary,
    mask_corners,
    mean_unguided_rollout,
    partition,
    planned_guidance,
    reference_rollout,
    timestamp,
    timestamp_idx,
    timestamps,
    unguided_rollout,
    upper_boundary,
    var,
    var_idx,
    w_slider,
):
    new_config = {
        "guidance_flag": True,
        "guidance_mode": guidance_mode_dropdown.value, 
        "rollout_id": config["rollout_id"],
        "N": N,
        "M": M,
        "mask_corners": mask_corners,
        "timestamp": str(timestamp),
        "timestamp_idx": int(timestamp_idx),
        "partition": partition,
        "level": None if level is None else str(level),
        "level_idx": None if level_idx is None else int(level_idx),
        "var": var,
        "var_idx": int(var_idx),
        "timestamps": [str(ts) for ts in timestamps],
        "init_mask_term": float(mean_unguided_rollout[0]),
        "ground_truth": list_tensors_to_floats(ground_truth),
        "unguided_rollout": [list_tensors_to_floats(list_) for list_ in unguided_rollout],
        "mean_rollout": list_tensors_to_floats(mean_unguided_rollout),
        "y": list_tensors_to_floats(planned_guidance),
        "lower_boundary": list_tensors_to_floats(lower_boundary), 
        "upper_boundary": list_tensors_to_floats(upper_boundary),
        "y_perc": alpha,
        "lambda_": list_tensors_to_floats(lambda_),
        "alpha": lambda_shape_slider.value,
        "w": w_slider.value,
        "reference": reference_rollout.value
    }
    return (new_config,)


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(CONFIGS, config, config_button, new_config, save_to_json):
    if config_button.value:
        config_dir = CONFIGS / "guided"
        save_to_json(new_config, config_dir, f"{config['rollout_id']}")
    return


if __name__ == "__main__":
    app.run()
