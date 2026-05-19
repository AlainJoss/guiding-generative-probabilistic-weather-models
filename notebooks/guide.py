import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", css_file="")


@app.cell
def _():
    import marimo as mo
    import torch

    return mo, torch


@app.cell
def _():
    from src.paths import ROLLOUTS, RUN_CONFIGS
    from src.rollout_config import GUIDANCE_REFERENCES, MASK_MODES
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.analysis.analysis_plots import plot_guidance_tracking
    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory

    from src.utils.dataset_utils import get_x_cond
    from src.utils.converters import list_tensors_to_floats, get_var_idx, get_level_idx
    from src.utils.setup import get_now_timestamp
    from src.utils.dataset_utils import get_timestamps, get_N_timestamps, get_N_slices, get_slices
    from src.utils.read_write import (
        get_xr_dataset,
        save_to_json, read_json, get_rollout_ids, get_rollout_xr,
        get_rollout_config
    )

    from src.funcs import N_schedule, T_schedule

    from src.mask import get_masked_slices, get_masked_mean, get_mask_2d, get_normal_mask, get_mask_from_corners
    from src.target import get_target_trajectory, get_reference_trajectory

    return (
        GUIDANCE_REFERENCES,
        LEVELS_DICT,
        MASK_MODES,
        N_schedule,
        PARTITIONS,
        T_schedule,
        VARIABLES_DICT,
        get_N_timestamps,
        get_level_idx,
        get_mask_2d,
        get_masked_mean,
        get_reference_trajectory,
        get_rollout_config,
        get_rollout_ids,
        get_rollout_xr,
        get_slices,
        get_target_trajectory,
        get_var_idx,
        get_xr_dataset,
        plot_dual_trajectory,
        plot_trajectory,
        save_to_json,
        visualize_map,
    )


@app.cell
def _(get_xr_dataset):
    ds = get_xr_dataset()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Move ui elements to getters in python to reuse them throughout the 3 notebooks.
    - Rename variables, some are really shit.
    - The lower bound and upper bound thing shouldn't have a function in here.
    - The part where I compute the mask average should be a really simple function with N, M, and any list I can possibly pass and return the value in a dict.
    - Write a getter for the timestamps from the xarray object, also format them correctly.
    - The experiment iterator file run all configs should be renamed, but is really good in principle.
    - Here I should define the delta_trajectory using the notion of tail or quantile.
    """)
    return


@app.cell
def _(config, get_N_timestamps):
    timestamps = get_N_timestamps(config.timestamp, config.N+1)
    return (timestamps,)


@app.cell
def _(PARTITIONS, config, mo):
    partition = mo.ui.dropdown(PARTITIONS, value=config.partition, label="partition: ")
    return (partition,)


@app.cell
def _(LEVELS_DICT, config, mo, partition):
    LEVELS = LEVELS_DICT[partition.value]
    level = mo.ui.slider(
        steps=LEVELS,
        value=config.level,
        label="level: ",
        show_value=True,
        debounce=True,
    )
    return (level,)


@app.cell
def _(VARIABLES_DICT, config, mo, partition):
    VARIABLES = VARIABLES_DICT[partition.value]
    VARIABLE_DEFAULT = config.var if partition.value == config.partition else VARIABLES[0]
    var = mo.ui.dropdown(VARIABLES, value=VARIABLE_DEFAULT, label="var: ")
    return (var,)


@app.cell
def _(get_level_idx, get_var_idx, level, partition, var):
    var_idx = get_var_idx(partition.value, var.value)
    level_idx = get_level_idx(partition.value, level.value)
    return level_idx, var_idx


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
def _(config, mo):
    peak_at_slider = mo.ui.slider(
        start=1,
        stop=config.N,
        step=1,
        value=config.N // 2,
        label="peak @ n: ",
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
    delta_alpha = alpha_slider.value / 100
    return (delta_alpha,)


@app.cell
def _(GUIDANCE_REFERENCES, mo):
    guidance_reference = mo.ui.dropdown(GUIDANCE_REFERENCES, value=GUIDANCE_REFERENCES[0], label="guidance reference: ")
    return (guidance_reference,)


@app.cell
def _(MASK_MODES, mo):
    mask_mode = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0])
    return (mask_mode,)


@app.cell
def _(N_masked_means_gt, lower_boundary, unguided_member, upper_boundary):
    reference_rollouts = {
        "unguided_members": unguided_member,
        "ground_truth": N_masked_means_gt,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary
    }
    return (reference_rollouts,)


@app.cell
def _(M_N_masked_means_ung, m):
    unguided_member = M_N_masked_means_ung[m.value]
    return (unguided_member,)


@app.cell
def _(M_N_masked_means_ung):
    lower_boundary, upper_boundary = M_N_masked_means_ung.min(axis=0), M_N_masked_means_ung.max(axis=0)
    return lower_boundary, upper_boundary


@app.cell
def _(N_schedule, config, delta_alpha, peak, peak_at_slider, shape_slider):
    delta_trajectory = N_schedule(
        config.N, shape_slider.value, peak, delta_alpha, peak_at_n=peak_at_slider.value
    )
    return (delta_trajectory,)


@app.cell
def _(guidance_reference, reference_rollouts):
    reference_trajectory = reference_rollouts[guidance_reference.value]
    return (reference_trajectory,)


@app.cell
def _(config, get_reference_trajectory, guidance_reference, m):
    reference_rollout = get_reference_trajectory(guidance_reference.value, config.rollout_id, m=m.value)
    return (reference_rollout,)


@app.cell
def _(
    delta_trajectory,
    get_slices,
    get_target_trajectory,
    level,
    level_idx,
    partition,
    reference_rollout,
    var,
    var_idx,
):
    planned_guidance = get_target_trajectory(
        partition.value, var_idx, level_idx,
        delta_trajectory, reference_rollout
    )
    N_slices_planned_guidance = get_slices(planned_guidance, partition.value, var.value, level.value)
    return (N_slices_planned_guidance,)


@app.cell
def _(N_slices_planned_guidance, get_masked_mean, mask):
    planned_trajectory = get_masked_mean(N_slices_planned_guidance, mask)
    return (planned_trajectory,)


@app.cell
def _(torch):
    def get_ensemble_upper_bound(unguided_rollout):
        rows = [[float(v) for v in row] for row in unguided_rollout]
        M = min(len(row) for row in rows)
        trimmed = [row[:M] for row in rows]
        lower_boundary = [torch.tensor(min(row)) for row in trimmed]
        upper_boundary = [torch.tensor(max(row)) for row in trimmed]
        return lower_boundary, upper_boundary

    return


@app.cell
def _(config, mo):
    n_slider = mo.ui.slider(
        start=0, 
        stop=config.N,
        step=1,
        label="n: ",
        value=0,
        debounce=True,
        show_value=True
    )
    return (n_slider,)


@app.cell
def _(N_slices_gt, n_slider):
    slice = N_slices_gt[n_slider.value]
    return (slice,)


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="refresh")
    return (refresh_button,)


@app.cell
def _(get_rollout_ids, mo, refresh_button):
    if refresh_button.value:
        pass

    unguided_rollouts = get_rollout_ids("unguided")
    pick_unguided_rollout_dropdown = mo.ui.dropdown(
        label="Rollout: ", value=unguided_rollouts[0], options=unguided_rollouts
    )
    return (pick_unguided_rollout_dropdown,)


@app.cell
def _(get_rollout_config, pick_unguided_rollout_dropdown):
    config = get_rollout_config("unguided", pick_unguided_rollout_dropdown.value)
    return (config,)


@app.cell
def _(config, get_rollout_xr):
    ground_truth_xr = get_rollout_xr(config.rollout_id, "ground_truth")
    unguided_xr = get_rollout_xr(config.rollout_id,"unguided")
    return ground_truth_xr, unguided_xr


@app.cell
def _(get_slices, ground_truth_xr, level, partition, unguided_xr, var):
    N_slices_gt = get_slices(ground_truth_xr, partition.value, var.value, level.value)
    N_slices_ung = get_slices(unguided_xr, partition.value, var.value, level.value)
    return N_slices_gt, N_slices_ung


@app.cell
def _(N_slices_gt, N_slices_ung, get_masked_mean, mask):
    N_masked_means_gt = get_masked_mean(N_slices_gt, mask)
    M_N_masked_means_ung = get_masked_mean(N_slices_ung, mask)
    return M_N_masked_means_ung, N_masked_means_gt


@app.cell
def _(unguided_xr):
    members = [int(val) for val in list(unguided_xr.member.values)]
    return (members,)


@app.cell
def _(M_N_masked_means_ung):
    mean_unguided_rollout = M_N_masked_means_ung.mean(axis=0)
    return (mean_unguided_rollout,)


@app.cell
def _(timestamps, unguided_member):
    len(unguided_member), len(timestamps)
    return


@app.cell
def _(
    M_N_masked_means_ung,
    N_masked_means_gt,
    delta_trajectory,
    m,
    mean_unguided_rollout,
    planned_trajectory,
    plot_dual_trajectory,
    reference_trajectory,
    timestamps,
    unguided_member,
    var,
):
    guidance_plot = plot_dual_trajectory(
        timestamps=timestamps,
        unguided_member=unguided_member,
        reference_trajectory=reference_trajectory,
        m=m.value,
        mean_rollout=mean_unguided_rollout,
        ground_truth=N_masked_means_gt,
        planned_guidance=planned_trajectory,
        y_trajectory=delta_trajectory,
        var=var.value,
        ensemble_rollout=M_N_masked_means_ung,
        ymin_left=None,
        ymax_left=None,
        figsize=(17.5, 5.5),
        title=f"{var.value} trajectory",
        subtitle="Unguided rollout ensemble, planned guidance, ground truth, and percentage trajectory",
        dpi=500
    )
    return (guidance_plot,)


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
def _(mo, pick_unguided_rollout_dropdown, refresh_button):
    mo.hstack(
        [pick_unguided_rollout_dropdown, refresh_button],
        justify="start",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure guidance experiment
    """)
    return


@app.cell
def _(config, get_mask_2d):
    mask = get_mask_2d(config.mask_mode, config.mask_params)
    return (mask,)


@app.cell
def _(mask, slice, visualize_map):
    map_widget = visualize_map(
        slice,
        title="Mask region",
        interactive=True,
        mask_2d=mask,
        show_mask=True,
        vmin=slice.min(),
        vmax=slice.max(),
        center=slice.mean(),
        figsize=(45,25.24),
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
def _(map_widget, mask_mode, mo):
    mo.vstack([mask_mode, map_widget])
    return


@app.cell
def _(members, mo):
    m = mo.ui.slider(steps=members, label="m: ", show_value=True, debounce=True)
    return (m,)


@app.cell
def _(
    alpha_slider,
    guidance_plot,
    guidance_reference,
    level,
    m,
    max_perc_granularity_slider,
    min_max_lambda_slider,
    mo,
    partition,
    peak_at_slider,
    peak_granularity_slider,
    peak_slider,
    shape_slider,
    var,
):
    mo.vstack(
        [
            guidance_reference,
            m,
            mo.hstack(
                [
                    partition,
                    var,
                    level,
                ],
                justify="start",
            ),
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
            guidance_plot,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Guidance @ diffusion time
    """)
    return


@app.cell
def _(lambda_, plot_trajectory):
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
        mo.md(r"""
            \( \lambda_t = w \left(\sin\left(\frac{\pi t}{T-1}\right)\right)^\alpha, \quad t = 0, \dots, T-1 \).
        """),
        mo.vstack([
            lambda_shape_slider,
            w_slider,
            lambda_trajectory_plot,
        ])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save config
    """)
    return


@app.cell
def _(
    config,
    delta_trajectory,
    guidance_reference,
    lambda_,
    lambda_shape_slider,
    level,
    mask_mode,
    mean_unguided_rollout,
    partition,
    var,
    w_slider,
):
    new_config = {
        "rollout_id": config.rollout_id,
        "guidance_flag": True,
        "N": config.N,
        "M": config.M,
        "timestamp": config.timestamp,
        "partition": partition.value,
        "level": level.value,
        "var": var.value,
        "mask_mode": mask_mode.value,
        "mask_params": config.mask_params,
        "init_mask_term": float(mean_unguided_rollout[0]),
        "guidance_reference": guidance_reference.value,
        "delta_trajectory": delta_trajectory,
        "alpha": lambda_shape_slider.value,
        "w": w_slider.value,
        "lambda_": lambda_
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
