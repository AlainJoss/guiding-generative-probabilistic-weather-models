import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", css_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _():
    import random
    from pathlib import Path 
    from datetime import datetime, timedelta

    import marimo as mo
    import numpy as np
    import geopandas as gpd
    import geodatasets
    import torch

    from wigglystuff import ChartPuck
    from scipy.interpolate import CubicSpline

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import colors

    import cartopy.feature as cfeature
    from cartopy.crs import PlateCarree

    method_state = {"value": "CubicSpline"}
    return Path, mo, np, plt, torch


@app.cell
def _():
    from src.interaction import (
        visualize_map, get_mask_corners_from_widget, 
        get_mask_from_corners, plot_trajectory, plot_dual_trajectory
    )
    from src.funcs import avg_over_mask, get_guidance_trajectory, N_schedule, T_schedule, compute_mean_rollout
    from src.rollout import rollout
    from src.utils import (
        ensure_rollout_dir,
        get_dataset, get_model, state_to_device,
        read_state, get_slice, save_to_json, read_json,
        read_states, xr_to_torch, list_tens_to_floats, get_now_timestamp
    )
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.visualization import visualize_mask_terms_over_N

    return (
        LEVELS_DICT,
        N_schedule,
        PARTITIONS,
        T_schedule,
        VARIABLES_DICT,
        avg_over_mask,
        compute_mean_rollout,
        ensure_rollout_dir,
        get_dataset,
        get_guidance_trajectory,
        get_mask_corners_from_widget,
        get_mask_from_corners,
        get_model,
        get_now_timestamp,
        get_slice,
        list_tens_to_floats,
        plot_dual_trajectory,
        plot_trajectory,
        read_json,
        read_states,
        rollout,
        save_to_json,
        state_to_device,
        visualize_map,
        visualize_mask_terms_over_N,
        xr_to_torch,
    )


@app.cell
def _():
    from src.paths import ROLLOUTS

    return (ROLLOUTS,)


@app.cell
def _(device, get_dataset, get_model):
    ds = get_dataset()
    model = get_model(device)
    return ds, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactivity
    """)
    return


@app.cell
def _():
    device = "mps"

    MAX_PERC_DELTA = 100/100

    VMIN = None
    VMAX = None
    return (device,)


@app.cell
def _(PARTITIONS, mo, unguided_cfg):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=unguided_cfg["partition"], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(LEVELS_DICT, mo, partition, unguided_cfg):
    LEVELS = LEVELS_DICT[partition]
    level_slider = mo.ui.slider(steps=LEVELS, value=unguided_cfg["level"], label="level: ", show_value=True, debounce=True)
    return LEVELS, level_slider


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition, unguided_cfg):
    VARIABLES = VARIABLES_DICT[partition]
    VARIABLE_DEFAULT = unguided_cfg["var"] if partition == unguided_cfg["partition"] else VARIABLES[0]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLE_DEFAULT, label="variable : ")
    return VARIABLES, var_dropdown


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(slice, visualize_map):
    map_widget = visualize_map(
        slice,
        title="Select mask region",
        interactive=True,
        vmin=slice.min(),
        vmax=slice.max(),
        center= slice.mean()
    )
    return (map_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### flow-time ui elements
    """)
    return


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
def _(lambda_, plot_trajectory):
    lambda_trajectory_plot = plot_trajectory(lambda_, "$\lambda_t$", ymax=3, ymin=0, title="Guidance strength-schedule $\{\lambda_t\}_{t=0}^{T-1}$")
    return (lambda_trajectory_plot,)


@app.cell
def _(mo):
    min_max_lambda_slider = mo.ui.slider(
        steps=[5, 10, 25, 50, 100], value=10, label="bounds (%): ", show_value=True
    )
    return (min_max_lambda_slider,)


@app.cell
def _(min_max_lambda_slider):
    min_max_lambda = min_max_lambda_slider.value
    return (min_max_lambda,)


@app.cell
def _(min_max_lambda, mo):
    alpha_slider = mo.ui.slider(
        -min_max_lambda,
        min_max_lambda,
        value=0,
        step=1,
        label="max percentage change (%): ",
        show_value=True,
    )
    return (alpha_slider,)


@app.cell
def _(alpha_slider):
    alpha = alpha_slider.value / 100
    return (alpha,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### weather-time ui elements
    """)
    return


@app.cell
def _(mo):
    GUIDANCE_MODES = ["manual_trajectory", "ground_truth", "lower_boundary", "upper_boundary"]
    guidance_mode_dropdown = mo.ui.dropdown(GUIDANCE_MODES, value=GUIDANCE_MODES[0], label="guidance mode: ")
    return GUIDANCE_MODES, guidance_mode_dropdown


@app.cell
def _():
    from src.funcs import get_inverse_guidance_trajectory

    return (get_inverse_guidance_trajectory,)


@app.cell
def _(
    GUIDANCE_MODES,
    N,
    N_schedule,
    alpha,
    get_guidance_trajectory,
    get_inverse_guidance_trajectory,
    ground_truth,
    guidance_mode_dropdown,
    mean_unguided_rollout,
    unguided_rollout,
):
    if guidance_mode_dropdown.value == GUIDANCE_MODES[0]:
        y_trajectory = N_schedule(N, 1.0, alpha)
        planned_guidance = get_guidance_trajectory(
            y_trajectory, mean_unguided_rollout
        )
    elif guidance_mode_dropdown.value == GUIDANCE_MODES[1]:
        planned_guidance = ground_truth
        y_trajectory = get_inverse_guidance_trajectory(
            planned_guidance, mean_unguided_rollout
        )
    elif guidance_mode_dropdown.value == GUIDANCE_MODES[2]:
        planned_guidance, _ = get_ensemble_upper_bound(unguided_rollout)
        y_trajectory = get_inverse_guidance_trajectory(
            planned_guidance, mean_unguided_rollout
        )
    else:
        _, planned_guidance = get_ensemble_upper_bound(unguided_rollout)
        y_trajectory = get_inverse_guidance_trajectory(
            planned_guidance, mean_unguided_rollout
        )
    return planned_guidance, y_trajectory


@app.function
def get_ensemble_upper_bound(unguided_rollout):
    rows = [[float(v) for v in row] for row in unguided_rollout]
    M = min(len(row) for row in rows)
    trimmed = [row[:M] for row in rows]
    lower = [min(row) for row in trimmed]
    upper = [max(row) for row in trimmed]
    return lower, upper


@app.cell
def _(ds):
    STRIDE = int(ds.lead_time_hours) // int(ds.timedelta)
    TIMESTAMPS = [str(ts[2]).split(".")[0] for ts in ds.timestamps][STRIDE:-STRIDE]
    return STRIDE, TIMESTAMPS


@app.cell
def _(unguided_cfg):
    timestamp = unguided_cfg["timestamp"]
    M = unguided_cfg["M"]
    N = unguided_cfg["N"]
    return M, N, timestamp


@app.cell
def _(N, STRIDE, TIMESTAMPS, timestamp_idx):
    timestamps = [TIMESTAMPS[timestamp_idx + STRIDE * k] for k in range(N + 1)]
    return (timestamps,)


@app.cell
def _(LEVELS, TIMESTAMPS, VARIABLES, ds, level, timestamp, var):
    timestamp_idx = TIMESTAMPS.index(timestamp)
    var_idx = VARIABLES.index(var)
    level_idx = LEVELS.index(level)
    x_start = ds[timestamp_idx]
    return level_idx, timestamp_idx, var_idx, x_start


@app.cell
def _(get_mask_corners_from_widget, get_mask_from_corners, map_widget):
    mask_corners = get_mask_corners_from_widget(map_widget)
    mask = get_mask_from_corners(*mask_corners)
    return mask, mask_corners


@app.cell
def _(ds, level_idx, partition, var_idx, x_start):
    slice = ds.denormalize(x_start["state"])[partition][var_idx, level_idx]
    return (slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The rollout experiment stuff
    """)
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="refresh")
    return (refresh_button,)


@app.cell
def _(mo):
    SUBFOLDERS = ["old_model", "new_model", "unguided"]
    subfolder_selector = mo.ui.dropdown(label="Subfolder", value=SUBFOLDERS[2], options=SUBFOLDERS)
    return (subfolder_selector,)


@app.cell
def _(subfolder_selector):
    subfolder = subfolder_selector.value
    return (subfolder,)


@app.cell
def _(Path, ROLLOUTS, mo, refresh_button, subfolder):
    if refresh_button.value:
        pass


    def has_config_json(path: Path) -> bool:
        return (path / "config.json").exists()


    unguided_rollouts = Path(ROLLOUTS, subfolder).glob("2026*")
    unguided_rollouts = sorted(
        [p.name for p in unguided_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_unguided_rollout_dropdown = mo.ui.dropdown(
        label="Experiment: ", value=unguided_rollouts[0], options=unguided_rollouts
    )
    return (pick_unguided_rollout_dropdown,)


@app.cell
def _(Path, ROLLOUTS, pick_unguided_rollout_dropdown, read_json, subfolder):
    unguided_rollout_dir = Path(
        ROLLOUTS, subfolder, pick_unguided_rollout_dropdown.value
    )
    unguided_cfg = read_json(unguided_rollout_dir, "config")
    return unguided_cfg, unguided_rollout_dir


@app.cell
def _(mo, pick_unguided_rollout_dropdown, subfolder_selector, unguided_cfg):
    experiment_dropdown = mo.vstack([
        mo.md("Pick unguided rollout experiment."),
        mo.hstack([
            subfolder_selector,
            pick_unguided_rollout_dropdown
        ], justify="start"),
        mo.accordion(
            {
                "Experiment params": mo.md("<br>".join(f"{k}: {v}" for k, v in unguided_cfg.items()))
            }
        )
    ])
    return (experiment_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mask terms over rollouts
    """)
    return


@app.cell
def _(
    M,
    STRIDE,
    TIMESTAMPS,
    avg_over_mask,
    compute_mean_rollout,
    ds,
    get_slice,
    level,
    level_idx,
    mask,
    partition,
    read_states,
    slice,
    timestamp_idx,
    unguided_cfg,
    unguided_rollout_dir,
    var,
    var_idx,
    xr_to_torch,
):
    ground_truth = []
    unguided_rollout = []
    det_rollout = []

    init_value = avg_over_mask(slice, mask)
    ground_truth.append(init_value)
    unguided_rollout.append([init_value] * M)
    det_rollout.append([init_value] * M)

    # TODO: adapt to new structure
    for n in range(1, unguided_cfg["N"] + 1):
        timestamp_n = TIMESTAMPS[timestamp_idx + STRIDE * n]

        # ground truth
        gt_state_n = ds[timestamp_idx + STRIDE * n]["state"]
        gt_slice_n = ds.denormalize(gt_state_n)[partition][var_idx, level_idx]
        ground_truth.append(avg_over_mask(gt_slice_n, mask))

        # unguided
        unguided_states = read_states(unguided_rollout_dir, "unguided", n)

        unguided_slices = [
            get_slice(state, partition, level, var, timestamp_n)
            for state in unguided_states
        ]
        unguided_slices = [xr_to_torch(s) for s in unguided_slices]
        unguided_avgs = [avg_over_mask(s, mask) for s in unguided_slices]
        unguided_rollout.append(unguided_avgs)

        # det
        det_states = read_states(unguided_rollout_dir, "det", n)
        det_slices = [
            get_slice(state, partition, level, var, timestamp_n)
            for state in det_states
        ]
        det_slices = [xr_to_torch(s) for s in det_slices]
        det_avgs = [avg_over_mask(s, mask) for s in det_slices]
        det_rollout.append(det_avgs)

    # means
    mean_unguided_rollout = compute_mean_rollout(unguided_rollout)
    mean_det_rollout = compute_mean_rollout(det_rollout)
    return det_rollout, ground_truth, mean_unguided_rollout, unguided_rollout


@app.cell
def _(det_rollout, ground_truth, np, plt, unguided_rollout, var):
    def plot_ensemble_rmse(gen_rmse, det_rmse, var):
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=110)
        steps = list(range(len(gen_rmse)))
        ax.plot(
            steps,
            gen_rmse,
            marker="o",
            label="generative (M-ensemble)",
            color="C0",
        )
        ax.plot(steps, det_rmse, marker="s", label="deterministic (M)", color="C3")
        ax.set_xlabel("rollout step n")
        ax.set_ylabel(f"RMSE  [{var}]")
        ax.set_title("Ensemble RMSE vs ground truth (avg-over-mask scalar)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig


    def ensemble_rmse_per_step(ensemble, ground_truth):
        return [
            float(np.sqrt(np.mean([(float(m) - float(gt)) ** 2 for m in members])))
            for members, gt in zip(ensemble, ground_truth)
        ]


    gen_rmse_per_step = ensemble_rmse_per_step(unguided_rollout, ground_truth)
    det_rmse_per_step = ensemble_rmse_per_step(det_rollout, ground_truth)
    rmse_plot = plot_ensemble_rmse(gen_rmse_per_step, det_rmse_per_step, var)
    rmse_plot
    return


@app.cell
def _(det_rollout, ground_truth, np, plt, unguided_rollout, var):
    def member_rmse_over_steps(ensemble, ground_truth):
        M = len(ensemble[0])
        out = []
        for m in range(M):
            sq = [
                (float(step_vals[m]) - float(gt)) ** 2
                for step_vals, gt in zip(ensemble, ground_truth)
            ]
            out.append(float(np.sqrt(np.mean(sq))))
        return out


    def plot_paired_member_rmse(gen_rmse_m, det_rmse_m, var):
        M = len(gen_rmse_m)
        x = np.arange(M)
        width = 0.4
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * M + 2), 3.5), dpi=110)
        ax.bar(x - width / 2, gen_rmse_m, width, label="generative", color="C0")
        ax.bar(x + width / 2, det_rmse_m, width, label="deterministic", color="C3")
        ax.set_xticks(x)
        ax.set_xticklabels([f"m{m+1}" for m in range(M)])
        ax.set_xlabel("ensemble member")
        ax.set_ylabel(f"temporal RMSE  [{var}]")
        ax.set_title("Per-member RMSE over N steps (vs ground truth)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig


    gen_rmse_per_member = member_rmse_over_steps(unguided_rollout, ground_truth)
    det_rmse_per_member = member_rmse_over_steps(det_rollout, ground_truth)
    paired_rmse_plot = plot_paired_member_rmse(
        gen_rmse_per_member, det_rmse_per_member, var
    )
    paired_rmse_plot
    return


@app.cell
def _(det_rollout, ground_truth, np, plt, unguided_rollout, var):
    def relative_error_matrix(unguided, det, ground_truth):
        """[M, N+1] matrix of |gen - gt| - |det - gt|. Negative = generative closer."""
        M = len(unguided[0])
        N1 = len(unguided)
        mat = np.zeros((M, N1))
        for n in range(N1):
            gt_n = float(ground_truth[n])
            for m in range(M):
                gen_err = abs(float(unguided[n][m]) - gt_n)
                det_err = abs(float(det[n][m]) - gt_n)
                mat[m, n] = gen_err - det_err
        return mat


    def plot_relative_perf_matrix(mat, var):
        M, N1 = mat.shape
        vmax = float(np.max(np.abs(mat))) or 1.0
        fig, ax = plt.subplots(
            figsize=(max(6, 0.45 * N1 + 2), max(3, 0.35 * M + 1.5)), dpi=110
        )
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(N1))
        ax.set_xticklabels([str(n) for n in range(N1)])
        ax.set_yticks(np.arange(M))
        ax.set_yticklabels([f"m{m+1}" for m in range(M)])
        ax.set_xlabel("rollout step n")
        ax.set_ylabel("ensemble member m")
        ax.set_title(
            f"|gen - gt| - |det - gt|   [{var}]   (blue = gen better, red = det better)"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(f"abs-error diff [{var}]")
        fig.tight_layout()
        return fig


    rel_perf_matrix = relative_error_matrix(
        unguided_rollout, det_rollout, ground_truth
    )
    rel_perf_plot = plot_relative_perf_matrix(rel_perf_matrix, var)
    rel_perf_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


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
    )
    return (y_trajectory_plot,)


@app.cell
def _(
    ground_truth,
    mean_unguided_rollout,
    timestamps,
    unguided_rollout,
    var,
    visualize_mask_terms_over_N,
):
    ensemble_rollout_plot = visualize_mask_terms_over_N(
        var,
        timestamps,
        mean_rollout=mean_unguided_rollout,
        ensemble_rollout=unguided_rollout,
        ground_truth=ground_truth,
        # gen_det_rollout=det_rollout,
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mask
    """)
    return


@app.cell
def _(level_slider, map_widget, mo, partition_dropdown, var_dropdown):
    mo.vstack([
        mo.md("The mask specifies the variable, level, and region of interest. By default, these values are taken from the unguided rollout experiment's `config.json` file."),
        mo.hstack([
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        map_widget
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Guidance @ weather time
    """)
    return


@app.cell
def _(
    GUIDANCE_MODES,
    alpha_slider,
    guidance_mode_dropdown,
    min_max_lambda_slider,
    mo,
    y_trajectory_plot,
):
    if guidance_mode_dropdown.value == GUIDANCE_MODES[0]:
        weather_time_vstack = mo.vstack(
            [
                mo.md("The guidance trajectory is the sequence of target values for the masked spatial average that we aim to steer the generative model towards over the $N$ weather steps."),
                guidance_mode_dropdown,
                mo.hstack([alpha_slider, min_max_lambda_slider], justify="start"),
                y_trajectory_plot,
            ]
        )
    else:
        weather_time_vstack = mo.vstack(
            [guidance_mode_dropdown, y_trajectory_plot]
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
    ## Run experiment
    """)
    return


@app.cell
def _(mo):
    get_status, set_status = mo.state("IDLE")
    return get_status, set_status


@app.cell
def _(run_button, set_status):
    set_status("IDLE")
    if run_button.value:
        set_status("RUNNING")
    return


@app.cell
def _(get_status):
    status = get_status()
    return (status,)


@app.cell
def _(mo):
    test_flag_checkbox = mo.ui.checkbox(value=False, label="test")
    return (test_flag_checkbox,)


@app.cell
def _(mo, test_flag_checkbox):
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([
        test_flag_checkbox,
        run_button,
        # mo.md(f"Experiment status: **{get_status()}**")
    ])
    return (run_button,)


@app.cell
def _(
    M,
    N,
    alpha,
    ground_truth,
    guidance_mode_dropdown,
    lambda_,
    level,
    level_idx,
    list_tens_to_floats,
    mask_corners,
    mean_unguided_rollout,
    partition,
    timestamp,
    timestamp_idx,
    timestamps,
    unguided_cfg,
    unguided_rollout,
    var,
    var_idx,
    w_slider,
    y_trajectory,
):
    config = {
        "guidance_mode": guidance_mode_dropdown.value, 
        "unguided_rollout_id": unguided_cfg["rollout_id"],
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
        "timestamps": timestamps,
        "ground_truth": list_tens_to_floats(ground_truth),
        "ensemble_rollout": [list_tens_to_floats(list_) for list_ in unguided_rollout],
        "mean_rollout": list_tens_to_floats(mean_unguided_rollout),
        "y_trajectory": list_tens_to_floats(y_trajectory),
        "lambda_": list_tens_to_floats(lambda_),
        "alpha": alpha,
        "w": w_slider.value,
        "init_mask_term": float(mean_unguided_rollout[0]),
    }
    return (config,)


@app.cell
def _(
    N,
    Path,
    ROLLOUTS,
    config,
    device,
    ds,
    ensure_rollout_dir,
    get_now_timestamp,
    lambda_,
    level_idx,
    mask_corners,
    mean_unguided_rollout,
    model,
    partition,
    planned_guidance,
    rollout,
    run_button,
    save_to_json,
    state_to_device,
    status,
    test_flag_checkbox,
    torch,
    var_idx,
    x_start,
):
    if run_button.value and status == "RUNNING":
        rollout_id = get_now_timestamp()
        rollout_dir = ensure_rollout_dir("guided", N, rollout_id)
        config["rollout_id"] = rollout_id
        rollout(
            guidance_flag=True,
            rollout_dir=rollout_dir,
            ds=ds,
            x_start=state_to_device(x_start, device),
            gen_model=model,
            mask_corners=mask_corners,
            init_mask_term=torch.as_tensor(mean_unguided_rollout[0]),
            y=torch.as_tensor(planned_guidance),
            lambda_=lambda_,
            N=N,
            partition=partition,
            level_idx=level_idx,
            var_idx=var_idx,
            m=1,
            test=test_flag_checkbox.value
        )
        save_to_json(config, Path(ROLLOUTS, "guided", rollout_dir), "config")
    return


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(CONFIGS, config, config_button, get_now_timestamp, save_to_json):
    if config_button.value:
        config_id = get_now_timestamp()
        config_dir = CONFIGS / "guided"
        # no need to add the config_id to the config
        save_to_json(config, config_dir, f"{config_id}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
