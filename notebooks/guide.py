import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium", css_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boiler plate
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pipeline
    - write data analyzer for playing around with N and mask before generating the rollout for the M distribution and start the guidance experiment.
    - generate ensemble rollout as base trajectory
    - define mask (or evolving masks) and define guidance (y) trajectory (extremified version of ensemble rollout trajectory over mask)
    - set other params
    - define weighted average Gaussian Kernel or future difference around region in loss function. Refine latex-notes with new definition of mask.
    - Try out regularization term z^tK^z or just z^tIz=||z||^2
    - Guide using the ground truth and see whether the accuracy of other variables improves.
    - Guuide using dynamic mask instead of fixed one (future rollout but also regin difference)
    - plot the sum of relative absolute change in variable from gen sample to gen guided sample.
    - Define an ensemble of $G$ guided models.
    - plot masks over N

    ## todos
    - capture total relative change per variable in mask and overall.
    - correct the loss function in latex doc and explain why y not possible in denormalized space.
    - convert temperature to degrees celsius
    - define masks with physical priors
    - define masks dynamically in N
    - experiment with multiple variables (and masks correspondingly)
    - increase T and decrease lambda
    - define fitness of

    ## questions
    - is the spatially interplay of variables respected when guiding samples?
    - how do variables spatially interplay?

    ## ideas
    - as baseline compute some basic facts about ArchesWeatherGen. For instance, how well it does (compared to its deterministic brother)? How does performance degrade as N of rollout increases?
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
    return Path, mo, torch


@app.cell
def _():
    from src.visualization import visualize_mask_terms_over_N

    return (visualize_mask_terms_over_N,)


@app.cell
def _():
    from src.utils import read_states, xr_to_torch, list_tens_to_floats

    return list_tens_to_floats, read_states, xr_to_torch


@app.cell
def _():
    from src.funcs import N_schedule, T_schedule, compute_mean_rollout

    return N_schedule, T_schedule, compute_mean_rollout


@app.cell
def _():
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    return LEVELS_DICT, PARTITIONS, VARIABLES_DICT


@app.cell
def _():
    from src.interaction import (
        visualize_map, get_mask_corners_from_widget, 
        get_mask_from_corners, plot_trajectory, plot_dual_trajectory
    )
    from src.funcs import avg_over_mask, get_guidance_trajectory
    from src.rollout import rollout
    from src.utils import (
        ensure_rollout_dir,
        get_dataset, get_model, state_to_device,
        read_state, get_slice, save_to_json, read_json
    )

    return (
        avg_over_mask,
        ensure_rollout_dir,
        get_dataset,
        get_guidance_trajectory,
        get_mask_corners_from_widget,
        get_mask_from_corners,
        get_model,
        get_slice,
        plot_dual_trajectory,
        plot_trajectory,
        read_json,
        rollout,
        save_to_json,
        state_to_device,
        visualize_map,
    )


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
    level_slider = mo.ui.slider(steps=LEVELS, value=unguided_cfg["level"], label="level: ")
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


@app.cell
def _(mo):
    w_slider = mo.ui.slider(0.1, 5, value=1.0, label="w: ", step=0.1)
    return (w_slider,)


@app.cell
def _(w_slider):
    w = w_slider.value
    return (w,)


@app.cell
def _(mo):
    lambda_shape_slider = mo.ui.slider(1.0, 10.0, step=0.1, value=1.0, label="shape: ")
    y_shape_slider = mo.ui.slider(1.0, 10.0, step=0.1, value=1.0, label="shape: ")
    return lambda_shape_slider, y_shape_slider


@app.cell
def _(lambda_shape_slider):
    lambda_shape = lambda_shape_slider.value
    return (lambda_shape,)


@app.cell
def _(T_schedule, lambda_shape, w):
    lambda_ = T_schedule(25, lambda_shape, w) 
    return (lambda_,)


@app.cell
def _(lambda_, plot_trajectory):
    lambda_trajectory_plot = plot_trajectory(lambda_, "lambda", ymax=5, ymin=0)
    return (lambda_trajectory_plot,)


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(-1, 1, value=0.5, label="alpha: ", step=0.01)
    return (alpha_slider,)


@app.cell
def _(alpha_slider):
    alpha = alpha_slider.value
    return (alpha,)


@app.cell
def _(y_shape_slider):
    y_shape = y_shape_slider.value
    return (y_shape,)


@app.cell
def _(N, N_schedule, alpha, y_shape):
    y_trajectory = N_schedule(N, y_shape, alpha)
    return (y_trajectory,)


@app.cell
def _(ds):
    TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
    return (TIMESTAMPS,)


@app.cell
def _(unguided_cfg):
    timestamp = unguided_cfg["timestamp"]
    M = unguided_cfg["M"]
    N = unguided_cfg["N"]
    return M, N, timestamp


@app.cell
def _(N, TIMESTAMPS, timestamp_idx):
    timestamps = TIMESTAMPS[timestamp_idx:timestamp_idx+N+1] 
    return (timestamps,)


@app.cell
def _(LEVELS, TIMESTAMPS, VARIABLES, ds, level, timestamp, var):
    timestamp_idx = TIMESTAMPS.index(timestamp)
    var_idx = VARIABLES.index(var)
    level_idx = LEVELS.index(level) - 1
    x_start = ds[timestamp_idx]
    return level_idx, timestamp_idx, var_idx, x_start


@app.cell
def _(get_mask_corners_from_widget, get_mask_from_corners, map_widget):
    mask_corners = get_mask_corners_from_widget(map_widget)
    mask = get_mask_from_corners(*mask_corners)
    return mask, mask_corners


@app.cell
def _(ds, level_idx, partition, var_idx, x_start):
    # don't really like the batch dim ... [0]
    slice = ds.denormalize(x_start["state"])[partition][var_idx, level_idx]
    return (slice,)


@app.cell
def _(get_guidance_trajectory, mean_rollout, y_trajectory):
    planned_guidance = get_guidance_trajectory(y_trajectory, mean_rollout)
    return (planned_guidance,)


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
def _(Path, mo, refresh_button):
    if refresh_button.value:
        pass

    def has_config_json(path: Path) -> bool:
        return (path / "config.json").exists()


    unguided_rollouts = Path("rollouts", "unguided").glob("2026*")
    unguided_rollouts = sorted(
        [p for p in unguided_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_unguided_rollout_dropdown = mo.ui.dropdown(label="Pick unguided rollout", value=unguided_rollouts[0], options=unguided_rollouts)
    return (pick_unguided_rollout_dropdown,)


@app.cell
def _(pick_unguided_rollout_dropdown, read_json):
    unguided_rollout_dir = pick_unguided_rollout_dropdown.value
    unguided_cfg = read_json(unguided_rollout_dir, "config")
    return unguided_cfg, unguided_rollout_dir


@app.cell
def _(mo, pick_unguided_rollout_dropdown, refresh_button, unguided_cfg):
    experiment_dropdown = mo.vstack([
        mo.hstack([
            pick_unguided_rollout_dropdown, refresh_button
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
    ### Ensemble rollout
    """)
    return


@app.cell
def _(
    M,
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
    ensemble_rollout = []
    ensemble_rollout.append([avg_over_mask(slice, mask)]*M)
    ground_truth.append(avg_over_mask(slice, mask))

    for n in range(1, unguided_cfg["N"]+1):
        timestamp_n = TIMESTAMPS[timestamp_idx+n]
        state_n = ds[timestamp_idx+n]["state"]
        slice_n = ds.denormalize(state_n)[partition][var_idx, level_idx]
        ground_truth.append(avg_over_mask(slice_n, mask))
        states = read_states(unguided_rollout_dir, n) 
        slices = [get_slice(state, partition, level, var, timestamp_n) for state in states]
        slices = [xr_to_torch(slice_) for slice_ in slices]
        avgs = [avg_over_mask(slice_, mask) for slice_ in slices]

        ensemble_rollout.append(avgs)

    mean_rollout = compute_mean_rollout(ensemble_rollout)
    return ensemble_rollout, ground_truth, mean_rollout


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(planned_guidance, plot_dual_trajectory, timestamps, var, y_trajectory):
    y_trajectory_plot = plot_dual_trajectory(
        timestamps=timestamps,
        y_trajectory=y_trajectory,
        guidance_trajectory=planned_guidance,
        var=var,
        ymin_left=-1,
        ymax_left=1
    )
    return (y_trajectory_plot,)


@app.cell
def _(
    ensemble_rollout,
    ground_truth,
    mean_rollout,
    planned_guidance,
    timestamps,
    var,
    visualize_mask_terms_over_N,
):
    rollout_dist_plot = visualize_mask_terms_over_N(
        var, timestamps, ensemble_rollout, mean_rollout, ground_truth, planned_guidance
    )
    return (rollout_dist_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guidance experiments
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mask config
    """)
    return


@app.cell
def _(
    experiment_dropdown,
    level_slider,
    map_widget,
    mo,
    partition_dropdown,
    var_dropdown,
):
    mo.vstack([
        experiment_dropdown,
        mo.hstack([
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        mo.vstack([
            map_widget
        ])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Guidance config
    """)
    return


@app.cell
def _(
    alpha_slider,
    lambda_shape_slider,
    lambda_trajectory_plot,
    mo,
    rollout_dist_plot,
    w_slider,
    y_shape_slider,
    y_trajectory_plot,
):
    mo.hstack([
        mo.vstack([
            lambda_shape_slider,
            w_slider,
            lambda_trajectory_plot,
        ]),
        mo.vstack([
            y_shape_slider,
            alpha_slider,
            y_trajectory_plot,
        ]),
        rollout_dist_plot
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run guided rollout
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
def _(get_status, mo):
    status = get_status()
    mo.md(f"Experiment status: **{status}**")
    return (status,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run experiment")
    run_button
    return (run_button,)


@app.cell
def _(
    N,
    alpha,
    device,
    ds,
    ensemble_rollout,
    ensure_rollout_dir,
    ground_truth,
    lambda_,
    level,
    level_idx,
    list_tens_to_floats,
    mask_corners,
    mean_rollout,
    model,
    partition,
    planned_guidance,
    rollout,
    run_button,
    save_to_json,
    state_to_device,
    status,
    timestamp,
    timestamp_idx,
    timestamps,
    torch,
    unguided_cfg,
    var,
    var_idx,
    w,
    x_start,
    y_trajectory,
):
    if run_button.value and status == "RUNNING":
        # try:
        rollout_dir = ensure_rollout_dir("guided", N)

        rollout(
            guidance_flag=True,
            rollout_dir=rollout_dir,
            ds=ds,
            x_start=state_to_device(x_start, device),
            gen_model=model,
            mask_corners=mask_corners,
            init_mask_term=torch.as_tensor(mean_rollout[0]),
            y=torch.as_tensor(y_trajectory),  # needs to happen only here
            lambda_=lambda_,
            N=N,
            partition=partition,
            level_idx=level_idx,
            var_idx=var_idx,
        )

        config = {
            "unguided_rollout_dir": unguided_cfg["unguided_rollout_dir"],
            "N": N,
            "mask_corners": mask_corners,
            "timestamp": str(timestamp),
            "timestamp_idx": int(timestamp_idx),
            "partition": partition,
            "level": None if level is None else str(level),
            "level_idx": None if level_idx is None else int(level_idx),
            "var": var,
            "var_idx": int(var_idx),
            "timestamps": timestamps,
            "planned_guidance": list_tens_to_floats(planned_guidance),
            "ground_truth": list_tens_to_floats(ground_truth),
            "ensemble_rollout": [list_tens_to_floats(list_) for list_ in ensemble_rollout],
            "mean_rollout": list_tens_to_floats(mean_rollout),
            "y_perc": list_tens_to_floats(y_trajectory),
            "lambda_": list_tens_to_floats(lambda_),
            "alpha": alpha,
            "w": w
        }


        save_to_json(config, rollout_dir, "config")

        #     set_status("IDLE")
        # except Exception as e:
        #     print(f"Error: {type(e).__name__}: {e}")
        #     raise

        # finally:
        #     set_status("IDLE")
    return


if __name__ == "__main__":
    app.run()
