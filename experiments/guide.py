import marimo

__generated_with = "0.21.1"
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
    ## todos
    - normalize the colormaps around zero
    - refactor plots to use a dix min and max
    - solve how to steer in denormalized and latent space
    - refactor guided_flow model to take in partition, level_index, variable_index
    - refactor this whole notebook to outsource functionality
    - the start date is 18:00 the 31.12.2019!
    - convert temperature to degrees celsius
    - capture progress bar in marimo app-mode
    - ready for N-step rollout?
    - compare to ground-truth and also capture and compute no-guidance generation

    ## ideas
    - steer model towards grount truth in mask and measure divergence across states
    - generate multiple (vars, levels, partitions) no-guidance N rollouts and save in experiment folder -> need a way to encode the experiment (log of experiments or yaml with keys).
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
    return Path, mo, np, torch


@app.cell
def _():
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    return LEVELS_DICT, PARTITIONS, VARIABLES_DICT


@app.cell
def _():
    from src.utils import get_last_experiment_dir
    from src.interaction import visualize_map, get_mask_corners_from_widget, get_mask_from_corners, plot_trajectory
    from src.funcs import avg_over_mask, get_guidance_trajectory
    from src.utils import state_to_device

    return (
        avg_over_mask,
        get_guidance_trajectory,
        get_mask_corners_from_widget,
        get_mask_from_corners,
        plot_trajectory,
        state_to_device,
        visualize_map,
    )


@app.cell
def _(device):
    from src.utils import get_dataset, get_model
    ds = get_dataset()
    model = get_model(device)
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    - funcs
    - constants
    - sliders and dropdowns
    - ids
    """)
    return


@app.cell
def _(torch):
    import math

    def N_schedule(T: int, flatness: float, peak: float, alpha: float = 0.0):
        if T <= 0:
            return []

        return [
            torch.tensor(
                alpha + peak * (math.sin(0.5 * math.pi * (t + 1) / T) ** flatness),
                dtype=torch.float32,
            )
            for t in range(T)
        ]

    def T_schedule(T: int, flatness: float, peak: float): 
        if T == 1: 
            return [torch.tensor(0.0, dtype=torch.float32)] 
        return [
            torch.tensor( peak * (math.sin(math.pi * t / (T - 1)) ** flatness), dtype=torch.float32, ) for t in range(T)
               ] 

    return N_schedule, T_schedule


@app.cell
def _():
    device = "mps"

    MAX_PERC_DELTA = 100/100

    VMIN = None
    VMAX = None
    return (device,)


@app.cell
def _(ds, mo):
    # remove first and last, since we have two tensordicts less (because of prev and next)
    TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
    timestamp_dropdown = mo.ui.dropdown(TIMESTAMPS, value=TIMESTAMPS[0], label="start state : ")
    return TIMESTAMPS, timestamp_dropdown


@app.cell
def _(timestamp_dropdown):
    timestamp = timestamp_dropdown.value
    return (timestamp,)


@app.cell
def _(PARTITIONS, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(LEVELS_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition]
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level: ")
    return LEVELS, level_slider


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition):
    VARIABLES = VARIABLES_DICT[partition]
    if partition == "surface":
        VARIABLES_VALUE = VARIABLES[2]
    else:
        VARIABLES_VALUE = VARIABLES[3]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLES_VALUE, label="variable : ")
    return VARIABLES, var_dropdown


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(mo):
    N_slider = mo.ui.slider(1, 20, value=1, label="N: ")
    return (N_slider,)


@app.cell
def _(N_slider):
    N = N_slider.value
    return (N,)


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
    lambda_trajectory = T_schedule(25, lambda_shape, w) 
    return (lambda_trajectory,)


@app.cell
def _(lambda_trajectory, plot_trajectory):
    lambda_trajectory_plot = plot_trajectory(lambda_trajectory, "lambda", ymax=5)
    return (lambda_trajectory_plot,)


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(-5, 5, value=1, label="alpha: ", step=0.1)
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
def _(N, N_schedule, alpha, torch, y_shape):
    y_trajectory = N_schedule(N, y_shape, alpha) 
    y_trajectory = [torch.tensor(0.0, dtype=torch.float32)] + y_trajectory
    return (y_trajectory,)


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
def _(avg_over_mask, get_guidance_trajectory, mask, slice, y_trajectory):
    avg_over_mask_denormalized  = avg_over_mask(slice, mask)
    guidance_terms_denormalized  = get_guidance_trajectory(y_trajectory, avg_over_mask_denormalized)

    guidance_trajectory = guidance_terms_denormalized
    return guidance_terms_denormalized, guidance_trajectory


@app.cell
def _(guidance_trajectory, var, y_trajectory):
    from src.interaction import plot_dual_trajectory
    y_trajectory_plot = plot_dual_trajectory(
        y_trajectory=y_trajectory,
        guidance_trajectory=guidance_trajectory,
        var=var,
        ymin_left=-1,
        ymax_left=1
    )
    return (y_trajectory_plot,)


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


@app.cell
def _(
    N,
    N_slider,
    alpha_slider,
    lambda_shape_slider,
    lambda_trajectory_plot,
    level_slider,
    map_widget,
    mo,
    partition_dropdown,
    timestamp_dropdown,
    var_dropdown,
    w_slider,
    y_shape_slider,
    y_trajectory_plot,
):
    mo.vstack([
        mo.hstack([
            timestamp_dropdown,
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        mo.hstack([N_slider, mo.md(f"{N}")], justify="start"),
        mo.hstack([
            mo.vstack([
                y_shape_slider,
                alpha_slider,
                y_trajectory_plot,
            ]),
            mo.vstack([
                lambda_shape_slider,
                w_slider,
                lambda_trajectory_plot,
            ]),
        ]),
        map_widget
    ])
    return


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
    experiments = Path("data", "ensemble_rollouts").glob("2026*")
    experiments = sorted(experiments)[::-1]
    pick_dropdown = mo.ui.dropdown(label="Pick experiment", value=experiments[0], options=experiments)
    return (pick_dropdown,)


@app.cell
def _(pick_dropdown):
    from src.utils import read_config
    experiment_dir = pick_dropdown.value
    cfg = read_config(experiment_dir)
    return cfg, experiment_dir


@app.cell
def _(cfg, mo, pick_dropdown, refresh_button):
    mo.vstack([
        mo.hstack([
            pick_dropdown, refresh_button,
        ], justify="start"),
        mo.vstack([
            mo.md("#### ----- Experiment params -----"),
            mo.md("<br>".join(f"{k}: {v}" for k, v in cfg.items()))
        ]),
    ])
    return


@app.cell
def _(cfg):
    M = cfg["M"]
    return (M,)


@app.cell
def _():
    from src.utils import read_state, get_slice

    return get_slice, read_state


@app.cell
def _(
    M,
    avg_over_mask,
    cfg,
    experiment_dir,
    get_slice,
    level,
    mask,
    np,
    partition,
    read_state,
    timestamp,
    torch,
    var,
):
    avgs_avgs = []
    for n in range(1, cfg["N"]+1):
        states = [read_state(experiment_dir, str(n), str(m)) for m in range(M)]
        slices = [get_slice(state, partition, level, var, timestamp) for state in states]
        slices = [torch.tensor(np.asarray(slice)) for slice in slices]
        avgs = [avg_over_mask(slice, mask) for slice in slices]
        avgs_avgs.append(avgs)
    return (avgs_avgs,)


@app.cell
def _(avgs_avgs):
    avgs_avgs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run guided rollout
    """)
    return


@app.cell
def _():
    from src.utils import save_config, ensure_results_dir
    from src.rollout import rollout

    return (save_config,)


@app.cell
def _(device, gen_model, rollout_step, state_to_device, torch):
    def run(
        result_dir, 
        ds,
        x_start,
        guidance_terms_denormalized,
        lambda_,
        mask_corners,
        N,
        timestamp,
        timestamp_idx,
        partition,
        level,
        level_idx,
        var,
        var_idx,
    ):
        x_start = state_to_device(x_start, device)
        y = torch.as_tensor(guidance_terms_denormalized, device=device)

        return rollout_step(
            result_dir=result_dir,
            ds=ds,
            x_start=x_start,
            gen_model=gen_model,
            mask_corners=mask_corners,
            y=y,
            lambda_=lambda_,
            N=N,
            partition=partition,
            level_idx=level_idx,
            var_idx=var_idx
        )

    return (run,)


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
    ds,
    ensure_result_dir,
    guidance_terms_denormalized,
    lambda_trajectory,
    level,
    level_idx,
    mask_corners,
    partition,
    run,
    run_button,
    save_config,
    set_status,
    status,
    timestamp,
    timestamp_idx,
    var,
    var_idx,
    x_start,
    y_trajectory,
):
    if run_button.value and status == "RUNNING":
        try:
            result_dir = ensure_result_dir()

            for mc in [mask_corners, None]:
                run(
                    result_dir,
                    ds,
                    x_start,
                    guidance_terms_denormalized,
                    lambda_trajectory,
                    mc,
                    N,
                    timestamp,
                    timestamp_idx,
                    partition,
                    level,
                    level_idx,
                    var,
                    var_idx,
                )

            config = {
                "N": N,
                "mask_corners": mask_corners,
                "timestamp": str(timestamp),
                "timestamp_idx": int(timestamp_idx),
                "partition": partition,
                "level": None if level is None else str(level),
                "level_idx": None if level_idx is None else int(level_idx),
                "var": var,
                "var_idx": int(var_idx),
                "y": [g_t.item() for g_t in guidance_terms_denormalized],
                "y_perc": [y_t.item() for y_t in y_trajectory],
                "lambda_": [l_t.item() for l_t in lambda_trajectory]
            }

            save_config(result_dir, config)

            set_status("IDLE")
        except:
            set_status("IDLE")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
