import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", css_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # todos
    - define masks with physical priors
    - define masks dynamically in N
    - experiment with multiple variables (and masks correspondingly)
    - define weighted average Gaussian Kernel or future difference around region in loss function. Refine latex-notes with new definition of mask.
    - Try out regularization term z^tK^z or just z^tIz=||z||^2
    - Guide using the ground truth and see whether the accuracy of other variables improves.
    - Define an ensemble of $G$ guided models.
    - as baseline compute some basic facts about ArchesWeatherGen. For instance, how well it does (compared to its deterministic brother)? How does performance degrade as N of rollout increases?
    - swap rollout_dist_plot with newer version present in analyze.py
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
    return Path, datetime, mo, np, torch


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
        read_states, xr_to_torch, list_tens_to_floats
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
    level_slider = mo.ui.slider(steps=LEVELS, value=unguided_cfg["level"], label="level: ", debounce=True)
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
def _(mo, np):
    min_max_lambda_slider = mo.ui.slider(steps=np.logspace(-5,5,100), label="alpha boundaries: ")
    return (min_max_lambda_slider,)


@app.cell
def _(min_max_lambda_slider):
    min_max_lambda = min_max_lambda_slider.value
    return (min_max_lambda,)


@app.cell
def _(min_max_lambda, mo):
    alpha_slider = mo.ui.slider(-min_max_lambda, min_max_lambda, value=min_max_lambda/2, label="alpha: ", step=0.001)
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
    # align with "state" timestamps
    # first timestamp in ds is only available as "previous"!
    TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
    return (TIMESTAMPS,)


@app.cell
def _():
    # ds[0]
    # tensor_timestamp_to_string(torch.tensor([1577858400]))
    return


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
def _():
    # unguided_cfg, timestamps, timestamp_idx, TIMESTAMPS
    return


@app.cell
def _(LEVELS, TIMESTAMPS, VARIABLES, ds, level, timestamp, var):
    timestamp_idx = TIMESTAMPS.index(timestamp)
    var_idx = VARIABLES.index(var)
    level_idx = LEVELS.index(level) - 1
    x_start = ds[timestamp_idx]
    return level_idx, timestamp_idx, var_idx, x_start


@app.cell
def _(datetime, torch):
    def tensor_timestamp_to_string(timestamp: torch.Tensor, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Convert a torch tensor containing a Unix timestamp to a formatted string.

        Example:
            tensor(1577923200) -> "2020-01-02 00:00:00"
        """
        ts = timestamp.item()
        return datetime.fromtimestamp(ts).strftime(fmt)

    # NOTE: it's in UTC time
    # tensor_timestamp_to_string(x_start["timestamp"])
    return


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
def _(mo):
    SUBFOLDERS = ["old_model", "new_model", "unguided"]
    subfolder_selector = mo.ui.dropdown(label="Subfolder", value=SUBFOLDERS[0], options=SUBFOLDERS)
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
        [p for p in unguided_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_unguided_rollout_dropdown = mo.ui.dropdown(label="Experiment: ", value=unguided_rollouts[0], options=unguided_rollouts)
    return (pick_unguided_rollout_dropdown,)


@app.cell
def _(pick_unguided_rollout_dropdown, read_json):
    unguided_rollout_dir = pick_unguided_rollout_dropdown.value
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

    # TODO: implement
    deterministic_rollout = []

    for n in range(1, unguided_cfg["N"]+1):
        timestamp_n = TIMESTAMPS[timestamp_idx+n]
        state_n = ds[timestamp_idx+n]["state"]
        slice_n = ds.denormalize(state_n)[partition][var_idx, level_idx]
        ground_truth.append(avg_over_mask(slice_n, mask))
        states = read_states(unguided_rollout_dir, n) 
        # print(states)
        # print("---")
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
def _(
    mean_rollout,
    planned_guidance,
    plot_dual_trajectory,
    timestamps,
    var,
    y_trajectory,
):
    y_trajectory_plot = plot_dual_trajectory(
        timestamps=timestamps,
        mean_rollout=mean_rollout,
        planned_guidance=planned_guidance,
        y_trajectory=y_trajectory,
        var=var,
        ymin_left=None,
        ymax_left=None,
    )
    return (y_trajectory_plot,)


@app.cell
def _(
    ensemble_rollout,
    ground_truth,
    mean_rollout,
    timestamps,
    var,
    visualize_mask_terms_over_N,
):
    ensemble_rollout_plot = visualize_mask_terms_over_N(
        var, timestamps, mean_rollout=mean_rollout, ensemble_rollout=ensemble_rollout, ground_truth=ground_truth, 
    )
    return (ensemble_rollout_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PHASE 2 - guidance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick base unguided rollout
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
    ## Configure mask
    """)
    return


@app.cell
def _(level_slider, map_widget, mo, partition_dropdown, var_dropdown):
    mo.vstack([
        mo.md("Define mask: variable + region of interest. By default the values are set to the ones defined in the unguided rollout experiment config.json file."),
        mo.hstack([
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        map_widget
    ])
    return


@app.cell
def _(ensemble_rollout_plot, mo):
    mo.vstack([
        mo.md("Compares the the average over the defined mask of the M-models generative ensemble to the ground truth."), 
        ensemble_rollout_plot,
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("Compares the the average over the defined mask of the M-models generative ensemble to the deterministic predictions at each step, to see how much the residual diverges from the deterministic prediction."), 
        # ensemble_rollout_plot,
    ])
    return


@app.cell
def _():
    # TODO: create plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure guidance
    """)
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
        mo.md("$\lambda_t$ controls how much the guidance vector is conditioning the vector field $u_t^{\\theta}$ at diffusion timestep $t$."),
        mo.vstack([
            mo.hstack([
                lambda_shape_slider,
                mo.md("controls the smoothness of the lambda trajectory")
            ], justify="start"),
            mo.hstack([
                w_slider,
                mo.md("controls the maximum value of lambda (always @ step 12)")
            ], justify="start"),
            lambda_trajectory_plot,
        ])
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
    alpha_slider,
    min_max_lambda_slider,
    mo,
    y_shape_slider,
    y_trajectory_plot,
):
    mo.vstack([
        mo.vstack([
            mo.md("Configure trajectory over N steps."), 
            y_shape_slider,
            min_max_lambda_slider,
            alpha_slider,
            y_trajectory_plot,
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
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([
        run_button,
        # mo.md(f"Experiment status: **{get_status()}**")
    ])
    return (run_button,)


@app.cell
def _():
    TEST=True
    return (TEST,)


@app.cell
def _(
    N,
    Path,
    ROLLOUTS,
    TEST,
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
            m=1,
            seed=None,
            test=TEST
        )

        config = {
            "unguided_rollout_dir": unguided_cfg["rollout_dir"],
            "guided_rollout_dir": str(rollout_dir),
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


        save_to_json(config, Path(ROLLOUTS, "guided", rollout_dir), "config")

    #     set_status("IDLE")
    # except Exception as e:
    #     print(f"Error: {type(e).__name__}: {e}")
    #     raise

    # finally:
    #     set_status("IDLE")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
