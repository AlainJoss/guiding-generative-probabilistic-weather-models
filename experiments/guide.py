import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", css_file="")


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

    ## ideas
    - steer model towards grount truth in mask and measure divergence across states
    - generate multiple (vars, levels, partitions) no-guidance N rollouts and save in experiment folder -> need a way to encode the experiment (log of experiments or yaml with keys).
    - as baseline compute some basic facts about ArchesWeatherGen. For instance, how well it does (compared to its deterministic brother)? How does performance degrade as N of rollout increases?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
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
    return ChartPuck, CubicSpline, method_state, mo, np, torch


@app.cell
def _():
    from src.utils import get_last_experiment_dir, get_mask_corners_from_widget, get_mask_from_corners
    from src.interaction import visualize_map
    from src.funcs import avg_over_mask, get_guidance_trajectory
    from src.utils import state_to_device

    from geoarches.lightning_modules import load_module
    from geoarches.dataloaders.era5 import Era5Forecast

    return (
        Era5Forecast,
        avg_over_mask,
        get_guidance_trajectory,
        get_mask_corners_from_widget,
        get_mask_from_corners,
        load_module,
        state_to_device,
        visualize_map,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spline Puckchart
    """)
    return


@app.cell
def _(ChartPuck, CubicSpline, MAX_PERC_DELTA, N_slider, method_state, np):
    ##### splinechart

    def draw_spline_chart(ax, x_pucks, y_pucks, method):
        # Add fixed left anchor at (0, 0)
        x_all = np.concatenate([[0], x_pucks])
        y_all = np.concatenate([[0], y_pucks])

        # Sort points by x for proper spline fitting
        sorted_indices = np.argsort(x_all)
        x_sorted = x_all[sorted_indices]
        y_sorted = y_all[sorted_indices]

        # Ensure strictly increasing x by adding small perturbations to duplicates
        for i in range(1, len(x_sorted)):
            if x_sorted[i] <= x_sorted[i - 1]:
                x_sorted[i] = x_sorted[i - 1] + 1e-6

        # Dense x values over the full axis 0 ... N-1
        x_dense = np.linspace(0, N_slider.value, 200)

        spline = CubicSpline(x_sorted, y_sorted)
        y_dense = spline(x_dense)

        ax.plot(x_dense, y_dense, "b-", linewidth=2)
        ax.set_xlim(0, N_slider.value)
        ax.set_ylim(-MAX_PERC_DELTA, MAX_PERC_DELTA)
        ax.set_xticks(np.arange(0, N_slider.value, 1))
        ax.set_xlabel("N")
        ax.set_ylabel("Percentage change")
        ax.set_title("Trajectory")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

        # Fixed anchor
        ax.plot([0], [0], "ko", markersize=8, zorder=5)

    def draw_spline(ax, widget):
        # N total points INCLUDING the fixed anchor => N draggable pucks
        fixed_x = np.linspace(1, N_slider.value, len(widget.y)).tolist()

        try:
            widget.x = fixed_x
        except Exception:
            pass

        draw_spline_chart(ax, fixed_x, list(widget.y), method_state["value"])

    # Slider value N = total number of points INCLUDING the anchor
    _n = N_slider.value

    # Draggable pucks are at x = 1, 2, ..., N
    _init_x = np.linspace(1, _n, _n).tolist()
    _init_y = (0.15 * np.sin(np.pi * np.array(_init_x) / (_n))).tolist()

    spline_puck = ChartPuck.from_callback(
        draw_fn=draw_spline,
        x_bounds=(1, _n),
        y_bounds=(-MAX_PERC_DELTA, MAX_PERC_DELTA),
        figsize=(6, 4),
        x=_init_x,
        y=_init_y,
        puck_color="#9c27b0",
        drag_y_bounds=(-MAX_PERC_DELTA, MAX_PERC_DELTA),
    )
    return (spline_puck,)


@app.cell
def _(method_state, spline_puck):
    def on_method_change(new_val):
        method_state["value"] = new_val
        spline_puck.redraw()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guidance experiments
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load
    Data and model.
    """)
    return


@app.cell
def _(Era5Forecast):
    ds = Era5Forecast(
        path="data/era5_240/full",  # default path
        domain="test", # domain to consider. domain = 'test' loads the 2020 period
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=6
    )
    return (ds,)


@app.cell
def _(device, load_module):
    module_target = "geoarches.lightning_modules.{}"  # appendix
    gen_model, gen_config = load_module(
        "archesweathergen",
        module_target=module_target.format("guided_diffusion.GuidedFlow"),
    )
    gen_model = gen_model.to(device)
    return (gen_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constants
    """)
    return


@app.cell
def _(ds):
    # remove first and last, since we have two tensordicts less (because of prev and next)
    TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
    return (TIMESTAMPS,)


@app.cell
def _():
    PARTITIONS = ["surface", "level"]
    return (PARTITIONS,)


@app.cell
def _(partition):
    LEVELS_DICT = {
        "surface": [i+1 for i in range(1)],
        "level": [i+1 for i in range(13)]
    }
    LEVELS = LEVELS_DICT[partition]
    return (LEVELS,)


@app.cell
def _(partition):
    VARIABLES_DICT = {
        "surface": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure"
        ],
        "level": [
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            "vertical_velocity"
        ]
    }
    VARIABLES = VARIABLES_DICT[partition]
    return (VARIABLES,)


@app.cell
def _():
    device = "mps"

    MAX_PERC_DELTA = 100/100

    VMIN = None
    VMAX = None
    return MAX_PERC_DELTA, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(TIMESTAMPS, mo):
    timestamp_dropdown = mo.ui.dropdown(TIMESTAMPS, value=TIMESTAMPS[0], label="start state : ")
    return (timestamp_dropdown,)


@app.cell
def _(PARTITIONS, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(LEVELS, mo):
    level_dropdown = mo.ui.dropdown(LEVELS, value=LEVELS[0], label="level: ")
    return (level_dropdown,)


@app.cell
def _(VARIABLES, mo):
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLES[2], label="variable : ")
    return (var_dropdown,)


@app.cell
def _(mo):
    N_slider = mo.ui.slider(3, 20, value=6, label="N: ")
    return (N_slider,)


@app.cell
def _(mo, spline_puck):
    spline_widget = mo.ui.anywidget(spline_puck)  # cannot be put with n_pucks_slider
    return (spline_widget,)


@app.cell
def _(denormalized_slice, visualize_map):
    map_widget = visualize_map(
        denormalized_slice,
        title="Select mask region",
        interactive=True,
        undo_roll=True,
        vmin=denormalized_slice.min(),
        vmax=denormalized_slice.max(),
        center= denormalized_slice.mean()
    )
    return (map_widget,)


@app.cell
def _(
    N_slider,
    level_dropdown,
    map_widget,
    mo,
    partition_dropdown,
    spline_widget,
    timestamp_dropdown,
    var_dropdown,
):
    mo.vstack([
        timestamp_dropdown,
        var_dropdown,
        partition_dropdown,
        level_dropdown,
        N_slider,
        spline_widget,
        map_widget
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derived quantities
    - dropdown variables and associated ids
    - ids
    - mask
    - trajectory
    """)
    return


@app.cell
def _(timestamp_dropdown):
    timestamp = timestamp_dropdown.value
    return (timestamp,)


@app.cell
def _(level_dropdown):
    level = level_dropdown.value
    return (level,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(N_slider):
    N = N_slider.value
    return (N,)


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
def _(gen_model, level_idx, partition, var_idx, x_start):
    denormalized_slice = gen_model.denormalize(x_start["state"])[partition][var_idx, level_idx]
    return (denormalized_slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The denormalized prediction at time $t$ is defined as follows:

    \begin{align*}
    \hat{x}^\text{denorm} &= \mu^\text{denorm} + (\hat{x}^\text{norm}) \odot \sigma^\text{denorm} \\
    &= \mu^\text{denorm} + (\hat{\mu}^\text{det} + \sigma^\text{residual} \odot z_t) \odot \sigma^\text{denorm}
    \end{align*}

    Importantly, we have to perform two denormalizations. One in the residual space, and one in the data space. The second one is standard machine learning preprocessing, whilst the first one comes from the way the target of the generative model is built, which is as follows:

    \begin{align*}
    r^\text{gt} &=  \frac{x^\text{gt} - \hat{x}^\text{det}}{\sigma^\text{residual}}
    \end{align*}

    Note, $\sigma^\text{residual}$ is computed as standard deviation of the residuals $\sigma^\text{residual}_{n \in [N]}(x^\text{gt}_n - \hat{x}_n^\text{det})$ over the dataset.

    We are interested in defining the guidance in denormalized space.
    The loss is then defined as follows:

    \begin{align*}
    \mathcal{L}^\text{denorm}(y_t, \hat{x}^\text{denorm}) &=  \frac{1}{2} \left\lVert y_t - \frac{\sum_i \hat{x}^\text{denorm}_i \odot m_i}{\sum_i m_i} \right\lVert^2_2
    \end{align*}
    """)
    return


@app.cell
def _(
    avg_over_mask,
    denormalized_slice,
    get_guidance_trajectory,
    mask,
    spline_widget,
):
    avg_over_mask_denormalized  = avg_over_mask(denormalized_slice, mask)
    guidance_terms_denormalized  = get_guidance_trajectory(spline_widget.y, avg_over_mask_denormalized)

    guidance_trajectory = [avg_over_mask_denormalized] + guidance_terms_denormalized

    # avg_over_mask_denormalized, guidance_terms_denormalized
    return guidance_terms_denormalized, guidance_trajectory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run experiments
    """)
    return


@app.cell
def _(device, gen_model, state_to_device, torch):
    from src.rollout_step import rollout_step

    def run(
        ds,
        x_start,
        guidance_terms_denormalized,
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
        batch = state_to_device(x_start, device)
        y_t = torch.as_tensor(guidance_terms_denormalized, device=gen_model.device)[0]

        return rollout_step(
            ds=ds,
            x_start=batch,
            gen_model=gen_model,
            mask_corners=mask_corners,
            y_t=y_t,
            N=N,
            timestamp=timestamp,
            timestamp_idx=timestamp_idx,
            partition=partition,
            level=level,
            level_idx=level_idx,
            var=var,
            var_idx=var_idx,
        )

    return (run,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run experiment")
    run_button
    return (run_button,)


@app.cell
def _(
    N,
    ds,
    guidance_terms_denormalized,
    level,
    level_idx,
    mask_corners,
    partition,
    run,
    run_button,
    timestamp,
    timestamp_idx,
    var,
    var_idx,
    x_start,
):
    if run_button.value:
        run(
            ds,
            x_start,
            guidance_terms_denormalized,
            mask_corners,
            N,
            timestamp,
            timestamp_idx,
            partition,
            level,
            level_idx,
            var,
            var_idx,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis
    """)
    return


@app.cell
def _(guidance_trajectory, var):
    from src.interaction import plot_trajectory
    fig = plot_trajectory(guidance_trajectory, var)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
