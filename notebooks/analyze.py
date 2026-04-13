import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boiler plate
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
    import marimo as mo
    import torch
    # import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, mo, np, pd, plt


@app.cell
def _():
    from src.utils import (
        read_json,
        read_state,
        get_last_experiment_dir,
        get_slice
    )
    from src.interaction import visualize_map, get_mask_from_corners, get_mask_center

    return (
        get_mask_center,
        get_mask_from_corners,
        get_slice,
        read_state,
        visualize_map,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _(np, plt):
    def extract_zoom_values(
        array_2d,
        *,
        zoom,
        center_lon=0.0,
        center_lat=0.0,
    ):
        array_2d = np.asarray(array_2d)
        ny, nx = array_2d.shape

        lon_e = np.linspace(-180.0, 180.0, nx + 1, endpoint=True)
        lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)

        lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
        lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

        zoom = max(1, int(zoom))

        full_lon_span = 360.0
        full_lat_span = 180.0

        lon_span = full_lon_span / zoom
        lat_span = full_lat_span / zoom

        lon_min = max(-180.0, center_lon - lon_span / 2)
        lon_max = min(180.0, center_lon + lon_span / 2)
        lat_min = max(-90.0, center_lat - lat_span / 2)
        lat_max = min(90.0, center_lat + lat_span / 2)

        lon_mask = (lon_c >= lon_min) & (lon_c <= lon_max)
        lat_mask = (lat_c >= lat_min) & (lat_c <= lat_max)

        zoom_values = array_2d[np.ix_(lat_mask, lon_mask)]
        return zoom_values[np.isfinite(zoom_values)]


    def plot_zoom_histogram(
        array_2d,
        *,
        zoom,
        center_lon=0.0,
        center_lat=0.0,
        bins=30,
        title="Value distribution in zoom zone",
        figsize=(11.75, 5.45),
        dpi=200,
    ):
        values = extract_zoom_values(
            array_2d,
            zoom=zoom,
            center_lon=center_lon,
            center_lat=center_lat,
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        return fig

    return (plot_zoom_histogram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _():
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    return LEVELS_DICT, PARTITIONS, VARIABLES_DICT


@app.cell
def _(PARTITIONS, mo):
    partition_dropdown = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition_dropdown,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(LEVELS_DICT, partition):
    LEVELS = LEVELS_DICT[partition][::-1]
    return (LEVELS,)


@app.cell
def _(LEVELS, mo):
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level: ")
    return (level_slider,)


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, partition):
    VARIABLES = VARIABLES_DICT[partition]
    if partition == "surface":
        VARIABLES_VALUE = VARIABLES[2]
    else:
        VARIABLES_VALUE = VARIABLES[3]
    return VARIABLES, VARIABLES_VALUE


@app.cell
def _(VARIABLES, VARIABLES_VALUE, mo):
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLES_VALUE, label="variable: ")
    return (var_dropdown,)


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(mo):
    zoom_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=1,
        label="Zoom",
    )
    return (zoom_slider,)


@app.cell
def _(zoom_slider):
    zoom = zoom_slider.value
    return (zoom,)


@app.cell
def _(mo):
    show_mask_switch = mo.ui.switch(label="show mask")
    return (show_mask_switch,)


@app.cell
def _(show_mask_switch):
    show_mask = show_mask_switch.value
    return (show_mask,)


@app.cell
def _(get_mask_center, mask):
    zoom_centers = get_mask_center(mask)
    return (zoom_centers,)


@app.cell
def _(analysis_type_dropdown):
    analysis_type = analysis_type_dropdown.value
    return (analysis_type,)


@app.cell
def _(mo):
    analysis_types = ["absolute", "difference"]
    analysis_type_dropdown = mo.ui.dropdown(analysis_types, value=analysis_types[0], label="analysis type: ")
    return (analysis_type_dropdown,)


@app.cell
def _(mo):
    show_values_checkbox = mo.ui.checkbox(label="show_values")
    return (show_values_checkbox,)


@app.cell
def _(show_values_checkbox):
    show_values = show_values_checkbox.value
    return (show_values,)


@app.cell
def _(guided_unguided, mo):
    center_ = 0.0
    max_ = max(guided_unguided.max().item(), abs(guided_unguided.min().item()))
    mean_ = (max_ - center_)*9 / 10
    value_threshold_slider  = mo.ui.slider(
        start=center_,
        stop=max_,
        step=mean_/10,
        value=mean_,
        label="Text thresh",
    )
    return (value_threshold_slider,)


@app.cell
def _(value_threshold_slider):
    value_threshold = value_threshold_slider.value
    return (value_threshold,)


@app.cell
def _(cfg, mo):
    n_slider = mo.ui.slider(steps=range(1, cfg["N"]+1), value=1, label="n: ")
    return (n_slider,)


@app.cell
def _(n_slider):
    n = n_slider.value
    return (n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _():
    from geoarches.dataloaders.era5 import Era5Forecast

    ds = Era5Forecast(
        path="data/era5_240/full",  # default path
        domain="test", # domain to consider. domain = 'test' loads the 2020 period
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=6
    )
    return (ds,)


@app.cell
def _(cfg, ds, n, unix_tensor_to_iso):
    x_start = ds[cfg["timestamp_idx"]+n]
    x_start = ds.denormalize(x_start)
    current = ds.convert_to_xarray(x_start["state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))
    next = ds.convert_to_xarray(x_start["next_state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))

    lead_time_seconds = x_start["lead_time_hours"] * 3600
    timestamp = x_start["timestamp"] - lead_time_seconds
    timestamp = unix_tensor_to_iso(timestamp)
    return current, next, timestamp, x_start


@app.cell
def _(pd):
    def unix_tensor_to_iso(ts):
        return pd.to_datetime(int(ts.item()), unit="s").strftime("%Y-%m-%dT%H:%M:%S")

    return (unix_tensor_to_iso,)


@app.cell
def _(
    current,
    get_slice,
    guided,
    level,
    next,
    partition,
    timestamp,
    unguided,
    var,
):
    guided_slice = get_slice(guided, partition, level, var, timestamp)
    unguided_slice = get_slice(unguided, partition, level, var, timestamp)
    current_slice = get_slice(current, partition, level, var, timestamp)
    next_slice = get_slice(next, partition, level, var, timestamp)
    return current_slice, guided_slice, next_slice, unguided_slice


@app.cell
def _(current_slice, guided_slice, next_slice, unguided_slice):
    guided_unguided = guided_slice - unguided_slice
    next_guided = next_slice - guided_slice
    next_unguided = next_slice - unguided_slice
    next_current = next_slice - current_slice 
    return guided_unguided, next_current, next_guided, next_unguided


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results analysis
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
    refresh_button = mo.ui.button(label="refresh")
    return (refresh_button,)


@app.cell
def _(Path, mo, refresh_button):
    if refresh_button.value:
        pass
    experiments = Path("data", "guided_rollouts").glob("2026*")
    experiments = sorted(experiments)[::-1]
    pick_dropdown = mo.ui.dropdown(label="Pick experiment", value=experiments[0], options=experiments)
    return (pick_dropdown,)


@app.cell
def _(read_config, result_dir):
    cfg = read_config(result_dir)
    return (cfg,)


@app.cell
def _(pick_dropdown):
    result_dir = pick_dropdown.value
    return (result_dir,)


@app.cell
def _(n, read_state, result_dir):
    guided = read_state(result_dir, "guided", n)
    unguided = read_state(result_dir, "unguided", n)
    return guided, unguided


@app.cell
def _(cfg, get_mask_from_corners):
    mask = get_mask_from_corners(*cfg["mask_corners"])
    return (mask,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## n-snapshots
    """)
    return


@app.cell
def _(
    analysis_type,
    current_slice,
    guided_slice,
    guided_unguided,
    mask,
    next_current,
    next_guided,
    next_slice,
    next_unguided,
    np,
    show_mask,
    show_values,
    unguided_slice,
    value_threshold,
    visualize_map,
    zoom,
    zoom_centers,
    zoom_slider,
):
    if analysis_type == "absolute":
        state_map = visualize_map(
            current_slice,
            mask_2d=np.asarray(mask),
            title="$x_t$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center= current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )

        next_map = visualize_map(
            next_slice,
            mask_2d=np.asarray(mask),
            title="$x_{t+1}$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center= current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )

        guided_map = visualize_map(
            guided_slice,
            mask_2d=np.asarray(mask),
            title="$x_{t+1}^{guide}$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center=current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )

        unguided_map = visualize_map(
            unguided_slice,
            mask_2d=np.asarray(mask),
            title="$x_{t+1}^{gen}$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center=current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    else:
        next_current_map = visualize_map(
            next_current,
            mask_2d=np.asarray(mask),
            title="next-current",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center= next_current.mean(),
            show_mask=show_mask,
            zoom=zoom,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
        print(guided_unguided.max())
        guided_unguided_map = visualize_map(
            guided_unguided,
            mask_2d=np.asarray(mask),
            title="guided-unguided",
            vmin=guided_unguided.min(),
            vmax=guided_unguided.max(),
            center= guided_unguided.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
            show_values=show_values,
            value_threshold=value_threshold,
            value_fontsize=5,
        )

        next_guided_map = visualize_map(
            next_guided,
            mask_2d=np.asarray(mask),
            title="next-guided",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center=next_current.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )

        next_unguided_map = visualize_map(
            next_unguided,
            mask_2d=np.asarray(mask),
            title="next-unguided",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center=next_current.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    return (
        guided_map,
        guided_unguided_map,
        next_current_map,
        next_map,
        next_unguided_map,
        state_map,
        unguided_map,
    )


@app.cell
def _(guided_unguided, plot_zoom_histogram, zoom_centers, zoom_slider):
    next_guided_hist = plot_zoom_histogram(
        guided_unguided,
        zoom=zoom_slider.value,
        center_lon=zoom_centers[0],
        center_lat=zoom_centers[1],
        title="unguided-guided - value dist in zoom area",
    )
    return (next_guided_hist,)


@app.cell
def _(
    analysis_type,
    analysis_type_dropdown,
    guided_map,
    guided_unguided_map,
    level,
    level_slider,
    mo,
    n_slider,
    next_current_map,
    next_guided_hist,
    next_map,
    next_unguided_map,
    partition_dropdown,
    show_mask_switch,
    show_values_checkbox,
    state_map,
    unguided_map,
    value_threshold_slider,
    var_dropdown,
    zoom_slider,
):
    if analysis_type == "absolute":
        to_show = mo.vstack([
            analysis_type_dropdown,
            n_slider, 
            mo.hstack(
                [partition_dropdown, var_dropdown, mo.hstack([level_slider, mo.md(f"{level}")], justify="start")],
                justify="start",
            ),
            mo.hstack([show_mask_switch], justify="start"),
            zoom_slider,
            mo.hstack([state_map, next_map]),
            mo.hstack([unguided_map, guided_map]),
        ])
    else:
        to_show = mo.vstack([
            analysis_type_dropdown,
            n_slider,
            mo.hstack([partition_dropdown, var_dropdown, mo.hstack([level_slider, mo.md(f"{level}")], justify="start")], justify="start"),
            mo.hstack([show_mask_switch, show_values_checkbox, value_threshold_slider], justify="start"),
            zoom_slider,
            mo.hstack([next_current_map, guided_unguided_map]),
            mo.hstack([next_unguided_map, next_guided_hist])
        ])
    to_show
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N-trajectory
    """)
    return


@app.cell
def _():
    from src.funcs import avg_over_mask

    return (avg_over_mask,)


@app.cell
def _(
    avg_over_mask,
    cfg,
    ds,
    get_slice,
    mask,
    read_state,
    result_dir,
    unix_tensor_to_iso,
    x_start,
):
    realized_y_guided = [] 
    realized_y_unguided = [] 
    ground_truth = []

    for n_ in range(1, cfg["N"]+1):
        x_start_ = ds[cfg["timestamp_idx"]]
        lead_time_seconds_ = x_start_["lead_time_hours"] * 3600
        timestamp_ = x_start["timestamp"]
        timestamp_ = unix_tensor_to_iso(timestamp_)

        state = ds.convert_to_xarray(x_start["next_state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))
        state = get_slice(state, cfg["partition"], cfg["level"], cfg["var"], timestamp_)
        state = state.to_numpy()
        avg_ = avg_over_mask(state, mask)
        ground_truth.append(avg_)

        state = read_state(result_dir, "guided", n_)
        state = get_slice(state, cfg["partition"], cfg["level"], cfg["var"], timestamp_)
        state = state.to_numpy()
        avg_ = avg_over_mask(state, mask)
        realized_y_guided.append(avg_)

        state = read_state(result_dir, "unguided", n_)
        state = get_slice(state, cfg["partition"], cfg["level"], cfg["var"], timestamp_)
        state = state.to_numpy()
        avg_ = avg_over_mask(state, mask)
        realized_y_unguided.append(avg_)

    ground_truth = [cfg["y"][0]] + ground_truth
    realized_y_guided = [cfg["y"][0]] + realized_y_guided
    realized_y_unguided = [cfg["y"][0]] + realized_y_unguided
    return ground_truth, realized_y_guided, realized_y_unguided


@app.cell
def _(np, plt):
    def plot_realized_trajectories(planned, guided, unguided, ground_truth, var: str):
        x = np.arange(len(guided))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(x, planned, "-o", linewidth=2, markersize=5, label="planned")
        ax.plot(x, guided, "-o", linewidth=2, markersize=5, label="guided")
        ax.plot(x, unguided, "-o", linewidth=2, markersize=5, label="unguided")
        ax.plot(x, ground_truth, "-o", linewidth=2, markersize=5, label="ground truth")

        ax.set_xlim(0, len(x) - 1 if len(x) > 1 else 1)
        ax.set_xticks(x)
        ax.set_xlabel("N")
        ax.set_ylabel(var)
        ax.set_title("Realized trajectories")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    return (plot_realized_trajectories,)


@app.cell
def _(
    cfg,
    ground_truth,
    plot_realized_trajectories,
    realized_y_guided,
    realized_y_unguided,
    var,
):
    plot_realized_trajectories(cfg["y"], realized_y_guided, realized_y_unguided, ground_truth, var)
    return


@app.cell
def _():
    # the planning should really happening with respect to the forecast itself
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
