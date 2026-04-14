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
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, mo, np, pd, plt


@app.cell
def _():
    from src.utils import (
        read_json,
        read_state,
        read_states,
        get_last_experiment_dir,
        get_slice
    )
    from src.interaction import visualize_map, get_mask_from_corners, get_mask_center
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.funcs import avg_over_mask

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        get_mask_center,
        get_mask_from_corners,
        get_slice,
        read_json,
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


@app.cell
def _(pd):
    def unix_tensor_to_iso(ts):
        return pd.to_datetime(int(ts.item()), unit="s").strftime("%Y-%m-%dT%H:%M:%S")

    return (unix_tensor_to_iso,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


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
def _(guided_cfg, mo):
    n_slider = mo.ui.slider(steps=range(1, guided_cfg["N"]+1), value=1, label="n: ")
    return (n_slider,)


@app.cell
def _(n_slider):
    n = n_slider.value
    return (n,)


@app.cell
def _(mo, unguided_cfg):
    m_slider = mo.ui.slider(
        start=1,
        stop=unguided_cfg["M"],
        step=1,
        value=1,
        label="m: ",
    )
    return (m_slider,)


@app.cell
def _(m_slider):
    m = m_slider.value
    return (m,)


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
def _(ds, guided_cfg, n, unix_tensor_to_iso):
    x_start = ds[guided_cfg["timestamp_idx"]+n]
    x_start = ds.denormalize(x_start)
    current = ds.convert_to_xarray(x_start["state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))
    next = ds.convert_to_xarray(x_start["next_state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))

    lead_time_seconds = x_start["lead_time_hours"] * 3600
    timestamp = x_start["timestamp"] - lead_time_seconds
    timestamp = unix_tensor_to_iso(timestamp)
    return current, next, timestamp


@app.cell
def _(
    current,
    get_slice,
    guided_state,
    level,
    next,
    partition,
    timestamp,
    unguided_state,
    var,
):
    guided_slice = get_slice(guided_state, partition, level, var, timestamp)
    unguided_slice = get_slice(unguided_state, partition, level, var, timestamp)
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
    # refresh_button
    return (refresh_button,)


@app.cell
def _(Path, mo, refresh_button):
    if refresh_button.value:
        pass

    def has_config_json(path: Path) -> bool:
        return (path / "config.json").exists()


    guided_rollouts = Path("rollouts", "guided").glob("2026*")
    guided_rollouts = sorted(
        [p for p in guided_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_guided_rollout_dropdown = mo.ui.dropdown(label="Pick guided rollout", value=guided_rollouts[0], options=guided_rollouts)
    # pick_guided_rollout_dropdown
    return (pick_guided_rollout_dropdown,)


@app.cell
def _(pick_guided_rollout_dropdown):
    guided_rollout_dir = pick_guided_rollout_dropdown.value
    return (guided_rollout_dir,)


@app.cell
def _(Path, guided_rollout_dir, read_json):
    guided_cfg = read_json(guided_rollout_dir, "config")
    unguided_rollout_dir = Path(guided_cfg["unguided_rollout_dir"])
    unguided_cfg = read_json(unguided_rollout_dir, "config")
    return guided_cfg, unguided_cfg, unguided_rollout_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Realized guidance
    """)
    return


@app.cell
def _(get_guidance, plt):
    def realized_guidance_branches(
        realized_terms: list[float],
        y_perc: list[float],
    ) -> list[float]:
        if len(realized_terms) != len(y_perc):
            raise ValueError("realized_terms and y_perc must have the same length")

        branches = [realized_terms[0]]
        for n in range(1, len(realized_terms)):
            branches.append(get_guidance(y_perc[n], realized_terms[n - 1]))
        return branches


    def plot_guidance_branching(
        timestamps: list[str],
        realized_terms: list[float],
        y_perc: list[float],
        planned_guidance: list[float] | None = None,
        ground_truth: list[float] | None = None,
        mean_rollout: list[float] | None = None,
        ensemble_rollout: list[list[float]] | None = None,
    ):
        N = len(timestamps)
        if N != len(realized_terms) or len(realized_terms) != len(y_perc):
            raise ValueError("timestamps, realized_terms, and y_perc must have the same length")
        x = list(range(N))
        branch_targets = realized_guidance_branches(realized_terms, y_perc)

        fig, ax = plt.subplots(figsize=(9, 4.5))

        # realized trajectory
        ax.plot(x, realized_terms, marker="o", linewidth=2, label="Realized guidance")

        # branching segments:
        # from realized[n-1] to target computed for step n
        for n in range(1, len(x)):
            ax.plot(
                [x[n - 1], x[n]],
                [realized_terms[n - 1], branch_targets[n]],
                linestyle="--",
                marker="o",
                alpha=0.9,
                label="Online planned guidance" if n == 1 else None,
            )
        
        if ensemble_rollout is not None:
            M = len(ensemble_rollout[0])

            for i in range(M):
                y = [ensemble_rollout[n][i] for n in range(N)]
                ax.plot(x, y, marker="o", alpha=0.8)

            lower = [min(row) for row in ensemble_rollout]
            upper = [max(row) for row in ensemble_rollout]
            ax.fill_between(x, lower, upper, alpha=0.2)

        if planned_guidance is not None:
            ax.plot(x, planned_guidance, marker="o", linewidth=2, label="Offline planned guidance")

        if ground_truth is not None:
            ax.plot(x, ground_truth, marker="o", linewidth=2, label="Ground truth")
        
        if mean_rollout is not None:
            ax.plot(x, mean_rollout, marker="o", linewidth=2, label="Mean rollout")

        tick_idx = [i for i, ts in enumerate(timestamps) if ts.endswith("00:00:00")]
        if 0 not in tick_idx:
            tick_idx = [0] + tick_idx
        if len(timestamps) - 1 not in tick_idx:
            tick_idx.append(len(timestamps) - 1)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels([timestamps[i] for i in tick_idx], rotation=45, ha="right")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Mask term")
        ax.set_title("Realized guidance analysis")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    return (plot_guidance_branching,)


@app.cell
def _():
    from src.funcs import get_guidance

    return (get_guidance,)


@app.cell
def _(guided_cfg, guided_rollout_dir, read_json):
    timestamps = guided_cfg["timestamps"]
    y_perc = guided_cfg["y_perc"]
    planned_guidance = guided_cfg["planned_guidance"]
    ground_truth_ = guided_cfg["ground_truth"]

    # this should come from rollout_dir/mask_terms.json
    realized_terms = read_json(guided_rollout_dir, "mask_terms")["final_mask_terms"]
    return ground_truth_, planned_guidance, realized_terms, timestamps, y_perc


@app.cell
def _(
    ground_truth_,
    guided_cfg,
    planned_guidance,
    plot_guidance_branching,
    realized_terms,
    timestamps,
    y_perc,
):
    # NOTE: 
    # - first branch of online and offline planned should coincide
    realized_guidance_plot = plot_guidance_branching(
        timestamps=timestamps,
        realized_terms=realized_terms,
        y_perc=y_perc,
        planned_guidance=planned_guidance,
        ground_truth=ground_truth_,
        mean_rollout=guided_cfg["mean_rollout"],
        ensemble_rollout=guided_cfg["ensemble_rollout"]
    )
    return (realized_guidance_plot,)


@app.cell
def _(guided_rollout_dir, m, read_state, unguided_rollout_dir):
    paths = [path for path in guided_rollout_dir.glob(f"{1}/*")]
    guided_state = read_state(paths[0])
    paths = [path for path in unguided_rollout_dir.glob(f"{1}/*")]
    unguided_state = read_state(paths[m-1])
    return guided_state, unguided_state


@app.cell
def _(get_mask_from_corners, guided_cfg):
    mask = get_mask_from_corners(*guided_cfg["mask_corners"])
    return (mask,)


@app.cell
def _(guided_cfg, mo, pick_guided_rollout_dropdown, refresh_button):
    mo.vstack([
        mo.hstack([
            pick_guided_rollout_dropdown, refresh_button,
        ], justify="start"),
        mo.accordion(
            {
                "Experiment params": mo.md("<br>".join(f"{k}: {v}" for k, v in guided_cfg.items())),
            }
        )
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
        # print(guided_unguided.max())
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
    m_slider,
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
            mo.hstack([n_slider, m_slider], justify="start"), 
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
            mo.hstack([n_slider, m_slider], justify="start"), 
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
    ## Static analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Realized guidance during sampling
    """)
    return


@app.cell
def _(guided_rollout_dir, read_json):
    mask_terms = read_json(guided_rollout_dir, "mask_terms")
    final_mask_terms = mask_terms["final_mask_terms"]
    all_mask_terms = mask_terms["all_mask_terms"]
    # all_mask_terms
    return (all_mask_terms,)


@app.cell
def _():
    from src.interaction import plot_trajectory

    return (plot_trajectory,)


@app.cell
def _(all_mask_terms, plot_trajectory):
    plots=[]
    for mask_list in  all_mask_terms:
        plots.append(plot_trajectory(mask_list, "mask term", ymax=None, ymin=None, title="Realized guidance @ diffusion time"))
    plots
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Realized guidance analysis
    """)
    return


@app.cell
def _(realized_guidance_plot):
    realized_guidance_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variable change analysis
    """)
    return


@app.cell
def _():
    # analysis of change wrt to the unguided states
    # compute the N differences
    return


if __name__ == "__main__":
    app.run()
