import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PHASE 1 - unguided rollout
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch
    import numpy as np
    from datetime import datetime, timedelta, date, time
    import calendar

    return date, datetime, mo, np, time


@app.cell
def _():
    from src.paths import ROLLOUTS, RUN_CONFIGS
    from src.rollout_config import GUIDANCE_REFERENCES, MASK_MODES
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.ui.map import visualize_map
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory

    from src.utils.setup import get_now_timestamp
    from src.utils.read_write import (
        get_xr_dataset,
        save_to_json, get_dict_from_json, get_rollout_ids, get_rollout_files
    )
    from src.utils.converters import list_tensors_to_floats, get_var_idx, get_level_idx
    from src.utils.dataset_utils import get_timestamps, get_N_timestamps, get_N_slices

    from src.funcs import N_schedule, T_schedule

    from src.mask import get_masked_slices, get_masked_mean, get_mask_2d, get_normal_mask, get_mask_from_corners, get_mu_sigma

    return (
        LEVELS_DICT,
        MASK_MODES,
        PARTITIONS,
        RUN_CONFIGS,
        VARIABLES_DICT,
        get_N_slices,
        get_N_timestamps,
        get_level_idx,
        get_mask_2d,
        get_masked_mean,
        get_masked_slices,
        get_mu_sigma,
        get_now_timestamp,
        get_timestamps,
        get_var_idx,
        get_xr_dataset,
        plot_dual_trajectory,
        save_to_json,
        visualize_map,
    )


@app.cell
def _(get_timestamps, get_xr_dataset):
    ds = get_xr_dataset()
    era5_timestamps = get_timestamps(ds)
    return (ds,)


@app.cell
def _(N, date, mo):
    datetime_dropdown = mo.ui.date(start=date(2020, 1, 2), stop=date(2020, 12, 31-N.value))
    return (datetime_dropdown,)


@app.cell
def _(mo):
    hour_slider = mo.ui.slider(0, 18, value=0, step=6, label="hour: ", show_value=True, debounce=True)
    return (hour_slider,)


@app.cell
def _(datetime, datetime_dropdown, hour_slider, time):
    def get_timestamp_from_sliders(date, hour):
        return datetime.combine(date, time(hour=hour))

    timestamp = get_timestamp_from_sliders(datetime_dropdown.value, hour_slider.value)
    return (timestamp,)


@app.cell
def _(mo):
    # not more than 30 ow the datetime_dropdown will fail
    N = mo.ui.slider(1, 30, value=1, label="N: ", show_value=True, debounce=True)
    return (N,)


@app.cell
def _(mo):
    M = mo.ui.slider(1, 20, value=1, label="M: ", show_value=True, debounce=True)
    return (M,)


@app.cell
def _(PARTITIONS, mo):
    partition = mo.ui.dropdown(PARTITIONS, value=PARTITIONS[0], label="partition: ")
    return (partition,)


@app.cell
def _(LEVELS_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition.value]
    level = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level: ", show_value=True, debounce=True)
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition):
    VARIABLES = VARIABLES_DICT[partition.value]
    if partition.value == "surface":
        VARIABLES_VALUE = VARIABLES[2]
    else:
        VARIABLES_VALUE = VARIABLES[3]
    var = mo.ui.dropdown(VARIABLES, value=VARIABLES_VALUE, label="variable : ")
    return (var,)


@app.cell
def _(get_level_idx, get_var_idx, level, partition, var):
    var_idx = get_var_idx(partition.value, var.value)
    level_idx = get_level_idx(partition.value, level.value)
    return


@app.cell
def _(MASK_MODES, mo):
    mask_mode = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
    return (mask_mode,)


@app.cell
def _(N, mo):
    n_slider = mo.ui.slider(
        start=0, 
        stop=N.value,
        step=1,
        label="n: ",
        value=0,
        debounce=True,
        show_value=True
    )
    return (n_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - finally separate plots in different files
    - disable first timestamps ...
    - need to visualize historical patterns so I can find a good input state for a real extreme trajectory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure rollout
    """)
    return


@app.cell
def _(ground_truth, n_slider, plot_dual_trajectory, rollout_timestamps, var):
    rollout_dist_plot = plot_dual_trajectory(
        rollout_timestamps, var.value, m=0, ground_truth=ground_truth, right_axis=False, figsize=(12.4, 4), dpi=200,
        subtitle="mask weighted average", n=n_slider.value
    )
    return (rollout_dist_plot,)


@app.cell
def _(
    M,
    N,
    datetime_dropdown,
    hour_slider,
    level,
    map_widget,
    mask_map,
    mask_mode,
    mo,
    n_slider,
    partition,
    rollout_dist_plot,
    var,
):
    mo.vstack([
        mo.md("Rollout params:"),
        mo.hstack([
            mo.vstack(
                [
                    mo.hstack(
                        [mo.md("date: "), datetime_dropdown, hour_slider],
                        justify="start",
                    ),
                    M,
                    N,
                    n_slider,
                ]
            ),
        ]),
    rollout_dist_plot,
    mask_mode,
    mo.hstack(
        [
            partition,
            var,
            level,
        ],
        justify="start",
    ),
    mo.hstack([map_widget, mask_map], justify="start")
    ])
    return


@app.cell
def _(N, get_N_timestamps, timestamp):
    rollout_timestamps = get_N_timestamps(timestamp, N.value+1)
    return (rollout_timestamps,)


@app.cell
def _(N, ds, get_N_slices, level, partition, timestamp, var):
    N_slices = get_N_slices(ds, N.value+1, timestamp, partition.value, var.value, level.value)
    return (N_slices,)


@app.cell
def _(N_slices, n_slider):
    slice = N_slices[n_slider.value]
    return (slice,)


@app.cell
def _(N_slices, get_masked_mean, get_masked_slices, mask):
    N_masked_slices = get_masked_slices(N_slices, mask)
    ground_truth = get_masked_mean(N_masked_slices, mask)
    return N_masked_slices, ground_truth


@app.cell
def _(mo):
    get_corners, set_corners = mo.state((-10.0, 2.0, 35.0, 45.0))
    return get_corners, set_corners


@app.cell
def _(N_slices, get_corners, np, set_corners, slice, visualize_map):
    lon_left, lon_right, lat_bottom, lat_top = get_corners()
    map_widget = visualize_map(
        slice,
        title="Mask region",
        interactive=True,
        vmin=np.min(N_slices),
        vmax=np.max(N_slices),
        center=np.mean(N_slices),
        rectangle_x=(lon_left, lon_right),
        rectangle_y=(lat_bottom, lat_top),
        figsize=(12, 5),
        dpi=200,
    )
    map_widget.widget.observe(
        lambda _c: set_corners(
            (*sorted(map_widget.widget.x), *sorted(map_widget.widget.y))
        ),
        names=["x", "y"],
    )
    return lat_bottom, lat_top, lon_left, lon_right, map_widget


@app.cell
def _(
    get_mask_2d,
    get_mu_sigma,
    lat_bottom,
    lat_top,
    lon_left,
    lon_right,
    mask_mode,
):
    mask_params = None
    mu, sigma = get_mu_sigma(lon_left, lon_right, lat_bottom, lat_top)
    match mask_mode.value:
        case "bbox":
            mask_params = [lon_left, lon_right, lat_bottom, lat_top]
        case "normal":
            mask_params = [mu, sigma]
        case _:
            pass

    mask = get_mask_2d(mask_mode.value, mask_params)
    return mask, mask_params


@app.cell
def _(N_masked_slices, n_slider):
    masked_slice = N_masked_slices[n_slider.value]
    return


@app.cell
def _(lat_bottom, lat_top, lon_left, lon_right, mask, np, visualize_map):
    mask_map = visualize_map(
        mask,
        title="Mask",
        interactive=False,
        vmin=np.min(mask),
        vmax=np.max(mask),
        center=np.mean(mask),
        rectangle_x=(lon_left, lon_right),
        rectangle_y=(lat_bottom, lat_top),
        figsize=(12, 5),
        dpi=200,
    )
    return (mask_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save config
    """)
    return


@app.cell
def _():
    from src.rollout_config import RolloutConfig

    return (RolloutConfig,)


@app.cell
def _(
    M,
    N,
    RolloutConfig,
    level,
    mask_mode,
    mask_params,
    partition,
    timestamp,
    var,
):
    config = RolloutConfig(
        guidance_flag=False,
        M=M.value,
        N=N.value,
        timestamp=timestamp,  # datetime
        level=level.value,
        partition=partition.value,
        var=var.value,
        mask_mode=mask_mode.value,
        mask_params=mask_params,
    )
    return (config,)


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(RUN_CONFIGS, config, config_button, get_now_timestamp, save_to_json):
    if config_button.value:
        rollout_id = get_now_timestamp()
        config.rollout_id=rollout_id
        run_dir = RUN_CONFIGS / "unguided"
        save_to_json(config.to_dict(), run_dir, f"{rollout_id}")
    return


if __name__ == "__main__":
    app.run()
