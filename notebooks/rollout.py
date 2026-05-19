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
    from src.utils.setup import get_now_timestamp
    from src.utils.read_write import (
        get_td_dataset,
        get_model,
        save_to_json,
        get_arches_era5
    )
    from src.ui.map import (
        visualize_map, get_mask_corners_from_widget, 
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory
    from src.funcs import N_schedule, T_schedule
    from src.utils.read_write import (
        get_td_dataset, get_model,
        save_to_json, read_json, get_rollout_ids, get_rollout_xr,
        get_rollout_config
    )
    from src.utils.dataset_utils import get_x_cond_from_ts, get_N_timestamps
    from src.utils.converters import list_tensors_to_floats, get_var_idx, get_level_idx
    from src.config import GUIDANCE_REFERENCES
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.config import MASK_MODES
    from src.rollout import rollout
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory
    from src.utils.dataset_utils import get_timestamps
    from src.utils.dataset_utils import get_N_slices
    from src.mask import get_masked_slices, get_masked_mean

    return (
        LEVELS_DICT,
        MASK_MODES,
        PARTITIONS,
        RUN_CONFIGS,
        VARIABLES_DICT,
        get_N_slices,
        get_N_timestamps,
        get_arches_era5,
        get_masked_mean,
        get_masked_slices,
        get_now_timestamp,
        get_timestamps,
        plot_dual_trajectory,
        save_to_json,
        visualize_map,
    )


@app.cell
def _():
    from src.mask import get_mask_2d, get_normal_mask, get_mask_from_corners

    return (get_mask_2d,)


@app.cell
def _(get_arches_era5, get_timestamps):
    ds = get_arches_era5()
    era5_timestamps = get_timestamps(ds)
    return (ds,)


@app.cell
def _(mo):
    clip = lambda x, a, b: min(max(x, a), b)
    get_month, set_month = mo.state(1)
    get_day, set_day = mo.state(1)

    month_slider = mo.ui.slider(
        start=1, 
        stop=12, 
        value=1, 
        step=1, 
        label="month: ", 
        show_value=True, 
        debounce=True, 
        on_change=set_month
    )

    day_slider = ...
    return


@app.cell
def _(date, mo):
    datetime_dropdown = mo.ui.date(start=date(2020, 1, 1), stop=date(2020, 12, 31))
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
    N_slider = mo.ui.slider(1, 30, value=1, label="N: ", show_value=True, debounce=True)
    return (N_slider,)


@app.cell
def _(mo):
    M_slider = mo.ui.slider(1, 20, value=1, label="M: ", show_value=True, debounce=True)
    return (M_slider,)


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
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level  ", show_value=True, debounce=True)
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
def _(LEVELS, VARIABLES, level, var):
    var_idx = VARIABLES.index(var)
    level_idx = LEVELS.index(level)
    return


@app.cell
def _(M_slider, N_slider):
    M = M_slider.value
    N = N_slider.value
    return M, N


@app.cell
def _(MASK_MODES, mo):
    mask_mode = mo.ui.dropdown(options=MASK_MODES, value=MASK_MODES[0], label="mask_mode: ")
    return (mask_mode,)


@app.cell
def _(N, mo):
    n_slider = mo.ui.slider(
        start=0, 
        stop=N,
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
        rollout_timestamps, var, m=0, ground_truth=ground_truth, right_axis=False, figsize=(12.4, 4), dpi=200,
        subtitle="mask weighted average", n=n_slider.value
    )
    return (rollout_dist_plot,)


@app.cell
def _(
    M_slider,
    N_slider,
    datetime_dropdown,
    hour_slider,
    level_slider,
    map_widget,
    mask_map,
    mask_mode,
    mo,
    n_slider,
    partition_dropdown,
    rollout_dist_plot,
    var_dropdown,
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
                    M_slider,
                    N_slider,
                    n_slider,
                ]
            ),
        ]),
    rollout_dist_plot,
    mask_mode,
    mo.hstack(
        [
            partition_dropdown,
            var_dropdown,
            level_slider,
        ],
        justify="start",
    ),
    mo.hstack([map_widget, mask_map], justify="start")
    ])
    return


@app.cell
def _(N, get_N_timestamps, timestamp):
    rollout_timestamps = get_N_timestamps(timestamp, N+1)
    return (rollout_timestamps,)


@app.cell
def _(N, ds, get_N_slices, level, partition, timestamp, var):
    N_slices = get_N_slices(ds, N+1, timestamp, partition, var, level)
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
def _(lat_bottom, lat_top, lon_left, lon_right):
    def get_mu_sigma(lon_left, lon_right, lat_bottom, lat_top):
        H, W = 121, 240

        mu_lon = (lon_left + lon_right) / 2
        mu_lat = (lat_bottom + lat_top) / 2

        sigma_lon = lon_right - lon_left
        sigma_lat = lat_top - lat_bottom

        mu_row = (90.0 - mu_lat) / 180.0 * H
        mu_col = (mu_lon + 180.0) / 360.0 * W

        sigma_row = sigma_lat / 180.0 * H
        sigma_col = sigma_lon / 360.0 * W

        mu = (mu_row, mu_col)
        sigma = (sigma_row, sigma_col)
        return mu, sigma

    mu, sigma = get_mu_sigma(lon_left, lon_right, lat_bottom, lat_top)
    return mu, sigma


@app.cell
def _(
    get_mask_2d,
    lat_bottom,
    lat_top,
    lon_left,
    lon_right,
    mask_mode,
    mu,
    sigma,
):
    mask_dict = {}

    match mask_mode.value:
        case "bbox":
            mask_params = [lon_left, lon_right, lat_bottom, lat_top]
        case "normal":
            mask_params = [mu, sigma]
        case _:
            pass

    mask_dict["mode"] = mask_mode.value
    mask_dict["params"] = mask_params

    mask = get_mask_2d(mask_dict)
    return mask, mask_dict


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
def _(M, N, level, mask_dict, partition, timestamp, var):
    config = {
        "guidance_flag": False,
        "M": M,
        "N": N,
        "timestamp": str(timestamp),  # is datetime
        "level": level,
        "partition": partition,
        "var": var, 
        "mask_dict": mask_dict
    }
    return (config,)


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(RUN_CONFIGS, config, config_button, get_now_timestamp, save_to_json):
    if config_button.value:
        config_id = get_now_timestamp()
        config["rollout_id"]=config_id
        config_dir = RUN_CONFIGS / "unguided"
        save_to_json(config, config_dir, f"{config_id}")
    return


if __name__ == "__main__":
    app.run()
