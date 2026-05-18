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
    from datetime import datetime

    return (mo,)


@app.cell
def _():
    from src.paths import ROLLOUTS, RUN_CONFIGS

    return


@app.cell
def _():
    from src.utils.setup import get_now_timestamp, get_device
    from src.utils.read_write import (
        get_dataset,
        get_model,
        save_to_json,
    )
    from src.ui.interaction import (
        visualize_map, 
        get_mask_from_corners,
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.ui.plot_dual_trajectory import plot_dual_trajectory
    from src.funcs import avg_over_mask, get_guidance_trajectory, N_schedule, T_schedule, compute_mean_rollout
    from src.utils.read_write import (
        get_dataset, get_model,
        save_to_json, read_json, get_rollout_ids, get_rollout_xr,
        get_rollout_config
    )
    from src.utils.dataset_utils import get_x_cond_from_ts
    from src.utils.setup import get_device
    from src.utils.converters import list_tensors_to_floats, get_var_idx, get_level_idx
    from src.config import GUIDANCE_REFERENCES
    from src.dimensions import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.config import MASK_MODES
    from src.rollout import rollout
    from src.funcs import avg_over_mask
    from src.ui.interaction import (
        visualize_map, get_mask_corners_from_widget, 
        get_mask_from_corners, plot_trajectory
    )
    from src.ui.plot_dual_trajectory import plot_dual_trajectory

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        avg_over_mask,
        get_dataset,
        get_device,
        get_now_timestamp,
        get_x_cond_from_ts,
        plot_dual_trajectory,
        save_to_json,
        visualize_map,
    )


@app.cell
def _(get_device):
    device = get_device()
    return


@app.cell
def _(get_dataset):
    ds = get_dataset()
    return (ds,)


@app.cell
def _(ds):
    STRIDE = int(ds.lead_time_hours) // int(ds.timedelta)
    return (STRIDE,)


@app.cell
def _(STRIDE, ds, mo):
    # remove first and last (we have two tensordicts less due to prev/next)
    TIMESTAMPS = [str(ts[2]).split(".")[0] for ts in ds.timestamps][STRIDE:-STRIDE]
    hour_slider = mo.ui.slider(
        start=0,
        stop=18,
        step=6,
        value=0,
        label="hour: ",
        show_value=True,
        debounce=True,
    )
    get_day, set_day = mo.state(2)
    get_month, set_month = mo.state(1)
    return TIMESTAMPS, get_day, get_month, hour_slider, set_day, set_month


@app.cell
def _(N_slider, STRIDE, TIMESTAMPS, get_month, mo, set_month):
    _max_valid_idx = len(TIMESTAMPS) - 1 - STRIDE * N_slider.value
    _valid_months = sorted(
        {int(t[5:7]) for i, t in enumerate(TIMESTAMPS) if i <= _max_valid_idx}
    )
    _max_month = _valid_months[-1] if _valid_months else 1
    _cur_m = min(max(get_month(), 1), _max_month)
    month_slider = mo.ui.slider(
        start=1,
        stop=_max_month,
        step=1,
        value=_cur_m,
        label="month: ",
        show_value=True,
        debounce=True,
        on_change=set_month,
    )
    return (month_slider,)


@app.cell
def _(N_slider, STRIDE, TIMESTAMPS, get_day, mo, month_slider, set_day):
    import calendar

    _max_valid_idx = len(TIMESTAMPS) - 1 - STRIDE * N_slider.value
    _month_days = sorted(
        {
            int(t[8:10])
            for i, t in enumerate(TIMESTAMPS)
            if int(t[5:7]) == month_slider.value and i <= _max_valid_idx
        }
    )
    if not _month_days:
        _month_days = sorted(
            {int(t[8:10]) for t in TIMESTAMPS if int(t[5:7]) == month_slider.value}
        )
    _min_day = _month_days[0]
    _max_day = _month_days[-1]
    _cur = min(max(get_day(), _min_day), _max_day)
    day_slider = mo.ui.slider(
        start=_min_day,
        stop=_max_day,
        step=1,
        value=_cur,
        label="day: ",
        show_value=True,
        on_change=set_day,
        debounce=True,
    )
    return (day_slider,)


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
    return level_idx, var_idx


@app.cell
def _(TIMESTAMPS, day_slider, hour_slider, month_slider):
    _target = f"2020-{month_slider.value:02d}-{day_slider.value:02d}T{hour_slider.value:02d}:00:00"
    if _target in TIMESTAMPS:
        timestamp = _target
    else:
        _candidates = [t for t in TIMESTAMPS if t <= _target]
        timestamp = _candidates[-1] if _candidates else TIMESTAMPS[0]
    return (timestamp,)


@app.cell
def _(M_slider, N_slider):
    M = M_slider.value
    N = N_slider.value
    return M, N


@app.cell
def _(N, STRIDE, TIMESTAMPS, timestamp_idx):
    timestamps = TIMESTAMPS[
        timestamp_idx : timestamp_idx + STRIDE * N + 1 : STRIDE
    ]
    return (timestamps,)


@app.cell
def _(ds, get_x_cond_from_ts, level_idx, partition, timestamp, var_idx):
    x_cond = get_x_cond_from_ts(ds, timestamp)
    slice = ds.denormalize(x_cond["state"])[partition][var_idx, level_idx]
    return (slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - need new mask mode and save params and mode to config
    - use get mask using mask mode
    - need to change interaction plot
    - finally separate plots in different files
    - need to read xarray and disable first timestamps
    - need to visualize historical patterns so I can find a good input state for a real extreme trajectory.
    """)
    return


@app.cell
def _():
    # TODO: still need this with also the other modes
    # get_corners, set_corners = mo.state((-10.0, 2.0, 45.0, 35.0))
    return


@app.cell
def _(get_corners, set_corners, slice, visualize_map):
    # enables to save state of widget
    lx0, lx1, ly0, ly1 = get_corners()
    map_widget = visualize_map(
        slice,
        title="Select mask region",
        interactive=True,
        vmin=slice.min(),
        vmax=slice.max(),
        center=slice.mean(),
        rectangle_x=(lx0, lx1),
        rectangle_y=(ly0, ly1),
        figsize=(12,5),
        dpi=200
    )
    map_widget.widget.observe(
        lambda _c: set_corners(
            (*map_widget.widget.x, *map_widget.widget.y)
        ),
        names=["x", "y"],
    )
    return (map_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure rollout
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Write a getter for the following thing, also use in guide.py --> get_slice_average(mask, slice)  # numpy not torch
    """)
    return


@app.cell
def _(
    N,
    STRIDE,
    avg_over_mask,
    ds,
    level_idx,
    mask,
    partition,
    timestamp_idx,
    var_idx,
):
    ground_truth = []
    for n in range(N + 1):
        state_n = ds[timestamp_idx + STRIDE * n]["state"]
        slice_n = ds.denormalize(state_n)[partition][var_idx, level_idx]
        avg = avg_over_mask(slice_n, mask)
        ground_truth.append(avg)
    return (ground_truth,)


@app.cell
def _(ground_truth, plot_dual_trajectory, timestamps, var):
    rollout_dist_plot = plot_dual_trajectory(
        timestamps, var, ground_truth=ground_truth, right_axis=False, figsize=(12.4, 4), dpi=200
    )
    return (rollout_dist_plot,)


@app.cell
def _(
    M_slider,
    N_slider,
    day_slider,
    hour_slider,
    level_slider,
    mo,
    month_slider,
    partition_dropdown,
    rollout_dist_plot,
    timestamp,
    var_dropdown,
):
    mo.vstack([
        mo.md("Define rollout parameters:"),
        mo.hstack([
            mo.vstack(
                [
                    mo.hstack(
                        [mo.md("Experiment start: "), month_slider, day_slider, hour_slider, mo.md(f"({timestamp})")],
                        justify="start",
                    ),
                    mo.hstack(
                        [N_slider, mo.md("(24h model steps)")], justify="start"
                    ),
                    mo.hstack([M_slider, mo.md("(ensemble members)")], justify="start"),
                    mo.hstack(
                        [
                            mo.md("Mask: "),
                            partition_dropdown,
                            var_dropdown,
                            level_slider,
                        ],
                        justify="start",
                    ),
            ]),
        ]),
    rollout_dist_plot,
    ])
    return


@app.cell
def _(map_widget, mo):
    mo.vstack(
        [mo.md("Define mask region:"), map_widget],justify="start",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save config
    """)
    return


@app.cell
def _(M, N, level, level_idx, partition, timestamp, var, var_idx):
    config = {
        "guidance_flag": False,
        "M": M,
        "N": N,
        "timestamp": str(timestamp),
        "level": level,
        "level_idx": level_idx,
        "partition": partition,
        "var": var, 
        "var_idx": var_idx,
    }
    return (config,)


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(CONFIGS, config, config_button, get_now_timestamp, save_to_json):
    if config_button.value:
        config_id = get_now_timestamp()
        config["rollout_id"]=config_id
        config_dir = CONFIGS / "unguided"
        save_to_json(config, config_dir, f"{config_id}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
