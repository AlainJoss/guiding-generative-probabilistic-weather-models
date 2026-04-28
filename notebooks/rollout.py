import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PHASE 1 - unguided rollout

    The aim of this first phase is to rollout an ensemble of M unguided models,
    producing a trajectory over N model steps.

    Proceed as follows:
    - Define parameters:
        - N: number of rollout steps (6h freq)
        - M: number of non-guided ensemble-members
        - timestamp: start datetime of the experiment
        - mask: region and variable of interest
    - Wait for the experiment to end (~3min for each sampling procedure).
    - Start the guide.py notebook and define the guidance experiment there.
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
    from src.paths import ROLLOUTS, CONFIGS

    return (CONFIGS,)


@app.cell
def _():
    from src.utils import (
        get_dataset, get_model, ensure_rollout_dir, save_to_json, state_to_device, get_device, get_slice,
        tensor_timestamp_to_string, get_now_timestamp
    )

    return (
        ensure_rollout_dir,
        get_dataset,
        get_device,
        get_model,
        get_now_timestamp,
        save_to_json,
        state_to_device,
    )


@app.cell
def _():
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    return LEVELS_DICT, PARTITIONS, VARIABLES_DICT


@app.cell
def _():
    from src.rollout import rollout

    return (rollout,)


@app.cell
def _():
    from src.funcs import avg_over_mask

    return (avg_over_mask,)


@app.cell
def _():
    from src.visualization import visualize_mask_terms_over_N

    return (visualize_mask_terms_over_N,)


@app.cell
def _():
    from src.interaction import (
        visualize_map, get_mask_corners_from_widget, 
        get_mask_from_corners, plot_trajectory, plot_dual_trajectory
    )

    return get_mask_corners_from_widget, get_mask_from_corners, visualize_map


@app.cell
def _(get_device):
    device = get_device()
    return (device,)


@app.cell
def _(device, get_dataset, get_model):
    ds = get_dataset()
    model = get_model(device)
    return ds, model


@app.cell
def _(ds):
    STRIDE = int(ds.lead_time_hours) // int(ds.timedelta)
    return (STRIDE,)


@app.cell
def _(STRIDE, ds, mo):
    # remove first and last (we have two tensordicts less due to prev/next)
    TIMESTAMPS = [str(ts[2]).split(".")[0] for ts in ds.timestamps][STRIDE:-STRIDE]
    month_slider = mo.ui.slider(
        start=1, stop=12, step=1, value=1, label="month: ", show_value=True, debounce=True
    )
    hour_slider = mo.ui.slider(
        start=0, stop=18, step=6, value=0, label="hour: ", show_value=True, debounce=True
    )
    get_day, set_day = mo.state(2)
    return TIMESTAMPS, get_day, hour_slider, month_slider, set_day


@app.cell(hide_code=True)
def _(get_day, mo, month_slider, set_day):
    import calendar

    _max_day = calendar.monthrange(2020, month_slider.value)[1]
    _cur = min(get_day(), _max_day)
    day_slider = mo.ui.slider(
        start=1,
        stop=_max_day,
        step=1,
        value=_cur,
        label="day: ",
        show_value=True,
        on_change=set_day,
        debounce=True
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
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level: ", show_value=True, debounce=True)
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
    level_idx = LEVELS.index(level) - 1
    return level_idx, var_idx


@app.cell
def _(
    M_slider,
    N_slider,
    STRIDE,
    TIMESTAMPS,
    day_slider,
    hour_slider,
    month_slider,
):
    # build timestamp from sliders; fall back to closest preceding TIMESTAMP if invalid (e.g. day=31 in Feb)
    _target = f"2020-{month_slider.value:02d}-{day_slider.value:02d}T{hour_slider.value:02d}:00:00"
    if _target in TIMESTAMPS:
        timestamp = _target
    else:
        _candidates = [t for t in TIMESTAMPS if t <= _target]
        timestamp = _candidates[-1] if _candidates else TIMESTAMPS[0]
    M = M_slider.value
    N = N_slider.value
    timestamp_idx = TIMESTAMPS.index(timestamp)
    timestamps = TIMESTAMPS[
        timestamp_idx : timestamp_idx + STRIDE * N + 1 : STRIDE
    ]
    return M, N, timestamp, timestamp_idx, timestamps


@app.cell
def _(ds, level_idx, partition, timestamp_idx, var_idx):
    x_start = ds[timestamp_idx]
    slice = ds.denormalize(x_start["state"])[partition][var_idx, level_idx]
    return slice, x_start


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
def _(get_mask_corners_from_widget, get_mask_from_corners, map_widget):
    mask_corners = get_mask_corners_from_widget(map_widget)
    mask = get_mask_from_corners(*mask_corners)
    return mask, mask_corners


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
def _(ground_truth, timestamps, var, visualize_mask_terms_over_N):
    rollout_dist_plot = visualize_mask_terms_over_N(var, timestamps, ground_truth=ground_truth)
    return (rollout_dist_plot,)


@app.cell
def _(mo):
    test_flag_checkbox = mo.ui.checkbox(value=False, label="test")
    return (test_flag_checkbox,)


@app.cell
def _(test_flag_checkbox):
    TEST=test_flag_checkbox.value
    return (TEST,)


@app.cell
def _():
    # tensor_timestamp_to_string(x_start["timestamp"])
    return


@app.cell
def _(
    M_slider,
    N_slider,
    day_slider,
    hour_slider,
    level_slider,
    map_widget,
    mo,
    month_slider,
    partition_dropdown,
    rollout_dist_plot,
    test_flag_checkbox,
    timestamp,
    var_dropdown,
):
    mo.vstack(
        [
            test_flag_checkbox,
            mo.hstack(
                [month_slider, day_slider, hour_slider, mo.md(f"→ {timestamp}")],
                justify="start",
            ),
            mo.hstack(
                [N_slider, mo.md("→ 24h model steps")], justify="start"
            ),
            mo.hstack([M_slider, mo.md("→ ensemble members")], justify="start"),
            mo.hstack(
                [
                    partition_dropdown,
                    var_dropdown,
                    level_slider,
                ],
                justify="start",
            ),
            rollout_dist_plot,
            map_widget,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run experiment
    Either rollout or save config for later run.
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Rollout")
    run_button
    return (run_button,)


@app.cell
def _(
    M,
    N,
    TEST,
    device,
    ds,
    ensure_rollout_dir,
    level,
    mask_corners,
    model,
    partition,
    rollout,
    run_button,
    save_to_json,
    state_to_device,
    timestamp,
    var,
    x_start,
):
    def run_rollout():
        rollout_dir = ensure_rollout_dir("unguided", N)
        for m in range(1, M + 1):
            print(f"m: {m}/{M}")
            rollout(
                guidance_flag=False,
                rollout_dir=rollout_dir,
                ds=ds,
                x_start=state_to_device(x_start, device),
                gen_model=model,
                init_mask_term=None,
                mask_corners=None,  # mask_corners
                y=None,  # y
                lambda_=None,  # lambda_
                N=N,
                partition=None,  # partition
                level_idx=None,  # level_idx
                var_idx=None,  # var_idx
                m=m,
                seed=None,
                test=TEST,
            )

        rollout_config = {
            "rollout_dir": str(rollout_dir),
            "M": M,
            "N": N,
            "timestamp": str(timestamp),
            "level": level,
            "partition": partition,
            "var": var,
            "mask_corners": mask_corners,
        }

        save_to_json(rollout_config, rollout_dir, "config")


    if run_button.value:
        run_rollout()
    return


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(
    CONFIGS,
    M,
    N,
    config_button,
    get_now_timestamp,
    level,
    level_idx,
    mask_corners,
    partition,
    save_to_json,
    timestamp,
    timestamp_idx,
    var,
    var_idx,
):
    if config_button.value:
        config_id = get_now_timestamp()
        config_dir = CONFIGS / "unguided"
        experiment_config = {
            "guidance_flag": False,
            "M": M,
            "N": N,
            "timestamp": str(timestamp),
            "timestamp_idx": timestamp_idx,
            "level": level,
            "level_idx": level_idx,
            "partition": partition,
            "var": var,
            "var_idx": var_idx,
            "mask_corners": mask_corners,
            "init_mask_term": None,
            "y": None,
            "lambda_": None
        }

        save_to_json(experiment_config, config_dir, f"{config_id}")
    return


if __name__ == "__main__":
    app.run()
