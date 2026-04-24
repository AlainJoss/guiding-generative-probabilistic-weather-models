import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TODO
    - check start date is actually 00:60, seems odd
    - need to think about how to set up ablation study
    - which other plots
    - implement a logger for experiments and print error to log file instead of this
    - remember: we are currently using one model only for the deterministic prediction
    - should also number the deterministic predictions (otherwise they get overwritten)!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PHASE 1

    The aim of this first phase is to rollout an ensemble of M unguided models,
    producing a trajectory over N model steps.

    Proceed as follows:
    - Define parameters:
        - N: number of rollout steps (6h freq)
        - M: number of ensemble-members
        - timestamp: start datetime of the experiment
        - mask: region and variable of interest
    - Wait for the experiment to end (~3min for each sampling procedure).
    - Start the guide.py notebook and define the guidance experiment there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch
    import numpy as np

    return (mo,)


@app.cell
def _():
    from src.paths import ROLLOUTS

    return


@app.cell
def _():
    from src.utils import (
        get_dataset, get_model, ensure_rollout_dir, save_to_json, state_to_device, get_device, get_slice
    )

    return (
        ensure_rollout_dir,
        get_dataset,
        get_device,
        get_model,
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
def _(ds, mo):
    # remove first and last, since we have two tensordicts less (because of prev and next)
    TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
    timestamp_dropdown = mo.ui.dropdown(TIMESTAMPS, value=TIMESTAMPS[0], label="start state : ")
    return TIMESTAMPS, timestamp_dropdown


@app.cell
def _(mo):
    N_slider = mo.ui.slider(1, 20, value=1, label="N: ")
    return (N_slider,)


@app.cell
def _(mo):
    M_slider = mo.ui.slider(1, 20, value=1, label="M: ")
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
def _(LEVELS, VARIABLES, level, var):
    var_idx = VARIABLES.index(var)
    level_idx = LEVELS.index(level) - 1
    return level_idx, var_idx


@app.cell
def _(M_slider, N_slider, TIMESTAMPS, timestamp_dropdown):
    timestamp = timestamp_dropdown.value
    M = M_slider.value
    N = N_slider.value * 4
    timestamp_idx = TIMESTAMPS.index(timestamp)
    timestamps = TIMESTAMPS[timestamp_idx:timestamp_idx+N+1] 
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
    avg_over_mask,
    ds,
    level_idx,
    mask,
    partition,
    timestamp_idx,
    var_idx,
):
    ground_truth = []
    for n in range(N+1):
        state_n = ds[timestamp_idx+n]["state"]
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
def _(
    M_slider,
    N,
    N_slider,
    level_slider,
    map_widget,
    mo,
    partition_dropdown,
    rollout_dist_plot,
    test_flag_checkbox,
    timestamp_dropdown,
    var_dropdown,
):
    mo.vstack([
        test_flag_checkbox,
        timestamp_dropdown,
        mo.hstack([N_slider, mo.md(f"days ({N} steps)")], justify="start"),
        mo.hstack([M_slider, mo.md(f"ensemble members")], justify="start"),
        mo.hstack([
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        rollout_dist_plot,
        map_widget
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
def _(mo):
    run_button = mo.ui.run_button(label="Run")
    run_button
    return (run_button,)


@app.cell
def _(get_status, mo):
    status = get_status()
    mo.md(f"Experiment status: **{status}**")
    return (status,)


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
    status,
    timestamp,
    var,
    x_start,
):
    if run_button.value and status == "RUNNING":
        # try:
        rollout_dir = ensure_rollout_dir("unguided", N)
        for m in range(1, M+1):
            print(f"m: {m}/{M}")
            rollout(
                guidance_flag=False,
                rollout_dir=rollout_dir,
                ds=ds, 
                x_start=state_to_device(x_start, device),
                gen_model=model,
                init_mask_term=None,
                mask_corners=None, # mask_corners
                y=None, # y
                lambda_=None, # lambda_
                N=N,
                partition=None, # partition
                level_idx=None, # level_idx
                var_idx=None, # var_idx
                m=m,
                seed=None,
                test=TEST
            )

        config = {
            "rollout_dir": str(rollout_dir),
            "M": M,
            "N": N,
            "timestamp": str(timestamp),
            "level": level,
            "partition": partition,
            "var": var,
            "mask_corners": mask_corners
        } 

        save_to_json(config, rollout_dir, "config")
        #     set_status("IDLE")

        # except Exception as e:
        #     rollout_dir.unlink()
        #     print(f"Error: {type(e).__name__}: {e}")
        #     raise


        # finally:
        #     set_status("IDLE")
    return (rollout_dir,)


@app.cell
def _(mo):
    config_button = mo.ui.run_button(label="Save config")
    config_button
    return (config_button,)


@app.cell
def _(
    M,
    N,
    config_button,
    level,
    mask_corners,
    partition,
    rollout_dir,
    save_to_json,
    timestamp,
    var,
):
    if config_button.value:
        _config = {
            "rollout_dir": str(rollout_dir),
            "M": M,
            "N": N,
            "timestamp": str(timestamp),
            "level": level,
            "partition": partition,
            "var": var,
            "mask_corners": mask_corners
        } 

        save_to_json(_config, rollout_dir, "config")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
