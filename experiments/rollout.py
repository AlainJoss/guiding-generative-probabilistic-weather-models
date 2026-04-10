import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ensemble rollout
    - Define N
    - Define M: number of members in ensemble
    - Define TIMESTAMP: starting datetime
    - Retrieve x_start from TIMESTAMP
    - -> Rollout ensemble and collect trajectories
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.cell
def _():
    from src.utils import (
        get_dataset, get_model, ensure_rollout_dir, save_to_json, state_to_device
    )
    from src.rollout import rollout

    return (
        ensure_rollout_dir,
        get_dataset,
        get_model,
        rollout,
        save_to_json,
        state_to_device,
    )


@app.cell
def _():
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    return LEVELS_DICT, PARTITIONS, VARIABLES_DICT


@app.cell
def _():
    device = "mps"
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
    ROLLOUT_TYPES = ["gt-weather", "model-ensemble"]
    rollout_type_dropdown = mo.ui.dropdown(ROLLOUT_TYPES, value=ROLLOUT_TYPES[0], label="rollout type: ")
    return ROLLOUT_TYPES, rollout_type_dropdown


@app.cell
def _(ROLLOUT_TYPES, mo, rollout_type):
    max_M = 1 if rollout_type == ROLLOUT_TYPES[0] else 20
    M_slider = mo.ui.slider(1, max_M, value=1, label="M: ")
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
def _(ds, level_idx, partition, timestamp_idx, var_idx):
    x_start = ds[timestamp_idx]
    slice = ds.denormalize(x_start["state"])[partition][var_idx, level_idx]
    return slice, x_start


@app.cell
def _():
    from src.interaction import (
        visualize_map, get_mask_corners_from_widget, 
        get_mask_from_corners, plot_trajectory, plot_dual_trajectory
    )

    return (visualize_map,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(
    M_slider,
    N_slider,
    level_slider,
    map_widget,
    mo,
    partition_dropdown,
    rollout_type_dropdown,
    timestamp_dropdown,
    var_dropdown,
):
    mo.vstack([
        rollout_type_dropdown,
        timestamp_dropdown,
        N_slider,
        M_slider, 
        mo.hstack([
            partition_dropdown,
            var_dropdown,
            level_slider,
        ], justify="start"),
        map_widget
    ])
    return


@app.cell
def _(M_slider, N_slider, mo, rollout_type_dropdown, timestamp_dropdown):
    mo.vstack([
        rollout_type_dropdown,
        timestamp_dropdown,
        N_slider,
        M_slider 
    ])
    return


@app.cell
def _(rollout_type_dropdown):
    rollout_type = rollout_type_dropdown.value
    return (rollout_type,)


@app.cell
def _(M_slider, N_slider, TIMESTAMPS, timestamp_dropdown):
    timestamp = timestamp_dropdown.value
    M = M_slider.value
    N = N_slider.value
    timestamp_idx = TIMESTAMPS.index(timestamp)
    return M, N, timestamp, timestamp_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run
    """)
    return


@app.cell
def _(Path, ds, model, rollout, timestamp_idx):
    def ensemble_rollout(rollout_dir: Path, M: int, N: int, x_start):
        for m in range(M):
            print(f"m: {m+1}/{M}")
            rollout(
                sampling_flag=True,
                guidance_flag=False,
                ensemble_flag=True,
                rollout_dir=rollout_dir,
                ds=ds, 
                x_start=x_start,
                gen_model=model,
                mask_corners=None, # mask_corners
                y=None, # y
                lambda_=None, # lambda_
                N=N,
                partition=None, # partition
                level_idx=None, # level_idx
                var_idx=None, # var_idx
                timestamp_idx=None,
                ensemble_step=m+1
            )

    def weather_rollout(rollout_dir: Path, M: int, N: int, x_start):
        rollout(
            sampling_flag=False,
            guidance_flag=False,
            ensemble_flag=False,
            rollout_dir=rollout_dir,
            ds=ds, 
            x_start=x_start,
            gen_model=model,
            mask_corners=None, # mask_corners
            y=None, # y
            lambda_=None, # lambda_
            N=N,
            partition=None, # partition
            level_idx=None, # level_idx
            var_idx=None, # var_idx
            timestamp_idx=timestamp_idx,
            ensemble_step=None
        )

    return ensemble_rollout, weather_rollout


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
    run_button = mo.ui.run_button(label="Run experiment")
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
    Path,
    ROLLOUT_TYPES,
    device,
    ensemble_rollout,
    ensure_rollout_dir,
    rollout_type,
    run_button,
    save_to_json,
    set_status,
    state_to_device,
    status,
    timestamp,
    weather_rollout,
    x_start,
):
    if run_button.value and status == "RUNNING":
        try:
            rollout_dir = ensure_rollout_dir(Path(rollout_type), N)
    
            if rollout_type == ROLLOUT_TYPES[0]:
                weather_rollout(rollout_dir, M, N, state_to_device(x_start, device))
            else:
                ensemble_rollout(rollout_dir, M, N, state_to_device(x_start, device))
    
            config = {
                "M": M,
                "N": N,
                "timestamp": str(timestamp),
            }  # add the other stuff when you extract it 
    
            save_to_json(config, rollout_dir, "config")
            set_status("IDLE")

        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            raise

        finally:
            set_status("IDLE")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
