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

    return (mo,)


@app.cell
def _():
    from src.utils import (
        get_dataset, get_model,
        ensure_ensemble_rollouts_dir, save_config, state_to_device
    )
    from src.rollout import rollout

    return (
        ensure_ensemble_rollouts_dir,
        get_dataset,
        get_model,
        save_config,
        state_to_device,
    )


@app.cell
def _():
    device = "mps"
    return (device,)


@app.cell
def _(device, get_dataset, get_model):
    ds = get_dataset()
    model = get_model(device)
    return (ds,)


@app.cell
def _(ds, mo):
    # remove first and last, since we have two tensordicts less (because of prev and next)
    TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
    timestamp_dropdown = mo.ui.dropdown(TIMESTAMPS, value=TIMESTAMPS[0], label="start state : ")
    return TIMESTAMPS, timestamp_dropdown


@app.cell
def _(mo):
    M_slider = mo.ui.slider(1, 10, value=5, label="M: ")
    N_slider = mo.ui.slider(1, 20, value=2, label="N: ")
    return M_slider, N_slider


@app.cell
def _(M_slider, N_slider):
    M = M_slider.value
    N = N_slider.value
    return M, N


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(M_slider, N_slider, mo, timestamp_dropdown):
    mo.vstack([
        timestamp_dropdown,
        N_slider,
        M_slider
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run
    """)
    return


@app.cell
def _(TIMESTAMPS, device, ds, state_to_device, timestamp_dropdown):
    timestamp = timestamp_dropdown.value
    timestamp_idx = TIMESTAMPS.index(timestamp)
    x_start = ds[timestamp_idx]
    x_start = state_to_device(x_start, device)
    return timestamp, timestamp_idx, x_start


app._unparsable_cell(
    r"""
    def ensemble_rollout(result_dir: Path, M: int, N: int, x_start):
        for m in range(M):
            print(f"m: {m+1}/{M}")
            rollout(
                guidance_flag=False,
                ensemble_flag=True,
                result_dir,
                ds, 
                x_start,
                model,
                None, # mask_corners
                None, # y
                None, # lambda_
                N,
                None, # partition
                None, # level_idx
                None, # var_idx
                ensemble_step=m
            )
    """,
    name="_"
)


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
    ensemble_rollout,
    ensure_ensemble_rollouts_dir,
    run_button,
    save_config,
    set_status,
    status,
    timestamp,
    timestamp_idx,
    x_start,
):
    if run_button.value and status == "RUNNING":
        try:
            result_dir = ensure_ensemble_rollouts_dir(N)
            ensemble_rollout(result_dir, M, N, x_start)

            config = {
                "M": M,
                "N": N,
                "timestamp": str(timestamp),
                "timestamp_idx": int(timestamp_idx),
            }

            save_config(result_dir, config)
            set_status("IDLE")
        except:
            set_status("IDLE")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
