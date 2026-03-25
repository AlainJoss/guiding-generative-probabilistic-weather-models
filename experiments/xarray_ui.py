import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import xarray as xr
    import numpy as np

    from src.utils import (
        read_config,
        read_state,
        get_last_experiment_dir, 
        get_mask_from_corners
    )
    from src.interaction import visualize_map

    return (
        get_last_experiment_dir,
        get_mask_from_corners,
        mo,
        np,
        read_config,
        read_state,
        visualize_map,
    )


@app.cell
def _(get_last_experiment_dir, get_mask_from_corners, read_config, read_state):
    result_dir = get_last_experiment_dir() 
    state = read_state(result_dir)
    cfg = read_config(result_dir)
    mask = get_mask_from_corners(*cfg["mask_corners"])
    cfg
    return cfg, mask, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Snapshot
    """)
    return


@app.cell
def _(state):
    state
    return


@app.cell
def _():
    # TODO: save mask in state and guidance partition + level + var
    # use the mask to visualize the rectangle of guidance (use red for gvar and blue for other ones)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interact
    """)
    return


@app.cell
def _():
    # I also want the ground truth to come alongside and the deterministic prediction and maybe a non-guided sampling
    return


@app.cell
def _():


    return


@app.cell
def _():
    # todo: shit I have to unroll both data and mask ...
    return


@app.cell
def _(cfg, state):
    slice = state[cfg["var"]].sel(time=cfg["timestamp"])
    return (slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(mask, np, slice, visualize_map):
    static_map = visualize_map(
        slice,
        mask_2d=np.asarray(mask),
        title="Experiment with saved mask",
        undo_roll=False,
        vmin=slice.min(),
        vmax=slice.max(),
        center= slice.mean()
    )
    return (static_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display
    """)
    return


@app.cell
def _(mask, static_map):
    print(mask.shape, mask.sum())
    static_map
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
