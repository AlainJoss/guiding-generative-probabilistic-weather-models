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

    from src.utils import get_last_experiment_id, get_mask_from_widget
    from src.interaction import visualize_map

    return get_last_experiment_id, mo, np, visualize_map, xr


@app.cell
def _(get_last_experiment_id, xr):
    path = get_last_experiment_id() / "state.nc"
    state = xr.open_dataset(path, engine="netcdf4")
    return (state,)


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
    timestamp = "2020-01-02T00:00:00" # this is actually annoying since there is only one
    partition = "surface"  # str
    var = "2m_temperature"
    var_idx = 2  # int
    level_idx = 0  # int  # NOTE: some variables do not have level coordinate
    return timestamp, var


@app.cell
def _():
    # todo: shit I have to unroll both data and mask ...
    return


@app.cell
def _():
    # todo: enable variable selection from the interface
    return


@app.cell
def _(state, timestamp, var):
    slice = state[var].sel(time=timestamp)
    return (slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(np):
    mask = np.zeros((121, 240), dtype=np.float32)
    mask[100, 100] = 1
    return (mask,)


@app.cell
def _(mask, slice, visualize_map):
    static_map = visualize_map(
        slice,
        mask_2d=mask,
        title="Experiment with saved mask",
        undo_roll=False,
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
