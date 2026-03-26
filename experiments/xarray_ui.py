import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
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
        get_last_experiment_dir
    )
    from src.interaction import visualize_map, get_mask_from_corners

    return (
        get_last_experiment_dir,
        get_mask_from_corners,
        mo,
        np,
        read_config,
        read_state,
        visualize_map,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interact
    """)
    return


@app.cell
def _():
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    return LEVELS_DICT, PARTITIONS, VARIABLES_DICT


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
    level_dropdown = mo.ui.dropdown(LEVELS, value=LEVELS[0], label="level: ")
    return (level_dropdown,)


@app.cell
def _(level_dropdown):
    level = level_dropdown.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition):
    VARIABLES = VARIABLES_DICT[partition]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLES[2], label="variable : ")
    return (var_dropdown,)


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(cfg, level, partition, state, var):
    def get_slice(state, partition, level, var):
        if partition == "surface":
            return state[var].sel(time=cfg["timestamp"])
        else: 
            return state[var].sel(time=cfg["timestamp"], level=level)
        
    slice = get_slice(state, partition, level, var)
    return (slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw data
    """)
    return


@app.cell
def _(state):
    state
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment config
    """)
    return


@app.cell
def _(
    get_last_experiment_dir,
    get_mask_from_corners,
    mo,
    read_config,
    read_state,
):
    result_dir = get_last_experiment_dir() 
    state = read_state(result_dir)
    cfg = read_config(result_dir)
    mask = get_mask_from_corners(*cfg["mask_corners"])
    from pprint import pprint
    mo.vstack([
        mo.md("<br>".join(f"{k}: {v}" for k, v in cfg.items()))
    ])
    return cfg, mask, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Result
    """)
    return


@app.cell
def _(mo):
    show_mask_switch = mo.ui.switch(label="show mask")
    return (show_mask_switch,)


@app.cell
def _(show_mask_switch):
    show_mask = show_mask_switch.value
    return (show_mask,)


@app.cell
def _(mask, np, show_mask, slice, visualize_map):
    result_map = visualize_map(
        slice,
        mask_2d=np.asarray(mask),
        title="Experiment with saved mask",
        vmin=slice.min(),
        vmax=slice.max(),
        center= slice.mean(),
        show_mask=show_mask
    )

    gt_map = visualize_map(
        slice,
        mask_2d=np.asarray(mask),
        title="Experiment with saved mask",
        vmin=slice.min(),
        vmax=slice.max(),
        center= slice.mean(),
        show_mask=show_mask
    )
    return gt_map, result_map


@app.cell
def _(
    gt_map,
    level_dropdown,
    mo,
    partition_dropdown,
    result_map,
    show_mask_switch,
    var_dropdown,
):
    mo.vstack([
        partition_dropdown,
        level_dropdown,
        var_dropdown,
        show_mask_switch, 
        result_map,
        gt_map
    ])

    return


@app.cell
def _():
    return


@app.cell
def _():
    # TODO: stupid mask haha
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
