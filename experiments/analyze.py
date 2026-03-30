import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import torch
    # import xarray as xr
    import numpy as np

    from src.utils import (
        read_config,
        read_state,
        get_last_experiment_dir
    )
    from src.interaction import visualize_map, get_mask_from_corners, get_mask_center

    return (
        Path,
        get_mask_center,
        get_mask_from_corners,
        mo,
        np,
        read_config,
        read_state,
        visualize_map,
    )


@app.cell
def _(cfg):
    def get_slice(state, partition, level, var):
        if partition == "surface":
            return state[var].sel(time=cfg["timestamp"])
        else: 
            return state[var].sel(time=cfg["timestamp"], level=level)

    return (get_slice,)


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
def _(LEVELS_DICT, partition):
    LEVELS = LEVELS_DICT[partition][::-1]
    return (LEVELS,)


@app.cell
def _(LEVELS, mo):
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level: ")
    return (level_slider,)


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, partition):
    VARIABLES = VARIABLES_DICT[partition]
    return (VARIABLES,)


@app.cell
def _(VARIABLES, mo):
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLES[2], label="variable : ")
    return (var_dropdown,)


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(mo):
    zoom_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=1,
        label="Zoom",
    )
    return (zoom_slider,)


@app.cell
def _(zoom_slider):
    zoom = zoom_slider.value
    return (zoom,)


@app.cell
def _(mo):
    show_mask_switch = mo.ui.switch(label="show mask")
    return (show_mask_switch,)


@app.cell
def _(show_mask_switch):
    show_mask = show_mask_switch.value
    return (show_mask,)


@app.cell
def _(get_mask_center, mask):
    zoom_centers = get_mask_center(mask)
    return (zoom_centers,)


@app.cell
def _(analysis_type_dropdown):
    analysis_type = analysis_type_dropdown.value
    return (analysis_type,)


@app.cell
def _(mo):
    analysis_types = ["absolute", "difference"]
    analysis_type_dropdown = mo.ui.dropdown(analysis_types, value=analysis_types[0], label="analysis: ")
    return (analysis_type_dropdown,)


@app.cell
def _(mo):
    show_values_checkbox = mo.ui.checkbox(label="show_values")
    return (show_values_checkbox,)


@app.cell
def _(show_values_checkbox):
    show_values = show_values_checkbox.value
    return (show_values,)


@app.cell
def _(show_values):
    show_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ground truth
    """)
    return


@app.cell
def _():
    from geoarches.dataloaders.era5 import Era5Forecast

    ds = Era5Forecast(
        path="data/era5_240/full",  # default path
        domain="test", # domain to consider. domain = 'test' loads the 2020 period
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=6
    )
    return (ds,)


@app.cell
def _(cfg, ds):
    x_start = ds[cfg["timestamp_idx"]]
    x_start = ds.denormalize(x_start)
    current = ds.convert_to_xarray(x_start["state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))
    next = ds.convert_to_xarray(x_start["next_state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))
    return current, next


@app.cell
def _(current, get_slice, guided, level, next, partition, unguided, var):
    guided_slice = get_slice(guided, partition, level, var)
    unguided_slice = get_slice(unguided, partition, level, var)
    current_slice = get_slice(current, partition, level, var)
    next_slice = get_slice(next, partition, level, var)
    return current_slice, guided_slice, next_slice, unguided_slice


@app.cell
def _(current_slice, guided_slice, next_slice, unguided_slice):
    unguided_guided = unguided_slice - guided_slice
    next_guided = next_slice - guided_slice
    next_unguided = next_slice - unguided_slice
    next_current = next_slice - current_slice 
    return next_current, next_guided, next_unguided, unguided_guided


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data snapshot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment config
    """)
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="Refresh")
    refresh_button
    return (refresh_button,)


@app.cell
def _(Path, mo, refresh_button):
    if refresh_button.value:
        pass
    experiments = Path("data", "results").glob("2026*")
    experiments = sorted(experiments)[::-1]
    pick_dropdown = mo.ui.dropdown(label="Pick experiment", value=experiments[0], options=experiments)
    pick_dropdown
    return (pick_dropdown,)


@app.cell
def _(get_mask_from_corners, mo, pick_dropdown, read_config, read_state):
    result_dir = pick_dropdown.value
    guided = read_state(result_dir, "guided")
    unguided = read_state(result_dir, "unguided")
    cfg = read_config(result_dir)
    mask = get_mask_from_corners(*cfg["mask_corners"])
    mo.vstack([
        mo.md("<br>".join(f"{k}: {v}" for k, v in cfg.items()))
    ])
    return cfg, guided, mask, unguided


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Result
    """)
    return


@app.cell
def _(
    analysis_type,
    current_slice,
    guided_slice,
    mask,
    next_current,
    next_guided,
    next_slice,
    next_unguided,
    np,
    show_mask,
    show_values,
    unguided_guided,
    unguided_slice,
    visualize_map,
    zoom,
    zoom_centers,
    zoom_slider,
):
    if analysis_type == "absolute":
        state_map = visualize_map(
            current_slice,
            mask_2d=np.asarray(mask),
            title="$x_t$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center= current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    
        next_map = visualize_map(
            next_slice,
            mask_2d=np.asarray(mask),
            title="$x_{t+1}$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center= current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    
        guided_map = visualize_map(
            guided_slice,
            mask_2d=np.asarray(mask),
            title="$x_{t+1}^{guide}$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center=current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    
        unguided_map = visualize_map(
            unguided_slice,
            mask_2d=np.asarray(mask),
            title="$x_{t+1}^{gen}$",
            vmin=current_slice.min(),
            vmax=current_slice.max(),
            center=current_slice.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    else:
        next_current_map = visualize_map(
            next_current,
            mask_2d=np.asarray(mask),
            title="next-current",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center= next_current.mean(),
            show_mask=show_mask,
            zoom=zoom,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    
        unguided_guided_map = visualize_map(
            unguided_guided,
            mask_2d=np.asarray(mask),
            title="unguided-guided",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center= next_current.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
            show_values=show_values,
            value_threshold=0.1,
            value_fontsize=5,
        )
    
        next_guided_map = visualize_map(
            next_guided,
            mask_2d=np.asarray(mask),
            title="next-guided",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center=next_current.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    
        next_unguided_map = visualize_map(
            next_unguided,
            mask_2d=np.asarray(mask),
            title="next-unguided",
            vmin=next_current.min(),
            vmax=next_current.max(),
            center=next_current.mean(),
            show_mask=show_mask,
            zoom=zoom_slider.value,
            zoom_center_lon=zoom_centers[0],
            zoom_center_lat=zoom_centers[1],
        )
    return (
        guided_map,
        next_current_map,
        next_guided_map,
        next_map,
        next_unguided_map,
        state_map,
        unguided_guided_map,
        unguided_map,
    )


@app.cell
def _(analysis_type_dropdown):
    analysis_type_dropdown
    return


@app.cell
def _(
    analysis_type,
    guided_map,
    level,
    level_slider,
    mo,
    next_current_map,
    next_guided_map,
    next_map,
    next_unguided_map,
    partition_dropdown,
    show_mask_switch,
    show_values_checkbox,
    state_map,
    unguided_guided_map,
    unguided_map,
    var_dropdown,
    zoom_slider,
):
    if analysis_type == "absolute":
        to_show = mo.vstack([
            partition_dropdown,
            mo.hstack([level_slider, mo.md(f"{level}")], justify="start"),
            var_dropdown,
            show_mask_switch, 
            zoom_slider,
            mo.hstack([state_map, next_map]),
            mo.hstack([unguided_map, guided_map])
        ])
    else:
        to_show = mo.vstack([
            mo.hstack([partition_dropdown, var_dropdown], justify="start"),
            mo.hstack([level_slider, mo.md(f"{level}")], justify="start"),
            mo.hstack([show_mask_switch, show_values_checkbox], justify="start"),
            zoom_slider,
            mo.hstack([next_current_map, unguided_guided_map]),
            mo.hstack([next_unguided_map, next_guided_map])
        ])
    to_show
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
