import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path 
    from datetime import datetime, timedelta

    import marimo as mo
    import numpy as np
    import geopandas as gpd
    import geodatasets
    import torch

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import colors

    import cartopy.feature as cfeature
    from cartopy.crs import PlateCarree

    return Path, mo


@app.cell
def _():
    from src.interaction import (
        visualize_map, get_mask_corners_from_widget, 
        get_mask_from_corners, plot_trajectory, plot_dual_trajectory
    )
    from src.funcs import avg_over_mask, get_guidance_trajectory, N_schedule, T_schedule, compute_mean_rollout
    from src.rollout import rollout
    from src.utils import (
        ensure_rollout_dir,
        get_dataset, get_model, state_to_device,
        read_state, get_slice, save_to_json, read_json,
        read_states, xr_to_torch, list_tens_to_floats
    )
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.visualization import visualize_mask_terms_over_N

    return get_dataset, read_json


@app.cell
def _(get_dataset):
    ds = get_dataset()
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="refresh")
    return (refresh_button,)


@app.cell
def _(Path, mo, refresh_button):
    if refresh_button.value:
        pass

    def has_config_json(path: Path) -> bool:
        return (path / "config.json").exists()


    unguided_rollouts = Path("rollouts", "unguided").glob("2026*")
    unguided_rollouts = sorted(
        [p for p in unguided_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_unguided_rollout_dropdown = mo.ui.dropdown(label="Pick unguided rollout", value=unguided_rollouts[0], options=unguided_rollouts)

    old_model_rollouts = Path("rollouts", "old_model").glob("2026*")
    old_model_rollouts = sorted(
        [p for p in old_model_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_old_model_rollouts_rollout_dropdown = mo.ui.dropdown(label="Pick old model rollout", value=old_model_rollouts[0], options=old_model_rollouts)

    mo.vstack([
        pick_unguided_rollout_dropdown,
        pick_old_model_rollouts_rollout_dropdown
    ])
    return (
        pick_old_model_rollouts_rollout_dropdown,
        pick_unguided_rollout_dropdown,
    )


@app.cell
def _(
    pick_old_model_rollouts_rollout_dropdown,
    pick_unguided_rollout_dropdown,
    read_json,
):
    unguided_rollout_dir = pick_unguided_rollout_dropdown.value
    unguided_cfg = read_json(unguided_rollout_dir, "config")
    old_model_rollout_dir = pick_old_model_rollouts_rollout_dropdown.value
    old_model_cfg = read_json(old_model_rollout_dir, "config")
    return old_model_cfg, unguided_cfg


@app.cell
def _(old_model_cfg, unguided_cfg):
    unguided_cfg, old_model_cfg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
