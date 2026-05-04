import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison - my vs. their's implementation
    """)
    return


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

    return Path, mo, np, torch


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

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        get_dataset,
        get_slice,
        read_json,
        read_state,
        visualize_map,
    )


@app.cell
def _(get_dataset):
    ds = get_dataset()
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="refresh")
    refresh_button
    return (refresh_button,)


@app.cell
def _(Path, mo, refresh_button):
    if refresh_button.value:
        pass

    def has_config_json(path: Path) -> bool:
        return (path / "config.json").exists()


    unguided_rollouts = Path("rollouts", "new_model").glob("2026*")
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
    return (
        old_model_cfg,
        old_model_rollout_dir,
        unguided_cfg,
        unguided_rollout_dir,
    )


@app.cell
def _(old_model_cfg, unguided_cfg):
    unguided_cfg, old_model_cfg
    return


@app.cell
def _(
    old_model_cfg,
    old_model_rollout_dir,
    read_state,
    torch,
    unguided_cfg,
    unguided_rollout_dir,
):
    import pandas as pd

    N_cmp = min(unguided_cfg["N"], old_model_cfg["N"])
    M_cmp = min(unguided_cfg["M"], old_model_cfg["M"])


    def _state_max_diff(state_a, state_b):
        max_diff = 0.0
        ok = True
        for v in state_a.data_vars:
            a = torch.as_tensor(state_a[v].values)
            b = torch.as_tensor(state_b[v].values)
            d = (a - b).abs().max().item()
            if d > max_diff:
                max_diff = d
            if not torch.allclose(a, b, rtol=1e-6, atol=1e-7):
                ok = False
        return max_diff, ok


    rows = []
    for n in range(1, N_cmp + 1):
        old_paths = sorted(old_model_rollout_dir.glob(f"{n}/*"))[:M_cmp]
        new_paths = sorted(unguided_rollout_dir.glob(f"{n}/*"))[:M_cmp]
        assert (
            len(old_paths) == M_cmp
        ), f"old_model missing members at n={n}: {len(old_paths)}"
        assert (
            len(new_paths) == M_cmp
        ), f"unguided missing members at n={n}: {len(new_paths)}"
        for m, (op, np_) in enumerate(zip(old_paths, new_paths)):
            s_old = read_state(op)
            s_new = read_state(np_)
            mx, ok = _state_max_diff(s_old, s_new)
            rows.append({"n": n, "m": m, "max_abs_diff": mx, "allclose": ok})

    comparison_df = pd.DataFrame(rows)
    return N_cmp, comparison_df


@app.cell(hide_code=True)
def _(comparison_df):
    per_step = comparison_df.groupby("n").agg(
        max_abs_diff=("max_abs_diff", "max"),
        all_allclose=("allclose", "all"),
    )
    per_step
    return


@app.cell(hide_code=True)
def _(comparison_df, mo):
    all_ok = bool(comparison_df["allclose"].all())
    worst = float(comparison_df["max_abs_diff"].max())
    mo.vstack(
        [
            mo.md(f"Allclose over whole state? -> {all_ok}"),
            mo.md(f"Worst max-abs-diff -> {worst:.3e}")
        ]
    )
    return


@app.cell(hide_code=True)
def _(N_cmp, PARTITIONS, mo, unguided_cfg):
    _cmp_default_partition = unguided_cfg.get("partition", PARTITIONS[0])
    cmp_partition_dropdown = mo.ui.dropdown(
        PARTITIONS, value=_cmp_default_partition, label="partition: "
    )
    cmp_n_slider = mo.ui.slider(
        steps=list(range(1, N_cmp + 1)), value=1, label="n: "
    )
    return cmp_n_slider, cmp_partition_dropdown


@app.cell(hide_code=True)
def _(LEVELS_DICT, VARIABLES_DICT, cmp_partition_dropdown, mo, unguided_cfg):
    _cmp_partition_val = cmp_partition_dropdown.value
    _cmp_vars = VARIABLES_DICT[_cmp_partition_val]
    _cmp_levels = LEVELS_DICT[_cmp_partition_val]
    _cmp_cfg_partition = unguided_cfg.get("partition")
    if _cmp_partition_val == _cmp_cfg_partition:
        _cmp_default_var = unguided_cfg.get("var", _cmp_vars[0])
        _cmp_default_level = unguided_cfg.get("level", _cmp_levels[0])
    else:
        _cmp_default_var = _cmp_vars[0]
        _cmp_default_level = _cmp_levels[0]
    cmp_var_dropdown = mo.ui.dropdown(
        _cmp_vars, value=_cmp_default_var, label="variable: "
    )
    cmp_level_slider = mo.ui.slider(
        steps=_cmp_levels, value=_cmp_default_level, label="level: "
    )
    return cmp_level_slider, cmp_var_dropdown


@app.cell(hide_code=True)
def _(
    cmp_level_slider,
    cmp_n_slider,
    cmp_partition_dropdown,
    cmp_var_dropdown,
    get_slice,
    old_model_rollout_dir,
    read_state,
    unguided_rollout_dir,
):
    cmp_var = cmp_var_dropdown.value
    cmp_level = cmp_level_slider.value
    cmp_n = cmp_n_slider.value
    cmp_partition = cmp_partition_dropdown.value

    cmp_old_path = sorted(old_model_rollout_dir.glob(f"{cmp_n}/*"))[0]
    cmp_new_path = sorted(unguided_rollout_dir.glob(f"{cmp_n}/*"))[0]
    cmp_old_state = read_state(cmp_old_path)
    cmp_new_state = read_state(cmp_new_path)

    _cmp_ts = cmp_old_state.time.values[0]
    cmp_old_slice = get_slice(
        cmp_old_state, cmp_partition, cmp_level, cmp_var, _cmp_ts
    )
    cmp_new_slice = get_slice(
        cmp_new_state, cmp_partition, cmp_level, cmp_var, _cmp_ts
    )
    cmp_diff_slice = cmp_old_slice - cmp_new_slice
    return cmp_diff_slice, cmp_level, cmp_new_slice, cmp_old_slice, cmp_var


@app.cell(hide_code=True)
def _(
    cmp_diff_slice,
    cmp_level,
    cmp_level_slider,
    cmp_n_slider,
    cmp_new_slice,
    cmp_old_slice,
    cmp_partition_dropdown,
    cmp_var,
    cmp_var_dropdown,
    mo,
    np,
    visualize_map,
):
    _cmp_abs_min = min(
        float(np.asarray(cmp_old_slice).min()),
        float(np.asarray(cmp_new_slice).min()),
    )
    _cmp_abs_max = max(
        float(np.asarray(cmp_old_slice).max()),
        float(np.asarray(cmp_new_slice).max()),
    )
    if _cmp_abs_max <= _cmp_abs_min:
        _cmp_abs_max = _cmp_abs_min + 1e-9
    _cmp_abs_center = 0.5 * (_cmp_abs_min + _cmp_abs_max)
    _cmp_abs_center = min(
        max(_cmp_abs_center, _cmp_abs_min + 1e-9), _cmp_abs_max - 1e-9
    )
    _cmp_diff_absmax = float(np.abs(np.asarray(cmp_diff_slice)).max()) or 1e-8

    cmp_old_map = visualize_map(
        cmp_old_slice,
        title=f"old_model  {cmp_var} @ {cmp_level}",
        vmin=_cmp_abs_min,
        vmax=_cmp_abs_max,
        center=_cmp_abs_center,
    )
    cmp_new_map = visualize_map(
        cmp_new_slice,
        title=f"unguided  {cmp_var} @ {cmp_level}",
        vmin=_cmp_abs_min,
        vmax=_cmp_abs_max,
        center=_cmp_abs_center,
    )
    cmp_diff_map = visualize_map(
        cmp_diff_slice,
        title="old - unguided",
        vmin=-_cmp_diff_absmax,
        vmax=_cmp_diff_absmax,
        center=0.0,
    )

    mo.vstack(
        [   mo.md(f"### diff analysis"),
            mo.hstack(
                [
                    cmp_partition_dropdown,
                    cmp_var_dropdown,
                    cmp_level_slider,
                    cmp_n_slider,
                ],
                justify="start",
            ),
            mo.hstack([cmp_old_map, cmp_new_map, cmp_diff_map]),
        ]
    )
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
