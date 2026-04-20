import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Boiler plate
    """)
    return


@app.cell
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
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, mo, np, pd, plt, xr


@app.cell
def _():
    from src.utils import (
        read_json,
        read_state,
        read_states,
        get_last_experiment_dir,
        get_slice
    )
    from src.interaction import visualize_map, get_mask_from_corners, get_mask_center
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.funcs import avg_over_mask

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        get_mask_center,
        get_mask_from_corners,
        get_slice,
        read_json,
        read_state,
        visualize_map,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _(np, plt):
    def extract_zoom_values(
        array_2d,
        *,
        zoom,
        center_lon=0.0,
        center_lat=0.0,
    ):
        array_2d = np.asarray(array_2d)
        ny, nx = array_2d.shape

        lon_e = np.linspace(-180.0, 180.0, nx + 1, endpoint=True)
        lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)

        lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
        lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

        zoom = max(1, int(zoom))

        full_lon_span = 360.0
        full_lat_span = 180.0

        lon_span = full_lon_span / zoom
        lat_span = full_lat_span / zoom

        lon_min = max(-180.0, center_lon - lon_span / 2)
        lon_max = min(180.0, center_lon + lon_span / 2)
        lat_min = max(-90.0, center_lat - lat_span / 2)
        lat_max = min(90.0, center_lat + lat_span / 2)

        lon_mask = (lon_c >= lon_min) & (lon_c <= lon_max)
        lat_mask = (lat_c >= lat_min) & (lat_c <= lat_max)

        zoom_values = array_2d[np.ix_(lat_mask, lon_mask)]
        return zoom_values[np.isfinite(zoom_values)]


    def plot_zoom_histogram(
        array_2d,
        *,
        zoom,
        center_lon=0.0,
        center_lat=0.0,
        bins=30,
        title="Value distribution in zoom zone",
        figsize=(11.75, 5.45),
        dpi=200,
    ):
        values = extract_zoom_values(
            array_2d,
            zoom=zoom,
            center_lon=center_lon,
            center_lat=center_lat,
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        return fig

    return (plot_zoom_histogram,)


@app.cell
def _(pd):
    def unix_tensor_to_iso(ts):
        return pd.to_datetime(int(ts.item()), unit="s").strftime("%Y-%m-%dT%H:%M:%S")

    return (unix_tensor_to_iso,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


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
    if partition == "surface":
        VARIABLES_VALUE = VARIABLES[2]
    else:
        VARIABLES_VALUE = VARIABLES[3]
    return VARIABLES, VARIABLES_VALUE


@app.cell
def _(VARIABLES, VARIABLES_VALUE, mo):
    var_dropdown = mo.ui.dropdown(VARIABLES, value=VARIABLES_VALUE, label="variable: ")
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
    return


@app.cell
def _(mo):
    show_mask_switch = mo.ui.checkbox(label="show mask")
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
    analysis_type_dropdown = mo.ui.dropdown(analysis_types, value=analysis_types[0], label="analysis type: ")
    return (analysis_type_dropdown,)


@app.cell
def _(mo):
    show_values_checkbox = mo.ui.checkbox(label="show values")
    return (show_values_checkbox,)


@app.cell
def _(show_values_checkbox):
    show_values = show_values_checkbox.value
    return (show_values,)


@app.cell
def _(guided_unguided, mo):
    center_ = 0.0
    max_ = max(guided_unguided.max().item(), abs(guided_unguided.min().item()))
    mean_ = (max_ - center_)*9 / 10
    value_threshold_slider  = mo.ui.slider(
        start=center_,
        stop=max_,
        step=mean_/10,
        value=mean_,
        label="text thresh: ",
    )
    return (value_threshold_slider,)


@app.cell
def _(value_threshold_slider):
    value_threshold = value_threshold_slider.value
    return (value_threshold,)


@app.cell
def _(guided_cfg, mo):
    n_slider = mo.ui.slider(steps=range(1, guided_cfg["N"]+1), value=1, label="n: ")
    return (n_slider,)


@app.cell
def _(n_slider):
    n = n_slider.value
    return (n,)


@app.cell
def _(mo, unguided_cfg):
    m_slider = mo.ui.slider(
        start=1,
        stop=unguided_cfg["M"],
        step=1,
        value=1,
        label="m: ",
    )
    return (m_slider,)


@app.cell
def _(m_slider):
    m = m_slider.value
    return (m,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data
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
def _(ds, guided_cfg, n, unix_tensor_to_iso):
    x_start = ds[guided_cfg["timestamp_idx"]+n]
    x_start = ds.denormalize(x_start)
    current = ds.convert_to_xarray(x_start["state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))
    next = ds.convert_to_xarray(x_start["next_state"].unsqueeze(0), x_start["timestamp"].unsqueeze(0))

    lead_time_seconds = x_start["lead_time_hours"] * 3600
    timestamp = x_start["timestamp"] - lead_time_seconds
    timestamp = unix_tensor_to_iso(timestamp)
    return current, next, timestamp, x_start


@app.cell
def _(
    current,
    get_slice,
    guided_state_n,
    level,
    next,
    partition,
    timestamp,
    unguided_state_n,
    var,
):
    guided_slice = get_slice(guided_state_n, partition, level, var, timestamp)
    unguided_slice = get_slice(unguided_state_n, partition, level, var, timestamp)
    current_slice = get_slice(current, partition, level, var, timestamp)
    next_slice = get_slice(next, partition, level, var, timestamp)

    guided_unguided = guided_slice - unguided_slice
    guided_gt = guided_slice - next_slice
    unguided_gt = unguided_slice - next_slice
    next_current = next_slice - current_slice
    guided_current = guided_slice - current_slice
    unguided_current = unguided_slice - current_slice
    return (
        current_slice,
        guided_current,
        guided_gt,
        guided_slice,
        guided_unguided,
        next_current,
        next_slice,
        unguided_current,
        unguided_gt,
        unguided_slice,
    )


@app.function
def get_rollout_path(rollout_dir, n, m):
    return rollout_dir / f"{n}" / f"{m}.nc"


@app.cell
def _(mo):
    mo.md(r"""
    # Results analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment
    """)
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


    guided_rollouts = Path("rollouts", "guided").glob("2026*")
    guided_rollouts = sorted(
        [p for p in guided_rollouts if has_config_json(p)],
        reverse=True,
    )
    pick_guided_rollout_dropdown = mo.ui.dropdown(label="Pick guided rollout", value=guided_rollouts[0], options=guided_rollouts)
    return (pick_guided_rollout_dropdown,)


@app.cell
def _(pick_guided_rollout_dropdown):
    guided_rollout_dir = pick_guided_rollout_dropdown.value
    return (guided_rollout_dir,)


@app.cell
def _(Path, guided_rollout_dir, read_json):
    guided_cfg = read_json(guided_rollout_dir, "config")
    unguided_rollout_dir = Path(guided_cfg["unguided_rollout_dir"])
    unguided_cfg = read_json(unguided_rollout_dir, "config")
    return guided_cfg, unguided_cfg, unguided_rollout_dir


@app.cell
def _(mo):
    mo.md(r"""
    ## Realized guidance
    """)
    return


@app.cell
def _(get_guidance, plt):
    def realized_guidance_branches(
        realized_terms: list[float],
        y_perc: list[float],
    ) -> list[float]:
        if len(realized_terms) != len(y_perc):
            raise ValueError("realized_terms and y_perc must have the same length")

        branches = [realized_terms[0]]
        for n in range(1, len(realized_terms)):
            branches.append(get_guidance(y_perc[n], realized_terms[n - 1]))
        return branches


    def plot_guidance_branching(
        timestamps: list[str],
        realized_terms: list[float],
        y_perc: list[float],
        planned_guidance: list[float] | None = None,
        ground_truth: list[float] | None = None,
        mean_rollout: list[float] | None = None,
        ensemble_rollout: list[list[float]] | None = None,
        title: str | None = "Realized guidance",
        subtitle: str | None = None,
    ):
        N = len(timestamps)
        if N != len(realized_terms) or len(realized_terms) != len(y_perc):
            raise ValueError(
                "timestamps, realized_terms, and y_perc must have the same length"
            )
        x = list(range(N))
        branch_targets = realized_guidance_branches(realized_terms, y_perc)

        C = {
            "realized": "#1f77b4",
            "online": "#d62728",
            "offline": "#ff7f0e",
            "gt": "#2ca02c",
            "mean": "#9467bd",
            "ensemble": "#7f7f7f",
        }

        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

        if ensemble_rollout is not None:
            M = len(ensemble_rollout[0])
            for i in range(M):
                y = [ensemble_rollout[n][i] for n in range(N)]
                ax.plot(
                    x,
                    y,
                    "-",
                    color=C["ensemble"],
                    linewidth=0.6,
                    alpha=0.35,
                    zorder=1,
                )
            lower = [min(row) for row in ensemble_rollout]
            upper = [max(row) for row in ensemble_rollout]
            ax.fill_between(
                x,
                lower,
                upper,
                color=C["ensemble"],
                alpha=0.12,
                label=f"Ensemble range (M={M})",
                zorder=1,
            )

        for n in range(1, N):
            ax.plot(
                [x[n - 1], x[n]],
                [realized_terms[n - 1], branch_targets[n]],
                linestyle="--",
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=C["online"],
                alpha=0.85,
                zorder=2,
                label="Online planned" if n == 1 else None,
            )

        if planned_guidance is not None:
            ax.plot(
                x,
                planned_guidance,
                "-",
                marker="s",
                markersize=3.5,
                linewidth=1.6,
                color=C["offline"],
                alpha=0.9,
                label="Offline planned",
                zorder=3,
            )

        if mean_rollout is not None:
            ax.plot(
                x,
                mean_rollout,
                "-",
                marker="D",
                markersize=3.5,
                linewidth=1.6,
                color=C["mean"],
                alpha=0.9,
                label="Mean rollout",
                zorder=3,
            )

        if ground_truth is not None:
            ax.plot(
                x,
                ground_truth,
                "-",
                marker="^",
                markersize=4.5,
                linewidth=1.8,
                color=C["gt"],
                alpha=0.95,
                label="Ground truth",
                zorder=4,
            )

        ax.plot(
            x,
            realized_terms,
            "-",
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            color=C["realized"],
            label="Realized",
            zorder=5,
        )

        tick_idx = [
            i for i, ts in enumerate(timestamps) if ts.endswith("00:00:00")
        ]
        if 0 not in tick_idx:
            tick_idx = [0] + tick_idx
        if N - 1 not in tick_idx:
            tick_idx.append(N - 1)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [timestamps[i] for i in tick_idx], rotation=35, ha="right", fontsize=8
        )
        ax.set_xlabel("Timestamp", fontsize=10)
        ax.set_ylabel("Mask term", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=9,
        )

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        if subtitle:
            fig.text(
                0.5,
                0.955,
                subtitle,
                ha="center",
                va="top",
                fontsize=9,
                color="#555",
            )

        fig.tight_layout(
            rect=(0.0, 0.0, 0.82, 0.93 if (title or subtitle) else 1.0)
        )
        return fig

    return (plot_guidance_branching,)


@app.cell
def _():
    from src.funcs import get_guidance

    return (get_guidance,)


@app.cell
def _(guided_cfg, guided_rollout_dir, read_json):
    timestamps = guided_cfg["timestamps"]
    y_perc = guided_cfg["y_perc"]
    planned_guidance = guided_cfg["planned_guidance"]
    ground_truth_ = guided_cfg["ground_truth"]

    # this should come from rollout_dir/mask_terms.json
    realized_terms = read_json(guided_rollout_dir, "mask_terms")["final_mask_terms"]
    return ground_truth_, planned_guidance, realized_terms, timestamps, y_perc


@app.cell
def _(
    ground_truth_,
    guided_cfg,
    planned_guidance,
    plot_guidance_branching,
    realized_terms,
    timestamps,
    y_perc,
):
    # NOTE:
    # - first branch of online and offline planned should coincide
    realized_guidance_plot = plot_guidance_branching(
        timestamps=timestamps,
        realized_terms=realized_terms,
        y_perc=y_perc,
        planned_guidance=planned_guidance,
        ground_truth=ground_truth_,
        mean_rollout=guided_cfg["mean_rollout"],
        ensemble_rollout=guided_cfg["ensemble_rollout"],
        title="Realized guidance analysis",
        subtitle="mask term along the rollout: realized vs online / offline plans, ground truth, mean & ensemble",
    )
    return (realized_guidance_plot,)


@app.cell
def _(guided_rollout_dir, m, n, read_state, unguided_rollout_dir):
    guided_state_n = read_state(get_rollout_path(guided_rollout_dir, n, m))
    unguided_state_n = read_state(get_rollout_path(unguided_rollout_dir, n, m))
    return guided_state_n, unguided_state_n


@app.cell
def _(get_mask_from_corners, guided_cfg):
    mask = get_mask_from_corners(*guided_cfg["mask_corners"])
    return (mask,)


@app.cell
def _(guided_cfg, mo, pick_guided_rollout_dropdown, refresh_button):
    mo.vstack([
        mo.hstack([
            pick_guided_rollout_dropdown, refresh_button,
        ], justify="start"),
        mo.accordion(
            {
                "Experiment params": mo.md("<br>".join(f"{k}: {v}" for k, v in guided_cfg.items())),
            }
        )
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## n-snapshots

    Single rollout step $n$, single ensemble member $m$, picked from the
    sliders below. Compare the **guided** posterior $x_{t+1}^{\text{guide}}$,
    the **unguided** sample $x_{t+1}^{\text{gen}}$, the ground-truth next
    state $x_{t+1}$, and the current state $x_t$. The `analysis_type`
    dropdown toggles between absolute fields and difference maps
    (`guided - unguided`, `next - current`, etc.). The mask outline and zoom
    controls below let you focus on the guidance region.
    """)
    return


@app.cell
def _(
    analysis_type,
    current_slice,
    guided_current,
    guided_gt,
    guided_slice,
    guided_unguided,
    mask,
    next_current,
    next_slice,
    np,
    show_mask,
    show_values,
    unguided_current,
    unguided_gt,
    unguided_slice,
    value_threshold,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    def _mask_bool(mask_2d):
        return np.asarray(mask_2d).astype(bool)


    def _stats_abs(arr, mask_2d):
        v = np.asarray(arr)[_mask_bool(mask_2d)]
        if v.size == 0:
            return float("nan"), float("nan")
        return float(v.mean()), float(np.abs(v).max())


    def _stats_diff(arr, mask_2d):
        v = np.asarray(arr)[_mask_bool(mask_2d)]
        if v.size == 0:
            return float("nan"), float("nan")
        return float(np.sqrt((v**2).mean())), float(np.abs(v).max())


    def _fmt(x):
        return f"{x:.3g}"


    _mask_np = np.asarray(mask)

    if analysis_type == "absolute":
        _abs_panels = [
            ("$x_t$", current_slice),
            ("$x_{t+1}$", next_slice),
            ("$x_{t+1}^{gen}$", unguided_slice),
            ("$x_{t+1}^{guide}$", guided_slice),
        ]
        _abs_vmin = min(float(np.asarray(a).min()) for _, a in _abs_panels)
        _abs_vmax = max(float(np.asarray(a).max()) for _, a in _abs_panels)
        if _abs_vmax <= _abs_vmin:
            _abs_vmax = _abs_vmin + 1e-9
        _abs_center = 0.5 * (_abs_vmin + _abs_vmax)
        _abs_center = min(max(_abs_center, _abs_vmin + 1e-9), _abs_vmax - 1e-9)

        _abs_maps = {}
        for _label, _arr in _abs_panels:
            _mu, _mx = _stats_abs(_arr, _mask_np)
            _abs_maps[_label] = visualize_map(
                _arr,
                mask_2d=_mask_np,
                title=f"{_label}   $\mu$={_fmt(_mu)}  |max|={_fmt(_mx)}",
                vmin=_abs_vmin,
                vmax=_abs_vmax,
                center=_abs_center,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
            )

        state_map = _abs_maps["$x_t$"]
        next_map = _abs_maps["$x_{t+1}$"]
        unguided_map = _abs_maps["$x_{t+1}^{gen}$"]
        guided_map = _abs_maps["$x_{t+1}^{guide}$"]

        unguided_gt_map = None
        guided_gt_map = None
        guided_unguided_map = None
        next_current_map = None
        unguided_current_map = None
        guided_current_map = None
    else:
        _diff_panels = [
            ("unguided-gt", unguided_gt),
            ("guided-gt", guided_gt),
            ("guided-unguided", guided_unguided),
            ("next-current", next_current),
            ("unguided-current", unguided_current),
            ("guided-current", guided_current),
        ]
        _diff_absmax = max(
            float(np.abs(np.asarray(a)).max()) for _, a in _diff_panels
        )
        if _diff_absmax <= 0:
            _diff_absmax = 1e-8

        _diff_maps = {}
        for _label, _arr in _diff_panels:
            _mu, _mx = _stats_diff(_arr, _mask_np)
            _diff_maps[_label] = visualize_map(
                _arr,
                mask_2d=_mask_np,
                title=f"{_label}   $\mu$={_fmt(_mu)}  |max|={_fmt(_mx)}",
                vmin=-_diff_absmax,
                vmax=_diff_absmax,
                center=0.0,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
                show_values=show_values,
                value_threshold=value_threshold,
                value_fontsize=5,
            )

        unguided_gt_map = _diff_maps["unguided-gt"]
        guided_gt_map = _diff_maps["guided-gt"]
        guided_unguided_map = _diff_maps["guided-unguided"]
        next_current_map = _diff_maps["next-current"]
        unguided_current_map = _diff_maps["unguided-current"]
        guided_current_map = _diff_maps["guided-current"]

        state_map = None
        next_map = None
        unguided_map = None
        guided_map = None
    return (
        guided_current_map,
        guided_gt_map,
        guided_map,
        guided_unguided_map,
        next_current_map,
        next_map,
        state_map,
        unguided_current_map,
        unguided_gt_map,
        unguided_map,
    )


@app.cell
def _(guided_unguided, plot_zoom_histogram, zoom_centers, zoom_slider):
    next_guided_hist = plot_zoom_histogram(
        guided_unguided,
        zoom=zoom_slider.value,
        center_lon=zoom_centers[0],
        center_lat=zoom_centers[1],
        title="unguided-guided - value dist in zoom area",
    )
    return (next_guided_hist,)


@app.cell
def _(
    analysis_type,
    analysis_type_dropdown,
    guided_current_map,
    guided_gt_map,
    guided_map,
    guided_unguided_map,
    level,
    level_slider,
    m_slider,
    mo,
    n_slider,
    next_current_map,
    next_guided_hist,
    next_map,
    partition_dropdown,
    show_mask_switch,
    show_values_checkbox,
    state_map,
    unguided_current_map,
    unguided_gt_map,
    unguided_map,
    value_threshold_slider,
    var_dropdown,
    zoom_slider,
):
    if analysis_type == "absolute":
        to_show = mo.vstack(
            [
                analysis_type_dropdown,
                mo.hstack([n_slider, m_slider], justify="start"),
                mo.hstack(
                    [
                        partition_dropdown,
                        var_dropdown,
                        mo.hstack(
                            [level_slider, mo.md(f"{level}")], justify="start"
                        ),
                    ],
                    justify="start",
                ),
                mo.hstack([show_mask_switch], justify="start"),
                zoom_slider,
                mo.hstack([state_map, next_map]),
                mo.hstack([unguided_map, guided_map]),
            ]
        )
    else:
        to_show = mo.vstack(
            [
                analysis_type_dropdown,
                mo.hstack([n_slider, m_slider], justify="start"),
                mo.hstack(
                    [
                        partition_dropdown,
                        var_dropdown,
                        mo.hstack(
                            [level_slider, mo.md(f"{level}")], justify="start"
                        ),
                    ],
                    justify="start",
                ),
                mo.hstack(
                    [
                        show_mask_switch,
                        show_values_checkbox,
                        value_threshold_slider,
                    ],
                    justify="start",
                ),
                zoom_slider,
                mo.md("**Errors vs ground truth**"),
                mo.hstack([unguided_gt_map, guided_gt_map, guided_unguided_map]),
                mo.md("**Evolutions from current**"),
                mo.hstack(
                    [next_current_map, unguided_current_map, guided_current_map]
                ),
                mo.md("**Distribution in zoom region**"),
                next_guided_hist,
            ]
        )
    to_show
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Static analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Masks over N

    Per rollout iteration $n = 1, \dots, N$, compare the **guided** and
    **unguided** posterior samples (ensemble member $m$) against the ERA5
    **ground truth** $x_{t_n+1}$ and the **current** state $x_{t_n}$ entering
    the sampler. Each row corresponds to a rollout step; each column to one of
    the four fields. Per-row color scale centers on the current field so
    spatial structure matches the single-step GUI above.

    Subplots are framed to the mask bounding box — the red outline shows the
    mask region used to guide generation.

    Use `analysis_type = "difference"` to display the residual fields (ground
    truth minus current / guided minus unguided / ground truth minus guided /
    ground truth minus unguided) with a symmetric color scale centered at zero.
    """)
    return


@app.function
def get_ds_state(ds, idx):
    state = ds[idx]
    return ds.convert_to_xarray(state["state"].unsqueeze(0), state["timestamp"].unsqueeze(0))


@app.cell
def _(guided_cfg):
    timestamp_idx = guided_cfg["timestamp_idx"]
    return (timestamp_idx,)


@app.cell
def _():
    from src.interaction import plot_states_over_n

    return (plot_states_over_n,)


@app.cell
def _(
    N,
    analysis_type,
    ds,
    get_slice,
    guided_rollout_dir,
    m,
    mask,
    partition,
    plot_states_over_n,
    read_state,
    timestamp,
    timestamp_idx,
    unguided_rollout_dir,
    var,
):
    masks_over_n_plot,_ = plot_states_over_n(
        ds=ds,
        guided_rollout_dir=guided_rollout_dir,
        unguided_rollout_dir=unguided_rollout_dir,
        get_rollout_path=get_rollout_path,
        read_state=read_state,
        get_slice=get_slice,
        timestamp=timestamp,
        timestamp_idx=timestamp_idx,
        N=N,
        m=m,
        partition=partition,
        var=var,
        level=None,
        mask_2d=mask,
        show_mask=True,
        analysis_type=analysis_type,
        title=f"States over N — {analysis_type}",
        subtitle=f"var={var}  |  member m={m}  |  timestamp={timestamp}",
    )
    masks_over_n_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Realized guidance during sampling

    For each rollout step $n=1,\dots,N$, the curve below traces the
    mask-averaged loss term across the diffusion denoising steps (inner
    loop of one sampling call). Lower values mean the sample better matches
    the target observation $y$ inside the mask; a non-monotone trajectory
    indicates the guidance competes with the prior score during sampling.
    """)
    return


@app.cell
def _(guided_rollout_dir, read_json):
    mask_terms = read_json(guided_rollout_dir, "mask_terms")
    final_mask_terms = mask_terms["final_mask_terms"]
    all_mask_terms = mask_terms["all_mask_terms"]
    # all_mask_terms
    return (all_mask_terms,)


@app.cell
def _():
    from src.interaction import plot_trajectory, plot_trajectories_over_n

    return (plot_trajectories_over_n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TODO
    - annotate with hline the planned guidance.
    - also overlay on the right y-axis the "difference curve" and compare to the "planned curve".
    """)
    return


@app.cell
def _(all_mask_terms, plot_trajectories_over_n):
    trajectories_over_n_plot,_ = plot_trajectories_over_n(
        trajectories=all_mask_terms,
        var="mask term",
        title="Realized guidance over N",
        subtitle="mask-averaged loss term across diffusion steps, one panel per rollout step n",
    )
    trajectories_over_n_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Realized guidance analysis

    End-of-sampling comparison along the rollout. The **realized** line is
    the mask term actually reached by the guided sample at each step. The
    **online planned** dashed segments show, for each step $n$, the target
    the online planner aims for given the previous realized value. The
    **offline planned** curve is the full trajectory planned before sampling
    begins. **Ground truth** is the mask term evaluated on ERA5; **mean
    rollout** and the **ensemble** envelope show what an unguided ensemble
    would produce over the same window.
    """)
    return


@app.cell
def _(realized_guidance_plot):
    realized_guidance_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Variable change analysis

    For each rollout step $n=1,\dots,N$ and each variable–level key (e.g.
    `u_component_of_wind-500`, `2m_temperature-surface`), we measure how much
    the **guided** state differs from the **unguided** state in the normalized
    space, summed spatially. Scores are then min-max normalized per ensemble
    member so that the leading channels at each step sit near 1 and the
    quiescent channels near 0.

    The plot below is a **parallel-coordinates** view: each key is a polyline
    across rollout steps, with error bars showing the ensemble spread (std).
    Use the slider to filter to the top-$k$ channels ranked by their maximum
    score across $N$.
    """)
    return


@app.cell
def _(mo):
    top_k_slider = mo.ui.slider(
        start=1, stop=50, value=12, step=1, label="top-k channels"
    )
    rank_by_radio = mo.ui.radio(
        options=["max", "mean"], value="max", label="rank by"
    )
    mo.hstack([top_k_slider, rank_by_radio], justify="start")
    return rank_by_radio, top_k_slider


@app.cell
def _():
    from src.interaction import plot_variable_change_parallel

    return (plot_variable_change_parallel,)


@app.cell
def _(unguided_cfg):
    N = unguided_cfg["N"]
    M = unguided_cfg["M"]
    return M, N


@app.cell
def _(
    M,
    N,
    ds,
    guided_rollout_dir,
    minmax_normalize_dict,
    per_var_level_sum_dict,
    read_state,
    unguided_rollout_dir,
    x_start,
    xr,
):
    minmax_analysis_list = []

    for n_iter in range(1, N + 1):
        guided_paths_n_minmax = sorted((guided_rollout_dir / f"{n_iter}").glob("*"))
        unguided_paths_n_minmax = sorted((unguided_rollout_dir / f"{n_iter}").glob("*"))

        M_eff = min(len(guided_paths_n_minmax), len(unguided_paths_n_minmax), M)
        abs_sum_dict_minmax_list = []

        for m_iter in range(M_eff):
            guided_state_n_minmax = read_state(guided_paths_n_minmax[m_iter])
            guided_state_n_minmax = read_state(unguided_paths_n_minmax[m_iter])

            guided_normalized = ds.convert_to_tensordict(guided_state_n_minmax)
            guided_normalized = guided_normalized.apply(lambda x: x.squeeze(1).unsqueeze(0))
            guided_normalized = ds.normalize(guided_normalized)

            unguided_normalized = ds.convert_to_tensordict(guided_state_n_minmax)
            unguided_normalized = unguided_normalized.apply(lambda x: x.squeeze(1).unsqueeze(0))
            unguided_normalized = ds.normalize(unguided_normalized)

            diff_guided_unguided_norm = guided_normalized - unguided_normalized
            diff_guided_unguided_norm = ds.convert_to_xarray(
                diff_guided_unguided_norm,
                x_start["timestamp"].unsqueeze(0),
            )
            diff_guided_unguided_norm = xr.ufuncs.abs(diff_guided_unguided_norm)

            abs_sum_dict = per_var_level_sum_dict(diff_guided_unguided_norm)
            abs_sum_dict_minmax = minmax_normalize_dict(abs_sum_dict)

            abs_sum_dict_minmax_list.append(abs_sum_dict_minmax)

        minmax_analysis_list.append(abs_sum_dict_minmax_list)
    return (minmax_analysis_list,)


@app.cell
def _(np):
    def aggregate_dicts(dict_list, error="std"):
        keys = list(dict_list[0].keys())
        arr = np.array([[d[k] for k in keys] for d in dict_list], dtype=float)

        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        sem = std / np.sqrt(arr.shape[0])

        err = std if error == "std" else sem

        return {
            "keys": keys,
            "mean": mean,
            "err": err,
            "values": arr,
        }

    return (aggregate_dicts,)


@app.cell
def _(np, plt):
    def per_var_level_sum_dict(ds_abs):
        out = {}

        for var_name, da in ds_abs.data_vars.items():
            if "level" in da.dims:
                for level in da.level.values:
                    key = f"{var_name}-{int(level)}"
                    out[key] = float(da.sel(level=level).sum().item())
            else:
                key = f"{var_name}-surface"
                out[key] = float(da.sum().item())

        return out

    def minmax_normalize_dict(d):
        vals = list(d.values())
        vmin = min(vals)
        vmax = max(vals)

        if vmax == vmin:
            return {k: 0.0 for k in d}

        return {
            k: (v - vmin) / (vmax - vmin)
            for k, v in d.items()
        }

    def plot_errorbar_panels(
        aggregated_per_n,
        top_k=20,
        figsize_per_panel=(12, 4),
        rotate_xticks=90,
    ):
        n_panels = len(aggregated_per_n)

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_panels),
            squeeze=False
        )
        axes = axes.flatten()

        for i, agg in enumerate(aggregated_per_n):
            keys = np.array(agg["keys"])
            mean = np.array(agg["mean"])
            err = np.array(agg["err"])

            order = np.argsort(mean)[::-1]
            if top_k is not None:
                order = order[:top_k]

            keys_plot = keys[order]
            mean_plot = mean[order]
            err_plot = err[order]

            x = np.arange(len(keys_plot))
            ax = axes[i]

            ax.errorbar(
                x,
                mean_plot,
                yerr=err_plot,
                fmt="o",
                ls="--",
                capsize=5,
                capthick=1,
                ecolor="black",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(keys_plot, rotation=rotate_xticks)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("min-max score")
            ax.set_title(f"N = {i+1}")
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig, axes

    return minmax_normalize_dict, per_var_level_sum_dict


@app.cell
def _(
    aggregate_dicts,
    minmax_analysis_list,
    plot_variable_change_parallel,
    rank_by_radio,
    top_k_slider,
):
    aggregated_per_n = [
        aggregate_dicts(dict_list, error="std")
        for dict_list in minmax_analysis_list
    ]
    var_change_fig, _ = plot_variable_change_parallel(
        aggregated_per_n,
        top_k=top_k_slider.value,
        rank_by=rank_by_radio.value,
        title="Variable change across rollout steps",
        subtitle="min-max normalized |guided - unguided| per variable-level; error bars = ensemble std",
    )
    var_change_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Variable change (level-aggregated)

    Same analysis as above, but with pressure-level scores **summed per
    variable** (surface-only variables are dropped). Each polyline is a
    variable, tracked across rollout steps with ensemble error bars. Use the
    slider to show the top-$k$ variables by max/mean score.
    """)
    return


@app.cell
def _(aggregate_dicts, minmax_analysis_list, plot_variable_change_parallel):
    def collapse_level_keys_to_vars(d):
        out = {}
        for key, val in d.items():
            var, suffix = key.rsplit("-", 1)
            if suffix == "surface":
                continue
            out[var] = out.get(var, 0.0) + val
        return out


    var_analysis_list = [
        [collapse_level_keys_to_vars(d) for d in dict_list]
        for dict_list in minmax_analysis_list
    ]

    aggregated_vars_per_n = [
        aggregate_dicts(dict_list, error="std") for dict_list in var_analysis_list
    ]

    var_agg_fig, _ = plot_variable_change_parallel(
        aggregated_vars_per_n,
        top_k=None,
        rank_by="max",
        title="Variable change across rollout steps (level-aggregated)",
        subtitle="pressure-level keys summed per variable; error bars = ensemble std",
        ylim=None,
        ylabel="score (sum over levels)",
    )
    var_agg_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### RMSE analysis

    For each rollout step $n=1,\dots,N$ and each ensemble member $m$, compute
    the **normalized-space RMSE on the mask** of guided vs ground truth and
    unguided vs ground truth. The two parallel-coordinates plots below show
    the **relative improvement**

    $$
    r = \frac{\mathrm{RMSE}_{\text{unguided}} - \mathrm{RMSE}_{\text{guided}}}{\mathrm{RMSE}_{\text{unguided}}}
    $$

    (positive = guidance helps) per variable-level and per variable. The third
    plot shows the absolute mask RMSE of guided and unguided sides by side
    across $n$ (ensemble mean, shaded std).
    """)
    return


@app.cell
def _(np):
    def per_var_level_rmse_dict_on_mask(ds_diff, mask_2d):
        m2 = np.asarray(mask_2d).astype(bool)
        out = {}
        for var_name, da in ds_diff.data_vars.items():
            arr = np.asarray(da.squeeze("time", drop=True)) ** 2
            if "level" in da.dims:
                for i, lvl in enumerate(da.level.values):
                    vals = arr[i][m2]
                    out[f"{var_name}-{int(lvl)}"] = float(np.sqrt(vals.mean()))
            else:
                vals = arr[m2]
                out[f"{var_name}-surface"] = float(np.sqrt(vals.mean()))
        return out


    def per_var_rmse_dict_on_mask(ds_diff, mask_2d, drop_surface=True):
        m2 = np.asarray(mask_2d).astype(bool)
        out = {}
        for var_name, da in ds_diff.data_vars.items():
            arr = np.asarray(da.squeeze("time", drop=True)) ** 2
            if "level" in da.dims:
                pooled = arr[..., m2].ravel()
                out[var_name] = float(np.sqrt(pooled.mean()))
            elif not drop_surface:
                out[f"{var_name}-surface"] = float(np.sqrt(arr[m2].mean()))
        return out


    def total_rmse_on_mask(ds_diff, mask_2d):
        m2 = np.asarray(mask_2d).astype(bool)
        accum = []
        for _, da in ds_diff.data_vars.items():
            arr = np.asarray(da.squeeze("time", drop=True)) ** 2
            if "level" in da.dims:
                accum.append(arr[..., m2].ravel())
            else:
                accum.append(arr[m2])
        return float(np.sqrt(np.concatenate(accum).mean()))


    def rel_improvement(rmse_unguided_dict, rmse_guided_dict, eps=1e-12):
        return {
            k: (rmse_unguided_dict[k] - rmse_guided_dict[k])
            / max(rmse_unguided_dict[k], eps)
            for k in rmse_guided_dict
        }


    def to_norm_xr(state_xr, ds_, timestamp):
        td = ds_.convert_to_tensordict(state_xr)
        td = td.apply(lambda x: x.squeeze(1).unsqueeze(0))
        td = ds_.normalize(td)
        return ds_.convert_to_xarray(td, timestamp)


    def total_rmse_full(ds_diff):
        accum = []
        for _, da in ds_diff.data_vars.items():
            arr = np.asarray(da.squeeze("time", drop=True)) ** 2
            accum.append(arr.ravel())
        return float(np.sqrt(np.concatenate(accum).mean()))

    return (
        per_var_level_rmse_dict_on_mask,
        per_var_rmse_dict_on_mask,
        rel_improvement,
        to_norm_xr,
        total_rmse_full,
        total_rmse_on_mask,
    )


@app.cell
def _(
    M,
    N,
    ds,
    guided_rollout_dir,
    mask,
    np,
    per_var_level_rmse_dict_on_mask,
    per_var_rmse_dict_on_mask,
    read_state,
    rel_improvement,
    timestamp_idx,
    to_norm_xr,
    total_rmse_full,
    total_rmse_on_mask,
    unguided_rollout_dir,
):
    rmse_level_rel_list = []
    rmse_var_rel_list = []
    rmse_guided_total = np.full((N, M), np.nan)
    rmse_unguided_total = np.full((N, M), np.nan)
    rmse_guided_full = np.full((N, M), np.nan)
    rmse_unguided_full = np.full((N, M), np.nan)

    for _n in range(1, N + 1):
        g_paths = sorted((guided_rollout_dir / f"{_n}").glob("*"))
        u_paths = sorted((unguided_rollout_dir / f"{_n}").glob("*"))
        rmse_M_eff = min(len(g_paths), len(u_paths), M)

        gt_td = ds[timestamp_idx + _n]
        gt_ts = gt_td["timestamp"].unsqueeze(0)
        gt_norm_xr = ds.convert_to_xarray(gt_td["next_state"].unsqueeze(0), gt_ts)

        level_rel_m = []
        var_rel_m = []

        for _m in range(rmse_M_eff):
            g_xr = to_norm_xr(read_state(g_paths[_m]), ds, gt_ts)
            u_xr = to_norm_xr(read_state(u_paths[_m]), ds, gt_ts)

            diff_g = g_xr - gt_norm_xr
            diff_u = u_xr - gt_norm_xr

            lvl_g = per_var_level_rmse_dict_on_mask(diff_g, mask)
            lvl_u = per_var_level_rmse_dict_on_mask(diff_u, mask)
            level_rel_m.append(rel_improvement(lvl_u, lvl_g))

            var_g = per_var_rmse_dict_on_mask(diff_g, mask)
            var_u = per_var_rmse_dict_on_mask(diff_u, mask)
            var_rel_m.append(rel_improvement(var_u, var_g))

            rmse_guided_total[_n - 1, _m] = total_rmse_on_mask(diff_g, mask)
            rmse_unguided_total[_n - 1, _m] = total_rmse_on_mask(diff_u, mask)
            rmse_guided_full[_n - 1, _m] = total_rmse_full(diff_g)
            rmse_unguided_full[_n - 1, _m] = total_rmse_full(diff_u)

        rmse_level_rel_list.append(level_rel_m)
        rmse_var_rel_list.append(var_rel_m)
    return (
        rmse_guided_full,
        rmse_guided_total,
        rmse_level_rel_list,
        rmse_unguided_full,
        rmse_unguided_total,
        rmse_var_rel_list,
    )


@app.cell
def _(
    aggregate_dicts,
    mo,
    np,
    rmse_guided_full,
    rmse_guided_total,
    rmse_level_rel_list,
    rmse_unguided_full,
    rmse_unguided_total,
    rmse_var_rel_list,
):
    agg_level_rel_per_n = [
        aggregate_dicts(dl, error="std") for dl in rmse_level_rel_list
    ]
    agg_var_rel_per_n = [
        aggregate_dicts(dl, error="std") for dl in rmse_var_rel_list
    ]

    rmse_guided_mean = np.nanmean(rmse_guided_total, axis=1)
    rmse_unguided_mean = np.nanmean(rmse_unguided_total, axis=1)
    rmse_guided_std = np.nanstd(rmse_guided_total, axis=1)
    rmse_unguided_std = np.nanstd(rmse_unguided_total, axis=1)

    rmse_guided_full_mean = np.nanmean(rmse_guided_full, axis=1)
    rmse_unguided_full_mean = np.nanmean(rmse_unguided_full, axis=1)
    rmse_guided_full_std = np.nanstd(rmse_guided_full, axis=1)
    rmse_unguided_full_std = np.nanstd(rmse_unguided_full, axis=1)

    top_k_rmse_slider = mo.ui.slider(
        start=1, stop=50, value=12, step=1, label="top-k var-level"
    )
    rank_by_rmse_radio = mo.ui.radio(
        options=["max", "mean"], value="max", label="rank by"
    )
    mo.hstack([top_k_rmse_slider, rank_by_rmse_radio], justify="start")
    return (
        agg_level_rel_per_n,
        agg_var_rel_per_n,
        rank_by_rmse_radio,
        rmse_guided_full_mean,
        rmse_guided_full_std,
        rmse_guided_mean,
        rmse_guided_std,
        rmse_unguided_full_mean,
        rmse_unguided_full_std,
        rmse_unguided_mean,
        rmse_unguided_std,
        top_k_rmse_slider,
    )


@app.cell
def _(
    agg_level_rel_per_n,
    plot_variable_change_parallel,
    rank_by_rmse_radio,
    top_k_rmse_slider,
):
    rmse_level_fig, _ = plot_variable_change_parallel(
        agg_level_rel_per_n,
        top_k=top_k_rmse_slider.value,
        rank_by=rank_by_rmse_radio.value,
        title="Relative RMSE improvement — per variable-level",
        subtitle="(RMSE_unguided - RMSE_guided) / RMSE_unguided on mask (normalized space); error bars = ensemble std",
        ylim=None,
        ylabel="relative RMSE improvement",
    )
    rmse_level_fig
    return


@app.cell
def _(agg_var_rel_per_n, plot_variable_change_parallel):
    rmse_var_fig, _ = plot_variable_change_parallel(
        agg_var_rel_per_n,
        top_k=None,
        rank_by="max",
        title="Relative RMSE improvement — per variable (level-pooled)",
        subtitle="per-variable RMSE pooled across pressure levels; error bars = ensemble std",
        ylim=None,
        ylabel="relative RMSE improvement",
    )
    rmse_var_fig
    return


@app.cell
def _(
    M,
    rmse_guided_mean,
    rmse_guided_std,
    rmse_unguided_mean,
    rmse_unguided_std,
):
    from src.interaction import plot_rmse_over_n

    rmse_over_n_fig, _ = plot_rmse_over_n(
        rmse_guided=rmse_guided_mean,
        rmse_unguided=rmse_unguided_mean,
        err_guided=rmse_guided_std,
        err_unguided=rmse_unguided_std,
        title="Mask RMSE over rollout",
        subtitle=f"ensemble mean +/- std on mask, normalized space (M={M})",
    )
    rmse_over_n_fig
    return (plot_rmse_over_n,)


@app.cell
def _(
    M,
    plot_rmse_over_n,
    rmse_guided_full_mean,
    rmse_guided_full_std,
    rmse_unguided_full_mean,
    rmse_unguided_full_std,
):
    rmse_over_n_full_fig, _ = plot_rmse_over_n(
        rmse_guided=rmse_guided_full_mean,
        rmse_unguided=rmse_unguided_full_mean,
        err_guided=rmse_guided_full_std,
        err_unguided=rmse_unguided_full_std,
        title="Whole-state RMSE over rollout",
        subtitle=f"ensemble mean +/- std on the full domain, normalized space (M={M})",
        ylabel="normalized RMSE (whole state)",
    )
    rmse_over_n_full_fig
    return


if __name__ == "__main__":
    app.run()
