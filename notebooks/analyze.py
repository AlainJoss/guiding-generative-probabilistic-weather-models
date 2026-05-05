import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Analyze rollouts

    This notebook uses the new rollout format:

    - `ground_truth.nc`
    - `unguided.nc`
    - `guided.nc`
    - `config.json`
    - `mask_terms.json`

    The ground truth contains the initial timestamp plus the future timestamps.
    Guided and unguided rollouts contain only the future timestamps.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import xarray as xr
    import matplotlib.pyplot as plt

    return mo, np, plt, torch


@app.cell
def _():
    from src.paths import ROLLOUTS
    from src.utils import (
        read_json,
        read_nc,
        get_experiment_ids,
        get_rollout_dir,
    )
    from src.interaction import (
        visualize_map,
        get_mask_from_corners,
        get_mask_center,
        plot_dual_trajectory,
        visualize_mask_terms_over_N
    )
    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.funcs import avg_over_mask, compute_mean_rollout

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        avg_over_mask,
        compute_mean_rollout,
        get_experiment_ids,
        get_mask_center,
        get_mask_from_corners,
        get_rollout_dir,
        plot_dual_trajectory,
        read_json,
        read_nc,
        visualize_map,
        visualize_mask_terms_over_N,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Pick experiments
    """)
    return


@app.cell
def _(get_experiment_ids, mo):
    refresh_button = mo.ui.button(label="refresh")

    guided_ids = get_experiment_ids("guided")
    unguided_ids = get_experiment_ids("unguided")

    if not guided_ids:
        raise ValueError("No guided rollouts found with guided.nc and config.json.")
    if not unguided_ids:
        raise ValueError("No unguided rollouts found with unguided.nc and config.json.")

    guided_dropdown = mo.ui.dropdown(
        guided_ids,
        value=guided_ids[0],
        label="guided rollout: ",
    )

    unguided_dropdown = mo.ui.dropdown(
        unguided_ids,
        value=unguided_ids[0],
        label="unguided rollout: ",
    )

    mo.vstack(
        [
            mo.hstack([guided_dropdown, unguided_dropdown, refresh_button], justify="start"),
        ]
    )
    return guided_dropdown, unguided_dropdown


@app.cell
def _(get_rollout_dir, guided_dropdown, read_json, read_nc, unguided_dropdown):
    guided_rollout_dir = get_rollout_dir(guided_dropdown.value)
    unguided_rollout_dir = get_rollout_dir(unguided_dropdown.value)

    guided_cfg = read_json(guided_rollout_dir, "config")
    unguided_cfg = read_json(unguided_rollout_dir, "config")

    guided_xr = read_nc(guided_rollout_dir, "guided")
    unguided_xr = read_nc(unguided_rollout_dir, "unguided")

    # Either rollout has ground_truth.nc, but usually use the guided one for the guided config.
    ground_truth_xr = read_nc(guided_rollout_dir, "ground_truth")
    return (
        ground_truth_xr,
        guided_cfg,
        guided_rollout_dir,
        guided_xr,
        unguided_cfg,
        unguided_xr,
    )


@app.cell
def _(guided_cfg, guided_dropdown, mo, unguided_cfg, unguided_dropdown):
    mo.accordion(
        {
            "Guided config": mo.md(
                "<br>".join(f"{k}: {v}" for k, v in guided_cfg.items())
            ),
            "Unguided config": mo.md(
                "<br>".join(f"{k}: {v}" for k, v in unguided_cfg.items())
            ),
            "Selected ids": mo.md(
                f"guided: `{guided_dropdown.value}`<br>"
                f"unguided: `{unguided_dropdown.value}`"
            ),
        }
    )
    return


@app.cell
def _(ground_truth_xr, guided_cfg, guided_xr, unguided_xr):
    N = int(guided_cfg["N"])

    guided_members = list(guided_xr.member.values) if "member" in guided_xr.dims else [None]
    unguided_members = (
        list(unguided_xr.member.values) if "member" in unguided_xr.dims else [None]
    )

    M_guided = len(guided_members)
    M_unguided = len(unguided_members)
    M = min(M_guided, M_unguided)

    # ground truth has init + future
    gt_times = list(ground_truth_xr.time.values)

    # guided and unguided have only future
    guided_times = list(guided_xr.time.values)
    unguided_times = list(unguided_xr.time.values)

    future_times = list(unguided_xr.time.values)
    plot_times = [gt_times[0]] + future_times

    timestamps = [str(t).split(".")[0] for t in plot_times]
    future_timestamps = [str(t).split(".")[0] for t in future_times]
    return (
        M,
        N,
        future_times,
        future_timestamps,
        guided_members,
        plot_times,
        timestamps,
        unguided_members,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Variable, mask, and controls
    """)
    return


@app.cell
def _(PARTITIONS, guided_cfg, mo):
    default_partition = guided_cfg.get("partition", PARTITIONS[0])
    partition_dropdown = mo.ui.dropdown(
        PARTITIONS,
        value=default_partition,
        label="partition: ",
    )
    return (partition_dropdown,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(LEVELS_DICT, guided_cfg, partition):
    LEVELS = LEVELS_DICT[partition]

    if partition == guided_cfg.get("partition") and guided_cfg.get("level") is not None:
        default_level = guided_cfg["level"]
        try:
            default_level = int(default_level)
        except Exception:
            pass
    else:
        default_level = LEVELS[0]
    return LEVELS, default_level


@app.cell
def _(LEVELS, default_level, mo):
    level_slider = mo.ui.slider(
        steps=LEVELS,
        value=default_level,
        label="level: ",
        show_value=True,
    )
    return (level_slider,)


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, guided_cfg, partition):
    VARIABLES = VARIABLES_DICT[partition]

    if partition == guided_cfg.get("partition"):
        default_var = guided_cfg.get("var", VARIABLES[0])
    elif partition == "surface":
        default_var = "2m_temperature" if "2m_temperature" in VARIABLES else VARIABLES[0]
    else:
        default_var = "temperature" if "temperature" in VARIABLES else VARIABLES[0]
    return VARIABLES, default_var


@app.cell
def _(VARIABLES, default_var, mo):
    var_dropdown = mo.ui.dropdown(
        VARIABLES,
        value=default_var,
        label="variable: ",
    )
    return (var_dropdown,)


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(M, mo):
    m_slider = mo.ui.slider(
        start=1,
        stop=max(1, M),
        step=1,
        value=1,
        label="member: ",
        show_value=True,
    )
    return (m_slider,)


@app.cell
def _(m_slider):
    m_idx = int(m_slider.value) - 1
    return (m_idx,)


@app.cell
def _(N, mo):
    n_slider = mo.ui.slider(
        steps=range(1, N + 1),
        value=1,
        label="rollout step n: ",
        show_value=True,
    )
    return (n_slider,)


@app.cell
def _(n_slider):
    n = int(n_slider.value)
    return (n,)


@app.cell
def _(mo):
    analysis_type_dropdown = mo.ui.dropdown(
        ["absolute", "difference"],
        value="absolute",
        label="analysis type: ",
    )

    show_mask_switch = mo.ui.checkbox(label="show mask", value=True)
    show_values_checkbox = mo.ui.checkbox(label="show values", value=False)

    zoom_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=1,
        label="zoom: ",
        show_value=True,
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    analysis_type_dropdown,
                    show_mask_switch,
                    show_values_checkbox,
                    zoom_slider,
                ],
                justify="start",
            )
        ]
    )
    return (
        analysis_type_dropdown,
        show_mask_switch,
        show_values_checkbox,
        zoom_slider,
    )


@app.cell
def _(
    analysis_type_dropdown,
    show_mask_switch,
    show_values_checkbox,
    zoom_slider,
):
    analysis_type = analysis_type_dropdown.value
    show_mask = show_mask_switch.value
    show_values = show_values_checkbox.value
    zoom = zoom_slider.value
    return analysis_type, show_mask, show_values, zoom


@app.cell
def _(get_mask_center, get_mask_from_corners, guided_cfg):
    mask = get_mask_from_corners(*guided_cfg["mask_corners"])
    zoom_center_lon, zoom_center_lat = get_mask_center(mask)
    return mask, zoom_center_lat, zoom_center_lon


@app.cell
def _(mo):
    mo.md(r"""
    ## Helpers
    """)
    return


@app.cell
def _(avg_over_mask, np, torch):
    def select_da(xr_ds, var, timestamp, level=None, member=None):
        da = xr_ds[var].sel(time=timestamp)

        if member is not None and "member" in da.dims:
            da = da.sel(member=member)

        if "level" in da.dims and level is not None:
            da = da.sel(level=int(level))

        return da

    def da_to_torch(da, dtype=None):
        if dtype is None:
            return torch.tensor(da.values)
        return torch.tensor(da.values, dtype=dtype)

    def avg_xr_over_mask(xr_ds, var, timestamp, mask, level=None, member=None):
        da = select_da(
            xr_ds=xr_ds,
            var=var,
            timestamp=timestamp,
            level=level,
            member=member,
        )
        x = torch.tensor(da.values, dtype=mask.dtype)
        return avg_over_mask(x, mask)

    def as_np_2d(da):
        arr = np.asarray(da.values)
        return np.squeeze(arr)

    return as_np_2d, avg_xr_over_mask, select_da


@app.cell
def _(guided_members, m_idx, unguided_members):
    guided_member = guided_members[m_idx]
    unguided_member = unguided_members[m_idx]
    return guided_member, unguided_member


@app.cell
def _(
    future_times,
    ground_truth_xr,
    guided_member,
    guided_xr,
    level,
    n,
    select_da,
    unguided_member,
    unguided_xr,
    var,
):
    # n = 1..N
    current_time = ground_truth_xr.time.values[n - 1]
    future_time = future_times[n - 1]

    current_da = select_da(
        ground_truth_xr,
        var=var,
        timestamp=current_time,
        level=level,
        member=None,
    )
    gt_da = select_da(
        ground_truth_xr,
        var=var,
        timestamp=future_time,
        level=level,
        member=None,
    )
    guided_da = select_da(
        guided_xr,
        var=var,
        timestamp=future_time,
        level=level,
        member=guided_member,
    )
    unguided_da = select_da(
        unguided_xr,
        var=var,
        timestamp=future_time,
        level=level,
        member=unguided_member,
    )
    return current_da, gt_da, guided_da, unguided_da


@app.cell
def _(as_np_2d, current_da, gt_da, guided_da, unguided_da):
    current_slice = as_np_2d(current_da)
    gt_slice = as_np_2d(gt_da)
    guided_slice = as_np_2d(guided_da)
    unguided_slice = as_np_2d(unguided_da)

    guided_unguided = guided_slice - unguided_slice
    guided_gt = guided_slice - gt_slice
    unguided_gt = unguided_slice - gt_slice
    gt_current = gt_slice - current_slice
    guided_current = guided_slice - current_slice
    unguided_current = unguided_slice - current_slice
    return (
        current_slice,
        gt_current,
        gt_slice,
        guided_current,
        guided_gt,
        guided_slice,
        guided_unguided,
        unguided_current,
        unguided_gt,
        unguided_slice,
    )


@app.cell
def _(guided_unguided, mo, np):
    absmax = float(np.nanmax(np.abs(guided_unguided)))
    if absmax <= 0:
        absmax = 1.0

    value_threshold_slider = mo.ui.slider(
        start=0.0,
        stop=absmax,
        step=absmax / 20,
        value=0.8 * absmax,
        label="text threshold: ",
        show_value=True,
    )
    return (value_threshold_slider,)


@app.cell
def _(value_threshold_slider):
    value_threshold = value_threshold_slider.value
    return (value_threshold,)


@app.cell
def _(mo):
    mo.md(r"""
    # Trajectory analysis
    """)
    return


@app.cell
def _(
    avg_xr_over_mask,
    compute_mean_rollout,
    future_times,
    ground_truth_xr,
    guided_cfg,
    guided_members,
    guided_xr,
    level,
    mask,
    plot_times,
    unguided_members,
    unguided_xr,
    var,
):
    # ground_truth has init + future times
    # guided / unguided only have future times

    ground_truth = []
    guided_rollout = []
    unguided_rollout = []

    init_time = plot_times[0]

    init_avg = avg_xr_over_mask(
        ground_truth_xr,
        var=var,
        timestamp=init_time,
        mask=mask,
        level=level,
    )

    ground_truth.append(init_avg)
    guided_rollout.append([init_avg] * len(guided_members))
    unguided_rollout.append([init_avg] * len(unguided_members))

    for timestamp_n in future_times:
        gt_avg = avg_xr_over_mask(
            ground_truth_xr,
            var=var,
            timestamp=timestamp_n,
            mask=mask,
            level=level,
        )
        ground_truth.append(gt_avg)

        guided_avgs = [
            avg_xr_over_mask(
                guided_xr,
                var=var,
                timestamp=timestamp_n,
                mask=mask,
                level=level,
                member=member,
            )
            for member in guided_members
        ]
        guided_rollout.append(guided_avgs)

        unguided_avgs = [
            avg_xr_over_mask(
                unguided_xr,
                var=var,
                timestamp=timestamp_n,
                mask=mask,
                level=level,
                member=member,
            )
            for member in unguided_members
        ]
        unguided_rollout.append(unguided_avgs)

    mean_guided_rollout = compute_mean_rollout(guided_rollout)
    mean_unguided_rollout = compute_mean_rollout(unguided_rollout)

    planned_guidance = guided_cfg.get("y", None)
    y_trajectory = guided_cfg.get("y_perc", None)

    # If y_perc is a scalar in newer configs, use y from config as right-axis trajectory if available.
    if not isinstance(y_trajectory, list):
        y_trajectory = guided_cfg.get("y", [0.0] * len(ground_truth))
    return (
        ground_truth,
        guided_rollout,
        mean_guided_rollout,
        mean_unguided_rollout,
        planned_guidance,
        unguided_rollout,
        y_trajectory,
    )


@app.cell
def _(
    ground_truth,
    mean_unguided_rollout,
    planned_guidance,
    plot_dual_trajectory,
    timestamps,
    unguided_rollout,
    var,
    y_trajectory,
):
    unguided_trajectory_plot = plot_dual_trajectory(
        timestamps=timestamps,
        var=var,
        mean_rollout=mean_unguided_rollout,
        ground_truth=ground_truth,
        planned_guidance=planned_guidance,
        y_trajectory=y_trajectory,
        ensemble_rollout=unguided_rollout,
        ymin_left=None,
        ymax_left=None,
        figsize=(12, 4),
    )
    return (unguided_trajectory_plot,)


@app.cell
def _(unguided_trajectory_plot):
    unguided_trajectory_plot
    return


@app.cell
def _(
    ground_truth,
    guided_rollout,
    mean_guided_rollout,
    planned_guidance,
    plot_dual_trajectory,
    timestamps,
    var,
    y_trajectory,
):
    guided_trajectory_plot = plot_dual_trajectory(
        timestamps=timestamps,
        var=var,
        mean_rollout=mean_guided_rollout,
        ground_truth=ground_truth,
        planned_guidance=planned_guidance,
        y_trajectory=y_trajectory,
        ensemble_rollout=guided_rollout,
        ymin_left=None,
        ymax_left=None,
        figsize=(12, 4),
    )
    return (guided_trajectory_plot,)


@app.cell
def _(guided_trajectory_plot):
    guided_trajectory_plot
    return


@app.cell
def _(
    ground_truth,
    mean_guided_rollout,
    mean_unguided_rollout,
    timestamps,
    var,
    visualize_mask_terms_over_N,
):
    mean_comparison_plot = visualize_mask_terms_over_N(
        var=var,
        timestamps=timestamps,
        mean_rollout=mean_guided_rollout,
        ground_truth=ground_truth,
        planned_guidance=mean_unguided_rollout,
        title=f"Guided mean vs unguided mean — {var}",
        subtitle="orange curve is unguided mean, purple curve is guided mean",
    )
    return (mean_comparison_plot,)


@app.cell
def _(mean_comparison_plot):
    mean_comparison_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Inspect states
    """)
    return


@app.cell
def _(
    analysis_type,
    current_slice,
    gt_current,
    gt_slice,
    guided_current,
    guided_gt,
    guided_slice,
    guided_unguided,
    mask,
    np,
    show_mask,
    show_values,
    unguided_current,
    unguided_gt,
    unguided_slice,
    value_threshold,
    visualize_map,
    zoom,
    zoom_center_lat,
    zoom_center_lon,
):
    mask_np = np.asarray(mask)

    if analysis_type == "absolute":
        panels = [
            ("current", current_slice),
            ("ground truth", gt_slice),
            ("unguided", unguided_slice),
            ("guided", guided_slice),
        ]

        vmin = min(float(np.nanmin(arr)) for _, arr in panels)
        vmax = max(float(np.nanmax(arr)) for _, arr in panels)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        center = 0.5 * (vmin + vmax)

        maps = {
            label: visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=vmin,
                vmax=vmax,
                center=center,
                show_mask=show_mask,
                zoom=zoom,
                zoom_center_lon=zoom_center_lon,
                zoom_center_lat=zoom_center_lat,
            )
            for label, arr in panels
        }

        current_map = maps["current"]
        gt_map = maps["ground truth"]
        unguided_map = maps["unguided"]
        guided_map = maps["guided"]

        gt_current_map = None
        guided_current_map = None
        unguided_current_map = None
        guided_gt_map = None
        unguided_gt_map = None
        guided_unguided_map = None

    else:
        panels = [
            ("gt-current", gt_current),
            ("unguided-current", unguided_current),
            ("guided-current", guided_current),
            ("unguided-gt", unguided_gt),
            ("guided-gt", guided_gt),
            ("guided-unguided", guided_unguided),
        ]

        absmax = max(float(np.nanmax(np.abs(arr))) for _, arr in panels)
        if absmax <= 0:
            absmax = 1e-8

        maps = {
            label: visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=-absmax,
                vmax=absmax,
                center=0.0,
                show_mask=show_mask,
                zoom=zoom,
                zoom_center_lon=zoom_center_lon,
                zoom_center_lat=zoom_center_lat,
                show_values=show_values,
                value_threshold=value_threshold,
                value_fontsize=5,
            )
            for label, arr in panels
        }

        current_map = None
        gt_map = None
        unguided_map = None
        guided_map = None

        gt_current_map = maps["gt-current"]
        unguided_current_map = maps["unguided-current"]
        guided_current_map = maps["guided-current"]
        unguided_gt_map = maps["unguided-gt"]
        guided_gt_map = maps["guided-gt"]
        guided_unguided_map = maps["guided-unguided"]
    return (
        current_map,
        gt_current_map,
        gt_map,
        guided_current_map,
        guided_gt_map,
        guided_map,
        guided_unguided_map,
        unguided_current_map,
        unguided_gt_map,
        unguided_map,
    )


@app.cell
def _(
    analysis_type,
    analysis_type_dropdown,
    current_map,
    gt_current_map,
    gt_map,
    guided_current_map,
    guided_gt_map,
    guided_map,
    guided_unguided_map,
    level,
    level_slider,
    m_slider,
    mo,
    n_slider,
    partition_dropdown,
    show_mask_switch,
    show_values_checkbox,
    unguided_current_map,
    unguided_gt_map,
    unguided_map,
    value_threshold_slider,
    var_dropdown,
    zoom_slider,
):
    controls = mo.vstack(
        [
            mo.hstack(
                [
                    analysis_type_dropdown,
                    n_slider,
                    m_slider,
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    partition_dropdown,
                    var_dropdown,
                    mo.hstack([level_slider, mo.md(f"{level}")], justify="start"),
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    show_mask_switch,
                    show_values_checkbox,
                    value_threshold_slider,
                    zoom_slider,
                ],
                justify="start",
            ),
        ]
    )

    if analysis_type == "absolute":
        display = mo.vstack(
            [
                controls,
                mo.hstack([current_map, gt_map]),
                mo.hstack([unguided_map, guided_map]),
            ]
        )
    else:
        display = mo.vstack(
            [
                controls,
                mo.hstack([gt_current_map, guided_unguided_map]),
                mo.hstack([unguided_gt_map, guided_gt_map]),
                mo.hstack([unguided_current_map, guided_current_map]),
            ]
        )

    display
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Realized guidance
    """)
    return


@app.cell
def _(guided_rollout_dir, read_json):
    def _parse_mask_terms(mask_terms_raw, member_key="1"):
        """
        Expected saved structure:
            {
              "1": [init_mask_term, [T terms for n=1], [T terms for n=2], ...],
              "2": ...
            }

        Returns:
            final_terms: [init, final_n1, final_n2, ...]
            all_terms: [[T terms n1], [T terms n2], ...]
        """
        if member_key not in mask_terms_raw:
            member_key = sorted(mask_terms_raw.keys())[0]

        values = mask_terms_raw[member_key]

        if not values:
            return [], []

        init = float(values[0])
        all_terms = values[1:]

        final_terms = [init]
        for terms_n in all_terms:
            if isinstance(terms_n, list):
                final_terms.append(float(terms_n[-1]))
            else:
                final_terms.append(float(terms_n))

        return final_terms, all_terms

    try:
        mask_terms_raw = read_json(guided_rollout_dir, "mask_terms")
    except FileNotFoundError:
        mask_terms_raw = {}
    return (mask_terms_raw,)


@app.cell
def _(m_idx, mask_terms_raw):
    member_key = str(m_idx + 1)
    realized_terms, all_mask_terms = _parse_mask_terms(mask_terms_raw, member_key)
    return all_mask_terms, realized_terms


@app.cell
def _(mo):
    mo.md(r"""
    ## Realized final mask term
    """)
    return


@app.cell
def _(
    ground_truth,
    planned_guidance,
    realized_terms,
    timestamps,
    var,
    visualize_mask_terms_over_N,
):
    if realized_terms:
        realized_guidance_plot = visualize_mask_terms_over_N(
            var=var,
            timestamps=timestamps,
            mean_rollout=realized_terms,
            ground_truth=ground_truth,
            planned_guidance=planned_guidance,
            ensemble_rollout=None,
            title="Realized guidance",
            subtitle="purple = realized mask term; orange = planned guidance",
        )
    else:
        realized_guidance_plot = None
    return (realized_guidance_plot,)


@app.cell
def _(realized_guidance_plot):
    realized_guidance_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inner diffusion-time mask terms
    """)
    return


@app.cell
def _(all_mask_terms, plt):
    def plot_inner_mask_terms(all_mask_terms):
        if not all_mask_terms:
            return None

        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

        for n, terms_n in enumerate(all_mask_terms, start=1):
            y = [float(v) for v in terms_n]
            x = list(range(len(y)))
            ax.plot(x, y, label=f"n={n}", linewidth=1.2)

        ax.set_xlabel("diffusion step")
        ax.set_ylabel("mask term")
        ax.set_title("Mask term over diffusion steps")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        return fig

    inner_mask_terms_plot = plot_inner_mask_terms(all_mask_terms)
    return (inner_mask_terms_plot,)


@app.cell
def _(inner_mask_terms_plot):
    inner_mask_terms_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    # RMSE analysis

    This uses the new xarray files directly. RMSE is computed over the selected mask
    for the currently selected variable and level.
    """)
    return


@app.cell
def _(
    future_times,
    ground_truth_xr,
    guided_members,
    guided_xr,
    level,
    mask,
    np,
    select_da,
    torch,
    unguided_members,
    unguided_xr,
    var,
):
    def rmse_da_on_mask(pred_da, gt_da, mask):
        pred = torch.tensor(np.asarray(pred_da.values), dtype=mask.dtype)
        gt = torch.tensor(np.asarray(gt_da.values), dtype=mask.dtype)

        pred = pred.squeeze()
        gt = gt.squeeze()

        diff2 = (pred - gt) ** 2
        return float(torch.sqrt((diff2 * mask).sum() / mask.sum()).item())

    rmse_guided = []
    rmse_unguided = []

    M_eff = min(len(guided_members), len(unguided_members))

    for timestamp_n in future_times:
        gt_da_n = select_da(
            ground_truth_xr,
            var=var,
            timestamp=timestamp_n,
            level=level,
        )

        guided_m = []
        unguided_m = []

        for i in range(M_eff):
            g_da = select_da(
                guided_xr,
                var=var,
                timestamp=timestamp_n,
                level=level,
                member=guided_members[i],
            )
            u_da = select_da(
                unguided_xr,
                var=var,
                timestamp=timestamp_n,
                level=level,
                member=unguided_members[i],
            )

            guided_m.append(rmse_da_on_mask(g_da, gt_da_n, mask))
            unguided_m.append(rmse_da_on_mask(u_da, gt_da_n, mask))

        rmse_guided.append(guided_m)
        rmse_unguided.append(unguided_m)

    rmse_guided_arr = np.asarray(rmse_guided, dtype=float)
    rmse_unguided_arr = np.asarray(rmse_unguided, dtype=float)

    rmse_guided_mean = rmse_guided_arr.mean(axis=1)
    rmse_unguided_mean = rmse_unguided_arr.mean(axis=1)

    rmse_guided_std = rmse_guided_arr.std(axis=1)
    rmse_unguided_std = rmse_unguided_arr.std(axis=1)
    return (
        rmse_guided_mean,
        rmse_guided_std,
        rmse_unguided_mean,
        rmse_unguided_std,
    )


@app.cell
def _(
    future_timestamps,
    plt,
    rmse_guided_mean,
    rmse_guided_std,
    rmse_unguided_mean,
    rmse_unguided_std,
    var,
):
    x = list(range(1, len(future_timestamps) + 1))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    ax.plot(x, rmse_unguided_mean, label="unguided", linewidth=1.8)
    ax.fill_between(
        x,
        rmse_unguided_mean - rmse_unguided_std,
        rmse_unguided_mean + rmse_unguided_std,
        alpha=0.18,
    )

    ax.plot(x, rmse_guided_mean, label="guided", linewidth=1.8)
    ax.fill_between(
        x,
        rmse_guided_mean - rmse_guided_std,
        rmse_guided_mean + rmse_guided_std,
        alpha=0.18,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(future_timestamps, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("RMSE on mask")
    ax.set_title(f"Mask RMSE — {var}")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper right", frameon=False)

    rmse_plot = fig
    return (rmse_plot,)


@app.cell
def _(rmse_plot):
    rmse_plot
    return


if __name__ == "__main__":
    app.run()
