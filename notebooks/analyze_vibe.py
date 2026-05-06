import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr

    return mo, np, pd, plt, xr


@app.cell
def _():
    from src.interaction import (
        visualize_map,
        get_mask_from_corners,
        get_mask_center,
        plot_rmse_over_n,
        plot_variable_change_parallel,
        plot_trajectory,
        plot_trajectories_over_n,
        plot_states_over_n,
    )

    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.funcs import get_guidance

    from src.utils import (
        read_nc,
        read_json,
        get_dataset,
        get_slice,
        get_rollout_dir,
        get_experiment_ids,
    )

    from src.paths import ROLLOUTS

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        get_dataset,
        get_experiment_ids,
        get_guidance,
        get_mask_center,
        get_mask_from_corners,
        get_rollout_dir,
        plot_rmse_over_n,
        plot_trajectories_over_n,
        plot_variable_change_parallel,
        read_json,
        read_nc,
        visualize_map,
    )


@app.cell
def _(np, pd):
    def format_time_value(t) -> str:
        return pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")


    def get_time_values(xr_ds) -> list:
        return list(xr_ds.time.values)


    def clean_coord_value(value):
        if isinstance(value, np.generic):
            return value.item()

        return value


    def get_member_values(xr_ds) -> list:
        if xr_ds is None:
            return [None]

        if "member" in xr_ds.dims:
            return [clean_coord_value(val) for val in xr_ds.member.values]

        return [None]


    def member_label(member) -> str:
        if member is None:
            return "single"

        return str(member)


    def as_numpy_2d(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()

        return np.asarray(x)


    def select_xr_field(xr_ds, var, timestamp, level=None, member=None):
        da = xr_ds[var].sel(time=timestamp)

        if member is not None and "member" in da.dims:
            da = da.sel(member=member)

        if "level" in da.dims and level is not None:
            da = da.sel(level=int(level))

        return da


    def xr_field_to_array(xr_ds, var, timestamp, level=None, member=None):
        da = select_xr_field(
            xr_ds=xr_ds,
            var=var,
            timestamp=timestamp,
            level=level,
            member=member,
        )

        return np.asarray(da.values)


    def masked_mean_xr(xr_ds, var, timestamp, mask, level=None, member=None):
        arr = xr_field_to_array(
            xr_ds=xr_ds,
            var=var,
            timestamp=timestamp,
            level=level,
            member=member,
        )

        mask_np = as_numpy_2d(mask).astype(bool)
        vals = arr[mask_np]

        if vals.size == 0:
            return float("nan")

        return float(vals.mean())


    def build_mask_rollouts(
        ground_truth_xr,
        unguided_xr,
        guided_xr,
        var,
        level,
        mask,
    ):
        """
        Builds trajectories with the phase-2 convention:

        ground_truth:
            length N + 1

        unguided_rollout:
            length N + 1, each entry is a list over members

        guided_rollout:
            length N + 1, each entry is a list over members

        The first entry is the initial ground-truth mask average, repeated over
        members, because guided.nc and unguided.nc normally start at the first
        future forecast time.
        """
        gt_times = get_time_values(ground_truth_xr)
        unguided_times = get_time_values(unguided_xr)
        guided_times = get_time_values(guided_xr)

        unguided_members = get_member_values(unguided_xr)
        guided_members = get_member_values(guided_xr)

        init_time = gt_times[0]

        init_avg = masked_mean_xr(
            ground_truth_xr,
            var=var,
            timestamp=init_time,
            mask=mask,
            level=level,
        )

        ground_truth_terms = [init_avg]
        unguided_rollout_terms = [[init_avg] * len(unguided_members)]
        guided_rollout_terms = [[init_avg] * len(guided_members)]

        N_eff = min(
            len(unguided_times),
            len(guided_times),
            len(gt_times) - 1,
        )

        for idx in range(N_eff):
            gt_time = gt_times[idx + 1]
            unguided_time = unguided_times[idx]
            guided_time = guided_times[idx]

            gt_avg = masked_mean_xr(
                ground_truth_xr,
                var=var,
                timestamp=gt_time,
                mask=mask,
                level=level,
            )

            ground_truth_terms.append(gt_avg)

            unguided_member_terms = [
                masked_mean_xr(
                    unguided_xr,
                    var=var,
                    timestamp=unguided_time,
                    mask=mask,
                    level=level,
                    member=member,
                )
                for member in unguided_members
            ]

            guided_member_terms = [
                masked_mean_xr(
                    guided_xr,
                    var=var,
                    timestamp=guided_time,
                    mask=mask,
                    level=level,
                    member=member,
                )
                for member in guided_members
            ]

            unguided_rollout_terms.append(unguided_member_terms)
            guided_rollout_terms.append(guided_member_terms)

        timestamps = [format_time_value(t) for t in gt_times[: len(ground_truth_terms)]]

        return (
            timestamps,
            ground_truth_terms,
            unguided_rollout_terms,
            guided_rollout_terms,
        )


    def mean_rollout_terms(rollout_terms):
        return [float(np.mean(row)) for row in rollout_terms]


    def member_rollout_terms(rollout_terms, member_index):
        return [float(row[member_index]) for row in rollout_terms]


    def align_series_length(values, target_length, prepend_value=None):
        if values is None:
            return [float("nan")] * target_length

        values = list(values)

        if len(values) == target_length:
            return values

        if len(values) == target_length - 1 and prepend_value is not None:
            return [prepend_value] + values

        if len(values) > target_length:
            return values[:target_length]

        if len(values) < target_length:
            pad_value = values[-1] if len(values) > 0 else float("nan")
            return values + [pad_value] * (target_length - len(values))


    return (
        align_series_length,
        build_mask_rollouts,
        format_time_value,
        get_member_values,
        mean_rollout_terms,
        member_rollout_terms,
        xr_field_to_array,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # PHASE 3 - results analysis
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
def _(get_experiment_ids, mo, refresh_button):
    if refresh_button.value:
        pass

    unguided_rollouts = get_experiment_ids("unguided")
    print(unguided_rollouts)
    pick_unguided_rollout_dropdown = mo.ui.dropdown(
        label="Experiment: ", value=unguided_rollouts[0], options=unguided_rollouts
    )

    pick_rollout_dropdown = mo.ui.dropdown(
        options=unguided_rollouts,
        value=unguided_rollouts[0],
        label="Pick rollout: ",
    )
    return pick_rollout_dropdown, pick_unguided_rollout_dropdown


@app.cell
def _(get_rollout_dir, pick_unguided_rollout_dropdown, read_json):
    rollout_dir = get_rollout_dir(pick_unguided_rollout_dropdown.value)
    unguided_config = read_json(rollout_dir, "config")
    return (rollout_dir,)


@app.cell
def _(read_nc, rollout_dir):
    unguided_xr = read_nc(rollout_dir, "unguided")
    ground_truth_xr = read_nc(rollout_dir, "ground_truth")
    return ground_truth_xr, unguided_xr


@app.cell
def _(config, mo, pick_rollout_dropdown, refresh_button):
    experiment_picker = mo.vstack(
        [
            mo.hstack(
                [
                    pick_rollout_dropdown,
                    refresh_button,
                ],
                justify="start",
            ),
            mo.accordion(
                {
                    "Experiment params": mo.md(
                        "<br>".join(f"{k}: {v}" for k, v in config.items())
                    ),
                }
            ),
        ]
    )
    return (experiment_picker,)


@app.cell
def _(mo, read_json, rollout_dir):
    experiment_params = read_json(rollout_dir, "experiment_params")

    guidance_mode_dropdown = mo.ui.dropdown(
        options=experiment_params["guidance_mode"],
        value=experiment_params["guidance_mode"][0],
        label="guidance mode",
    )

    alpha_slider = mo.ui.slider(
        start=min(experiment_params["alpha"]),
        stop=max(experiment_params["alpha"]),
        step=experiment_params["alpha"][1] - experiment_params["alpha"][0]
        if len(experiment_params["alpha"]) > 1
        else 1,
        value=experiment_params["alpha"][0],
        label="alpha",
    )

    w_slider = mo.ui.slider(
        start=min(experiment_params["w"]),
        stop=max(experiment_params["w"]),
        step=experiment_params["w"][1] - experiment_params["w"][0]
        if len(experiment_params["w"]) > 1
        else 1,
        value=experiment_params["w"][0],
        label="w",
    )

    mo.vstack([
        guidance_mode_dropdown,
        alpha_slider,
        w_slider,
    ])
    return alpha_slider, guidance_mode_dropdown, w_slider


@app.cell
def _(alpha_slider, guidance_mode_dropdown, rollout_dir, w_slider, xr):
    params = {
        "guidance_mode": guidance_mode_dropdown.value,
        "alpha": alpha_slider.value,
        "w": w_slider.value,
    }

    from src.utils import make_hash
    guided_id = make_hash(params)
    guided_xr = xr.open_dataset(rollout_dir / "guided" / guided_id / "guided.nc")
    return (guided_xr,)


@app.cell
def _(experiment_picker):
    experiment_picker
    return


@app.cell
def _(PARTITIONS, config, mo):
    partition_default = config.get("partition", PARTITIONS[0])

    if partition_default not in PARTITIONS:
        partition_default = PARTITIONS[0]

    partition_dropdown = mo.ui.dropdown(
        PARTITIONS,
        value=partition_default,
        label="partition: ",
    )
    return (partition_dropdown,)


@app.cell
def _(partition_dropdown):
    partition = partition_dropdown.value
    return (partition,)


@app.cell
def _(LEVELS_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition]
    level_slider = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level  ", show_value=True, debounce=True)
    return (level_slider,)


@app.cell
def _(level_slider):
    level = level_slider.value
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition):
    VARIABLES = VARIABLES_DICT[partition]
    if partition == "surface":
        variable_default = VARIABLES[2]
    else:
        variable_default = VARIABLES[3]
    var_dropdown = mo.ui.dropdown(VARIABLES, value=variable_default, label="variable : ")
    return (var_dropdown,)


@app.cell
def _(var_dropdown):
    var = var_dropdown.value
    return (var,)


@app.cell
def _(config, mo):
    N = config["N"]
    n_slider = mo.ui.slider(
        steps=range(1, N + 1),
        value=1,
        label="n: ",
        show_value=True,
        debounce=True,
    )
    return N, n_slider


@app.cell
def _(n_slider):
    n = n_slider.value
    return (n,)


@app.cell
def _(config, get_member_values, guided_xr, mo, unguided_xr):
    M = config["M"]
    member_values = get_member_values(unguided_xr)
    guided_member_values = get_member_values(guided_xr)

    m_slider = mo.ui.slider(
        steps=member_values,
        value=member_values[0],
        label="m: ",
        show_value=True,
        debounce=True,
    )
    return guided_member_values, m_slider, member_values


@app.cell
def _(m_slider):
    m = m_slider.value
    return (m,)


@app.cell
def _(guided_member_values, m, member_values):
    member_index = member_values.index(m)

    if len(guided_member_values) == len(member_values):
        guided_member = guided_member_values[member_index]
    elif len(guided_member_values) == 1:
        guided_member = guided_member_values[0]
    else:
        guided_member = guided_member_values[min(member_index, len(guided_member_values) - 1)]

    unguided_member = m
    return guided_member, member_index, unguided_member


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
    show_mask_switch = mo.ui.checkbox(label="show mask", value=True)
    return (show_mask_switch,)


@app.cell
def _(show_mask_switch):
    show_mask = show_mask_switch.value
    return (show_mask,)


@app.cell
def _(mo):
    analysis_types = ["absolute", "difference"]

    analysis_type_dropdown = mo.ui.dropdown(
        analysis_types,
        value=analysis_types[0],
        label="analysis type: ",
    )
    return (analysis_type_dropdown,)


@app.cell
def _(analysis_type_dropdown):
    analysis_type = analysis_type_dropdown.value
    return (analysis_type,)


@app.cell
def _(mo):
    show_values_checkbox = mo.ui.checkbox(label="show values")
    return (show_values_checkbox,)


@app.cell
def _(show_values_checkbox):
    show_values = show_values_checkbox.value
    return (show_values,)


@app.cell
def _(config, get_mask_from_corners):
    mask_corners = tuple(config["mask_corners"])
    mask = get_mask_from_corners(*mask_corners)
    return (mask,)


@app.cell
def _(get_mask_center, mask):
    zoom_centers = get_mask_center(mask)
    return (zoom_centers,)


@app.cell
def _(
    align_series_length,
    build_mask_rollouts,
    config,
    ground_truth_xr,
    guided_xr,
    level,
    mask,
    mean_rollout_terms,
    member_index,
    member_rollout_terms,
    unguided_xr,
    var,
):
    (
        timestamps,
        ground_truth_terms,
        unguided_rollout_terms,
        guided_rollout_terms,
    ) = build_mask_rollouts(
        ground_truth_xr=ground_truth_xr,
        unguided_xr=unguided_xr,
        guided_xr=guided_xr,
        var=var,
        level=level,
        mask=mask,
    )

    mean_unguided_rollout_terms = mean_rollout_terms(unguided_rollout_terms)
    mean_guided_rollout_terms = mean_rollout_terms(guided_rollout_terms)

    selected_guided_terms = member_rollout_terms(
        guided_rollout_terms,
        member_index,
    )

    selected_unguided_terms = member_rollout_terms(
        unguided_rollout_terms,
        member_index,
    )

    planned_guidance = config.get("y", config.get("planned_guidance", None))

    planned_guidance = align_series_length(
        planned_guidance,
        target_length=len(timestamps),
        prepend_value=ground_truth_terms[0],
    )
    return (
        ground_truth_terms,
        mean_unguided_rollout_terms,
        planned_guidance,
        selected_guided_terms,
        timestamps,
        unguided_rollout_terms,
    )


@app.cell
def _(get_dataset):
    ds = get_dataset()
    return


@app.cell
def _(ground_truth_xr, guided_xr, n, unguided_xr):
    ground_truth_times = list(ground_truth_xr.time.values)
    guided_times = list(guided_xr.time.values)
    unguided_times = list(unguided_xr.time.values)

    current_time = ground_truth_times[n - 1]
    ground_truth_time = ground_truth_times[n]
    guided_time = guided_times[n - 1]
    unguided_time = unguided_times[n - 1]
    return current_time, ground_truth_time, guided_time, unguided_time


@app.cell
def _(
    current_time,
    ground_truth_time,
    ground_truth_xr,
    guided_member,
    guided_time,
    guided_xr,
    level,
    unguided_member,
    unguided_time,
    unguided_xr,
    var,
    xr_field_to_array,
):
    current_slice = xr_field_to_array(
        ground_truth_xr,
        var=var,
        timestamp=current_time,
        level=level,
    )

    next_slice = xr_field_to_array(
        ground_truth_xr,
        var=var,
        timestamp=ground_truth_time,
        level=level,
    )

    unguided_slice = xr_field_to_array(
        unguided_xr,
        var=var,
        timestamp=unguided_time,
        level=level,
        member=unguided_member,
    )

    guided_slice = xr_field_to_array(
        guided_xr,
        var=var,
        timestamp=guided_time,
        level=level,
        member=guided_member,
    )

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


@app.cell
def _(guided_unguided, mo, np):
    # text thresh
    center_value_for_threshold = 0.0

    max_abs_diff_for_threshold = float(
        max(
            np.nanmax(guided_unguided),
            abs(np.nanmin(guided_unguided)),
        )
    )

    if max_abs_diff_for_threshold <= 0:
        max_abs_diff_for_threshold = 1e-8

    default_value_threshold = max_abs_diff_for_threshold * 0.9

    value_threshold_slider = mo.ui.slider(
        start=center_value_for_threshold,
        stop=max_abs_diff_for_threshold,
        step=max_abs_diff_for_threshold / 20,
        value=default_value_threshold,
        label="text thresh: ",
    )
    return (value_threshold_slider,)


@app.cell
def _(value_threshold_slider):
    value_threshold = value_threshold_slider.value
    return (value_threshold,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Realized guidance

    This compares the planned guidance trajectory from the config with the
    realized mask averages measured directly from `guided.nc`.
    """)
    return


@app.cell
def _(get_guidance, plt):
    def realized_guidance_branches(
        realized_terms: list[float],
        planned_guidance: list[float],
        mean_rollout: list[float],
    ) -> list[float]:
        if (
            len(realized_terms) != len(planned_guidance)
            or len(realized_terms) != len(mean_rollout)
        ):
            raise ValueError(
                "realized_terms, planned_guidance, and mean_rollout must have the same length"
            )

        branches = [realized_terms[0]]

        for idx in range(1, len(realized_terms)):
            mean_value = mean_rollout[idx]
            abs_mean_value = abs(mean_value) if mean_value != 0 else 1.0
            guidance_delta = (planned_guidance[idx] - mean_value) / abs_mean_value

            branches.append(
                get_guidance(
                    guidance_delta,
                    realized_terms[idx - 1],
                )
            )

        return branches


    def plot_guidance_branching(
        timestamps: list[str],
        realized_terms: list[float],
        planned_guidance: list[float],
        mean_rollout: list[float],
        ground_truth: list[float] | None = None,
        ensemble_rollout: list[list[float]] | None = None,
        title: str | None = "Realized guidance",
        subtitle: str | None = None,
    ):
        num_steps = len(timestamps)

        if (
            num_steps != len(realized_terms)
            or len(realized_terms) != len(planned_guidance)
        ):
            raise ValueError(
                "timestamps, realized_terms, and planned_guidance must have the same length"
            )

        x = list(range(num_steps))

        branch_targets = realized_guidance_branches(
            realized_terms=realized_terms,
            planned_guidance=planned_guidance,
            mean_rollout=mean_rollout,
        )

        colors = {
            "realized": "#1f77b4",
            "online": "#d62728",
            "offline": "#ff7f0e",
            "gt": "#2ca02c",
            "mean": "#9467bd",
            "ensemble": "#7f7f7f",
        }

        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

        if ensemble_rollout is not None:
            num_members = len(ensemble_rollout[0])

            for member_idx in range(num_members):
                y = [
                    ensemble_rollout[step_idx][member_idx]
                    for step_idx in range(num_steps)
                ]

                ax.plot(
                    x,
                    y,
                    "-",
                    color=colors["ensemble"],
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
                color=colors["ensemble"],
                alpha=0.12,
                label=f"Unguided ensemble range (M={num_members})",
                zorder=1,
            )

        for step_idx in range(1, num_steps):
            ax.plot(
                [x[step_idx - 1], x[step_idx]],
                [realized_terms[step_idx - 1], branch_targets[step_idx]],
                linestyle="--",
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=colors["online"],
                alpha=0.85,
                zorder=2,
                label="Online planned" if step_idx == 1 else None,
            )

        ax.plot(
            x,
            planned_guidance,
            "-",
            marker="s",
            markersize=3.5,
            linewidth=1.6,
            color=colors["offline"],
            alpha=0.9,
            label="Offline planned",
            zorder=3,
        )

        ax.plot(
            x,
            mean_rollout,
            "-",
            marker="D",
            markersize=3.5,
            linewidth=1.6,
            color=colors["mean"],
            alpha=0.9,
            label="Mean unguided rollout",
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
                color=colors["gt"],
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
            color=colors["realized"],
            label="Realized guided",
            zorder=5,
        )

        tick_idx = [
            idx for idx, ts in enumerate(timestamps) if ts.endswith("00:00:00")
        ]

        if 0 not in tick_idx:
            tick_idx = [0] + tick_idx

        if num_steps - 1 not in tick_idx:
            tick_idx.append(num_steps - 1)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [timestamps[idx] for idx in tick_idx],
            rotation=35,
            ha="right",
            fontsize=8,
        )

        ax.set_xlabel("Timestamp", fontsize=10)
        ax.set_ylabel("Mask term", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle=":")

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
def _(
    ground_truth_terms,
    mean_unguided_rollout_terms,
    planned_guidance,
    plot_guidance_branching,
    selected_guided_terms,
    timestamps,
    unguided_rollout_terms,
):
    realized_guidance_plot = plot_guidance_branching(
        timestamps=timestamps,
        realized_terms=selected_guided_terms,
        planned_guidance=planned_guidance,
        mean_rollout=mean_unguided_rollout_terms,
        ground_truth=ground_truth_terms,
        ensemble_rollout=unguided_rollout_terms,
        title="Realized guidance analysis",
        subtitle="mask term along the rollout",
    )
    return (realized_guidance_plot,)


@app.cell
def _(realized_guidance_plot):
    realized_guidance_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect states

    Analyze all relevant weather states interactively.
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
    def safe_abs_limits(arrays):
        vmin = min(float(np.nanmin(np.asarray(arr))) for arr in arrays)
        vmax = max(float(np.nanmax(np.asarray(arr))) for arr in arrays)

        if vmax <= vmin:
            vmax = vmin + 1e-9

        center = 0.5 * (vmin + vmax)
        center = min(max(center, vmin + 1e-9), vmax - 1e-9)

        return vmin, vmax, center


    def safe_diff_absmax(arrays):
        absmax = max(
            float(np.nanmax(np.abs(np.asarray(arr))))
            for arr in arrays
        )

        if absmax <= 0:
            absmax = 1e-8

        return absmax


    mask_np = np.asarray(mask)

    if analysis_type == "absolute":
        absolute_panels = [
            ("$x_t$", current_slice),
            ("$x_{t+1}$", next_slice),
            ("$x_{t+1}^{unguided}$", unguided_slice),
            ("$x_{t+1}^{guided}$", guided_slice),
        ]

        abs_vmin, abs_vmax, abs_center = safe_abs_limits(
            [arr for _, arr in absolute_panels]
        )

        absolute_maps = {}

        for label, arr in absolute_panels:
            absolute_maps[label] = visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=abs_vmin,
                vmax=abs_vmax,
                center=abs_center,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
            )

        state_map = absolute_maps["$x_t$"]
        next_map = absolute_maps["$x_{t+1}$"]
        unguided_map = absolute_maps["$x_{t+1}^{unguided}$"]
        guided_map = absolute_maps["$x_{t+1}^{guided}$"]

        next_current_map = None
        unguided_current_map = None
        guided_current_map = None
        unguided_gt_map = None
        guided_gt_map = None
        guided_unguided_map = None

    else:
        difference_panels = [
            ("$x_{t+1} - x_t$", next_current),
            ("$x_{t+1}^{unguided} - x_t$", unguided_current),
            ("$x_{t+1}^{guided} - x_t$", guided_current),
            ("$x_{t+1}^{unguided} - x_{t+1}$", unguided_gt),
            ("$x_{t+1}^{guided} - x_{t+1}$", guided_gt),
            ("$x_{t+1}^{guided} - x_{t+1}^{unguided}$", guided_unguided),
        ]

        diff_absmax = safe_diff_absmax(
            [arr for _, arr in difference_panels]
        )

        difference_maps = {}

        for label, arr in difference_panels:
            difference_maps[label] = visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=-diff_absmax,
                vmax=diff_absmax,
                center=0.0,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
                show_values=show_values,
                value_threshold=value_threshold,
                value_fontsize=5,
            )

        next_current_map = difference_maps["$x_{t+1} - x_t$"]
        unguided_current_map = difference_maps["$x_{t+1}^{unguided} - x_t$"]
        guided_current_map = difference_maps["$x_{t+1}^{guided} - x_t$"]
        unguided_gt_map = difference_maps["$x_{t+1}^{unguided} - x_{t+1}$"]
        guided_gt_map = difference_maps["$x_{t+1}^{guided} - x_{t+1}$"]
        guided_unguided_map = difference_maps[
            "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"
        ]

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
def _(
    analysis_type,
    analysis_type_dropdown,
    current_time,
    format_time_value,
    ground_truth_time,
    guided_current_map,
    guided_gt_map,
    guided_map,
    guided_time,
    guided_unguided_map,
    level,
    level_slider,
    m_slider,
    mo,
    n_slider,
    next_current_map,
    next_map,
    partition_dropdown,
    show_mask_switch,
    show_values_checkbox,
    state_map,
    unguided_current_map,
    unguided_gt_map,
    unguided_map,
    unguided_time,
    value_threshold_slider,
    var_dropdown,
    zoom_slider,
):
    time_info = mo.accordion(
        {
            "Selected times": mo.md(
                "<br>".join(
                    [
                        f"current / init: `{format_time_value(current_time)}`",
                        f"ground truth target: `{format_time_value(ground_truth_time)}`",
                        f"unguided forecast: `{format_time_value(unguided_time)}`",
                        f"guided forecast: `{format_time_value(guided_time)}`",
                    ]
                )
            )
        }
    )

    common_controls = [
        analysis_type_dropdown,
        mo.hstack([n_slider, m_slider], justify="start"),
        mo.hstack(
            [
                partition_dropdown,
                var_dropdown,
                mo.hstack(
                    [level_slider, mo.md(f"{level}")],
                    justify="start",
                ),
            ],
            justify="start",
        ),
        time_info,
    ]

    if analysis_type == "absolute":
        inspect_states_ui = mo.vstack(
            [
                *common_controls,
                mo.hstack([show_mask_switch], justify="start"),
                zoom_slider,
                mo.md("Absolute states:"),
                mo.hstack([state_map, next_map]),
                mo.hstack([unguided_map, guided_map]),
            ]
        )

    else:
        inspect_states_ui = mo.vstack(
            [
                *common_controls,
                mo.hstack(
                    [
                        show_mask_switch,
                        show_values_checkbox,
                        value_threshold_slider,
                    ],
                    justify="start",
                ),
                zoom_slider,
                mo.md("Difference over states:"),
                mo.hstack([next_current_map, guided_unguided_map]),
                mo.hstack([unguided_current_map, guided_current_map]),
                mo.hstack([unguided_gt_map, guided_gt_map]),
            ]
        )

    inspect_states_ui
    return


@app.cell
def _(current_slice, guided_slice, next_slice, np, unguided_slice):
    slice_shapes = {
        "current_slice": np.asarray(current_slice).shape,
        "next_slice": np.asarray(next_slice).shape,
        "unguided_slice": np.asarray(unguided_slice).shape,
        "guided_slice": np.asarray(guided_slice).shape,
    }
    return (slice_shapes,)


@app.cell
def _(mo, slice_shapes):
    mo.accordion(
        {
            "Slice shapes": mo.md(
                "<br>".join(f"{k}: `{v}`" for k, v in slice_shapes.items())
            )
        }
    )
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

    Per rollout step \(n = 1, \dots, N\), compare the selected guided and
    unguided ensemble member against the ERA5 ground truth.

    This version reads directly from:

    - `ground_truth.nc`
    - `unguided.nc`
    - `guided.nc`

    No old `n/m.nc` rollout folders are used anymore.
    """)
    return


@app.cell
def _(np, plt):
    def mask_bbox_with_padding(mask_2d, padding=4):
        mask_np = np.asarray(mask_2d).astype(bool)

        rows, cols = np.where(mask_np)

        if len(rows) == 0 or len(cols) == 0:
            return (
                0,
                mask_np.shape[0],
                0,
                mask_np.shape[1],
            )

        row_min = max(int(rows.min()) - padding, 0)
        row_max = min(int(rows.max()) + padding + 1, mask_np.shape[0])
        col_min = max(int(cols.min()) - padding, 0)
        col_max = min(int(cols.max()) + padding + 1, mask_np.shape[1])

        return row_min, row_max, col_min, col_max


    def crop_array_to_bbox(arr, bbox):
        row_min, row_max, col_min, col_max = bbox
        return np.asarray(arr)[row_min:row_max, col_min:col_max]


    def crop_mask_to_bbox(mask_2d, bbox):
        row_min, row_max, col_min, col_max = bbox
        return np.asarray(mask_2d).astype(bool)[row_min:row_max, col_min:col_max]


    def safe_abs_limits_for_row(arrays):
        vmin = min(float(np.nanmin(np.asarray(arr))) for arr in arrays)
        vmax = max(float(np.nanmax(np.asarray(arr))) for arr in arrays)

        if vmax <= vmin:
            vmax = vmin + 1e-9

        return vmin, vmax


    def safe_symmetric_limits_for_row(arrays):
        absmax = max(
            float(np.nanmax(np.abs(np.asarray(arr))))
            for arr in arrays
        )

        if absmax <= 0:
            absmax = 1e-8

        return -absmax, absmax


    def draw_mask_outline(ax, mask_2d):
        mask_np = np.asarray(mask_2d).astype(bool)

        if mask_np.any():
            ax.contour(
                mask_np.astype(float),
                levels=[0.5],
                linewidths=1.0,
            )


    def plot_states_over_n_from_xr(
        *,
        ground_truth_xr,
        unguided_xr,
        guided_xr,
        xr_field_to_array,
        N,
        var,
        level,
        mask_2d,
        unguided_member=None,
        guided_member=None,
        analysis_type="absolute",
        padding=4,
        figsize_per_row=(12, 2.5),
        title=None,
        subtitle=None,
    ):
        ground_truth_times = list(ground_truth_xr.time.values)
        unguided_times = list(unguided_xr.time.values)
        guided_times = list(guided_xr.time.values)

        N_eff = min(
            int(N),
            len(ground_truth_times) - 1,
            len(unguided_times),
            len(guided_times),
        )

        if N_eff <= 0:
            raise ValueError("No valid rollout steps found for plotting.")

        bbox = mask_bbox_with_padding(mask_2d, padding=padding)
        mask_crop = crop_mask_to_bbox(mask_2d, bbox)

        if analysis_type == "absolute":
            column_titles = [
                "$x_t$",
                "$x_{t+1}$",
                "$x_{t+1}^{unguided}$",
                "$x_{t+1}^{guided}$",
            ]
        else:
            column_titles = [
                "$x_{t+1} - x_t$",
                "$x_{t+1}^{unguided} - x_t$",
                "$x_{t+1}^{guided} - x_t$",
                "$x_{t+1}^{guided} - x_{t+1}^{unguided}$",
            ]

        fig, axes = plt.subplots(
            N_eff,
            len(column_titles),
            figsize=(figsize_per_row[0], figsize_per_row[1] * N_eff),
            squeeze=False,
            dpi=160,
        )

        for step_idx in range(N_eff):
            current_time_for_row = ground_truth_times[step_idx]
            target_time_for_row = ground_truth_times[step_idx + 1]
            unguided_time_for_row = unguided_times[step_idx]
            guided_time_for_row = guided_times[step_idx]

            current_arr = xr_field_to_array(
                ground_truth_xr,
                var=var,
                timestamp=current_time_for_row,
                level=level,
            )

            target_arr = xr_field_to_array(
                ground_truth_xr,
                var=var,
                timestamp=target_time_for_row,
                level=level,
            )

            unguided_arr = xr_field_to_array(
                unguided_xr,
                var=var,
                timestamp=unguided_time_for_row,
                level=level,
                member=unguided_member,
            )

            guided_arr = xr_field_to_array(
                guided_xr,
                var=var,
                timestamp=guided_time_for_row,
                level=level,
                member=guided_member,
            )

            if analysis_type == "absolute":
                row_arrays = [
                    current_arr,
                    target_arr,
                    unguided_arr,
                    guided_arr,
                ]
                vmin, vmax = safe_abs_limits_for_row(row_arrays)

            else:
                row_arrays = [
                    target_arr - current_arr,
                    unguided_arr - current_arr,
                    guided_arr - current_arr,
                    guided_arr - unguided_arr,
                ]
                vmin, vmax = safe_symmetric_limits_for_row(row_arrays)

            for col_idx, arr in enumerate(row_arrays):
                ax = axes[step_idx, col_idx]
                arr_crop = crop_array_to_bbox(arr, bbox)

                im = ax.imshow(
                    arr_crop,
                    vmin=vmin,
                    vmax=vmax,
                    origin="upper",
                    aspect="auto",
                )

                draw_mask_outline(ax, mask_crop)

                if step_idx == 0:
                    ax.set_title(column_titles[col_idx], fontsize=10)

                if col_idx == 0:
                    ax.set_ylabel(f"n={step_idx + 1}", fontsize=10)

                ax.set_xticks([])
                ax.set_yticks([])

                fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.02,
                )

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

        if subtitle:
            fig.text(
                0.5,
                0.975,
                subtitle,
                ha="center",
                va="top",
                fontsize=9,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.96 if title or subtitle else 1))

        return fig

    return (plot_states_over_n_from_xr,)


@app.cell
def _(
    N,
    analysis_type,
    ground_truth_xr,
    guided_member,
    guided_xr,
    level,
    m,
    mask,
    plot_states_over_n_from_xr,
    unguided_member,
    unguided_xr,
    var,
    xr_field_to_array,
):
    states_over_n_plot = plot_states_over_n_from_xr(
        ground_truth_xr=ground_truth_xr,
        unguided_xr=unguided_xr,
        guided_xr=guided_xr,
        xr_field_to_array=xr_field_to_array,
        N=N,
        var=var,
        level=level,
        mask_2d=mask,
        unguided_member=unguided_member,
        guided_member=guided_member,
        analysis_type=analysis_type,
        padding=4,
        title=f"States over N — {analysis_type}",
        subtitle=f"var={var} | level={level} | member={m}",
    )
    return (states_over_n_plot,)


@app.cell
def _(m_slider, mo, states_over_n_plot):
    mo.vstack([
        m_slider,
        states_over_n_plot
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Realized guidance during sampling

    If `mask_terms.json` exists in the rollout directory, this section shows
    the mask-term trajectory over diffusion time \(T\) for each rollout step \(n\).

    If the file does not exist, the notebook simply skips this plot.
    """)
    return


@app.cell
def _(read_json, rollout_dir):
    if rollout_dir is None:
        sampling_mask_terms = None

    elif (rollout_dir / "mask_terms.json").exists():
        sampling_mask_terms = read_json(rollout_dir, "mask_terms")

    else:
        sampling_mask_terms = None
    return (sampling_mask_terms,)


@app.cell
def _(mo, plot_trajectories_over_n, sampling_mask_terms):
    if sampling_mask_terms is None:
        sampling_mask_terms_view = mo.md(
            "`mask_terms.json` was not found for this rollout, so the diffusion-time guidance plot is skipped."
        )

    else:
        sampling_all_mask_terms = sampling_mask_terms.get("all_mask_terms", None)

        if sampling_all_mask_terms is None:
            sampling_mask_terms_view = mo.md(
                "`mask_terms.json` exists, but it does not contain `all_mask_terms`."
            )

        else:
            sampling_mask_terms_plot, _ = plot_trajectories_over_n(
                trajectories=sampling_all_mask_terms,
                var="mask term",
                title="Realized guidance over diffusion time",
            )

            sampling_mask_terms_view = sampling_mask_terms_plot
    return (sampling_mask_terms_view,)


@app.cell
def _(sampling_mask_terms_view):
    sampling_mask_terms_view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Variable change analysis

    For each rollout step \(n = 1, \dots, N\) and each variable-level key
    such as `u_component_of_wind-500` or `2m_temperature-surface`, we measure

    \[
    \sum_{i,j} |x^{guided}_{n,m,i,j} - x^{unguided}_{n,m,i,j}|
    \]

    directly from `guided.nc` and `unguided.nc`.

    Scores are min-max normalized **within each member and rollout step**, so
    the strongest changed channels at each step sit near 1.
    """)
    return


@app.cell
def _(np):
    def select_member_if_present(da, member=None):
        if member is not None and "member" in da.dims:
            return da.sel(member=member)

        return da


    def per_var_level_abs_sum_from_xr_diff(diff_xr):
        out = {}

        for var_name, da in diff_xr.data_vars.items():
            da_abs = abs(da)

            if "level" in da_abs.dims:
                for level_value in da_abs.level.values:
                    key = f"{var_name}-{int(level_value)}"
                    out[key] = float(
                        da_abs.sel(level=level_value).sum(skipna=True).item()
                    )

            else:
                key = f"{var_name}-surface"
                out[key] = float(da_abs.sum(skipna=True).item())

        return out


    def minmax_normalize_dict(d):
        vals = list(d.values())

        if len(vals) == 0:
            return {}

        vmin = min(vals)
        vmax = max(vals)

        if vmax == vmin:
            return {k: 0.0 for k in d}

        return {
            k: (v - vmin) / (vmax - vmin)
            for k, v in d.items()
        }


    def aggregate_dicts(dict_list, error="std"):
        if len(dict_list) == 0:
            return {
                "keys": [],
                "mean": np.array([]),
                "err": np.array([]),
                "values": np.array([]),
            }

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


    def build_variable_change_scores_from_xr(
        *,
        guided_xr,
        unguided_xr,
        member_values,
        guided_member_values,
        N,
    ):
        guided_times = list(guided_xr.time.values)
        unguided_times = list(unguided_xr.time.values)

        N_eff = min(
            int(N),
            len(guided_times),
            len(unguided_times),
        )

        minmax_analysis_list = []

        for step_idx in range(N_eff):
            guided_time = guided_times[step_idx]
            unguided_time = unguided_times[step_idx]

            per_member_scores = []

            for member_index, unguided_member in enumerate(member_values):
                if len(guided_member_values) == len(member_values):
                    guided_member = guided_member_values[member_index]
                elif len(guided_member_values) == 1:
                    guided_member = guided_member_values[0]
                else:
                    guided_member = guided_member_values[
                        min(member_index, len(guided_member_values) - 1)
                    ]

                guided_step = guided_xr.sel(time=guided_time)
                unguided_step = unguided_xr.sel(time=unguided_time)

                guided_step = guided_step.map(
                    lambda da: select_member_if_present(da, guided_member)
                )
                unguided_step = unguided_step.map(
                    lambda da: select_member_if_present(da, unguided_member)
                )

                diff_step = guided_step - unguided_step

                abs_sum_dict = per_var_level_abs_sum_from_xr_diff(diff_step)
                abs_sum_dict_minmax = minmax_normalize_dict(abs_sum_dict)

                per_member_scores.append(abs_sum_dict_minmax)

            minmax_analysis_list.append(per_member_scores)

        return minmax_analysis_list

    return (
        aggregate_dicts,
        build_variable_change_scores_from_xr,
        select_member_if_present,
    )


@app.cell
def _(mo):
    top_k_slider = mo.ui.slider(
        start=1,
        stop=50,
        value=12,
        step=1,
        label="top-k channels",
    )

    rank_by_radio = mo.ui.radio(
        options=["max", "mean"],
        value="max",
        label="rank by",
    )

    variable_change_controls = mo.hstack(
        [top_k_slider, rank_by_radio],
        justify="start",
    )
    return rank_by_radio, top_k_slider, variable_change_controls


@app.cell
def _(variable_change_controls):
    variable_change_controls
    return


@app.cell
def _(
    N,
    build_variable_change_scores_from_xr,
    guided_member_values,
    guided_xr,
    member_values,
    unguided_xr,
):
    minmax_analysis_list = build_variable_change_scores_from_xr(
        guided_xr=guided_xr,
        unguided_xr=unguided_xr,
        member_values=member_values,
        guided_member_values=guided_member_values,
        N=N,
    )
    return (minmax_analysis_list,)


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
    return (var_change_fig,)


@app.cell
def _(var_change_fig):
    var_change_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Variable change analysis, level-aggregated

    Same analysis as above, but pressure-level scores are summed per variable.
    Surface-only variables are dropped.
    """)
    return


@app.cell
def _(aggregate_dicts, minmax_analysis_list, plot_variable_change_parallel):
    def collapse_level_keys_to_vars(d):
        out = {}

        for key, val in d.items():
            var_name, suffix = key.rsplit("-", 1)

            if suffix == "surface":
                continue

            out[var_name] = out.get(var_name, 0.0) + val

        return out


    var_analysis_list = [
        [collapse_level_keys_to_vars(d) for d in dict_list]
        for dict_list in minmax_analysis_list
    ]

    aggregated_vars_per_n = [
        aggregate_dicts(dict_list, error="std")
        for dict_list in var_analysis_list
    ]

    var_agg_fig, _ = plot_variable_change_parallel(
        aggregated_vars_per_n,
        top_k=None,
        rank_by="max",
        title="Variable change across rollout steps, level-aggregated",
        subtitle="pressure-level keys summed per variable; error bars = ensemble std",
        ylim=None,
        ylabel="score, summed over levels",
    )
    return (var_agg_fig,)


@app.cell
def _(var_agg_fig):
    var_agg_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### RMSE analysis

    For each rollout step \(n = 1,\dots,N\) and each ensemble member \(m\),
    compute the RMSE of guided and unguided forecasts against the ERA5 ground
    truth.

    This version computes RMSE directly in the value space stored in the
    NetCDF files:

    - `guided.nc`
    - `unguided.nc`
    - `ground_truth.nc`

    Positive relative improvement means guidance helped:

    \[
    r =
    \frac{
        \mathrm{RMSE}_{unguided} - \mathrm{RMSE}_{guided}
    }{
        \mathrm{RMSE}_{unguided}
    }
    \]
    """)
    return


@app.cell
def _(np):
    def per_var_level_rmse_dict_on_mask_from_xr(diff_xr, mask_2d):
        mask_bool = np.asarray(mask_2d).astype(bool)
        out = {}

        for var_name, da in diff_xr.data_vars.items():
            da_squeezed = da.squeeze(drop=True)

            if "level" in da_squeezed.dims:
                for level_value in da_squeezed.level.values:
                    arr = np.asarray(
                        da_squeezed.sel(level=level_value).values
                    )
                    vals = arr[mask_bool]
                    out[f"{var_name}-{int(level_value)}"] = float(
                        np.sqrt(np.nanmean(vals**2))
                    )

            else:
                arr = np.asarray(da_squeezed.values)
                vals = arr[mask_bool]
                out[f"{var_name}-surface"] = float(
                    np.sqrt(np.nanmean(vals**2))
                )

        return out


    def per_var_rmse_dict_on_mask_from_xr(
        diff_xr,
        mask_2d,
        drop_surface=True,
    ):
        mask_bool = np.asarray(mask_2d).astype(bool)
        out = {}

        for var_name, da in diff_xr.data_vars.items():
            da_squeezed = da.squeeze(drop=True)
            arr = np.asarray(da_squeezed.values)

            if "level" in da_squeezed.dims:
                vals = arr[..., mask_bool].ravel()
                out[var_name] = float(np.sqrt(np.nanmean(vals**2)))

            elif not drop_surface:
                vals = arr[mask_bool]
                out[f"{var_name}-surface"] = float(
                    np.sqrt(np.nanmean(vals**2))
                )

        return out


    def total_rmse_on_mask_from_xr(diff_xr, mask_2d):
        mask_bool = np.asarray(mask_2d).astype(bool)
        accum = []

        for _, da in diff_xr.data_vars.items():
            arr = np.asarray(da.squeeze(drop=True).values)

            if arr.ndim == 3:
                accum.append(arr[..., mask_bool].ravel())
            else:
                accum.append(arr[mask_bool].ravel())

        vals = np.concatenate(accum)

        return float(np.sqrt(np.nanmean(vals**2)))


    def total_rmse_full_from_xr(diff_xr):
        accum = []

        for _, da in diff_xr.data_vars.items():
            arr = np.asarray(da.squeeze(drop=True).values)
            accum.append(arr.ravel())

        vals = np.concatenate(accum)

        return float(np.sqrt(np.nanmean(vals**2)))


    def relative_rmse_improvement_dict(
        rmse_unguided_dict,
        rmse_guided_dict,
        eps=1e-12,
    ):
        common_keys = sorted(
            set(rmse_unguided_dict.keys()).intersection(rmse_guided_dict.keys())
        )

        return {
            key: (
                rmse_unguided_dict[key] - rmse_guided_dict[key]
            )
            / max(rmse_unguided_dict[key], eps)
            for key in common_keys
        }


    def build_rmse_analysis_from_xr(
        *,
        ground_truth_xr,
        guided_xr,
        unguided_xr,
        member_values,
        guided_member_values,
        N,
        mask_2d,
        select_member_if_present,
    ):
        ground_truth_times = list(ground_truth_xr.time.values)
        guided_times = list(guided_xr.time.values)
        unguided_times = list(unguided_xr.time.values)

        N_eff = min(
            int(N),
            len(ground_truth_times) - 1,
            len(guided_times),
            len(unguided_times),
        )

        M_eff = len(member_values)

        rmse_level_rel_list = []
        rmse_var_rel_list = []

        rmse_guided_total = np.full((N_eff, M_eff), np.nan)
        rmse_unguided_total = np.full((N_eff, M_eff), np.nan)
        rmse_guided_full = np.full((N_eff, M_eff), np.nan)
        rmse_unguided_full = np.full((N_eff, M_eff), np.nan)

        for step_idx in range(N_eff):
            ground_truth_time = ground_truth_times[step_idx + 1]
            guided_time = guided_times[step_idx]
            unguided_time = unguided_times[step_idx]

            ground_truth_step = ground_truth_xr.sel(time=ground_truth_time)
            guided_step_all = guided_xr.sel(time=guided_time)
            unguided_step_all = unguided_xr.sel(time=unguided_time)

            level_rel_for_members = []
            var_rel_for_members = []

            for member_index, unguided_member in enumerate(member_values):
                if len(guided_member_values) == len(member_values):
                    guided_member = guided_member_values[member_index]
                elif len(guided_member_values) == 1:
                    guided_member = guided_member_values[0]
                else:
                    guided_member = guided_member_values[
                        min(member_index, len(guided_member_values) - 1)
                    ]

                guided_step = guided_step_all.map(
                    lambda da: select_member_if_present(da, guided_member)
                )
                unguided_step = unguided_step_all.map(
                    lambda da: select_member_if_present(da, unguided_member)
                )

                diff_guided = guided_step - ground_truth_step
                diff_unguided = unguided_step - ground_truth_step

                level_guided = per_var_level_rmse_dict_on_mask_from_xr(
                    diff_guided,
                    mask_2d,
                )
                level_unguided = per_var_level_rmse_dict_on_mask_from_xr(
                    diff_unguided,
                    mask_2d,
                )

                level_rel_for_members.append(
                    relative_rmse_improvement_dict(
                        level_unguided,
                        level_guided,
                    )
                )

                var_guided = per_var_rmse_dict_on_mask_from_xr(
                    diff_guided,
                    mask_2d,
                    drop_surface=True,
                )
                var_unguided = per_var_rmse_dict_on_mask_from_xr(
                    diff_unguided,
                    mask_2d,
                    drop_surface=True,
                )

                var_rel_for_members.append(
                    relative_rmse_improvement_dict(
                        var_unguided,
                        var_guided,
                    )
                )

                rmse_guided_total[step_idx, member_index] = (
                    total_rmse_on_mask_from_xr(diff_guided, mask_2d)
                )
                rmse_unguided_total[step_idx, member_index] = (
                    total_rmse_on_mask_from_xr(diff_unguided, mask_2d)
                )
                rmse_guided_full[step_idx, member_index] = (
                    total_rmse_full_from_xr(diff_guided)
                )
                rmse_unguided_full[step_idx, member_index] = (
                    total_rmse_full_from_xr(diff_unguided)
                )

            rmse_level_rel_list.append(level_rel_for_members)
            rmse_var_rel_list.append(var_rel_for_members)

        return {
            "rmse_level_rel_list": rmse_level_rel_list,
            "rmse_var_rel_list": rmse_var_rel_list,
            "rmse_guided_total": rmse_guided_total,
            "rmse_unguided_total": rmse_unguided_total,
            "rmse_guided_full": rmse_guided_full,
            "rmse_unguided_full": rmse_unguided_full,
        }

    return (build_rmse_analysis_from_xr,)


@app.cell
def _(
    N,
    build_rmse_analysis_from_xr,
    ground_truth_xr,
    guided_member_values,
    guided_xr,
    mask,
    member_values,
    select_member_if_present,
    unguided_xr,
):
    rmse_analysis = build_rmse_analysis_from_xr(
        ground_truth_xr=ground_truth_xr,
        guided_xr=guided_xr,
        unguided_xr=unguided_xr,
        member_values=member_values,
        guided_member_values=guided_member_values,
        N=N,
        mask_2d=mask,
        select_member_if_present=select_member_if_present,
    )
    return (rmse_analysis,)


@app.cell
def _(rmse_analysis):
    rmse_level_rel_list = rmse_analysis["rmse_level_rel_list"]
    rmse_var_rel_list = rmse_analysis["rmse_var_rel_list"]

    rmse_guided_total = rmse_analysis["rmse_guided_total"]
    rmse_unguided_total = rmse_analysis["rmse_unguided_total"]

    rmse_guided_full = rmse_analysis["rmse_guided_full"]
    rmse_unguided_full = rmse_analysis["rmse_unguided_full"]
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
        aggregate_dicts(dict_list, error="std")
        for dict_list in rmse_level_rel_list
    ]

    agg_var_rel_per_n = [
        aggregate_dicts(dict_list, error="std")
        for dict_list in rmse_var_rel_list
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
        start=1,
        stop=50,
        value=12,
        step=1,
        label="top-k var-level",
    )

    rank_by_rmse_radio = mo.ui.radio(
        options=["max", "mean"],
        value="max",
        label="rank by",
    )

    rmse_controls = mo.hstack(
        [top_k_rmse_slider, rank_by_rmse_radio],
        justify="start",
    )
    return (
        agg_level_rel_per_n,
        agg_var_rel_per_n,
        rank_by_rmse_radio,
        rmse_controls,
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
def _(rmse_controls):
    rmse_controls
    return


@app.cell
def _(mo):
    mo.md(r"""
    > **Reading the plots:** values above 0 are good: guidance reduced RMSE.
    Values below 0 are bad: guidance increased RMSE.
    """)
    return


@app.cell
def _(
    agg_level_rel_per_n,
    plot_variable_change_parallel,
    rank_by_rmse_radio,
    top_k_rmse_slider,
):
    rmse_level_fig, rmse_level_ax = plot_variable_change_parallel(
        agg_level_rel_per_n,
        top_k=top_k_rmse_slider.value,
        bottom_k=top_k_rmse_slider.value,
        rank_by=rank_by_rmse_radio.value,
        title="Relative RMSE improvement, per variable-level",
        subtitle="(RMSE_unguided - RMSE_guided) / RMSE_unguided on mask; error bars = ensemble std",
        ylim=None,
        ylabel="relative RMSE improvement",
    )

    rmse_level_ax.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    return (rmse_level_fig,)


@app.cell
def _(rmse_level_fig):
    rmse_level_fig
    return


@app.cell
def _(agg_var_rel_per_n, plot_variable_change_parallel):
    rmse_var_fig, rmse_var_ax = plot_variable_change_parallel(
        agg_var_rel_per_n,
        top_k=None,
        rank_by="max",
        title="Relative RMSE improvement, per variable",
        subtitle="per-variable RMSE pooled across pressure levels; error bars = ensemble std",
        ylim=None,
        ylabel="relative RMSE improvement",
    )

    rmse_var_ax.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    return (rmse_var_fig,)


@app.cell
def _(rmse_var_fig):
    rmse_var_fig
    return


@app.cell
def _(
    member_values,
    plot_rmse_over_n,
    rmse_guided_mean,
    rmse_guided_std,
    rmse_unguided_mean,
    rmse_unguided_std,
):
    rmse_over_n_fig, _ = plot_rmse_over_n(
        rmse_guided=rmse_guided_mean,
        rmse_unguided=rmse_unguided_mean,
        err_guided=rmse_guided_std,
        err_unguided=rmse_unguided_std,
        title="Mask RMSE over rollout",
        subtitle=f"ensemble mean +/- std on mask, NetCDF value space (M={len(member_values)})",
    )
    return (rmse_over_n_fig,)


@app.cell
def _(rmse_over_n_fig):
    rmse_over_n_fig
    return


@app.cell
def _(
    member_values,
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
        subtitle=f"ensemble mean +/- std on full domain, NetCDF value space (M={len(member_values)})",
        ylabel="RMSE, whole state",
    )
    return (rmse_over_n_full_fig,)


@app.cell
def _(rmse_over_n_full_fig):
    rmse_over_n_full_fig
    return


if __name__ == "__main__":
    app.run()
