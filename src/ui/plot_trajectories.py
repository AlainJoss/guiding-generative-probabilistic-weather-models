import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator


def plot_trajectories(
    *,
    # selected members
    guided_member: list[float] | None = None,
    unguided_member: list[float] | None = None,
    unguided_guided_member: list[float] | None = None,

    # ensemble bands
    guided_ensemble: list[list[float]] | None = None,
    unguided_ensemble: list[list[float]] | None = None,
    target_ensemble: list[list[float]] | None = None,
    target_guidance_ensemble: list[list[float]] | None = None,

    # optional summaries / references
    mean_unguided_rollout: list[float] | None = None,
    mean_guided_rollout: list[float] | None = None,
    target_trajectory: list[float] | None = None,
    target_guidance_trajectory: list[float] | None = None,
    ground_truth: list[float] | None = None,
    reference_trajectory: list[float] | None = None,

    # percentage axis, used if y_trajectory is not None
    delta_trajectory: list[float] | None = None,

    # display
    show_guided_mean: bool = False,
    show_unguided_mean: bool = False,

    timestamps: list[str],
    var: str,
    m: int | None = None,
    n: int | None = None,
    dpi: int = 180,
    figsize: tuple[float, float] = (17.5, 5.5),
    title: str | None = None,
    subtitle: str | None = None,
    ylabel: str | None = None,
    ymin_left: float | None = None,
    ymax_left: float | None = None,
):
    num_steps = len(timestamps)
    time_values = pd.to_datetime(timestamps)

    if n is not None and not (0 <= n < num_steps):
        raise ValueError(f"n must be in [0, {num_steps - 1}], got n={n}")

    def _as_1d(values, name: str, gt0=None):
        if values is None:
            return None

        values = np.asarray(values, dtype=float)

        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {values.shape}")

        if len(values) == num_steps:
            return values

        if gt0 is not None and len(values) == num_steps - 1:
            return np.concatenate([[gt0], values])

        raise ValueError(f"{name} must have the same length as timestamps")

    def _as_step_member_array(values, name: str, gt0=None):
        if values is None:
            return None

        values = np.asarray(values, dtype=float)

        if values.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {values.shape}")

        if values.shape[0] == num_steps:
            return values

        if values.shape[1] == num_steps:
            return values.T

        if gt0 is not None and num_steps - 1 in values.shape:
            if values.shape[1] == num_steps - 1:
                values = values.T
            num_members = values.shape[1]
            return np.vstack([np.full((1, num_members), gt0), values])

        raise ValueError(
            f"{name} must have one dimension equal to num_steps={num_steps}. "
            f"Got shape {values.shape}."
        )

    # Ground truth carries the initial (n=0) step; gt0 is prepended to every
    # forecast array that is one step short (length num_steps - 1) so all
    # trajectories start from the common initial point.
    ground_truth = _as_1d(ground_truth, "ground_truth")
    gt0 = ground_truth[0] if ground_truth is not None else None

    guided_member = _as_1d(guided_member, "guided_member", gt0=gt0)
    unguided_member = _as_1d(unguided_member, "unguided_member", gt0=gt0)
    unguided_guided_member = _as_1d(unguided_guided_member, "unguided_guided_member", gt0=gt0)

    mean_unguided_rollout = _as_1d(
        mean_unguided_rollout,
        "mean_unguided_rollout",
        gt0=gt0,
    )
    mean_guided_rollout = _as_1d(
        mean_guided_rollout,
        "mean_guided_rollout",
        gt0=gt0,
    )

    target_trajectory = _as_1d(target_trajectory, "planned_guidance", gt0=gt0)
    target_guidance_trajectory = _as_1d(target_guidance_trajectory, "target_guidance_trajectory", gt0=gt0)
    reference_trajectory = _as_1d(reference_trajectory, "reference_trajectory", gt0=gt0)

    delta_trajectory = (
        _as_1d(delta_trajectory, "y_trajectory", gt0=gt0) * 100.0
        if delta_trajectory is not None
        else None
    )

    guided_ensemble = _as_step_member_array(guided_ensemble, "guided_ensemble", gt0=gt0)
    unguided_ensemble = _as_step_member_array(unguided_ensemble, "unguided_ensemble", gt0=gt0)
    target_ensemble = _as_step_member_array(target_ensemble, "target_ensemble", gt0=gt0)
    target_guidance_ensemble = _as_step_member_array(target_guidance_ensemble, "target_guidance_ensemble", gt0=gt0)

    colors = {
        "guided": "#0072B2",
        "unguided": "#6E6E6E",
        "target": "#D55E00",
        "target_guidance": "#B7950B",
        "ground_truth": "#009E73",
        "reference_trajectory": "#E6B800",
        "y": "#7B2CBF",
        "unguided_guided": "#4A6FA5",
        "grid_major": "#D7D7D7",
        "grid_minor": "#EAEAEA",
        "text": "#222222",
        "n_marker": "#222222",
    }

    def _member_label(prefix: str):
        return f"{prefix} member, m={m}" if m is not None else f"{prefix} member"

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        y_values = []

        def _plot_ensemble_band(
            ensemble,
            color: str,
            label_prefix: str,
            zorder_fill: int,
            zorder_line: int,
            selected_member=None,
            show_mean: bool = False,
        ):
            num_members = ensemble.shape[1]

            ens_min = np.nanmin(ensemble, axis=1)
            ens_max = np.nanmax(ensemble, axis=1)
            ens_mean = np.nanmean(ensemble, axis=1)

            if selected_member is not None:
                if not np.all(
                    (selected_member >= ens_min - 1e-10)
                    & (selected_member <= ens_max + 1e-10)
                ):
                    raise ValueError(
                        f"{label_prefix.lower()} selected member is outside the "
                        f"{label_prefix.lower()} ensemble range. Check timestamp/member alignment."
                    )

            ax.fill_between(
                time_values,
                ens_min,
                ens_max,
                color=color,
                alpha=0.16,
                linewidth=0,
                label=f"{label_prefix} ensemble range, M={num_members}",
                zorder=zorder_fill,
            )

            ax.plot(
                time_values,
                ens_min,
                linestyle="-",
                linewidth=0.9,
                color=color,
                alpha=0.45,
                zorder=zorder_line,
                label="_nolegend_",
            )

            ax.plot(
                time_values,
                ens_max,
                linestyle="-",
                linewidth=0.9,
                color=color,
                alpha=0.45,
                zorder=zorder_line,
                label="_nolegend_",
            )

            y_values.append(ensemble.reshape(-1))

            if show_mean:
                ax.plot(
                    time_values,
                    ens_mean,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.85,
                    label=f"{label_prefix} ensemble mean",
                    zorder=zorder_line + 2,
                )
                y_values.append(ens_mean)

        # ------------------------------------------------------------
        # Ensemble bands
        # ------------------------------------------------------------
        if unguided_ensemble is not None:
            _plot_ensemble_band(
                ensemble=unguided_ensemble,
                color=colors["unguided"],
                label_prefix="Unguided",
                zorder_fill=1,
                zorder_line=2,
                selected_member=unguided_member,
                show_mean=show_unguided_mean,
            )

        if guided_ensemble is not None:
            _plot_ensemble_band(
                ensemble=guided_ensemble,
                color=colors["guided"],
                label_prefix="Guided",
                zorder_fill=3,
                zorder_line=4,
                selected_member=guided_member,
                show_mean=show_guided_mean,
            )

        if target_ensemble is not None:
            _plot_ensemble_band(
                ensemble=target_ensemble,
                color=colors["target"],
                label_prefix="Target",
                zorder_fill=2,
                zorder_line=3,
                selected_member=None,
                show_mean=False,
            )

        if target_guidance_ensemble is not None:
            _plot_ensemble_band(
                ensemble=target_guidance_ensemble,
                color=colors["target_guidance"],
                label_prefix="Target guidance",
                zorder_fill=2,
                zorder_line=3,
                selected_member=None,
                show_mean=False,
            )

        # ------------------------------------------------------------
        # Planned guidance
        # ------------------------------------------------------------
        planned_equals_ground_truth = (
            target_trajectory is not None
            and ground_truth is not None
            and np.allclose(target_trajectory, ground_truth, equal_nan=True)
        )

        if target_trajectory is not None:
            ax.plot(
                time_values,
                target_trajectory,
                linestyle="--" if planned_equals_ground_truth else "-",
                linewidth=2.4 if planned_equals_ground_truth else 2.0,
                color=colors["target"],
                alpha=0.98,
                label="Planned guidance",
                zorder=9 if planned_equals_ground_truth else 5,
                path_effects=(
                    [
                        pe.Stroke(
                            linewidth=4.2,
                            # foreground="white",
                            alpha=0.9,
                        ),
                        pe.Normal(),
                    ]
                    if planned_equals_ground_truth
                    else None
                ),
            )
            y_values.append(target_trajectory)

        # ------------------------------------------------------------
        # Target guidance: (1 + delta_n) * unguided masked mean
        # ------------------------------------------------------------
        if target_guidance_trajectory is not None:
            ax.plot(
                time_values,
                target_guidance_trajectory,
                linestyle="-",
                linewidth=2.0,
                color=colors["target_guidance"],
                alpha=0.95,
                label="Target guidance",
                zorder=5,
            )
            y_values.append(target_guidance_trajectory)

        # ------------------------------------------------------------
        # Mean rollouts
        # ------------------------------------------------------------
        if mean_unguided_rollout is not None:
            ax.plot(
                time_values,
                mean_unguided_rollout,
                linestyle="--",
                linewidth=1.5,
                color=colors["unguided"],
                alpha=0.85,
                label="Mean unguided rollout",
                zorder=5,
            )
            y_values.append(mean_unguided_rollout)

        if mean_guided_rollout is not None:
            ax.plot(
                time_values,
                mean_guided_rollout,
                linestyle="--",
                linewidth=1.6,
                color=colors["guided"],
                alpha=0.85,
                label="Mean guided rollout",
                zorder=6,
            )
            y_values.append(mean_guided_rollout)

        # ------------------------------------------------------------
        # Selected unguided member
        # ------------------------------------------------------------
        if unguided_member is not None:
            unguided_linewidth = 2.2

            ax.plot(
                time_values,
                unguided_member,
                linestyle="-",
                linewidth=unguided_linewidth,
                color=colors["unguided"],
                alpha=0.98,
                label=_member_label("Unguided"),
                zorder=8,
                path_effects=[
                    pe.Stroke(
                        linewidth=unguided_linewidth + 1,
                        # foreground="white",
                        alpha=0.95,
                    ),
                    pe.Normal(),
                ],
            )
            y_values.append(unguided_member)

        # ------------------------------------------------------------
        # Selected guided member
        # ------------------------------------------------------------
        if guided_member is not None:
            guided_linewidth = 2.8

            ax.plot(
                time_values,
                guided_member,
                linestyle="-",
                linewidth=guided_linewidth,
                color=colors["guided"],
                alpha=0.99,
                label=_member_label("Guided"),
                zorder=9,
                path_effects=[
                    pe.Stroke(
                        linewidth=guided_linewidth + 1,
                        # foreground="white",
                        alpha=0.95,
                    ),
                    pe.Normal(),
                ],
            )
            y_values.append(guided_member)

        # ------------------------------------------------------------
        # Unguided-guided: point per n + dashed branch from previous guided
        # ------------------------------------------------------------
        if unguided_guided_member is not None:
            if guided_member is not None:
                for n_idx in range(1, num_steps):
                    ug = unguided_guided_member[n_idx]
                    g_prev = guided_member[n_idx - 1]
                    if np.isnan(ug) or np.isnan(g_prev):
                        continue
                    ax.plot(
                        [time_values[n_idx - 1], time_values[n_idx]],
                        [g_prev, ug],
                        linestyle="--",
                        linewidth=1.2,
                        color=colors["unguided_guided"],
                        alpha=0.7,
                        zorder=7,
                        label="_nolegend_",
                    )
            ax.scatter(
                time_values,
                unguided_guided_member,
                s=42,
                color=colors["unguided_guided"],
                edgecolors="white",
                linewidths=0.8,
                zorder=9,
                label=_member_label("Unguided-guided"),
            )
            y_values.append(unguided_guided_member)

        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------
        if ground_truth is not None:
            ax.plot(
                time_values,
                ground_truth,
                linestyle="-",
                linewidth=2.2,
                color=colors["ground_truth"],
                alpha=0.70 if planned_equals_ground_truth else 0.95,
                label="Ground truth",
                zorder=7,
            )
            y_values.append(ground_truth)

        # ------------------------------------------------------------
        # Reference trajectory
        # ------------------------------------------------------------
        if reference_trajectory is not None:
            ax.plot(
                time_values,
                reference_trajectory,
                linestyle="--",
                linewidth=2.0,
                color=colors["reference_trajectory"],
                alpha=0.95,
                label="Reference trajectory",
                zorder=8,
            )
            y_values.append(reference_trajectory)

        # ------------------------------------------------------------
        # Vertical grid lines
        # ------------------------------------------------------------
        for step_idx, step_time in enumerate(time_values):
            if n is not None and step_idx == n:
                continue

            ax.axvline(
                step_time,
                color=colors["grid_major"],
                linestyle="-",
                linewidth=0.65,
                alpha=0.28,
                label="_nolegend_",
                zorder=0,
            )

        # ------------------------------------------------------------
        # Current n marker and automatic delta annotation
        # ------------------------------------------------------------
        if n is not None:
            ax.axvline(
                time_values[n],
                color=colors["n_marker"],
                linestyle=(0, (4, 4)),
                linewidth=1.2,
                alpha=0.75,
                label="_nolegend_",
                zorder=10,
            )

            ax.annotate(
                f"n={n}",
                xy=(time_values[n], 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(6, -8),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=9,
                color=colors["n_marker"],
                alpha=0.85,
                zorder=12,
            )

        # ------------------------------------------------------------
        # Axis styling
        # ------------------------------------------------------------
        ax.set_xlabel("Forecast time")
        ax.set_ylabel(ylabel if ylabel is not None else var)

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(False)

        ax.yaxis.grid(
            True,
            which="major",
            color=colors["grid_major"],
            linewidth=0.75,
            linestyle="-",
            alpha=0.55,
        )

        ax.yaxis.grid(
            True,
            which="minor",
            color=colors["grid_minor"],
            linewidth=0.55,
            linestyle="-",
            alpha=0.45,
        )

        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")

        ax.tick_params(
            axis="both",
            colors=colors["text"],
            length=4,
            width=0.8,
        )

        # ------------------------------------------------------------
        # Y-limits with padding
        # ------------------------------------------------------------
        if y_values:
            y_all = np.concatenate(y_values)
            y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
            y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 1.0

            left_min = ymin_left if ymin_left is not None else y_min - y_pad
            left_max = ymax_left if ymax_left is not None else y_max + y_pad

            if left_min == left_max:
                left_min -= 1.0
                left_max += 1.0

            ax.set_ylim(left_min, left_max)

        # ------------------------------------------------------------
        # Optional right axis for percentage trajectory
        # ------------------------------------------------------------
        has_right_axis = delta_trajectory is not None

        if has_right_axis:
            ax2 = ax.twinx()

            ax2.plot(
                time_values,
                delta_trajectory,
                linestyle="-",
                linewidth=1.6,
                color=colors["y"],
                alpha=0.5,
                label="Percentage change",
                zorder=4,
            )

            ax2.axhline(
                0.0,
                color=colors["grid_major"],
                linewidth=0.8,
                alpha=0.7,
                zorder=0,
            )

            ax2.set_ylabel("")
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))

            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color("#BBBBBB")

            ax2.tick_params(
                axis="both",
                colors=colors["y"],
                length=4,
                width=0.8,
            )

            left_min, left_max = ax.get_ylim()
            y_min = float(np.nanmin(delta_trajectory))
            y_max = float(np.nanmax(delta_trajectory))

            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0

            def map_left_to_right(v):
                return y_min + (v - left_min) * (y_max - y_min) / (left_max - left_min)

            ax2.set_ylim(
                map_left_to_right(left_min),
                map_left_to_right(left_max),
            )

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()

            handles = handles1 + handles2
            labels = labels1 + labels2
        else:
            handles, labels = ax.get_legend_handles_labels()

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        unique = dict(zip(labels, handles))
        legend_x = 1.05 if has_right_axis else 1.015

        ax.legend(
            unique.values(),
            unique.keys(),
            loc="center left",
            bbox_to_anchor=(legend_x, 0.5),
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
        )

        # ------------------------------------------------------------
        # Titles
        # ------------------------------------------------------------
        if title is None:
            title = f"{var} trajectory"

        if title:
            fig.suptitle(
                title,
                x=0.06,
                y=0.95,
                ha="left",
                fontsize=15,
                fontweight="bold",
                color=colors["text"],
            )

        if subtitle:
            fig.text(
                0.06,
                0.9,
                subtitle,
                ha="left",
                va="top",
                fontsize=9.5,
                color="#555555",
            )

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------
        right_margin = 0.80 if has_right_axis else 0.84

        fig.tight_layout(
            rect=(0.0, 0.0, right_margin, 0.90 if (title or subtitle) else 1.0)
        )

    return fig