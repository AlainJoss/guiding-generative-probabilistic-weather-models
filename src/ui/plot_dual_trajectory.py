import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator


def plot_dual_trajectory(
    timestamps: list[str],
    var: str,
    m: int, 
    mean_rollout: list[float] | None = None,
    reference_trajectory: list[float] | None = None,
    unguided_member: list[float] | None = None,
    planned_guidance: list[float] | None = None,
    ground_truth: list[float] | None = None,
    y_trajectory: list[float] | None = None,
    ensemble_rollout: list[list[float]] | None = None,
    ymin_left: float | None = None,
    ymax_left: float | None = None,
    right_axis: bool = True,
    n: int | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    ylabel: str | None = None,
    dpi: int = 180,
    figsize: tuple[float, float] = (17.5, 5.5),
):
    num_steps = len(timestamps)
    time_values = pd.to_datetime(timestamps)

    if n is not None and not (0 <= n < num_steps):
        raise ValueError(f"n must be in [0, {num_steps - 1}], got n={n}")

    mean_rollout = (
        np.asarray(mean_rollout, dtype=float)
        if mean_rollout is not None
        else None
    )
    unguided_member = (
        np.asarray(unguided_member, dtype=float)
        if unguided_member is not None
        else None
    )
    planned_guidance = (
        np.asarray(planned_guidance, dtype=float)
        if planned_guidance is not None
        else None
    )
    ground_truth = (
        np.asarray(ground_truth, dtype=float)
        if ground_truth is not None
        else None
    )
    reference_trajectory = (
        np.asarray(reference_trajectory, dtype=float)
        if reference_trajectory is not None
        else None
    )
    y_trajectory = (
        np.asarray(y_trajectory, dtype=float) * 100.0
        if y_trajectory is not None
        else None
    )

    for name, values in {
        "unguided_member": unguided_member,
        "mean_rollout": mean_rollout,
        "planned_guidance": planned_guidance,
        "ground_truth": ground_truth,
        "reference_trajectory": reference_trajectory,
        "y_trajectory": y_trajectory,
    }.items():
        if values is not None and len(values) != num_steps:
            raise ValueError(f"{name} must have the same length as timestamps")

    colors = {
        "mean": "#6E6E6E",
        "ensemble": "#6E6E6E",
        "target": "#D55E00",
        "reference": "#009E73",
        "reference_trajectory": "#E6B800",
        "y": "#7B2CBF",
        "grid_major": "#D7D7D7",
        "grid_minor": "#EAEAEA",
        "text": "#222222",
        "n_marker": "#222222",
    }

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

        # ------------------------------------------------------------
        # Ensemble shadow
        # Expected shape: (num_steps, num_members)
        # ------------------------------------------------------------
        if ensemble_rollout is not None:
            ensemble_rollout = np.asarray(ensemble_rollout, dtype=float)

            if ensemble_rollout.ndim != 2:
                raise ValueError(
                    f"ensemble_rollout must be 2D, got shape {ensemble_rollout.shape}"
                )

            if ensemble_rollout.shape[0] != num_steps:
                raise ValueError(
                    "ensemble_rollout must have shape (num_steps, num_members). "
                    f"Got shape {ensemble_rollout.shape}, num_steps={num_steps}."
                )

            num_members = ensemble_rollout.shape[1]

            ens_min = np.nanmin(ensemble_rollout, axis=1)
            ens_max = np.nanmax(ensemble_rollout, axis=1)

            ax.fill_between(
                time_values,
                ens_min,
                ens_max,
                color=colors["ensemble"],
                alpha=0.16,
                linewidth=0,
                label=f"Ensemble range, M={num_members}",
                zorder=1,
            )

            ax.plot(
                time_values,
                ens_min,
                linestyle="-",
                linewidth=0.9,
                color=colors["ensemble"],
                alpha=0.45,
                zorder=2,
                label="_nolegend_",
            )

            ax.plot(
                time_values,
                ens_max,
                linestyle="-",
                linewidth=0.9,
                color=colors["ensemble"],
                alpha=0.45,
                zorder=2,
                label="_nolegend_",
            )

            y_values.append(ensemble_rollout.reshape(-1))

        # ------------------------------------------------------------
        # Planned guidance
        # ------------------------------------------------------------
        planned_equals_ground_truth = (
            planned_guidance is not None
            and ground_truth is not None
            and np.allclose(planned_guidance, ground_truth, equal_nan=True)
        )

        if planned_guidance is not None:
            ax.plot(
                time_values,
                planned_guidance,
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
                            foreground="white",
                            alpha=0.9,
                        ),
                        pe.Normal(),
                    ]
                    if planned_equals_ground_truth
                    else None
                ),
            )
            y_values.append(planned_guidance)


        # ------------------------------------------------------------
        # Mean rollout
        # ------------------------------------------------------------
        if mean_rollout is not None:
            ax.plot(
                time_values,
                mean_rollout,
                linestyle="-",
                linewidth=0,
                color=colors["mean"],
                alpha=0.16,
                label="Mean rollout",
                zorder=6,
                path_effects=[
                    pe.Stroke(linewidth=1, alpha=0.16),
                    pe.Normal(),
                ],
            )
            y_values.append(mean_rollout)


        if unguided_member is not None:
            ax.plot(
                time_values,
                unguided_member,
                linestyle="-",
                linewidth=2.4,
                color=colors["mean"],
                alpha=0.98,
                label=f"Unguided member {m}",
                zorder=6,
                path_effects=[
                    pe.Stroke(linewidth=2.2, alpha=0.9),
                    pe.Normal(),
                ],
            )
            y_values.append(unguided_member)
        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------
        if ground_truth is not None:
            ax.plot(
                time_values,
                ground_truth,
                linestyle="-",
                linewidth=2.2,
                color=colors["reference"],
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
        # Vertical grid lines at every forecast step
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
        # Optional current n marker
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
        has_right_axis = right_axis and y_trajectory is not None

        if has_right_axis:
            ax2 = ax.twinx()

            ax2.plot(
                time_values,
                y_trajectory,
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

            # No right-axis label; show percent directly on tick labels.
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
            y_min = float(np.nanmin(y_trajectory))
            y_max = float(np.nanmax(y_trajectory))

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

        unique = dict(zip(labels, handles))

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
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
                y=0.98,
                ha="left",
                fontsize=15,
                fontweight="bold",
                color=colors["text"],
            )

        if subtitle:
            fig.text(
                0.06,
                0.925,
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