import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import marimo as mo
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

def plot_trajectory(
    trajectory: list[np.float64] | dict[str, list[np.float64]],
    var: str | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    title: str | None = "Trajectory",
    subtitle: str | None = None,
    dpi: int = 180,
    figsize: tuple[float, float] = (17.5, 4.0),
    step: int | None = None,
    right_trajectory: list[np.float64] | None = None,
    right_label: str = r"$\lambda_t$",
    right_color: str = "#7B2CBF",
    xlabel:str="$t$",
    color_map: dict | None = None,
    right_percentage: bool = False,
    bands: dict[str, tuple] | None = None,
    prepend_zero: bool = False,
    start_index: int = 0,
    mirror_right_axis: bool = False,
):
    # start_index: x-axis index of the first data point (default 0). Set to 1 for a
    # 1-indexed axis (e.g. "over t" plots that start at t=1) without prepend_zero's
    # artificial (0, 0) anchor point.
    trajectory_dict = trajectory if isinstance(trajectory, dict) else {var: trajectory}
    trajectory_dict = {k: np.asarray(v, dtype=float) for k, v in trajectory_dict.items()}

    # drop all-NaN traces (e.g. flow traces at forecast steps where guidance was
    # off, so nothing was recorded); if none survive, render a placeholder figure
    # instead of letting matplotlib fail on NaN axis limits.
    finite_dict = {k: v for k, v in trajectory_dict.items() if np.isfinite(v).any()}
    if trajectory_dict and not finite_dict:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_axis_off()
        ax.text(
            0.5, 0.5,
            "no finite data for this selection",
            ha="center", va="center",
            fontsize=12, color="#888888",
            transform=ax.transAxes,
        )
        if title:
            fig.suptitle(title, x=0.06, y=0.98, ha="left", fontsize=15,
                         fontweight="bold", color="#222222")
        if subtitle:
            fig.text(0.06, 0.925, subtitle, ha="left", va="top",
                     fontsize=9.5, color="#555555")
        return fig
    trajectory_dict = finite_dict

    if prepend_zero:
        # anchor every line at the origin (0, 0): the value for step i moves to x=i+1,
        # so x=0 is a baseline and the x-axis index matches the (1-indexed) selected
        # step instead of being off by one. bands + right-axis lines shift in lockstep.
        trajectory_dict = {k: np.concatenate([[0.0], v]) for k, v in trajectory_dict.items()}
        if bands:
            bands = {
                k: (np.concatenate([[0.0], np.asarray(lo, dtype=float)]),
                    np.concatenate([[0.0], np.asarray(hi, dtype=float)]))
                for k, (lo, hi) in bands.items()
            }
        if right_trajectory is not None:
            if isinstance(right_trajectory, dict):
                right_trajectory = {k: np.concatenate([[0.0], np.asarray(v, dtype=float)])
                                    for k, v in right_trajectory.items()}
            else:
                right_trajectory = np.concatenate([[0.0], np.asarray(right_trajectory, dtype=float)])

    num_steps = max(len(v) for v in trajectory_dict.values())

    if num_steps == 0:
        raise ValueError("trajectory must contain at least one value")

    x = np.arange(num_steps) + start_index

    colors = {
        "line": "#7B2CBF",
        "marker": "#7B2CBF",
        "zero": "#BBBBBB",
        "grid_major": "#D7D7D7",
        "grid_minor": "#EAEAEA",
        "text": "#222222",
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

        # ------------------------------------------------------------
        # Vertical grid lines at every timestep
        # ------------------------------------------------------------
        for tick in x:
            ax.axvline(
                tick,
                color=colors["grid_major"],
                linestyle="-",
                linewidth=0.65,
                alpha=0.28,
                label="_nolegend_",
                zorder=0,
            )

        # ------------------------------------------------------------
        # Zero line
        # ------------------------------------------------------------
        ax.axhline(
            0.0,
            color=colors["zero"],
            linewidth=0.8,
            alpha=0.75,
            label="_nolegend_",
            zorder=1,
        )

        # ------------------------------------------------------------
        # Main trajectories
        # ------------------------------------------------------------
        single = len(trajectory_dict) == 1
        palette = list(plt.get_cmap("tab10").colors)

        for i, (label, traj) in enumerate(trajectory_dict.items()):
            if color_map and label in color_map:
                color = color_map[label]
            else:
                color = colors["line"] if single else palette[i % len(palette)]
            ax.plot(
                x[: len(traj)],
                traj,
                linestyle="-",
                linewidth=2.2,
                color=color,
                alpha=0.95,
                label=label,
                zorder=4,
                path_effects=[
                    pe.Stroke(linewidth=4.0, foreground="white", alpha=0.85),
                    pe.Normal(),
                ] if single else None,
            )

            ax.scatter(
                x[: len(traj)],
                traj,
                s=28,
                color=color,
                alpha=0.95,
                zorder=5,
                edgecolors="white",
                linewidths=0.7,
            )

            if bands and label in bands:
                band_lo, band_hi = bands[label]
                ax.fill_between(
                    x[: len(band_lo)],
                    band_lo,
                    band_hi,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    label="_nolegend_",
                    zorder=3,
                )

        # ------------------------------------------------------------
        # Axis styling
        # ------------------------------------------------------------
        ax.set_xlim(start_index, start_index + max(num_steps - 1, 1))
        ax.set_xticks(x)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(var)

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
        y_all = np.concatenate(list(trajectory_dict.values()))
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 1.0

        final_ymin = ymin if ymin is not None else y_min - y_pad
        final_ymax = ymax if ymax is not None else y_max + y_pad

        if not (np.isfinite(final_ymin) and np.isfinite(final_ymax)):
            final_ymin, final_ymax = -1.0, 1.0
        elif final_ymin == final_ymax:
            final_ymin -= 1.0
            final_ymax += 1.0

        ax.set_ylim(final_ymin, final_ymax)

        if step is not None:
            ax.axvline(
                step,
                color=colors["text"],
                linestyle=(0, (4, 4)),
                linewidth=1.2,
                alpha=0.75,
                label="_nolegend_",
                zorder=10,
            )

            ax.annotate(
                f"{xlabel}={step}",
                xy=(step, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(6, -8),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=9,
                color=colors["text"],
                alpha=0.85,
                zorder=12,
            )


        # ------------------------------------------------------------
        # Optional right axis
        # ------------------------------------------------------------
        has_right_axis = right_trajectory is not None

        if has_right_axis:
            # dict -> multiple right-axis lines (shared scale); plain array -> one
            rights = (
                right_trajectory
                if isinstance(right_trajectory, dict)
                else {right_label: right_trajectory}
            )
            right_colors = (
                right_color if isinstance(right_color, dict) else {label: right_color for label in rights}
            )
            rights = {
                label: np.asarray(vals, dtype=float) * (100.0 if right_percentage else 1.0)
                for label, vals in rights.items()
            }
            ax2 = ax.twinx()
            for label, right in rights.items():
                ax2.plot(
                    x[: len(right)],
                    right,
                    linestyle="-",
                    linewidth=1.6,
                    color=right_colors.get(label, "#7B2CBF"),
                    alpha=0.7,
                    label=label,
                    zorder=4,
                )
            ax2.axhline(0.0, color=colors["grid_major"], linewidth=0.8, alpha=0.7, zorder=0)
            ax2.set_ylabel("")
            if right_percentage:
                ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color("#BBBBBB")
            ax2.tick_params(axis="both", colors=next(iter(right_colors.values())), length=4, width=0.8)

            right_all = np.concatenate(list(rights.values()))
            r_min, r_max = float(np.nanmin(right_all)), float(np.nanmax(right_all))
            if not (np.isfinite(r_min) and np.isfinite(r_max)):
                r_min, r_max = -1.0, 1.0
            elif r_min == r_max:
                r_min -= 1.0
                r_max += 1.0
            r_pad = 0.08 * (r_max - r_min)
            ax2.set_ylim(r_min - r_pad, r_max + r_pad)

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles, labels = handles1 + handles2, labels1 + labels2
        else:
            if mirror_right_axis:
                # phantom right axis: mirror the left y-ticks as little lines (no
                # numbers, no data) so plots without a real right axis line up with
                # the cross-check plots that do have one.
                ax_mirror = ax.twinx()
                ax_mirror.set_ylim(ax.get_ylim())
                ax_mirror.set_yticks(ax.get_yticks())
                ax_mirror.set_yticklabels([])
                ax_mirror.set_ylabel("")
                ax_mirror.spines["top"].set_visible(False)
                ax_mirror.spines["right"].set_color("#BBBBBB")
                ax_mirror.tick_params(axis="y", colors=colors["text"], length=4, width=0.8)
            handles, labels = ax.get_legend_handles_labels()

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        ax.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.05 if (has_right_axis or mirror_right_axis) else 1.015, 0.5),
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
        )

        # ------------------------------------------------------------
        # Titles
        # ------------------------------------------------------------
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

        # reserve a fixed-WIDTH strip (in inches) for the right-hand legend rather
        # than a fixed fraction, so the axes box scales with the figure width: a
        # narrow figure (few steps) keeps a usable legend without stretching the axes.
        legend_inches = 3.4 if (has_right_axis or mirror_right_axis) else 3.0
        fig.subplots_adjust(
            left=0.03,
            right=max(0.55, 1.0 - legend_inches / figsize[0]),
            top=0.88 if (title or subtitle) else 0.97,
            bottom=0.10,
        )

    return fig