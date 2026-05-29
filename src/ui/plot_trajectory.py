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
from matplotlib.ticker import AutoMinorLocator

def plot_trajectory(
    trajectory: list[np.float64] | dict[str, list[np.float64]],
    var: str | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    title: str | None = "Trajectory",
    subtitle: str | None = None,
    dpi: int = 180,
    figsize: tuple[float, float] = (17.5, 4.0),
    t: int | None = None,
    right_trajectory: list[np.float64] | None = None,
    right_label: str = r"$\lambda_t$",
    right_color: str = "#7B2CBF",
    xlabel:str="$t$"
):
    trajectory_dict = trajectory if isinstance(trajectory, dict) else {var: trajectory}
    trajectory_dict = {k: np.asarray(v, dtype=float) for k, v in trajectory_dict.items()}
    num_steps = max(len(v) for v in trajectory_dict.values())

    if num_steps == 0:
        raise ValueError("trajectory must contain at least one value")

    x = np.arange(num_steps)

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
        for step in x:
            ax.axvline(
                step,
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

        # ------------------------------------------------------------
        # Axis styling
        # ------------------------------------------------------------
        ax.set_xlim(0, max(num_steps - 1, 1))
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

        if final_ymin == final_ymax:
            final_ymin -= 1.0
            final_ymax += 1.0

        ax.set_ylim(final_ymin, final_ymax)

        if t is not None:
            ax.axvline(
                t,
                color=colors["text"],
                linestyle=(0, (4, 4)),
                linewidth=1.2,
                alpha=0.75,
                label="_nolegend_",
                zorder=10,
            )

            if single:
                only = next(iter(trajectory_dict.values()))
                subtract = only[t-1] if t > 0 else only[t]
                lambda_diff = only[t] - subtract
                annotation = f"t={t}\nΔ={lambda_diff:.3f}"

                ax.annotate(
                    annotation,
                    xy=(t, 1.0),
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
            right = np.asarray(right_trajectory, dtype=float)
            ax2 = ax.twinx()
            ax2.plot(
                x[: len(right)],
                right,
                linestyle="-",
                linewidth=1.6,
                color=right_color,
                alpha=0.7,
                label=right_label,
                zorder=4,
            )
            ax2.axhline(0.0, color=colors["grid_major"], linewidth=0.8, alpha=0.7, zorder=0)
            ax2.set_ylabel("")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color("#BBBBBB")
            ax2.tick_params(axis="both", colors=right_color, length=4, width=0.8)

            r_min, r_max = float(np.nanmin(right)), float(np.nanmax(right))
            if r_min == r_max:
                r_min -= 1.0
                r_max += 1.0
            r_pad = 0.08 * (r_max - r_min)
            ax2.set_ylim(r_min - r_pad, r_max + r_pad)

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles, labels = handles1 + handles2, labels1 + labels2
        else:
            handles, labels = ax.get_legend_handles_labels()

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        ax.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.05 if has_right_axis else 1.015, 0.5),
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

        fig.tight_layout(
            rect=(0.0, 0.0, 0.80 if has_right_axis else 0.84, 0.90 if (title or subtitle) else 1.0)
        )

    return fig