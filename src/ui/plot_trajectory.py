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
    trajectory: list[np.float64],
    var: str,
    ymin: float | None = None,
    ymax: float | None = None,
    title: str | None = "Trajectory",
    subtitle: str | None = None,
    dpi: int = 180,
    figsize: tuple[float, float] = (17.5, 4.0),
):
    trajectory = np.asarray(trajectory, dtype=float)
    num_steps = len(trajectory)

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
        for t in x:
            ax.axvline(
                t,
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
        # Main trajectory
        # ------------------------------------------------------------
        ax.plot(
            x,
            trajectory,
            linestyle="-",
            linewidth=2.2,
            color=colors["line"],
            alpha=0.95,
            label=var,
            zorder=4,
            path_effects=[
                pe.Stroke(linewidth=4.0, foreground="white", alpha=0.85),
                pe.Normal(),
            ],
        )

        ax.scatter(
            x,
            trajectory,
            s=28,
            color=colors["marker"],
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

        ax.set_xlabel(r"$t$")
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
        y_min = float(np.nanmin(trajectory))
        y_max = float(np.nanmax(trajectory))
        y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 1.0

        final_ymin = ymin if ymin is not None else y_min - y_pad
        final_ymax = ymax if ymax is not None else y_max + y_pad

        if final_ymin == final_ymax:
            final_ymin -= 1.0
            final_ymax += 1.0

        ax.set_ylim(final_ymin, final_ymax)

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.015, 0.5),
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
            rect=(0.0, 0.0, 0.84, 0.90 if (title or subtitle) else 1.0)
        )

    return fig