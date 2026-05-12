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

import geopandas as gpd
import geodatasets
from wigglystuff import ChartPuck
def plot_variable_change_parallel(
    aggregated_per_n: list[dict],
    top_k: int | None = 20,
    bottom_k: int | None = None,
    rank_by: str = "max",
    figsize: tuple[float, float] = (17.5, 6.0),
    dpi: int = 180,
    title: str | None = None,
    subtitle: str | None = None,
    cmap: str = "viridis",
    jitter: float = 0.0,
    ylim: tuple[float, float] | None = (0, 1.05),
    ylabel: str = "min-max score",
    show_unselected: bool = False,
):
    """Parallel-coordinates view of per-variable-level scores across N."""

    N = len(aggregated_per_n)
    if N == 0:
        raise ValueError("aggregated_per_n is empty")

    keys = list(aggregated_per_n[0]["keys"])
    K = len(keys)

    means = np.full((N, K), np.nan)
    errs = np.full((N, K), np.nan)

    for n, agg in enumerate(aggregated_per_n):
        agg_keys = list(agg["keys"])
        idx = {k: j for j, k in enumerate(agg_keys)}

        for j, k in enumerate(keys):
            if k in idx:
                means[n, j] = agg["mean"][idx[k]]
                errs[n, j] = agg["err"][idx[k]]

    if rank_by == "max":
        score = np.nanmax(means, axis=0)
    elif rank_by == "mean":
        score = np.nanmean(means, axis=0)
    else:
        raise ValueError(f"rank_by must be 'max' or 'mean', got {rank_by!r}")

    order = np.argsort(-score)

    sel = []
    if top_k is not None:
        sel.extend(order[:top_k].tolist())
    else:
        sel.extend(order.tolist())

    if bottom_k is not None:
        for j in order[::-1][:bottom_k].tolist():
            if j not in sel:
                sel.append(j)

    order = np.array(sel, dtype=int)

    sel_keys = [keys[j] for j in order]
    sel_means = means[:, order]
    sel_errs = errs[:, order]

    x = np.arange(1, N + 1)

    colors = {
        "grid_major": "#D7D7D7",
        "grid_minor": "#EAEAEA",
        "text": "#222222",
        "spine": "#BBBBBB",
    }

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        cmap_ = plt.get_cmap(cmap)
        K_sel = len(sel_keys)
        line_colors = [cmap_(i / max(K_sel - 1, 1)) for i in range(K_sel)]

        # ------------------------------------------------------------
        # Vertical grid lines at every N
        # ------------------------------------------------------------
        for x_i in x:
            ax.axvline(
                x_i,
                color=colors["grid_major"],
                linestyle="-",
                linewidth=0.65,
                alpha=0.28,
                label="_nolegend_",
                zorder=0,
            )

        # ------------------------------------------------------------
        # Faded background — all unselected keys
        # ------------------------------------------------------------
        if show_unselected:
            sel_set = set(order.tolist())
            for j in range(K):
                if j in sel_set:
                    continue
                ax.plot(
                    x,
                    means[:, j],
                    "-",
                    linewidth=0.6,
                    color="#999999",
                    alpha=0.18,
                    label="_nolegend_",
                    zorder=1,
                )

        # ------------------------------------------------------------
        # Lines + error bars
        # ------------------------------------------------------------
        for j, key in enumerate(sel_keys):
            if jitter:
                x_j = x + (j - K_sel / 2) * jitter / max(K_sel, 1)
            else:
                x_j = x

            ax.errorbar(
                x_j,
                sel_means[:, j],
                yerr=sel_errs[:, j],
                fmt="-o",
                markersize=4.2,
                linewidth=1.5,
                capsize=2.8,
                capthick=0.8,
                elinewidth=0.8,
                color=line_colors[j],
                alpha=0.88,
                label=key,
                zorder=3,
            )

        # ------------------------------------------------------------
        # Axis styling
        # ------------------------------------------------------------
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in x])

        ax.set_xlim(0.5, N + 0.5)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_ylabel(ylabel)

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

        ax.spines["left"].set_color(colors["spine"])
        ax.spines["bottom"].set_color(colors["spine"])

        ax.tick_params(
            axis="both",
            colors=colors["text"],
            length=4,
            width=0.8,
        )

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        ncol = 1 if K_sel <= 20 else 2

        if bottom_k:
            legend_title = f"top {top_k or '∞'} + bottom {bottom_k} by {rank_by}"
        else:
            legend_title = f"top {K_sel} by {rank_by}"

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.015, 0.5),
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
            ncol=ncol,
            title=legend_title,
            title_fontsize=9,
            columnspacing=1.2,
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

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------
        right_margin = 0.78 if ncol == 1 else 0.70

        fig.tight_layout(
            rect=(0.0, 0.0, right_margin, 0.90 if (title or subtitle) else 1.0)
        )

    return fig, ax