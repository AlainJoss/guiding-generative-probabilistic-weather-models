import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import marimo as mo

import geopandas as gpd
import geodatasets
from wigglystuff import ChartPuck


def get_mask_corners_from_widget(map_widget):
    x0, x1 = map_widget.value["x"]
    y0, y1 = map_widget.value["y"]

    lon_left, lon_right = sorted([x0, x1])
    lat_bottom, lat_top = sorted([y0, y1])

    return lon_left, lon_right, lat_bottom, lat_top


def get_mask_from_corners(lon_left, lon_right, lat_bottom, lat_top):
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    lon_mask = (lon_grid >= lon_left) & (lon_grid <= lon_right)
    lat_mask = (lat_grid >= lat_bottom) & (lat_grid <= lat_top)

    mask = (lon_mask & lat_mask).astype(np.float32)
    return torch.as_tensor(mask)

def plot_dual_trajectory(
    timestamps: list[str],
    y_trajectory: list[float],
    guidance_trajectory: dict[str, float],
    var: str,
    ymin_left: float | None = None,
    ymax_left: float | None = None,
):
    y = np.asarray(y_trajectory, dtype=float)
    g = np.asarray(guidance_trajectory, dtype=float)

    x = np.arange(len(timestamps))

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)

    ax1.plot(x, y, "b-", linewidth=2)
    ax1.plot(x, y, "o", color="#9c27b0", markersize=5)

    ax1.set_xlim((-0.5, 0.5) if len(y) == 1 else (0, len(y) - 1))

    xtick_positions = {0, len(timestamps) - 1}
    xtick_positions.update(
        i for i, ts in enumerate(timestamps) if str(ts).endswith("00:00:00")
    )
    xtick_positions = sorted(xtick_positions)
    xtick_labels = [timestamps[i] for i in xtick_positions]

    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Percentage change")
    ax1.set_title("Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.5)

    cur_ymin, cur_ymax = ax1.get_ylim()
    left_min = ymin_left if ymin_left is not None else cur_ymin
    left_max = ymax_left if ymax_left is not None else cur_ymax
    if left_min == left_max:
        left_min -= 1
        left_max += 1
    ax1.set_ylim(left_min, left_max)

    ax2 = ax1.twinx()
    ax2.plot(x, g, "-", linewidth=2)
    ax2.plot(x, g, "s", markersize=4)
    ax2.set_ylabel(var)

    y0, y1 = np.nanmin(y), np.nanmax(y)
    g0, g1 = np.nanmin(g), np.nanmax(g)

    if y0 == y1:
        y0 -= 1
        y1 += 1
    if g0 == g1:
        g0 -= 1
        g1 += 1

    def map_left_to_right(v):
        return g0 + (v - y0) * (g1 - g0) / (y1 - y0)

    ax2.set_ylim(map_left_to_right(left_min), map_left_to_right(left_max))

    fig.tight_layout()
    return fig


def plot_trajectories_over_n(
    trajectories: list[list[np.float64]],
    var: str = "mask term",
    ymin: float | None = None,
    ymax: float | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    figsize_per_row: tuple[float, float] = (7.0, 1.8),
    dpi: int = 160,
    shared_y: bool = True,
):
    N = len(trajectories)
    fig, axes = plt.subplots(
        N,
        1,
        figsize=(figsize_per_row[0], N * figsize_per_row[1]),
        dpi=dpi,
        squeeze=False,
        sharex=False,
        sharey=shared_y,
    )

    if shared_y:
        flat = [v for traj in trajectories for v in traj]
        g_min = float(np.nanmin(flat))
        g_max = float(np.nanmax(flat))
        pad = 0.05 * (g_max - g_min if g_max > g_min else 1.0)
        y_lo = ymin if ymin is not None else g_min - pad
        y_hi = ymax if ymax is not None else g_max + pad
    else:
        y_lo, y_hi = ymin, ymax

    line_color = "#1f77b4"
    marker_color = "#9c27b0"

    for i, traj in enumerate(trajectories):
        ax = axes[i, 0]
        x = np.arange(len(traj))
        ax.plot(x, traj, "-", color=line_color, linewidth=1.6, alpha=0.9)
        ax.plot(x, traj, "o", color=marker_color, markersize=3.5, zorder=3)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.7)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_xlim(0, max(len(traj) - 1, 1))
        ax.set_ylabel(f"N={i + 1}", rotation=0, ha="right", va="center", labelpad=18)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if y_lo is not None and y_hi is not None:
            ax.set_ylim(y_lo, y_hi)
        if i < N - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("diffusion step")

    fig.supylabel(var, fontsize=10)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=9, color="#555")

    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.96 if (title or subtitle) else 1.0))
    return fig, axes


def plot_rmse_over_n(
    rmse_guided: np.ndarray,
    rmse_unguided: np.ndarray,
    err_guided: np.ndarray | None = None,
    err_unguided: np.ndarray | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    figsize: tuple[float, float] = (9, 4.5),
    dpi: int = 160,
    ylabel: str = "normalized RMSE (mask)",
    highlight_n: int | None = None,
):
    """Two lines of N points each: RMSE guided vs unguided across rollout steps."""
    rmse_guided = np.asarray(rmse_guided)
    rmse_unguided = np.asarray(rmse_unguided)
    N = len(rmse_guided)
    x = np.arange(1, N + 1)

    C = {"guided": "#1f77b4", "unguided": "#d62728"}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _draw(y, err, color, marker, label):
        if err is not None:
            err = np.asarray(err)
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.15, zorder=1)
            ax.errorbar(
                x, y, yerr=err,
                fmt="-" + marker, linewidth=1.8, markersize=5,
                capsize=3, capthick=0.8, elinewidth=0.8,
                color=color, alpha=0.95, label=label, zorder=3,
            )
        else:
            ax.plot(x, y, "-" + marker, color=color, linewidth=1.8,
                    markersize=5, alpha=0.95, label=label, zorder=3)

    _draw(rmse_unguided, err_unguided, C["unguided"], "s", "unguided")
    _draw(rmse_guided, err_guided, C["guided"], "o", "guided")

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in x], fontsize=9)
    ax.set_xlim(0.5, N + 0.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("rollout step", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="best", frameon=False, fontsize=9)

    if highlight_n is not None and 1 <= highlight_n <= N:
        ax.axvline(highlight_n, color="black", linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter(
            [highlight_n], [rmse_guided[highlight_n - 1]],
            s=70, facecolor="white", edgecolor=C["guided"],
            linewidth=1.8, zorder=4,
        )
        ax.scatter(
            [highlight_n], [rmse_unguided[highlight_n - 1]],
            s=70, facecolor="white", edgecolor=C["unguided"],
            linewidth=1.8, zorder=4,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93 if (title or subtitle) else 1.0))
    _place_axes_title(fig, ax, title=title, subtitle=subtitle)
    return fig, ax


def _place_axes_title(fig, ax, *, title=None, subtitle=None,
                      title_fontsize=13, subtitle_fontsize=9):
    if not (title or subtitle):
        return
    bbox = ax.get_position()
    cx = (bbox.x0 + bbox.x1) / 2
    top = bbox.y1
    fig_h = fig.get_figheight()
    dy_title = 0.35 / fig_h
    dy_sub = 0.20 / fig_h
    if title and subtitle:
        fig.text(cx, top + dy_title + dy_sub, title,
                 ha="center", va="bottom",
                 fontsize=title_fontsize, fontweight="bold")
        fig.text(cx, top + dy_sub * 0.6, subtitle,
                 ha="center", va="bottom",
                 fontsize=subtitle_fontsize, color="#555")
    elif title:
        fig.text(cx, top + dy_sub, title, ha="center", va="bottom",
                 fontsize=title_fontsize, fontweight="bold")
    else:
        fig.text(cx, top + dy_sub, subtitle, ha="center", va="bottom",
                 fontsize=subtitle_fontsize, color="#555")


def plot_variable_change_parallel(
    aggregated_per_n: list[dict],
    top_k: int | None = 20,
    rank_by: str = "max",
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 160,
    title: str | None = None,
    subtitle: str | None = None,
    cmap: str = "viridis",
    jitter: float = 0.0,
    ylim: tuple[float, float] | None = (0, 1.05),
    ylabel: str = "min-max score",
):
    """Parallel-coordinates view of per-variable-level min-max scores across N.

    Each key (var-level) is a line across rollout steps n=1..N; markers at
    each N show the ensemble mean, with vertical error bars (std or sem
    depending on what was aggregated). Top-k ranks keys by the chosen
    statistic across N ('max' or 'mean').
    """
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
    if top_k is not None:
        order = order[:top_k]

    sel_keys = [keys[j] for j in order]
    sel_means = means[:, order]
    sel_errs = errs[:, order]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.arange(1, N + 1)
    cmap_ = plt.get_cmap(cmap)
    K_sel = len(sel_keys)
    colors = [cmap_(i / max(K_sel - 1, 1)) for i in range(K_sel)]

    for j, key in enumerate(sel_keys):
        x_j = x + (j - K_sel / 2) * jitter / max(K_sel, 1) if jitter else x
        ax.errorbar(
            x_j,
            sel_means[:, j],
            yerr=sel_errs[:, j],
            fmt="-o",
            markersize=4,
            linewidth=1.3,
            capsize=2.5,
            capthick=0.8,
            elinewidth=0.8,
            color=colors[j],
            alpha=0.85,
            label=key,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in x], fontsize=9)
    ax.set_xlim(0.5, N + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    ncol = 1 if K_sel <= 20 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        ncol=ncol,
        title=f"top {K_sel} by {rank_by}",
        title_fontsize=9,
    )

    right = 0.78 if ncol == 1 else 0.70
    fig.tight_layout(rect=(0.0, 0.0, right, 0.93 if (title or subtitle) else 1.0))
    _place_axes_title(fig, ax, title=title, subtitle=subtitle)
    return fig, ax


def plot_trajectory(
    trajectory: list[np.float64],
    var: str,
    ymin: float | None = None,
    ymax: float | None = None,
    title: str | None = "Trajectory"
):
    x = np.arange(len(trajectory))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.plot(x, trajectory, "b-", linewidth=2)
    ax.plot(x, trajectory, "o", color="#9c27b0", markersize=5)
    ax.set_xlim(0, len(trajectory) - 1)
    ax.set_xticks(np.arange(len(trajectory)))
    ax.set_xlabel("step")
    ax.set_ylabel(f"{var}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    current_ymin, current_ymax = ax.get_ylim()
    final_ymin = ymin if ymin is not None else current_ymin
    final_ymax = ymax if ymax is not None else current_ymax
    ax.set_ylim(final_ymin, final_ymax)

    return fig


# ------------------------------------------------------------
# ERA5 grid preparation
# ------------------------------------------------------------
def prepare_era5_plot_grid(array_2d: np.ndarray, undo_roll: bool = False):
    ny, nx = array_2d.shape
    assert (ny, nx) == (121, 240), f"Expected (121, 240), got {(ny, nx)}"

    lon_e = np.linspace(-180.0, 180.0, nx + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    return {
        "array_plot": array_2d,
        "lon_c_plot": lon_c,
        "lat_c": lat_c,
        "lon_e_plot": lon_e,
        "lat_e_plot": lat_e,
    }


def make_norm(array_2d_plot, vmin=None, vmax=None, center=None):
    if center is not None:
        if vmin is None or vmax is None:
            absmax = np.nanmax(np.abs(array_2d_plot))
            if not np.isfinite(absmax) or absmax == 0:
                absmax = 1e-8
            if vmin is None:
                vmin = -absmax
            if vmax is None:
                vmax = absmax
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ------------------------------------------------------------
# mask helpers
# ------------------------------------------------------------
def mask_bbox_from_active_cells(mask_plot: np.ndarray, lon_e_plot: np.ndarray, lat_e_plot: np.ndarray):
    active = mask_plot > 0
    if not np.any(active):
        return None

    ii, jj = np.where(active)

    i_min, i_max = ii.min(), ii.max()
    j_min, j_max = jj.min(), jj.max()

    lon_left = lon_e_plot[j_min]
    lon_right = lon_e_plot[j_max + 1]
    lat_top = lat_e_plot[i_min]
    lat_bottom = lat_e_plot[i_max + 1]

    return lon_left, lon_right, lat_bottom, lat_top


def compute_fit_zoom(mask_2d, *, max_zoom=12, padding=1.15):
    """Integer zoom factor that frames the mask bbox.

    Matches apply_zoom's integer zoom convention so the slider can
    be snapped directly to the return value.
    """
    geom = prepare_era5_plot_grid(np.asarray(mask_2d).astype(float))
    bbox = mask_bbox_from_active_cells(
        geom["array_plot"] > 0,
        geom["lon_e_plot"],
        geom["lat_e_plot"],
    )
    if bbox is None:
        return 1
    lon_l, lon_r, lat_b, lat_t = bbox
    lon_span = max(lon_r - lon_l, 1.0) * padding
    lat_span = max(lat_t - lat_b, 1.0) * padding
    z = max(360.0 / lon_span, 180.0 / lat_span)
    return max(1, min(int(z), max_zoom))


def draw_mask_outline(
    ax,
    *,
    mask_plot,
    lon_e_plot,
    lat_e_plot,
    edgecolor="red",
    linewidth=1.5,
    with_points=False,
):
    bbox = mask_bbox_from_active_cells(mask_plot, lon_e_plot, lat_e_plot)
    if bbox is None:
        return None

    lon_left, lon_right, lat_bottom, lat_top = bbox

    rect = mpatches.Rectangle(
        (lon_left, lat_bottom),
        lon_right - lon_left,
        lat_top - lat_bottom,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=10,
    )
    ax.add_patch(rect)

    if with_points:
        ax.plot(
            [lon_left, lon_right],
            [lat_top, lat_bottom],
            "o",
            color=edgecolor,
            markersize=5,
            zorder=11,
        )

    return rect


def get_mask_center(mask_2d):
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    bbox = mask_bbox_from_active_cells(np.asarray(mask_2d), lon_e, lat_e)
    if bbox is None:
        return 0.0, 0.0

    lon_left, lon_right, lat_bottom, lat_top = bbox
    return 0.5 * (lon_left + lon_right), 0.5 * (lat_bottom + lat_top)


# ------------------------------------------------------------
# zoom helper
# ------------------------------------------------------------
def apply_zoom(
    ax,
    *,
    zoom: int,
    center_lon: float = 0.0,
    center_lat: float = 0.0,
):
    """
    zoom=1  -> full map
    zoom>1  -> zoom into (center_lon, center_lat) by shrinking the visible window
    """
    zoom = max(1, int(zoom))

    full_lon_span = 360.0
    full_lat_span = 180.0

    lon_span = full_lon_span / zoom
    lat_span = full_lat_span / zoom

    lon_min = center_lon - lon_span / 2
    lon_max = center_lon + lon_span / 2
    lat_min = center_lat - lat_span / 2
    lat_max = center_lat + lat_span / 2

    lon_min = max(-180.0, lon_min)
    lon_max = min(180.0, lon_max)
    lat_min = max(-90.0, lat_min)
    lat_max = min(90.0, lat_max)

    if lon_max - lon_min < lon_span:
        if lon_min <= -180.0:
            lon_max = min(180.0, lon_min + lon_span)
        elif lon_max >= 180.0:
            lon_min = max(-180.0, lon_max - lon_span)

    if lat_max - lat_min < lat_span:
        if lat_min <= -90.0:
            lat_max = min(90.0, lat_min + lat_span)
        elif lat_max >= 90.0:
            lat_min = max(-90.0, lat_max - lat_span)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

def annotate_cell_values(
    ax,
    *,
    array_2d_plot,
    lon_c_plot,
    lat_c,
    fmt=".2f",
    fontsize=6,
    color="black",
    threshold=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
):
    ii, jj = np.where(np.isfinite(array_2d_plot))

    for i, j in zip(ii, jj):
        lon = lon_c_plot[j]
        lat = lat_c[i]
        value = array_2d_plot[i, j]

        if threshold is not None and abs(value) < threshold:
            continue

        if lon_min is not None and lon < lon_min:
            continue
        if lon_max is not None and lon > lon_max:
            continue
        if lat_min is not None and lat < lat_min:
            continue
        if lat_max is not None and lat > lat_max:
            continue
            
        ax.text(
            lon,
            lat,
            format(value, fmt),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=20,
            clip_on=True,
        )

# ------------------------------------------------------------
# visual-only plotting
# ------------------------------------------------------------
def draw_base_map(
    ax,
    *,
    array_2d_plot=None,
    lon_e_plot=None,
    lat_e_plot=None,
    lon_c_plot=None,
    lat_c=None,
    cmap="coolwarm",
    norm=None,
    title=None,
    add_colorbar=True,
    world=None,
    show_values=False,
    value_fmt=".2f",
    value_fontsize=6,
    value_color="black",
    value_threshold=None,
    value_lon_min=None,
    value_lon_max=None,
    value_lat_min=None,
    value_lat_max=None,
):
    im = ax.pcolormesh(
        lon_e_plot,
        lat_e_plot,
        array_2d_plot,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    if world is None:
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    world.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=5)
    if show_values:
       annotate_cell_values(
            ax,
            array_2d_plot=array_2d_plot,
            lon_c_plot=lon_c_plot,
            lat_c=lat_c,
            fmt=value_fmt,
            fontsize=value_fontsize,
            color=value_color,
            threshold=value_threshold,
            lon_min=value_lon_min,
            lon_max=value_lon_max,
            lat_min=value_lat_min,
            lat_max=value_lat_max,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if title is not None:
        ax.set_title(title)

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        ax.figure.colorbar(im, cax=cax)

    return im


def plot_map_static(
    array_2d,
    *,
    mask_2d=None,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    center=None,
    figsize=(10, 5),
    dpi=100,
    title=None,
    mask_edgecolor="red",
    mask_linewidth=1.0,
    mask_with_points=False,
    show=True,
    show_mask=False,
    zoom=1,
    zoom_center_lon=0.0,
    zoom_center_lat=0.0,
    show_values=False,
    value_fmt=".2f",
    value_fontsize=6,
    value_color="black",
    value_threshold=None,
    value_lon_min=None,
    value_lon_max=None,
    value_lat_min=None,
    value_lat_max=None,
):
    grid = prepare_era5_plot_grid(array_2d)
    norm = make_norm(grid["array_plot"], vmin=vmin, vmax=vmax, center=center)
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    zoom_ = max(1, int(zoom))

    full_lon_span = 360.0
    full_lat_span = 180.0

    lon_span = full_lon_span / zoom_
    lat_span = full_lat_span / zoom_

    lon_min = max(-180.0, zoom_center_lon - lon_span / 2)
    lon_max = min(180.0, zoom_center_lon + lon_span / 2)
    lat_min = max(-90.0, zoom_center_lat - lat_span / 2)
    lat_max = min(90.0, zoom_center_lat + lat_span / 2)

    draw_base_map(
        ax,
        array_2d_plot=grid["array_plot"],
        lon_e_plot=grid["lon_e_plot"],
        lat_e_plot=grid["lat_e_plot"],
        lon_c_plot=grid["lon_c_plot"],
        lat_c=grid["lat_c"],
        cmap=cmap,
        norm=norm,
        title=title,
        add_colorbar=True,
        world=world,
        show_values=show_values,
        value_fmt=value_fmt,
        value_fontsize=value_fontsize,
        value_color=value_color,
        value_threshold=value_threshold,
    )

    if mask_2d is not None and show_mask:
        draw_mask_outline(
            ax,
            mask_plot=np.asarray(mask_2d),
            lon_e_plot=grid["lon_e_plot"],
            lat_e_plot=grid["lat_e_plot"],
            edgecolor=mask_edgecolor,
            linewidth=mask_linewidth,
            with_points=mask_with_points,
        )

    apply_zoom(
        ax,
        zoom=zoom,
        center_lon=zoom_center_lon,
        center_lat=zoom_center_lat,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


# ------------------------------------------------------------
# interactive layer only
# ------------------------------------------------------------
def make_interactive_map(
    array_2d,
    *,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    center=None,
    title=None,
    rectangle_x=(-10.0, 2.0),
    rectangle_y=(45.0, 35.0),
):
    grid = prepare_era5_plot_grid(array_2d)
    norm = make_norm(grid["array_plot"], vmin=vmin, vmax=vmax, center=center)
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    def draw_map(ax, widget):
        ax.clear()

        fig = ax.figure
        while len(fig.axes) > 1:
            fig.delaxes(fig.axes[-1])

        draw_base_map(
            ax,
            array_2d_plot=grid["array_plot"],
            lon_e_plot=grid["lon_e_plot"],
            lat_e_plot=grid["lat_e_plot"],
            cmap=cmap,
            norm=norm,
            title=title,
            add_colorbar=True,
            world=world,
        )

        x0, x1 = widget.x
        y0, y1 = widget.y

        lon_left, lon_right = sorted([x0, x1])
        lat_bottom, lat_top = sorted([y0, y1])

        rect = mpatches.Rectangle(
            (lon_left, lat_bottom),
            lon_right - lon_left,
            lat_top - lat_bottom,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
            zorder=10,
        )
        ax.add_patch(rect)

    return mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_map,
            x=list(rectangle_x),
            y=list(rectangle_y),
            puck_color=["green", "red"],
            figsize=(12, 5.7),
            x_bounds=(-180.0, 180.0),
            y_bounds=(-90.0, 90.0),
        )
    )


# ------------------------------------------------------------
# wrapper
# ------------------------------------------------------------
def visualize_map(
    array_2d,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    center=0,
    figsize=(10, 5),
    interactive=False,
    title=None,
    dpi=200,
    mask_2d=None,
    show=False,
    show_mask=False,
    zoom=1,
    zoom_center_lon=0.0,
    zoom_center_lat=0.0,
    show_values=False,
    value_fmt=".2f",
    value_fontsize=6,
    value_color="black",
    value_threshold=None,
):
    if interactive:
        return make_interactive_map(
            array_2d,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            title=title,
        )

    return plot_map_static(
        array_2d,
        mask_2d=mask_2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        figsize=figsize,
        dpi=dpi,
        title=title,
        show=show,
        show_mask=show_mask,
        zoom=zoom,
        zoom_center_lon=zoom_center_lon,
        zoom_center_lat=zoom_center_lat,
        show_values=show_values,
        value_fmt=value_fmt,
        value_fontsize=value_fontsize,
        value_color=value_color,
        value_threshold=value_threshold,
    )


_NATIVE_UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "temperature": "K",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "geopotential": "m^2/s^2",
    "specific_humidity": "kg/kg",
}


def to_display_units(array, var, *, is_difference=False):
    """Convert a slice to display units.

    For 2m_temperature in absolute mode: Kelvin -> Celsius.
    For difference panels: values are unchanged (ΔK ≡ Δ°C), only the
    label changes to °C so the colorbar reads naturally.

    Returns (array_display, unit_label).
    """
    native = _NATIVE_UNITS.get(var, "")
    if var == "2m_temperature":
        if is_difference:
            return array, "°C"
        return np.asarray(array) - 273.15, "°C"
    return array, native


def visualize_grid(
    panels,
    *,
    nrows,
    ncols,
    vmin,
    vmax,
    center=None,
    cmap="coolwarm",
    unit_label=None,
    mask_2d=None,
    show_mask=False,
    zoom=1,
    zoom_center_lon=0.0,
    zoom_center_lat=0.0,
    figsize_per_panel=(4.8, 3.4),
    dpi=150,
    tick_formatter=None,
    show_values=False,
    value_fmt=".2f",
    value_fontsize=5,
    value_color="black",
    value_threshold=None,
):
    """Grid of map panels sharing one figure-level colorbar.

    panels: sequence of (title, array_2d). Length must equal nrows*ncols.
    vmin/vmax/center: shared norm across all panels.
    unit_label: appended to the colorbar label; if given, shown as
        "value [unit_label]".
    tick_formatter: optional matplotlib Formatter applied to the
        colorbar ticks (e.g. to render Kelvin arrays as °C).
    """
    if len(panels) != nrows * ncols:
        raise ValueError(
            f"panels has {len(panels)} entries but grid is "
            f"{nrows}x{ncols}={nrows * ncols}"
        )

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    first_arr = np.asarray(panels[0][1])
    grid_geom = prepare_era5_plot_grid(first_arr)
    shared_norm = make_norm(
        grid_geom["array_plot"], vmin=vmin, vmax=vmax, center=center
    )

    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False
    )

    im = None
    for idx, (title, array_2d) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        geom = prepare_era5_plot_grid(array_2d)

        draw_base_map(
            ax,
            array_2d_plot=geom["array_plot"],
            lon_e_plot=geom["lon_e_plot"],
            lat_e_plot=geom["lat_e_plot"],
            lon_c_plot=geom["lon_c_plot"],
            lat_c=geom["lat_c"],
            cmap=cmap,
            norm=shared_norm,
            title=title,
            add_colorbar=False,
            world=world,
            show_values=show_values,
            value_fmt=value_fmt,
            value_fontsize=value_fontsize,
            value_color=value_color,
            value_threshold=value_threshold,
        )

        if mask_2d is not None and show_mask:
            draw_mask_outline(
                ax,
                mask_plot=np.asarray(mask_2d),
                lon_e_plot=geom["lon_e_plot"],
                lat_e_plot=geom["lat_e_plot"],
                edgecolor="red",
                linewidth=1.0,
                with_points=False,
            )

        apply_zoom(
            ax,
            zoom=zoom,
            center_lon=zoom_center_lon,
            center_lat=zoom_center_lat,
        )

        im = ax.collections[0]

    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    if unit_label:
        cbar.set_label(unit_label)
    if tick_formatter is not None:
        cbar.ax.yaxis.set_major_formatter(tick_formatter)

    return fig


def plot_states_over_n(
    *,
    ds,
    guided_rollout_dir,
    unguided_rollout_dir,
    get_rollout_path,
    read_state,
    get_slice,
    timestamp,
    timestamp_idx,
    N,
    m,
    partition,
    var,
    level=None,
    mask_2d=None,
    show_mask=False,
    analysis_type="absolute",
    cmap="coolwarm",
    figsize_per_panel=(4.8, 3.4),
    dpi=160,
    title=None,
    subtitle=None,
):
    if analysis_type == "absolute":
        col_titles = ["guided", "unguided", "ground truth", "current"]
    else:
        col_titles = [
            "ground truth - current",
            "guided - unguided",
            "ground truth - guided",
            "ground truth - unguided",
        ]

    fig, axes = plt.subplots(
        N,
        4,
        figsize=(4 * figsize_per_panel[0], N * figsize_per_panel[1]),
        dpi=dpi,
        squeeze=False,
    )

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    last_norm = None

    for n_iter in range(1, N + 1):
        guided_state = read_state(get_rollout_path(guided_rollout_dir, n_iter, m))
        unguided_state = read_state(get_rollout_path(unguided_rollout_dir, n_iter, m))

        sample = ds.denormalize(ds[timestamp_idx + n_iter])
        ts = sample["timestamp"].unsqueeze(0)
        current_state = ds.convert_to_xarray(sample["state"].unsqueeze(0), ts)
        ground_truth_state = ds.convert_to_xarray(sample["next_state"].unsqueeze(0), ts)

        guided_s = np.asarray(get_slice(guided_state, partition, level, var, timestamp))
        unguided_s = np.asarray(get_slice(unguided_state, partition, level, var, timestamp))
        gt_s = np.asarray(get_slice(ground_truth_state, partition, level, var, timestamp))
        cur_s = np.asarray(get_slice(current_state, partition, level, var, timestamp))

        if analysis_type == "absolute":
            arrays = {
                "guided": guided_s,
                "unguided": unguided_s,
                "ground truth": gt_s,
                "current": cur_s,
            }
            row_min = min(np.nanmin(a) for a in arrays.values())
            row_max = max(np.nanmax(a) for a in arrays.values())
            row_center = float(np.nanmean(cur_s))
            row_center = min(max(row_center, row_min + 1e-9), row_max - 1e-9)
        else:
            arrays = {
                "ground truth - current": gt_s - cur_s,
                "guided - unguided": guided_s - unguided_s,
                "ground truth - guided": gt_s - guided_s,
                "ground truth - unguided": gt_s - unguided_s,
            }
            absmax = max(np.nanmax(np.abs(a)) for a in arrays.values())
            absmax = absmax if absmax > 0 else 1e-8
            row_min, row_max, row_center = -absmax, absmax, 0.0

        for col, name in enumerate(col_titles):
            ax = axes[n_iter - 1, col]
            arr = arrays[name]

            grid = prepare_era5_plot_grid(arr)
            norm = make_norm(
                grid["array_plot"],
                vmin=row_min,
                vmax=row_max,
                center=row_center,
            )
            last_norm = norm

            draw_base_map(
                ax,
                array_2d_plot=grid["array_plot"],
                lon_e_plot=grid["lon_e_plot"],
                lat_e_plot=grid["lat_e_plot"],
                lon_c_plot=grid["lon_c_plot"],
                lat_c=grid["lat_c"],
                cmap=cmap,
                norm=norm,
                title=None,
                add_colorbar=False,
                world=world,
            )

            if mask_2d is not None and show_mask:
                draw_mask_outline(
                    ax,
                    mask_plot=np.asarray(mask_2d),
                    lon_e_plot=grid["lon_e_plot"],
                    lat_e_plot=grid["lat_e_plot"],
                    edgecolor="red",
                    linewidth=1.0,
                    with_points=False,
                )

            if mask_2d is not None:
                bbox = mask_bbox_from_active_cells(
                    np.asarray(mask_2d),
                    grid["lon_e_plot"],
                    grid["lat_e_plot"],
                )
                if bbox is not None:
                    lon_l, lon_r, lat_b, lat_t = bbox
                    ax.set_xlim(lon_l, lon_r)
                    ax.set_ylim(lat_b, lat_t)

            if n_iter == 1:
                ax.set_title(name)

            if col == 0:
                ax.set_ylabel(f"N={n_iter}")

    sm = plt.cm.ScalarMappable(norm=last_norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
    )
    cbar.set_label(var if level is None else f"{var} @ {level}")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=10)

    return fig, axes