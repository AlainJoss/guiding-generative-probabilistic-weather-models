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
    y_trajectory: list[float],
    guidance_trajectory: list[float],
    var: str,
    ymin_left: float | None = None,
    ymax_left: float | None = None,
):
    y = np.asarray(y_trajectory, dtype=float)
    g = np.asarray(guidance_trajectory, dtype=float)

    if len(y) != len(g):
        raise ValueError("y_trajectory and guidance_trajectory must have the same length")

    x = np.arange(len(y))

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)

    ax1.plot(x, y, "b-", linewidth=2)
    ax1.plot(x, y, "o", color="#9c27b0", markersize=5)

    ax1.set_xlim((-0.5, 0.5) if len(y) == 1 else (0, len(y) - 1))
    ax1.set_xticks(np.arange(len(y)))
    ax1.set_xlabel("N")
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

    return fig


def plot_trajectory(
    trajectory: list[np.float64],
    var: str,
    ymin: float | None = None,
    ymax: float | None = None,
):
    x = np.arange(len(trajectory))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.plot(x, trajectory, "b-", linewidth=2)
    ax.plot(x, trajectory, "o", color="#9c27b0", markersize=5)
    ax.set_xlim(0, len(trajectory) - 1)
    ax.set_xticks(np.arange(len(trajectory)))
    ax.set_xlabel("steps")
    ax.set_ylabel(f"{var}")
    ax.set_title("Trajectory")
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

    print(array_2d_plot.max())

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