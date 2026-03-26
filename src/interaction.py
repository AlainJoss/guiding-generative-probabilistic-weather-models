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


def plot_trajectory(trajectory: list[np.float64], var: str):
    x = np.arange(len(trajectory))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, trajectory, "b-", linewidth=2)
    ax.plot(x, trajectory, "o", color="#9c27b0", markersize=5)
    ax.set_xlim(0, len(trajectory) - 1)
    ax.set_xticks(np.arange(len(trajectory)))
    ax.set_xlabel("N")
    ax.set_ylabel(f"{var}")
    ax.set_title("Trajectory")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
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
    """
    Infer the outer rectangle from active mask cells in plot coordinates.

    Parameters
    ----------
    mask_plot : np.ndarray
        Shape (121, 240), already reordered to plot longitude order
    lon_e_plot : np.ndarray
        Longitude edges in plot coordinates
    lat_e_plot : np.ndarray
        Latitude edges

    Returns
    -------
    (lon_left, lon_right, lat_bottom, lat_top) or None if mask is empty
    """
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


# ------------------------------------------------------------
# visual-only plotting
# ------------------------------------------------------------

def draw_base_map(
    ax,
    *,
    array_2d_plot,
    lon_e_plot,
    lat_e_plot,
    cmap="coolwarm",
    norm=None,
    title=None,
    add_colorbar=True,
    world=None,
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

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
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
    mask_linewidth=0.5,
    mask_with_points=False,
    show=True,
    show_mask=False
):
    grid = prepare_era5_plot_grid(array_2d)
    norm = make_norm(grid["array_plot"], vmin=vmin, vmax=vmax, center=center)
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

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

    if mask_2d is not None:
        if show_mask:
            draw_mask_outline(
                ax,
                mask_plot=np.asarray(mask_2d),
                lon_e_plot=grid["lon_e_plot"],
                lat_e_plot=grid["lat_e_plot"],
                edgecolor=mask_edgecolor,
                linewidth=mask_linewidth,
                with_points=mask_with_points,
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
    show_mask=False
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
        show_mask=show_mask
    )