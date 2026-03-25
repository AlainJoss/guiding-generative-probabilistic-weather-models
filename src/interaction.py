
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import marimo as mo

import geopandas as gpd
import geodatasets
from wigglystuff import ChartPuck



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
    """
    Prepare ERA5 121x240 field for plotting on lon in [-180, 180].

    Parameters
    ----------
    array_2d : np.ndarray
        Shape (121, 240)
    undo_roll : bool
        Set True only if the incoming array is still in the rolled format.

    Returns
    -------
    dict with:
        array_plot : reordered array
        lon_c_plot : longitude centers in [-180, 180]
        lat_c      : latitude centers
        lon_e_plot : longitude edges in [-180, 180]
        lat_e_plot : latitude edges
        sort_idx   : sorting index used to move from ERA5 [0,360] to plot [-180,180]
    """
    ny, nx = array_2d.shape
    assert (ny, nx) == (121, 240), f"Expected (121, 240), got {(ny, nx)}"

    if undo_roll:
        array_unrolled = np.roll(array_2d, -nx // 2, axis=1)
    else:
        array_unrolled = array_2d

    lon_e = np.linspace(0.0, 360.0, nx + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_c_plot = ((lon_c + 180.0) % 360.0) - 180.0
    sort_idx = np.argsort(lon_c_plot)

    lon_c_plot = lon_c_plot[sort_idx]
    array_plot = array_unrolled[:, sort_idx]

    dlon = lon_c_plot[1] - lon_c_plot[0]
    lon_e_plot = np.concatenate([
        [lon_c_plot[0] - 0.5 * dlon],
        0.5 * (lon_c_plot[:-1] + lon_c_plot[1:]),
        [lon_c_plot[-1] + 0.5 * dlon],
    ])

    dlat = lat_c[0] - lat_c[1]
    lat_e_plot = np.concatenate([
        [lat_c[0] + 0.5 * dlat],
        0.5 * (lat_c[:-1] + lat_c[1:]),
        [lat_c[-1] - 0.5 * dlat],
    ])

    return {
        "array_plot": array_plot,
        "lon_c_plot": lon_c_plot,
        "lat_c": lat_c,
        "lon_e_plot": lon_e_plot,
        "lat_e_plot": lat_e_plot,
        "sort_idx": sort_idx,
    }


def reorder_mask_for_plot(mask_2d: np.ndarray, sort_idx: np.ndarray, undo_roll: bool = False):
    """
    Reorder a mask from ERA5 storage coordinates [0,360] to plot coordinates [-180,180].

    Parameters
    ----------
    mask_2d : np.ndarray
        Shape (121, 240), built on ERA5 lon centers in [0, 360]
    sort_idx : np.ndarray
        Longitude sorting index from prepare_era5_plot_grid
    undo_roll : bool
        If the mask was produced on rolled data coordinates, set True.
        Usually False for masks built from get_mask().
    """
    ny, nx = mask_2d.shape
    assert (ny, nx) == (121, 240), f"Expected mask shape (121, 240), got {(ny, nx)}"

    if undo_roll:
        mask_unrolled = np.roll(mask_2d, -nx // 2, axis=1)
    else:
        mask_unrolled = mask_2d

    mask_plot = mask_unrolled[:, sort_idx]
    return mask_plot


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
    mask_linewidth=1.5,
    mask_with_points=False,
    undo_roll=False,
    show=True,
):
    """
    Non-interactive experiment plot.
    
    Parameters
    ----------
    array_2d : np.ndarray
        Field to plot, shape (121, 240)
    mask_2d : np.ndarray or None
        Mask in ERA5 storage coordinates, shape (121, 240)
    undo_roll : bool
        Applies to array_2d only. Usually False.
    """
    grid = prepare_era5_plot_grid(array_2d, undo_roll=undo_roll)
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
        mask_plot = reorder_mask_for_plot(
            mask_2d,
            sort_idx=grid["sort_idx"],
            undo_roll=False,
        )
        draw_mask_outline(
            ax,
            mask_plot=mask_plot,
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
    undo_roll=False,
):
    """
    Interactive wrapper around the same visual helpers.
    Requires `mo` and `ChartPuck` to exist in the notebook/app environment.
    """
    grid = prepare_era5_plot_grid(array_2d, undo_roll=undo_roll)
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

        ax.plot(
            [lon_left, lon_right],
            [lat_top, lat_bottom],
            "o",
            color="red",
            markersize=5,
            zorder=11,
        )

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
    undo_roll=False,
    mask_2d=None,
    show=False,
):
    if interactive:
        return make_interactive_map(
            array_2d,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            title=title,
            undo_roll=undo_roll,
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
        undo_roll=undo_roll,
        show=show,
    )