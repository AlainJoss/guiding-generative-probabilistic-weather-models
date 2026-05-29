import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import marimo as mo
from matplotlib.ticker import FormatStrFormatter

import geopandas as gpd
import geodatasets
from wigglystuff import ChartPuck

plt.rcParams["font.family"] = "Menlo"    # INter


def get_mask_corners_from_widget(map_widget):
    x0, x1 = map_widget.value["x"]
    y0, y1 = map_widget.value["y"]

    lon_left, lon_right = sorted([x0, x1])
    lat_bottom, lat_top = sorted([y0, y1])

    return lon_left, lon_right, lat_bottom, lat_top



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
    ax.set_xlim(float(lon_e_plot[0]), float(lon_e_plot[-1]))
    ax.set_ylim(float(lat_e_plot[-1]), float(lat_e_plot[0]))
    ax.margins(0)

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
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    if title is not None:
        ax.set_title(title, fontsize=13)

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=9)

    return im


def plot_map_static(
    array_2d,
    *,
    mask_2d=None,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    center=None,
    figsize=(14, 5),
    dpi=100,
    title=None,
    suptitle=None,
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
    contour_2d=None,
    contour_levels=8,
    contour_color="red",
    contour_linewidth=0.8,
):
    grid = prepare_era5_plot_grid(array_2d)
    norm = make_norm(grid["array_plot"], vmin=vmin, vmax=vmax, center=center)
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.88)

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

    if contour_2d is not None:
        ax.contour(
            grid["lon_c_plot"],
            grid["lat_c"],
            np.asarray(contour_2d),
            levels=contour_levels,
            colors=contour_color,
            linewidths=contour_linewidth,
            zorder=6,
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

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.2)
        s.set_zorder(10)

    # plt.tight_layout()

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
    suptitle=None,
    rectangle_x=(-10.0, 2.0),
    rectangle_y=(45.0, 35.0),
):
    plt.rcParams["font.family"] = "Menlo"
    grid = prepare_era5_plot_grid(array_2d)
    norm = make_norm(grid["array_plot"], vmin=vmin, vmax=vmax, center=center)
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    def draw_map(ax, widget):
        ax.clear()

        fig = ax.figure
        while len(fig.axes) > 1:
            fig.delaxes(fig.axes[-1])

        if suptitle is not None:
            fig.suptitle(suptitle)

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
        # fig.tight_layout()

        # Important: do not call fig.tight_layout() here.

    return mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_map,
            x=list(rectangle_x),
            y=list(rectangle_y),
            puck_color=["green", "red"],
            figsize=(10.31, 5),
            x_bounds=(-180.0, 180.0),
            y_bounds=(-90.0, 90.0),
            throttle=1000,
            puck_radius=3,
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
    figsize=(15, 5),
    interactive=False,
    title=None,
    suptitle=None,
    dpi=100,
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
    rectangle_x=(-10.0, 2.0),
    rectangle_y=(45.0, 35.0),
    contour_2d=None,
    contour_levels=8,
    contour_color="red",
    contour_linewidth=0.8,
):
    if interactive:
        return make_interactive_map(
            array_2d,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            title=title,
            suptitle=suptitle,
            rectangle_x=rectangle_x,
            rectangle_y=rectangle_y,
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
        suptitle=suptitle,
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
        contour_2d=contour_2d,
        contour_levels=contour_levels,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
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
    dpi=100,
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

    # fig.tight_layout()
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    if unit_label:
        cbar.set_label(unit_label)
    if tick_formatter is not None:
        cbar.ax.yaxis.set_major_formatter(tick_formatter)

    return fig