import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Experiment builder
    """)
    return


@app.cell
def _():
    import os
    from datetime import timedelta

    import marimo as mo
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    import earthkit.plots as ekp
    import cartopy.crs as ccrs

    from src.utils import ensure_rollout_dir, get_now_timestamp, dump_json, get_xr_dataset
    from src.mask import get_mask_2d, get_mask_center, effective_bbox
    from src.ui.map import visualize_map, to_display_units
    from src.ui.helpers import get_timestamp_from_sliders, max_day
    from src.rollout_config import RolloutConfig

    SURFACE_PAIR = {"temperature": "2m_temperature",
                    "u_component_of_wind": "10m_u_component_of_wind",
                    "v_component_of_wind": "10m_v_component_of_wind"}
    return (
        RolloutConfig,
        SURFACE_PAIR,
        ccrs,
        dump_json,
        effective_bbox,
        ekp,
        ensure_rollout_dir,
        get_mask_2d,
        get_mask_center,
        get_now_timestamp,
        get_xr_dataset,
        mcolors,
        mo,
        np,
        plt,
        timedelta,
        to_display_units,
        visualize_map,
        xr,
    )


@app.cell
def _(mcolors, np, plt):
    white_zero_cmap = plt.get_cmap("RdBu_r").copy()
    white_zero_cmap.set_bad("white")
    # single-signed fields use only the warm/cool half (white stays at 0), like guidance.py
    warm_half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_warm", white_zero_cmap(np.linspace(0.5, 1.0, 256)))
    cool_half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_cool", white_zero_cmap(np.linspace(0.0, 0.5, 256)))

    def abs_style(arr):
        """(cmap, vmin, vmax, center) for an absolute field, guidance.py convention:
        white anchored at 0 for straddling ranges; single-signed -> warm/cool half."""
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if vmin < 0.0 < vmax:
            return white_zero_cmap, vmin, vmax, 0.0
        if vmin >= 0.0:
            return warm_half_cmap, vmin, vmax, None
        return cool_half_cmap, vmin, vmax, None

    def add_map_stats(map_obj, arr):
        """Stamp min/max (left) and mean/std (right) at title height, like guidance.py.
        Static maps are (fig, ax) tuples; interactive widgets are returned unchanged."""
        if map_obj is None or not isinstance(map_obj, tuple):
            return map_obj
        _, ax = map_obj
        a = np.asarray(arr, dtype=float)
        if np.isfinite(a).any():
            ax.set_title(f"min = {np.nanmin(a):.3g} | max = {np.nanmax(a):.3g}", loc="left", fontsize=8)
            ax.set_title(f"mean = {np.nanmean(a):.3g} | std = {np.nanstd(a):.3g}", loc="right", fontsize=8)
        return map_obj

    return abs_style, add_map_stats


@app.cell
def _(ensure_rollout_dir, get_now_timestamp):
    save_id = get_now_timestamp()   # one rollout id for the session

    def experiment_dir():
        # created lazily (only when a button actually writes), shared by both writers
        return ensure_rollout_dir(save_id)

    return (experiment_dir,)


@app.cell
def _(mo):
    get_center, set_center = mo.state((-4.0, 40.0))
    return get_center, set_center


@app.cell
def _(mo):
    mask_mode_dropdown = mo.ui.dropdown(["BBOX", "ELLIPTICAL"], value="BBOX", label="mask_mode: ")
    side_lon_slider = mo.ui.slider(1.5, 90, step=1.5, value=12, label="lon side: ", show_value=True, debounce=True)
    side_lat_slider = mo.ui.slider(1.5, 90, step=1.5, value=10, label="lat side: ", show_value=True, debounce=True)
    sigma_div_slider = mo.ui.slider(steps=[0.25, 0.5, 1, 2, 4], value=2, label="sigma div: ", show_value=True)
    zoom_slider = mo.ui.slider(1, 12, step=1, value=1, label="zoom: ", show_value=True)
    dpi_slider = mo.ui.slider(steps=[60, 100, 140, 200], value=100, label="dpi: ", show_value=True)
    var_dropdown = mo.ui.dropdown(
        ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature",
         "specific_humidity", "vertical_velocity", "mean_sea_level_pressure"],
        value="temperature", label="var: ")
    level_slider = mo.ui.slider(steps=[0, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
                                value=0, label="level: ", show_value=True)
    M_slider = mo.ui.slider(1, 10, value=3, step=1, label="M: ", show_value=True)
    T_slider = mo.ui.slider(1, 25, value=25, step=1, label="T: ", show_value=True)
    return (
        M_slider,
        T_slider,
        dpi_slider,
        level_slider,
        mask_mode_dropdown,
        side_lat_slider,
        side_lon_slider,
        sigma_div_slider,
        var_dropdown,
        zoom_slider,
    )


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="save config")
    return (run_button,)


@app.cell
def _(SURFACE_PAIR, level_slider, var_dropdown):
    def resolve_var_level(base, lvl):
        if lvl == 0:
            if base in SURFACE_PAIR:
                return SURFACE_PAIR[base], "surface", 0
            if base == "mean_sea_level_pressure":
                return base, "surface", 0
            return base, "level", 50           # invalid: level var at the surface tick
        if base == "mean_sea_level_pressure":
            return base, "surface", 0          # invalid: msl at a pressure level
        return base, "level", lvl

    var, partition, level = resolve_var_level(var_dropdown.value, int(level_slider.value))
    return level, partition, var


@app.cell
def _(
    effective_bbox,
    get_center,
    get_mask_2d,
    get_mask_center,
    mask_mode_dropdown,
    side_lat_slider,
    side_lon_slider,
    sigma_div_slider,
):
    def build_mask():
        center_lon, center_lat = get_center()
        corners = (
            center_lon - side_lon_slider.value / 2, center_lon + side_lon_slider.value / 2,
            center_lat - side_lat_slider.value / 2, center_lat + side_lat_slider.value / 2,
        )
        weights = get_mask_2d(mask_mode_dropdown.value, corners, sigma_div=sigma_div_slider.value)
        drawn = effective_bbox(*corners, sigma_div=sigma_div_slider.value)  # snapped box = pixel edges
        return weights, drawn

    mask, mask_corners = build_mask()
    zoom_centers = get_mask_center(*mask_corners)   # recenter the maps on the box
    return mask, mask_corners, zoom_centers


@app.cell
def _(clim_ds, get_slices, get_xr_dataset, hw, np, to_display_units):
    def weather_field():
        # the 2m_temperature field at the heatwave PEAK (the max slice from the right chart),
        # rolled onto the mask grid and in display units; falls back to the 2020 mean if no heatwave
        if hw is None or clim_ds is None:
            return to_display_units(get_xr_dataset(2020)["2m_temperature"].mean("time").to_numpy(), "2m_temperature")[0]
        slices = get_slices(clim_ds, "surface", "2m_temperature", 0)          # (time,121,240) geographic, K
        ti = int(np.argmin(np.abs(clim_ds.time.values - np.datetime64(hw["peak_ts"]))))
        peak = np.roll(slices[ti], slices.shape[-1] // 2, axis=-1)            # align to the mask grid (rolled)
        return to_display_units(peak, "2m_temperature")[0]

    bg_disp = weather_field()
    return (bg_disp,)


@app.cell
def _(
    abs_style,
    add_map_stats,
    bg_disp,
    dpi_slider,
    hw,
    mask,
    mask_corners,
    mask_mode_dropdown,
    set_center,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    def build_weather_map():
        cmap, vmin, vmax, center = abs_style(bg_disp)
        m = visualize_map(
            bg_disp,
            suptitle=(f"2m_temperature  ·  heatwave peak {hw['peak_ts']:%Y-%m-%d}"
                      if hw is not None else "2m_temperature (2020 mean)"),
            title="click to set box center",
            interactive=True,
            cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            puck_center=((mask_corners[0] + mask_corners[1]) / 2,
                         (mask_corners[2] + mask_corners[3]) / 2),
            side_lon=mask_corners[1] - mask_corners[0],
            side_lat=mask_corners[3] - mask_corners[2],
            contour_2d=None if mask_mode_dropdown.value == "BBOX" else mask,
            contour_levels=8, contour_color="black", contour_linewidth=0.5,
            zoom=zoom_slider.value,   # 1 = full map (all grid points); >1 zooms into the box
            zoom_center_lon=zoom_centers[0], zoom_center_lat=zoom_centers[1],
            figsize=(14, 8), dpi=dpi_slider.value,
        )
        m = add_map_stats(m, bg_disp)
        m.widget.observe(
            lambda change: set_center((m.widget.x[0], m.widget.y[0])), names=["x", "y"])
        return m

    weather_map = build_weather_map()
    return (weather_map,)


@app.cell
def _(
    abs_style,
    ccrs,
    crs_mode_dropdown,
    ekp,
    get_center,
    make_crs,
    mask_corners,
    np,
    temp_field,
    zoom_centers,
    zoom_slider,
):
    def domain_globe():
        center_lon, center_lat = get_center()
        m = ekp.Map(crs=make_crs(crs_mode_dropdown.value, center_lon, center_lat))
        cmap, vmin, vmax, center = abs_style(temp_field.values)
        if center == 0.0:                      # straddles 0 -> symmetric so white sits at 0
            absmax = max(abs(vmin), abs(vmax))
            vmin, vmax = -absmax, absmax
        m.pcolormesh(temp_field, cmap=cmap, vmin=vmin, vmax=vmax)
        m.coastlines()
        m.legend(label="2m_temperature [°C]")
        ax = m.fig.axes[0]
        lon_l, lon_r, lat_b, lat_t = mask_corners
        # densely sample the edges (const-lat top/bottom, const-lon sides) so the outline follows
        # the parallels/meridians and curves with the projection instead of 4 straight chords
        _n = 120
        _lon, _lat = np.linspace(lon_l, lon_r, _n), np.linspace(lat_b, lat_t, _n)
        _bx = np.concatenate([_lon, np.full(_n, lon_r), _lon[::-1], np.full(_n, lon_l)])
        _by = np.concatenate([np.full(_n, lat_b), _lat, np.full(_n, lat_t), _lat[::-1]])
        ax.plot(_bx, _by, color="red", linewidth=1.5, transform=ccrs.PlateCarree(), zorder=10)
        # zoom=1 -> full projected globe; zoom>1 -> gradually zoom in on the box.
        zoom = max(1, int(zoom_slider.value))
        if zoom > 1:
            if crs_mode_dropdown.value == "PlateCarree":
                # flat map: geographic crop centered on the box (no limb to worry about)
                lon_span, lat_span = 360.0 / zoom, 180.0 / zoom
                cx, cy = zoom_centers
                ax.set_extent([max(-180.0, cx - lon_span / 2), min(180.0, cx + lon_span / 2),
                               max(-90.0, cy - lat_span / 2), min(90.0, cy + lat_span / 2)],
                              crs=ccrs.PlateCarree())
            else:
                # perspective globe: a wide lon/lat extent hits the limb and distorts, so
                # shrink the projected disk about its centre (= the box) by 1/zoom instead
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                hx, hy = (x1 - x0) / (2 * zoom), (y1 - y0) / (2 * zoom)
                ax.set_xlim(xm - hx, xm + hx)
                ax.set_ylim(ym - hy, ym + hy)
        return m.fig

    domain_map = domain_globe()
    return (domain_map,)


@app.cell
def _(
    abs_style,
    ccrs,
    crs_mode_dropdown,
    ekp,
    get_center,
    make_crs,
    mask,
    mask_corners,
    np,
    xr,
    zoom_centers,
    zoom_slider,
):
    def globe_mask():
        # the mask WEIGHTS on the globe (like domain_globe, but plotting the mask instead of T)
        center_lon, center_lat = get_center()
        m = ekp.Map(crs=make_crs(crs_mode_dropdown.value, center_lon, center_lat))
        _le = np.linspace(-180, 180, 241); _lc = 0.5 * (_le[:-1] + _le[1:])   # display grid (geographic)
        _la = np.linspace(90, -90, 122);   _lac = 0.5 * (_la[:-1] + _la[1:])
        mask_da = xr.DataArray(mask, dims=("latitude", "longitude"),
                               coords={"latitude": _lac, "longitude": _lc})
        cmap, vmin, vmax, center = abs_style(mask)
        m.pcolormesh(mask_da, cmap=cmap, vmin=vmin, vmax=vmax)
        m.coastlines()
        m.legend(label="mask weights")
        ax = m.fig.axes[0]
        ax.set_title("mask weights")
        lon_l, lon_r, lat_b, lat_t = mask_corners
        _n = 120
        _lon, _lat = np.linspace(lon_l, lon_r, _n), np.linspace(lat_b, lat_t, _n)
        _bx = np.concatenate([_lon, np.full(_n, lon_r), _lon[::-1], np.full(_n, lon_l)])
        _by = np.concatenate([np.full(_n, lat_b), _lat, np.full(_n, lat_t), _lat[::-1]])
        ax.plot(_bx, _by, color="red", linewidth=1.5, transform=ccrs.PlateCarree(), zorder=10)
        zoom = max(1, int(zoom_slider.value))
        if zoom > 1:
            if crs_mode_dropdown.value == "PlateCarree":
                lon_span, lat_span = 360.0 / zoom, 180.0 / zoom
                cx, cy = zoom_centers
                ax.set_extent([max(-180.0, cx - lon_span / 2), min(180.0, cx + lon_span / 2),
                               max(-90.0, cy - lat_span / 2), min(90.0, cy + lat_span / 2)],
                              crs=ccrs.PlateCarree())
            else:
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                hx, hy = (x1 - x0) / (2 * zoom), (y1 - y0) / (2 * zoom)
                ax.set_xlim(xm - hx, xm + hx); ax.set_ylim(ym - hy, ym + hy)
        return m.fig

    globe_mask_fig = globe_mask()
    return (globe_mask_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(
    M_slider,
    N_slider,
    T_slider,
    build_result,
    level_slider,
    mo,
    run_button,
    var_dropdown,
):
    mo.vstack([
        mo.hstack([var_dropdown, level_slider], justify="start"),
        mo.hstack([M_slider, N_slider, T_slider], justify="start"),
        mo.hstack([run_button], justify="start"),
        build_result,
    ], align="start")
    return


@app.cell
def _(
    crs_mode_dropdown,
    domain_map,
    dpi_slider,
    globe_mask_fig,
    mask_map,
    mask_mode_dropdown,
    mo,
    side_lat_slider,
    side_lon_slider,
    sigma_div_slider,
    weather_map,
    zoom_slider,
):
    mo.vstack([
        mo.md("## Mask"),
        mo.hstack([mask_mode_dropdown, side_lon_slider, side_lat_slider, sigma_div_slider,
                   zoom_slider, dpi_slider], justify="start"),
        mo.hstack([crs_mode_dropdown], justify="start"),
        mo.hstack([mo.vstack([weather_map, mask_map]), globe_mask_fig, domain_map],
                  justify="start", align="start"),
    ], align="start")
    return


@app.cell
def _(mo):
    crs_mode_dropdown = mo.ui.dropdown(
        ["NearsidePerspective", "PlateCarree"],
        value="NearsidePerspective", label="projection: ")
    return (crs_mode_dropdown,)


@app.cell
def _(ccrs, get_xr_dataset, np, xr):
    def make_crs(mode, lon, lat):
        builders = {
            "NearsidePerspective": lambda: ccrs.NearsidePerspective(central_latitude=lat, central_longitude=lon),
            "Orthographic": lambda: ccrs.Orthographic(central_longitude=lon, central_latitude=lat),
            "PlateCarree": lambda: ccrs.PlateCarree(central_longitude=lon),
            "Robinson": lambda: ccrs.Robinson(central_longitude=lon),
            "Mollweide": lambda: ccrs.Mollweide(central_longitude=lon),
            "LambertAzimuthalEqualArea": lambda: ccrs.LambertAzimuthalEqualArea(central_longitude=lon, central_latitude=lat),
        }
        return builders[mode]()

    def geo_temp_field():
        # get_xr_dataset returns arches_era5.nc, whose lon content is rolled 180 (Europe-centered).
        # unroll it so content matches the 0-360 labels for the geographic cartopy plot.
        da = get_xr_dataset(2020)["2m_temperature"].mean("time") - 273.15  # degC
        unrolled = np.roll(da.values, da.shape[-1] // 2, axis=-1)
        return xr.DataArray(unrolled, coords=da.coords, dims=da.dims)

    temp_field = geo_temp_field()
    return make_crs, temp_field


@app.cell
def _(xr):
    from src.paths import GT
    from src.mask import get_masked_mean
    from src.utils import get_slices
    import pandas as pd
    import matplotlib.dates as mdates

    clim_path = GT / "clim_2m_temperature.nc"
    clim_ds = xr.open_dataset(clim_path).load() if clim_path.exists() else None
    return clim_ds, clim_path, get_masked_mean, get_slices, mdates


@app.cell
def _(
    clim_ds,
    clim_path,
    dpi_slider,
    get_masked_mean,
    get_slices,
    highlight_year_slider,
    hw,
    mask,
    mdates,
    mo,
    np,
    pl,
    plt,
    rolling_window_slider,
):
    def region_climatology_plot():
        if clim_ds is None:
            return mo.md(f"_climatology file not found — run `python -m src.download_climatology --out {clim_path}`_")
        slices = get_slices(clim_ds, "surface", "2m_temperature", 0)   # (time,121,240), lon 0-360
        slices = np.roll(slices, slices.shape[-1] // 2, axis=-1)        # align to the mask grid (-180..180)
        v_all = get_masked_mean(slices, mask) - 273.15
        t = clim_ds.time.values                                        # numpy datetime64
        years_arr = t.astype("datetime64[Y]").astype(int) + 1970
        doy_all = np.clip((t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1, 1, 365)
        window = int(rolling_window_slider.value)
        sel = int(highlight_year_slider.value)
        years = sorted(set(int(y) for y in years_arr))
        ref = np.datetime64("2001-01-01")
        smooth = lambda a: pl.Series(np.asarray(a, float)).rolling_mean(window_size=window, min_samples=1, center=True).to_numpy()
        xdate = lambda d: ref + (np.asarray(d) - 1).astype("timedelta64[D]")

        def year_series(yr):                                # (doy, v) for a year, sorted by doy
            mk = years_arr == yr
            d, vv = doy_all[mk], v_all[mk]
            o = np.argsort(d)
            return d[o], vv[o]

        fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi_slider.value)
        for yr in years:                                     # non-neighbour years: light gray
            if yr in (sel - 1, sel, sel + 1):
                continue
            d, vv = year_series(yr)
            ax.plot(xdate(d), smooth(vv), color="#cfcfcf", linewidth=0.8, alpha=1, zorder=1)
        # previous (yellow, shaded), next (dark brown, shaded), selected (red, solid, on top).
        # sel-1 / sel+1 only exist in `years` when present -> 2020 has no prev, 2026 no next.
        for yr, color, lw, a, z in [(sel - 1, "#f2a900", 1.7, 0.4, 3),
                                    (sel + 1, "#7a1f1f", 1.7, 0.25, 3),
                                    (sel,     "#e63946", 2.1, 1.0, 6)]:
            if yr in years:
                d, vv = year_series(yr)
                ax.plot(xdate(d), smooth(vv), color=color, linewidth=lw, alpha=a, label=str(yr), zorder=z)
        uniq_doy = np.unique(doy_all)
        mean_v = np.array([v_all[doy_all == d].mean() for d in uniq_doy])
        ax.plot(xdate(uniq_doy), smooth(mean_v), color="#444444", linewidth=1.6, linestyle="--",
                label=f"{years[0]}-{years[-1]} average", zorder=5)
        ax.plot([], [], color="#cfcfcf", linewidth=1.2, label="other years")

        # month guide lines at each month start, drawn as a uniform x-grid on minor ticks.
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1))    # gridlines at month starts
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=16))   # mid-month -> centered labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(False, axis="x", which="major")   # no mid-month lines
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xlim(ref, ref + np.timedelta64(365, "D"))
        ax.set_ylabel("mask-avg 2m_temperature [°C]")
        ax.set_title(f"Region surface air temperature  ·  daily average"
                     + (f"  ·  {window}-day rolling" if window > 1 else ""), loc="left", fontsize=13)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=0.16, top=0.93, left=0.09, right=0.97)
        # mark the highlighted year's rank-th peak (temperature + date), and the start/end
        if hw is not None:
            _d = lambda doy_: ref + int(doy_ - 1) * np.timedelta64(1, "D")
            peak_x, start_x, end_x = _d(hw["peak_doy"]), _d(hw["start_doy"]), _d(hw["end_doy"])
            ax.axvline(start_x, color="#e63946", linewidth=0.8, linestyle=":", zorder=3)         # start day
            ax.axvline(end_x, color="#e63946", linewidth=0.9, linestyle=":", zorder=3)          # heatwave end
            ax.plot([end_x], [hw["end_val"]], "o", color="#e63946", markersize=6, zorder=7)
            ax.annotate(f"{hw['end_val']:.1f}°C", xy=(end_x, hw["end_val"]), xytext=(0, -13),
                        textcoords="offset points", ha="center", color="#e63946", fontsize=10, fontweight="bold", zorder=8)
            ax.plot([peak_x], [hw["peak_val"]], "o", color="#e63946", markersize=6, zorder=7)
            ax.annotate(f"{hw['peak_val']:.1f}°C", xy=(peak_x, hw["peak_val"]), xytext=(0, 7),
                        textcoords="offset points", ha="center", color="#e63946", fontsize=10,
                        fontweight="bold", zorder=8)
            ax.annotate(f"{hw['peak_ts']:%d %b %Y}", xy=(peak_x, hw["peak_val"]), xytext=(0, 21),
                        textcoords="offset points", ha="center", color="#e63946", fontsize=9, zorder=8)  # date over the max temp
            ax.plot([start_x], [hw["start_val"]], "o", color="#e63946", markersize=6, zorder=7)
            ax.annotate(f"{hw['start_val']:.1f}°C", xy=(start_x, hw["start_val"]), xytext=(0, -14),
                        textcoords="offset points", ha="center", color="#e63946", fontsize=10, fontweight="bold", zorder=8)
        _ylo, _yhi = ax.get_ylim()
        ax.set_ylim(_ylo, _yhi + 0.10 * (_yhi - _ylo))   # headroom so the peak temp + date clear the title
        return fig

    region_clim_plot = region_climatology_plot()
    return (region_clim_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Climatology
    """)
    return


@app.cell
def _(
    add_button,
    clear_button,
    dates,
    highlight_year_slider,
    mo,
    peak_rank_slider,
    region_clim_plot,
    region_trajectory_fig,
    rolling_window_slider,
    start_day_slider,
):
    mo.vstack([
        mo.hstack([highlight_year_slider, peak_rank_slider, rolling_window_slider], justify="start"),
        mo.hstack([
            region_clim_plot,
            mo.vstack([
                start_day_slider,
                mo.hstack([add_button, clear_button], justify="start"),
                mo.md("**picked dates:** " + (", ".join(d.strftime("%Y-%m-%d") for d in dates) if dates else "_none_")),
                region_trajectory_fig,
            ], align="start"),
        ], justify="start", align="end"),
    ], align="start")
    return


@app.cell
def _(mo):
    rolling_window_slider = mo.ui.slider(1, 31, value=1, step=1,
        label="rolling window (days): ", show_value=True, debounce=True)
    return (rolling_window_slider,)


@app.cell
def _(clim_ds, mo, np):
    def make_year_slider():
        if clim_ds is None:
            return mo.ui.slider(2020, 2026, value=2026, step=1, label="highlight year: ", show_value=True)
        yrs = np.unique(clim_ds.time.values.astype("datetime64[Y]").astype(int) + 1970)
        return mo.ui.slider(int(yrs.min()), int(yrs.max()), value=int(yrs.max()), step=1,
                            label="highlight year: ", show_value=True)

    highlight_year_slider = make_year_slider()
    return (highlight_year_slider,)


@app.cell
def _(mo):
    peak_rank_slider = mo.ui.slider(1, 5, value=1, step=1, label="peak rank: ", show_value=True)
    return (peak_rank_slider,)


@app.cell
def _(
    clim_ds,
    get_masked_mean,
    get_slices,
    highlight_year_slider,
    mask,
    np,
    peak_rank_slider,
    pl,
    rolling_window_slider,
):
    from scipy.signal import find_peaks

    def heatwave_analysis():
        """For the highlighted year: the rolling region-avg series, its rank-th peak,
        and the heatwave start (walk back from the peak to the local min)."""
        if clim_ds is None:
            return None
        slices = get_slices(clim_ds, "surface", "2m_temperature", 0)
        slices = np.roll(slices, slices.shape[-1] // 2, axis=-1)        # align to mask grid (-180..180)
        series = get_masked_mean(slices, mask) - 273.15
        t = clim_ds.time.values                                        # numpy datetime64
        yr = int(highlight_year_slider.value)
        m = (t.astype("datetime64[Y]").astype(int) + 1970) == yr
        doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1
        doy = np.clip(doy[m], 1, 365)
        order_doy = np.argsort(doy)
        times = t[m][order_doy]                     # datetime64, sorted by doy
        doy = doy[order_doy]
        v_raw = series[m][order_doy]
        v = pl.Series(v_raw).rolling_mean(window_size=int(rolling_window_slider.value),
                                          min_samples=1, center=True).to_numpy()
        peaks, _ = find_peaks(v)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(v))])
        ranked = peaks[np.argsort(v[peaks])[::-1]]                 # local maxima, tallest first
        p = int(ranked[min(int(peak_rank_slider.value), len(ranked)) - 1])
        i = p                                                       # heatwave start: back to local min
        while i > 0 and v[i - 1] < v[i]:
            i -= 1
        j = p                       # heatwave end: forward to local min
        while j < len(v) - 1 and v[j + 1] < v[j]:
            j += 1
        _pyts = lambda k: times[k].astype("datetime64[s]").astype(object)   # numpy -> python datetime
        return dict(year=yr, doy=doy, v=v, v_raw=v_raw, peak_i=p, start_i=i, end_i=j,
                    start_ts=_pyts(i), end_ts=_pyts(j), peak_ts=_pyts(p),
                    peak_doy=int(doy[p]), peak_val=float(v[p]),
                    start_doy=int(doy[i]), start_val=float(v[i]), end_doy=int(doy[j]), end_val=float(v[j]))

    hw = heatwave_analysis()
    return (hw,)


@app.cell
def _(
    N_slider,
    dpi_slider,
    hw,
    mo,
    np,
    plt,
    region_clim_plot,
    start_day_slider,
):
    def region_trajectory_plot():
        if hw is None:
            return mo.md("_no heatwave selected_")
        v, s, e = hw["v_raw"], hw["start_i"], hw["end_i"]
        n = int(start_day_slider.value)                  # selected start offset (0 = heatwave start)
        N = int(N_slider.value)
        dur = e - s
        lo_n, hi_n = min(-5, n), max(dur + 5, n + N)     # x-range: heatwave +/- margin and the horizon
        lo, hi = max(0, s + lo_n), min(len(v), s + hi_n + 1)
        seg = v[lo:hi]
        steps = np.arange(lo - s, hi - s)
        size = tuple(region_clim_plot.get_size_inches()) if hasattr(region_clim_plot, "get_size_inches") else (14, 8)
        fig, ax = plt.subplots(figsize=size, dpi=dpi_slider.value)
        ax.axvspan(n, n + N, color="#457b9d", alpha=0.12, zorder=1)                # rollout horizon [n, n+N]
        ax.axvline(n, color="#457b9d", linewidth=1.4, linestyle="--", zorder=4)     # selected start
        ax.plot(steps, seg, color="#e63946", linewidth=2.1, zorder=6)
        ax.plot([0, hw["peak_i"] - s, e - s], [v[s], v[hw["peak_i"]], v[e]], "o",
                color="#e63946", markersize=6, zorder=7)
        for px, py in [(0, v[s]), (e - s, v[e])]:
            ax.annotate(f"{py:.1f}°C", xy=(px, py), xytext=(0, -14), textcoords="offset points",
                        ha="center", color="#e63946", fontsize=10, fontweight="bold", zorder=8)
        ax.axvline(0, color="#e63946", linewidth=0.8, linestyle=":", zorder=3)
        ax.axvline(e - s, color="#e63946", linewidth=0.9, linestyle=":", zorder=3)
        ax.axvline(hw["peak_i"] - s, color="#e63946", linewidth=0.9, linestyle=":", zorder=3)
        ax.annotate(f"{v[hw['peak_i']]:.1f}°C", xy=(hw["peak_i"] - s, v[hw["peak_i"]]),
                    xytext=(0, 7), textcoords="offset points", ha="center", color="#e63946",
                    fontsize=10, fontweight="bold", zorder=8)
        ax.set_xlabel("n  (days from heatwave start)")
        ax.set_ylabel("mask-avg 2m_temperature [°C]")
        ax.set_title(f"Masked-average over heatwave  ·  start n={n}, N={N}", loc="left", fontsize=13)
        tick = steps if len(steps) <= 20 else steps[::(len(steps) // 20 + 1)]
        ax.set_xticks(tick)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.9)
        ax.set_axisbelow(True)
        fig.subplots_adjust(bottom=0.16, top=0.93, left=0.09, right=0.97)
        _ylo, _yhi = ax.get_ylim()
        ax.set_ylim(_ylo, _yhi + 0.05 * (_yhi - _ylo))
        return fig

    region_trajectory_fig = region_trajectory_plot()
    return (region_trajectory_fig,)


@app.cell
def _(mo):
    N_slider = mo.ui.slider(1, 30, value=15, step=1, label="N (rollout days): ", show_value=True)
    return (N_slider,)


@app.cell
def _(hw, mo):
    def make_start_day():
        if hw is None:
            return mo.ui.slider(0, 1, value=0, label="start day (n):")
        dur = int(hw["end_i"] - hw["start_i"])
        return mo.ui.slider(start=-10, stop=dur + 10, step=1, value=0, show_value=True,
                            full_width=True, label="start day (n, 0 = heatwave start):")

    start_day_slider = make_start_day()
    return (start_day_slider,)


@app.cell
def _(mo):
    get_dates, set_dates = mo.state([])
    return get_dates, set_dates


@app.cell
def _(hw, start_day_slider, timedelta):
    current_date = (hw["start_ts"] + timedelta(days=int(start_day_slider.value))
                    if hw is not None else None)
    return (current_date,)


@app.cell
def _(current_date, get_dates, mo, set_dates):
    add_button = mo.ui.button(
        label="add start",
        on_click=lambda _: set_dates(sorted(set(
            get_dates() + ([current_date] if current_date is not None else [])))))
    clear_button = mo.ui.button(label="clear dates", on_click=lambda _: set_dates([]))
    return add_button, clear_button


@app.cell
def _(get_dates):
    dates = get_dates()
    return (dates,)


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(
    abs_style,
    add_map_stats,
    dpi_slider,
    mask,
    mask_mode_dropdown,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    def build_mask_map():
        # the mask weights (arches lat weights, sum to 1): white at 0, warm toward the max
        cmap, vmin, vmax, center = abs_style(mask)
        m = visualize_map(
            mask,
            suptitle="mask",
            title="mask weights",
            interactive=False,
            cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            contour_2d=None if mask_mode_dropdown.value == "BBOX" else mask,
            contour_levels=8, contour_color="black", contour_linewidth=0.5,
            zoom=zoom_slider.value,   # 1 = full map (all grid points); >1 zooms into the box
            zoom_center_lon=zoom_centers[0], zoom_center_lat=zoom_centers[1],
            figsize=(11.2, 8), dpi=dpi_slider.value,
        )
        return add_map_stats(m, mask)[0]   # (fig, ax) -> fig for display

    mask_map = build_mask_map()
    return (mask_map,)


@app.cell
def _(
    M_slider,
    N_slider,
    RolloutConfig,
    T_slider,
    dates,
    dump_json,
    experiment_dir,
    hw,
    level,
    mask_corners,
    mo,
    partition,
    run_button,
    var,
):
    def write_config():
        if not dates or hw is None:
            return mo.md("_pick a start and click 'add start'_")
        fmt = "%Y-%m-%d_%H:%M:%S"                     # subdir/id convention (matches get_now_timestamp)
        outer = experiment_dir()
        N = int(N_slider.value)
        names = [d.strftime(fmt) for d in dates]
        # OUTER config = the picked date list + rollout length. run.py loops STARTS;
        # `python -m src.download --rollout_id <id>` derives the per-date windows from STARTS + N.
        dump_json({"STARTS": names, "N": N}, outer / "config.json")
        for d, name in zip(dates, names):
            cfg = RolloutConfig(
                M=M_slider.value, N=N, T=T_slider.value, START_TS=d,
                PARTITION=partition, LEVEL=level, VAR=var, MASK_CORNERS=list(mask_corners))
            (outer / name).mkdir(parents=True, exist_ok=True)
            dump_json(cfg.to_dict(), outer / name / "config.json")
        return mo.md(
            f"✅ saved {len(names)} dates (N={N}) under `{outer}`\n\n"
            f"download the model input with `python -m src.download --rollout_id {outer.name}`")

    def build_experiment():
        # the button only writes the configs; the model input is fetched separately via
        # `python -m src.download --rollout_id <exp_id>`
        return write_config()

    build_result = (build_experiment() if run_button.value
                    else mo.md(""))
    return (build_result,)


if __name__ == "__main__":
    app.run()
