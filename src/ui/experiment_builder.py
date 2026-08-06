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
    from weatherbench2 import regridding

    from src.utils import ensure_rollout_dir, get_now_timestamp, dump_json, get_xr_dataset
    from src.mask import get_mask_2d, get_mask_center
    from src.ui.map import visualize_map, to_display_units
    from src.ui.helpers import get_timestamp_from_sliders, max_day
    from src.rollout_config import RolloutConfig

    ARCO_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    ARCO_STORAGE = {"token": "anon"}
    # ArchesWeather target grid (regrid needs ascending lat; flip to descending after)
    ARCHES_LON = np.arange(0, 360, 1.5)              # 240, ascending 0..358.5
    ARCHES_LAT_ASC = np.arange(-90, 90.001, 1.5)     # 121, ascending -90..90

    SURFACE = ["10m_u_component_of_wind", "10m_v_component_of_wind",
               "2m_temperature", "mean_sea_level_pressure"]
    LEVELVARS = ["geopotential", "u_component_of_wind", "v_component_of_wind",
                 "temperature", "specific_humidity", "vertical_velocity"]
    MODEL_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    UNITS = {
        "10m_u_component_of_wind": "m s**-1", "10m_v_component_of_wind": "m s**-1",
        "2m_temperature": "K", "mean_sea_level_pressure": "Pa",
        "geopotential": "m**2 s**-2", "u_component_of_wind": "m s**-1",
        "v_component_of_wind": "m s**-1", "temperature": "K",
        "specific_humidity": "kg kg**-1", "vertical_velocity": "Pa s**-1",
    }
    SURFACE_PAIR = {"temperature": "2m_temperature",
                    "u_component_of_wind": "10m_u_component_of_wind",
                    "v_component_of_wind": "10m_v_component_of_wind"}
    return (
        ARCHES_LAT_ASC,
        ARCHES_LON,
        ARCO_STORAGE,
        ARCO_URI,
        LEVELVARS,
        MODEL_LEVELS,
        SURFACE,
        SURFACE_PAIR,
        UNITS,
        ccrs,
        dump_json,
        ekp,
        ensure_rollout_dir,
        get_mask_center,
        get_now_timestamp,
        get_xr_dataset,
        mcolors,
        mo,
        np,
        os,
        plt,
        regridding,
        timedelta,
        to_display_units,
        xr,
    )


@app.cell
def _(
    ARCHES_LAT_ASC,
    ARCHES_LON,
    ARCO_STORAGE,
    ARCO_URI,
    LEVELVARS,
    MODEL_LEVELS,
    SURFACE,
    UNITS,
    regridding,
    xr,
):
    # ---- ARCO access + WeatherBench2 regrid + arches-schema netCDF ----
    def open_arco():
        return xr.open_zarr(ARCO_URI, storage_options=ARCO_STORAGE, chunks={})

    def sel_12z(ds):
        # WeatherBench2-style instantaneous synoptic subsampling, keeping only 12:00Z
        return ds.sel(time=ds.time.dt.hour == 12)

    def regrid_to_arches(ds):
        # WB2 ConservativeRegridder needs ascending lat + in-memory numpy
        ds = ds.sortby("latitude").load()
        source = regridding.Grid.from_degrees(lon=ds.longitude.values, lat=ds.latitude.values)
        target = regridding.Grid.from_degrees(lon=ARCHES_LON, lat=ARCHES_LAT_ASC)
        out = regridding.ConservativeRegridder(source, target).regrid_dataset(ds)
        return out.sortby("latitude", ascending=False).astype("float32")  # -> descending

    def regrid_batched(ds, size):
        batches = [regrid_to_arches(ds.isel(time=slice(i, i + size)))
                   for i in range(0, ds.sizes["time"], size)]
        return xr.concat(batches, dim="time")

    def to_arches_netcdf(ds, path, batch=2):
        # ARCO subset -> 1.5deg arches_era5.nc schema netCDF
        ds = ds[SURFACE + LEVELVARS].sel(level=MODEL_LEVELS)
        ds = regrid_batched(ds, size=batch)
        ds = ds.assign_coords(level=ds.level.astype("int64"))[SURFACE + LEVELVARS]
        # arches order: surface (time,lat,lon); level (time,level,lat,lon)
        ds = ds.transpose("time", "level", "latitude", "longitude", missing_dims="ignore")
        for name in ds.data_vars:
            ds[name].attrs["units"] = UNITS[name]
        ds.to_netcdf(path)
        return ds

    return open_arco, sel_12z, to_arches_netcdf


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
    side_lat_slider = mo.ui.slider(1.5, 60, step=1.5, value=10, label="lat side: ", show_value=True, debounce=True)
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
    run_button = mo.ui.run_button(label="save config + download data")
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
    get_center,
    get_mask_center,
    mask_mod,
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
        weights = mask_mod.get_mask_2d(mask_mode_dropdown.value, corners, sigma_div=sigma_div_slider.value)
        drawn = mask_mod.effective_bbox(*corners, sigma_div=sigma_div_slider.value)  # snapped box = pixel edges
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
    map_mod,
    mask,
    mask_corners,
    mask_mode_dropdown,
    set_center,
    zoom_centers,
    zoom_slider,
):
    def build_weather_map():
        cmap, vmin, vmax, center = abs_style(bg_disp)
        m = map_mod.visualize_map(
            bg_disp,
            suptitle=(f"2m_temperature  ·  heatwave peak {hw['peak_ts']:%Y-%m-%d}"
                      if hw is not None else "2m_temperature (2020 mean)"),
            title="click to set the box center",
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
    M_slider,
    T_slider,
    dump_json,
    exp_starts,
    experiment_dir,
    hw,
    level,
    mask_corners,
    mo,
    open_arco,
    os,
    partition,
    run_button,
    sel_12z,
    timedelta,
    to_arches_netcdf,
    var,
    xr,
):
    def write_config():
        if exp_starts is None or not exp_starts["starts"] or hw is None:
            return mo.md("_no heatwave / start range selected_")
        import importlib
        import src.rollout_config as rc
        importlib.reload(rc)                          # pick up new RolloutConfig fields in the live kernel
        fmt = "%Y-%m-%d_%H:%M:%S"                     # subdir/id convention (matches get_now_timestamp)
        outer = experiment_dir()
        end_ts = exp_starts["end_ts"]
        win_start = exp_starts["window_start"]
        hw_s = hw["start_ts"]
        hw_e = hw["end_ts"]
        starts = exp_starts["starts"]
        names = [s.strftime(fmt) for s in starts]
        # OUTER config: just the start sweep + common end; run.py loops STARTS and runs each subdir
        dump_json({"STARTS": names, "END_TS": rc.datetime_to_string(end_ts)},
                  outer / "config.json")
        for s, name in zip(starts, names):
            N = (end_ts - s).days                                   # per-start horizon to END_TS
            cfg = rc.RolloutConfig(
                M=M_slider.value, N=int(N), T=T_slider.value, START_TS=s,
                WINDOW_START=win_start, WINDOW_END=end_ts,
                HEATWAVE_START=hw_s, HEATWAVE_END=hw_e,
                PARTITION=partition, LEVEL=level, VAR=var, MASK_CORNERS=list(mask_corners))
            (outer / name).mkdir(parents=True, exist_ok=True)
            dump_json(cfg.to_dict(), outer / name / "config.json")
        return mo.md(f"✅ saved outer config + {len(names)} start configs under `{outer}`")

    def download_model_input():
        if exp_starts is None or not exp_starts["starts"]:
            return mo.md("_no start range selected_")
        path = str(experiment_dir() / "era5_input.nc")
        if os.path.exists(path):
            ds = xr.open_dataset(path)
        else:
            window = sel_12z(open_arco()).sel(
                time=slice(exp_starts["window_start"] - timedelta(days=1), exp_starts["end_ts"]))
            ds = to_arches_netcdf(window, path, batch=2)
        return mo.md(
            f"✅ `{path}`\n\nsteps={ds.sizes['time']} (WINDOW_START−1d .. END_TS) · "
            f"vars={len(ds.data_vars)} · levels={ds.sizes.get('level')} · "
            f"grid={ds.sizes['latitude']}×{ds.sizes['longitude']}"
        )

    def build_experiment():
        # single button: write the per-start configs, then fetch + regrid the model input
        return mo.vstack([write_config(), download_model_input()], align="start")

    build_result = (build_experiment() if run_button.value
                    else mo.md(""))
    return (build_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(
    M_slider,
    T_slider,
    build_result,
    level_slider,
    mo,
    run_button,
    var_dropdown,
):
    mo.vstack([
        mo.hstack([var_dropdown, level_slider], justify="start"),
        mo.hstack([M_slider, T_slider], justify="start"),
        mo.hstack([run_button], justify="start"),
        build_result,
    ], align="start")
    return


@app.cell
def _(
    crs_mode_dropdown,
    domain_map,
    dpi_slider,
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
        mo.hstack([weather_map, mask_map, domain_map], justify="start", align="start"),
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

        fig, ax = plt.subplots(figsize=(14, 8), dpi=500)
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
    N_minus_slider,
    N_plus_slider,
    highlight_year_slider,
    mo,
    peak_rank_slider,
    region_clim_plot,
    region_trajectory_fig,
    rolling_window_slider,
    start_range_slider,
):
    mo.vstack([
        mo.hstack([highlight_year_slider, peak_rank_slider, rolling_window_slider], justify="start"),
        mo.hstack([
            region_clim_plot,
            mo.vstack([
                mo.hstack([N_plus_slider, N_minus_slider], justify="start"),
                start_range_slider,
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
    N_minus_slider,
    N_plus_slider,
    hw,
    mo,
    np,
    plt,
    region_clim_plot,
    start_range_slider,
):
    def region_trajectory_plot():
        if hw is None:
            return mo.md("_no heatwave selected_")
        v, s, Np, Nm = hw["v_raw"], hw["start_i"], int(N_plus_slider.value), int(N_minus_slider.value)
        e = hw["end_i"]                                     # heatwave end index
        lo, hi = max(0, s - Nm), min(len(v), e + Np + 1)    # N- before start .. N+ after end
        seg = v[lo:hi]
        steps = np.arange(lo - s, hi - s)                # x relative to start (0 = heatwave start)
        size = tuple(region_clim_plot.get_size_inches()) if hasattr(region_clim_plot, "get_size_inches") else (14, 8)
        fig, ax = plt.subplots(figsize=size, dpi=500)   # same size + dpi as the left chart
        fn, ln = (int(x) for x in start_range_slider.value)      # experiment-start sweep range (n)
        ax.axvspan(fn, ln, color="#457b9d", alpha=0.12, zorder=1)             # start-sweep band
        ax.axvline(fn, color="#457b9d", linewidth=1.1, linestyle="--", zorder=4)
        ax.axvline(ln, color="#457b9d", linewidth=1.1, linestyle="--", zorder=4)
        # trajectory drawn like the left chart's selected year: solid red, same weight
        ax.plot(steps, seg, color="#e63946", linewidth=2.1, zorder=6)
        pts_x = [0, hw["peak_i"] - s, e - s]                     # start, peak, end
        pts_y = [v[s], v[hw["peak_i"]], v[e]]                     # real-data temps there
        ax.plot(pts_x, pts_y, "o", color="#e63946", markersize=6, zorder=7)   # markers as on the left
        for px, py in [(0, v[s]), (e - s, v[e])]:                 # start & end temps (peak annotated above)
            ax.annotate(f"{py:.1f}°C", xy=(px, py), xytext=(0, -14), textcoords="offset points",
                        ha="center", color="#e63946", fontsize=10, fontweight="bold", zorder=8)
        ax.axvline(0, color="#e63946", linewidth=0.8, linestyle=":", zorder=3)  # heatwave start (n=0)
        ax.axvline(hw["end_i"] - s, color="#e63946", linewidth=0.9, linestyle=":", zorder=3)  # heatwave end
        ax.axvline(hw["peak_i"] - s, color="#e63946", linewidth=0.9, linestyle=":", zorder=3)  # peak
        ax.annotate(f"{v[hw['peak_i']]:.1f}°C", xy=(hw["peak_i"] - s, v[hw["peak_i"]]),
                    xytext=(0, 7), textcoords="offset points", ha="center", color="#e63946",
                    fontsize=10, fontweight="bold", zorder=8)
        ax.set_xlabel("n  (days from heatwave start)")
        ax.set_ylabel("mask-avg 2m_temperature [°C]")
        ax.set_title(f"Masked-average over heatwave", loc="left", fontsize=13)
        ax.set_xticks(steps)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.9)
        ax.set_axisbelow(True)
        fig.subplots_adjust(bottom=0.16, top=0.93, left=0.09, right=0.97)   # match left chart margins
        _ylo, _yhi = ax.get_ylim()
        ax.set_ylim(_ylo, _yhi + 0.05 * (_yhi - _ylo))   # same title headroom as the left chart
        return fig

    region_trajectory_fig = region_trajectory_plot()
    return (region_trajectory_fig,)


@app.cell
def _(mo):
    N_plus_slider = mo.ui.slider(1, 30, value=14, step=1, label="N⁺ (days after start): ", show_value=True)
    N_minus_slider = mo.ui.slider(0, 10, value=2, step=1, label="N⁻ (days before start): ", show_value=True)
    return N_minus_slider, N_plus_slider


@app.cell
def _(N_minus_slider, N_plus_slider, hw, start_range_slider, timedelta):
    def experiment_starts():
        if hw is None:
            return None
        hw_start = hw["start_ts"]                                    # python datetime
        Nm, Np = int(N_minus_slider.value), int(N_plus_slider.value)
        dur = int(hw["end_i"] - hw["start_i"])
        end_ts = hw_start + timedelta(days=dur + Np)               # common END_TS = WINDOW_END
        window_start = hw_start - timedelta(days=Nm)               # WINDOW_START
        first_n, last_n = (int(x) for x in start_range_slider.value)
        starts = [hw_start + timedelta(days=n) for n in range(first_n, last_n + 1)]
        starts = [s for s in starts if s < end_ts]                # keep N >= 1
        return dict(end_ts=end_ts, window_start=window_start, first_n=first_n, last_n=last_n, starts=starts)

    exp_starts = experiment_starts()
    return (exp_starts,)


@app.cell
def _(N_minus_slider, N_plus_slider, hw, mo):
    def make_start_range():
        # experiment-start sweep, indexed by n (= right-chart step index; 0 = heatwave start).
        # bounds depend on hw / N⁻ / N⁺, so this cell recomputes (and the slider resets) like day_slider did.
        if hw is None:
            return mo.ui.range_slider(start=0, stop=1, value=[0, 1], label="experiment starts (n):")
        Nm, Np = int(N_minus_slider.value), int(N_plus_slider.value)
        dur = int(hw["end_i"] - hw["start_i"])
        lo = -Nm
        hi = max(lo + 1, dur + Np - 1)                  # last start < END_TS so every run has N >= 1
        return mo.ui.range_slider(start=lo, stop=hi, step=1, value=[lo, 0],
                                  show_value=True, full_width=True, label="experiment starts (n):")

    start_range_slider = make_start_range()
    return (start_range_slider,)


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(
    abs_style,
    add_map_stats,
    dpi_slider,
    map_mod,
    mask,
    mask_mode_dropdown,
    zoom_centers,
    zoom_slider,
):
    def build_mask_map():
        # the mask weights (arches lat weights, sum to 1): white at 0, warm toward the max
        cmap, vmin, vmax, center = abs_style(mask)
        m = map_mod.visualize_map(
            mask,
            suptitle="mask",
            title="mask weights m(lon, lat)  ·  arches lat weights, sums to 1",
            interactive=False,
            cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            contour_2d=None if mask_mode_dropdown.value == "BBOX" else mask,
            contour_levels=8, contour_color="black", contour_linewidth=0.5,
            zoom=zoom_slider.value,   # 1 = full map (all grid points); >1 zooms into the box
            zoom_center_lon=zoom_centers[0], zoom_center_lat=zoom_centers[1],
            figsize=(14, 8), dpi=dpi_slider.value,
        )
        return add_map_stats(m, mask)[0]   # (fig, ax) -> fig for display

    mask_map = build_mask_map()
    return (mask_map,)


@app.cell
def _():
    import importlib
    import src.ui.map as map_mod
    import src.mask as mask_mod
    importlib.reload(map_mod)    # pick up map.py edits (no white grid; interactive zoom)
    importlib.reload(mask_mod)   # pick up the reverted (previous) mask weighting
    return map_mod, mask_mod


if __name__ == "__main__":
    app.run()
