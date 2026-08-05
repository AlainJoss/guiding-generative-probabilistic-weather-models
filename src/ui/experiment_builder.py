import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Experiment builder

    Author an experiment the way `guidance.py` does in `unguided_rollout` mode: pick the
    variable/level, draw the **region of interest** on the map (click = center, sliders =
    box size), set the start date and `M/N/T`, then **save the config**. A separate
    section downloads the model's 14-day input for that start date.

    *Climatology (to choose the start date) is a standalone script:*
    `python -m src.download_climatology --out local_data/era5/clim_2m_temperature.nc`
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
        RolloutConfig,
        SURFACE,
        SURFACE_PAIR,
        UNITS,
        ccrs,
        dump_json,
        ekp,
        ensure_rollout_dir,
        get_mask_2d,
        get_mask_center,
        get_now_timestamp,
        get_timestamp_from_sliders,
        get_xr_dataset,
        max_day,
        mcolors,
        mo,
        np,
        os,
        plt,
        regridding,
        timedelta,
        to_display_units,
        visualize_map,
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


    return abs_style, add_map_stats, white_zero_cmap


@app.cell
def _(ensure_rollout_dir, get_now_timestamp):
    save_id = get_now_timestamp()   # one rollout id for the session

    def experiment_dir():
        # created lazily (only when a button actually writes), shared by both writers
        return ensure_rollout_dir(save_id)

    return experiment_dir, save_id


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
    year_dropdown = mo.ui.dropdown([str(y) for y in range(2020, 2027)], value="2024", label="year: ")
    month_slider = mo.ui.slider(1, 12, value=7, step=1, label="month: ", show_value=True)
    hour_slider = mo.ui.slider(0, 18, value=12, step=6, label="hour: ", show_value=True)
    M_slider = mo.ui.slider(1, 10, value=3, step=1, label="M: ", show_value=True)
    N_slider = mo.ui.slider(1, 30, value=2, step=1, label="N: ", show_value=True)
    T_slider = mo.ui.slider(5, 50, value=25, step=1, label="T: ", show_value=True)
    return (
        M_slider,
        N_slider,
        T_slider,
        dpi_slider,
        hour_slider,
        level_slider,
        mask_mode_dropdown,
        month_slider,
        side_lat_slider,
        side_lon_slider,
        sigma_div_slider,
        var_dropdown,
        year_dropdown,
        zoom_slider,
    )


@app.cell
def _(max_day, mo, month_slider):
    day_slider = mo.ui.slider(1, max_day(month_slider.value), value=1, label="day: ", show_value=True)
    return (day_slider,)


@app.cell
def _(mo):
    save_config_button = mo.ui.run_button(label="save config.json")
    model_button = mo.ui.run_button(label="download 14-day model input (all vars)")
    return model_button, save_config_button


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
        return weights, corners

    mask, mask_corners = build_mask()
    zoom_centers = get_mask_center(*mask_corners)   # recenter the maps on the box
    return mask, mask_corners, zoom_centers


@app.cell
def _(
    day_slider,
    get_timestamp_from_sliders,
    hour_slider,
    month_slider,
    year_dropdown,
):
    start_ts = get_timestamp_from_sliders(
        int(year_dropdown.value), month_slider.value, day_slider.value, hour_slider.value)
    return (start_ts,)


@app.cell
def _(get_xr_dataset, to_display_units):
    bg_disp = to_display_units(
        get_xr_dataset(2020)["2m_temperature"].mean("time").to_numpy(), "2m_temperature")[0]
    return (bg_disp,)


@app.cell
def _(
    abs_style,
    add_map_stats,
    bg_disp,
    dpi_slider,
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
            suptitle="ERA5 2m_temperature (2020 mean)",
            title="click to set the box center",
            interactive=True,
            cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            puck_center=((mask_corners[0] + mask_corners[1]) / 2,
                         (mask_corners[2] + mask_corners[3]) / 2),
            side_lon=mask_corners[1] - mask_corners[0],
            side_lat=mask_corners[3] - mask_corners[2],
            contour_2d=None if mask_mode_dropdown.value == "BBOX" else mask,
            contour_levels=8, contour_color="black", contour_linewidth=0.5,
            zoom=zoom_slider.value, zoom_center_lon=zoom_centers[0], zoom_center_lat=zoom_centers[1],
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
    add_map_stats,
    dpi_slider,
    mask,
    mask_mode_dropdown,
    mcolors,
    np,
    visualize_map,
    white_zero_cmap,
    zoom_centers,
    zoom_slider,
):
    mask_min, mask_max = float(np.min(mask)), float(np.max(mask))
    # warm half of the map so 0 is white and the max saturates to red (guidance style)
    mask_cmap = (mcolors.LinearSegmentedColormap.from_list(
        "mask_warm", white_zero_cmap(np.linspace(0.5, 1.0, 256)))
        if mask_min == 0.0 else white_zero_cmap)
    mask_map = visualize_map(
        mask,
        suptitle="mask",
        title="mask weights (region of interest)",
        interactive=False,
        cmap=mask_cmap,
        vmin=mask_min if mask_min < mask_max else -0.001,
        vmax=mask_max if mask_min < mask_max else 0.001,
        center=(None if mask_min == 0.0 else 0.5 * (mask_min + mask_max)) if mask_min < mask_max else 0.0,
        contour_2d=None if mask_mode_dropdown.value == "BBOX" else mask,
        contour_levels=8, contour_color="black", contour_linewidth=0.5,
        zoom=zoom_slider.value, zoom_center_lon=zoom_centers[0], zoom_center_lat=zoom_centers[1],
        figsize=(14, 8), dpi=dpi_slider.value,
    )
    mask_map = add_map_stats(mask_map, mask)[0]  # (fig, ax) -> fig for display
    return (mask_map,)


@app.cell
def _(
    abs_style,
    ccrs,
    crs_mode_dropdown,
    ekp,
    get_center,
    make_crs,
    mask_corners,
    temp_field,
):
    # 2m_temperature on the selected projection, centered on the box (earthkit-plots),
    # same white-at-0 coloring convention as the Mask-section maps
    def domain_globe():
        center_lon, center_lat = get_center()
        m = ekp.Map(crs=make_crs(crs_mode_dropdown.value, center_lon, center_lat))
        cmap, vmin, vmax, center = abs_style(temp_field.values)
        if center == 0.0:                      # straddles 0 -> symmetric so white sits at 0
            absmax = max(abs(vmin), abs(vmax))
            vmin, vmax = -absmax, absmax
        m.pcolormesh(temp_field, cmap=cmap, vmin=vmin, vmax=vmax)
        m.coastlines()
        m.borders()
        m.legend()
        ax = m.fig.axes[0]
        lon_l, lon_r, lat_b, lat_t = mask_corners
        ax.plot([lon_l, lon_r, lon_r, lon_l, lon_l], [lat_b, lat_b, lat_t, lat_t, lat_b],
                color="red", linewidth=1.5, transform=ccrs.PlateCarree(), zorder=10)
        return m.fig

    domain_map = domain_globe()

    return (domain_map,)


@app.cell
def _(
    M_slider,
    N_slider,
    T_slider,
    level,
    mask_corners,
    mo,
    partition,
    save_id,
    start_ts,
    var,
):
    config_preview = mo.md(
        f"**save_id** `{save_id}`  ·  **M/N/T** {M_slider.value}/{N_slider.value}/{T_slider.value}  ·  "
        f"**VAR** `{var}` ({partition}, L{level})  ·  **START_TS** `{start_ts}`  ·  "
        f"**MASK_CORNERS** `{tuple(round(c, 2) for c in mask_corners)}`"
    )
    return (config_preview,)


@app.cell
def _(
    M_slider,
    N_slider,
    RolloutConfig,
    T_slider,
    dump_json,
    experiment_dir,
    level,
    mask_corners,
    mo,
    partition,
    save_config_button,
    start_ts,
    var,
):
    def write_config():
        cfg = RolloutConfig(
            M=M_slider.value, N=N_slider.value, T=T_slider.value,
            START_TS=start_ts, PARTITION=partition, LEVEL=level, VAR=var,
            MASK_CORNERS=list(mask_corners),
        )
        path = experiment_dir() / "config.json"
        dump_json(cfg.to_dict(), path)
        return mo.md(f"✅ wrote `{path}`\n\n```json\n{cfg.to_dict()}\n```")

    save_config_result = (write_config() if save_config_button.value
                          else mo.md("press **save config.json** to write the config"))
    return (save_config_result,)


@app.cell
def _(
    experiment_dir,
    mo,
    model_button,
    open_arco,
    os,
    sel_12z,
    start_ts,
    timedelta,
    to_arches_netcdf,
    xr,
):
    def download_model_input():
        path = str(experiment_dir() / "era5_input_14d.nc")
        if os.path.exists(path):
            ds = xr.open_dataset(path)
        else:
            window = sel_12z(open_arco()).sel(
                time=slice(start_ts - timedelta(days=1), start_ts + timedelta(days=14)))
            ds = to_arches_netcdf(window, path, batch=2)
        return mo.md(
            f"✅ `{path}`\n\nsteps={ds.sizes['time']} (n=−1, n=0, +14 GT) · "
            f"vars={len(ds.data_vars)} · levels={ds.sizes.get('level')} · "
            f"grid={ds.sizes['latitude']}×{ds.sizes['longitude']}"
        )

    model_input_msg = (download_model_input() if model_button.value
                       else mo.md("press **download 14-day model input** to fetch + regrid"))

    return (model_input_msg,)


@app.cell
def _(
    M_slider,
    N_slider,
    T_slider,
    config_preview,
    day_slider,
    hour_slider,
    level_slider,
    mo,
    month_slider,
    save_config_button,
    save_config_result,
    var_dropdown,
    year_dropdown,
):
    mo.vstack([
        mo.md("## Experiment"),
        mo.hstack([var_dropdown, level_slider], justify="start"),
        mo.hstack([M_slider, N_slider, T_slider], justify="start"),
        mo.hstack([year_dropdown, month_slider, day_slider, hour_slider], justify="start"),
        config_preview,
        mo.hstack([save_config_button], justify="start"),
        save_config_result,
    ], align="start")
    return


@app.cell
def _(
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
        mo.hstack([weather_map, mask_map], justify="start", align="start"),
    ], align="start")
    return


@app.cell
def _(crs_mode_dropdown, domain_map, mo):
    mo.vstack([
        mo.md("## Domain (2m_temperature, 2020 mean — centered on the box)"),
        mo.hstack([crs_mode_dropdown], justify="start"),
        domain_map,
    ], align="start")

    return


@app.cell
def _(mo, model_button, model_input_msg):
    mo.vstack([mo.md("## Model input"), model_button, model_input_msg], align="start")
    return


@app.cell
def _(mo):
    crs_mode_dropdown = mo.ui.dropdown(
        ["NearsidePerspective", "Orthographic", "PlateCarree", "Robinson",
         "Mollweide", "LambertAzimuthalEqualArea"],
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

    return clim_ds, clim_path, get_masked_mean, get_slices, mdates, pd


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
    pd,
    plt,
    rolling_window_slider,
):
    def region_climatology_plot():
        if clim_ds is None:
            return mo.md(f"_climatology file not found — run `python -m src.download_climatology --out {clim_path}`_")
        slices = get_slices(clim_ds, "surface", "2m_temperature", 0)   # (time,121,240), lon 0-360
        slices = np.roll(slices, slices.shape[-1] // 2, axis=-1)        # align to the mask grid (-180..180)
        series = get_masked_mean(slices, mask) - 273.15
        times = pd.to_datetime(clim_ds.time.values)
        df = pd.DataFrame({"v": series}, index=times)
        df["year"] = df.index.year
        df["doy"] = np.clip(df.index.dayofyear.to_numpy(), 1, 365)
        window = int(rolling_window_slider.value)
        sel = int(highlight_year_slider.value)
        years = sorted(df["year"].unique())
        ref = pd.Timestamp("2001-01-01")
        smooth = lambda v: pd.Series(v).rolling(window, center=True, min_periods=1).mean().values
        xdate = lambda g: ref + (g["doy"].values - 1) * pd.Timedelta("1D")

        fig, ax = plt.subplots(figsize=(14, 8), dpi=500)
        for yr, g in df.groupby("year"):                     # non-neighbour years: light gray
            if yr in (sel - 1, sel, sel + 1):
                continue
            g = g.sort_values("doy")
            ax.plot(xdate(g), smooth(g["v"].values), color="#cfcfcf", linewidth=0.8, alpha=1, zorder=1)
        # previous (yellow, shaded), next (dark brown, shaded), selected (red, on top).
        # sel-1 / sel+1 only exist in `years` when present -> 2020 has no prev, 2026 no next.
        for yr, color, lw, a, z in [(sel - 1, "#f2a900", 1.7, 0.4, 3),
                                    (sel + 1, "#7a1f1f", 1.7, 0.25, 3),
                                    (sel,     "#e63946", 2.1, 1.0, 6)]:
            if yr in years:
                g = df[df["year"] == yr].sort_values("doy")
                ax.plot(xdate(g), smooth(g["v"].values), color=color, linewidth=lw, alpha=a,
                        label=str(yr), zorder=z)
        mean_by_doy = df.groupby("doy")["v"].mean()
        ax.plot(ref + (mean_by_doy.index.values - 1) * pd.Timedelta("1D"), smooth(mean_by_doy.values),
                color="#444444", linewidth=1.6, linestyle="--", label=f"{years[0]}-{years[-1]} average", zorder=5)
        ax.plot([], [], color="#cfcfcf", linewidth=1.2, label="other years")

        # month guide lines at each month start, drawn as a uniform x-grid on minor ticks.
        # (individual axvlines render unevenly from sub-pixel rounding.)  CHANGE COLOR HERE:
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1))    # gridlines at month starts
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=16))   # mid-month -> centered labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        # ax.grid(True, axis="x", which="minor", color="#c8c8c8", linewidth=0.7, zorder=0)   # month-start lines (CHANGE COLOR HERE)
        ax.grid(False, axis="x", which="major")   # no mid-month lines
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xlim(ref, ref + pd.Timedelta("365D"))
        ax.set_ylabel("mask-avg 2m_temperature [°C]")
        ax.set_title(f"Region surface air temperature  ·  daily average"
                     + (f"  ·  {window}-day rolling" if window > 1 else ""), loc="left", fontsize=13)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=0.16, top=0.93, left=0.09, right=0.97)
        # mark the highlighted year's rank-th peak (hline + temperature) and the heatwave start
        if hw is not None:
            peak_x = ref + (hw["peak_doy"] - 1) * pd.Timedelta("1D")
            start_x = ref + (hw["start_doy"] - 1) * pd.Timedelta("1D")
            # ax.axhline(hw["peak_val"], color="#e63946", linewidth=0.9, linestyle=":", zorder=3)
            # ax.axhline(hw["start_val"], color="#e63946", linewidth=0.9, linestyle=":", zorder=3)  # start temp
            ax.axvline(peak_x, color="#e63946", linewidth=0.8, linestyle=":", zorder=3)          # peak day
            ax.axvline(start_x, color="#e63946", linewidth=0.8, linestyle=":", zorder=3)         # start day
            ax.plot([peak_x], [hw["peak_val"]], "o", color="#e63946", markersize=6, zorder=7)
            ax.annotate(f"{hw['peak_val']:.1f}°C", xy=(peak_x, hw["peak_val"]), xytext=(0, 7),
                        textcoords="offset points", ha="center", color="#e63946", fontsize=10,
                        fontweight="bold", zorder=8)
            ax.plot([start_x], [hw["start_val"]], "o", color="#e63946", markersize=6, zorder=7)
            ax.annotate(f"{hw['start_val']:.1f}°C", xy=(start_x, hw["start_val"]), xytext=(0, -14),
                        textcoords="offset points", ha="center", color="#e63946", fontsize=10, fontweight="bold", zorder=8)
        _ylo, _yhi = ax.get_ylim()
        ax.set_ylim(_ylo, _yhi + 0.05 * (_yhi - _ylo))   # headroom so the peak label clears the title
        return fig

    region_clim_plot = region_climatology_plot()

    return (region_clim_plot,)


@app.cell
def _(
    H_slider,
    N_slider,
    highlight_year_slider,
    mo,
    peak_rank_slider,
    region_clim_plot,
    region_trajectory_fig,
    rolling_window_slider,
):
    mo.vstack([
        mo.md("## Region climatology + heatwave trajectory"),
        mo.hstack([highlight_year_slider, peak_rank_slider, rolling_window_slider], justify="start"),
        mo.hstack([
            region_clim_plot,
            mo.vstack([
                mo.hstack([N_slider, H_slider], justify="start"),
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
def _(clim_ds, mo, np, pd):
    def make_year_slider():
        if clim_ds is None:
            return mo.ui.slider(2020, 2026, value=2026, step=1, label="highlight year: ", show_value=True)
        yrs = np.unique(pd.to_datetime(clim_ds.time.values).year)
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
    pd,
    peak_rank_slider,
    rolling_window_slider,
):
    from scipy.signal import find_peaks

    def heatwave_analysis():
        """For the highlighted year: the rolling region-avg series, its rank-th peak,
        and the heatwave start (walk back from the peak to the local min)."""
        if clim_ds is None:
            return None
        slices = np.roll(get_slices(clim_ds, "surface", "2m_temperature", 0), 
                         get_slices(clim_ds, "surface", "2m_temperature", 0).shape[-1] // 2, axis=-1)
        series = get_masked_mean(slices, mask) - 273.15
        t = pd.to_datetime(clim_ds.time.values)
        yr = int(highlight_year_slider.value)
        m = t.year == yr
        doy = np.clip(t.dayofyear.to_numpy()[m], 1, 365)
        order_doy = np.argsort(doy)
        doy = doy[order_doy]
        v_raw = series[m][order_doy]
        v = pd.Series(v_raw).rolling(int(rolling_window_slider.value), center=True, min_periods=1).mean().values
        peaks, _ = find_peaks(v)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(v))])
        ranked = peaks[np.argsort(v[peaks])[::-1]]                 # local maxima, tallest first
        p = int(ranked[min(int(peak_rank_slider.value), len(ranked)) - 1])
        i = p                                                       # heatwave start: back to local min
        while i > 0 and v[i - 1] < v[i]:
            i -= 1
        return dict(year=yr, doy=doy, v=v, v_raw=v_raw, peak_i=p, start_i=i,
                    peak_doy=int(doy[p]), peak_val=float(v[p]),
                    start_doy=int(doy[i]), start_val=float(v[i]))

    hw = heatwave_analysis()

    return (hw,)


@app.cell
def _(H_slider, N_slider, hw, mo, np, plt, region_clim_plot):
    def region_trajectory_plot():
        if hw is None:
            return mo.md("_no heatwave selected_")
        v, s, N, H = hw["v_raw"], hw["start_i"], int(N_slider.value), int(H_slider.value)
        lo, hi = max(0, s - H), min(len(v), s + N + 1)   # H days before start .. N after
        seg = v[lo:hi]
        steps = np.arange(lo - s, hi - s)                # x relative to start (0 = heatwave start)
        size = tuple(region_clim_plot.get_size_inches()) if hasattr(region_clim_plot, "get_size_inches") else (11, 8)
        fig, ax = plt.subplots(figsize=size, dpi=130)   # equal size to the left chart
        ax.plot(steps, seg, "-o", color="#e63946", linewidth=1.6, markersize=4, zorder=5)  # same red as left
        ax.axhline(v[s], color="#e63946", linewidth=0.9, linestyle=":", zorder=3)  # start temp at x=0 (left-chart style)
        ax.axvline(0, color="#e63946", linewidth=0.9, linestyle=":", zorder=3)  # heatwave start (n=0)
        ax.set_xlabel("n  (days from heatwave start)")
        ax.set_ylabel("mask-avg 2m_temperature [°C]")
        ax.set_title(f"Region-average trajectory from heatwave start  ·  {hw['year']}  ·  N={N}", loc="left", fontsize=13)
        ax.set_xticks(steps)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.9)
        ax.set_axisbelow(True)
        fig.subplots_adjust(bottom=0.16, top=0.93, left=0.11, right=0.97)
        return fig

    region_trajectory_fig = region_trajectory_plot()

    return (region_trajectory_fig,)


@app.cell
def _(mo):
    H_slider = mo.ui.slider(0, 10, value=2, step=1, label="H (days prior): ", show_value=True)

    return (H_slider,)


if __name__ == "__main__":
    app.run()
