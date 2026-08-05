import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ERA5 point download (WeatherBench2)

    Click the map to move the **center puck**; the red rectangle is the bounding box
    (width/height sliders). The center's coordinates drive a **WeatherBench2** ERA5
    request: the public, cloud-optimized Zarr on GCS (no credentials) — we pull only
    the **nearest grid point's** `2m_temperature` time series, all days at **12:00Z**.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt

    from src.utils import get_xr_dataset
    from src.ui.map import visualize_map, to_display_units

    # WeatherBench2 ERA5, 1.5 deg / 240x121 equiangular, 6-hourly (matches the model grid).
    # Public bucket -> anonymous GCS access. Coverage: 1959-01-01 .. 2023-01-10.
    WB2_URI = (
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    )
    WB2_STORAGE = {"token": "anon"}
    return (
        WB2_STORAGE,
        WB2_URI,
        get_xr_dataset,
        mo,
        np,
        pd,
        plt,
        to_display_units,
        visualize_map,
        xr,
    )


@app.cell
def _(get_xr_dataset, to_display_units):
    # map background: local ERA5 2m_temperature annual mean (same 240x121 grid, instant).
    _field = get_xr_dataset(2020)["2m_temperature"].mean("time").to_numpy()
    bg_disp, bg_unit = to_display_units(_field, "2m_temperature")
    return bg_disp, bg_unit


@app.cell
def _(mo):
    # center puck coordinate (lon, lat), display convention lon in [-180, 180]
    get_center, set_center = mo.state((-4.0, 40.0))
    return get_center, set_center


@app.cell
def _(mo):
    side_lon_slider = mo.ui.slider(1, 60, value=12, step=1, label="bbox width (lon°): ", show_value=True)
    side_lat_slider = mo.ui.slider(1, 60, value=10, step=1, label="bbox height (lat°): ", show_value=True)
    start_year_slider = mo.ui.slider(1959, 2023, value=2000, step=1, label="start year: ", show_value=True)
    download_button = mo.ui.run_button(label="⬇ Download ERA5 point series (WeatherBench2)")
    return download_button, side_lat_slider, side_lon_slider, start_year_slider


@app.cell
def _(
    download_button,
    mo,
    side_lat_slider,
    side_lon_slider,
    start_year_slider,
):
    mo.hstack(
        [side_lon_slider, side_lat_slider, start_year_slider, download_button],
        justify="start", align="center",
    )
    return


@app.cell
def _(
    bg_disp,
    bg_unit,
    get_center,
    set_center,
    side_lat_slider,
    side_lon_slider,
    visualize_map,
):
    era5_map = visualize_map(
        bg_disp,
        interactive=True,
        cmap="coolwarm",
        puck_center=get_center(),
        side_lon=side_lon_slider.value,
        side_lat=side_lat_slider.value,
        title=f"ERA5 2m_temperature (2020 mean, {bg_unit}) — click to move the center",
    )
    # sync the puck position back into the center state (same pattern as guidance.py)
    era5_map.widget.observe(
        lambda _ch: set_center((era5_map.widget.x[0], era5_map.widget.y[0])),
        names=["x", "y"],
    )
    era5_map
    return


@app.cell
def _(get_center, mo):
    lon_c, lat_c = get_center()
    mo.md(f"**Center:** lon `{lon_c:.2f}`, lat `{lat_c:.2f}`  →  nearest ERA5 grid point")
    return lat_c, lon_c


@app.cell
def _(
    WB2_STORAGE,
    WB2_URI,
    download_button,
    lat_c,
    lon_c,
    mo,
    start_year_slider,
    xr,
):
    # gated on the button: pull only the nearest grid point's 2m_temperature at 12:00Z.
    # Point extraction reads whole time-chunks, so multi-decade pulls move ~GBs -> be patient.
    if download_button.value:
        _ds = xr.open_zarr(WB2_URI, storage_options=WB2_STORAGE, chunks={})
        _pt = _ds["2m_temperature"].sel(
            latitude=lat_c, longitude=lon_c % 360.0, method="nearest",
        )
        _pt = _pt.sel(time=_pt.time.dt.hour == 12)
        _pt = _pt.sel(time=slice(f"{start_year_slider.value}-01-01", None))  # to dataset end
        era5_point = _pt.compute()
        _msg = mo.md(
            f"Downloaded **{era5_point.size}** steps @ 12:00Z | "
            f"grid point lat `{float(era5_point.latitude):.2f}`, lon `{float(era5_point.longitude):.2f}`"
        )
    else:
        era5_point = None
        _msg = mo.md("_press the download button to fetch the point series_")
    _msg
    return (era5_point,)


@app.cell
def _(era5_point, np, pd, plt, to_display_units):
    if era5_point is None:
        _fig = None
    else:
        _t = pd.to_datetime(era5_point.time.values)
        _y, _unit = to_display_units(era5_point.values, "2m_temperature")
        _roll = pd.Series(_y, index=_t).rolling(365, center=True, min_periods=30).mean()

        _fig, _ax = plt.subplots(figsize=(22, 6), dpi=140)
        _ax.plot(_t, _y, color="#C9A7E8", linewidth=0.5, alpha=0.8, label="daily 12:00Z")
        _ax.plot(_roll.index, _roll.values, color="#7B2CBF", linewidth=2.0, label="365-day rolling mean")
        _ax.set_ylabel(f"2m_temperature [{_unit}]")
        _ax.set_title(
            f"WeatherBench2 ERA5 2m_temperature @ nearest point   |   "
            f"n={_y.size}   mean={np.nanmean(_y):.1f}{_unit}"
        )
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)
        _ax.grid(True, axis="y", color="#D7D7D7", linewidth=0.75, alpha=0.55)
        _ax.set_axisbelow(True)
        _ax.legend(loc="upper left", frameon=False, fontsize=9)
        _fig.subplots_adjust(top=0.9, left=0.06, right=0.98, bottom=0.12)
    _fig
    return


if __name__ == "__main__":
    app.run()
