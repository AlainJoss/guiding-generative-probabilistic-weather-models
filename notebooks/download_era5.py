import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import cdsapi

    return


@app.cell
def _():
    # dataset = "reanalysis-era5-single-levels"
    # request = {
    #     "product_type": ["reanalysis"],
    #     "variable": [
    #         "10m_u_component_of_wind",
    #         "10m_v_component_of_wind",
    #         "2m_temperature",
    #         "mean_sea_level_pressure",
    #     ],
    #     "year": ["2026"],
    #     "month": ["05"],
    #     "day": [
    #         "01", "02", "03",
    #         "04", "05", "06",
    #         "07", "08", "09",
    #         "10", "11", "12",
    #         "13", "14", "15",
    #         "16", "17", "18",
    #         "19", "20", "21",
    #         "22", "23", "24",
    #         "25", "26", "27",
    #         "28", "29",
    #     ],
    #     "time": [
    #         "00:00",
    #         "06:00",
    #         "12:00",
    #         "18:00",
    #     ],
    #     "grid": [1.5, 1.5],
    #     "data_format": "netcdf",
    #     "download_format": "zip",
    # }

    # client = cdsapi.Client()
    # client.retrieve(dataset, request).download()
    return


@app.cell
def _():
    # dataset = "reanalysis-era5-pressure-levels"
    # request = {
    #     "product_type": ["reanalysis"],
    #     "variable": [
    #         "geopotential",
    #         "specific_humidity",
    #         "temperature",
    #         "u_component_of_wind",
    #         "v_component_of_wind",
    #         "vertical_velocity"
    #     ],
    #     "year": ["2026"],
    #     "month": ["05"],
    #     "day": [
    #         "01", "02", "03",
    #         "04", "05", "06",
    #         "07", "08", "09",
    #         "10", "11", "12",
    #         "13", "14", "15",
    #         "16", "17", "18",
    #         "19", "20", "21",
    #         "22", "23", "24",
    #         "25", "26", "27",
    #         "28", "29"
    #     ],
    #     "time": [
    #         "00:00",
    #         "06:00",
    #         # "12:00",
    #         # "18:00",
    #     ],
    #     "grid": [1.5, 1.5],
    #     "pressure_level": [
    #         "50", "100", "150",
    #         "200", "250", "300",
    #         "400", "500", "600",
    #         "700", "850", "925",
    #         "1000"
    #     ], 
    #     "data_format": "netcdf",
    #     "download_format": "zip"
    # }

    # client = cdsapi.Client()
    # client.retrieve(dataset, request).download()
    return


@app.cell
def _():
    # dataset = "reanalysis-era5-pressure-levels"
    # request = {
    #     "product_type": ["reanalysis"],
    #     "variable": [
    #         "geopotential",
    #         "specific_humidity",
    #         "temperature",
    #         "u_component_of_wind",
    #         "v_component_of_wind",
    #         "vertical_velocity"
    #     ],
    #     "year": ["2026"],
    #     "month": ["05"],
    #     "day": [
    #         "01", "02", "03",
    #         "04", "05", "06",
    #         "07", "08", "09",
    #         "10", "11", "12",
    #         "13", "14", "15",
    #         "16", "17", "18",
    #         "19", "20", "21",
    #         "22", "23", "24",
    #         "25", "26", "27",
    #         "28", "29"
    #     ],
    #     "time": [
    #         # "00:00",
    #         # "06:00",
    #         "12:00",
    #         "18:00",
    #     ],
    #     "grid": [1.5, 1.5],
    #     "pressure_level": [
    #         "50", "100", "150",
    #         "200", "250", "300",
    #         "400", "500", "600",
    #         "700", "850", "925",
    #         "1000"
    #     ],
    #     "data_format": "netcdf",
    #     "download_format": "zip"
    # }

    # client = cdsapi.Client()
    # client.retrieve(dataset, request).download()
    return


@app.cell
def _():
    # changed names for easier read in
    return


@app.cell
def _():
    import xarray as xr

    return (xr,)


@app.cell
def _(xr):
    surface = xr.open_dataset("2026-surface.nc", engine="netcdf4")
    surface
    return (surface,)


@app.cell
def _(xr):
    level_00_06 = xr.open_dataset("2026-level-00-06.nc", engine="netcdf4")
    level_00_06
    return (level_00_06,)


@app.cell
def _(xr):
    level_12_18 = xr.open_dataset("2026-level-12-18.nc", engine="netcdf4")
    level_12_18
    return (level_12_18,)


@app.cell
def _(level_00_06, level_12_18, xr):
    level = xr.concat([level_00_06, level_12_18], dim="valid_time")
    level = level.sortby("valid_time")
    level
    return (level,)


@app.cell
def _():
    # rename valid_time -> time, pressure_level -> level
    return


@app.cell
def _():
    # join datasets on time 
    return


@app.cell
def _(level, surface):
    full = surface.merge(level, join="exact", compat="no_conflicts")

    full = full.rename({
        "valid_time": "time",
        "pressure_level": "level",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "t2m": "2m_temperature",
        "msl": "mean_sea_level_pressure",
        "z": "geopotential",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "t": "temperature",
        "q": "specific_humidity",
        "w": "vertical_velocity",
    })

    # drops non-dimension coords
    full = full.drop_vars(["expver", "number"])
    full.coords
    return (full,)


@app.cell
def _():
    from src.paths import SWITCHDRIVE_DATA
    from pathlib import Path

    return SWITCHDRIVE_DATA, Path


@app.cell(disabled=True)
def _(DATA, Path, full):
    full.to_netcdf(Path(DATA, "arches_era5_26.nc"))
    return


@app.cell
def _():
    from src.utils.read_write import get_xr_dataset
    ds_26 = get_xr_dataset(2026)
    ds_26 = ds_26.drop_attrs()

    ds_26 = ds_26.roll(longitude=ds_26.sizes["longitude"] // 2, roll_coords=False)

    ds_26.transpose(..., "level", "latitude", "longitude")
    return (ds_26,)


@app.cell(disabled=True)
def _(DATA, Path, ds_26):
    ds_26.to_netcdf(Path(DATA, "arches_era5_26.nc"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
