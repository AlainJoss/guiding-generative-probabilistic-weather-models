"""Download an ERA5 2m_temperature climatology and regrid it to the ArchesWeather grid.

Source: ARCO-ERA5 (public GCS, anonymous), hourly 0.25deg, to ~present. We keep only the
12:00Z instantaneous slices (WeatherBench2-style synoptic subsampling), regrid 0.25deg ->
1.5deg (240x121) with WeatherBench2's ConservativeRegridder, and save a netCDF. The result
feeds the experiment-builder notebook's climatology view (mask-average -> annual cycle).

NOTE: ARCO chunks are time=1 full-globe, so a single 2m_temperature field is ~4 MB and the
pull is GB-scale (2020->present 12Z-daily ~9 GB read). Regridding is done in memory-bounded
time batches. The saved 1.5deg netCDF is small (~250 MB for 6 years).

Usage:
    python -m src.download_climatology --out local_data/era5/clim_2m_temperature.nc
    python -m src.download_climatology --out clim.nc --start-year 2020 --end-year 2026 --batch 64
"""
import argparse
import time

import numpy as np
import xarray as xr
from weatherbench2 import regridding

ARCO_URI = ("gs://gcp-public-data-arco-era5/ar/"
            "full_37-1h-0p25deg-chunk-1.zarr-v3")
ARCO_STORAGE = {"token": "anon"}
# ArchesWeather target grid. The regridder needs ascending latitude; we flip to the
# model's descending convention (90 -> -90) afterwards.
ARCHES_LON = np.arange(0, 360, 1.5)              # 240, ascending 0..358.5
ARCHES_LAT_ASC = np.arange(-90, 90.001, 1.5)     # 121, ascending -90..90


def regrid_to_arches(ds: xr.Dataset) -> xr.Dataset:
    """Conservatively regrid an ARCO subset onto the 1.5deg ArchesWeather grid."""
    ds = ds.sortby("latitude").load()            # WB2 needs ascending lat + in-memory numpy
    src = regridding.Grid.from_degrees(lon=ds.longitude.values, lat=ds.latitude.values)
    tgt = regridding.Grid.from_degrees(lon=ARCHES_LON, lat=ARCHES_LAT_ASC)
    out = regridding.ConservativeRegridder(src, tgt).regrid_dataset(ds)
    return out.sortby("latitude", ascending=False).astype("float32")  # -> descending lat


def last_valid_time(da: xr.DataArray):
    """Last time index with real (finite) data.

    ARCO's `time` axis is pre-allocated far into the future (to 2050) and filled with
    NaN beyond the latest available analysis (~a couple months behind present). Data is
    finite up to a cutoff then NaN, so we binary-search a single grid point's finiteness
    (each probe reads one ~4 MB chunk) to find the last real timestep — avoids pulling
    tens of GB of unfilled future dates.
    """
    pt = da.isel(latitude=da.sizes["latitude"] // 2, longitude=da.sizes["longitude"] // 2)
    finite = lambda i: bool(np.isfinite(float(pt.isel(time=i).values)))
    lo, hi = 0, da.sizes["time"] - 1
    if not finite(lo):
        raise ValueError("no finite data at the requested start")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if finite(mid):
            lo = mid
        else:
            hi = mid - 1
    return da.time.values[lo]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output netCDF path")
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=None,
                    help="inclusive; default = latest available in ARCO")
    ap.add_argument("--batch", type=int, default=64,
                    help="time steps regridded per in-memory batch")
    args = ap.parse_args()

    ds = xr.open_zarr(ARCO_URI, storage_options=ARCO_STORAGE, chunks={})
    da = ds[["2m_temperature"]]
    da = da.sel(time=da.time.dt.hour == 12)                       # 12:00Z only (instantaneous)
    end = None if args.end_year is None else f"{args.end_year}-12-31"
    da = da.sel(time=slice(f"{args.start_year}-01-01", end))
    # cap at the last real timestep (ARCO's axis runs to 2050 with NaN in the unfilled future)
    _lv = last_valid_time(da["2m_temperature"])
    da = da.sel(time=slice(None, _lv))
    n = da.sizes["time"]
    print(f"{n} 12Z steps: {str(da.time.values.min())[:10]} .. {str(da.time.values.max())[:10]} "
          f"(capped at last real data)")
    print(f"downloading + regridding in batches of {args.batch} (this pulls several GB) ...")

    outs = []
    t0 = time.time()
    for i in range(0, n, args.batch):
        outs.append(regrid_to_arches(da.isel(time=slice(i, i + args.batch))))
        done = min(i + args.batch, n)
        print(f"  {done}/{n} regridded ({time.time() - t0:.0f}s)")
    out = xr.concat(outs, dim="time")

    out.to_netcdf(args.out)
    print(f"saved {args.out}  |  dims={dict(out.sizes)}  |  "
          f"grid={out.sizes['latitude']}x{out.sizes['longitude']}")


if __name__ == "__main__":
    main()
