"""Download the ArchesWeather model input for an experiment and regrid it to the model grid.

Source: ARCO-ERA5 (public GCS, anonymous), hourly 0.25deg. We keep only the 12:00Z
instantaneous slices (WeatherBench2-style synoptic subsampling), select the ArchesWeather
82-field stack (4 surface + 6 upper-air x 13 levels), regrid 0.25deg -> 1.5deg (240x121)
with WeatherBench2's ConservativeRegridder, and save an `era5_input.nc` in the experiment
(outer rollout) dir. The download is GLOBAL (the model forecasts the whole globe); only the
time axis is sliced.

The timesteps come from the experiment config (written by the experiment-builder notebook):
the dates `STARTS` and the rollout length `N`. Each date d needs `[d-1 day .. d+N days]`
(prev_state through the last rollout step), and we write ONE `era5_input.nc` PER date
subfolder (`<exp>/<date>/era5_input.nc`) holding exactly those few days -- never the contiguous
min..max span (dates can sit years apart). At run time `get_x_cond` (via `find_era5_input`)
reads the file from the date's own subfolder.

NOTE: ARCO chunks are time=1 full-globe, so the pull is GB-scale (minutes). The saved 1.5deg
netCDF is small.

Usage:
    python -m src.download --rollout_id 2026-08-05_12:00:00
    python -m src.download --rollout_id <id> --batch 4 --force
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from weatherbench2 import regridding

from src.utils import get_rollout_dir, get_dict_from_json, get_config

STARTS_FMT = "%Y-%m-%d_%H:%M:%S"   # STARTS names = the per-date subdir names

ARCO_URI = ("gs://gcp-public-data-arco-era5/ar/"
            "full_37-1h-0p25deg-chunk-1.zarr-v3")
ARCO_STORAGE = {"token": "anon"}
# ArchesWeather target grid. The regridder needs ascending latitude; we flip to the
# model's descending convention (90 -> -90) afterwards.
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


def open_arco():
    # chunks=None (the ARCO-docs pattern) reads zarr chunks directly, NO dask: opening is ~1s.
    # chunks={} builds a dask graph over the full 1940-2050 hourly axis, which (with the old
    # per-timestep .dt.hour==12 scan) made opening hang for minutes.
    return xr.open_zarr(ARCO_URI, storage_options=ARCO_STORAGE, chunks=None, consolidated=True)


def regrid_to_arches(ds):
    # WB2 ConservativeRegridder needs ascending lat + in-memory numpy
    ds = ds.sortby("latitude").load()
    source = regridding.Grid.from_degrees(lon=ds.longitude.values, lat=ds.latitude.values)
    target = regridding.Grid.from_degrees(lon=ARCHES_LON, lat=ARCHES_LAT_ASC)
    out = regridding.ConservativeRegridder(source, target).regrid_dataset(ds)
    return out.sortby("latitude", ascending=False).astype("float32")  # -> descending


def to_arches_netcdf(ds, path, batch=2):
    # ARCO subset -> 1.5deg arches_era5.nc schema netCDF
    ds = ds[SURFACE + LEVELVARS].sel(level=MODEL_LEVELS)
    n = ds.sizes["time"]
    outs, t0 = [], time.time()
    for i in range(0, n, batch):                       # time-batched to bound memory + show progress
        outs.append(regrid_to_arches(ds.isel(time=slice(i, i + batch))))
        print(f"  regridded {min(i + batch, n)}/{n} timesteps ({time.time() - t0:.0f}s)", flush=True)
    ds = xr.concat(outs, dim="time")
    ds = ds.assign_coords(level=ds.level.astype("int64"))[SURFACE + LEVELVARS]
    # arches order: surface (time,lat,lon); level (time,level,lat,lon)
    ds = ds.transpose("time", "level", "latitude", "longitude", missing_dims="ignore")
    for name in ds.data_vars:
        ds[name].attrs["units"] = UNITS[name]
    ds.to_netcdf(path)
    return ds


def download_model_input(times, path, batch=2, force=False):
    """Pull the requested 12:00Z timesteps, regrid to arches, save to `path`.

    `times` is a list of numpy datetime64[ns] at 12:00 (see date_times). Only these exact
    steps are fetched via a direct index select -- NO full-axis `.dt.hour==12` scan (which,
    with dask-on-open, made this hang for minutes). Idempotent: an existing file is reopened
    (not re-downloaded) unless `force`.
    """
    path = str(path)
    if os.path.exists(path) and not force:
        ds = xr.open_dataset(path)
        print(f"exists, skipping: {path}  |  steps={ds.sizes['time']}")
        return ds
    print(f"opening ARCO-ERA5 and selecting {len(times)} timesteps "
          f"({str(times[0])[:10]} .. {str(times[-1])[:10]}) ...", flush=True)
    ds = open_arco()
    want = np.array(times, dtype="datetime64[ns]")
    have = np.isin(want, ds.time.to_numpy())
    if not have.all():   # future/unfilled dates: ARCO lags ~2 months behind present
        print(f"  WARNING: {int((~have).sum())} requested timesteps not in ARCO "
              f"(future/unfilled): {[str(w)[:10] for w in want[~have]]}", flush=True)
    window = ds.sel(time=want[have])   # exact 12:00 steps, direct index lookup
    print(f"downloading + regridding {int(have.sum())} timesteps "
          f"(~{int(have.sum()) * 0.34:.1f} GB) ...", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds_out = to_arches_netcdf(window, path, batch=batch)
    print(f"saved {path}  |  steps={ds_out.sizes['time']} · vars={len(ds_out.data_vars)} · "
          f"levels={ds_out.sizes.get('level')} · grid={ds_out.sizes['latitude']}x{ds_out.sizes['longitude']}")
    return ds_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_id", required=True)
    parser.add_argument("--batch", type=int, default=2,
                        help="time steps regridded per in-memory batch")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if era5_input.nc already exists")
    return parser.parse_args()


def date_times(start, N):
    """The 12:00Z timestamps one date's era5_input.nc must contain: [start-1 .. start+N]
    (prev_state through the last rollout step), as numpy datetime64[ns]."""
    return [np.datetime64(start + timedelta(days=k), "ns") for k in range(-1, int(N) + 1)]


def main() -> None:
    args = parse_args()
    rollout_dir = get_rollout_dir(args.rollout_id)
    d = get_dict_from_json(rollout_dir / "config.json")
    # ONE era5_input.nc per date subfolder, each holding just that date's [d-1 .. d+N] window
    if "STARTS" in d:
        # legacy outer configs predate the top-level "N"; read it from a per-date config
        N = int(d["N"]) if "N" in d else get_config(f"{args.rollout_id}/{d['STARTS'][0]}").N
        items = [(datetime.strptime(name, STARTS_FMT), rollout_dir / name / "era5_input.nc")
                 for name in d["STARTS"]]
    else:
        cfg = get_config(args.rollout_id)              # plain rollout: file in the rollout dir
        N, items = int(cfg.N), [(cfg.START_TS, rollout_dir / "era5_input.nc")]
    for start, out in items:
        print(f"=== {out.parent.name} ===", flush=True)
        download_model_input(date_times(start, N), out, batch=args.batch, force=args.force)


if __name__ == "__main__":
    main()
