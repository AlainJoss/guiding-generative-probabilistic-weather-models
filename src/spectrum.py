import numpy as np
import pyshtools as pysh


def power_spectrum(field, latitudes=None):
    """Spherical-harmonic power spectrum of a single (lat, lon) field.

    Same pyshtools Driscoll-Healy expansion as
    geoarches.metrics.spherical_power_spectrum.PowerSpectrum, but for one 2D
    slice instead of a batched torchmetric -- handy for comparing individual
    states (gt / ung / ung_gui / gui) in the analyze notebook.

    Args:
        field: 2D array (nlat, nlon) on a regular lat-lon grid.
        latitudes: optional 1D latitude coordinate. If given and ascending
            (south->north), the field is flipped to run north->south, which is
            the ordering the DH grid expects.

    Returns:
        (degrees, power): degrees is an int array [0..lmax]; power[l] is the
        power (variance) at spherical-harmonic degree l.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"expected a 2D (lat, lon) slice, got shape {field.shape}")

    if latitudes is not None and np.asarray(latitudes)[0] < np.asarray(latitudes)[-1]:
        field = field[::-1]  # order north -> south

    # DH grid needs nlon == 2*nlat; weather grids carry both poles (nlat odd),
    # so drop the trailing (south) pole row to make nlat even.
    nlat, nlon = field.shape
    if nlon == 2 * (nlat - 1):
        field = field[:-1]

    power = pysh.SHGrid.from_array(field).expand().spectrum()
    return np.arange(power.shape[0]), power


def activity(field, latitudes, climatology=None):
    """Activity: latitude-weighted spatial standard deviation of the
    climatology-removed forecast -- a smoothness diagnostic alongside the power
    spectrum (deterministic/over-smoothed fields show lower activity).

    Args:
        field: 2D array (nlat, nlon) on a regular lat-lon grid.
        latitudes: 1D latitude coordinate (degrees), length nlat.
        climatology: optional 2D (nlat, nlon) climatology aligned to `field`
            (e.g. from `climatology_slice`). If given, the anomaly is
            field - climatology (paper definition). If None, the climatology is
            proxied by the field's zonal mean (longitude average per latitude),
            i.e. the stationary-eddy field -- use when no climatology is available.
    """
    field = np.asarray(field, dtype=np.float64)
    if climatology is not None:
        anomaly = field - np.asarray(climatology, dtype=np.float64)
    else:
        anomaly = field - field.mean(axis=-1, keepdims=True)  # zonal-mean proxy
    w = np.clip(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0, None)
    weights = np.broadcast_to(w[:, None], anomaly.shape)
    mean = np.average(anomaly, weights=weights)
    var = np.average((anomaly - mean) ** 2, weights=weights)
    return float(np.sqrt(var))


def climatology_slice(clim, variable, valid_time, latitudes, level=None):
    """2D (nlat, nlon) climatology for `variable` at the day-of-year and hour of
    `valid_time`, reordered to `latitudes` (the WeatherBench2 climatology is
    stored south->north, the rollout grids north->south).

    Args:
        clim: climatology Dataset (dims: dayofyear, hour, latitude, longitude,
            and level for level variables) -- e.g. xr.open_dataset(DATA/"era5_240_clim.nc").
        variable: data variable name (e.g. "2m_temperature", "geopotential").
        valid_time: forecast valid time (np.datetime64 / datetime / str).
        latitudes: target latitude ordering (1D array of degrees).
        level: pressure level to select for level variables; None for surface.
    """
    import pandas as pd

    ts = pd.Timestamp(valid_time)
    da = clim[variable]
    if level is not None and "level" in da.dims:
        da = da.sel(level=level, method="nearest")
    da = da.sel(dayofyear=int(ts.dayofyear)).sel(hour=int(ts.hour), method="nearest")
    da = da.transpose("latitude", "longitude")
    # match the target latitude ordering by sorting (not reindex: the grids share
    # the same points but stored in opposite order, and float coords won't match exactly)
    latitudes = np.asarray(latitudes)
    da = da.sortby("latitude", ascending=bool(latitudes[0] < latitudes[-1]))
    return np.asarray(da.values, dtype=np.float64)


def log_spectral_distance(p, q, l_min=1, l_max=None):
    """RMS log-ratio between two power spectra -- a single scale-aware summary of
    how different they are: sqrt(mean_l (ln p_l - ln q_l)^2) over l in [l_min, l_max).

    The log turns it into a per-degree ratio, so over/under-power at every scale
    counts equally (a raw |p-q| is dominated by the low-l, high-power modes).
    l=0 (the field mean) is skipped by default.
    """
    p = np.asarray(p, dtype=np.float64)[l_min:l_max]
    q = np.asarray(q, dtype=np.float64)[l_min:l_max]
    return float(np.sqrt(np.mean((np.log(p) - np.log(q)) ** 2)))


def spectral_bias(p, q, l_min=1, l_max=None):
    """Signed mean log-ratio mean_l ln(p_l / q_l) over l in [l_min, l_max).

    Direction companion to log_spectral_distance: < 0 means p is under-powered
    (smoother) than q, > 0 means over-powered (rougher). l=0 skipped by default.
    """
    p = np.asarray(p, dtype=np.float64)[l_min:l_max]
    q = np.asarray(q, dtype=np.float64)[l_min:l_max]
    return float(np.mean(np.log(p) - np.log(q)))
