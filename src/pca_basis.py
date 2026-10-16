"""Persisted regional PCA bases for flow-time analysis.

Fit a basis on the ERA5 climatology for a guided channel and pixel region, save
the mean and principal components, and reuse them to project flow trajectories.
Cache keys include the variable, level, region, source, and UTC hour.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.paths import PCA_BASIS
from src.utils import get_xr_dataset


def mask_bbox(mask_2d, rel_threshold: float = 0.5):
    """Row/col slices of the mask footprint at ``rel_threshold * max`` (relative, so
    mask normalization does not move the bbox)."""
    m = np.asarray(mask_2d, dtype=float)
    rows, cols = np.where(m >= rel_threshold * float(m.max()))
    return (slice(int(rows.min()), int(rows.max()) + 1),
            slice(int(cols.min()), int(cols.max()) + 1))


@dataclass
class PCABasis:
    mu: np.ndarray          # (F,)
    components: np.ndarray  # (n_components, F) -- first right-singular vectors
    evr: np.ndarray         # (n_components,)
    meta: dict

    @property
    def F(self):
        return int(self.mu.shape[0])


def project(basis: PCABasis, x) -> np.ndarray:
    """Project bbox-latent vectors ``(..., F)`` onto the saved basis -> ``(..., n_components)``."""
    x = np.asarray(x, dtype=float)
    assert x.shape[-1] == basis.F, (
        f"feature dim {x.shape[-1]} != basis F {basis.F}; wrong region/basis")
    return (x - basis.mu) @ basis.components.T


def pathlength(proj) -> float:
    """Sum of consecutive segment lengths of a projected trajectory ``(T, k)``."""
    proj = np.asarray(proj, dtype=float)
    if proj.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(proj, axis=0), axis=1).sum())


# --- generalized region basis: PCA on an arbitrary pixel region, fit on one UTC hour -----
# These functions take a boolean pixel selector so PCA can be fit/projected on the globe, mask
# footprint, or its complement -- and (per the experiments) only on the 12:00 climatology
# states -- to see what the guidance does at different "levels".

def region_bool(mask_2d, mode="mask", rel_threshold: float = 0.5):
    """Boolean pixel selector ``(H, W)`` for the PCA region.
    ``'globe'`` = all pixels; ``'mask'`` = footprint (``mask >= rel_threshold*max``);
    ``'!mask'`` = the complement of the footprint."""
    m = np.asarray(mask_2d, dtype=float)
    if mode == "globe":
        return np.ones(m.shape, dtype=bool)
    foot = m >= rel_threshold * float(m.max())
    return foot if mode == "mask" else ~foot


def region_latent(field, region):
    """Flatten the selected pixels of ``(..., H, W)`` fields into feature vectors ``(..., F)``."""
    field = np.asarray(field, dtype=float)
    return field[..., np.asarray(region, dtype=bool)].reshape(*field.shape[:-2], -1)


def _region_tag(region, mode, time_hour) -> str:
    reg = np.ascontiguousarray(np.asarray(region, dtype=bool))
    h = hashlib.md5(reg.tobytes()).hexdigest()[:8]
    ht = "all" if time_hour is None else f"{int(time_hour):02d}"
    return f"{mode}__{reg.shape[0]}x{reg.shape[1]}__t{ht}__{h}"


def region_basis_path(var, level, region, source, mode, time_hour) -> Path:
    lev = "sfc" if level is None else f"L{level}"
    return PCA_BASIS / f"pca__{var}__{lev}__{_region_tag(region, mode, time_hour)}__{source}.npz"


def _cloud_region(var, level, region, source, *, time_hour=None, max_points=None):
    """``(N_points, F)`` climatology cloud on the pixel ``region``, optionally restricted to a
    single UTC hour (e.g. ``time_hour=12`` -> only the 12:00 states of the year)."""
    year = 2026 if source == "era5_26" else 2020
    ds = get_xr_dataset(year)
    da = ds[var]
    if level is not None and "level" in da.dims:
        da = da.sel(level=level)
    if time_hour is not None and "time" in da.dims:
        da = da.isel(time=(da["time"].dt.hour == int(time_hour)).values)
    arr = np.asarray(da.values, dtype=float)                # (time, H, W)
    rows = arr[:, np.asarray(region, dtype=bool)].reshape(arr.shape[0], -1)   # (time, F)
    if max_points is not None and rows.shape[0] > max_points:
        idx = np.linspace(0, rows.shape[0] - 1, max_points).astype(int)
        rows = rows[idx]
    return rows


def ensure_region_basis(var, level, region, source="era5", *, mode="mask", time_hour=12,
                        n_components=3) -> PCABasis:
    """Load (or fit+persist once) the PCA basis for a pixel ``region`` of VAR@LEVEL, fit on the
    ``time_hour`` climatology states. Keyed by (var, level, mode, hour, region hash)."""
    path = region_basis_path(var, level, region, source, mode, time_hour)
    if not path.exists():
        rows = _cloud_region(var, level, region, source, time_hour=time_hour)
        mu = rows.mean(axis=0)
        _, sv, vt = np.linalg.svd(rows - mu, full_matrices=False)
        components = vt[:n_components]
        evr = (sv ** 2 / (sv ** 2).sum())[:n_components]
        meta = {
            "var": var, "level": level, "mode": mode, "time_hour": time_hour,
            "F": int(mu.shape[0]), "n_points": int(rows.shape[0]),
            "n_components": int(n_components), "source": source,
            "evr": [float(x) for x in evr],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=mu, components=components, evr=evr, meta=json.dumps(meta))
    z = np.load(path, allow_pickle=False)
    return PCABasis(mu=z["mu"], components=z["components"], evr=z["evr"],
                    meta=json.loads(str(z["meta"])))


def region_cloud_sample(var, level, region, source="era5", *, time_hour=12, max_points=400):
    """A subsample of the region climatology cloud (12:00 states) for the PCA-plot background."""
    return _cloud_region(var, level, region, source, time_hour=time_hour, max_points=max_points)
