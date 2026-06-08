import torch
import xarray as xr

from geoarches.dataloaders.era5 import level_variables, pressure_levels, surface_variables
from geoarches.paths import STATS_PATH


class XarrayNormalizer:
    """Pangu-style mean/std normalization for xr.Datasets keyed by ERA5 variable name."""

    def __init__(self, stats_file: str = "pangu_norm_stats2_with_w.pt"):
        s = torch.load(STATS_PATH / stats_file, weights_only=True)
        lm = s["level_mean"].squeeze(-1).squeeze(-1).numpy()  # (6, 13)
        ls = s["level_std"].squeeze(-1).squeeze(-1).numpy()
        sm = s["surface_mean"].squeeze().numpy()              # (4,)
        ss = s["surface_std"].squeeze().numpy()
        self._mean = {v: lm[i] for i, v in enumerate(level_variables)} | {v: sm[i] for i, v in enumerate(surface_variables)}
        self._std = {v: ls[i] for i, v in enumerate(level_variables)} | {v: ss[i] for i, v in enumerate(surface_variables)}

    def _as_da(self, arr):
        return xr.DataArray(arr, dims=["level"], coords={"level": pressure_levels}) if arr.ndim == 1 else float(arr)

    def normalize(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.map(lambda da: (da - self._as_da(self._mean[da.name])) / self._as_da(self._std[da.name]))

    def denormalize(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.map(lambda da: da * self._as_da(self._std[da.name]) + self._as_da(self._mean[da.name]))
