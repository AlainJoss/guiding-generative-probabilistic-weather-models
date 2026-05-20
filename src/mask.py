import xarray as xr
import torch
import numpy as np

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply

from src.dimensions import LEVELS_DICT, VARIABLES_DICT
from src.rollout_config import RolloutConfig


def get_mu_sigma(lon_left, lon_right, lat_bottom, lat_top):
    H, W = 121, 240

    mu_lon = (lon_left + lon_right) / 2
    mu_lat = (lat_bottom + lat_top) / 2

    sigma_lon = lon_right - lon_left
    sigma_lat = lat_top - lat_bottom

    mu_row = (90.0 - mu_lat) / 180.0 * H
    mu_col = (mu_lon + 180.0) / 360.0 * W

    sigma_row = sigma_lat / 180.0 * H
    sigma_col = sigma_lon / 360.0 * W

    mu = (mu_row, mu_col)
    sigma = (sigma_row, sigma_col)
    return mu, sigma


def get_masked_mean(N_slices: np.ndarray, mask: np.ndarray):
    """
    mask == normal: weights sum to 1.
    mask == bbox: weights sum to count(mask!=0)
    """
    return N_slices.sum(axis=(-1,-2)) / mask.sum()


def get_mask_from_corners(lon_left, lon_right, lat_bottom, lat_top):
    """
    Normalizes to 1.
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    lon_mask = (lon_grid >= lon_left) & (lon_grid <= lon_right)
    lat_mask = (lat_grid >= lat_bottom) & (lat_grid <= lat_top)

    mask = (lon_mask & lat_mask).astype(np.float32)
    return mask / mask.sum()

# TODO: test
def get_mask_tensordict(example_tdict: TensorDict, partition: str, var_idx: int, level_idx: int, mask_2d: torch.Tensor):
    if mask_2d is not None:
        mask = tensordict_apply(lambda x: torch.zeros_like(x), example_tdict)
        mask[partition][..., var_idx, level_idx, :, :] = mask_2d
    else:
        mask = tensordict_apply(lambda x: torch.ones_like(x), example_tdict)
    return mask

def get_masked_slices(N_slices: np.ndarray, mask: np.ndarray):
    return N_slices * mask


def minmax_slice(slice_: torch.Tensor):
    max_ = slice_.max()
    min_ = slice_.min()
    assert max_ != min_
    return (slice_ -  min_) / (max_ - min_)


def get_normal_mask(mu, sigma, shape=(121, 240)):
    y, x = np.indices(shape)
    my, mx = mu
    sy, sx = sigma
    H, W = shape

    dx = np.minimum(abs(x - mx), W - abs(x - mx))
    dy = y - my

    z = np.exp(-0.5 * ((dy / sy) ** 2 + (dx / sx) ** 2))
    return z / z.sum()


def get_mask_2d(
    mask_mode: str,
    mask_params: any
):  
    """
    examples:
    {
        "mask_mode": "bbox",
        "corners": [...],
    }
    {
        "mask_mode": "normal",
        "mu": (100, 120),
        "sigma": (20, 30),
    }
    """
    match mask_mode:
        case "bbox":
            mask_2d = get_mask_from_corners(*mask_params)
        case "normal":
            mu = mask_params[0]
            sigma = mask_params[1]
            mask_2d = get_normal_mask(mu, sigma)
        case _:
            raise ValueError(f"Invalid mask_params {mask_params}")
    return mask_2d


def get_mask_tdict(
    config: RolloutConfig,
    state_xr: xr.Dataset,
    example_tdict: TensorDict,
):
    def _xr_slice_at_time(state_xr: xr.Dataset, partition: str, var_idx: int, level_idx: int) -> torch.Tensor:
        var_name = VARIABLES_DICT[partition][var_idx]
        if partition == "surface":
            slice_xr = state_xr[var_name]
        else:
            level_val = LEVELS_DICT["level"][level_idx]
            slice_xr = state_xr[var_name].sel(level=level_val)
        return torch.as_tensor(slice_xr.values, dtype=torch.float32)

    device = example_tdict[config.partition].device

    match config.mask_mode:
        case "bbox":
            mask_2d = get_mask_from_corners(*config.mask["corners"]).to(device)

        case "normal":
            slice_ = _xr_slice_at_time(state_xr, config.partition, config.var_idx, config.level_idx)
            mask_2d = get_normal_mask(
                mu=config.mask["mu"],
                sigma=config.mask["sigma"],
                shape=tuple(slice_.shape),
                device=device,
            )

        case "normal_minmax":
            slice_ = _xr_slice_at_time(state_xr, config.partition, config.var_idx, config.level_idx).to(device)
            normal_mask = get_normal_mask(
                mu=config.mask["mu"],
                sigma=config.mask["sigma"],
                shape=tuple(slice_.shape),
                device=device,
            )
            norm_slice = minmax_slice(slice_)
            mask_2d = normal_mask * norm_slice
            mask_2d = mask_2d / mask_2d.sum().clamp_min(1e-8)

        case "slice":
            mask_2d = torch.ones(121, 240, device=device)

        case "state":
            raise NotImplementedError("mask_mode='state' not yet implemented")

        case _:
            raise ValueError(f"Invalid mask_mode '{config.mask_mode}'")

    mask = get_mask_tensordict(
        example_tdict,
        config.partition,
        config.var_idx,
        config.level_idx,
        mask_2d,
    )

    return mask
