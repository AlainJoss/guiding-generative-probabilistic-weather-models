import xarray as xr
import torch
import numpy as np

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply


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


def get_bbox_mask(lon_left, lon_right, lat_bottom, lat_top):
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


def get_normal_mask(lon_left, lon_right, lat_bottom, lat_top):
    mu, sigma = get_mu_sigma(lon_left, lon_right, lat_bottom, lat_top)
    shape = (121, 240)
    y, x = np.indices(shape)
    my, mx = mu
    sy, sx = sigma
    H, W = shape

    dx = np.minimum(abs(x - mx), W - abs(x - mx))
    dy = y - my

    z = np.exp(-0.5 * ((dy / sy) ** 2 + (dx / sx) ** 2))
    return z / z.sum()


def get_mask_center(lon_left, lon_right, lat_bottom, lat_top):
    x=lon_right+lon_left
    y=lat_top+lat_bottom
    return x/2,y/2


def get_masked_mean(N_slices: np.ndarray, mask: np.ndarray):
    masked_slices = N_slices * mask
    return masked_slices.sum(axis=(-1,-2))


def get_mask_tdict(example_tdict: TensorDict, partition: str, var_idx: int, level_idx: int, mask_2d: np.ndarray):
    assert mask_2d is not None
    mask_2d = torch.tensor(mask_2d)
    mask = tensordict_apply(lambda x: torch.zeros_like(x), example_tdict)
    mask[partition][..., var_idx, level_idx, :, :] = mask_2d
    return mask


def get_mask_2d(
    mask_mode: str,
    mask_corners: any
):  
    match mask_mode:
        case "BBOX":
            mask_2d = get_bbox_mask(*mask_corners)
        case "GAUSSIAN":
            mask_2d = get_normal_mask(*mask_corners)
        case _:
            raise ValueError(f"Invalid mask_mode {mask_mode!r} (params {mask_corners})")
    return mask_2d