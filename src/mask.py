import xarray as xr
import torch
import numpy as np

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply


def _wrap_lon(lon):
    """Wrap a longitude (or array) into [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def get_mu_sigma(lon_left, lon_right, lat_bottom, lat_top):
    H, W = 121, 240

    mu_lon = _wrap_lon((lon_left + lon_right) / 2)
    mu_lat = (lat_bottom + lat_top) / 2

    # half the box side per axis (matches SPHERICAL's half-diagonal convention)
    sigma_lon = (lon_right - lon_left) / 2
    sigma_lat = (lat_top - lat_bottom) /2

    mu_row = (90.0 - mu_lat) / 180.0 * H
    mu_col = (mu_lon + 180.0) / 360.0 * W

    sigma_row = sigma_lat / 180.0 * H
    sigma_col = sigma_lon / 360.0 * W

    mu = (mu_row, mu_col)
    sigma = (sigma_row, sigma_col)
    return mu, sigma


def get_bbox_mask(lon_left, lon_right, lat_bottom, lat_top):
    """
    Normalizes to 1. Boxes may cross the dateline (lon_left/right outside
    [-180, 180] or left > right after wrapping): membership wraps in lon.
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    if lon_right - lon_left >= 360.0:
        lon_mask = np.ones_like(lon_grid, dtype=bool)
    else:
        left, right = _wrap_lon(lon_left), _wrap_lon(lon_right)
        if left <= right:
            lon_mask = (lon_grid >= left) & (lon_grid <= right)
        else:  # crosses the dateline
            lon_mask = (lon_grid >= left) | (lon_grid <= right)
    # a rectangle cannot extend past a pole: clamp (corners may be unclamped)
    lat_mask = (lat_grid >= max(lat_bottom, -90.0)) & (lat_grid <= min(lat_top, 90.0))

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


def _haversine(lat1, lon1, lat2, lon2):
    # great-circle angle between two points; all arguments in radians
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def get_spherical_mask(lon_left, lon_right, lat_bottom, lat_top, sigma_div=2.0):
    """
    Isotropic (discrete) Gaussian on the sphere: kernel in great-circle
    distance from the box center, one scale sigma = box diagonal / sigma_div,
    normalized to sum to 1. cos(lat) cell-area weights make the masked
    statistic an AREA-TRUE average on the sphere (the equiangular grid
    oversamples high latitudes). The kernel center is precompensated
    poleward by sigma^2 tan(mu_lat) so the PEAK of kernel*cos sits exactly
    on the chosen center.
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(np.radians(lon_c), np.radians(lat_c))

    mu_lat = np.radians((lat_bottom + lat_top) / 2)
    mu_lon = np.radians(_wrap_lon((lon_left + lon_right) / 2))

    # one isotropic scale: box diagonal / sigma_div in physical (angular)
    # units. computed from the side lengths (tangent plane at the center),
    # NOT from a corner-to-corner haversine, so it stays well-defined when
    # the box sticks out over a pole (|corner lat| > 90)
    dlat = np.radians(lat_top - lat_bottom)
    dlon = np.radians(lon_right - lon_left) * np.cos(mu_lat)
    sigma = np.hypot(dlat, dlon) / sigma_div

    # precompensate: the cos(lat) weight pulls the product's peak equatorward;
    # shifting the kernel center poleward by sigma^2 tan(mu) puts the peak of
    # exp(-d^2/2s^2)*cos(lat) exactly back on mu (stationarity at mu)
    mu_lat_k = np.clip(
        mu_lat + sigma**2 * np.tan(mu_lat),
        np.radians(-89.5), np.radians(89.5),
    )

    d = _haversine(lat_grid, lon_grid, mu_lat_k, mu_lon)

    z = np.exp(-0.5 * (d / sigma) ** 2)
    z = z * np.cos(lat_grid)  # cell-area weight -> area-true statistic
    return (z / z.sum()).astype(np.float32)


def get_elliptical_mask(lon_left, lon_right, lat_bottom, lat_top, sigma_div=2.0):
    """
    Gaussian on the sphere with a DIAGONAL covariance (anisotropic): the
    great-circle offset from the box center is decomposed into north-south /
    east-west components via the initial bearing, each scaled by its own
    sigma = corresponding box side / sigma_div (in physical angular units).
    Reduces to the isotropic behaviour for square boxes; wraps in lon and
    over the poles like get_spherical_mask. cos(lat) cell-area weights make
    the masked statistic an area-true average (see get_spherical_mask).
    Normalized to sum to 1.
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(np.radians(lon_c), np.radians(lat_c))

    mu_lat = np.radians((lat_bottom + lat_top) / 2)
    mu_lon = np.radians(_wrap_lon((lon_left + lon_right) / 2))

    # per-axis scales: box sides / sigma_div, east-west in physical units at
    # the center latitude (guarded away from zero for near-pole centers)
    sig_lat = np.radians(lat_top - lat_bottom) / sigma_div
    sig_lon = np.radians(lon_right - lon_left) * max(np.cos(mu_lat), 1e-3) / sigma_div

    # precompensate the kernel center so the peak of kernel*cos(lat) sits
    # exactly on mu (see get_spherical_mask)
    mu_lat_k = np.clip(
        mu_lat + sig_lat**2 * np.tan(mu_lat),
        np.radians(-89.5), np.radians(89.5),
    )

    d = _haversine(lat_grid, lon_grid, mu_lat_k, mu_lon)

    # initial bearing from the (shifted) kernel center to each grid point
    dlmb = lon_grid - mu_lon
    theta = np.arctan2(
        np.sin(dlmb) * np.cos(lat_grid),
        np.cos(mu_lat_k) * np.sin(lat_grid)
        - np.sin(mu_lat_k) * np.cos(lat_grid) * np.cos(dlmb),
    )
    d_ns = d * np.cos(theta)
    d_ew = d * np.sin(theta)

    z = np.exp(-0.5 * ((d_ns / sig_lat) ** 2 + (d_ew / sig_lon) ** 2))
    z = z * np.cos(lat_grid)  # cell-area weight -> area-true statistic
    return (z / z.sum()).astype(np.float32)


def get_great_circle_field(lon_left, lon_right, lat_bottom, lat_top):
    """
    Great-circle distance (in degrees) from the box center at every grid
    point. Contouring this field draws iso-distance rings radiating from
    the mask center (wraps in lon and over the poles like the masks).
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(np.radians(lon_c), np.radians(lat_c))

    mu_lat = np.radians((lat_bottom + lat_top) / 2)
    mu_lon = np.radians(_wrap_lon((lon_left + lon_right) / 2))

    d = _haversine(lat_grid, lon_grid, mu_lat, mu_lon)
    return np.degrees(d).astype(np.float32)


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
    mask_corners: any,
    sigma_div: float = 2.0,  # box extent / sigma (SPHERICAL and ELLIPTICAL only)
):
    match mask_mode:
        case "BBOX":
            mask_2d = get_bbox_mask(*mask_corners)
        case "GAUSSIAN":
            mask_2d = get_normal_mask(*mask_corners)
        case "SPHERICAL":
            mask_2d = get_spherical_mask(*mask_corners, sigma_div=sigma_div)
        case "ELLIPTICAL":
            mask_2d = get_elliptical_mask(*mask_corners, sigma_div=sigma_div)
        case _:
            raise ValueError(f"Invalid mask_mode {mask_mode!r} (params {mask_corners})")
    return mask_2d