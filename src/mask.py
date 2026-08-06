import xarray as xr
import torch
import numpy as np

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply
from geoarches.metrics.metric_base import compute_lat_weights_weatherbench


def _lat_area_weights(n_lat: int) -> np.ndarray:
    """Per-row latitude cell-area weight for a DESCENDING 90..-90 grid, using the exact
    WeatherBench/arches scheme (geoarches.metrics.metric_base.compute_lat_weights_weatherbench):
    a_i = sin(upper_i) - sin(lower_i) = the integral of cos(lat) over cell i, with the pole
    cells clamped to +/-90. arches returns weights for ascending -90..90 -> flip to our order."""
    w = compute_lat_weights_weatherbench(n_lat).squeeze(-1).cpu().numpy().astype(np.float64)
    return w[::-1].copy()


def _wrap_lon(lon):
    """Wrap a longitude (or array) into [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def snap_bbox_corners(lon_left, lon_right, lat_bottom, lat_top):
    """Snap each corner to the nearest display-grid cell edge (lon edges = linspace(-180,180,241);
    lat edges = linspace(90,-90,122)). Combined with centre-membership this makes the mask enclose
    WHOLE pixels and lets the drawn rectangle land exactly on pixel boundaries."""
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, 121 + 1, endpoint=True)
    snap = lambda v, e: float(e[int(np.argmin(np.abs(e - v)))])
    return (snap(lon_left, lon_e), snap(lon_right, lon_e),
            snap(lat_bottom, lat_e), snap(lat_top, lat_e))


def effective_bbox(lon_left, lon_right, lat_bottom, lat_top, sigma_div=2.0):
    """The box the mask actually uses and that should be DRAWN: rescale about the centre by
    1/sigma_div (extent/sigma_div convention), then snap the corners to the grid cell edges."""
    mu_lon, mu_lat = (lon_left + lon_right) / 2.0, (lat_bottom + lat_top) / 2.0
    half_lon, half_lat = (lon_right - lon_left) / sigma_div, (lat_top - lat_bottom) / sigma_div
    return snap_bbox_corners(mu_lon - half_lon, mu_lon + half_lon, mu_lat - half_lat, mu_lat + half_lat)


def get_bbox_mask(lon_left, lon_right, lat_bottom, lat_top, sigma_div=2.0):
    """
    cos(lat) cell-area weighted (like the elliptical mask), normalized to 1:
    the masked statistic is an AREA-TRUE average on the sphere. Boxes may
    cross the dateline (lon_left/right outside
    [-180, 180] or left > right after wrapping): membership wraps in lon.
    sigma_div rescales the box about its center with the same extent/sigma_div
    convention as the elliptical mask (half-side = side / sigma_div): the
    default 2.0 reproduces the drawn base box, 4.0 halves it, 1.0 doubles it.
    """
    # rescale by 1/sigma_div and snap corners to the display-grid cell edges
    lon_left, lon_right, lat_bottom, lat_top = effective_bbox(
        lon_left, lon_right, lat_bottom, lat_top, sigma_div)

    # membership grid = the DISPLAY grid (pcolormesh) so the mask lines up with the drawn pixels:
    # cell edges at linspace, centres at their midpoints
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
    # snapped corners are cell edges; a cell is IN iff its centre is between them (whole pixels)
    lat_mask = (lat_grid >= min(lat_bottom, lat_top)) & (lat_grid <= max(lat_bottom, lat_top))

    mask = (lon_mask & lat_mask).astype(np.float32)
    mask = mask * _lat_area_weights(len(lat_c))[:, None]  # arches/WeatherBench per-lat cell area
    return (mask / mask.sum()).astype(np.float32)


def _haversine(lat1, lon1, lat2, lon2):
    # great-circle angle between two points; all arguments in radians
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def get_elliptical_mask(lon_left, lon_right, lat_bottom, lat_top, sigma_div=2.0):
    """
    Gaussian on the sphere with a DIAGONAL covariance (anisotropic): the
    great-circle offset from the box center is decomposed into north-south /
    east-west components via the initial bearing, each scaled by its own
    sigma = corresponding box side / sigma_div (in physical angular units).
    Reduces to the isotropic behaviour for square boxes; wraps in lon and
    over the poles. cos(lat) cell-area weights make the masked statistic an
    AREA-TRUE average on the sphere (the equiangular grid oversamples high
    latitudes). Normalized to sum to 1.
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = np.linspace(90.0, -90.0, 121)   # arches data latitudes (poles included)

    lon_grid, lat_grid = np.meshgrid(np.radians(lon_c), np.radians(lat_c))

    mu_lat = np.radians((lat_bottom + lat_top) / 2)
    mu_lon = np.radians(_wrap_lon((lon_left + lon_right) / 2))

    # per-axis scales: box sides / sigma_div, east-west in physical units at
    # the center latitude (guarded away from zero for near-pole centers)
    sig_lat = np.radians(lat_top - lat_bottom) / sigma_div
    sig_lon = np.radians(lon_right - lon_left) * max(np.cos(mu_lat), 1e-3) / sigma_div

    # precompensate the kernel center: the cos(lat) weight pulls the product's
    # peak equatorward; shifting poleward by sig^2 tan(mu) puts it back on mu
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
    z = z * _lat_area_weights(len(lat_c))[:, None]  # arches/WeatherBench per-lat cell area
    return (z / z.sum()).astype(np.float32)


def get_great_circle_field(lon_left, lon_right, lat_bottom, lat_top):
    """
    Great-circle distance (in degrees) from the box center at every grid
    point. Contouring this field draws iso-distance rings radiating from
    the mask center (wraps in lon and over the poles like the masks).
    """
    lon_e = np.linspace(-180.0, 180.0, 240 + 1, endpoint=True)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = np.linspace(90.0, -90.0, 121)   # arches data latitudes (poles included)

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


def shift_corners(mask_corners, mask_shift: str = "none"):
    """Translate (lon_left, lon_right, lat_bottom, lat_top) by a pixel shift.

    mask_shift is "none" or "dir@px" with dir in {right, left, up, down};
    1 px = 1.5 deg on the 240x121 grid. The mask builders re-apply their own
    lon wrap / pole clamp and cos(lat) area weights at the shifted location.
    """
    if not mask_shift or mask_shift == "none":
        return mask_corners
    direction, px = mask_shift.split("@", 1)
    step = 1.5 * float(px)
    dlon = {"right": step, "left": -step}.get(direction, 0.0)
    dlat = {"up": step, "down": -step}.get(direction, 0.0)
    if dlon == 0.0 and dlat == 0.0:
        raise ValueError(f"Invalid mask_shift {mask_shift!r} (dir must be right/left/up/down)")
    lon_left, lon_right, lat_bottom, lat_top = mask_corners
    return (lon_left + dlon, lon_right + dlon, lat_bottom + dlat, lat_top + dlat)


def get_mask_2d(
    mask_mode: str,
    mask_corners: any,
    sigma_div: float = 2.0,  # box extent / scale, all modes (2.0 = base box)
    mask_shift: str = "none",  # "none" or "dir@px" translation (1 px = 1.5 deg)
):
    mask_corners = shift_corners(mask_corners, mask_shift)
    match mask_mode:
        case "BBOX":
            mask_2d = get_bbox_mask(*mask_corners, sigma_div=sigma_div)
        case "ELLIPTICAL":
            mask_2d = get_elliptical_mask(*mask_corners, sigma_div=sigma_div)
        case _:
            raise ValueError(f"Invalid mask_mode {mask_mode!r} (params {mask_corners})")
    return mask_2d