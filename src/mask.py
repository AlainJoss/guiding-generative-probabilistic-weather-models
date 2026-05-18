import xarray as xr
import torch

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply

from src.dimensions import LEVELS_DICT, VARIABLES_DICT
from src.config import RolloutConfig
from src.ui.interaction import get_mask_from_corners

# TODO: test
def get_mask_tensordict(example_tdict: TensorDict, partition: str, var_idx: int, level_idx: int, mask_2d: torch.Tensor):
    if mask_2d is not None:
        mask = tensordict_apply(lambda x: torch.zeros_like(x), example_tdict)
        mask[partition][..., var_idx, level_idx, :, :] = mask_2d
    else:
        mask = tensordict_apply(lambda x: torch.ones_like(x), example_tdict)
    return mask


def minmax_slice(slice_: torch.Tensor):
    max_ = slice_.max()
    min_ = slice_.min()
    assert max_ != min_
    return (slice_ -  min_) / (max_ - min_)


def get_normal_slice(
    mu: tuple[float, float],
    sigma: tuple[float, float],
    shape: tuple[int, int] = (121, 240),
    device: torch.device | None = None,
):
    """
    Broad 2D Gaussian mask.

    mu:    (lat_idx, lon_idx)
    sigma: (sigma_lat, sigma_lon)
    """
    H, W = shape
    mu_y, mu_x = mu
    sigma_y, sigma_x = sigma

    y = torch.arange(H, dtype=torch.float32, device=device)
    x = torch.arange(W, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    mask = torch.exp(
        -0.5 * (
            ((yy - mu_y) / sigma_y) ** 2
            + ((xx - mu_x) / sigma_x) ** 2
        )
    )

    return mask / mask.sum().clamp_min(1e-8)


def _xr_slice_at_time(state_xr: xr.Dataset, partition: str, var_idx: int, level_idx: int) -> torch.Tensor:
    var_name = VARIABLES_DICT[partition][var_idx]
    if partition == "surface":
        slice_xr = state_xr[var_name]
    else:
        level_val = LEVELS_DICT["level"][level_idx]
        slice_xr = state_xr[var_name].sel(level=level_val)
    return torch.as_tensor(slice_xr.values, dtype=torch.float32)


def get_mask(
    config: RolloutConfig,
    state_xr: xr.Dataset,
    example_tdict: TensorDict,
):
    device = example_tdict[config.partition].device

    match config.mask_mode:
        case "bbox":
            mask_2d = get_mask_from_corners(*config.mask["corners"]).to(device)

        case "normal":
            slice_ = _xr_slice_at_time(state_xr, config.partition, config.var_idx, config.level_idx)
            mask_2d = get_normal_slice(
                mu=config.mask["mu"],
                sigma=config.mask["sigma"],
                shape=tuple(slice_.shape),
                device=device,
            )

        case "normal_minmax":
            slice_ = _xr_slice_at_time(state_xr, config.partition, config.var_idx, config.level_idx).to(device)
            normal_mask = get_normal_slice(
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
