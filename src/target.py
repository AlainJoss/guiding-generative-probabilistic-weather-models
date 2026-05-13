import xarray as xr
import torch

from src.utils import (
    get_slice, get_rollout,
)
from src.config import GuidedConfig
from src.funcs import get_mask_tensordict, minmax_slice  # TODO: , get_normal_slice
from src.ui.interaction import get_mask_from_corners

from tensordict.tensordict import TensorDict


def get_target_trajectory(
    config,  # let's say I use the differentials
    reference_trajectory: xr.Dataset  # the m dimension should not exist
):
    config.delta_trajectory
    config.var_idx
    config.level_idx
    config.partition

    # TODO: multiply by (1+delta_trajectory[n]) each of the reference_trajectory n

    # TODO: return torch.tensor!

def get_reference_trajectory(
    config,
    m: int | None = None
):
    """these 5 are the guidance modes defined in config, might enhance with ENUM"""
    rollout = get_rollout(config.rollout_id, config.reference)
    match config.reference:
        case "ground_truth":
            reference_trajectory = rollout
        case "unguided_m":
            reference_trajectory = ... # get m-th rollout
        case "lower_boundary":
            reference_trajectory = ... # get lower_boundary rollout
        case "upper_boundary":
            reference_trajectory = ... # get upper_boundary rollout
        case "sampled_trajectory":
            reference_trajectory = None  
            # TODO: need to add the keyword to the possible something mode
        case _:
            raise ValueError(f"Invalid reference {config.reference}")
    return reference_trajectory

def get_normal_slice(mu, sigma):
    ...

def get_mask(
    config: GuidedConfig,
    state: TensorDict, 
):
    match config.mask_mode:
        case "bbox":
            mask_2d = get_mask_from_corners(config.mask["corners"])
        case "normal":
            mask_2d = get_normal_slice(config.mask["mu"], config.mask["sigma"])
        case "normal_minmax":
            mask_2d = get_normal_slice(config.mask["mu"], config.mask["sigma"])
            slice_ = get_slice(state, config.partition, config.var_idx, config.level_idx)
            norm_slice = minmax_slice(slice_)
            mask_2d = torch.mul(norm_slice, mask_2d)
            # TODO: need a function to bring the magnitudes up similar to case "normal"
            #       maybe match maximums of two slices and sum up like that ...
        case "state":
            mask_2d = torch.ones((121, 240))
        case _:
            raise ValueError(f"Invalid mask_mode '{config.mask_mode}'")
    
    mask = get_mask_tensordict(state, config.partition, config.var_idx, config.level_idx, mask_2d)
