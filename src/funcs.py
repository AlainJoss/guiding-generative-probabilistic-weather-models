import torch
import numpy as np

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply


def avg_over_mask(state, mask):
    avg = torch.sum(mask * state) / torch.sum(mask)
    return avg

def get_mask_tensordict(example_tdict: TensorDict, partition: str, var_idx: int, level_idx: int, mask_2d: torch.Tensor):
    mask = tensordict_apply(lambda x: torch.zeros_like(x), example_tdict)
    mask[partition][var_idx, level_idx] = mask_2d
    return mask

def get_guidance(drift: float, anchor: float):
    return anchor + drift * np.abs(anchor)

def get_guidance_trajectory(drift: list[float], anchor: float):
    return [get_guidance(d, anchor) for d in drift]