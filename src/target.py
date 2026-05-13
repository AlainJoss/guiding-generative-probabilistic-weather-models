import xarray as xr
import torch

from src.utils import (
    save_to_json,
    rollout_to_xarray,
    batchify_and_move,
    get_dataset,
    get_x_cond,
    make_hash,
    list_floats_to_tensors,
    update_experiment_params,
    ensure_rollout_dir
)
from src.funcs import get_mask_tensordict
from src.ui.interaction import get_mask_from_corners

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict

def get_masked_target(mask, y, state, config):
    # maybe config is still a param so I can pick what I need in dict format with different dephts 
    # depending on the mask_mode
    config.partition
    config.level_idx
    config.var_idx

    pass
