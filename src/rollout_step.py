from tensordict.tensordict import TensorDict

from src.utils import (
    save_state, 
    ensure_result_dir,
    save_config
)
from src.funcs import get_mask_tensordict
from src.interaction import get_mask_from_corners

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow

# TODO: create run func and call from marimo after setup!

##### load #####

def rollout_step(
        ds: Era5Forecast, x_start: dict[TensorDict], 
        gen_model: GuidedFlow, 
        mask_corners, y_t, lambda_, N,
        timestamp, timestamp_idx, 
        partition, level, level_idx, var, var_idx
    ):
    device = gen_model.device
    mask = get_mask_from_corners(*mask_corners)
    mask = mask.to(device)
    y_t = y_t.to(device)

    # note: something is not unrolled here ...
    mask = get_mask_tensordict(x_start["state"][0], partition, var_idx, level_idx, mask)

    ##### run #####
    guided_state = gen_model.rollout_step(
        x_cond=x_start, 
        mask=mask,
        y_n=y_t,
        lambda_=lambda_
    )
    unguided_state = gen_model.rollout_step(
        x_cond=x_start, 
        mask=None,
        y_n=None,
        lambda_=None
    )
    # guided_state = x_start["state"]
    # unguided_state = x_start["state"]
    # TODO: this has later to be moved to rollout(N)
    guided_state = ds.denormalize(guided_state).cpu()
    unguided_state = ds.denormalize(unguided_state).cpu()

    ##### save #####
    result_dir = ensure_result_dir()

    guided_state = ds.convert_to_xarray(guided_state, x_start["timestamp"].cpu())
    unguided_state = ds.convert_to_xarray(unguided_state, x_start["timestamp"].cpu())
    save_state(result_dir, guided_state, "guided")
    save_state(result_dir, unguided_state, "unguided")

    config = {
        "N": N,
        "mask_corners": mask_corners,
        "timestamp": str(timestamp),
        "timestamp_idx": int(timestamp_idx),
        "partition": partition,
        "level": None if level is None else str(level),
        "level_idx": None if level_idx is None else int(level_idx),
        "var": var,
        "var_idx": int(var_idx),
        "y_t": y_t.detach().cpu().tolist(),
    }
    save_config(result_dir, config)