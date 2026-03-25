import torch

from src.utils import (
    save_state, 
    state_to_device,
)

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

# TODO: create run func and call from marimo after setup!

##### to move

from tensordict.tensordict import TensorDict
from geoarches.utils.tensordict_utils import tensordict_apply
def get_mask_tensordict(example_tdict: TensorDict, partition: str, var_idx: int, level_idx: int, spatial_mask: torch.Tensor):
    mask = tensordict_apply(lambda x: torch.zeros_like(x), example_tdict)
    mask[partition][var_idx, level_idx] = spatial_mask
    return mask

##### load #####

# data
ds = Era5Forecast(
    path="data/era5_240/full",  # default path
    load_prev=True,  # whether to load previous state
    norm_scheme="pangu",  # default normalization scheme
    domain="test",  # domain to consider. domain = 'test' loads the 2020 period
)

# model
model = 1
module_target = "geoarches.lightning_modules.{}"
MODELS = ["diffusion.DiffusionModule", "guided_diffusion.GuidedFlow"]
device = "mps"
gen_model, gen_config = load_module(
    "archesweathergen",
    module_target=module_target.format(MODELS[model]),
)
gen_model = gen_model.to(device)

##### run config #####



##### set up #####

# select X_start
X_start = ds[0]
X_start = state_to_device(X_start, device)

partition = "surface"  # str
var_idx = 2  # int
level_idx = 0  # int
spatial_mask = torch.zeros((121, 240), device=device, dtype=torch.float32)
spatial_mask[100, 100] = 1

mask = get_mask_tensordict(X_start["state"][0], partition, var_idx, level_idx, spatial_mask)

y_t = torch.tensor(1, device=device, dtype=torch.float32)

##### run #####

sampled_state = gen_model.rollout_step(
    x_cond=X_start, 
    y_t=y_t, 
    mask=mask
).cpu()

# sampled_state = X_start["state"].cpu()
# print(sampled_state)

##### save #####


timestamp = X_start["timestamp"].cpu()
sampled_state = ds.convert_to_xarray(sampled_state, timestamp)

save_state(sampled_state)