import json

import torch

from src.utils import (
    save_state, 
    state_to_device,
)
from src.funcs import get_mask_tensordict

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

# TODO: create run func and call from marimo after setup!

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

##### set up #####

# select X_start
X_start = ds[0]
X_start = state_to_device(X_start, device)

partition = "surface"  # str
var_idx = 2  # int
level_idx = 0  # int
spatial_mask = torch.zeros((121, 240), device=device, dtype=torch.float32)
spatial_mask[100, 100] = 1

# note: something is not unrolled here ...
mask = get_mask_tensordict(X_start["state"][0], partition, var_idx, level_idx, spatial_mask)

y_t = torch.tensor(200, device=device, dtype=torch.float32)

##### run #####

sampled_state = gen_model.rollout_step(
    x_cond=X_start, 
    y_t=y_t, 
    mask=mask
).cpu()
# TODO: this has later to be moved to rollout(N)
sampled_state = gen_model.denormalize(sampled_state)

##### save #####

# TODO: get from ui
timestamp = X_start["timestamp"].cpu()
sampled_state = ds.convert_to_xarray(sampled_state, timestamp)

save_state(sampled_state)

timestamp = "2020-01-02T00:00:00" # this is actually annoying since there is only one
start_ts_idx = ...  # enables to retrieve ground truth from timestamp to timestamp+N
partition = "surface"  # str
var = "2m_temperature"
var_idx = 2  # int
level = ...
level_idx = 0  # int  # NOTE: some variables do not have level coordinate
mask = ...
N = ...


def save_config(config):
    path = ""
    with open(path, "w"):
        json.dump(config)