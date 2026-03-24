from pathlib import Path 
from datetime import datetime

import torch

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

# >>>> CHOOSE MODEL <<<<
model = 1

module_target = "geoarches.lightning_modules.{}"
MODELS = ["diffusion.DiffusionModule", "guided_diffusion.GuidedFlow"]

device = "mps"
gen_model, gen_config = load_module(
    "archesweathergen",
    module_target=module_target.format(MODELS[model]),
)
gen_model = gen_model.to(device)

ds = Era5Forecast(
    path="data/era5_240/full",  # default path
    load_prev=True,  # whether to load previous state
    norm_scheme="pangu",  # default normalization scheme
    domain="test",  # domain to consider. domain = 'test' loads the 2020 period
)
X_start = ds[0]
X_start = {k: v[None].to(device) for k, v in X_start.items()}

sampled_state = gen_model.rollout_step(
    X_start, 
    torch.tensor(1, device=X_start["state"].device, dtype=torch.float32), 
    torch.zeros((121, 240), device=X_start["state"].device, dtype=torch.float32)
).cpu()

def save_run(sampled_state):
    date, time = str(datetime.now().replace(microsecond=0)).split(" ")
    now = date + "_" + time
    experiment_path = Path("experiments", f"{now}")
    experiment_path.mkdir(parents=True, exist_ok=True)
    experiment_path = experiment_path /f"{MODELS[model]}.pt"
    torch.save(sampled_state["surface"][0][2, 0], experiment_path)
    print("run saved")

save_run(sampled_state)