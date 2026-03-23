from geoarches.lightning_modules import load_module

# load sample from dataloader
from geoarches.dataloaders.era5 import Era5Forecast

ds = Era5Forecast(
    path="data/era5_240/full",  # default path
    load_prev=True,  # whether to load previous state
    norm_scheme="pangu",  # default normalization scheme
    domain="test",  # domain to consider. domain = 'test' loads the 2020 period
)

# loading ArchesWeatherFlow
device = "mps"

# load_module will look in modelstore/
gen_model, gen_config = load_module("archesweathergen")

gen_model = gen_model.to(device)

# run model on a sample
seed = 0
num_steps = 25  # if not provided to model.sample, model will use the default value (25)
scale_input_noise = 1.05

batch = {k: v[None].to(device) for k, v in ds[0].items()}


sample = gen_model.sample(
    batch, seed=seed, num_steps=num_steps, scale_input_noise=scale_input_noise
).cpu()