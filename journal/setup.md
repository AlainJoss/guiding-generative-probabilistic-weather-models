# Setup

Steps:
```bash
# download dependencies
uv sync
# now remove the poetry.lock

# for storing my stuff
mkdir experiments

# for storing downloaded/generated data
mkdir data evalstore modelstore wandblogs

# download model weights https://geoarches.readthedocs.io/en/latest/archesweather/setup/
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
MODELS=("archesweather-m-seed0" "archesweather-m-seed1" "archesweather-m-skip-seed0" "archesweather-m-skip-seed1" "archesweathergen")

for MOD in "${MODELS[@]}"; do
    mkdir -p "modelstore/$MOD/checkpoints"
    wget -O "modelstore/$MOD/checkpoints/checkpoint.ckpt" "$src/${MOD}_checkpoint.ckpt"
    wget -O "modelstore/$MOD/config.yaml" "$src/${MOD}_config.yaml"
done

# download ERA5 quantile statistics
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
wget -O geoarches/stats/era5-quantiles-2016_2022.nc $src/era5-quantiles-2016_2022.nc

# change download code of dl_era because google cloud complaints with auth
xr.open_zarr(
    climatology_path,
    storage_options={"token": "anon"}
)
# download data 
uv run geoarches/download/dl_era.py --years 2020
uv run geoarches/download/dl_era.py  # full dataset
# copy paste docs/archesweather/run.ipynb and change from cd ../.. to cd ..
```