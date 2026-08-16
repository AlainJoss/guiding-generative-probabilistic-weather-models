# Guiding Generative Probabilistic Weather Models

Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
uv run marimo edit src/ui/guidance.py --watch --no-token
uv run marimo edit src/ui/flow_time_tests.py --watch --no-token
uv run marimo edit src/ui/experiment_builder.py --watch --no-token
uv run marimo edit src/ui/latent_trajectories.py --watch --no-token

```

Experiment runner:
```bash
python -m src.run --rollout_id 2026-08-14_11:50:36 --rollout_type gui
```

latent_trajectories.py

Terminal setup:
```bash
# on Renku only once after creating the session 
cd guiding-generative-probabilistic-weather-models
ln -s ../data data 
mkdir modelstore stats
cp -r data/modelstore/* modelstore/
cp -r data/stats/* stats/

# locally
ln -s ~/switchdrive data
git push
git pull
```

Retrieve data with earth-kit:
```python
def retrieve_data(variable, date_range, lat, lng):
    # Define the dataset and request parameters
    dataset = "reanalysis-era5-single-levels-timeseries"
    request = {
        "variable": [
        variable,  # Variable to retrieve
        ],
        "date": date_range,  # Date range for the data
        "location": {"longitude": lng, "latitude": lat},  # Location coordinates
        "data_format": "netcdf"  # Format of the retrieved data
    }

    # Use "earthkit" to retrieve the data
    ekds = earthkit.data.from_source(
        "cds", dataset, request
    ).to_xarray()

    return ekds
```
... and the docs: https://earthkit-data.readthedocs.io/en/latest/concepts/inputs/from_source.html#data-sources-zarr

Access ARCO zarr files:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=analysis_ready_data

```python
import xarray as xr

# Geo-chunked data for access optimised along the time dimension (e.g. for time-series at a single point)
# Surface
surface_geo_url = "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
# Wave
wave_geo_url = "https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr"

# Time-chunked data for access optimised in spatial dimensions (e.g. for global maps)
# Surface
surface_time_url = "https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr"
# Wave
wave_time_url = "https://arco.datastores.ecmwf.int/cadl-arco-time-003/arco/reanalysis_era5_single_levels/wav/timeChunked.zarr"

# Open one of the Zarr objects with xarray, the default example opens the geo-chunked surface variables
ds = xr.open_zarr(
    surface_geo_url,
    consolidated=True,
    storage_options={
        "headers": {"Authorization": f"Bearer <CDS-API-KEY>"}
    }
)

# Inspect the variables
print(ds)
```