import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time Conversion in ArchesWeather

    In this notebook, I show that something is most likely going wrong with the time conversion in ArchesWeather.

    Specifically, when passing a weather state carrying a timestamp in seconds since 1970, the generative model fails to convert it correctly to a datetime. As a result, the hour of the day is extracted incorrectly.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import pandas as pd
    from datetime import datetime

    return datetime, mo, pd, torch


@app.cell
def _():
    from geoarches.dataloaders.era5 import Era5Forecast

    return


@app.cell
def _():
    from src.utils import get_dataset

    return (get_dataset,)


@app.cell
def _(datetime, torch):
    def tensor_timestamp_to_string(
        timestamp: torch.Tensor,
        fmt: str = "%Y-%m-%d %H:%M:%S",
    ) -> str:
        ts = timestamp.item()
        return datetime.utcfromtimestamp(ts).strftime(fmt)

    return (tensor_timestamp_to_string,)


@app.cell
def _(get_dataset):
    ds = get_dataset()
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First five timestamps in the dataset:
    """)
    return


@app.cell
def _(ds):
    all_timestamps = [ts[2] for ts in ds.timestamps]
    all_timestamps[0:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Select a weather state

    I now select the weather state from the `ERA5Forecast` object corresponding to the following timestamp:
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(0, 10, step=1, show_value=True)
    slider
    return (slider,)


@app.cell
def _(slider):
    idx = slider.value
    return (idx,)


@app.cell
def _(ds, idx):
    x_start = ds[idx]
    x_start
    return (x_start,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    second state and corresponding timestamp:
    """)
    return


@app.cell
def _(tensor_timestamp_to_string, x_start):
    tensor_timestamp_to_string(x_start["timestamp"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deterministic Model

    To make a prediction, we pass the weather state to the deterministic model. The model converts the timestamp as follows:
    ```python
    time = pd.to_datetime(x_start["timestamp"].cpu().numpy(), unit="s").tz_localize(None)

    print(time, torch.tensor(time.month), torch.tensor(time.hour))
    ```
    """)
    return


@app.cell
def _(pd, torch, x_start):
    # before embedding time in ForecastModuleWithCond
    times_det = pd.to_datetime(x_start["timestamp"].cpu().numpy(), unit="s").tz_localize(None)

    (f"timestamp: {times_det}, month: {torch.tensor(times_det.month)}, hour: {torch.tensor(times_det.hour)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The conversion works as expected.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generative Model

    We pass the same state to the generative model, which converts the timestamp as follows:
    ```python
    time = pd.to_datetime(x_start["timestamp"].cpu().numpy() * 10**9).tz_localize(None)

    print(time, torch.tensor(time.month), torch.tensor(time.hour))
    ```
    """)
    return


@app.cell
def _(pd, torch, x_start):
    # in the sampling procedure before embedding time
    times = pd.to_datetime(x_start["timestamp"].cpu().numpy() * 10**9).tz_localize(None)
    (f"timestamp: {times}, month: {torch.tensor(times.month)}, hour: {torch.tensor(times.hour)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The conversion is incorrect.

    As a consequence, the generative model cannot properly use the hour of the day as conditioning information. The same issue also affects the month, except for timestamps in January.
    """)
    return


if __name__ == "__main__":
    app.run()
