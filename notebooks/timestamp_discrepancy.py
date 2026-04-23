import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time encoding in ArchesWeather

    In this notebook I showcase that, most probably something is going south with the time encoding in ArchesWeather. This might help explain why the models seems to have no notion of time of the day.

    For instance, when we make a 3 day rollout (4*3=12 steps), the 2m_temperature (at any selected location) will not show the expected daily fluctuation when compared to the ground.
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

    return (Era5Forecast,)


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
def _(Era5Forecast):
    ds = Era5Forecast(
        path="data/era5_240/full",
        load_prev=True, 
        norm_scheme="pangu",
        domain="test",  
        lead_time_hours=6,
    )
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### List of all timestamps

    This is extracted from the dataset object containing all tensor_dicts (weather states).
    """)
    return


@app.cell
def _(ds):
    all_timestamps = [ts[2] for ts in ds.timestamps]
    all_timestamps[0:8]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Example weather state tensor dict.
    """)
    return


@app.cell
def _(ds):
    ds[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Index slider
    Slide the index of the state to retrieve. Each state has a timestamp in this form: "timestamp":1577858400.
    This represent the time has passed from 1970 in seconds, and has to be converted in a timestamps with date and time (year, month, day, hour).

    Spoiler: the conversion is correct in the deterministic model, wrong in the generative one.

    Slide the index of the weather states to retrieve different ones and compare the conversion done by the two models (det vs. gen).
    """)
    return


@app.cell
def _(mo):
    slide_x_start = mo.ui.slider(start=0, stop=16, step=1, label="slide start index: ")
    slide_x_start
    return (slide_x_start,)


@app.cell
def _(slide_x_start):
    x_start_index = slide_x_start.value
    return (x_start_index,)


@app.cell
def _(ds, x_start_index):
    x_start = ds[x_start_index]
    return (x_start,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### My timestamp extraction
    """)
    return


@app.cell
def _(tensor_timestamp_to_string, x_start):
    # makes sense, because we also load the previous
    # in fact, if we set previous to False the timestamp will start at 00:00, nice
    tensor_timestamp_to_string(x_start["timestamp"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here I use the datetime module instead of pandas to be sure about the conversion.

    The timestamp is correct. It is shifted by 6h wrt to the list of all timestamps since we also load the previous state. If we set the load_prev = False it correctly aligns with the list of all timestamps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time extraction in deterministic model
    """)
    return


@app.cell
def _(pd, torch, x_start):
    # before embedding time in ForecastModuleWithCond
    times_det = pd.to_datetime(x_start["timestamp"].cpu().numpy(), unit="s").tz_localize(None)

    times_det, torch.tensor(times_det.month), torch.tensor(times_det.hour)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Correct.

    Convertion used:
    ```python
    time = pd.to_datetime(x_start["timestamp"].cpu().numpy(), unit="s").tz_localize(None)

    print(time, torch.tensor(times_det.month), torch.tensor(times_det.hour))
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time extraction in generative model
    """)
    return


@app.cell
def _(pd, torch, x_start):
    # in the sampling procedure before embedding time
    times = pd.to_datetime(x_start["timestamp"].cpu().numpy() * 10**9).tz_localize(None)
    # should be rounded up, but the model will learn the right thing
    times, torch.tensor(times.month), torch.tensor(times.hour)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Incorrect.

    Convertion used:
    ```python
    time = pd.to_datetime(x_start["timestamp"].cpu().numpy() * 10**9).tz_localize(None)

    print(time, torch.tensor(times_det.month), torch.tensor(times_det.hour))
    ```

    The time extraction in the generative model goes south because multiplying the seconds by 10^9 doesn't achieve the correct result, which in turn is achieved when using unit="s". This conversion has probably not been tested before including it in the codebase.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Consequences

    Since this also affects the training of the diffusion model, I assume that the model doesn't have any notion of time, because it only sees 1 and 23 ... .

    Furthermore, I previously tested the RMSE of the deterministic model compared to the one of the generative one, which was comparatively (significantly) worse. This might explain why that is. Not having a (correct) notion of time of the day predictably really harmful for modeling weather correctly.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
