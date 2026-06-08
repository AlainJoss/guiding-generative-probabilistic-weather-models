import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # zarr playground
    """)
    return


@app.cell
def _():
    import marimo as mo
    import zarr
    import numpy as np
    import xarray as xr
    import dask.array as da

    return mo, np, xr, zarr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## creating a zarr
    """)
    return


@app.cell
def _():
    shape=(1000, 1000)
    store = "example.zarr"
    return shape, store


@app.cell(disabled=True)
def _(shape, store, zarr):
    zarr.create(
        store=store, shape=shape, chunks=(10, 10), dtype="f2", 
        fill_value=None, overwrite=True, zarr_version=2
    )
    return


@app.cell
def _(zarr):
    z = zarr.open("example.zarr")
    print(z.info)
    return (z,)


@app.cell
def _(np, shape, z):
    z[:, :] = np.random.random(shape)
    print(z.info)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## xr to zarr
    """)
    return


@app.cell
def _():
    from src.utils.read_write import get_xr_dataset

    return (get_xr_dataset,)


@app.cell
def _(get_xr_dataset):
    ds = get_xr_dataset()
    print(f"{ds.nbytes / 1_000_000_000:.2f} GB")
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell(disabled=True)
def _(ds, store):
    ds.to_zarr(store=store, consolidated=True)
    return


@app.cell
def _(store, xr):
    ds_new = xr.open_zarr(store, consolidated=True, chunks="auto")
    return (ds_new,)


@app.cell
def _(ds_new):
    for _name in ds_new.data_vars:
        print(_name, type(ds_new[_name].data), ds_new[_name].chunks)
    return


@app.cell
def _(ds_new):
    # create new variable and write directly to the zarr store!
    ds_new["wind_speed_10m"] = (
        ds_new["10m_u_component_of_wind"]**2
        + ds_new["10m_v_component_of_wind"]**2
    ) ** 0.5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## append xarray to zarr
    """)
    return


@app.cell
def _(ds_new, store):
    ds_out = ds_new.expand_dims({"m": [1]}).chunk({
        "m": 1,
        "time": 366,
        "level": 13,
        "latitude": 121,
        "longitude": 240,
    })

    for v in ds_out.variables:
        ds_out[v].encoding.pop("chunks", None)
        ds_out[v].encoding.pop("preferred_chunks", None)

    ds_out.to_zarr(
        store,
        mode="w",
        consolidated=True,
    )
    return (ds_out,)


@app.cell
def _(ds_out):
    ds_out
    return


@app.cell
def _():
    from src.utils.read_write import get_rollout_files
    ung_gui, _ = get_rollout_files("grad", "2026-05-28_11:44:34", guided_id="809a0e569e")
    ung_gui
    return (ung_gui,)


@app.cell
def _(ung_gui):
    ung_gui.dims
    return


@app.cell
def _(ung_gui):
    ung_gui.coords
    return


@app.cell
def _():
    # mode="a", append_dim="run",
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## API for rollouts
    """)
    return


@app.cell
def _():
    from src.rollout_config import RolloutConfig

    return


@app.cell
def _():
    from src.utils.read_write import get_run_config
    from src.utils.dataset_utils import get_N_timestamps
    config = get_run_config("unguided_rollout", "2026-06-04_15:31:06")
    return


@app.cell
def _(np, xr):
    from src.dimensions import SPATIAL_COORDS, LEVELS_DICT, VARIABLES_DICT
    from src.paths import ROLLOUTS

    def _create_zarr_container(rollout_id:str, rollout_type:str, M:int, N:int, t_dim:bool=False, sweep_params:dict[str,list]=None):
        m_steps = range(M)
        n_steps = range(N)
        t_steps = range(25)

        coords = {
            "m": m_steps,
            "n": n_steps,
            "level": LEVELS_DICT["level"],
            "latitude": SPATIAL_COORDS["latitude"],
            "longitude": SPATIAL_COORDS["longitude"],
        }

        coords.update(sweep_params)

        surface_dims = ["m", "n", "latitude", "longitude"]+list(sweep_params.keys())
        level_dims = ["m", "n", "level", "latitude", "longitude"]+list(sweep_params.keys())


        if t_dim:
            coords["t"] = t_steps
            surface_dims.append("t")
            level_dims.append("t")

        data_vars = {}

        for var in VARIABLES_DICT["surface"]:
            shape_ = tuple(len(coords[d]) for d in surface_dims)
            data_vars[var] = (
                surface_dims,
                np.full(shape_, np.nan, dtype=np.float32),
            )

        for var in VARIABLES_DICT["level"]:
            shape_ = tuple(len(coords[d]) for d in level_dims)
            data_vars[var] = (
                level_dims,
                np.full(shape_, np.nan, dtype=np.float32),
            )

        container_ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )

        path = ROLLOUTS / rollout_id / f"{rollout_type}.zarr"
        container_ds.to_zarr(path, mode="w")

    return (ROLLOUTS,)


@app.cell
def _():
    M, N = 2, 4
    sweep_params={"alpha": [1, 2], "w": [3,4,5]}
    return M, N, sweep_params


@app.cell
def _(M, N, create_zarr_container, sweep_params):
    create_zarr_container("2026-06-04_15:31:06", "grads", M, N, t_dim=True, sweep_params=sweep_params)
    # create_zarr_container("2026-06-04_15:31:06", "ung", False)
    return


@app.cell
def _(ROLLOUTS, xr):
    def get_rollout(rollout_id:str, rollout_type:str):
        path = ROLLOUTS / rollout_id / f"{rollout_type}.zarr"
        rollout = xr.open_zarr(path)
        return rollout

    return (get_rollout,)


@app.cell
def _(get_rollout):
    z_cont = get_rollout("2026-06-05_13:10:44", "ung")
    z_cont
    return (z_cont,)


@app.cell
def _(z_cont):
    print(f"{z_cont.nbytes / 1_000_000_000:.2f} GB") 
    return


@app.cell
def _(z_cont):
    print(z_cont.info)
    return


@app.cell
def _():
    # print(z_cont["10m_u_component_of_wind"])  
    # print(z_cont["10m_u_component_of_wind"].load())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
