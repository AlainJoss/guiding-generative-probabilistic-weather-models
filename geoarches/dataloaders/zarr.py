import shutil

import fasteners
import xarray as xr


class ZarrIterativeWriter:
    def __init__(self, path, force=True):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.synchronizer = fasteners.InterProcessLock(str(path.parent) + "/.lock")
        print("Creating ZarrIterativeWriter")
        with self.synchronizer:
            if path.exists() and force:
                print("path already exists")
                shutil.rmtree(path)

    def write(self, xr_dataset, append_dim="time"):
        with self.synchronizer:
            is_initialized = self.path.exists()
            args = (
                dict(append_dim=append_dim)
                if is_initialized
                else dict(encoding=dict(time=dict(units="hours since 2000-01-01")))
            )

            xr_dataset.to_zarr(self.path, **args)

    def to_netcdf(self, dump_id=0):
        with self.synchronizer:
            nc_path = self.path.parent / self.path.name.replace(".zarr", f"_{dump_id}.nc")

            if not nc_path.exists():
                ds = xr.open_zarr(self.path)
                try:
                    ds.to_netcdf(nc_path)
                finally:
                    ds.close()

            shutil.rmtree(self.path)
