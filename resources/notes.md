# Notes

- the data is already normalized. we go back to physical units only for visualization and interaction matters
- the data is shifted (not centered around lat,lon=0,0) 
![Data shift visualization](figures/data-shift.png)
- the testset (2020) starts in 2020-01-01T00:00:00 goes until 2020-12-31T18:00:00.00, thus has all 366*4=1464 timestamps. when initializing the Era5Forecast dataset object following gets printed "start time 2019-12-31T18:00:00". this must be wrong, since the timestep of the tensordict is actually '2020-01-01T06:00:00', such that "prev" must be at 00:00:00. The last tensordict has timestamp '2020-12-31T12:00:00', such that "next" is the last available state of 2020.
    ```
    t = ds[0]["timestamp"]
    timestamp_tensor_to_iso_t(t)
    # '2020-01-01T06:00:00'
    t = ds[len(ds)-1]["timestamp"]
    timestamp_tensor_to_iso_t(t)
    '2020-12-31T12:00:00'
    ```
- removed dask chunking from conversion to xarrays in era5 dataloader: 
```
# xr_dataset = xr_dataset.chunk(time=1)
```
