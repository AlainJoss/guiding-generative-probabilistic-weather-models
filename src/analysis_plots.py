import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_zoom_values(
    array_2d,
    *,
    zoom,
    center_lon=0.0,
    center_lat=0.0,
):
    array_2d = np.asarray(array_2d)
    ny, nx = array_2d.shape

    lon_e = np.linspace(-180.0, 180.0, nx + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)

    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    zoom = max(1, int(zoom))

    full_lon_span = 360.0
    full_lat_span = 180.0

    lon_span = full_lon_span / zoom
    lat_span = full_lat_span / zoom

    lon_min = max(-180.0, center_lon - lon_span / 2)
    lon_max = min(180.0, center_lon + lon_span / 2)
    lat_min = max(-90.0, center_lat - lat_span / 2)
    lat_max = min(90.0, center_lat + lat_span / 2)

    lon_mask = (lon_c >= lon_min) & (lon_c <= lon_max)
    lat_mask = (lat_c >= lat_min) & (lat_c <= lat_max)

    zoom_values = array_2d[np.ix_(lat_mask, lon_mask)]
    return zoom_values[np.isfinite(zoom_values)]
