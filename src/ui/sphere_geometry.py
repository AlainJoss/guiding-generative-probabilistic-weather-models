import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from src.ui.map import _get_world as get_world

    return get_world, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sphere vs. plate carrée

    Our ERA5 arrays live on a regular $1.5^\circ$ lat/lon grid — a **plate carrée**
    (equirectangular) picture of the sphere. Building a Gaussian mask on it needs two
    corrections, and this notebook shows each one interactively:

    1. **Great-circle distance** — measure *where things are* on the sphere,
       not in raw degree coordinates (fixes the kernel's **values**).
    2. **$\cos(\mathrm{lat})$ area weight** — count each pixel by the true area it
       represents (fixes the discrete **sum**: the grid oversamples high latitudes).
    """)
    return


@app.cell
def _(np):
    def haversine_deg(lat, lon, mu_lat, mu_lon):
        """Great-circle distance in degrees (all inputs in degrees)."""
        _p1, _l1 = np.radians(lat), np.radians(lon)
        _p2, _l2 = np.radians(mu_lat), np.radians(mu_lon)
        _a = (
            np.sin((_p2 - _p1) / 2) ** 2
            + np.cos(_p1) * np.cos(_p2) * np.sin((_l2 - _l1) / 2) ** 2
        )
        return np.degrees(2 * np.arcsin(np.sqrt(np.clip(_a, 0.0, 1.0))))

    # the ERA5 grid (cell centers)
    lon_c = 0.5 * (np.linspace(-180, 180, 241)[:-1] + np.linspace(-180, 180, 241)[1:])
    lat_c = 0.5 * (np.linspace(90, -90, 122)[:-1] + np.linspace(90, -90, 122)[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)
    return LAT, LON, haversine_deg, lat_c, lon_c


@app.cell
def _(mo):
    center_lat = mo.ui.slider(-85, 85, step=5, value=60, label="center lat: ", show_value=True, debounce=True)
    center_lon = mo.ui.slider(-180, 180, step=5, value=15, label="center lon: ", show_value=True, debounce=True)
    sigma_deg = mo.ui.slider(5, 60, step=5, value=20, label="sigma (deg): ", show_value=True, debounce=True)
    mo.hstack([center_lat, center_lon, sigma_deg], justify="start")
    return center_lat, center_lon, sigma_deg


@app.cell
def _(LAT, LON, center_lat, center_lon, haversine_deg, np):
    # the two distance fields from the chosen center
    d_gc = haversine_deg(LAT, LON, center_lat.value, center_lon.value)
    d_naive = np.hypot(LAT - center_lat.value, LON - center_lon.value)
    return d_gc, d_naive


@app.cell
def _(get_world, lat_c, lon_c):
    def draw_panel(ax, field, title, rings=12, cmap="coolwarm"):
        ax.imshow(
            field, extent=[-180, 180, -90, 90], origin="upper",
            cmap=cmap, aspect="auto",
        )
        if rings:
            ax.contour(lon_c, lat_c, field, levels=rings,
                       colors="black", linewidths=0.5)
        get_world().boundary.plot(ax=ax, color="black", linewidth=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)

    return (draw_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Great-circle distance: *where things are*

    Below, iso-distance rings from the center — left in **naive degree coordinates**
    $\sqrt{\Delta\mathrm{lat}^2 + \Delta\mathrm{lon}^2}$, right along **great circles**.

    The inversion to internalize: the naive rings look like perfect circles *on the map*
    but are wrong *on the Earth* (no dateline wrap, east–west physically shrunk at high
    latitude); the great-circle rings look distorted on the map but are true circles on
    the globe. Drag the center poleward and watch them diverge.
    """)
    return


@app.cell
def _(d_gc, d_naive, draw_panel, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.8), dpi=100)
    draw_panel(_axes[0], d_naive, "naive lat/lon degree distance (no wrap, no sphere)")
    draw_panel(_axes[1], d_gc, "great-circle distance (wraps, true on the sphere)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same comparison for the resulting **Gaussian kernels**
    $\exp(-d^2/2\sigma^2)$: the naive bump is round on the map but egg-shaped on the
    Earth; the great-circle bump is the reverse.
    """)
    return


@app.cell
def _(d_gc, d_naive, draw_panel, np, plt, sigma_deg):
    m_naive = np.exp(-0.5 * (d_naive / sigma_deg.value) ** 2)
    m_gc = np.exp(-0.5 * (d_gc / sigma_deg.value) ** 2)

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.8), dpi=100)
    draw_panel(_axes[0], m_naive, "naive Gaussian (round on map, wrong on Earth)", rings=0)
    draw_panel(_axes[1], m_gc, "great-circle Gaussian (round on Earth)", rings=0)
    _fig.tight_layout()
    _fig
    return m_gc, m_naive


@app.cell(hide_code=True)
def _(lat_c, mo, np):
    _pts = float((np.abs(lat_c) >= 60).mean())
    _area = 1.0 - np.sin(np.radians(60))
    mo.md(
        rf"""
    ## 2. $\cos(\mathrm{{lat}})$: *how much each pixel counts*

    On the map all cells look equal; on the sphere a cell's area shrinks by
    $\cos(\mathrm{{lat}})$ toward the poles. The grid therefore **oversamples** high
    latitudes: rows poleward of $60^\circ$ hold **{_pts:.0%} of the grid points** but
    only **{_area:.0%} of the Earth's area**. An unweighted sum $\sum m \odot x$
    over-counts the poleward part of any mask — multiplying each pixel by
    $\cos(\mathrm{{lat}})$ before normalizing restores the true area integral.
    """
    )
    return


@app.cell
def _(lat_c, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.2), dpi=100)

    _axes[0].plot(lat_c, np.cos(np.radians(lat_c)), color="#325D3D")
    _axes[0].fill_between(lat_c, np.cos(np.radians(lat_c)), alpha=0.2, color="#325D3D")
    _axes[0].set_xlabel("latitude")
    _axes[0].set_title("true area represented by one grid cell  (∝ cos lat)", fontsize=10)

    _axes[1].plot(lat_c, 1 / np.cos(np.radians(lat_c)), color="#B05C3A")
    _axes[1].set_ylim(0, 12)
    _axes[1].set_xlabel("latitude")
    _axes[1].set_title("grid points per unit area  (∝ 1/cos lat): pole oversampling", fontsize=10)

    for _ax in _axes:
        _ax.tick_params(labelsize=8)
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The globe doesn't lie

    The same two kernels, painted on the actual sphere. The **great-circle** bump is a
    round cap wherever you put it; switch to the **naive** one and move the center
    poleward — it pinches into a wedge. The black dots are every 4th grid point:
    watch them bunch up toward the poles (that's the $1/\cos$ oversampling).
    """)
    return


@app.cell
def _(mo):
    globe_field = mo.ui.radio(["great-circle", "naive lat/lon"], value="great-circle", label="kernel on the globe:")
    elev_slider = mo.ui.slider(-90, 90, step=10, value=35, label="elev: ", show_value=True, debounce=True)
    azim_slider = mo.ui.slider(-180, 180, step=10, value=20, label="azim: ", show_value=True, debounce=True)
    show_points = mo.ui.checkbox(value=True, label="show grid points")
    mo.hstack([globe_field, elev_slider, azim_slider, show_points], justify="start", align="center")
    return azim_slider, elev_slider, globe_field, show_points


@app.cell
def _(
    LAT,
    LON,
    azim_slider,
    elev_slider,
    get_world,
    globe_field,
    m_gc,
    m_naive,
    np,
    plt,
    show_points,
):
    _m = m_gc if globe_field.value == "great-circle" else m_naive

    _phi, _lmb = np.radians(LAT), np.radians(LON)
    _X = np.cos(_phi) * np.cos(_lmb)
    _Y = np.cos(_phi) * np.sin(_lmb)
    _Z = np.sin(_phi)

    _fig = plt.figure(figsize=(7.5, 7.5), dpi=100)
    _ax = _fig.add_subplot(projection="3d")
    _ax.plot_surface(
        _X, _Y, _Z,
        facecolors=plt.get_cmap("coolwarm")(_m / _m.max()),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
    )

    for _geom in get_world().boundary.geometry:
        for _line in getattr(_geom, "geoms", [_geom]):
            _xs, _ys = np.asarray(_line.xy[0]), np.asarray(_line.xy[1])
            _p, _l = np.radians(_ys), np.radians(_xs)
            _ax.plot(
                np.cos(_p) * np.cos(_l) * 1.002,
                np.cos(_p) * np.sin(_l) * 1.002,
                np.sin(_p) * 1.002,
                color="black", linewidth=0.3,
            )

    if show_points.value:
        _s = (slice(None, None, 4), slice(None, None, 4))
        _ax.scatter(
            _X[_s] * 1.004, _Y[_s] * 1.004, _Z[_s] * 1.004,
            s=0.6, color="black", alpha=0.6, depthshade=False,
        )

    _ax.set_box_aspect((1, 1, 1))
    _ax.set_axis_off()
    _ax.view_init(elev=elev_slider.value, azim=azim_slider.value)
    _ax.set_title(f"{globe_field.value} kernel on the sphere", fontsize=11)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the code

    `src/mask.py :: get_elliptical_mask` applies exactly these two corrections:
    the kernel is evaluated in **great-circle** offsets (haversine $+$ bearing
    decomposition $\rightarrow$ point 1), and the weights are multiplied by
    $\cos(\mathrm{lat})$ before normalizing ($\rightarrow$ point 2), so the masked
    statistic $M(x) = \sum m \odot x$ approximates a true area average on the sphere.
    """)
    return


if __name__ == "__main__":
    app.run()
