import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare implementations

    Proof that my implementation is up to a small numerical error equivalent to the original.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.colors as mcolors

    import geopandas as gpd
    import geodatasets


    import matplotlib.patches as mpatches
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    return (
        geodatasets,
        gpd,
        make_axes_locatable,
        mcolors,
        mo,
        mpatches,
        np,
        torch,
    )


@app.cell
def _():
    variable = "2m_temperature"
    level = 0
    return level, variable


@app.cell
def _(torch):
    path_ = "experiments/generated/{}.pt"
    diffusion_tensor = torch.load(path_.format("diffusion.DiffusionModule"), weights_only=False)
    flow_tensor = torch.load(path_.format("guided_diffusion.GuidedFlow"), weights_only=False)
    all_close = torch.allclose(diffusion_tensor, flow_tensor, rtol=1e-6, atol=1e-7)
    return all_close, diffusion_tensor, flow_tensor


@app.cell
def _(diffusion_tensor, flow_tensor):
    diff_tensor = diffusion_tensor - flow_tensor
    return (diff_tensor,)


@app.cell
def _(
    ChartPuck,
    geodatasets,
    gpd,
    level,
    make_axes_locatable,
    mcolors,
    mo,
    mpatches,
    np,
    variable,
):

    def lonlat_to_ij(lon, lat, lon_c, lat_c):
        j = int(np.argmin(np.abs(lon_c - lon)))
        i = int(np.argmin(np.abs(lat_c - lat)))
        return i, j


    def visualize_map(
        array_2d,
        cmap="coolwarm",
        vmin=None,
        vmax=None,
        center=None,
        figsize=(10, 5),
        with_rectangle=False,
        rectangle_x=(20.0, 60.0),
        rectangle_y=(50.0, 20.0),
    ):
        ny, nx = array_2d.shape
        assert (ny, nx) == (121, 240), f"Expected (121, 240), got {(ny, nx)}"

        # undo the 180° roll applied in convert_to_tensordict
        array_2d_unrolled = np.roll(array_2d, -nx // 2, axis=1)

        # original ERA5 grid in [0, 360]
        lon_e = np.linspace(0.0, 360.0, nx + 1, endpoint=True)
        lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)
        lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
        lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

        # convert longitudes to [-180, 180]
        lon_c_plot = ((lon_c + 180) % 360) - 180
        sort_idx = np.argsort(lon_c_plot)
        lon_c_plot = lon_c_plot[sort_idx]
        array_2d_plot = array_2d_unrolled[:, sort_idx]

        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        if center is not None:
            if vmin is None or vmax is None:
                absmax = np.nanmax(np.abs(array_2d_plot))
                if not np.isfinite(absmax) or absmax == 0:
                    absmax = 1e-8
                if vmin is None:
                    vmin = -absmax
                if vmax is None:
                    vmax = absmax
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def draw_base(ax):
            im = ax.pcolormesh(
                lon_c_plot,
                lat_c,
                array_2d_plot,
                cmap=cmap,
                norm=norm,
                shading="nearest",
            )

            world.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=5)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"{variable} @ level {level}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            ax.figure.colorbar(im, cax=cax)

            return im

        if not with_rectangle:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, dpi=100, figsize=figsize)
            draw_base(ax)
            plt.tight_layout()
            plt.show()
            return

        def draw_map(ax, widget):
            ax.clear()

            # remove previous colorbar axes if redraw happens
            fig = ax.figure
            while len(fig.axes) > 1:
                fig.delaxes(fig.axes[-1])

            draw_base(ax)

            x0, x1 = widget.x
            y0, y1 = widget.y

            lon_left = min(x0, x1)
            lon_right = max(x0, x1)
            lat_bottom = min(y0, y1)
            lat_top = max(y0, y1)

            try:
                widget.x = [lon_left, lon_right]
                widget.y = [lat_top, lat_bottom]
            except Exception:
                pass

            ul_i, ul_j = lonlat_to_ij(lon_left, lat_top, lon_c_plot, lat_c)
            lr_i, lr_j = lonlat_to_ij(lon_right, lat_bottom, lon_c_plot, lat_c)

            rect = mpatches.Rectangle(
                (lon_left, lat_bottom),
                lon_right - lon_left,
                lat_top - lat_bottom,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
                zorder=10,
            )
            ax.add_patch(rect)

            ax.plot(
                [lon_left, lon_right],
                [lat_top, lat_bottom],
                "o",
                color="red",
                markersize=5,
                zorder=11,
            )

        return mo.ui.anywidget(
            ChartPuck.from_callback(
                draw_fn=draw_map,
                x=list(rectangle_x),
                y=list(rectangle_y),
                puck_color=["green", "red"],
                figsize=(14.5, 8),   # or bigger
                x_bounds=(-180.0, 180.0),
                y_bounds=(-90.0, 90.0),
            )
        )

    return (visualize_map,)


@app.cell
def _():
    vmin = -3
    vmax = 3
    return vmax, vmin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Maps
    """)
    return


@app.cell
def _(all_close, mo):
    mo.md(f"""
    tensor values all close? {all_close}
    """)
    return


@app.cell
def _(diff_tensor, diffusion_tensor, flow_tensor, visualize_map, vmax, vmin):
    visualize_map(diffusion_tensor, vmin=vmin, vmax=vmax)
    visualize_map(flow_tensor, vmin=vmin, vmax=vmax)
    visualize_map(diff_tensor, vmin=vmin, vmax=vmax)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
