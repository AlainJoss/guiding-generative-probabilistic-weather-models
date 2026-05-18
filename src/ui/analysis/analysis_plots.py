import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator

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


def plot_guidance_tracking(
    timestamps: list[str],
    m: int,
    n: int,
    guided_member: list[float],
    unguided_member: list[float],
    target_schedule: list[float] | None = None,
    reference: list[float] | None = None,
    unguided_ensemble: list[list[float]] | None = None,
    guided_ensemble: list[list[float]] | None = None,
    show_unguided_mean: bool = False,
    show_guided_mean: bool = False,
    title: str | None = "Guidance tracking",
    subtitle: str | None = None,
    ylabel: str = "Mask-averaged value",
    figsize: tuple[int, int] = (12, 5),
    dpi: int = 100,
):
    num_steps = len(timestamps)
    time_values = pd.to_datetime(timestamps)

    if not (0 <= n < num_steps):
        raise ValueError(f"n must be in [0, {num_steps - 1}], got n={n}")

    guided_member = np.asarray(guided_member, dtype=float)
    unguided_member = np.asarray(unguided_member, dtype=float)

    if len(guided_member) != num_steps:
        raise ValueError("guided_member must have the same length as timestamps")

    if len(unguided_member) != num_steps:
        raise ValueError("unguided_member must have the same length as timestamps")

    if target_schedule is not None:
        target_schedule = np.asarray(target_schedule, dtype=float)
        if len(target_schedule) != num_steps:
            raise ValueError("target_schedule must have the same length as timestamps")

    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if len(reference) != num_steps:
            raise ValueError("reference must have the same length as timestamps")

    colors = {
        "guided": "#0072B2",
        "unguided": "#6E6E6E",
        "target": "#D55E00",
        # "target": "#E69F00",  # older planned-guidance color
        "reference": "#009E73",
        "grid_major": "#D7D7D7",
        "grid_minor": "#EAEAEA",
        "text": "#222222",
        "n_marker": "#222222",
    }

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        y_values = [
            guided_member,
            unguided_member,
        ]

        # ------------------------------------------------------------
        # Unguided ensemble shadow
        # Shape: (num_steps, num_members)
        # ------------------------------------------------------------
        if unguided_ensemble is not None:
            unguided_ensemble = np.asarray(unguided_ensemble, dtype=float)

            if unguided_ensemble.ndim != 2:
                raise ValueError(
                    f"unguided_ensemble must be 2D, got shape {unguided_ensemble.shape}"
                )

            if unguided_ensemble.shape[0] != num_steps:
                raise ValueError(
                    "unguided_ensemble must have shape (num_steps, num_members). "
                    f"Got shape {unguided_ensemble.shape}, num_steps={num_steps}."
                )

            num_unguided_members = unguided_ensemble.shape[1]

            unguided_min = np.nanmin(unguided_ensemble, axis=1)
            unguided_max = np.nanmax(unguided_ensemble, axis=1)
            unguided_mean = np.nanmean(unguided_ensemble, axis=1)

            if not np.all(
                (unguided_member >= unguided_min - 1e-10)
                & (unguided_member <= unguided_max + 1e-10)
            ):
                raise ValueError(
                    "unguided_member is outside the unguided ensemble range. "
                    "Check that it comes from the same ensemble and timestamp alignment."
                )

            ax.fill_between(
                time_values,
                unguided_min,
                unguided_max,
                color=colors["unguided"],
                alpha=0.16,
                linewidth=0,
                label=f"Unguided ensemble range, M={num_unguided_members}",
                zorder=1,
            )

            ax.plot(
                time_values,
                unguided_min,
                linestyle="-",
                linewidth=0.9,
                color=colors["unguided"],
                alpha=0.45,
                zorder=2,
                label="_nolegend_",
            )
            ax.plot(
                time_values,
                unguided_max,
                linestyle="-",
                linewidth=0.9,
                color=colors["unguided"],
                alpha=0.45,
                zorder=2,
                label="_nolegend_",
            )

            y_values.append(unguided_ensemble.reshape(-1))

            if show_unguided_mean:
                ax.plot(
                    time_values,
                    unguided_mean,
                    linestyle="--",
                    linewidth=1.5,
                    color=colors["unguided"],
                    alpha=0.85,
                    label="Unguided ensemble mean",
                    zorder=4,
                )

        # ------------------------------------------------------------
        # Guided ensemble shadow
        # Shape: (num_steps, num_members)
        # ------------------------------------------------------------
        if guided_ensemble is not None:
            guided_ensemble = np.asarray(guided_ensemble, dtype=float)

            if guided_ensemble.ndim != 2:
                raise ValueError(
                    f"guided_ensemble must be 2D, got shape {guided_ensemble.shape}"
                )

            if guided_ensemble.shape[0] != num_steps:
                raise ValueError(
                    "guided_ensemble must have shape (num_steps, num_members). "
                    f"Got shape {guided_ensemble.shape}, num_steps={num_steps}."
                )

            num_guided_members = guided_ensemble.shape[1]

            guided_min = np.nanmin(guided_ensemble, axis=1)
            guided_max = np.nanmax(guided_ensemble, axis=1)
            guided_mean = np.nanmean(guided_ensemble, axis=1)

            if not np.all(
                (guided_member >= guided_min - 1e-10)
                & (guided_member <= guided_max + 1e-10)
            ):
                raise ValueError(
                    "guided_member is outside the guided ensemble range. "
                    "Check that it comes from the same ensemble and timestamp alignment."
                )

            ax.fill_between(
                time_values,
                guided_min,
                guided_max,
                color=colors["guided"],
                alpha=0.18,
                linewidth=0,
                label=f"Guided ensemble range, M={num_guided_members}",
                zorder=3,
            )

            ax.plot(
                time_values,
                guided_min,
                linestyle="-",
                linewidth=0.9,
                color=colors["guided"],
                alpha=0.45,
                zorder=4,
                label="_nolegend_",
            )
            ax.plot(
                time_values,
                guided_max,
                linestyle="-",
                linewidth=0.9,
                color=colors["guided"],
                alpha=0.45,
                zorder=4,
                label="_nolegend_",
            )

            y_values.append(guided_ensemble.reshape(-1))

            if show_guided_mean:
                ax.plot(
                    time_values,
                    guided_mean,
                    linestyle="--",
                    linewidth=1.6,
                    color=colors["guided"],
                    alpha=0.85,
                    label="Guided ensemble mean",
                    zorder=5,
                )

        # ------------------------------------------------------------
        # Target schedule
        # ------------------------------------------------------------
        if target_schedule is not None:
            ax.plot(
                time_values,
                target_schedule,
                linestyle="-",
                linewidth=2.0,
                color=colors["target"],
                alpha=0.95,
                label="Planned guidance",
                zorder=6,
            )
            y_values.append(target_schedule)

        # ------------------------------------------------------------
        # Reference / ground truth
        # ------------------------------------------------------------
        if reference is not None:
            ax.plot(
                time_values,
                reference,
                linestyle="-",
                linewidth=2.0,
                color=colors["reference"],
                alpha=0.95,
                label="Ground truth",
                zorder=7,
            )
            y_values.append(reference)

        # ------------------------------------------------------------
        # Selected unguided member
        # ------------------------------------------------------------
        unguided_linewidth = 2.2
        ax.plot(
            time_values,
            unguided_member,
            linestyle="-",
            linewidth=unguided_linewidth,
            color=colors["unguided"],
            alpha=0.98,
            label=f"Unguided member, m={m}",
            zorder=8,
            path_effects=[
                pe.Stroke(
                    linewidth=unguided_linewidth + 2,
                    # foreground="white",
                    alpha=0.95,
                ),
                pe.Normal(),
            ],
        )

        # ------------------------------------------------------------
        # Selected guided member
        # ------------------------------------------------------------
        guided_linewidth = 2.8
        ax.plot(
            time_values,
            guided_member,
            linestyle="-",
            linewidth=guided_linewidth,
            color=colors["guided"],
            alpha=0.99,
            label=f"Guided member, m={m}",
            zorder=9,
            path_effects=[
                pe.Stroke(
                    linewidth=guided_linewidth + 2,
                    # foreground="white",
                    alpha=0.95,
                ),
                pe.Normal(),
            ],
        )

        # ------------------------------------------------------------
        # Vertical grid lines at every forecast step except current n
        # ------------------------------------------------------------
        for step_idx, step_time in enumerate(time_values):
            if step_idx == n:
                continue

            ax.axvline(
                step_time,
                color=colors["grid_major"],
                linestyle="-",
                linewidth=0.65,
                alpha=0.28,
                label="_nolegend_",
                zorder=0,
            )

        # ------------------------------------------------------------
        # Guidance step marker
        # ------------------------------------------------------------
        ax.axvline(
            time_values[n],
            color=colors["n_marker"],
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            alpha=0.75,
            label="_nolegend_",
            zorder=10,
        )

        member_diff = guided_member - unguided_member
        ax.annotate(
            f"n={n}\nΔ={member_diff[n]:.3f}",
            xy=(time_values[n], 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color=colors["n_marker"],
            alpha=0.85,
            zorder=12,
        )
        # ------------------------------------------------------------
        # Axis styling
        # ------------------------------------------------------------
        ax.set_xlabel("Forecast time")
        ax.set_ylabel(ylabel)

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(False)

        ax.yaxis.grid(
            True,
            which="major",
            color=colors["grid_major"],
            linewidth=0.75,
            linestyle="-",
            alpha=0.55,
        )

        ax.yaxis.grid(
            True,
            which="minor",
            color=colors["grid_minor"],
            linewidth=0.55,
            linestyle="-",
            alpha=0.45,
        )

        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")

        ax.tick_params(
            axis="both",
            colors=colors["text"],
            length=4,
            width=0.8,
        )

        # ------------------------------------------------------------
        # Y-limits with padding
        # ------------------------------------------------------------
        y_all = np.concatenate(y_values)
        y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
        y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 1.0
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        ax.legend(
            unique.values(),
            unique.keys(),
            loc="center left",
            bbox_to_anchor=(1.015, 0.5),
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
        )

        # ------------------------------------------------------------
        # Titles
        # ------------------------------------------------------------
        if title:
            fig.suptitle(
                title,
                x=0.06,
                y=0.98,
                ha="left",
                fontsize=15,
                fontweight="bold",
                color=colors["text"],
            )

        if subtitle:
            fig.text(
                0.06,
                0.925,
                subtitle,
                ha="left",
                va="top",
                fontsize=9.5,
                color="#555555",
            )

        fig.tight_layout(
            rect=(0.0, 0.0, 0.84, 0.90 if (title or subtitle) else 1.0)
        )

    return fig