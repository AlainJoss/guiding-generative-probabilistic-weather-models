import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_climatology(
    dates,
    series,
    rolling_days: int = 7,
    var: str | None = None,
    unit: str | None = None,
    title: str | None = "Climatology",
    subtitle: str | None = None,
    figsize: tuple[float, float] = (22.0, 6.0),
    dpi: int = 180,
    label: str | None = None,
):
    """Mask-averaged annual climatology with a rolling-average overlay.

    Plots the daily `series` (thin) plus a centered `rolling_days`-day rolling mean
    (bold), labels the x-axis by month, and marks with a dashed vertical line the START
    of the `rolling_days`-day window with the highest rolling average (the hottest
    period). `dates` is one datetime per point; `series` is already in display units.
    Styling mirrors `plot_trajectory` (bold left-aligned title, muted grid, purple line).
    """
    colors = {
        "raw": "#C9A7E8",        # light purple: raw daily line
        "roll": "#7B2CBF",       # bold purple: rolling average
        "vline": "#E03131",      # red: hottest-window start
        "grid_major": "#D7D7D7",
        "text": "#222222",
    }

    dates = pd.to_datetime(np.asarray(dates))
    y = np.asarray(series, dtype=float)
    W = int(np.clip(rolling_days, 1, len(y)))

    # centered rolling mean for the smooth line
    roll = pd.Series(y).rolling(W, center=True, min_periods=1).mean().to_numpy()

    # hottest window: the W-day window (indexed by its START) with the highest mean.
    # cumulative-sum trick gives mean(series[i:i+W]) for every valid start i.
    if len(y) >= W:
        _c = np.concatenate([[0.0], np.cumsum(y)])
        win_mean = (_c[W:] - _c[:-W]) / W            # len = N-W+1, indexed by start i
        i0 = int(np.argmax(win_mean))
    else:
        i0 = int(np.argmax(y))
    start_date = dates[i0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(dates, y, color=colors["raw"], linewidth=1.0, alpha=0.9, zorder=2,
            label=(label or var or "daily"))
    ax.plot(dates, roll, color=colors["roll"], linewidth=2.2, alpha=0.95, zorder=4,
            label=f"{W}-day rolling avg")

    # hottest-window start marker
    ax.axvline(start_date, color=colors["vline"], linewidth=1.6, linestyle="--",
               alpha=0.9, zorder=5)
    ax.annotate(
        f"peak {W}d start\n{start_date.strftime('%b %d')}",
        xy=(start_date, float(np.nanmax(roll))), xytext=(6, -2),
        textcoords="offset points", ha="left", va="top",
        fontsize=9, color=colors["vline"], zorder=6,
    )

    # month-only x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylabel(f"{var} [{unit}]" if (var and unit) else (var or ""))

    # grid / spines (match plot_trajectory)
    ax.grid(False)
    ax.yaxis.grid(True, which="major", color=colors["grid_major"], linewidth=0.75, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", colors=colors["text"], length=4, width=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    if title:
        fig.suptitle(title, x=0.06, y=0.98, ha="left", fontsize=15,
                     fontweight="bold", color="#222222")
    if subtitle:
        fig.text(0.06, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color="#555555")
    fig.subplots_adjust(top=0.88, left=0.06, right=0.98, bottom=0.12)
    return fig
