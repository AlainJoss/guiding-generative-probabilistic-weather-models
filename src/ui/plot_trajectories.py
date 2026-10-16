
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_rollout_thesis(
    *,
    # {name: {"det": arr, "det_res": arr, "band_lo": arr|None, "band_hi": arr|None,
    #         "color": str, "label": str}}  -- all ABSOLUTE (display units, e.g. deg C).
    # `det` = M_pi(x_det) deterministic core (member-independent); `det_res` =
    # M_pi(x_det + res) full state for the selected member m; `band_lo/hi` = the
    # ensemble min/max envelope over members. Series are forecast-length N or N+1
    # (already carrying the n=0 initial condition).
    ensembles: dict,
    target: list[float] | None = None,        # absolute target trajectory y*_n (member m)
    target_lo: list[float] | None = None,     # target ensemble min over members (distribution)
    target_hi: list[float] | None = None,     # target ensemble max over members (distribution)
    target_label: str = r"$y_{n,m}^\star$",
    ground_truth: list[float] | None = None,  # absolute ground truth
    ground_truth_label: str = "ground truth",
    # value at n=0 prepended to forecast-length (N) series so every trajectory
    # starts from the shared initial condition; defaults to ground_truth[0]
    n0: float | None = None,
    # last guided weather step H (1-based); draws a faint marker + shades [0, H]
    H: int | None = None,
    show_bands: bool = True,                   # per-ensemble min-max band
    m: int | None = None,                      # selected member (labels only)
    ylabel: str = r"mask-averaged temperature [$^\circ$C]",
    title: str | None = None,
    subtitle: str | None = None,
    figsize: tuple[float, float] = (11.0, 5.0),
    dpi: int = 180,
    # thin dashed vertical connectors det -> det+res at each n (the residual kick)
    connectors: bool = True,
):
    """Thesis-ready rollout-trajectories chart (absolute mask-averaged values).

    For each ensemble a SOLID line follows the selected member's full state
    (det+res, x_{n,m}); hollow markers show the deterministic core (det,
    \\hat{x}^{det}_n); and a dotted arrow leads from the core into the full state,
    i.e. the stochastic-residual contribution. An optional shaded band spans the
    ensemble min-max. Overlays: the absolute member target y*_{n,m}, ground truth,
    and a marker at the last guided step n=H. The per-step Delta annotations, the
    % twin axis and the scatter branches are removed.
    """
    # infer the forecast horizon N from the first series; normalise every series to
    # the common n-axis 0..N (prepend the shared initial condition n0 at n=0)
    _first = next(iter(ensembles.values()))
    N = len(np.asarray(_first["det_res"], dtype=float))
    num_steps = N + 1
    x = np.arange(num_steps)

    _gt_arr = np.asarray(ground_truth, dtype=float) if ground_truth is not None else None
    if n0 is None and _gt_arr is not None and len(_gt_arr) == num_steps:
        n0 = float(_gt_arr[0])

    def _prep(a):
        if a is None:
            return None
        a = np.asarray(a, dtype=float)
        if len(a) == num_steps:
            return a
        if len(a) == N:                       # forecast-only -> prepend the n=0 value
            _pv = n0 if n0 is not None else float(a[0])
            return np.concatenate([[_pv], a])
        raise ValueError(f"series length {len(a)} != N({N}) or N+1({num_steps})")

    colors = {
        "target": "#D55E00",
        "ground_truth": "#009E73",
        "grid_major": "#D7D7D7",
        "text": "#222222",
        "guide": "#9AA0A6",
    }

    y_values = []
    _any_band = False

    with plt.rc_context(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # guided-region shading + last-guided-step marker (drawn first, behind data)
        if H is not None and H > 0:
            ax.axvspan(0, H, color=colors["guide"], alpha=0.06, zorder=0)
            ax.axvline(H, color=colors["guide"], linewidth=1.0, linestyle=(0, (4, 3)),
                       alpha=0.8, zorder=1)
            ax.annotate(f"$n=H={H}$", (H, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(-3, -6), textcoords="offset points", ha="right", va="top",
                        fontsize=9, color=colors["guide"])

        # per-ensemble: optional band + SOLID full-state (det+res) line + deterministic-
        # core markers + a dotted arrow leading from the det core into the full state
        for _name, _spec in ensembles.items():
            _c = _spec["color"]
            _det = _prep(_spec.get("det"))
            _dr = _prep(_spec.get("det_res"))
            _lo = _prep(_spec.get("band_lo"))
            _hi = _prep(_spec.get("band_hi"))

            if show_bands and _lo is not None and _hi is not None:
                ax.fill_between(x, _lo, _hi, color=_c, alpha=0.12, linewidth=0, zorder=1)
                y_values.append(_lo)
                y_values.append(_hi)
                _any_band = True

            # dotted connector from the deterministic core into the full state (residual)
            if connectors and _det is not None and _dr is not None:
                for _xi in range(1, num_steps):
                    _y0, _y1 = float(_det[_xi]), float(_dr[_xi])
                    if not (np.isfinite(_y0) and np.isfinite(_y1)):
                        continue
                    ax.plot([_xi, _xi], [_y0, _y1], color=_c, linestyle=":",
                            linewidth=1.2, alpha=0.7, zorder=3)

            # full state x = det + res: SOLID line (primary) + endpoint marker
            if _dr is not None:
                ax.plot(x, _dr, color=_c, linestyle="-", linewidth=2.2, alpha=0.95,
                        solid_capstyle="round", zorder=4)
                ax.scatter(x[-1:], _dr[-1:], s=34, color=_c, edgecolors="white",
                           linewidths=0.8, zorder=6)
                y_values.append(_dr)
            # deterministic core: hollow markers at each n
            if _det is not None:
                ax.scatter(x, _det, s=26, facecolors="white", edgecolors=_c,
                           linewidths=1.3, zorder=5)
                y_values.append(_det)

        # absolute target (line sits below the realized ensemble lines so a gui
        # trajectory that lands on target stays visible; diamonds stay on top). The
        # target is member-specific, so its ensemble spread is drawn as a band.
        _tgt = _prep(target)
        _tlo = _prep(target_lo)
        _thi = _prep(target_hi)
        if _tgt is not None:
            if show_bands and _tlo is not None and _thi is not None:
                ax.fill_between(x, _tlo, _thi, color=colors["target"], alpha=0.13,
                                linewidth=0, zorder=1.2)
                y_values.append(_tlo)
                y_values.append(_thi)
                _any_band = True
            ax.plot(x, _tgt, color=colors["target"], linestyle=(0, (6, 3)), linewidth=2.0,
                    alpha=0.95, zorder=2.5)
            ax.scatter(x[1:], _tgt[1:], s=26, color=colors["target"], marker="D",
                       edgecolors="white", linewidths=0.6, zorder=7)
            y_values.append(_tgt)

        # ground truth
        _gt = _prep(ground_truth)
        if _gt is not None:
            ax.plot(x, _gt, color=colors["ground_truth"], linestyle="-", linewidth=1.6,
                    alpha=0.9, zorder=3)
            y_values.append(_gt)

        # ------------------------------------------------------------ axis chrome
        ax.set_xlabel("$n$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xlim(-0.15, N + 0.15)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(False)
        ax.yaxis.grid(True, which="major", color=colors["grid_major"], linewidth=0.7,
                      linestyle="-", alpha=0.45)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for _s in ("top", "right"):
            ax.spines[_s].set_visible(False)
        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.tick_params(axis="both", colors=colors["text"], length=4, width=0.8)

        if y_values:
            y_all = np.concatenate(y_values)
            if np.isfinite(y_all).any():
                y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
                pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.5
                ax.set_ylim(y_min - pad, y_max + pad)

        # ------------------------------------------------------------ legend
        # colour = ensemble (full-state trajectory); marker = deterministic core;
        # dotted arrow = residual; plus band / target / ground truth
        handles = [Line2D([0], [0], color=_s["color"], linewidth=2.4,
                          label=_s.get("label", _n))
                   for _n, _s in ensembles.items()]
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                              markerfacecolor="white", markeredgecolor=colors["text"],
                              markeredgewidth=1.3,
                              label=r"$\hat{x}_{n}^{\mathrm{det}}$"))
        if _any_band:
            handles.append(Patch(facecolor=colors["text"], alpha=0.18,
                                 label="ensemble range (min–max)"))
        if _tgt is not None:
            handles.append(Line2D([0], [0], color=colors["target"], linewidth=2.0,
                                  linestyle=(0, (6, 3)), marker="D", markersize=5,
                                  markeredgecolor="white", label=target_label))
        if _gt is not None:
            handles.append(Line2D([0], [0], color=colors["ground_truth"], linewidth=1.8,
                                  label=ground_truth_label))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.015, 0.5),
                  frameon=False, handlelength=2.6, borderaxespad=0.0)

        # ------------------------------------------------------------ titles
        if title:
            fig.suptitle(title, x=0.06, y=0.96, ha="left", fontsize=15, fontweight="bold",
                         color=colors["text"])
        if subtitle:
            fig.text(0.06, 0.905, subtitle, ha="left", va="top", fontsize=9.5, color="#555555")

        fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.90 if (title or subtitle) else 1.0))

    return fig
