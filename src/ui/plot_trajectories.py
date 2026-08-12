import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator


def plot_trajectories(
    *,
    # selected members
    guided_member: list[float] | None = None,
    unguided_member: list[float] | None = None,
    guided_unguided_member: list[float] | None = None,
    det_member: list[float] | None = None,

    # ensemble bands
    guided_ensemble: list[list[float]] | None = None,
    unguided_ensemble: list[list[float]] | None = None,
    target_ensemble: list[list[float]] | None = None,
    target_guidance_ensemble: list[list[float]] | None = None,

    # optional summaries / references
    mean_unguided_rollout: list[float] | None = None,
    mean_guided_rollout: list[float] | None = None,
    target_trajectory: list[float] | None = None,
    target_guidance_trajectory: list[float] | None = None,
    ground_truth: list[float] | None = None,
    ground_truth_label: str = "Ground truth",
    reference_trajectory: list[float] | None = None,

    # percentage axis, used if y_trajectory is not None
    delta_trajectory: list[float] | None = None,
    # per-step text markers on the percentage curve (fractions, like delta)
    cumulative_delta_trajectory: list[float] | None = None,

    # display
    show_guided_mean: bool = False,
    show_unguided_mean: bool = False,

    timestamps: list[str],
    var: str,
    m: int | None = None,
    n: int | None = None,
    dpi: int = 180,
    figsize: tuple[float, float] = (17.5, 5.5),
    title: str | None = None,
    subtitle: str | None = None,
    ylabel: str | None = None,
    ymin_left: float | None = None,
    ymax_left: float | None = None,
    # multiple delta trajectories (each aligned to the N+1 time points); plotted on
    # the right axis. Takes precedence over the single delta_trajectory above.
    delta_trajectories: list[list[float]] | None = None,
    # add value/delta hlines-to-axis on the target-guidance plan line(s). Guided
    # preview mode has no guided member, so the plan carries the annotation instead.
    annotate_target_guidance: bool = False,
):
    num_steps = len(timestamps)
    # x-axis is the forecast step index n (0..num_steps-1), not the dates
    time_values = np.arange(num_steps)
    # deterministic x-limits so the guided guide-lines can reach the y-axis (left spine);
    # left margin slightly wider so the annotations clear the first data point
    _x_pad = 0.02 * max(num_steps - 1, 1)
    _x_left = -1.6 * _x_pad

    if n is not None and not (0 <= n < num_steps):
        raise ValueError(f"n must be in [0, {num_steps - 1}], got n={n}")

    def _as_1d(values, name: str, gt0=None):
        if values is None:
            return None

        values = np.asarray(values, dtype=float)

        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {values.shape}")

        if len(values) == num_steps:
            return values

        if gt0 is not None and len(values) == num_steps - 1:
            return np.concatenate([[gt0], values])

        raise ValueError(f"{name} must have the same length as timestamps")

    def _as_step_member_array(values, name: str, gt0=None):
        if values is None:
            return None

        values = np.asarray(values, dtype=float)

        if values.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {values.shape}")

        # Ensembles are forecast-only (members x N steps, N = num_steps - 1); the
        # initial point is prepended from gt0. Check this orientation FIRST: when the
        # member count happens to equal num_steps (e.g. M=2, N=1 -> num_steps=2), the
        # bare `shape[0] == num_steps` test below would misread the member axis as the
        # step axis, orient the array wrong, and the band/selected-member check fails.
        if gt0 is not None and num_steps - 1 in values.shape:
            if values.shape[1] == num_steps - 1:
                values = values.T  # (members, steps) -> (steps, members)
            num_members = values.shape[1]
            return np.vstack([np.full((1, num_members), gt0), values])

        if values.shape[0] == num_steps:
            return values

        if values.shape[1] == num_steps:
            return values.T

        raise ValueError(
            f"{name} must have one dimension equal to num_steps={num_steps}. "
            f"Got shape {values.shape}."
        )

    # Ground truth carries the initial (n=0) step; gt0 is prepended to every
    # forecast array that is one step short (length num_steps - 1) so all
    # trajectories start from the common initial point.
    ground_truth = _as_1d(ground_truth, "ground_truth")
    gt0 = ground_truth[0] if ground_truth is not None else None

    guided_member = _as_1d(guided_member, "guided_member", gt0=gt0)
    unguided_member = _as_1d(unguided_member, "unguided_member", gt0=gt0)
    guided_unguided_member = _as_1d(guided_unguided_member, "guided_unguided_member", gt0=gt0)
    det_member = _as_1d(det_member, "det_member", gt0=gt0)

    mean_unguided_rollout = _as_1d(
        mean_unguided_rollout,
        "mean_unguided_rollout",
        gt0=gt0,
    )
    mean_guided_rollout = _as_1d(
        mean_guided_rollout,
        "mean_guided_rollout",
        gt0=gt0,
    )

    target_trajectory = _as_1d(target_trajectory, "planned_guidance", gt0=gt0)
    # target guidance: a single trajectory or a list of them (one per authored profile)
    if target_guidance_trajectory is None:
        target_guidance_trajectories = None
    else:
        _tg = target_guidance_trajectory
        _is_multi = isinstance(_tg, (list, tuple)) and len(_tg) > 0 and np.ndim(_tg[0]) >= 1
        target_guidance_trajectories = [
            _as_1d(_t, f"target_guidance_trajectory[{_j}]", gt0=gt0)
            for _j, _t in enumerate(list(_tg) if _is_multi else [_tg])
        ]
    reference_trajectory = _as_1d(reference_trajectory, "reference_trajectory", gt0=gt0)

    delta_trajectory = (
        _as_1d(delta_trajectory, "y_trajectory", gt0=gt0) * 100.0
        if delta_trajectory is not None
        else None
    )

    cumulative_delta_trajectory = (
        _as_1d(cumulative_delta_trajectory, "cumulative_delta_trajectory", gt0=gt0) * 100.0
        if cumulative_delta_trajectory is not None
        else None
    )

    # the right axis is a "%" axis (formatter appends %), and deltas are stored as
    # fractions -> scale to percent. delta_trajectories (plural) is what the notebook
    # actually passes and previously skipped this, so labels read 100x too small.
    delta_trajectories = (
        [_as_1d(_dt, "delta_trajectories", gt0=gt0) * 100.0 for _dt in delta_trajectories]
        if delta_trajectories is not None
        else None
    )

    guided_ensemble = _as_step_member_array(guided_ensemble, "guided_ensemble", gt0=gt0)
    unguided_ensemble = _as_step_member_array(unguided_ensemble, "unguided_ensemble", gt0=gt0)
    target_ensemble = _as_step_member_array(target_ensemble, "target_ensemble", gt0=gt0)
    target_guidance_ensemble = _as_step_member_array(target_guidance_ensemble, "target_guidance_ensemble", gt0=gt0)

    colors = {
        "guided": "#0072B2",
        "unguided": "#8B5A2B",
        "target": "#D55E00",
        "target_guidance": "#B7950B",
        "ground_truth": "#009E73",
        "reference_trajectory": "#E6B800",
        "y": "#7B2CBF",
        "guided_unguided": "#800080",
        "det": "#AAAAAA",
        "pct": "#FF7F0E",
        "grid_major": "#D7D7D7",
        "grid_minor": "#EAEAEA",
        "text": "#222222",
        "n_marker": "#222222",
    }

    def _member_label(prefix: str):
        return f"{prefix} member, m={m}" if m is not None else f"{prefix} member"

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

        y_values = []

        def _plot_ensemble_band(
            ensemble,
            color: str,
            label_prefix: str,
            zorder_fill: int,
            zorder_line: int,
            selected_member=None,
            show_mean: bool = False,
        ):
            num_members = ensemble.shape[1]

            # Partial-run tolerance: NaN steps (not-yet-written slices) simply leave a
            # gap in the band/lines; all-NaN steps just emit a RuntimeWarning we silence.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ens_min = np.nanmin(ensemble, axis=1)
                ens_max = np.nanmax(ensemble, axis=1)
                ens_mean = np.nanmean(ensemble, axis=1)

            ax.fill_between(
                time_values,
                ens_min,
                ens_max,
                color=color,
                alpha=0.16,
                linewidth=0,
                label=f"{label_prefix} ensemble range, M={num_members}",
                zorder=zorder_fill,
            )

            ax.plot(
                time_values,
                ens_min,
                linestyle="-",
                linewidth=0.9,
                color=color,
                alpha=0.45,
                zorder=zorder_line,
                label="_nolegend_",
            )

            ax.plot(
                time_values,
                ens_max,
                linestyle="-",
                linewidth=0.9,
                color=color,
                alpha=0.45,
                zorder=zorder_line,
                label="_nolegend_",
            )

            y_values.append(ensemble.reshape(-1))

            if show_mean:
                ax.plot(
                    time_values,
                    ens_mean,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.85,
                    label=f"{label_prefix} ensemble mean",
                    zorder=zorder_line + 2,
                )
                y_values.append(ens_mean)

        def _annotate_to_axis(traj, color, ung=None):
            # dashed horizontal guide from each point to the y-axis, labelled with the
            # value, the day delta Δ^day (from the previous step), the unguided gap
            # Δ^ung (vs the unguided member at the same step) and their cumulative
            # counterparts (Δ^day_cum = total change since n=0; Δ^ung_cum = the gap
            # integrated over steps, degree-day style). Shared by the guided member
            # (analyze mode) and the target-guidance plan (guided-preview mode).
            _ref = traj[0]  # initial (n=0) point: cumulative-delta baseline
            _ung_cum = 0.0
            for _i, (_xi, _yi) in enumerate(zip(time_values, traj)):
                if np.isnan(_yi):
                    continue
                ax.plot(
                    [_x_left, _xi],
                    [_yi, _yi],
                    linestyle="--",
                    linewidth=0.8,
                    color=color,
                    alpha=0.35,
                    zorder=1,
                    label="_nolegend_",
                )
                _prev = traj[_i - 1] if _i > 0 else np.nan
                _dtxt = "" if np.isnan(_prev) else rf", $\Delta^{{\mathrm{{day}}}}{_yi - _prev:+.2f}$"
                _cumtxt = "" if (_i == 0 or np.isnan(_ref)) else rf", $\Delta^{{\mathrm{{day}}}}_{{\mathrm{{cum}}}}{_yi - _ref:+.2f}$"
                _ungtxt = _ungcumtxt = ""
                if ung is not None and _i < len(ung) and not np.isnan(ung[_i]):
                    _gap = _yi - ung[_i]
                    _ung_cum += _gap
                    if _i > 0:
                        _ungtxt = rf", $\Delta^{{\mathrm{{ung}}}}{_gap:+.2f}$"
                        _ungcumtxt = rf", $\Delta^{{\mathrm{{ung}}}}_{{\mathrm{{cum}}}}{_ung_cum:+.2f}$"
                # label sits just right of the y-axis, at the height of the guide line
                ax.annotate(
                    rf"{_yi:.2f}{_dtxt}{_ungtxt}{_cumtxt}{_ungcumtxt}",
                    xy=(0.0, _yi),
                    xycoords=("axes fraction", "data"),
                    xytext=(3, 2),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    zorder=11,
                )

        # ------------------------------------------------------------
        # Ensemble bands
        # ------------------------------------------------------------
        if unguided_ensemble is not None:
            _plot_ensemble_band(
                ensemble=unguided_ensemble,
                color=colors["unguided"],
                label_prefix="Unguided",
                zorder_fill=1,
                zorder_line=2,
                selected_member=unguided_member,
                show_mean=show_unguided_mean,
            )

        if guided_ensemble is not None:
            _plot_ensemble_band(
                ensemble=guided_ensemble,
                color=colors["guided"],
                label_prefix="Guided",
                zorder_fill=3,
                zorder_line=4,
                selected_member=guided_member,
                show_mean=show_guided_mean,
            )

        if target_ensemble is not None:
            _plot_ensemble_band(
                ensemble=target_ensemble,
                color=colors["target"],
                label_prefix="Target",
                zorder_fill=2,
                zorder_line=3,
                selected_member=None,
                show_mean=False,
            )

        if target_guidance_ensemble is not None:
            _plot_ensemble_band(
                ensemble=target_guidance_ensemble,
                color=colors["target_guidance"],
                label_prefix="Target guidance",
                zorder_fill=2,
                zorder_line=3,
                selected_member=None,
                show_mean=False,
            )

        # ------------------------------------------------------------
        # Planned guidance
        # ------------------------------------------------------------
        planned_equals_ground_truth = (
            target_trajectory is not None
            and ground_truth is not None
            and np.allclose(target_trajectory, ground_truth, equal_nan=True)
        )

        if target_trajectory is not None:
            ax.plot(
                time_values,
                target_trajectory,
                linestyle="--" if planned_equals_ground_truth else "-",
                linewidth=2.4 if planned_equals_ground_truth else 2.0,
                color=colors["target"],
                alpha=0.98,
                label="Planned guidance",
                zorder=9 if planned_equals_ground_truth else 5,
                path_effects=(
                    [
                        pe.Stroke(
                            linewidth=4.2,
                            # foreground="white",
                            alpha=0.9,
                        ),
                        pe.Normal(),
                    ]
                    if planned_equals_ground_truth
                    else None
                ),
            )
            y_values.append(target_trajectory)

        # ------------------------------------------------------------
        # Target guidance: A_n = (1 + p_n) * baseline masked mean, one line
        # per profile (linestyle cycles to keep multiple profiles readable)
        # ------------------------------------------------------------
        if target_guidance_trajectories is not None:
            # always dashed (never solid) so the target line stays visible when it
            # overlaps the solid guided line
            _tg_styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
            for _j, _traj in enumerate(target_guidance_trajectories):
                ax.plot(
                    time_values,
                    _traj,
                    linestyle=_tg_styles[_j % len(_tg_styles)],
                    linewidth=2.0,
                    color=colors["target_guidance"],
                    alpha=0.95,
                    label="Target guidance" if len(target_guidance_trajectories) == 1
                    else f"Target guidance {_j}",
                    zorder=5,
                )
                y_values.append(_traj)
                if annotate_target_guidance:
                    # gap vs the unguided member (Δ^ung) on the plan too, matching the
                    # guided-member annotation; the target is defined as A=(1+p)·unguided
                    _annotate_to_axis(_traj, colors["target_guidance"], ung=unguided_member)

        # ------------------------------------------------------------
        # Mean rollouts
        # ------------------------------------------------------------
        if mean_unguided_rollout is not None:
            ax.plot(
                time_values,
                mean_unguided_rollout,
                linestyle="--",
                linewidth=1.5,
                color=colors["unguided"],
                alpha=0.85,
                label="Mean unguided rollout",
                zorder=5,
            )
            y_values.append(mean_unguided_rollout)

        if mean_guided_rollout is not None:
            ax.plot(
                time_values,
                mean_guided_rollout,
                linestyle="--",
                linewidth=1.6,
                color=colors["guided"],
                alpha=0.85,
                label="Mean guided rollout",
                zorder=6,
            )
            y_values.append(mean_guided_rollout)

        # ------------------------------------------------------------
        # Selected unguided member
        # ------------------------------------------------------------
        if unguided_member is not None:
            unguided_linewidth = 2.2

            ax.plot(
                time_values,
                unguided_member,
                linestyle="-",
                linewidth=unguided_linewidth,
                color=colors["unguided"],
                alpha=0.98,
                label=_member_label("Unguided"),
                zorder=8,
                path_effects=[
                    pe.Stroke(
                        linewidth=unguided_linewidth + 1,
                        # foreground="white",
                        alpha=0.95,
                    ),
                    pe.Normal(),
                ],
            )
            y_values.append(unguided_member)

        # ------------------------------------------------------------
        # Selected guided member
        # ------------------------------------------------------------
        if guided_member is not None:
            guided_linewidth = 2.8

            ax.plot(
                time_values,
                guided_member,
                linestyle="-",
                linewidth=guided_linewidth,
                color=colors["guided"],
                alpha=0.99,
                label=_member_label("Guided"),
                zorder=9,
                path_effects=[
                    pe.Stroke(
                        linewidth=guided_linewidth + 1,
                        # foreground="white",
                        alpha=0.95,
                    ),
                    pe.Normal(),
                ],
            )
            # dashed horizontal guide from each guided point to the y-axis, labelled
            # with the value and the day/unguided deltas
            _annotate_to_axis(guided_member, colors["guided"], ung=unguided_member)
            y_values.append(guided_member)

        # ------------------------------------------------------------
        # Guided-unguided: point per n + dashed branch from previous guided
        # ------------------------------------------------------------
        if guided_unguided_member is not None:
            if guided_member is not None:
                for n_idx in range(1, num_steps):
                    ug = guided_unguided_member[n_idx]
                    g_prev = guided_member[n_idx - 1]
                    if np.isnan(ug) or np.isnan(g_prev):
                        continue
                    ax.plot(
                        [time_values[n_idx - 1], time_values[n_idx]],
                        [g_prev, ug],
                        linestyle="--",
                        linewidth=1.2,
                        color=colors["guided_unguided"],
                        alpha=0.7,
                        zorder=7,
                        label="_nolegend_",
                    )
            ax.scatter(
                time_values,
                guided_unguided_member,
                s=42,
                color=colors["guided_unguided"],
                edgecolors="white",
                linewidths=0.8,
                zorder=9,
                label=_member_label("Guided-unguided"),
            )
            y_values.append(guided_unguided_member)

        # ------------------------------------------------------------
        # Deterministic: point per n + dashed branch from previous guided
        # (same grammar as guided_unguided, in grey)
        # ------------------------------------------------------------
        if det_member is not None:
            if guided_member is not None:
                for n_idx in range(1, num_steps):
                    dv = det_member[n_idx]
                    g_prev = guided_member[n_idx - 1]
                    if np.isnan(dv) or np.isnan(g_prev):
                        continue
                    ax.plot(
                        [time_values[n_idx - 1], time_values[n_idx]],
                        [g_prev, dv],
                        linestyle="--",
                        linewidth=1.2,
                        color=colors["det"],
                        alpha=0.8,
                        zorder=6,
                        label="_nolegend_",
                    )
            ax.scatter(
                time_values,
                det_member,
                s=42,
                color=colors["det"],
                edgecolors="white",
                linewidths=0.8,
                zorder=8,
                label=_member_label("Deterministic (gui_det)"),
            )
            y_values.append(det_member)

        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------
        if ground_truth is not None:
            ax.plot(
                time_values,
                ground_truth,
                linestyle="-",
                linewidth=2.2,
                color=colors["ground_truth"],
                alpha=0.70 if planned_equals_ground_truth else 0.95,
                label=ground_truth_label,
                zorder=7,
            )
            y_values.append(ground_truth)

        # ------------------------------------------------------------
        # Reference trajectory
        # ------------------------------------------------------------
        if reference_trajectory is not None:
            ax.plot(
                time_values,
                reference_trajectory,
                linestyle="--",
                linewidth=2.0,
                color=colors["reference_trajectory"],
                alpha=0.95,
                label="Reference trajectory",
                zorder=8,
            )
            y_values.append(reference_trajectory)

        # ------------------------------------------------------------
        # Axis styling  (x-axis = forecast step n; no vertical n/t markers)
        # ------------------------------------------------------------
        ax.set_xlabel("$n$")
        ax.set_ylabel(ylabel if ylabel is not None else var)

        ax.set_xticks(time_values)
        ax.set_xlim(_x_left, (num_steps - 1) + _x_pad)

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
        if y_values and np.isfinite(np.concatenate(y_values)).any():
            y_all = np.concatenate(y_values)
            y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
            y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 1.0

            left_min = ymin_left if ymin_left is not None else y_min - y_pad
            left_max = ymax_left if ymax_left is not None else y_max + y_pad

            if left_min == left_max:
                left_min -= 1.0
                left_max += 1.0

            ax.set_ylim(left_min, left_max)

        # ------------------------------------------------------------
        # Optional right axis for percentage trajectory
        # ------------------------------------------------------------
        # one or more delta trajectories on the right axis (delta_trajectories wins)
        _deltas = delta_trajectories if delta_trajectories is not None else (
            [delta_trajectory] if delta_trajectory is not None else None
        )
        has_right_axis = bool(_deltas)

        if has_right_axis:
            ax2 = ax.twinx()
            _single = len(_deltas) == 1

            for _i, _dt in enumerate(_deltas):
                ax2.plot(
                    time_values,
                    _dt,
                    linestyle="-",
                    linewidth=1.6,
                    color=colors["pct"],
                    alpha=0.95,
                    label="Percentage change" if _single else f"delta {_i}",
                    zorder=4,
                )

            ax2.axhline(
                0.0,
                color=colors["grid_major"],
                linewidth=0.8,
                alpha=0.7,
                zorder=0,
            )

            if _single and cumulative_delta_trajectory is not None:
                delta_trajectory = _deltas[0]
                for i, (t_val, d_val, c_val) in enumerate(
                    zip(time_values, delta_trajectory, cumulative_delta_trajectory)
                ):
                    # step 0 is the prepended initial point, nothing accumulated yet
                    if i == 0 or np.isnan(d_val) or np.isnan(c_val):
                        continue
                    ax2.annotate(
                        f"cum%={c_val:.2f}",
                        (t_val, d_val),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=7,
                        color=colors["y"],
                        zorder=5,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    )

            ax2.set_ylabel("")
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))

            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color("#BBBBBB")

            ax2.tick_params(
                axis="both",
                colors=colors["y"],
                length=4,
                width=0.8,
            )

            left_min, left_max = ax.get_ylim()
            _all = np.concatenate([np.asarray(_dt, dtype=float) for _dt in _deltas])
            y_min = float(np.nanmin(_all))
            y_max = float(np.nanmax(_all))

            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0

            def map_left_to_right(v):
                return y_min + (v - left_min) * (y_max - y_min) / (left_max - left_min)

            ax2.set_ylim(
                map_left_to_right(left_min),
                map_left_to_right(left_max),
            )

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()

            handles = handles1 + handles2
            labels = labels1 + labels2
        else:
            handles, labels = ax.get_legend_handles_labels()

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        unique = dict(zip(labels, handles))
        legend_x = 1.05 if has_right_axis else 1.015

        ax.legend(
            unique.values(),
            unique.keys(),
            loc="center left",
            bbox_to_anchor=(legend_x, 0.5),
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
        )

        # ------------------------------------------------------------
        # Titles
        # ------------------------------------------------------------
        if title is None:
            title = f"{var} trajectory"

        if title:
            fig.suptitle(
                title,
                x=0.06,
                y=0.95,
                ha="left",
                fontsize=15,
                fontweight="bold",
                color=colors["text"],
            )

        if subtitle:
            fig.text(
                0.06,
                0.9,
                subtitle,
                ha="left",
                va="top",
                fontsize=9.5,
                color="#555555",
            )

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------
        right_margin = 0.80 if has_right_axis else 0.84

        fig.tight_layout(
            rect=(0.0, 0.0, right_margin, 0.90 if (title or subtitle) else 1.0)
        )

    return fig