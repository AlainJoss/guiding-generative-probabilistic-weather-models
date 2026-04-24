import matplotlib.pyplot as plt


def _to_float_list(xs):
    if xs is None:
        return None
    return [float(v) for v in xs]


def visualize_mask_terms_over_N(
    var: str,
    timestamps: list[str],
    ensemble_rollout: list[list[float]] | None = None,   # shape: [N][M]
    mean_rollout: list[float] | None = None,
    ground_truth: list[float] | None = None,
    planned_guidance: list[float] | None = None,
    gen_det_rollout: list[float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
):
    if (
        ensemble_rollout is None
        and mean_rollout is None
        and ground_truth is None
        and planned_guidance is None
        and gen_det_rollout is None
    ):
        raise ValueError("At least one rollout must be provided")

    N = len(timestamps)
    x = list(range(N))

    mean_rollout = _to_float_list(mean_rollout)
    ground_truth = _to_float_list(ground_truth)
    planned_guidance = _to_float_list(planned_guidance)
    gen_det_rollout = _to_float_list(gen_det_rollout)

    C = {
        "gt": "#2ca02c",
        "mean": "#9467bd",
        "ensemble": "#7f7f7f",
        "offline": "#ff7f0e",
        "det": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    if ensemble_rollout is not None:
        rows = [[float(v) for v in row] for row in ensemble_rollout]
        M = min(len(row) for row in rows)
        trimmed = [row[:M] for row in rows]

        for i in range(M):
            y = [trimmed[n][i] for n in range(N)]
            ax.plot(
                x, y, "-",
                color=C["ensemble"], linewidth=0.6, alpha=0.35, zorder=1,
            )
        lower = [min(row) for row in trimmed]
        upper = [max(row) for row in trimmed]
        ax.fill_between(
            x, lower, upper,
            color=C["ensemble"], alpha=0.12,
            label=f"Ensemble range (M={M})", zorder=1,
        )
    else:
        trimmed = None

    if gen_det_rollout is not None:
        if mean_rollout is not None:
            anchor = mean_rollout
        elif trimmed is not None:
            anchor = [sum(row) / len(row) for row in trimmed]
        elif ground_truth is not None:
            anchor = ground_truth
        else:
            anchor = gen_det_rollout

        for n in range(1, N):
            ax.plot(
                [x[n - 1], x[n]],
                [anchor[n - 1], gen_det_rollout[n]],
                linestyle="--",
                linewidth=1.2,
                color=C["det"],
                alpha=0.85,
                zorder=2,
                label="Deterministic branch" if n == 1 else None,
            )

    if planned_guidance is not None:
        ax.plot(
            x, planned_guidance, "-",
            linewidth=1.6,
            color=C["offline"], alpha=0.9,
            label="Planned guidance", zorder=3,
        )

    if mean_rollout is not None:
        ax.plot(
            x, mean_rollout, "-",
            linewidth=1.6,
            color=C["mean"], alpha=0.9,
            label="Mean rollout", zorder=3,
        )

    if ground_truth is not None:
        ax.plot(
            x, ground_truth, "-",
            linewidth=1.8,
            color=C["gt"], alpha=0.95,
            label="Ground truth", zorder=4,
        )

    tick_idx = [i for i, ts in enumerate(timestamps) if ts.endswith("00:00:00")]
    if 0 not in tick_idx:
        tick_idx = [0] + tick_idx
    if N - 1 not in tick_idx:
        tick_idx.append(N - 1)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [timestamps[i] for i in tick_idx], rotation=35, ha="right", fontsize=8
    )
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel(var, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )

    _title = title if title is not None else f"Rollout distribution {var}"
    _axes_xmid = 0.41
    fig.suptitle(_title, fontsize=13, fontweight="bold", x=_axes_xmid, y=0.995)
    if subtitle:
        fig.text(
            _axes_xmid, 0.955, subtitle,
            ha="center", va="top", fontsize=9, color="#555",
        )

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.93))
    return fig
