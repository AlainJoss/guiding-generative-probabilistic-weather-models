import matplotlib.pyplot as plt


def visualize_mask_terms_over_N(
    var: str,
    timestamps: list[str],
    ensemble_rollout: list[list[float]] | None = None,   # shape: [N][M]
    mean_rollout: list[float] | None = None,
    ground_truth: list[float] | None = None,
    planned_guidance: list[float] | None = None,
):
    if (
        ensemble_rollout is None
        and mean_rollout is None
        and ground_truth is None
        and planned_guidance is None
    ):
        raise ValueError("At least one rollout must be provided")

    N = len(timestamps)
    x = list(range(N))

    fig, ax = plt.subplots(figsize=(8, 4))

    if ensemble_rollout is not None:
        M = len(ensemble_rollout[0])

        for i in range(M):
            y = [ensemble_rollout[n][i] for n in range(N)]
            ax.plot(x, y, marker="o", alpha=0.8)

        lower = [min(row) for row in ensemble_rollout]
        upper = [max(row) for row in ensemble_rollout]
        ax.fill_between(x, lower, upper, alpha=0.2)

    if mean_rollout is not None:
        ax.plot(x, mean_rollout, marker="o", linewidth=2, label="Mean rollout")

    if ground_truth is not None:
        ax.plot(x, ground_truth, marker="o", linewidth=2, label="Ground truth")

    if planned_guidance is not None:
        ax.plot(x, planned_guidance, marker="o", linewidth=2, label="Planned guidance")

    tick_idx = [i for i, ts in enumerate(timestamps) if ts.endswith("00:00:00")]
    if 0 not in tick_idx:
        tick_idx = [0] + tick_idx
    if N - 1 not in tick_idx:
        tick_idx.append(N - 1)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([timestamps[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.set_title(f"Rollout distribution {var}")
    ax.grid(alpha=0.3)

    if mean_rollout is not None or ground_truth is not None or planned_guidance is not None:
        ax.legend()

    fig.tight_layout()
    return fig


def plot_ensemble_rollout(
    timestamps: list[str],
    realized_terms: list[float],
    y_perc: list[float],
    planned_guidance: list[float] | None = None,
    ground_truth: list[float] | None = None,
    mean_rollout: list[float] | None = None,
    ensemble_rollout: list[list[float]] | None = None,
    title: str | None = "Realized guidance",
    subtitle: str | None = None,
):
    N = len(timestamps)
    if N != len(realized_terms) or len(realized_terms) != len(y_perc):
        raise ValueError(
            "timestamps, realized_terms, and y_perc must have the same length"
        )
    x = list(range(N))
    branch_targets = realized_guidance_branches(realized_terms, y_perc)

    C = {
        "realized": "#1f77b4",
        "online": "#d62728",
        "offline": "#ff7f0e",
        "gt": "#2ca02c",
        "mean": "#9467bd",
        "ensemble": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    if ensemble_rollout is not None:
        M = len(ensemble_rollout[0])
        for i in range(M):
            y = [ensemble_rollout[n][i] for n in range(N)]
            ax.plot(
                x,
                y,
                "-",
                color=C["ensemble"],
                linewidth=0.6,
                alpha=0.35,
                zorder=1,
            )
        lower = [min(row) for row in ensemble_rollout]
        upper = [max(row) for row in ensemble_rollout]
        ax.fill_between(
            x,
            lower,
            upper,
            color=C["ensemble"],
            alpha=0.12,
            label=f"Ensemble range (M={M})",
            zorder=1,
        )

    for n in range(1, N):
        ax.plot(
            [x[n - 1], x[n]],
            [realized_terms[n - 1], branch_targets[n]],
            linestyle="--",
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=C["online"],
            alpha=0.85,
            zorder=2,
            label="Online planned" if n == 1 else None,
        )

    if planned_guidance is not None:
        ax.plot(
            x,
            planned_guidance,
            "-",
            marker="s",
            markersize=3.5,
            linewidth=1.6,
            color=C["offline"],
            alpha=0.9,
            label="Offline planned",
            zorder=3,
        )

    if mean_rollout is not None:
        ax.plot(
            x,
            mean_rollout,
            "-",
            marker="D",
            markersize=3.5,
            linewidth=1.6,
            color=C["mean"],
            alpha=0.9,
            label="Mean rollout",
            zorder=3,
        )

    if ground_truth is not None:
        ax.plot(
            x,
            ground_truth,
            "-",
            marker="^",
            markersize=4.5,
            linewidth=1.8,
            color=C["gt"],
            alpha=0.95,
            label="Ground truth",
            zorder=4,
        )

    ax.plot(
        x,
        realized_terms,
        "-",
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        color=C["realized"],
        label="Realized",
        zorder=5,
    )

    tick_idx = [
        i for i, ts in enumerate(timestamps) if ts.endswith("00:00:00")
    ]
    if 0 not in tick_idx:
        tick_idx = [0] + tick_idx
    if N - 1 not in tick_idx:
        tick_idx.append(N - 1)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [timestamps[i] for i in tick_idx], rotation=35, ha="right", fontsize=8
    )
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel("Mask term", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(
            0.5,
            0.955,
            subtitle,
            ha="center",
            va="top",
            fontsize=9,
            color="#555",
        )

    fig.tight_layout(
        rect=(0.0, 0.0, 0.82, 0.93 if (title or subtitle) else 1.0)
    )
    return fig
