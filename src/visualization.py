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