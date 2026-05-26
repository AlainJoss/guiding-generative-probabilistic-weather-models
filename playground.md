def plot_dual_trajectory(
    # selected members
    guided_member: list[float] | None = None,
    unguided_member: list[float] | None = None,

    # ensemble bands
    guided_ensemble: list[list[float]] | None = None,
    unguided_ensemble: list[list[float]] | None = None,

    # optional summaries / references
    mean_unguided_rollout: list[float] | None = None,
    mean_guided_rollout: list[float] | None = None,
    planned_guidance: list[float] | None = None,
    ground_truth: list[float] | None = None,
    reference_trajectory: list[float] | None = None,

    # percentage axis (use if y_trajectory is not None)
    y_trajectory: list[float] | None = None,

    # display
    show_guided_mean: bool = False,
    show_unguided_mean: bool = False,
    # show_delta_annotation: bool = True -> don't use this, just do it inside without the need of this flag

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
):