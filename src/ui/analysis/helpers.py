import pandas as pd
import numpy as np

def format_time_value(t) -> str:
    return pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")


def get_time_values(xr_ds) -> list:
    return list(xr_ds.time.values)


def clean_coord_value(value):
    if isinstance(value, np.generic):
        return value.item()

    return value


def get_member_values(xr_ds) -> list:
    if xr_ds is None:
        return [None]

    if "member" in xr_ds.dims:
        return [clean_coord_value(val) for val in xr_ds.member.values]

    return [None]


def member_label(member) -> str:
    if member is None:
        return "single"

    return str(member)


def as_numpy_2d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def select_xr_field(xr_ds, var, timestamp, level=None, member=None):
    da = xr_ds[var].sel(time=timestamp)

    if member is not None and "member" in da.dims:
        da = da.sel(member=member)

    if "level" in da.dims and level is not None:
        da = da.sel(level=int(level))

    return da


def xr_field_to_array(xr_ds, var, timestamp, level=None, member=None):
    da = select_xr_field(
        xr_ds=xr_ds,
        var=var,
        timestamp=timestamp,
        level=level,
        member=member,
    )

    return np.asarray(da.values)


def masked_mean_xr(xr_ds, var, timestamp, mask, level=None, member=None):
    arr = xr_field_to_array(
        xr_ds=xr_ds,
        var=var,
        timestamp=timestamp,
        level=level,
        member=member,
    )

    mask_np = as_numpy_2d(mask).astype(bool)
    vals = arr[mask_np]

    if vals.size == 0:
        return float("nan")

    return float(vals.mean())


def build_mask_rollouts(
    ground_truth_xr,
    unguided_xr,
    guided_xr,
    var,
    level,
    mask,
):
    """
    Builds trajectories with the phase-2 convention:

    ground_truth:
        length N + 1

    unguided_rollout:
        length N + 1, each entry is a list over members

    guided_rollout:
        length N + 1, each entry is a list over members

    The first entry is the initial ground-truth mask average, repeated over
    members, because guided.nc and unguided.nc normally start at the first
    future forecast time.
    """
    gt_times = get_time_values(ground_truth_xr)
    unguided_times = get_time_values(unguided_xr)
    guided_times = get_time_values(guided_xr)

    unguided_members = get_member_values(unguided_xr)
    guided_members = get_member_values(guided_xr)

    init_time = gt_times[0]

    init_avg = masked_mean_xr(
        ground_truth_xr,
        var=var,
        timestamp=init_time,
        mask=mask,
        level=level,
    )

    ground_truth_terms = [init_avg]
    unguided_rollout_terms = [[init_avg] * len(unguided_members)]
    guided_rollout_terms = [[init_avg] * len(guided_members)]

    N_eff = min(
        len(unguided_times),
        len(guided_times),
        len(gt_times) - 1,
    )

    for idx in range(N_eff):
        gt_time = gt_times[idx + 1]
        unguided_time = unguided_times[idx]
        guided_time = guided_times[idx]

        gt_avg = masked_mean_xr(
            ground_truth_xr,
            var=var,
            timestamp=gt_time,
            mask=mask,
            level=level,
        )

        ground_truth_terms.append(gt_avg)

        unguided_member_terms = [
            masked_mean_xr(
                unguided_xr,
                var=var,
                timestamp=unguided_time,
                mask=mask,
                level=level,
                member=member,
            )
            for member in unguided_members
        ]

        guided_member_terms = [
            masked_mean_xr(
                guided_xr,
                var=var,
                timestamp=guided_time,
                mask=mask,
                level=level,
                member=member,
            )
            for member in guided_members
        ]

        unguided_rollout_terms.append(unguided_member_terms)
        guided_rollout_terms.append(guided_member_terms)

    timestamps = [format_time_value(t) for t in gt_times[: len(ground_truth_terms)]]

    return (
        timestamps,
        ground_truth_terms,
        unguided_rollout_terms,
        guided_rollout_terms,
    )


def mean_rollout_terms(rollout_terms):
    return [float(np.mean(row)) for row in rollout_terms]


def member_rollout_terms(rollout_terms, member_index):
    return [float(row[member_index]) for row in rollout_terms]