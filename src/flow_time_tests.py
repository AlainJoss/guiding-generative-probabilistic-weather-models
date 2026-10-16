"""Pure metric functions for the flow-time tests (docs/report/chapters/5-evaluation-framework.tex):

    T1 Reliability -- the gap to the guidance target is driven to zero; mean/std of the FINAL gap.
    T2 Stability   -- the trajectory is free of oscillation; pushback + overshoot/oscillation.
    T3 Closeness   -- guided vs unguided (gui_ung) deviation; total guidance, PCA pathlength,
                      and per-step cosine vs the gui_ung twin.

No marimo. Inputs are numpy arrays produced by the loaders in ``src/ui/comparison.py`` and
the reduction cubes in ``src/reductions.py``. A field masked mean here is ``S(x) = sum(mask*x)``
with the sum-1 normalized mask from ``get_mask_2d`` (identical to ``get_masked_mean``).

t = flow step (Euler steps within one ``sample()`` call).
"""
import numpy as np


# --- T1 -------------------------------------------------------------------------------
def target_A(base, p_n):
    """Absolute guidance target ``A = (1 + p_n) * base`` (masked-mean baseline)."""
    return (1.0 + np.asarray(p_n, dtype=float)) * np.asarray(base, dtype=float)


# --- T3 -------------------------------------------------------------------------------


def step_sums(base, kick):
    """T3 total guidance-step and pushback-step lengths (per-pixel RMS over F), summed over
    flow steps:

        guidance step_t = ||kick_t - base_t||        (base_t -> kick_t: the applied kick)
        pushback step_t = ||base_{t+1} - kick_t||    (kick_t -> base_{t+1}: the flow pull-back)

    ``base`` (gui_ung_over_t) and ``kick`` (guided) are the (T, F) bbox-latent branch pair from
    the flow-time notebook. Returns ``(guidance_sum, pushback_sum)``. Applicable to
    any schedule (the sum of the actual per-step moves inside the guided flow)."""
    base = np.asarray(base, dtype=float)
    kick = np.asarray(kick, dtype=float)
    f = np.sqrt(base.shape[1])
    guidance = np.linalg.norm(kick - base, axis=1) / f
    pushback = np.linalg.norm(base[1:] - kick[:-1], axis=1) / f
    return float(guidance.sum()), float(pushback.sum())


# --- Interference (measured on the guidance-convergence plot, average mask space) -----
def interference_from_convergence(states, land_ung):
    """All interference metrics read off the guidance-convergence plot for the TARGET
    variable, in average (masked-mean) space. ``states`` (T+1,) = masked mean - target
    (converging to 0, anchored at noise); ``land_ung`` (T,) = the raw-flow Euler landing.

        flow_move_t = land_ung_t - state_t     (the flow's own move; red candle)
        gui_move_t  = state_{t+1} - land_ung_t  (the guidance move; blue candle)

    Returns a dict:
      total_guidance   = sum_t |gui_move_t|                      (total guidance applied)
      pushback_count   = # steps whose flow_move pushes AWAY from target (same sign as state)
      pushback_amount  = sum of |flow_move_t| over those steps
      overshoot_count  = # steps where the state is PAST the target (opposite sign to the
                         initial gap state_0)
      overshoot_amount = total |state_t| past the target over those steps
    """
    s = np.asarray(states, dtype=float)
    lu = np.asarray(land_ung, dtype=float)
    T = lu.shape[0]
    flow = lu - s[:T]
    gui = s[1:] - lu
    total_guid = float(np.nansum(np.abs(gui)))
    pb = (flow * s[:T]) > 0                         # flow move increases |state| -> pushback
    pushback_count = int(np.nansum(pb))
    pushback_amount = float(np.nansum(np.abs(flow[pb])))
    s0 = s[0]
    excursion = (-np.sign(s0) * s) if s0 != 0 else np.zeros_like(s)   # >0 = past target
    ov = excursion > 0
    overshoot_count = int(np.nansum(ov))
    overshoot_amount = float(np.nansum(excursion[ov]))
    return {"total_guidance": total_guid,
            "pushback_count": pushback_count, "pushback_amount": pushback_amount,
            "overshoot_count": overshoot_count, "overshoot_amount": overshoot_amount}


# --- Realism ---------------------------------------------------------------------------
def realism_tv(gui_field, ung_field, mask):
    """Realism = total-variation distance between the normalized guidance-change field
    ``|gui - ung|`` and the (normalized) ``mask``, over the full field:

        p = |gui - ung| / sum|gui - ung|,   q = mask / sum(mask),   TV = 0.5 * sum|p - q|.

    0 = the guidance change is distributed exactly like the mask footprint; larger = the
    guidance leaks beyond / concentrates differently from the mask (less realistic control).
    ``gui_field``/``ung_field``/``mask`` are (lat, lon) for the target channel."""
    g = np.abs(np.asarray(gui_field, dtype=float) - np.asarray(ung_field, dtype=float))
    gs = g.sum()
    if not np.isfinite(gs) or gs <= 0:
        return np.nan
    p = g / gs
    q = np.asarray(mask, dtype=float)
    q = q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


# --- aggregation ----------------------------------------------------------------------
def aggregate_mean_std(values):
    """``(mean, std, n_pool)`` over a pool of scalars, ignoring NaN (population std)."""
    v = np.asarray(list(values), dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std()), int(v.size)
