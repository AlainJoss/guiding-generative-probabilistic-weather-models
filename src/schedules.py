import math
import numpy as np


def N_schedule(
    N: int,
    flatness: float,
    peak_magnitude: float,
    peak_at_n: int | None = None,
) -> list[float]:
    if N < 1:
        raise ValueError("N must be >= 1")
    if flatness <= 0:
        raise ValueError("flatness must be > 0")

    if peak_at_n is None:
        peak_at_n = N // 2

    if not (1 <= peak_at_n <= N):
        raise ValueError(f"peak_at_n must be in [1, {N}], got {peak_at_n}")

    values = []

    for n in range(N + 1):
        if n <= peak_at_n:
            x = n / peak_at_n
            s = math.sin(0.5 * math.pi * x)
        else:
            x = (N - n) / (N - peak_at_n)
            s = math.sin(0.5 * math.pi * x)

        values.append(peak_magnitude * s**flatness)

    return values


def guidance_weight_schedule(
    T: int,
    w_values: list[float],
    num_train_timesteps: int = 1000,
    lambda0: float = 1.0,
    c: float = 5.0,
) -> dict[str, list[float]]:
    """Per-flow-step guidance weight {f"w={w}": w * a_t} for a T-step Euler schedule.

    Mirrors GuidedFlow.a_t / get_flow_timesteps EXACTLY (the bounded version):
      s_t   = noise level running 1 -> 1/num_train_timesteps over T steps
      s_eff = clip(s_t - ds, 0, 1 - ds),  ds = 1/num_train_timesteps
      a     = s_eff / (1 - s_eff)
      a_t   = lambda0 * a / (a + c)   # approaches lambda0, never explodes; 0 at the last step
    """
    s = np.linspace(num_train_timesteps, 1, T) / num_train_timesteps
    ds = 1.0 / num_train_timesteps
    s_eff = np.clip(s - ds, 0.0, 1.0 - ds)
    a = s_eff / (1.0 - s_eff)
    a_t = lambda0 * a / (a + c)
    return {f"w={w:g}": (w * a_t).tolist() for w in w_values}


def delta_schedule(
    N: int,
    mode: str,
    peak_magnitude: float,
    peak_at_n: int | None = None,
    stop_at_n: int | None = None,
    flatness: float = 1.0,
    start_value: float = 0.0,
) -> list[float]:
    if N < 1:
        raise ValueError("N must be >= 1")
    if peak_at_n is None:
        peak_at_n = N // 2
    peak_at_n = max(0, min(peak_at_n, N))

    def shape(x):                      # x in [0, 1] -> [0, 1]
        if mode == "linear":
            return x
        if mode == "sinusoidal":
            return math.sin(0.5 * math.pi * x) ** flatness
        raise ValueError(f"Invalid delta_mode {mode}")

    values = []
    for n in range(N + 1):
        # natural full-span bump: rise from start_value to peak at peak_at_n, fall to 0 at N
        if n <= peak_at_n:
            # rise anchored at n=1 (first kept step after [1:]): exactly start_value, then -> peak
            denom = peak_at_n - 1
            x = max((n - 1) / denom, 0.0) if denom > 0 else 1.0
            v = start_value + (peak_magnitude - start_value) * shape(x)
        else:
            denom = N - peak_at_n
            x = (N - n) / denom if denom > 0 else 0.0             # fall: peak -> 0 at N
            v = peak_magnitude * shape(x)
        # hard cutoff: stop_at_n zeroes the tail, independent of peak_at_n
        if stop_at_n is not None and n >= stop_at_n:
            v = 0.0
        values.append(v)

    return values