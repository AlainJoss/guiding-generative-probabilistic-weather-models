from __future__ import annotations

from typing import Callable, Sequence
import torch


Tensor = torch.Tensor


def rollout(
    N: int,
    T: int,
    x_cond: Sequence[Tensor],
    *,
    deterministic: Callable[[Sequence[Tensor]], Tensor],
    sample: Callable[[int, Sequence[Tensor], Tensor, Callable[..., Tensor] | None, Tensor | None, Tensor | None], Tensor],
    guide: Callable[[int, Tensor, Tensor, Tensor, Sequence[Tensor], Tensor | None, Tensor | None], Tensor] | None = None,
    x_guide_trajectory: Sequence[Tensor] | None = None,
    mask: Tensor | None = None,
) -> list[Tensor]:
    """
    Outer autoregressive rollout over forecast time.

    Parameters
    ----------
    N:
        Number of rollout steps.
    T:
        Number of inner generative steps per rollout step.
    x_cond:
        Conditioning sequence, typically [X_{n-1}, X_n].
    deterministic:
        Callable returning the deterministic forecast mean mu from x_cond.
    sample:
        Callable implementing the inner generative loop.
    guide:
        Guidance function applied inside the inner generative loop.
        If None, sampling is unguided.
    x_guide_trajectory:
        Optional guide target per rollout step.
    mask:
        Optional broadcastable mask used by guidance.

    Returns
    -------
    trajectory:
        List of generated states x_hat for each rollout step.
    """

    trajectory: list[Tensor] = []

    for n in range(N):
        x_guide = None if x_guide_trajectory is None else x_guide_trajectory[n]

        x_hat = rollout_step(
            T=T,
            x_cond=x_cond,
            deterministic=deterministic,
            sample=sample,
            guide=guide,
            x_guide=x_guide,
            mask=mask,
        )
        trajectory.append(x_hat)

        # TODO: use tensor_dict apply and "state" key
        if n < N - 1:
            x_cond = [x_cond[1], x_hat]

    return trajectory

def rollout_step(
    T: int,
    x_cond: Sequence[Tensor],
    *,
    deterministic: Callable[[Sequence[Tensor]], Tensor],
    sample: Callable[[int, Sequence[Tensor], Tensor, Callable[..., Tensor] | None, Tensor | None, Tensor | None], Tensor],
    guide: Callable[[int, Tensor, Tensor, Tensor, Sequence[Tensor], Tensor | None, Tensor | None], Tensor] | None,
    x_guide: Tensor | None,
    mask: Tensor | None,
) -> Tensor:
    """
    Single outer rollout step:
    1. deterministic mean forecast mu
    2. sample latent residual z
    3. decode x_hat = mu + SIGMA * z

    Note:
    - The residual scaling SIGMA is handled inside `sample` or via closure there.
    - If you want it here instead, adapt `sample` to return z and move decode here.
    """
    mu = deterministic(x_cond)
    x_hat = sample(T, x_cond, mu, guide, x_guide, mask)
    return x_hat