from __future__ import annotations

from typing import Callable, Sequence
import torch


Tensor = torch.Tensor

def make_sampler(
    *,
    sigma: Tensor | float,
    denoiser: Callable[[Tensor, Sequence[Tensor], int], Tensor],
    decode_in_sampler: bool = True,
) -> Callable[[int, Sequence[Tensor], Tensor, Callable[..., Tensor] | None, Tensor | None, Tensor | None], Tensor]:
    """
    Factory that builds the inner generative sampler.

    Parameters
    ----------
    sigma:
        Residual scaling tensor or scalar.
    denoiser:
        Callable (z, x_cond, t) -> model_out
    decode_in_sampler:
        If True, returns x_hat = mu + sigma * z at the end.
        If False, returns z.

    Returns
    -------
    sample_fn:
        A function matching the expected `sample` protocol.
    """

    def sample(
        T: int,
        x_cond: Sequence[Tensor],
        mu: Tensor,
        guide: Callable[[int, Tensor, Tensor, Tensor, Sequence[Tensor], Tensor | None, Tensor | None], Tensor] | None,
        x_guide: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        z = ...  # TODO: Guassian noise

        for t in range(T):
            model_out = denoiser(z, x_cond, t)

            if guide is None:
                guided_out = model_out
            else:
                guided_out = guide(t, z, model_out, mu, x_cond, x_guide, mask)
            
            scheduler_step = ...  # TODO might define the function outside
            z = scheduler_step(z, guided_out, t)

        if decode_in_sampler:
            return mu + sigma * z
        return z

    return sample