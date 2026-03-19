
from __future__ import annotations

from typing import Callable, Sequence, TypeAlias
import torch


Tensor = torch.Tensor


GuideFn: TypeAlias = Callable[
    [int, Tensor, Tensor, Tensor, Sequence[Tensor], Tensor | None, Tensor | None],
    Tensor,
]


def make_no_guidance() -> GuideFn:
    def guide_fn(
        t: int,
        z: Tensor,
        model_out: Tensor,
        mu: Tensor,
        x_cond: Sequence[Tensor],
        x_guide: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        return model_out
    return guide_fn


def make_some_other_guidance(param1: float, param2: float) -> GuideFn:
    def guide_fn(
        t: int,
        z: Tensor,
        model_out: Tensor,
        mu: Tensor,
        x_cond: Sequence[Tensor],
        x_guide: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        # use param1, param2 here
        return model_out
    return guide_fn


# def make_masked_mean_guidance(
#     sigma: Tensor | float,
#     *,
#     strength_schedule: Callable[[int, int], float],
#     model_out_to_z_correction: Callable[[Tensor], Tensor] | None = None,
# ) -> Callable[[int, Tensor, Tensor, Tensor, Sequence[Tensor], Tensor | None, Tensor | None], Tensor]:
#     """
#     Factory for a simple gradient-based guidance function that nudges the
#     masked mean of the decoded sample x = mu + sigma * z toward x_guide.

#     Important:
#     - This computes gradients w.r.t. z.
#     - It then turns that gradient into a correction added to model_out.
#     - The exact correctness of adding this directly to model_out depends on
#       what your denoiser predicts. For now, this uses the identity map unless
#       you provide model_out_to_z_correction.

#     Parameters
#     ----------
#     sigma:
#         Residual scaling tensor or scalar. Must be broadcastable to z.
#     strength_schedule:
#         Function (t, T_minus_1) -> float controlling guidance strength.
#     model_out_to_z_correction:
#         Optional transform from grad_z(loss) to the space of model_out.
#         If omitted, the identity is used.

#     Returns
#     -------
#     guide_fn:
#         A callable with the same protocol as `guide`.
#     """
#     if model_out_to_z_correction is None:
#         model_out_to_z_correction = lambda g: g

#     def guide_fn(
#         t: int,
#         z: Tensor,
#         model_out: Tensor,
#         mu: Tensor,
#         x_cond: Sequence[Tensor],
#         x_guide: Tensor | None,
#         mask: Tensor | None,
#     ) -> Tensor:
#         if x_guide is None:
#             return model_out

#         # We need a differentiable view of z for the guide loss.
#         z_req = z.detach().clone().requires_grad_(True)
#         x_curr = mu + sigma * z_req
#         loss = (masked_mean(x_curr, mask) - x_guide) ** 2

#         grad_z = torch.autograd.grad(loss, z_req, allow_unused=False)[0]
#         lam = float(strength_schedule(t, t if t == 0 else t))  # overwritten below by wrapper if needed
#         correction = model_out_to_z_correction(grad_z)

#         # Minus sign: descend the guide loss.
#         return model_out - lam * correction

#     return guide_fn