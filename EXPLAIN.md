FGWNOLR ("no learning rate") finds w by treating the whole guided rollout as a scalar function of one variable and root-finding on its derivative. Step by step:

The objective. Fix the profile a_t = (1−η)^{t+1}. For a candidate w, run the entire guided flow with λ_t = w·a_t (in the new convention: kick norm w·a_t·‖g_t‖, i.e. raw multiplier w·a_t) and measure the final masked loss:

$$L_{\text{final}}(w) = e_T(w)^2, \qquad e_T = M\bigl(x^{\text{gui}}_T(w)\bigr) - y$$

Since the gap e_T is roughly affine in w, L(w) is roughly a parabola — one w closes it.

One evaluation = loss + exact derivative (_flow_loss_and_lambda_grads). The trick is that dL/dw is computable exactly (to first order) without backpropagating through the unrolled network:

Forward: run the guided Euler pass at this w; at every step cache the leaf state z_i and the (detached) gradient g_i computed on the realized path.
Final: evaluate L and its adjoint a = ∂L/∂z_T.
Backward adjoint loop (frozen-g): each step's kick enters the next state as −h_i·λ_i·g_i, so $$\frac{\partial L}{\partial \lambda_i} = -h_i,\langle a_{i+1},, g_i\rangle$$ and between steps the adjoint is pulled back through the model Jacobian with one VJP: a ← a + h·(∂u/∂z)ᵀa. "Frozen-g" means the ∂g/∂z curvature term is dropped — the only approximation.
Chain rule: λ_i = w·a_i, so $$\frac{dL}{dw} = \sum_i a_i \cdot \frac{\partial L}{\partial \lambda_i}$$
Cost: one forward with a gradient pass per step plus one VJP per step — about 2T model passes per evaluation.

The secant iteration. With (L, dL/dw) per point, solve dL/dw = 0:

Evaluate at w₀ = fgwnolr_w_init; bootstrap a second point w₁ = w₀ ∓ 10% (moving against the derivative's sign).
Then repeat the secant update — a finite-difference Newton step on the derivative: $$w_{k+1} = w_k - \frac{dL_k,(w_k - w_{k-1})}{dL_k - dL_{k-1}}$$
Safeguards: w is projected to ≥ 0; each step is capped at 3× the previous one (the secant shoots off when the derivative flattens); it stops when the loss drops under 10⁻⁶ (gap closed), the step shrinks below 10⁻³ (converged in w), the denominator degenerates, or 30 evaluations are hit.
Best-loss-wins: w* is the argmin of the measured losses over all visited points, not the last iterate — the secant targets a stationary point of an approximate derivative, so the safest answer is the best point actually seen.
Final pass. One more guided flow at w*, this time recording all the traces and the w_star sidecar.

The name captures the design: because dL/dw is a single exact scalar, there's no gradient-descent loop — hence no learning rate and no iteration-count hyperparameter to tune; the only knob is the starting point, and the optimization runs until the gap is closed or w converges. This is also why NOLR kept its secant while FGWRHO lost it: NOLR's λ = w·a doesn't depend on the realized path, so the frozen-g derivative is only mildly biased and reliably crosses zero near the loss minimum; RHO's path-dependent ratio made its derivative biased enough (positive on both sides of the minimum in your log) that we switched it to derivative-free parabolic search on the measured losses.