Below is a **copy-paste AOWF baseline adapted to your `GuidedFlow` class**. It follows the AOWF optimizer: zero initialization, spatial-STD normalized gradients, momentum with (\beta=0.9), cosine-annealed step size, and projection to zero spatial mean and per-field spatial standard deviation (\le\epsilon). The original AOWF uses (N=50) attack iterations and a 2-step approximation; Ueyama et al. restrict the attack to the current atmospheric input, use 50 iterations, and set the budget to (0.07) normalized units. ([arXiv][1])

One adaptation is unavoidable: AOWF's approximation is designed around GenCast's denoiser. Here I implement its natural flow-matching analogue by running ArchesWeather with a reduced number of Euler steps during attack optimization. Set `aowf_approx_steps=2` for the paper-style baseline, or `None` to differentiate through all 25 ArchesWeather flow steps.

### 1. Add these methods inside `GuidedFlow`

```python
# ------------------------------------------------------------- AOWF baseline ---

def _aowf_scale_std(self, td, eps: float = 1e-12):
    """
    AOWF gradient normalization Π_1(g):
    independently for every batch/channel/level field,
    remove spatial mean and scale spatial std to 1.

    Spatial dimensions are assumed to be the final two dimensions.
    """
    def scale(x):
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)

        # A constant / unused field should remain zero.
        return torch.where(
            std > eps,
            x / std.clamp_min(eps),
            torch.zeros_like(x),
        )

    return td.apply(scale)


def _aowf_projection(self, td, epsilon: float, eps: float = 1e-12):
    """
    AOWF projection:

        mean(delta_v) = 0
        std(delta_v) <= epsilon

    independently for each atmospheric field / pressure level.

    `epsilon` is therefore measured in the normalized coordinates in which
    the perturbation is represented.
    """
    def project(x):
        x = x - x.mean(dim=(-2, -1), keepdim=True)

        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        factor = torch.minimum(
            torch.ones_like(std),
            epsilon / std.clamp_min(eps),
        )

        return x * factor

    return td.apply(project)


def _aowf_differentiable_forecast(
    self,
    x_cond,
    seed: int | None,
    approximation_steps: int | None = 2,
):
    """
    Differentiable one-weather-step ArchesWeather forecast used as the
    surrogate f() inside AOWF.

    approximation_steps:
        2     -> AOWF-style cheap surrogate
        None  -> full self.T-step ArchesWeather flow

    Returns the final DENORMALIZED weather state.
    """

    # IMPORTANT: unlike normal sampling, det_pred must remain in the graph
    # because AOWF differentiates the final forecast w.r.t. x_cond["state"].
    det_pred = self.det_model(x_cond)

    flow_cond = dict(x_cond)
    flow_cond["pred_state"] = det_pred
    flow_cond = {
        k: v for k, v in flow_cond.items()
        if "next" not in k
    }

    # Same sampled noise for the complete surrogate forecast.
    if seed is not None:
        self.generator.manual_seed(seed)

    z_t = flow_cond["state"].apply(
        lambda x: torch.empty_like(x).normal_(generator=self.generator)
    )
    z_t = z_t * self.scale_input_noise

    if approximation_steps is None:
        timesteps = self.flow_timesteps()
    else:
        approximation_steps = max(2, int(approximation_steps))

        # Reduced Euler discretization of the same s: 1 -> 0 flow.
        timesteps = torch.linspace(
            self.num_train_timesteps,
            1,
            approximation_steps,
            device=self.device,
        )

    for i in range(len(timesteps)):
        t = timesteps[i]
        s_t = t / self.num_train_timesteps

        if i < len(timesteps) - 1:
            s_next = timesteps[i + 1] / self.num_train_timesteps
            h = s_t - s_next
        else:
            h = s_t

        time_embedding = self.embed_time(flow_cond, t)
        input_state = self.velocity_input(z_t, flow_cond)

        u_t = self.velocity(
            flow_cond,
            time_embedding,
            input_state,
            z_t,
            s_t,
        )

        z_t = self.euler_step(z_t, u_t, h)

    return self.final_prediction(det_pred, z_t)


def _aowf_flow(
    self,
    x_cond,
    delta_t: torch.Tensor,
    mask: TensorDict,
    x_ref: TensorDict,
    seed: int,
    aowf_epsilon: float = 0.07,
    aowf_steps: int = 50,
    aowf_beta: float = 0.9,
    aowf_approx_steps: int | None = 2,
    aowf_attack_mask: TensorDict | None = None,
    aowf_resample_noise: bool = True,
):
    """
    AOWF baseline adapted to ArchesWeather.

    Following Ueyama et al., only the CURRENT atmospheric state is attacked:
        x_n -> x_n + delta

    x_{n-1} is kept unchanged.

    The attack minimizes exactly the same target loss as our guidance:

        L = [ M(x_hat_n) - y_n^* ]^2

          = [ M(x_hat_n)
              - (1 + delta_n) M(x_ref) ]^2

    subject to, independently for every field/level,

        mean_spatial(delta) = 0
        std_spatial(delta) <= epsilon.

    The optimized perturbation is applied ONCE to x_cond["state"], after
    which a normal unguided ArchesWeather forecast is generated.

    `aowf_attack_mask` is optional. It should preferably select whole
    channels / pressure levels rather than individual spatial pixels.
    """

    base_state = x_cond["state"].detach()

    perturbation = base_state.apply(torch.zeros_like)
    momentum = base_state.apply(torch.zeros_like)

    loss_history = []

    for k in range(aowf_steps):
        # Fresh leaf tensors for this attack iteration.
        perturbation = perturbation.apply(
            lambda x: x.detach().requires_grad_(True)
        )

        attacked_state = base_state + perturbation

        attacked_cond = dict(x_cond)
        attacked_cond["state"] = attacked_state

        # Original AOWF uses a different stochastic realization across attack
        # iterations to avoid overfitting the perturbation to one noise sample.
        attack_seed = (
            seed + k
            if aowf_resample_noise and seed is not None
            else seed
        )

        with torch.enable_grad():
            x_hat = self._aowf_differentiable_forecast(
                attacked_cond,
                seed=attack_seed,
                approximation_steps=aowf_approx_steps,
            )

            loss = self.masked_loss(
                x_hat,
                x_ref,
                delta_t,
                mask,
            )

            keys = list(perturbation.keys())

            grads = torch.autograd.grad(
                loss,
                [perturbation[key] for key in keys],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grad_td = perturbation.__class__(
            {
                key: (
                    torch.zeros_like(perturbation[key])
                    if grad is None
                    else grad
                )
                for key, grad in zip(keys, grads)
            },
            batch_size=perturbation.batch_size,
            device=perturbation.device,
        )

        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)

        with torch.no_grad():
            # Optional restriction to selected atmospheric channels.
            if aowf_attack_mask is not None:
                grad_td = tensordict_apply(
                    torch.mul,
                    grad_td,
                    aowf_attack_mask,
                )

            # Π_1(g): zero spatial mean + unit spatial std.
            grad_td = self._aowf_scale_std(grad_td)

            # Momentum:
            #
            #   m_k = beta m_{k-1} + (1-beta) Π_1(g_k)
            momentum = tensordict_apply(
                lambda m, g:
                    aowf_beta * m + (1.0 - aowf_beta) * g,
                momentum,
                grad_td,
            )

            # Original AOWF cosine schedule:
            #
            # alpha'_k goes from approximately 2 epsilon
            # down to epsilon / N.
            eta_0 = 2.0 * aowf_epsilon
            eta_min = aowf_epsilon / aowf_steps

            alpha_prime = (
                eta_min
                + 0.5
                * (eta_0 - eta_min)
                * (
                    1.0
                    + math.cos(
                        k * math.pi / aowf_steps
                    )
                )
            )

            # Bias correction used by the authors' implementation.
            alpha = alpha_prime / (
                1.0 - aowf_beta ** (k + 1)
            )

            perturbation = tensordict_apply(
                lambda d, m: d - alpha * m,
                perturbation,
                momentum,
            )

            if aowf_attack_mask is not None:
                perturbation = tensordict_apply(
                    torch.mul,
                    perturbation,
                    aowf_attack_mask,
                )

            # Π_epsilon(delta):
            #
            # zero mean and spatial std <= epsilon independently
            # for every field / pressure level.
            perturbation = self._aowf_projection(
                perturbation,
                aowf_epsilon,
            )

        print(
            f"AOWF iter {k:02d}: "
            f"loss={loss_value:.6e}",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------ final attacked forecast

    attacked_cond = dict(x_cond)
    attacked_cond["state"] = base_state + perturbation.detach()

    # The actual reported forecast uses the normal full ArchesWeather sampler,
    # not the cheap differentiable surrogate used by the attacker.
    with torch.no_grad():
        det_pred = self.det_model(attacked_cond)

    attacked_cond["pred_state"] = det_pred
    attacked_cond = {
        k: v for k, v in attacked_cond.items()
        if "next" not in k
    }

    z_t, sampling_trace = self.unguided_flow(
        attacked_cond,
        det_pred,
        seed,
    )

    sampling_trace["aowf"] = {
        "epsilon": float(aowf_epsilon),
        "steps": int(aowf_steps),
        "approximation_steps": (
            None
            if aowf_approx_steps is None
            else int(aowf_approx_steps)
        ),
        "loss_history": loss_history,
        "perturbation": perturbation.detach().cpu(),
    }

    return z_t, sampling_trace, det_pred
```

The projection above is directly the paper's constraint
[
\mu_v=0,\qquad \sigma_v\le\epsilon,
]
implemented independently over the spatial dimensions of every field. ([arXiv][1]) The momentum, gradient standardization, cosine schedule, and projection order follow the authors' released `our_attack` implementation. ([GitHub][2])

### 2. Replace your `sample()` method with this version

The special branch is needed because AOWF must perturb `x_cond["state"]` **before** computing the deterministic prediction. Your current implementation computes `det_pred` before dispatching the guidance method, so AOWF cannot simply be added to the existing `flows` dictionary. 

```python
def sample(
    self,
    guidance_flag: bool,
    guidance_type: str,
    x_cond: dict,
    delta_n: torch.Tensor | None = None,
    mask: TensorDict | None = None,
    x_ref: TensorDict | None = None,
    seed: int | None = None,
    guidance_kwargs: dict | None = None,
):
    guidance_kwargs = guidance_kwargs or {}

    # ---------------------------------------------------------- AOWF attack
    #
    # AOWF is special: it modifies the atmospheric CONDITIONING STATE,
    # therefore it must run before det_pred is computed.
    if guidance_flag and guidance_type == "AOWF":
        z, sampling_trace, det_pred = self._aowf_flow(
            x_cond=x_cond,
            delta_t=delta_n,
            mask=mask,
            x_ref=x_ref,
            seed=seed,
            **guidance_kwargs,
        )

    else:
        # ------------------------------------------------ standard prediction
        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred

        x_cond = {
            k: v for k, v in x_cond.items()
            if "next" not in k
        }

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(
                x_cond,
                det_pred,
                seed,
            )

        else:
            flows = {
                "OPTIMIZE-GAIN": self._fgwnolr_flow,
                "CLOSE-GAP": self._fgwnogap_flow,
                "OPTIMIZE-SCHEDULE": self._fgwfree_flow,
                "UG": self._ug_flow,
            }

            if guidance_type not in flows:
                raise ValueError(
                    f"Unknown guidance_type: {guidance_type}"
                )

            z, sampling_trace = flows[guidance_type](
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **guidance_kwargs,
            )

    sampling_trace["det_pred"] = det_pred.detach()

    x_hat_norm = det_pred + tensordict_apply(
        torch.mul,
        z,
        self.residual_to_pangu_scale,
    )

    return x_hat_norm, sampling_trace
```

Then you can invoke it as:

```python
guidance_type = "AOWF"

guidance_kwargs = {
    "aowf_epsilon": 0.07,
    "aowf_steps": 50,
    "aowf_beta": 0.9,
    "aowf_approx_steps": 2,
}
```

Those `0.07` and `50` settings reproduce the baseline configuration reported by Ueyama et al.; they also attack only the current atmospheric input rather than both conditioning states. ([arxiv.org][3])

### One choice I would make differently for your thesis

For the actual comparison, I would probably run **two AOWF variants once as a sanity check**:

```python
"aowf_approx_steps": 2
```

for the paper-matched cheap attack, and

```python
"aowf_approx_steps": None
```

for attack optimization through the full 25-step Arches flow.

AOWF's two-level inference approximation was developed specifically because differentiating through GenCast's diffusion chain is expensive. ([arXiv][1]) In your case the generator is an Euler-discretized flow, so a two-step Euler surrogate is a somewhat stronger approximation than the two-denoiser-call approximation used in GenCast. If both produce comparable adversarial perturbations, you can safely use the cheaper one. If they differ substantially, I would use the full-flow version for the thesis and describe it as **AOWF adapted to ArchesWeather**, rather than pretending the GenCast-specific approximation transfers exactly.

[1]: https://arxiv.org/html/2504.15942 "https://arxiv.org/html/2504.15942"
[2]: https://github.com/mlsec-group/adversarial-observations/blob/main/src/utils/attacks.py "https://github.com/mlsec-group/adversarial-observations/blob/main/src/utils/attacks.py"
[3]: https://arxiv.org/html/2605.14317 "Guided Diffusion Sampling for Precipitation Forecast Interventions"
