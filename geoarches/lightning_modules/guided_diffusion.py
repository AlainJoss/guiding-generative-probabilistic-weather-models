import math
from collections import defaultdict

import pandas as pd
import torch

from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm.auto import tqdm

from geoarches.backbones.dit import TimestepEmbedder
from geoarches.lightning_modules import BaseLightningModule
from geoarches.lightning_modules.base_module import AvgModule, load_module
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

from geoarches.paths import STATS_PATH


class GuidedFlow(BaseLightningModule):
    """
    Rollout, sample, and guide predictions (det+gen).
    """
    def __init__(
        self,
        cfg,
        name="diffusion",
        cond_dim=256,
        num_train_timesteps=1000,
        scheduler="flow",  # only available option
        prediction_type="sample",  # or velocity
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.012,
        loss_weighting_strategy=None,
        conditional="",  # things that the model is conditioned
        load_deterministic_model=False,
        loss_delta_normalization=False,
        state_normalization=False,
        pow=2,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        learn_residual=False,
        sd3_timestep_sampling=True,
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(locals())
        print("initializing GuidedFlow")
        
        # some constants that were floating hardcoded around the codebase 
        # or are uselessly in the cfg (since they are fix anyway)
        self.scale_input_noise = 1.05  # NOTE: this is not sigma
        self.num_train_timesteps=1000
        self.cond_dim=256
        self.T = 25

        # LBG (Monte-Carlo loss-based guidance) hyperparameters (tunable)
        # n_mc: number of MC samples; r_t: std of the Gaussian q(x0|xt)=N(x_hat_t, r_t^2 I)
        # drawn in normalized space. n_mc=1, r_t=0 makes LBG reduce to DPS.

        self.cfg = cfg
        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)

        # NOTE: we could change this to use multiple det models
        if isinstance(load_deterministic_model, str):
            self.det_model, _ = load_module(load_deterministic_model)
        else:
            self.det_model = AvgModule(load_deterministic_model)

        # TODO: can I put this in a pipeline too?
        self.month_embedder = TimestepEmbedder(self.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cond_dim)
        self.timestep_embedder = TimestepEmbedder(self.cond_dim)

        # sigma scaling factor (shape like usual states)
        pangu_stats = torch.load(
            STATS_PATH / "pangu_norm_stats2_with_w.pt", weights_only=True
        )
        pangu_scaler = TensorDict(
            level=pangu_stats["level_std"], surface=pangu_stats["surface_std"]
        )
        scaler = TensorDict(
            **torch.load(
                STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False
            )
        )
        scaler["level"][-1] *= 3  # we don't care too much about vertical velocity
        self.residual_to_pangu_scale = scaler / pangu_scaler  # inverse because we divide by state_scaler
        # taken from era5.py
        self.data_mean = TensorDict(
            surface=pangu_stats["surface_mean"],
            level=pangu_stats["level_mean"],
        )
        self.data_std = TensorDict(
            surface=pangu_stats["surface_std"],
            level=pangu_stats["level_std"],
        )

        print("initialized GuidedFlow")


    ### utils ###

    def move_objects_to_device(self):
        # device comes from initialization pipeline, not visible in init
        self.residual_to_pangu_scale = self.residual_to_pangu_scale.to(self.device)
        self.data_mean = self.data_mean.to(self.device)
        self.data_std = self.data_std.to(self.device)
        self.generator = torch.Generator(self.device)


    def denormalize(self, batch):
        # TODO: not sure why the presence of surface should enable this 
        if "surface" in batch:
            # we can denormalize directly
            return batch * self.data_std + self.data_mean

        out = {k: (v * self.data_std + self.data_mean if "state" in k else v) for k, v in batch.items()}
        return out
    
    
    def normalize(self, batch):
        if "surface" in batch:
            return (batch - self.data_mean) / self.data_std

        out = {
            k: ((v - self.data_mean) / self.data_std if "state" in k else v)
            for k, v in batch.items()
        }
        return out
    

    ### flow funcs ###

    def embedd_time(self, batch, t):
        times = pd.to_datetime(
            batch["timestamp"].detach().cpu().numpy(),
            unit="s",
        ).tz_localize(None)
        month = torch.tensor(times.month).to(self.device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor(times.hour).to(self.device)
        hour_emb = self.hour_embedder(hour)
        timestep_emb = self.timestep_embedder(torch.tensor([t]).to(self.device))
        # print(f"embedding time for gen model - month:{int(month)}, hour:{int(hour)}")

        time_embedding = month_emb + hour_emb + timestep_emb
        return time_embedding
    

    def get_velocity_input_state(self, z, batch):
        # only init of concat (we need z later as is)
        assert "pred_state" in batch
        pred_state = batch["pred_state"]
        prev_state = batch["prev_state"]
        z = tensordict_cat([prev_state, z], dim=1)
        z = tensordict_cat([pred_state, z], dim=1)
        return z
    

    def velocity(self, batch, time_embedding, input_state, z_t, s_t):
        ##### compute residual #####

        # here we embedd prev_state (input_state[0]), current_state (batch["state"]), noisy_state (input_state[1])
        x = self.embedder.encode(batch["state"], input_state)
        x = self.backbone(x, time_embedding)
        r_t = self.embedder.decode(x)  # we get tdict

        ##### compute velocity from residual
        # u_t = r_t - eps_t := (r_t - z_t) / s_t
        u_t = (r_t - z_t).apply(lambda x: x / s_t)
        return u_t
    
    
    def guided_velocity(self, u_t, gui_vec, lambda_):
        return tensordict_apply(
            lambda u, g: u - lambda_ * g,
            u_t,
            gui_vec,
        )


    def euler_step(self, z_t, u_t, h):
        # z_new = z_t + h * u_t
        return tensordict_apply(lambda z, u: z + h * u, z_t, u_t)
    

    def init_noise(self, x_cond, seed=None):
        if seed is not None:
            self.generator.manual_seed(seed)

        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        return z_t * self.scale_input_noise


    def get_timesteps(self):
        return torch.linspace(
            self.num_train_timesteps,
            1,
            self.T,
            device=self.device,
        )
    

    def get_step_factors(self, i, timesteps):
        t = timesteps[i]
        s_t = t / self.num_train_timesteps

        if i < len(timesteps) - 1:
            s_next = timesteps[i + 1] / self.num_train_timesteps
            h = s_t - s_next
        else:
            h = s_t

        return t, s_t, h
 

    def clean_prediction(self, det_pred, z_t, u_t, s_t):
        x_hat_t_norm = det_pred + (
            z_t + tensordict_apply(torch.mul, s_t, u_t)
        ) * self.residual_to_pangu_scale

        return self.denormalize(x_hat_t_norm)
        

    def final_prediction(self, det_pred, z_t):
        x_hat_norm = det_pred + tensordict_apply(
            torch.mul,
            z_t,
            self.residual_to_pangu_scale,
        )
        return self.denormalize(x_hat_norm)

            
    def grad_loss(self, loss_, z_t, create_graph: bool = False):
        keys = list(z_t.keys())
        tensors = [z_t[k] for k in keys]

        grads = torch.autograd.grad(
            loss_,
            tensors,
            retain_graph=create_graph,
            create_graph=create_graph,
            allow_unused=True,
        )

        return z_t.__class__(
            {
                k: torch.zeros_like(z_t[k]) if g is None else g
                for k, g in zip(keys, grads)
            },
            batch_size=z_t.batch_size,
            device=z_t.device,
        )
    

    ### used outside to wire guidance type ###
        
    def sample(
        self,
        guidance_flag: bool,
        guidance_type: str,
        x_cond: dict,
        delta_t: torch.Tensor | None = None,
        mask: TensorDict | None = None,
        x_hat_ung: TensorDict | None = None,
        lambda_schedule: list[torch.Tensor] | None = None,
        seed: int | None = None,
        guidance_kwargs: dict | None = None,
    ):
        guidance_kwargs = guidance_kwargs or {}

        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred

        x_cond = {k: v for k, v in x_cond.items() if "next" not in k}

        gradient_guidance = {
            "DPS": self.dps_guidance,
            "LBG": self.lbg_guidance,
        }

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(
                x_cond=x_cond,
                seed=seed,
            )
        elif guidance_type in gradient_guidance:
            z, sampling_trace = self._gradient_flow(
                guidance_name=guidance_type,
                guidance_fn=gradient_guidance[guidance_type],
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "UG":
            z, sampling_trace = self._ug_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FLOWGRAD":
            z, sampling_trace = self._flowgrad_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                seed=seed,
                **guidance_kwargs,
            )
        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

        x_hat_norm = det_pred + tensordict_apply(
            torch.mul,
            z,
            self.residual_to_pangu_scale,
        )

        return x_hat_norm, sampling_trace

    ### base flow ###
    
    def unguided_flow(
        self,
        x_cond: dict,
        seed: int | None = None,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()

        for i in tqdm(range(len(timesteps))):
            t, s_t, h = self.get_step_factors(i, timesteps)

            with torch.no_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                z_t = self.euler_step(z_t, u_t, h)

        return z_t, None

    def _gradient_flow(
        self,
        guidance_name: str,
        guidance_fn,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: list[torch.Tensor],
        seed: int | None = None,
        create_graph: bool = False,
        differentiable: bool = False,
        trace: bool = True,
        **guidance_kwargs,
    ):
        # create_graph : keep higher-order graph through the guidance gradient
        # differentiable: keep the whole trajectory attached so the outer
        #                 objective stays differentiable w.r.t. lambda_schedule (FLOWGRAD)
        # trace         : record the per-step sampling_trace (skipped for FLOWGRAD inner passes)
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()
        sampling_trace = defaultdict(list)

        if differentiable:
            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

        for i in tqdm(range(len(timesteps)), desc=f"{guidance_name} sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)

            if not differentiable:
                z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

                x_hat_t_norm = det_pred + self.euler_step(
                    z_t,
                    u_t,
                    s_t,
                ) * self.residual_to_pangu_scale

                x_hat_t = self.denormalize(x_hat_t_norm)

                gui_vec = guidance_fn(
                    x_hat_t=x_hat_t,
                    x_hat_t_norm=x_hat_t_norm,
                    x_hat_ung=x_hat_ung,
                    delta_t=delta_t,
                    mask=mask,
                    z_t=z_t,
                    create_graph=create_graph,
                    **guidance_kwargs,
                )

            if trace:
                sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                sampling_trace["grads"].append(gui_vec.detach().cpu())
                sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

            u_t = self.guided_velocity(
                u_t=u_t,
                gui_vec=gui_vec,
                lambda_=lambda_schedule[i],
            )

            if trace:
                sampling_trace["gui_vfs"].append(
                    u_t.apply(lambda x: x * s_t).detach().cpu()
                )

            if differentiable:
                z_t = self.euler_step(z_t, u_t, h)
            else:
                with torch.no_grad():
                    z_t = self.euler_step(z_t, u_t, h)

        return z_t, sampling_trace


    ### loss + guidance-vector helpers (shared by all guidance types) ###

    def masked_residual(self, x_hat_t, x_hat_ung, delta_t, mask):
        pred = sum((mask[k] * x_hat_t[k]).sum() for k in x_hat_t.keys())
        target = sum((mask[k] * x_hat_ung[k]).sum() for k in x_hat_ung.keys())
        target = (1 + delta_t) * target
        return pred - target


    def similarity_loss(self, x_hat_t, x_hat_ung):
        return sum(
            ((x_hat_t[k] - x_hat_ung[k]) ** 2).mean()
            for k in x_hat_t.keys()
        )


    def build_loss(self, x_hat_t, x_hat_ung, delta_t, mask, regularized=False, beta=1e-4):
        # single return contract for every guidance type: (loss, |residual|)
        residual = self.masked_residual(x_hat_t, x_hat_ung, delta_t, mask)
        loss = residual ** 2
        if regularized:
            loss = loss + beta * self.similarity_loss(x_hat_t, x_hat_ung)
        return loss, residual.abs()


    def normalize_grad(self, gui_vec, residual, eps=1e-8):
        # divide the guidance vector by the (detached) residual magnitude
        scale = residual.detach().clamp_min(eps)
        return gui_vec.apply(lambda g: g / scale)


    def dps_guidance(
        self,
        *,
        x_hat_t,
        x_hat_t_norm=None,
        x_hat_ung,
        delta_t,
        mask,
        z_t,
        regularized=False,
        beta=1e-4,
        normalize=True,
        eps=1e-8,
        create_graph=False,
        **extra,
    ):
        loss_, residual = self.build_loss(
            x_hat_t, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
        )
        gui_vec = self.grad_loss(loss_, z_t, create_graph=create_graph)

        if normalize:
            gui_vec = self.normalize_grad(gui_vec, residual, eps=eps)

        return gui_vec


    def lbg_guidance(
        self,
        *,
        x_hat_t,
        x_hat_t_norm,
        x_hat_ung,
        delta_t,
        mask,
        z_t,
        regularized=False,
        beta=1e-4,
        normalize=False,
        eps=1e-8,
        n_mc=4,
        r_t=1.0,
        create_graph=False,
        **extra,
    ):
        per_sample_losses = []

        for _ in range(n_mc):
            x_i_norm = x_hat_t_norm.apply(
                lambda x: x + r_t * torch.empty_like(x).normal_(generator=self.generator)
            )
            x_i = self.denormalize(x_i_norm)

            loss_i, _ = self.build_loss(
                x_i, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
            )
            per_sample_losses.append(loss_i)

        losses = torch.stack(per_sample_losses)
        loss_ = math.log(n_mc) - torch.logsumexp(-losses, dim=0)
        gui_vec = self.grad_loss(loss_, z_t, create_graph=create_graph)

        if normalize:
            _, residual = self.build_loss(x_hat_t, x_hat_ung, delta_t, mask)
            gui_vec = self.normalize_grad(gui_vec, residual, eps=eps)

        return gui_vec


    ### flow variant 3 ###

    def _ug_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: list[torch.Tensor],
        seed: int | None = None,
        S: int = 4,
        m: int = 5,
        delta_lr: float = 1e-1,
        regularized: bool = False,
        beta: float = 1e-4,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps)), desc="UG sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)
            s_next = s_t - h  # h == s_t on the final step -> s_next == 0

            time_embedding = self.embedd_time(x_cond, t)

            for k in range(S):
                with torch.no_grad():
                    input_state = self.get_velocity_input_state(z_t, x_cond)
                    u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                    x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)

                delta_x = self.UG_clean_shift(
                    x_hat_t=x_hat_t,
                    x_hat_ung=x_hat_ung,
                    delta_t=delta_t,
                    mask=mask,
                    m=m,
                    lr=delta_lr,
                    regularized=regularized,
                    beta=beta,
                )

                gui_vec = self.UG_shift_to_velocity(
                    delta_x=delta_x,
                    s_t=s_t,
                )

                if normalize:
                    _, residual = self.build_loss(x_hat_t, x_hat_ung, delta_t, mask)
                    gui_vec = self.normalize_grad(gui_vec, residual, eps=eps)

                if k == S - 1:
                    sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                    sampling_trace["grads"].append(gui_vec.detach().cpu())
                    sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

                with torch.no_grad():
                    u_t = self.guided_velocity(
                        u_t=u_t,
                        gui_vec=gui_vec,
                        lambda_=lambda_schedule[i],
                    )

                    if k == S - 1:
                        sampling_trace["gui_vfs"].append(
                            u_t.apply(lambda x: x * s_t).detach().cpu()
                        )

                    z_next = self.euler_step(z_t, u_t, h)

                    if k < S - 1:
                        z_t = self.renoise(z_next, s_t, s_next)

            z_t = z_next

        return z_t, sampling_trace
    
    def UG_clean_shift(
        self,
        x_hat_t: TensorDict,
        x_hat_ung: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        m: int = 5,
        lr: float = 1e-1,
        regularized: bool = False,
        beta: float = 1e-4,
    ):
        delta_x = x_hat_t.apply(
            lambda x: torch.zeros_like(x, requires_grad=True)
        )

        optimizer = torch.optim.SGD(
            [delta_x[k] for k in delta_x.keys()],
            lr=lr,
        )

        x_hat_t = x_hat_t.detach()
        x_hat_ung = x_hat_ung.detach()

        for _ in range(m):
            optimizer.zero_grad()

            shifted = x_hat_t + delta_x

            loss, _ = self.build_loss(
                shifted,
                x_hat_ung,
                delta_t,
                mask,
                regularized=regularized,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for k in delta_x.keys():
                    delta_x[k].mul_(mask[k])

        return delta_x.detach()
    
    def UG_shift_to_velocity(
        self,
        delta_x: TensorDict,
        s_t: torch.Tensor,
    ):
        # delta_x is in denormalized data space.
        # Map: denorm data -> normalized data -> residual space -> velocity space.
        delta_r = delta_x / (self.data_std * self.residual_to_pangu_scale)

        # guided_velocity does: u - lambda * gui_vec
        # To apply u + delta_r / s_t, we return negative vector.
        return delta_r.apply(lambda x: -x / s_t)
    

    def renoise(self, z, s_high, s_low):
        # Self-recurrence renoise: bring a sample at noise level s_low back up to s_high.
        # Rectified-flow translation of the DDPM renoise under alpha_bar ~ (1 - s)^2.
        ratio = (1 - s_high) / (1 - s_low)
        eps = z.apply(lambda x: torch.empty_like(x).normal_(generator=self.generator))
        eps = eps * self.scale_input_noise
        coeff = (1 - ratio ** 2) ** 0.5
        return tensordict_apply(lambda zz, ee: ratio * zz + coeff * ee, z, eps)
    

    ### flow variant 4 ###

    def _flowgrad_flow(
        self,
        x_cond,
        det_pred,
        delta_t,
        mask,
        x_hat_ung,
        seed: int,
        n_opt: int = 10,
        lr: float = 1e-2,
        gamma: float = 1e-3,
        init_lambda: float = -10.0,
        regularized: bool = True,
        beta: float = 1e-4,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        # Optimize a per-timestep lambda_schedule by backpropagating an outer
        # objective through the (differentiable) guided gradient flow, then do a
        # final, plain pass with the optimized schedule.
        raw_lambda = torch.nn.Parameter(
            torch.full((self.T,), init_lambda, device=self.device)
        )
        optimizer = torch.optim.Adam([raw_lambda], lr=lr)
        trace = defaultdict(list)

        def run_flow(lambda_schedule, *, create_graph, differentiable, do_trace):
            return self._gradient_flow(
                guidance_name="FLOWGRAD",
                guidance_fn=self.dps_guidance,
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
                create_graph=create_graph,
                differentiable=differentiable,
                trace=do_trace,
                regularized=regularized,
                beta=beta,
                normalize=normalize,
                eps=eps,
            )

        for _ in tqdm(range(n_opt), desc="FLOWGRAD lambda optimization"):
            optimizer.zero_grad()

            lambda_schedule = torch.nn.functional.softplus(raw_lambda)

            z_t, _ = run_flow(
                lambda_schedule,
                create_graph=True,
                differentiable=True,
                do_trace=False,
            )

            x_hat_gui = self.final_prediction(det_pred, z_t)
            target_loss, _ = self.build_loss(
                x_hat_gui, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
            )
            reg_loss = gamma * (lambda_schedule ** 2).sum()
            loss = target_loss + reg_loss

            loss.backward()
            optimizer.step()

            trace["loss"].append(loss.detach().cpu())
            trace["target_loss"].append(target_loss.detach().cpu())
            trace["reg_loss"].append(reg_loss.detach().cpu())
            trace["lambda_schedule"].append(lambda_schedule.detach().cpu())

        lambda_star = torch.nn.functional.softplus(raw_lambda).detach()

        z_t, sampling_trace = run_flow(
            lambda_star,
            create_graph=False,
            differentiable=False,
            do_trace=True,
        )

        sampling_trace["flowgrad"] = trace
        sampling_trace["lambda_star"] = lambda_star.detach().cpu()

        return z_t, sampling_trace