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
        print("initialized GuidedFlow")
        
        # some constants that were floating hardcoded around the codebase 
        # or are uselessly in the cfg (since they are fix anyway)
        self.scale_input_noise = 1.05  # NOTE: this is not sigma
        self.num_train_timesteps=1000
        self.cond_dim=256
        self.T = 25

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

    def denormalize(self, batch):
        means = self.data_mean #.to(self.device)
        stds = self.data_std #.to(self.device)
        # TODO: not sure why the presence of surface should enable this 
        if "surface" in batch:
            # we can denormalize directly
            return batch * stds + means

        out = {k: (v * stds + means if "state" in k else v) for k, v in batch.items()}
        return out

    def sample_rollout(
        self, 
        config,
        m: int,
        x_cond: dict, 
        y: list[TensorDict] | None = None,
        masks: list[TensorDict] | None = None
    ):
        guided_trajectory = []
        for n in range(0, config.N):
            y_n = None if y is None else y[n]  # TODO: check the index of the guidance
            mask_n = None if masks is None else masks[n]
            x_hat = self.sample(
                x_cond=x_cond,
                y_n=y_n,
                mask=mask_n,
                lambda_=config.lambda_,
                seed=m + 1000 * n  # + batch_nb * 10**6
            )
            guided_trajectory.append(x_hat.cpu())
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # after the last iteration no need to set this again
            if n < config.N-1:
                # in seconds
                next_timestamp = x_cond["timestamp"] + x_cond["lead_time_hours"] * 3600
                x_cond = {
                    "prev_state": x_cond["state"],
                    "state": x_hat,
                    "timestamp": next_timestamp,
                    "lead_time_hours": x_cond["lead_time_hours"],
                }
        # TODO: what is this format???
        return torch.stack(guided_trajectory, dim=1)
    
    def get_det_pred(self, x_cond):
        return self.det_model(x_cond)

    def sample(self,
        x_cond: dict,  # timestamp, TensorDict for "state", "prev", etc.
        y_n: TensorDict | None = None, 
        mask: TensorDict | None = None, 
        lambda_: list[torch.Tensor] | None = None,
        seed: int | None = None
    ):  
        # det prediction
        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred
            
            self.residual_to_pangu_scale = self.residual_to_pangu_scale.to(self.device)
            self.data_mean = self.data_mean.to(self.device)
            self.data_std = self.data_std.to(self.device)
        
        # remove next_state (save compute)
        x_cond = {k: v for k, v in x_cond.items() if "next" not in k} 
        z = self.flow(x_cond, det_pred, y_n, mask, lambda_, seed)
        # x_hat = x_det + r_hat (=sigma*z_T)
        x_hat = det_pred + tensordict_apply(torch.mul, z, self.residual_to_pangu_scale)
        return x_hat
    
    def flow(self,
        x_cond, 
        det_pred: TensorDict,
        y_n: TensorDict | None = None,  # used as guidance flag (None== no guidance)
        mask: TensorDict | None = None, 
        lambda_: list[torch.Tensor] | None = None,
        seed: int | None = None
    ):
        ##### init #####
        # draw noise
        generator = torch.Generator(device=self.device)
        if seed is not None:
            print(f"set seed: {seed}")
            generator.manual_seed(seed)
            
        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=generator)
        )
        z_t = z_t * self.scale_input_noise
    
        ##### sample #####
        timesteps = torch.linspace(self.num_train_timesteps, 1, self.T).to(self.device)
        for i in tqdm(range(len(timesteps))):
            t = timesteps[i]

            s_t = t / self.num_train_timesteps  # integration factor
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                s_next = t_next / self.num_train_timesteps
                dt = s_t - s_next
            else:
                dt = s_t

            # reset graph at each step and make z_t differentiable
            if y_n is not None:
                z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))
            
            time_embedding = self.embedd_time(x_cond, t)      
            input_state = self.get_velocity_input_state(z_t, x_cond)

            # vector field 
            with torch.no_grad():
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

            if y_n is not None:
                with torch.enable_grad():
                    sigma_z_t = tensordict_apply(torch.mul, z_t, self.residual_to_pangu_scale)
                    x_hat_norm_t = det_pred + sigma_z_t
                    x_hat_t = self.denormalize(x_hat_norm_t)
                    grad_l = self.grad_loss(x_hat_t, y_n, mask, z_t)

                u_t = tensordict_apply(
                    lambda u, g: u - (lambda_[i]) * g,
                    u_t,
                    grad_l,
                )

            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, dt)
        
        return z_t
    
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
        print(f"embedding time for gen model - month:{int(month)}, hour:{int(hour)}")

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
    
    def grad_loss(self, x_hat_t, y_n, mask, z_t):
        # in this block the shape is always "state" and type tensor_dict
        diff = x_hat_t - y_n
        masked_diff = torch.mul(mask, diff)

        # l2norm
        loss_ = torch.sum(masked_diff ** 2)

        # grad for each tensor in tensor dict
        keys = list(z_t.keys())
        tensors = [z_t[k] for k in keys]
        grads = torch.autograd.grad(
            loss_,
            tensors,
            retain_graph=False,
            create_graph=False,
        )
        grad_l = z_t.__class__(
            {k: g for k, g in zip(keys, grads)},
            batch_size=z_t.batch_size,
            device=z_t.device,
        )
        return grad_l
    
    def euler_step(self, z_t, u_t, dt):
        # z_new = z_t + h * u_t, where h = dt
        return tensordict_apply(lambda z, u: z + dt * u, z_t, u_t)