import importlib
import pandas as pd
import torch

from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm import tqdm

import geoarches.stats as geoarches_stats
from geoarches.backbones.dit import TimestepEmbedder
from geoarches.lightning_modules import BaseLightningModule
from geoarches.lightning_modules.base_module import AvgModule, load_module
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

geoarches_stats_path = importlib.resources.files(geoarches_stats)


class GuidedFlow(BaseLightningModule):
    """
    Rollout, sample, and guide predictions (det+gen).
    """
    def __init__(
        self,
        cfg,
        load_deterministic_model=None
    ):
        super().__init__()
        self.__dict__.update(locals())
        
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
        print("self.device", self.device)
        self.det_model = self.det_model.to(self.device)

        # TODO: can I put this in a pipeline too?
        self.month_embedder = TimestepEmbedder(self.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cond_dim)
        self.timestep_embedder = TimestepEmbedder(self.cond_dim)

        # sigma scaling factor (shape like usual states)
        pangu_stats = torch.load(
            geoarches_stats_path / "pangu_norm_stats2_with_w.pt", weights_only=True
        )
        pangu_scaler = TensorDict(
            level=pangu_stats["level_std"], surface=pangu_stats["surface_std"]
        )
        scaler = TensorDict(
            **torch.load(
                geoarches_stats_path / "deltapred24_aws_denorm.pt", weights_only=False
            )
        )
        scaler["level"][-1] *= 3  # we don't care too much about vertical velocity
        self.sigma = scaler / pangu_scaler  # inverse because we divide by state_scaler

    def rollout(
        self, 
        N, 
        x_cond,  # TODO: need a better name
        x_guide_trajectory: list[torch.Tensor] | None = None, # shape: 
        mask: torch.Tensor | None = None,  # shape: 
    ):
        realized_trajectory = []

        for n in range(N):
            x_guide = None if x_guide_trajectory is None else x_guide_trajectory[n]
            x_hat = self.rollout_step(
                x_cond=x_cond,
                y_t=x_guide,
                mask=mask,
            )
            realized_trajectory.append(x_hat)

            if n < N - 1:
                x_cond = tensordict_cat([x_cond["state"], x_hat], dim=1)

        return realized_trajectory

    def rollout_step(self,
        x_cond, y_t, mask
    ):  
        # det prediction
        with torch.no_grad():
            x_cond["pred_state"] = self.det_model(x_cond).detach()
            # TODO: should place this somewhere else
            self.sigma = self.sigma.to(self.det_model.device)
        
        # TODO: might remove this (it's redundant for now)
        mu = x_cond["pred_state"]
        # remove next_state (save compute)
        loop_batch = {k: v for k, v in x_cond.items() if "next" not in k} 
        z = self.sample(loop_batch, mu, y_t, mask).detach()
        # x_hat = x_det + r_hat (=sigma*z_T)
        x_hat = mu + tensordict_apply(torch.mul, z, self.sigma)
        return x_hat
    
    # TODO: do not use batch, separate object for clarity 
    #       also the name is utter bs
    def sample(self, batch, mu, y_t, mask):
        
        ##### init #####

        # draw noise
        generator = torch.Generator(device=batch["timestamp"].device)
        z = batch["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=generator)
        )
        # reason explained in paper
        # TODO: changed name ones we get rid of the name "sigma" in other contexts!
        z_t = z * self.scale_input_noise

        lambdas = [0.1 * self.T]  # TODO: should input a time dependent lambda list to guided_sampling
        w = 1
    
        ##### sample #####

        with torch.no_grad():
            for t in tqdm(range(1, self.T + 1)):
                # denoiser is composed of encoder-backbone-decoder
                # I must do some torch pipeline object
                time_embedding = self.embedd_time(batch, t)
                input_state = self.get_velocity_input_state(z_t, batch)
                u_t = self.velocity(batch, time_embedding, input_state, z_t, t)
                # TODO: need to decide how to pass the grad in case I need it
                grad_l = self.grad_loss(mask, mu, y_t, z_t)
                u_t_guided = u_t -  lambdas[0] * w * grad_l
                z_t = self.euler_step(z_t, u_t_guided)

        return z_t

        ##### compute final output #####
    
    def embedd_time(self, batch, t):
        times = pd.to_datetime(batch["timestamp"].cpu().numpy() * 10**9).tz_localize(None)
        month = torch.tensor(times.month).to(batch["timestamp"].device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor(times.hour).to(batch["timestamp"].device)
        hour_emb = self.hour_embedder(hour)
        timestep_emb = self.timestep_embedder(torch.tensor([t]).to(batch["timestamp"].device))

        time_embedding = month_emb + hour_emb + timestep_emb
        return time_embedding
    
    def get_velocity_input_state(self, z, batch):
        input_state = z  # only init of concat (we need z later as is)
        assert "pred_state" in batch
        pred_state = batch["pred_state"]
        prev_state = batch["prev_state"]
        input_state = tensordict_cat([prev_state, input_state], dim=1)
        input_state = tensordict_cat([pred_state, input_state], dim=1)
        return input_state

    def velocity(self, batch, time_embedding, input_state, z, t):

        ##### compute residual #####

        # TODO: can I put this into a torch pipeline object?
        #       may need to compute the grad through this object
        # here we embedd prev_state (input_state[0]), current_state (batch["state"]), noisy_state (input_state[1])
        x = self.embedder.encode(batch["state"], input_state)
        x = self.backbone(x, time_embedding)
        r_t = self.embedder.decode(x)  # we get tdict

        ##### compute velocity from residual

        # u_t = r_t - eps_t := (r_t - z_t) / s_t
        # TODO: check that this works in the forwards case ...
        s_t = t / self.T  # e.g. 959.5 / 1000
        # TODO: check that broadcasts correctly
        # s_t = s_t[:, None, None, None, None]  # shape (batch_size,1,1,1,1), to divide elementwise
        s_t = torch.tensor(t / self.T, device=batch["state"].device, dtype=torch.float32)
        u_t = (r_t - z).apply(lambda x: x / s_t)
        return u_t
    
    def grad_loss(self, mask, mu, y_t, z_t):
        # TODO: select tensor from inside tensor_dict to compute this ...
        #       need to define guiding variable and partition somewhere too!
        # TODO: can place it in helpers instead of class
        # TODO: sigma needs probably same treatment as in the translation ...
        # TODO: check that * does pointwise multiplication
        print(mask.device, mu.device, z_t.device, self.sigma.device)
        # loss_ = torch.square(
        #     y_t - torch.sum(mask * (mu + self.sigma * z_t)) / torch.sum(mask)
        # )
        return torch.tensor(0.0)

        def grad(loss, z_t):
            # TODO: can place it in helpers instead of clas
            return torch.tensor(0.0)

        grad_l = grad(loss_, z_t)
        return grad_l 
    
    def euler_step(self, z_t, u_t):
        h = 1 / self.T
        # z_new = z_t + h * u_t
        # the lambda corresponds to def f(z, u): return z + h * u, the rest are the arguments
        return tensordict_apply(lambda z, u: z + h * u, z_t, u_t)