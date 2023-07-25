"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import math
from typing import Optional, Tuple

from cruise.utilities.rank_zero import rank_zero_info
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = extend_dim(scale, x.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


class UniformDistribution:
    def __init__(self, vmin: float = 0., vmax: float = 1.):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class ARVSampler(nn.Module):
    def __init__(self, net: nn.Module, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.net = net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def sample_loop(
            self, current: Tensor, num_steps: int = 20, skip_steps: int = 0, show_progress: bool = False,
            angle_schedule: str = 'linear', classifier_free_guidance: int = 1, model_kwargs: dict = {}
    ) -> Tensor:
        B, C, T = current.shape
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=B, t=T)
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        context = model_kwargs["context"]
        if context[0].shape[0] == 1 and B > 1:
            context = (context[0].repeat(B, 1), context[1].repeat(B))
        prompt = model_kwargs["prompt"]
        if prompt.shape[0] == 1 and B > 1:
            prompt = prompt.repeat(B, 1)

        sigma = 1.
        if angle_schedule == 'linear':
            angle_schedule = np.linspace(1., 2., num_steps)
            angle_schedule /= angle_schedule.sum()
        elif angle_schedule == 'uniform':
            angle_schedule = [1 / num_steps,] * num_steps

        if skip_steps > 0:
            z = current
            current = torch.randn_like(z)

        for i in progress_bar:
            sigma_i = torch.ones(B, 1, T, device=self.device) * sigma
            if skip_steps > 0 and i < skip_steps:
                sigma = sigma - angle_schedule[i]
                angle = sigma * math.pi / 2
                current = np.cos(angle) * z + np.sin(angle) * torch.randn_like(z)
                continue
            else:
                v_pred = self.net(current, sigma_i, context=context, prompt=prompt)
                if classifier_free_guidance != 1:
                    v_uncond = self.net(current, sigma_i, cfg=True, context=context, prompt=prompt)
                    guidance_scale = classifier_free_guidance
                    v_pred = guidance_scale * v_pred + (1 - guidance_scale) * v_uncond
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            current = np.cos(omega) * current - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    def sample_start(self, xT: Tensor, num_steps: int, skip_steps: int, model_kwargs: dict = {}, **kwargs) -> Tensor:
        #b, c, t = num_items, self.in_channels, model_kwargs["z_lens"][0]#, self.num_chunks * self.split_length)
        #self.net.num_chunks = min(self.num_splits, self.num_chunks)
        # Sample start
        return self.sample_loop(current=xT,
                                num_steps=num_steps,
                                skip_steps=skip_steps,
                                model_kwargs=model_kwargs,
                                **kwargs)

    @torch.no_grad()
    def forward(
        self,
        num_items: int,
        num_steps: int,
        xT: Tensor = None,
        skip_steps: int = 0,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        angle_schedule: str = 'uniform',
        classifier_free_guidance = 1,
        model_kwargs=None,
    ) -> Tensor:
        #assert_message = f"required at least {self.num_splits} chunks"
        if model_kwargs is None:
            model_kwargs = {}
        if xT is None:
            xT = torch.randn(num_items, self.in_channels, model_kwargs["z_lens"].max(), device=self.device)
        else:
            if xT.shape[0] == 1 and num_items > 1:
                xT = xT.repeat(num_items, 1, 1)

        # Sample initial chunks
        start = self.sample_start(
            xT=xT,
            num_steps=num_steps,
            skip_steps=skip_steps,
            angle_schedule=angle_schedule,
            classifier_free_guidance=classifier_free_guidance,
            model_kwargs=model_kwargs)
        return start


class Diffusion:
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(
        self,
        steps,
        num_chunks,
        chunk_length,
        training=True,
    ):
        self.compute_loss_count = 0
        self.end2end = False
        self.num_chunks = num_chunks
        self.chunk_length = chunk_length
        self.training = training
        self.num_timesteps = steps
        self.loss_mean = 1000.  # no clamp at the beginning

        if self.training:
            self.angles = UniformDistribution()
        else:
            self.angles = np.linspace(0., math.pi / 2., steps + 1, dtype=np.float32)
            self.alphas = torch.from_numpy(np.cos(self.angles)).float()
            self.deltas = torch.from_numpy(np.sin(self.angles)).float()
    
    def to(self, device):
        self.alphas = self.alphas.to(device)
        self.deltas = self.deltas.to(device)
        return self

    def q_sample(self, x0, t, et=None, return_scales=False):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(xt | x0).

        :param x0: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x0.
        """
        if et is None:
            et = torch.randn_like(x0)
        assert noise.shape == x0.shape
        if self.training:
            angles = math.pi / 2. * t
            alphas, deltas = torch.cos(angles), torch.sin(angles)
        else:
            alphas = self.alphas
            deltas = _extract_into_tensor(self.deltas, t, x0.shape)
        if return_scales:
            return alphas * x0 + deltas * et, alphas, deltas
        return alphas * x0 + deltas * et

    def p_mean_variance(
        self,
        model,
        xt,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        classifier_free_guidance=1,
    ):
        """
        Apply the model to get p(x_{t-1} | xt), as well as a prediction of
        the initial x, x0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'pred_xstart': the prediction for x0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                x = x.clamp(-1, 1)
            return x

        B, C = xt.shape[:2]
        #print(xt.shape, classifier_free_guidance, flush=True)
        vt = model(xt, (t + 1) / self.num_timesteps, mulan_cfg=True, **model_kwargs)
        if classifier_free_guidance != 1:
            guidance_scale = classifier_free_guidance
            vt_uncond = model(xt, (t + 1) / self.num_timesteps, w2v_cfg=True, mulan_cfg=True, **model_kwargs)
            vt = guidance_scale * vt + (1 - guidance_scale) * vt_uncond
        pred_xstart = process_xstart(self._predict_xstart_from_v(xt, t + 1, vt))
        #pred_epsilon = process_epsilon(self._predict_epsilon_from_v(xt, t+1, v=vt))
        model_mean = extend_dim(self.alphas[t], xt.ndim) * pred_xstart\
            + extend_dim(self.deltas[t], vt.ndim) * torch.randn_like(pred_xstart)
            #+ extend_dim(self.deltas[t], vt.ndim) * pred_epsilon
        #print(model_mean.shape, flush=True)
        rank_zero_info(t / self.num_timesteps)
        rank_zero_info(self.alphas[t])
        rank_zero_info(self.deltas[t])
        rank_zero_info("Model output: Min={} Max={} Mean={} Std={}".format(
            vt.min(), vt.max(), vt.mean(), vt.std()))
        rank_zero_info("xt: Min={} Max={} Mean={} Std={}".format(
            xt.min(), xt.max(), xt.mean(), xt.std()))
        rank_zero_info("Predict x0: Min={} Max={} Mean={} Std={}".format(
            pred_xstart.min(), pred_xstart.max(), pred_xstart.mean(), pred_xstart.std()))
        rank_zero_info("Predict mean: Min={} Max={} Mean={} Std={}".format(
            model_mean.min(), model_mean.max(), model_mean.mean(), model_mean.std()))

        return {
            "mean": model_mean,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_v(self, xt, t, vt):
        assert xt.shape == vt.shape
        return (
            extend_dim(self.alphas[t], xt.ndim) * xt - extend_dim(self.deltas[t], vt.ndim) * vt
        )

    def _predict_epsilon_from_v(self, xt, t, vt):
        assert xt.shape == vt.shape
        return (
            extend_dim(self.deltas[t], xt.ndim) * xt + extend_dim(self.alphas[t], vt.ndim) * vt
        )

    def ddim_sample(
        self,
        model,
        xt,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        classifier_free_guidance=1,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            xt,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            classifier_free_guidance=classifier_free_guidance,
        )
        return {"sample": out["mean"], "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        x_init=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        classifier_free_guidance=1,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            x_init=x_init,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            classifier_free_guidance=classifier_free_guidance,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        x_init=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        classifier_free_guidance=1,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            xt = noise
        else:
            xt = torch.randn(*shape, device=device)

        if skip_timesteps and x_init is None:
            x_init = torch.zeros_like(xt)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if x_init is not None:
            t = torch.ones([shape[0]], device=device, dtype=torch.long) * indices[0]
            xt = self.q_sample(x_init, t, et=xt)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    xt,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    classifier_free_guidance=classifier_free_guidance,
                )
                yield out
                xt = out["sample"]

    def training_losses(self, model, x0, model_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x0: the [B x D x L] tensor of inputs.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, D, L, device, dtype = *x0.shape, x0.device, x0.dtype
        et = torch.randn_like(x0)
        t = torch.rand(B, 1, 1, device=device, dtype=dtype).repeat(1, 1, L)
        #t = torch.rand((B, 1, self.num_chunks), device=device, dtype=dtype) ** 1.5
        #t = repeat(t, "b 1 n -> b 1 (n l)", l=self.chunk_length)
        # See progressive distillation (https://arxiv.org/pdf/2202.00512.pdf)
        angles = math.pi / 2. * t
        alphas, deltas = torch.cos(angles), torch.sin(angles)
        xt = alphas * x0 + deltas * et
        vt = alphas * et - deltas * x0
        context = model_kwargs["context"]
        prompt = model_kwargs["prompt"]
        z_lens = model_kwargs["z_lens"]
        vt_pred = model(xt, t, context=context, prompt=prompt)
        if self.compute_loss_count % 10 == 0:
            x0_pred = alphas * xt - deltas * vt_pred.detach()
            rank_zero_info("t: {}".format(t[:, 0, 0].view(-1)))
            rank_zero_info("alphas: {}".format(alphas[:, 0, 0].view(-1)))
            rank_zero_info("deltas: {}".format(deltas[:, 0, 0].view(-1)))
            rank_zero_info("Target Velocity: Shape={} Min={} Max={} Mean={} Std={}".format(
                vt.shape, vt.min(), vt.max(), vt.mean(), vt.std()))
            rank_zero_info("Predict Velocity: Shape={} Min={} Max={} Mean={} Std={}".format(
                vt_pred.shape, vt_pred.min(), vt_pred.max(), vt_pred.mean(), vt_pred.std()))
            rank_zero_info("Target X_start: Shape={} Min={} Max={} Mean={} Std={}".format(
                x0.shape, x0.min(), x0.max(), x0.mean(), x0.std()))
            rank_zero_info("Predict X_start: Shape={} Min={} Max={} Mean={} Std={}".format(
                x0_pred.shape, x0_pred.min(), x0_pred.max(), x0_pred.mean(), x0_pred.std()))
        terms = {}
        #is_continuous = ('mean_logvar_conv' in autoencoder.__dict__)
        #if is_continuous:
        mask = (torch.arange(max(z_lens), device=device)[None, :] < z_lens[:, None].to(device)).unsqueeze(1)
        #terms["v_mse"] = mean_flat(((vt - vt_pred) * mask) ** 2.) / mean_flat((vt * mask) ** 2.)
        #terms["x_mse"] = mean_flat(((x0 - x0_pred) * mask) ** 2.) / mean_flat((x0 * mask) ** 2.)
        terms["v_mse"] = mean_flat(((vt - vt_pred) * mask) ** 2.)
        #terms["x_mse"] = mean_flat(((x0 - x0_pred) * mask) ** 2.) / mean_flat((x0 * mask) ** 2.)
        #terms["v_l1"] = mean_flat(((vt-vt_pred) * mask).abs())
        #terms["v_mse"] = mean_flat(torch.cat(((vt - vt_pred) ** 2.).chunk(self.num_chunks, -1), 0))
        #else:
        #terms["v_mse"] = mean_flat(torch.cat(self.quant_mse(autoencoder, x0, x0_pred).chunk(self.num_chunks, -1), 0))
        terms["loss"] = terms["v_mse"]# + terms["x_mse"]
        self.compute_loss_count += 1
        return terms, t[:, 0, 0].reshape(B)
        #return terms, t[:, :, ::self.chunk_length].view(-1)

    def quant_mse(self, autoencoder, x0, x0_pred, warmup=False):
        quant_outs = []
        losses = []
        encoder_out = x0 * autoencoder.div_scale
        predict_out = x0_pred * autoencoder.div_scale

        for quant_vae in autoencoder.quant_vaes:
            quant_out = quant_vae(encoder_out)[0]
            losses.append((quant_out - predict_out) ** 2.)
            encoder_out = encoder_out - quant_out.detach()
            predict_out = predict_out - quant_out.detach()

        quant_mse = sum(losses)
        return quant_mse
