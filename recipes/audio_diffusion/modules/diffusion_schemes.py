import math

import torch


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


class EDMScheme:  # pragma: no cover
    def __init__(self, sigma_data=0.5, P_mean=-1.2, P_std=1.2):
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std

    def train(self, reals, model, **kwargs):
        rnd_normal = torch.randn(
            [reals.shape[0]] + [1] * (len(reals.shape) - 1), device=reals.device
        )
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma.square() + self.sigma_data**2) / (
            sigma * self.sigma_data
        ).square()
        n = torch.randn_like(reals) * sigma
        denoised = self._calculate_preconditioned_model_output(
            reals + n, sigma, model, **kwargs
        )
        loss = torch.log10((weight * ((denoised - reals).square())).mean())
        return loss

    def sample(
        self,
        initial_noise,
        model,
        guidance_model=None,
        num_steps=50,
        rho=7,
        S_churn=0.02,
        S_min=0.001,
        S_max=float("inf"),
        S_noise=1.07,
        sigma_min=0.002,
        sigma_max=80,
        **kwargs,
    ):
        x = initial_noise
        device = x.device

        t_steps = torch.arange(num_steps, device=device) / (num_steps - 1)
        t_steps = (
            sigma_max ** (1 / rho)
            + t_steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1], device=device)])

        t_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)

        x_next = x * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = (
                min(S_churn / num_steps, math.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur, device=x.device)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur, device=device)

            derivative = self._calculate_preconditioned_model_derivative(
                x_hat, t_hat.expand(t_shape), model, guidance_model, **kwargs
            )
            d_cur = derivative / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < num_steps - 1:
                derivative = self._calculate_preconditioned_model_derivative(
                    x_next, t_next.expand(t_shape), model, guidance_model, **kwargs
                )
                d_prime = derivative / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def _calculate_preconditioned_model_output(self, x, sigma, model, **kwargs):
        sigma_data_squared = self.sigma_data**2
        sigma_squared = sigma.square()

        c_skip = sigma_data_squared / (sigma_squared + sigma_data_squared)
        c_out = sigma * self.sigma_data / (sigma_squared + sigma_data_squared).sqrt()
        c_in = 1 / (sigma_data_squared + sigma_squared).sqrt()
        c_noise = sigma.log() / 4

        F_x = model(c_in * x, c_noise, **kwargs)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def _calculate_preconditioned_model_derivative(
        self, x, sigma, model, guidance_model=None, **kwargs
    ):
        if guidance_model is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                D_x = self._calculate_preconditioned_model_output(
                    x, sigma, model, **kwargs
                )
                guidance_derivative = guidance_model(x, D_x)
                derivative = x.detach() - D_x.detach() + guidance_derivative.detach()
        else:
            D_x = self._calculate_preconditioned_model_output(x, sigma, model, **kwargs)
            derivative = x - D_x
        return derivative


class VDiffusionScheme:  # pragma: no cover
    def _get_alpha_beta(self, sigmas):
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def train(self, x, model, text_cond=None, conditioning_dropout_rate=0.1):
        rnd_uniform = torch.rand(
            [x.shape[0]] + [1] * (len(x.shape) - 1), device=x.device
        )
        sigmas = rnd_uniform
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self._get_alpha_beta(sigmas)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        if text_cond is not None:
            if model.training:
                uncond = torch.zeros_like(text_cond)
                dropout_mask = (
                    torch.rand(text_cond.shape[0], device=text_cond.device)
                    > conditioning_dropout_rate
                ).view(-1, 1, 1)
                text_cond = torch.where(dropout_mask, text_cond, uncond)
            v_pred = model(x_noisy, sigmas, text_cond)
        else:
            v_pred = model(x_noisy, sigmas)
        loss = torch.log10((((v_pred - v_target).square())).mean())
        return loss

    def sample(
        self,
        initial_noise,
        model,
        num_steps=100,
        text_cond=None,
        guidance_scale=7.0,
        **kwargs,
    ):
        x = initial_noise
        t_steps = 1.0 - torch.arange(num_steps, device=x.device) / (num_steps - 1)

        t_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)

        alphas, betas = self._get_alpha_beta(t_steps)

        for i in range(num_steps - 1):
            if text_cond is not None:
                uncond_input = torch.zeros_like(text_cond)
                cond_output = model(x, t_steps[i].expand(t_shape), text_cond)
                uncond_output = model(x, t_steps[i].expand(t_shape), uncond_input)
                v_pred = (
                    guidance_scale * cond_output
                    + (1.0 - guidance_scale) * uncond_output
                )
            else:
                v_pred = model(x, t_steps[i].expand(t_shape))
            x_pred = alphas[i] * x - betas[i] * v_pred
            noise_pred = betas[i] * x + alphas[i] * v_pred
            x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return x

    @torch.no_grad()
    def sample_dpm_2_ancestral(
        self,
        x,
        model,
        guidance_scale=7.0,
        eta=1.0,
        s_noise=1.0,
        num_steps=100,
        **kwargs,
    ):
        """Ancestral sampling with DPM-Solver second-order steps."""
        sigmas = 1.0 - torch.arange(num_steps, device=x.device) / (num_steps - 1)
        sigmas_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
        noise_sampler = default_noise_sampler(x)
        for i in range(len(sigmas) - 1):
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            d = self._calculate_model_derivative(
                x, sigmas[i].expand(sigmas_shape), model, **kwargs
            )
            if sigma_down == 0:
                # Euler method
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                x_2 = x + d * dt_1
                d_2 = self._calculate_model_derivative(
                    x_2, sigma_mid.expand(sigmas_shape), model, **kwargs
                )
                x = x + d_2 * dt_2
                if sigmas[i + 1] > 0.0:
                    x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        return x

    @torch.no_grad()
    def sample_heun(
        self,
        x,
        model,
        guidance_scale=7.0,
        num_steps=100,
        s_churn=0.2,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        **kwargs,
    ):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
        sigmas = 1.0 - torch.arange(num_steps, device=x.device) / (num_steps - 1)
        sigmas_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)

        for i in range(len(sigmas) - 1):
            gamma = (
                min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
                if s_tmin <= sigmas[i] <= s_tmax
                else 0.0
            )
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            d = self._calculate_model_derivative(
                x, sigma_hat.expand(sigmas_shape), model, **kwargs
            )
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                d_2 = self._calculate_model_derivative(
                    x_2, sigmas[i + 1].expand(sigmas_shape), model, **kwargs
                )
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x

    def _calculate_model_derivative(
        self, x, sigma, model, text_cond=None, guidance_scale=7.0, **kwargs
    ):
        if text_cond is not None:
            uncond_input = torch.zeros_like(text_cond)
            cond_output = model(x, sigma, text_cond)
            uncond_output = model(x, sigma, uncond_input)
            v_pred = (
                guidance_scale * cond_output + (1.0 - guidance_scale) * uncond_output
            )
        else:
            v_pred = model(x, sigma)
        return v_pred
