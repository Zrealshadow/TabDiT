"""
Diffusion Schedule and Sampling

Implements noise schedules for diffusion models:
- Linear beta schedule
- Cosine beta schedule (improved)
- DDPM forward process (adding noise)
- DDPM sampling (denoising)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tensor:
    """Linear noise schedule from DDPM paper.

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion timesteps
    beta_start : float
        Starting beta value
    beta_end : float
        Ending beta value

    Returns
    -------
    Tensor
        Beta values for each timestep [T]
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
) -> Tensor:
    """Cosine noise schedule from "Improved DDPM" paper.

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion timesteps
    s : float
        Small offset to prevent beta from being too small at t=0

    Returns
    -------
    Tensor
        Beta values for each timestep [T]
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps

    # f(t) = cos((t + s) / (1 + s) * pi / 2)^2
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Compute betas from cumulative alphas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tensor:
    """Sigmoid noise schedule.

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion timesteps
    beta_start : float
        Starting beta value
    beta_end : float
        Ending beta value

    Returns
    -------
    Tensor
        Beta values for each timestep [T]
    """
    betas = torch.linspace(-6, 6, num_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DiffusionScheduler(nn.Module):
    """
    DDPM Diffusion Scheduler.

    Handles:
    - Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    - Adding noise at arbitrary timesteps
    - Computing loss weights
    - DDPM sampling (reverse process)

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion timesteps
    schedule : str
        Noise schedule type: 'linear', 'cosine', or 'sigmoid'
    beta_start : float
        Starting beta value (for linear/sigmoid schedules)
    beta_end : float
        Ending beta value (for linear/sigmoid schedules)
    clip_sample : bool
        Whether to clip samples during denoising
    clip_sample_range : float
        Range for clipping samples [-range, range]
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Get beta schedule
        if schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        elif schedule == "sigmoid":
            betas = sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Compute derived quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register buffers (moved to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For forward process: q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # For posterior: q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        # Posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # For converting noise prediction to x_0 prediction
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def _extract(self, coeff: Tensor, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
        """Extract coefficients for given timesteps and reshape for broadcasting.

        Parameters
        ----------
        coeff : Tensor
            Coefficient tensor [T]
        t : Tensor
            Timesteps [B]
        x_shape : Tuple[int, ...]
            Shape of x for broadcasting

        Returns
        -------
        Tensor
            Extracted coefficients with shape [B, 1, 1, ...]
        """
        batch_size = t.shape[0]
        out = coeff[t]

        # Reshape for broadcasting: [B] -> [B, 1, 1, ...]
        return out.view(batch_size, *([1] * (len(x_shape) - 1)))

    def add_noise(
        self,
        x_0: Tensor,
        noise: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Forward diffusion process: q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Parameters
        ----------
        x_0 : Tensor
            Original clean data [B, ...]
        noise : Tensor
            Gaussian noise [B, ...]
        t : Tensor
            Timesteps [B]

        Returns
        -------
        Tensor
            Noisy data x_t [B, ...]
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

    def predict_x0_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)

        Parameters
        ----------
        x_t : Tensor
            Noisy data [B, ...]
        t : Tensor
            Timesteps [B]
        noise : Tensor
            Predicted noise [B, ...]

        Returns
        -------
        Tensor
            Predicted x_0 [B, ...]
        """
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def posterior_mean(
        self,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute posterior mean: q(x_{t-1} | x_t, x_0).

        Parameters
        ----------
        x_0 : Tensor
            Predicted clean data [B, ...]
        x_t : Tensor
            Noisy data [B, ...]
        t : Tensor
            Timesteps [B]

        Returns
        -------
        Tensor
            Posterior mean [B, ...]
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)

        return coef1 * x_0 + coef2 * x_t

    @torch.no_grad()
    def step(
        self,
        model_output: Tensor,
        t: Tensor,
        x_t: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Single denoising step: p(x_{t-1} | x_t).

        Parameters
        ----------
        model_output : Tensor
            Model's noise prediction [B, ...]
        t : Tensor
            Current timesteps [B]
        x_t : Tensor
            Current noisy samples [B, ...]
        generator : torch.Generator, optional
            Random generator for reproducibility

        Returns
        -------
        Tensor
            Denoised samples x_{t-1} [B, ...]
        """
        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, model_output)

        # Optionally clip x_0 prediction
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -self.clip_sample_range, self.clip_sample_range)

        # Compute posterior mean
        mean = self.posterior_mean(x_0_pred, x_t, t)

        # Get posterior variance
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # Sample from posterior (no noise at t=0)
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        """Generate samples using DDPM sampling.

        Parameters
        ----------
        model : nn.Module
            Denoising model that takes (x_t, t) and returns noise prediction
        shape : Tuple[int, ...]
            Shape of samples to generate [B, N, C]
        device : torch.device
            Device to generate samples on
        generator : torch.Generator, optional
            Random generator for reproducibility
        return_intermediates : bool
            Whether to return all intermediate samples

        Returns
        -------
        Tensor
            Generated samples [B, N, C] or list of intermediates
        """
        batch_size = shape[0]

        # Start from pure noise
        x_t = torch.randn(shape, generator=generator, device=device)

        intermediates = [x_t] if return_intermediates else None

        # Denoise step by step
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Get model prediction
            noise_pred = model(x_t, t)

            # Take denoising step
            x_t = self.step(noise_pred, t, x_t, generator=generator)

            if return_intermediates:
                intermediates.append(x_t)

        return intermediates if return_intermediates else x_t

    def get_loss_weight(self, t: Tensor) -> Tensor:
        """Get loss weighting for given timesteps.

        Uses SNR (signal-to-noise ratio) weighting by default.

        Parameters
        ----------
        t : Tensor
            Timesteps [B]

        Returns
        -------
        Tensor
            Loss weights [B]
        """
        # Simple uniform weighting (can be modified)
        return torch.ones_like(t, dtype=torch.float)


class DDIMScheduler(DiffusionScheduler):
    """
    DDIM (Denoising Diffusion Implicit Models) Scheduler.

    Allows for faster sampling with fewer steps while maintaining quality.
    Inherits from DiffusionScheduler and overrides sampling methods.

    Parameters
    ----------
    num_timesteps : int
        Number of training timesteps
    num_inference_steps : int
        Number of steps for inference (can be much smaller)
    eta : float
        Controls stochasticity. eta=0 is deterministic, eta=1 is DDPM-like
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Compute inference timesteps (evenly spaced)
        step_ratio = num_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.register_buffer("inference_timesteps", timesteps.flip(0))

    @torch.no_grad()
    def ddim_step(
        self,
        model_output: Tensor,
        t: int,
        t_prev: int,
        x_t: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Single DDIM denoising step.

        Parameters
        ----------
        model_output : Tensor
            Model's noise prediction [B, ...]
        t : int
            Current timestep
        t_prev : int
            Previous timestep
        x_t : Tensor
            Current noisy samples [B, ...]
        generator : torch.Generator, optional
            Random generator

        Returns
        -------
        Tensor
            Denoised samples [B, ...]
        """
        # Get alpha values
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)

        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -self.clip_sample_range, self.clip_sample_range)

        # Compute sigma
        sigma = self.eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        # Compute direction pointing to x_t
        pred_direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * model_output

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + pred_direction

        # Add noise if eta > 0
        if self.eta > 0:
            noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        """Generate samples using DDIM sampling.

        Parameters
        ----------
        model : nn.Module
            Denoising model
        shape : Tuple[int, ...]
            Shape of samples to generate
        device : torch.device
            Device to generate on
        generator : torch.Generator, optional
            Random generator
        return_intermediates : bool
            Whether to return intermediates

        Returns
        -------
        Tensor
            Generated samples
        """
        batch_size = shape[0]

        # Start from pure noise
        x_t = torch.randn(shape, generator=generator, device=device)

        intermediates = [x_t] if return_intermediates else None
        timesteps = self.inference_timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)

            # Get model prediction
            noise_pred = model(x_t, t_tensor)

            # Get previous timestep
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1

            # Take DDIM step
            x_t = self.ddim_step(noise_pred, t.item(), t_prev, x_t, generator=generator)

            if return_intermediates:
                intermediates.append(x_t)

        return intermediates if return_intermediates else x_t
