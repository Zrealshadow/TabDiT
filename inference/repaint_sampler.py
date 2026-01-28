"""
RePaint Sampler for Conditional Generation.

Implements the RePaint algorithm (CVPR 2022) for using unconditional
diffusion models for inpainting/conditional generation.

Key techniques:
1. Conditioning: At each denoising step, blend noisy ground truth into known regions
2. Resampling: Jump forward and denoise again to improve harmony

Reference:
    Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
    https://arxiv.org/abs/2201.09865
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from .utils import generate_repaint_schedule, generate_simple_schedule


@dataclass
class RePaintConfig:
    """Configuration for RePaint sampling."""

    # Resampling parameters
    jump_length: int = 10  # Steps to jump forward for resampling
    jump_n_sample: int = 10  # Number of resampling iterations

    # Sampling options
    use_resampling: bool = True  # Whether to use resampling
    clip_denoised: bool = True  # Clip x_0 predictions to [-1, 1]
    clip_range: float = 1.0  # Range for clipping

    # Progress
    show_progress: bool = True


class RePaintSampler:
    """
    RePaint sampler for conditional generation using unconditional diffusion models.

    The sampler performs inpainting by:
    1. At each timestep t, adding t-level noise to known values
    2. Blending noisy known values with generated unknown values
    3. Denoising with the model
    4. Optionally resampling (jumping forward and denoising again)

    Args:
        scheduler: DiffusionScheduler with noise schedule parameters
        config: RePaintConfig or None for defaults
    """

    def __init__(
        self,
        scheduler: nn.Module,  # DiffusionScheduler
        config: Optional[RePaintConfig] = None,
    ):
        self.scheduler = scheduler
        self.config = config or RePaintConfig()

    @property
    def num_timesteps(self) -> int:
        return self.scheduler.num_timesteps

    @property
    def device(self) -> torch.device:
        return self.scheduler.betas.device

    def _get_schedule(self) -> List[Tuple[int, int]]:
        """Get the sampling schedule."""
        if self.config.use_resampling:
            return generate_repaint_schedule(
                self.num_timesteps,
                self.config.jump_length,
                self.config.jump_n_sample,
            )
        else:
            return generate_simple_schedule(self.num_timesteps)

    def _add_noise_to_known(
        self,
        x_known: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Add t-level noise to known values.

        x_t^known = sqrt(alpha_bar_t) * x_known + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_known: Clean known values [B, N, C]
            t: Timesteps [B]
            noise: Optional pre-sampled noise

        Returns:
            Noisy known values at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_known)

        return self.scheduler.add_noise(x_known, noise, t)

    def _denoise_step(
        self,
        model: nn.Module,
        x_t: Tensor,
        x_known: Tensor,
        mask: Tensor,
        t: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Single denoising step with conditioning.

        1. Add t-level noise to known values
        2. Blend: x_t_input = mask * x_t_known + (1-mask) * x_t
        3. Denoise with model
        4. Return x_{t-1}

        Args:
            model: Denoising model
            x_t: Current noisy sample [B, N, C]
            x_known: Clean known values [B, N, C]
            mask: Binary mask (1=known, 0=unknown) [B, N, C]
            t: Current timestep (scalar)
            generator: Optional random generator

        Returns:
            x_{t-1}: Denoised sample
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Add t-level noise to known values
        noise_known = torch.randn(x_known.shape, generator=generator, device=device, dtype=x_known.dtype)
        x_t_known = self._add_noise_to_known(x_known, t_tensor, noise_known)

        # Blend known and unknown regions
        # x_t_input = mask * x_t_known + (1 - mask) * x_t
        x_t_input = mask * x_t_known + (1.0 - mask) * x_t

        # Get model prediction (noise)
        with torch.no_grad():
            noise_pred = model(x_t_input, t_tensor)

        # Take denoising step using scheduler
        x_t_minus_1 = self.scheduler.step(noise_pred, t_tensor, x_t_input, generator=generator)

        return x_t_minus_1

    def _resample_step(
        self,
        x_t: Tensor,
        t_from: int,
        t_to: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Resample step: add noise to jump forward.

        x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * noise

        Args:
            x_t: Current sample [B, N, C]
            t_from: Source timestep
            t_to: Target timestep (t_to > t_from)
            generator: Optional random generator

        Returns:
            x_t at target timestep
        """
        device = x_t.device

        # Add noise step by step
        x = x_t
        for t in range(t_from + 1, t_to + 1):
            # Get beta_t
            beta_t = self.scheduler.betas[t]

            # x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * noise
            noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
            x = torch.sqrt(1.0 - beta_t) * x + torch.sqrt(beta_t) * noise

        return x

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_known: Tensor,
        mask: Tensor,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, Tensor], None]] = None,
    ) -> Tensor:
        """
        RePaint sampling for conditional generation.

        Args:
            model: Trained denoising model that takes (x_t, t) -> noise prediction
            x_known: Known values (0 where unknown) [B, N, C]
            mask: Binary mask (1=known, 0=unknown) [B, N, C]
            generator: Optional random generator for reproducibility
            callback: Optional callback(step, x_t) called at each step

        Returns:
            x_0: Final sample with unknown regions filled [B, N, C]
        """
        device = x_known.device
        shape = x_known.shape

        # Initialize from pure noise
        x_t = torch.randn(shape, generator=generator, device=device, dtype=x_known.dtype)

        # Get schedule
        schedule = self._get_schedule()

        # Progress bar
        pbar = tqdm(schedule, desc="RePaint sampling", disable=not self.config.show_progress)

        for t_prev, t_cur in pbar:
            if t_cur < t_prev:
                # Denoise step: t_prev -> t_cur
                x_t = self._denoise_step(
                    model=model,
                    x_t=x_t,
                    x_known=x_known,
                    mask=mask,
                    t=t_prev,
                    generator=generator,
                )
            else:
                # Resample step: t_prev -> t_cur (jump forward)
                x_t = self._resample_step(
                    x_t=x_t,
                    t_from=t_prev,
                    t_to=t_cur,
                    generator=generator,
                )

            if callback is not None:
                callback(t_cur, x_t)

        # Final output: blend known values with generated unknown values
        # This ensures known regions are exactly preserved
        x_0 = mask * x_known + (1.0 - mask) * x_t

        return x_0

    @torch.no_grad()
    def sample_simple(
        self,
        model: nn.Module,
        x_known: Tensor,
        mask: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Simplified RePaint sampling without resampling.

        Faster but may produce less harmonious results.

        Args:
            model: Trained denoising model
            x_known: Known values [B, N, C]
            mask: Binary mask [B, N, C]
            generator: Optional random generator

        Returns:
            x_0: Final sample
        """
        device = x_known.device
        shape = x_known.shape

        # Initialize from pure noise
        x_t = torch.randn(shape, generator=generator, device=device, dtype=x_known.dtype)

        # Simple reverse diffusion with conditioning
        for t in tqdm(
            reversed(range(self.num_timesteps)),
            desc="Sampling",
            disable=not self.config.show_progress,
        ):
            x_t = self._denoise_step(
                model=model,
                x_t=x_t,
                x_known=x_known,
                mask=mask,
                t=t,
                generator=generator,
            )

        # Final blend
        x_0 = mask * x_known + (1.0 - mask) * x_t

        return x_0
