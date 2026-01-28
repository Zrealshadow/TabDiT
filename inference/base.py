"""
Base class for downstream inference tasks.

All downstream tasks (imputation, regression, classification) inherit from
BaseInference and implement task-specific input preparation and postprocessing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .repaint_sampler import RePaintSampler, RePaintConfig


@dataclass
class InferenceConfig:
    """Base configuration for inference tasks."""

    # Diffusion scheduler parameters
    num_timesteps: int = 1000
    schedule: str = "cosine"  # "linear", "cosine", "sigmoid"

    # RePaint parameters
    jump_length: int = 10
    jump_n_sample: int = 10
    use_resampling: bool = True

    # Sampling options
    clip_denoised: bool = True
    clip_range: float = 1.0

    # Normalization
    normalize_data: bool = True

    # Progress
    show_progress: bool = True

    # Device
    device: str = "cuda"


class BaseInference(ABC):
    """
    Base class for downstream inference tasks.

    Provides common functionality:
    - Model and scheduler management
    - RePaint sampling
    - Data normalization

    Subclasses implement:
    - prepare_input(): Task-specific input preparation
    - postprocess(): Task-specific output processing

    Args:
        model: Trained TabularDiffusion model
        config: InferenceConfig or None for defaults
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
    ):
        self.model = model
        self.config = config or InferenceConfig()

        # Set device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize scheduler
        self._init_scheduler()

        # Initialize RePaint sampler
        repaint_config = RePaintConfig(
            jump_length=self.config.jump_length,
            jump_n_sample=self.config.jump_n_sample,
            use_resampling=self.config.use_resampling,
            clip_denoised=self.config.clip_denoised,
            clip_range=self.config.clip_range,
            show_progress=self.config.show_progress,
        )
        self.sampler = RePaintSampler(self.scheduler, repaint_config)

        # Normalization stats (computed per call or from training data)
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None

    def _init_scheduler(self):
        """Initialize diffusion scheduler."""
        # Import here to avoid circular imports
        from train.diffusion import DiffusionScheduler

        self.scheduler = DiffusionScheduler(
            num_timesteps=self.config.num_timesteps,
            schedule=self.config.schedule,
            clip_sample=self.config.clip_denoised,
            clip_sample_range=self.config.clip_range,
        ).to(self.device)

    def set_normalization_stats(self, mean: Tensor, std: Tensor):
        """
        Set normalization statistics from training data.

        Args:
            mean: Per-feature mean [C]
            std: Per-feature std [C]
        """
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

    @abstractmethod
    def prepare_input(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Prepare input for RePaint sampling.

        Returns:
            x_known: Known values (0 where unknown) [B, N, C]
            mask: Binary mask (1=known, 0=unknown) [B, N, C]
        """
        pass

    @abstractmethod
    def postprocess(self, x_0: Tensor, *args, **kwargs) -> Any:
        """
        Postprocess RePaint output.

        Args:
            x_0: RePaint output [B, N, C]

        Returns:
            Task-specific output
        """
        pass

    def _normalize(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normalize data to zero mean and unit variance.

        Args:
            x: Input data [B, N, C]
            mask: Binary mask [B, N, C]

        Returns:
            x_norm: Normalized data
            mean: Mean used
            std: Std used
        """
        if not self.config.normalize_data:
            return x, torch.zeros(x.shape[-1], device=x.device), torch.ones(x.shape[-1], device=x.device)

        # Use provided stats or compute from known values
        if self.mean is not None and self.std is not None:
            mean, std = self.mean, self.std
        else:
            # Compute from known values only
            x_masked = x * mask
            count = mask.sum(dim=(0, 1)).clamp(min=1)
            mean = (x_masked).sum(dim=(0, 1)) / count
            var = ((x_masked - mean * mask) ** 2 * mask).sum(dim=(0, 1)) / count
            std = torch.sqrt(var + 1e-8)

        # Normalize
        x_norm = (x - mean) / std.clamp(min=1e-8)

        return x_norm, mean, std

    def _denormalize(self, x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        """
        Denormalize data back to original scale.

        Args:
            x: Normalized data [B, N, C]
            mean: Mean values [C]
            std: Std values [C]

        Returns:
            Denormalized data
        """
        return x * std + mean

    @torch.no_grad()
    def __call__(self, *args, **kwargs) -> Any:
        """
        Run inference.

        1. Prepare input (task-specific)
        2. Normalize data
        3. Run RePaint sampling
        4. Denormalize output
        5. Postprocess (task-specific)

        Returns:
            Task-specific output
        """
        # Prepare input
        x_known, mask = self.prepare_input(*args, **kwargs)

        # Move to device
        x_known = x_known.to(self.device)
        mask = mask.to(self.device)

        # Normalize
        x_known_norm, mean, std = self._normalize(x_known, mask)

        # RePaint sampling
        x_0_norm = self.sampler.sample(
            model=self.model,
            x_known=x_known_norm * mask,  # Zero out unknown values
            mask=mask,
        )

        # Denormalize
        x_0 = self._denormalize(x_0_norm, mean, std)

        # Postprocess
        return self.postprocess(x_0, *args, **kwargs)


class SimpleInference(BaseInference):
    """
    Simple inference without resampling for faster results.

    Uses simplified RePaint sampling (no jumping forward).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
    ):
        if config is None:
            config = InferenceConfig()
        config.use_resampling = False
        super().__init__(model, config)
