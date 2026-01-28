"""
Missing Value Imputation using RePaint Algorithm.

Imputes missing values in tabular data by treating it as an inpainting problem:
- Known values (non-NaN): mask = 1
- Missing values (NaN): mask = 0

The RePaint algorithm fills in missing values by conditioning on known values
at each denoising step.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from .base import BaseInference, InferenceConfig
from .utils import create_mask_from_nan, fill_nan_with_zero


@dataclass
class ImputationConfig(InferenceConfig):
    """Configuration for imputation task."""

    # Imputation-specific options
    preserve_known: bool = True  # Exactly preserve known values in output
    return_uncertainty: bool = False  # Return uncertainty estimates


class Imputer(BaseInference):
    """
    Missing value imputation using RePaint algorithm.

    Uses the unconditional diffusion model to fill in missing values (NaN)
    by conditioning on known values at each denoising step.

    Example:
        >>> model = TabularDiffusion.load("checkpoint.pt")
        >>> imputer = Imputer(model)
        >>>
        >>> # Data with missing values
        >>> x = torch.tensor([[1.0, float('nan'), 3.0],
        ...                   [4.0, 5.0, float('nan')]])
        >>>
        >>> # Impute missing values
        >>> x_imputed = imputer(x)

    Args:
        model: Trained TabularDiffusion model
        config: ImputationConfig or None for defaults
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ImputationConfig] = None,
    ):
        if config is None:
            config = ImputationConfig()
        super().__init__(model, config)
        self.imputation_config = config

    def prepare_input(
        self,
        x: Union[Tensor, np.ndarray],
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare input for imputation.

        Args:
            x: Input data with NaN for missing values [B, N, C] or [N, C]

        Returns:
            x_known: Data with NaN replaced by 0 [B, N, C]
            mask: Binary mask (1=known, 0=missing) [B, N, C]
        """
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Create mask from NaN values
        mask = create_mask_from_nan(x)

        # Replace NaN with 0
        x_known = fill_nan_with_zero(x)

        return x_known, mask

    def postprocess(
        self,
        x_0: Tensor,
        x: Union[Tensor, np.ndarray],
    ) -> Tensor:
        """
        Postprocess imputed output.

        Args:
            x_0: RePaint output [B, N, C]
            x: Original input with NaN [B, N, C] or [N, C]

        Returns:
            x_imputed: Data with missing values filled [B, N, C]
        """
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.to(x_0.device)

        if self.imputation_config.preserve_known:
            # Preserve original known values exactly
            mask = create_mask_from_nan(x)
            x_known = fill_nan_with_zero(x)
            x_imputed = mask * x_known + (1.0 - mask) * x_0
        else:
            x_imputed = x_0

        return x_imputed

    def impute(
        self,
        x: Union[Tensor, np.ndarray],
        return_numpy: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        """
        Impute missing values in data.

        Convenience method that handles input/output conversion.

        Args:
            x: Input data with NaN for missing values [B, N, C] or [N, C]
            return_numpy: If True, return numpy array

        Returns:
            x_imputed: Data with missing values filled
        """
        # Run inference
        x_imputed = self(x)

        # Remove batch dimension if input didn't have it
        input_was_2d = (isinstance(x, np.ndarray) and x.ndim == 2) or \
                       (isinstance(x, Tensor) and x.dim() == 2)
        if input_was_2d:
            x_imputed = x_imputed.squeeze(0)

        # Convert to numpy if requested
        if return_numpy:
            x_imputed = x_imputed.cpu().numpy()

        return x_imputed

    def impute_with_uncertainty(
        self,
        x: Union[Tensor, np.ndarray],
        n_samples: int = 10,
        return_numpy: bool = False,
    ) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
        """
        Impute missing values with uncertainty estimation.

        Runs multiple imputations and returns mean and std.

        Args:
            x: Input data with NaN for missing values [B, N, C] or [N, C]
            n_samples: Number of imputation samples
            return_numpy: If True, return numpy arrays

        Returns:
            mean: Mean of imputed samples
            std: Standard deviation (uncertainty) of imputed samples
        """
        samples = []

        for _ in range(n_samples):
            x_imputed = self(x)
            samples.append(x_imputed)

        # Stack samples: [n_samples, B, N, C]
        samples = torch.stack(samples, dim=0)

        # Compute mean and std
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        # Remove batch dimension if input didn't have it
        input_was_2d = (isinstance(x, np.ndarray) and x.ndim == 2) or \
                       (isinstance(x, Tensor) and x.dim() == 2)
        if input_was_2d:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        # Convert to numpy if requested
        if return_numpy:
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()

        return mean, std


class FastImputer(Imputer):
    """
    Fast imputation without resampling.

    Uses simplified RePaint sampling (no jumping forward) for faster results.
    May produce less harmonious imputations but is significantly faster.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ImputationConfig] = None,
    ):
        if config is None:
            config = ImputationConfig()
        config.use_resampling = False
        super().__init__(model, config)
