"""
Utility functions for inference module.

Provides helper functions for:
- Mask creation from missing values
- Data normalization/denormalization
- Schedule generation for RePaint
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import torch
from torch import Tensor


def create_mask_from_nan(x: Tensor) -> Tensor:
    """
    Create binary mask from data with NaN values.

    Args:
        x: Input tensor with NaN for missing values [B, N, C]

    Returns:
        mask: Binary mask where 1=known, 0=missing [B, N, C]
    """
    return (~torch.isnan(x)).float()


def fill_nan_with_zero(x: Tensor) -> Tensor:
    """
    Replace NaN values with zeros.

    Args:
        x: Input tensor with NaN values [B, N, C]

    Returns:
        Tensor with NaN replaced by 0.0
    """
    return torch.nan_to_num(x, nan=0.0)


def generate_repaint_schedule(
    num_timesteps: int,
    jump_length: int = 10,
    jump_n_sample: int = 10,
) -> List[Tuple[int, int]]:
    """
    Generate RePaint schedule with resampling jumps.

    The schedule includes:
    - Normal denoising steps: t -> t-1
    - Jump forward steps: t -> t+jump_length (resampling)

    At certain timesteps, we jump forward and denoise again multiple times
    to improve harmony between known and unknown regions.

    Args:
        num_timesteps: Total number of diffusion timesteps (T)
        jump_length: Number of steps to jump forward for resampling
        jump_n_sample: Number of times to resample at each jump point

    Returns:
        List of (t_prev, t_cur) tuples representing the schedule.
        If t_cur < t_prev: denoise step
        If t_cur > t_prev: resample (add noise) step

    Example:
        For num_timesteps=100, jump_length=10, jump_n_sample=2:
        At t=90: [(99,90), (90,100), (100,90), (90,100), (100,90)]
                  denoise   jump     denoise   jump     denoise
    """
    schedule = []
    t = num_timesteps - 1  # Start from T-1

    while t >= 0:
        # Determine if this is a jump point
        # Jump points are at t = 0, jump_length, 2*jump_length, ...
        is_jump_point = (t % jump_length == 0) and (t > 0)

        if is_jump_point and jump_n_sample > 1:
            # At jump points, we do multiple resample cycles
            for _ in range(jump_n_sample - 1):
                # Denoise from current to t - jump_length
                for step in range(jump_length):
                    if t - step > 0:
                        schedule.append((t - step, t - step - 1))

                # Jump forward (add noise back)
                # Go from (t - jump_length) back to t
                for step in range(jump_length):
                    if t - jump_length + step < t:
                        schedule.append((t - jump_length + step, t - jump_length + step + 1))

        # Normal denoising step
        if t > 0:
            schedule.append((t, t - 1))

        t -= 1

    return schedule


def generate_simple_schedule(num_timesteps: int) -> List[Tuple[int, int]]:
    """
    Generate simple denoising schedule without resampling.

    Args:
        num_timesteps: Total number of diffusion timesteps

    Returns:
        List of (t_prev, t_cur) tuples for denoising: T-1 -> T-2 -> ... -> 0
    """
    return [(t, t - 1) for t in range(num_timesteps - 1, 0, -1)]


def normalize_data(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Normalize data to zero mean and unit variance.

    Args:
        x: Input data [B, N, C]
        mean: Pre-computed mean (if None, computed from x)
        std: Pre-computed std (if None, computed from x)
        eps: Small value for numerical stability

    Returns:
        x_normalized: Normalized data
        mean: Mean values used
        std: Std values used
    """
    if mean is None:
        # Compute per-feature mean ignoring NaN
        mask = ~torch.isnan(x)
        x_masked = torch.where(mask, x, torch.zeros_like(x))
        mean = x_masked.sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)

    if std is None:
        # Compute per-feature std ignoring NaN
        mask = ~torch.isnan(x)
        x_centered = torch.where(mask, x - mean, torch.zeros_like(x))
        var = (x_centered ** 2).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)
        std = torch.sqrt(var + eps)

    # Normalize
    x_normalized = (x - mean) / std.clamp(min=eps)

    return x_normalized, mean, std


def denormalize_data(
    x: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """
    Denormalize data back to original scale.

    Args:
        x: Normalized data [B, N, C]
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized data
    """
    return x * std + mean


def create_icl_mask(
    n_train: int,
    n_test: int,
    n_features: int,
    target_col: int = -1,
    batch_size: int = 1,
    device: torch.device = None,
) -> Tensor:
    """
    Create mask for in-context learning (regression/classification).

    For ICL:
    - Train rows: all features known (mask=1)
    - Test rows: all features except target known (mask=1 for X, mask=0 for y)

    Args:
        n_train: Number of training rows
        n_test: Number of test rows
        n_features: Total number of features (including target)
        target_col: Index of target column (default -1 for last column)
        batch_size: Batch size
        device: Device for tensor

    Returns:
        mask: Binary mask [B, N, C] where N = n_train + n_test
    """
    N = n_train + n_test
    C = n_features

    # Initialize all as known
    mask = torch.ones(batch_size, N, C, device=device)

    # Set target column of test rows as unknown
    if target_col == -1:
        target_col = C - 1
    mask[:, n_train:, target_col] = 0.0

    return mask
