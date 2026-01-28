"""
Missing Value Pattern Generators and Dataset Wrapper.

Generates three types of missing data patterns for evaluation:
- MCAR: Missing Completely At Random
- MAR: Missing At Random (depends on observed data)
- MNAR: Missing Not At Random (depends on missing values themselves)

References:
    - Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.

Usage:
    >>> from prior.missing import MissingValueWrapper, MissingConfig, MissingType
    >>> from prior.dataset import PriorDataset
    >>>
    >>> # Wrap a dataset with MCAR missing pattern
    >>> dataset = PriorDataset(batch_size=32)
    >>> config = MissingConfig(missing_type=MissingType.MCAR, missing_rate=0.2)
    >>> wrapped = MissingValueWrapper(dataset, config)
    >>>
    >>> # Get batch with missing values
    >>> X, X_missing, mask, y, d, seq_lens, train_sizes = wrapped.get_batch()
"""

from __future__ import annotations

from typing import Optional, Tuple, Iterator, Any
from enum import Enum

from pydantic.dataclasses import dataclass
from pydantic import field_validator

import torch
import numpy as np
from torch import Tensor


class MissingType(Enum):
    """Types of missing data mechanisms."""
    MCAR = "mcar"  # Missing Completely At Random
    MAR = "mar"    # Missing At Random
    MNAR = "mnar"  # Missing Not At Random


@dataclass
class MissingConfig:
    """
    Configuration for missing pattern generation.

    Parameters
    ----------
    missing_type : MissingType
        Type of missing mechanism (MCAR, MAR, or MNAR)

    missing_rate : float
        Overall target missing rate (0.0 to 1.0)

    mar_obs_cols : int, optional
        Number of fully observed columns for MAR mechanism.
        If None, defaults to num_features // 3.

    mar_threshold_quantile : float
        For MAR: rows with observed column sum below this quantile
        have higher probability of missing values in other columns.

    mnar_threshold_quantile : float
        For MNAR: values below this quantile have higher missing probability.

    mnar_high_missing_prob : float
        For MNAR: missing probability for values below threshold.

    mnar_low_missing_prob : float
        For MNAR: missing probability for values above threshold.

    seed : int, optional
        Random seed for reproducibility.
    """
    missing_type: MissingType = MissingType.MCAR
    missing_rate: float = 0.2

    # MAR-specific parameters
    mar_obs_cols: Optional[int] = None
    mar_threshold_quantile: float = 0.5

    # MNAR-specific parameters
    mnar_threshold_quantile: float = 0.3
    mnar_high_missing_prob: float = 0.5
    mnar_low_missing_prob: float = 0.1

    seed: Optional[int] = None

    @field_validator("missing_rate")
    @classmethod
    def validate_missing_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"missing_rate must be in [0, 1], got {v}")
        return v

    @field_validator("missing_type", mode="before")
    @classmethod
    def validate_missing_type(cls, v):
        if isinstance(v, str):
            return MissingType(v)
        return v


def generate_mcar_mask(
    X: Tensor,
    missing_rate: float,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate MCAR (Missing Completely At Random) mask.

    Each value has equal probability of being missing, independent of
    both observed and unobserved data.

    Args:
        X: Input data [N, C] or [B, N, C]
        missing_rate: Probability of each value being missing
        seed: Random seed for reproducibility

    Returns:
        mask: Binary mask where 1=observed, 0=missing
    """
    if seed is not None:
        generator = torch.Generator(device=X.device).manual_seed(seed)
    else:
        generator = None

    mask = torch.ones_like(X)
    probs = torch.rand(X.shape, generator=generator, device=X.device)
    mask[probs < missing_rate] = 0.0

    return mask


def generate_mar_mask(
    X: Tensor,
    missing_rate: float,
    obs_cols: Optional[int] = None,
    threshold_quantile: float = 0.5,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate MAR (Missing At Random) mask.

    Missing values depend on observed data but not on the missing values themselves.
    Implementation: Some columns are fully observed; other columns have missing values
    that depend on the values in the observed columns.

    Args:
        X: Input data [N, C] or [B, N, C]
        missing_rate: Target overall missing rate
        obs_cols: Number of fully observed columns (default: C // 3)
        threshold_quantile: Rows with observed sum below this quantile have higher missing rate
        seed: Random seed for reproducibility

    Returns:
        mask: Binary mask where 1=observed, 0=missing
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Handle batch dimension
    if X.dim() == 2:
        X = X.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, N, C = X.shape
    device = X.device

    # Determine number of fully observed columns
    if obs_cols is None:
        obs_cols = max(1, C // 3)
    obs_cols = min(obs_cols, C - 1)  # At least 1 column must have missing values

    mask = torch.ones_like(X)

    for b in range(B):
        X_b = X[b]  # [N, C]

        # First obs_cols columns are fully observed (mask stays 1)
        # Remaining columns have MAR pattern

        # Compute sum of observed columns for each row
        obs_sum = X_b[:, :obs_cols].sum(dim=1)  # [N]

        # Rows with obs_sum below threshold have higher missing rate
        threshold = torch.quantile(obs_sum.float(), threshold_quantile)
        is_low = obs_sum < threshold  # [N]

        # Calculate missing probabilities to achieve target rate
        # Only (C - obs_cols) columns can have missing values
        # Target: missing_rate * C = actual_missing_rate * (C - obs_cols)
        adjusted_rate = (missing_rate * C) / (C - obs_cols) if C > obs_cols else missing_rate

        # Low rows have higher missing rate, high rows have lower
        high_rate = min(0.9, adjusted_rate * 1.5)
        low_rate = max(0.05, adjusted_rate * 0.5)

        for j in range(obs_cols, C):
            probs = torch.rand(N, device=device)
            missing_prob = torch.where(
                is_low,
                torch.full((N,), high_rate, device=device),
                torch.full((N,), low_rate, device=device)
            )
            mask[b, :, j] = (probs >= missing_prob).float()

    if squeeze_output:
        mask = mask.squeeze(0)

    return mask


def generate_mnar_mask(
    X: Tensor,
    threshold_quantile: float = 0.3,
    high_missing_prob: float = 0.5,
    low_missing_prob: float = 0.1,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate MNAR (Missing Not At Random) mask.

    Missing values depend on the values that would have been observed.
    Implementation: Low values (below threshold) are more likely to be missing.

    This simulates scenarios like:
    - People with low income less likely to report income
    - Patients with severe symptoms less likely to complete surveys

    Args:
        X: Input data [N, C] or [B, N, C]
        threshold_quantile: Values below this quantile have higher missing prob
        high_missing_prob: Missing probability for values below threshold
        low_missing_prob: Missing probability for values above threshold
        seed: Random seed for reproducibility

    Returns:
        mask: Binary mask where 1=observed, 0=missing
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Handle batch dimension
    if X.dim() == 2:
        X = X.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, N, C = X.shape
    device = X.device
    mask = torch.ones_like(X)

    for b in range(B):
        X_b = X[b]  # [N, C]

        for j in range(C):
            col = X_b[:, j]  # [N]
            threshold = torch.quantile(col.float(), threshold_quantile)
            is_low = col < threshold  # [N]

            probs = torch.rand(N, device=device)
            missing_prob = torch.where(
                is_low,
                torch.full((N,), high_missing_prob, device=device),
                torch.full((N,), low_missing_prob, device=device)
            )
            mask[b, :, j] = (probs >= missing_prob).float()

    if squeeze_output:
        mask = mask.squeeze(0)

    return mask


def generate_missing_mask(
    X: Tensor,
    config: MissingConfig,
) -> Tensor:
    """
    Generate missing mask based on configuration.

    Args:
        X: Input data [N, C] or [B, N, C]
        config: Missing pattern configuration

    Returns:
        mask: Binary mask where 1=observed, 0=missing
    """
    if config.missing_type == MissingType.MCAR:
        return generate_mcar_mask(
            X,
            config.missing_rate,
            config.seed
        )
    elif config.missing_type == MissingType.MAR:
        return generate_mar_mask(
            X,
            config.missing_rate,
            config.mar_obs_cols,
            config.mar_threshold_quantile,
            config.seed,
        )
    elif config.missing_type == MissingType.MNAR:
        return generate_mnar_mask(
            X,
            config.mnar_threshold_quantile,
            config.mnar_high_missing_prob,
            config.mnar_low_missing_prob,
            config.seed,
        )
    else:
        raise ValueError(f"Unknown missing type: {config.missing_type}")


def apply_missing_mask(X: Tensor, mask: Tensor) -> Tensor:
    """
    Apply missing mask to data, replacing missing values with NaN.

    Args:
        X: Input data
        mask: Binary mask (1=observed, 0=missing)

    Returns:
        X_missing: Data with NaN for missing values
    """
    X_missing = X.clone().float()
    X_missing[mask == 0] = float('nan')
    return X_missing


def compute_missing_statistics(mask: Tensor) -> dict:
    """
    Compute statistics about missing pattern.

    Args:
        mask: Binary mask (1=observed, 0=missing)

    Returns:
        Dictionary with missing statistics
    """
    total = mask.numel()
    missing = (mask == 0).sum().item()
    observed = (mask == 1).sum().item()

    # Per-column statistics
    if mask.dim() == 2:
        col_missing_rate = (mask == 0).float().mean(dim=0)
    elif mask.dim() == 3:  # [B, N, C]
        col_missing_rate = (mask == 0).float().mean(dim=(0, 1))
    else:
        col_missing_rate = torch.tensor([missing / total])

    return {
        "total_values": total,
        "missing_values": missing,
        "observed_values": observed,
        "missing_rate": missing / total if total > 0 else 0.0,
        "col_missing_rates": col_missing_rate.tolist(),
        "col_missing_rate_std": col_missing_rate.std().item() if len(col_missing_rate) > 1 else 0.0,
    }


class MissingValueWrapper:
    """
    Wrapper that applies missing value patterns to any dataset.

    Wraps a dataset and applies configured missing patterns to the features.
    Returns original data, data with missing values, and the mask.

    Parameters
    ----------
    dataset : Any
        Dataset with `get_batch()` method returning (X, y, d, seq_lens, train_sizes)

    config : MissingConfig
        Configuration for missing pattern generation

    Example
    -------
    >>> from prior.dataset import PriorDataset
    >>> from prior.missing import MissingValueWrapper, MissingConfig, MissingType
    >>>
    >>> dataset = PriorDataset(batch_size=32)
    >>> config = MissingConfig(missing_type=MissingType.MCAR, missing_rate=0.2)
    >>> wrapped = MissingValueWrapper(dataset, config)
    >>>
    >>> # Returns: X (original), X_missing (with NaN), mask, y, d, seq_lens, train_sizes
    >>> X, X_missing, mask, y, d, seq_lens, train_sizes = wrapped.get_batch()
    """

    def __init__(
        self,
        dataset: Any,
        config: MissingConfig,
    ):
        self.dataset = dataset
        self.config = config
        self._batch_count = 0

    def get_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Get a batch with missing values applied.

        Parameters
        ----------
        batch_size : int, optional
            Override batch size

        Returns
        -------
        X : Tensor
            Original complete data [B, N, C]

        X_missing : Tensor
            Data with NaN for missing values [B, N, C]

        mask : Tensor
            Binary mask (1=observed, 0=missing) [B, N, C]

        y : Tensor
            Labels [B, N]

        d : Tensor
            Number of features per table [B]

        seq_lens : Tensor
            Sequence lengths [B]

        train_sizes : Tensor
            Train/test split positions [B]
        """
        # Get batch from underlying dataset
        X, y, d, seq_lens, train_sizes = self.dataset.get_batch(batch_size)

        # Update seed for each batch if base seed is set
        config = self.config
        if config.seed is not None:
            # Create a new config with updated seed for this batch
            config = MissingConfig(
                missing_type=config.missing_type,
                missing_rate=config.missing_rate,
                mar_obs_cols=config.mar_obs_cols,
                mar_threshold_quantile=config.mar_threshold_quantile,
                mnar_threshold_quantile=config.mnar_threshold_quantile,
                mnar_high_missing_prob=config.mnar_high_missing_prob,
                mnar_low_missing_prob=config.mnar_low_missing_prob,
                seed=config.seed + self._batch_count,
            )
        self._batch_count += 1

        # Generate missing mask and apply to data
        mask = generate_missing_mask(X, config)
        X_missing = apply_missing_mask(X, mask)

        return X, X_missing, mask, y, d, seq_lens, train_sizes

    def __iter__(self) -> Iterator:
        """Returns an iterator that yields batches with missing values."""
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns the next batch with missing values applied."""
        return self.get_batch()

    def __repr__(self) -> str:
        return (
            f"MissingValueWrapper(\n"
            f"  dataset={self.dataset.__class__.__name__},\n"
            f"  missing_type={self.config.missing_type.value},\n"
            f"  missing_rate={self.config.missing_rate},\n"
            f")"
        )
