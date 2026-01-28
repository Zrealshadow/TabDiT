"""
Regression In-Context Learning using RePaint Algorithm.

Uses the unconditional diffusion model for regression by framing it as inpainting:
- Train rows: all features (X, y) are known → mask = 1
- Test rows: features X are known, target y is unknown → mask = 0 for y

The model learns to predict test y given the context of training examples.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from .base import BaseInference, InferenceConfig
from .utils import create_icl_mask


@dataclass
class RegressionICLConfig(InferenceConfig):
    """Configuration for regression ICL task."""

    # ICL-specific options
    target_col: int = -1  # Index of target column (default: last column)

    # Ensemble predictions
    n_ensemble: int = 1  # Number of ensemble predictions (for uncertainty)


class RegressionICL(BaseInference):
    """
    Regression In-Context Learning using RePaint algorithm.

    Given training examples (X_train, y_train) as context, predicts y_test
    for test examples X_test by treating it as an inpainting problem.

    The input is constructed as:
        [X_train | y_train]   <- Train rows (all known)
        [X_test  | ?      ]   <- Test rows (y unknown)

    The model fills in the unknown y_test values conditioned on the context.

    Example:
        >>> model = TabularDiffusion.load("checkpoint.pt")
        >>> regressor = RegressionICL(model)
        >>>
        >>> # Training data (context)
        >>> X_train = torch.randn(80, 10)  # 80 samples, 10 features
        >>> y_train = torch.randn(80, 1)   # 80 targets
        >>>
        >>> # Test data
        >>> X_test = torch.randn(20, 10)   # 20 samples to predict
        >>>
        >>> # Predict
        >>> y_pred = regressor.predict(X_train, y_train, X_test)

    Args:
        model: Trained TabularDiffusion model
        config: RegressionICLConfig or None for defaults
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[RegressionICLConfig] = None,
    ):
        if config is None:
            config = RegressionICLConfig()
        super().__init__(model, config)
        self.icl_config = config

    def prepare_input(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare input for regression ICL.

        Constructs the input tensor by stacking train and test data,
        with target column masked for test rows.

        Args:
            X_train: Training features [n_train, n_features] or [B, n_train, n_features]
            y_train: Training targets [n_train] or [n_train, 1] or [B, n_train, 1]
            X_test: Test features [n_test, n_features] or [B, n_test, n_features]

        Returns:
            x_known: Combined data [B, N, C] where N = n_train + n_test
            mask: Binary mask [B, N, C] with 0 for test targets
        """
        # Convert numpy to tensor if needed
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()

        # Handle dimensions
        # X_train: [n_train, n_features] -> [1, n_train, n_features]
        if X_train.dim() == 2:
            X_train = X_train.unsqueeze(0)
        # y_train: [n_train] -> [1, n_train, 1]
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(0).unsqueeze(-1)
        elif y_train.dim() == 2:
            y_train = y_train.unsqueeze(0) if y_train.shape[0] != X_train.shape[0] else y_train.unsqueeze(-1)
        # X_test: [n_test, n_features] -> [1, n_test, n_features]
        if X_test.dim() == 2:
            X_test = X_test.unsqueeze(0)

        B = X_train.shape[0]
        n_train = X_train.shape[1]
        n_test = X_test.shape[1]
        n_features = X_train.shape[2]

        # Concatenate X and y for train: [B, n_train, n_features + 1]
        train_data = torch.cat([X_train, y_train], dim=-1)

        # Create placeholder for test y (zeros): [B, n_test, 1]
        y_test_placeholder = torch.zeros(B, n_test, 1, device=X_test.device, dtype=X_test.dtype)
        test_data = torch.cat([X_test, y_test_placeholder], dim=-1)

        # Stack train and test: [B, n_train + n_test, n_features + 1]
        x_known = torch.cat([train_data, test_data], dim=1)

        # Create mask: all 1 except test target column
        n_total_features = n_features + 1  # +1 for target
        mask = create_icl_mask(
            n_train=n_train,
            n_test=n_test,
            n_features=n_total_features,
            target_col=self.icl_config.target_col,
            batch_size=B,
            device=x_known.device,
        )

        return x_known, mask

    def postprocess(
        self,
        x_0: Tensor,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
    ) -> Tensor:
        """
        Extract predicted y_test from RePaint output.

        Args:
            x_0: RePaint output [B, N, C]
            X_train, y_train, X_test: Original inputs (for dimensions)

        Returns:
            y_pred: Predicted targets for test samples [B, n_test] or [n_test]
        """
        # Get dimensions
        if isinstance(X_train, np.ndarray):
            n_train = X_train.shape[0] if X_train.ndim == 2 else X_train.shape[1]
        else:
            n_train = X_train.shape[0] if X_train.dim() == 2 else X_train.shape[1]

        # Extract test predictions (last column of test rows)
        target_col = self.icl_config.target_col
        y_pred = x_0[:, n_train:, target_col]

        return y_pred

    def predict(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
        return_numpy: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        """
        Predict target values for test samples given training context.

        Args:
            X_train: Training features [n_train, n_features]
            y_train: Training targets [n_train] or [n_train, 1]
            X_test: Test features [n_test, n_features]
            return_numpy: If True, return numpy array

        Returns:
            y_pred: Predicted targets [n_test]
        """
        # Run inference
        y_pred = self(X_train, y_train, X_test)

        # Remove batch dimension if inputs were 2D
        input_was_2d = (isinstance(X_train, np.ndarray) and X_train.ndim == 2) or \
                       (isinstance(X_train, Tensor) and X_train.dim() == 2)
        if input_was_2d and y_pred.dim() > 1:
            y_pred = y_pred.squeeze(0)

        # Convert to numpy if requested
        if return_numpy:
            y_pred = y_pred.cpu().numpy()

        return y_pred

    def predict_with_uncertainty(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
        n_samples: int = 10,
        return_numpy: bool = False,
    ) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
        """
        Predict with uncertainty estimation using multiple samples.

        Args:
            X_train: Training features [n_train, n_features]
            y_train: Training targets [n_train]
            X_test: Test features [n_test, n_features]
            n_samples: Number of prediction samples
            return_numpy: If True, return numpy arrays

        Returns:
            mean: Mean predictions [n_test]
            std: Standard deviation (uncertainty) [n_test]
        """
        samples = []

        for _ in range(n_samples):
            y_pred = self.predict(X_train, y_train, X_test)
            samples.append(y_pred)

        # Stack samples
        if isinstance(samples[0], np.ndarray):
            samples = np.stack(samples, axis=0)
            mean = samples.mean(axis=0)
            std = samples.std(axis=0)
        else:
            samples = torch.stack(samples, dim=0)
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)

        if return_numpy and isinstance(mean, Tensor):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()

        return mean, std


class FastRegressionICL(RegressionICL):
    """
    Fast regression ICL without resampling.

    Uses simplified RePaint sampling for faster predictions.
    May be less accurate but significantly faster.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[RegressionICLConfig] = None,
    ):
        if config is None:
            config = RegressionICLConfig()
        config.use_resampling = False
        super().__init__(model, config)
