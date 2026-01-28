"""
Classification In-Context Learning using RePaint Algorithm.

Uses the unconditional diffusion model for classification by framing it as inpainting:
- Train rows: all features (X, y) are known → mask = 1
- Test rows: features X are known, target y is unknown → mask = 0 for y

For classification, we handle discrete labels by:
1. Encoding labels as continuous values (normalized to [-1, 1])
2. Running RePaint to get continuous predictions
3. Decoding predictions back to discrete labels via nearest neighbor or rounding
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from .base import BaseInference, InferenceConfig
from .utils import create_icl_mask


@dataclass
class ClassificationICLConfig(InferenceConfig):
    """Configuration for classification ICL task."""

    # ICL-specific options
    target_col: int = -1  # Index of target column (default: last column)

    # Classification-specific
    num_classes: Optional[int] = None  # Number of classes (auto-detected if None)
    label_encoding: str = "normalized"  # "normalized" (0,1,..,K-1 -> -1 to 1) or "onehot"

    # Prediction
    n_ensemble: int = 1  # Number of ensemble predictions
    voting: str = "soft"  # "soft" (average probs) or "hard" (majority vote)


class ClassificationICL(BaseInference):
    """
    Classification In-Context Learning using RePaint algorithm.

    Given training examples (X_train, y_train) as context, predicts class labels
    for test examples X_test by treating it as an inpainting problem.

    Labels are encoded as continuous values for the diffusion process:
    - Normalized encoding: labels 0,1,...,K-1 mapped to evenly spaced values in [-1, 1]
    - Predictions are decoded back to discrete labels via nearest neighbor matching

    Example:
        >>> model = TabularDiffusion.load("checkpoint.pt")
        >>> classifier = ClassificationICL(model)
        >>>
        >>> # Training data (context)
        >>> X_train = torch.randn(80, 10)  # 80 samples, 10 features
        >>> y_train = torch.randint(0, 3, (80,))  # 3-class classification
        >>>
        >>> # Test data
        >>> X_test = torch.randn(20, 10)   # 20 samples to classify
        >>>
        >>> # Predict
        >>> y_pred = classifier.predict(X_train, y_train, X_test)

    Args:
        model: Trained TabularDiffusion model
        config: ClassificationICLConfig or None for defaults
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ClassificationICLConfig] = None,
    ):
        if config is None:
            config = ClassificationICLConfig()
        super().__init__(model, config)
        self.cls_config = config

        # Label encoding/decoding
        self._num_classes: Optional[int] = config.num_classes
        self._label_values: Optional[Tensor] = None  # Encoded label values

    def _setup_label_encoding(self, y: Tensor) -> None:
        """
        Setup label encoding based on observed labels.

        Args:
            y: Training labels [n_train]
        """
        if self._num_classes is None:
            # Auto-detect number of classes
            unique_labels = torch.unique(y)
            self._num_classes = len(unique_labels)

        # Create encoded values for each class
        # Map class indices to evenly spaced values in [-1, 1]
        if self._num_classes == 2:
            # Binary: 0 -> -1, 1 -> 1
            self._label_values = torch.tensor([-1.0, 1.0], device=y.device)
        else:
            # Multi-class: evenly spaced in [-1, 1]
            self._label_values = torch.linspace(-1, 1, self._num_classes, device=y.device)

    def _encode_labels(self, y: Tensor) -> Tensor:
        """
        Encode discrete labels to continuous values.

        Args:
            y: Discrete labels [n] with values in {0, 1, ..., K-1}

        Returns:
            Encoded values [n, 1] in [-1, 1]
        """
        if self._label_values is None:
            self._setup_label_encoding(y)

        # Map labels to encoded values
        y_encoded = self._label_values[y.long()]

        return y_encoded.unsqueeze(-1)  # [n, 1]

    def _decode_labels(self, y_continuous: Tensor) -> Tensor:
        """
        Decode continuous predictions to discrete labels.

        Uses nearest neighbor matching to find closest class.

        Args:
            y_continuous: Continuous predictions [n] or [n, 1]

        Returns:
            Discrete labels [n] with values in {0, 1, ..., K-1}
        """
        if y_continuous.dim() > 1:
            y_continuous = y_continuous.squeeze(-1)

        # Find nearest encoded label value for each prediction
        # [n, 1] - [1, K] -> [n, K] distances
        distances = torch.abs(
            y_continuous.unsqueeze(-1) - self._label_values.unsqueeze(0)
        )

        # Get index of minimum distance (predicted class)
        y_pred = distances.argmin(dim=-1)

        return y_pred

    def prepare_input(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare input for classification ICL.

        Args:
            X_train: Training features [n_train, n_features] or [B, n_train, n_features]
            y_train: Training labels [n_train] or [B, n_train]
            X_test: Test features [n_test, n_features] or [B, n_test, n_features]

        Returns:
            x_known: Combined data [B, N, C]
            mask: Binary mask [B, N, C]
        """
        # Convert numpy to tensor if needed
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train)
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()

        # Handle dimensions for X_train
        if X_train.dim() == 2:
            X_train = X_train.unsqueeze(0)
        if X_test.dim() == 2:
            X_test = X_test.unsqueeze(0)

        # Handle y_train dimensions and encode labels
        if y_train.dim() == 1:
            self._setup_label_encoding(y_train)
            y_train_encoded = self._encode_labels(y_train)  # [n_train, 1]
            y_train_encoded = y_train_encoded.unsqueeze(0)  # [1, n_train, 1]
        else:
            # Batched y_train [B, n_train]
            self._setup_label_encoding(y_train[0])
            y_train_encoded = torch.stack([
                self._encode_labels(y_train[b]) for b in range(y_train.shape[0])
            ], dim=0)  # [B, n_train, 1]

        B = X_train.shape[0]
        n_train = X_train.shape[1]
        n_test = X_test.shape[1]
        n_features = X_train.shape[2]

        # Concatenate X and encoded y for train: [B, n_train, n_features + 1]
        train_data = torch.cat([X_train, y_train_encoded.to(X_train.device)], dim=-1)

        # Create placeholder for test y: [B, n_test, 1]
        y_test_placeholder = torch.zeros(B, n_test, 1, device=X_test.device, dtype=X_test.dtype)
        test_data = torch.cat([X_test, y_test_placeholder], dim=-1)

        # Stack train and test: [B, n_train + n_test, n_features + 1]
        x_known = torch.cat([train_data, test_data], dim=1)

        # Create mask
        n_total_features = n_features + 1
        mask = create_icl_mask(
            n_train=n_train,
            n_test=n_test,
            n_features=n_total_features,
            target_col=self.cls_config.target_col,
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
        Extract and decode predicted labels from RePaint output.

        Args:
            x_0: RePaint output [B, N, C]
            X_train, y_train, X_test: Original inputs

        Returns:
            y_pred: Predicted class labels [B, n_test] or [n_test]
        """
        # Get n_train
        if isinstance(X_train, np.ndarray):
            n_train = X_train.shape[0] if X_train.ndim == 2 else X_train.shape[1]
        else:
            n_train = X_train.shape[0] if X_train.dim() == 2 else X_train.shape[1]

        # Extract continuous predictions for test rows
        target_col = self.cls_config.target_col
        y_continuous = x_0[:, n_train:, target_col]  # [B, n_test]

        # Decode to discrete labels
        y_pred = torch.stack([
            self._decode_labels(y_continuous[b]) for b in range(y_continuous.shape[0])
        ], dim=0)

        return y_pred

    def predict(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
        return_numpy: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        """
        Predict class labels for test samples given training context.

        Args:
            X_train: Training features [n_train, n_features]
            y_train: Training labels [n_train]
            X_test: Test features [n_test, n_features]
            return_numpy: If True, return numpy array

        Returns:
            y_pred: Predicted class labels [n_test]
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

    def predict_proba(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        X_test: Union[Tensor, np.ndarray],
        n_samples: int = 10,
        return_numpy: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        """
        Predict class probabilities using multiple samples.

        Runs multiple predictions and computes class probabilities
        based on frequency of each predicted class.

        Args:
            X_train: Training features [n_train, n_features]
            y_train: Training labels [n_train]
            X_test: Test features [n_test, n_features]
            n_samples: Number of prediction samples
            return_numpy: If True, return numpy array

        Returns:
            proba: Class probabilities [n_test, n_classes]
        """
        # Ensure label encoding is set up
        if isinstance(y_train, np.ndarray):
            y_train_tensor = torch.from_numpy(y_train)
        else:
            y_train_tensor = y_train
        self._setup_label_encoding(y_train_tensor)

        samples = []
        for _ in range(n_samples):
            y_pred = self.predict(X_train, y_train, X_test)
            samples.append(y_pred)

        # Stack samples: [n_samples, n_test]
        if isinstance(samples[0], np.ndarray):
            samples = np.stack(samples, axis=0)
            n_test = samples.shape[1]

            # Count class frequencies
            proba = np.zeros((n_test, self._num_classes))
            for i in range(n_test):
                for c in range(self._num_classes):
                    proba[i, c] = (samples[:, i] == c).mean()
        else:
            samples = torch.stack(samples, dim=0)
            n_test = samples.shape[1]

            # Count class frequencies
            proba = torch.zeros(n_test, self._num_classes, device=samples.device)
            for i in range(n_test):
                for c in range(self._num_classes):
                    proba[i, c] = (samples[:, i] == c).float().mean()

            if return_numpy:
                proba = proba.cpu().numpy()

        return proba


class FastClassificationICL(ClassificationICL):
    """
    Fast classification ICL without resampling.

    Uses simplified RePaint sampling for faster predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ClassificationICLConfig] = None,
    ):
        if config is None:
            config = ClassificationICLConfig()
        config.use_resampling = False
        super().__init__(model, config)
