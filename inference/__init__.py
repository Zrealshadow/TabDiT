"""
Inference module for downstream tasks using unconditional diffusion model.

All downstream tasks are framed as "inpainting" using the RePaint algorithm:
- Imputation: mask = missing values, fill them in
- Regression ICL: mask = target column for test rows, predict given context
- Classification ICL: mask = target column for test rows, predict class labels

Key components:
- RePaintSampler: Core RePaint algorithm for conditional generation
- BaseInference: Base class for all downstream tasks
- Imputer: Missing value imputation
- RegressionICL: In-context learning for regression
- ClassificationICL: In-context learning for classification
"""

from .repaint_sampler import RePaintSampler, RePaintConfig
from .base import BaseInference, InferenceConfig
from .imputation import Imputer, FastImputer, ImputationConfig
from .regression import RegressionICL, FastRegressionICL, RegressionICLConfig
from .classification import ClassificationICL, FastClassificationICL, ClassificationICLConfig

__all__ = [
    # Core
    "RePaintSampler",
    "RePaintConfig",
    "BaseInference",
    "InferenceConfig",
    # Imputation
    "Imputer",
    "FastImputer",
    "ImputationConfig",
    # Regression ICL
    "RegressionICL",
    "FastRegressionICL",
    "RegressionICLConfig",
    # Classification ICL
    "ClassificationICL",
    "FastClassificationICL",
    "ClassificationICLConfig",
]
