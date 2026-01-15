"""
Feature Positional Embedding

Provides unique identity to each feature/column position.
Inspired by TabPFN's subspace method.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeaturePositionalEmbedding(nn.Module):
    """
    Feature positional embedding using subspace projection.

    Each feature gets a unique embedding that distinguishes it from others.
    This ensures that the same value in different columns has different
    representations (e.g., "5" in age column vs "5" in rating column).

    Method:
    1. Generate random vectors in reduced dimension (d_model // 4)
    2. Learn a shared projection to full dimension
    3. Add to feature embeddings

    Args:
        d_model: Model dimension
        max_features: Maximum number of features
        subspace_dim: Dimension of random subspace (default: d_model // 4)
        seed: Random seed for reproducible embeddings
    """

    def __init__(
        self,
        d_model: int = 128,
        max_features: int = 500,
        subspace_dim: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_features = max_features
        self.subspace_dim = subspace_dim or (d_model // 4)
        self.seed = seed

        # Learnable projection: subspace -> full dimension
        self.proj = nn.Linear(self.subspace_dim, d_model)

        # Generate and register random subspace vectors
        self._init_subspace_vectors()

    def _init_subspace_vectors(self):
        """Initialize random subspace vectors with fixed seed."""
        generator = torch.Generator().manual_seed(self.seed)
        subspace_vectors = torch.randn(
            self.max_features,
            self.subspace_dim,
            generator=generator,
        )
        # Normalize for stability
        subspace_vectors = subspace_vectors / subspace_vectors.norm(dim=-1, keepdim=True)
        self.register_buffer("subspace_vectors", subspace_vectors)

    def forward(self, num_features: int) -> torch.Tensor:
        """
        Get positional embeddings for given number of features.

        Args:
            num_features: Number of features (C)

        Returns:
            Feature embeddings [C, D]
        """
        # Get subspace vectors for these features
        subspace = self.subspace_vectors[:num_features]  # [C, subspace_dim]

        # Project to full dimension
        embeddings = self.proj(subspace)  # [C, D]

        return embeddings

    def add_to_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add feature positional embeddings to input tensor.

        Args:
            x: Input tensor [B, N, C, D] or [B, C, N, D]
               Assumes C is at dim 2 for [B, N, C, D]

        Returns:
            Tensor with added positional embeddings, same shape as input
        """
        # Determine C dimension (assumes [B, N, C, D] format)
        C = x.shape[2]

        # Get embeddings
        pos_emb = self.forward(C)  # [C, D]

        # Add with broadcasting: [C, D] -> [1, 1, C, D]
        return x + pos_emb


class LearnedFeatureEmbedding(nn.Module):
    """
    Fully learned feature positional embeddings.

    Simpler alternative to subspace method - directly learns
    an embedding for each feature position.

    Args:
        d_model: Model dimension
        max_features: Maximum number of features
    """

    def __init__(
        self,
        d_model: int = 128,
        max_features: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_features = max_features

        # Learnable embeddings
        self.embeddings = nn.Parameter(
            torch.randn(max_features, d_model) * 0.02
        )

    def forward(self, num_features: int) -> torch.Tensor:
        """
        Get positional embeddings for given number of features.

        Args:
            num_features: Number of features (C)

        Returns:
            Feature embeddings [C, D]
        """
        return self.embeddings[:num_features]

    def add_to_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add feature positional embeddings to input tensor.

        Args:
            x: Input tensor [B, N, C, D]

        Returns:
            Tensor with added positional embeddings
        """
        C = x.shape[2]
        pos_emb = self.forward(C)  # [C, D]
        return x + pos_emb
