"""
Stage 1: Column-wise Encoder

Processes each column independently to capture distributional properties.
Uses Set Transformer with inducing points for O(n) complexity.
"""

import torch
import torch.nn as nn
from typing import Optional

from .components import SetTransformer


class ColumnEncoder(nn.Module):
    """
    Column-wise embedding with distribution-aware transformations.

    For each column/feature:
    1. Project scalar values to embeddings
    2. Apply Set Transformer to capture column distribution
    3. Generate distribution-aware weights and biases
    4. Transform embeddings with learned weights/biases

    Args:
        d_model: Model dimension
        num_blocks: Number of Set Transformer blocks
        num_heads: Number of attention heads
        num_inducing: Number of inducing points
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_inducing: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Initial projection: scalar -> embedding
        self.in_proj = nn.Linear(1, d_model)

        # Set Transformer for distribution-aware processing
        self.set_transformer = SetTransformer(
            d_model=d_model,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_inducing=num_inducing,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Distribution-aware transformation outputs
        self.out_weights = nn.Linear(d_model, d_model)
        self.out_biases = nn.Linear(d_model, d_model)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C] where:
                B = batch size
                N = number of rows
                C = number of columns/features
            mask: Optional mask [B, N] for padding

        Returns:
            Feature embeddings [B, N, C, D]
        """
        B, N, C = x.shape

        # 1. Reshape for column-wise processing
        # [B, N, C] -> [B, C, N, 1]
        x = x.transpose(1, 2).unsqueeze(-1)

        # 2. Initial projection: [B, C, N, 1] -> [B, C, N, D]
        features = self.in_proj(x)

        # 3. Apply Set Transformer per column
        # Reshape to process all columns together: [B*C, N, D]
        features_flat = features.view(B * C, N, self.d_model)

        # Set Transformer processes each column independently
        features_flat = self.set_transformer(features_flat, mask)

        # Reshape back: [B, C, N, D]
        features = features_flat.view(B, C, N, self.d_model)

        # 4. Generate distribution-aware weights and biases
        weights = self.out_weights(features)  # [B, C, N, D]
        biases = self.out_biases(features)    # [B, C, N, D]

        # 5. Apply transformation
        # Get original features (before Set Transformer)
        x_expanded = x.expand(-1, -1, -1, self.d_model)  # [B, C, N, D]
        features = x_expanded * weights + biases

        # 6. Normalize and reshape to [B, N, C, D]
        features = self.norm(features)
        features = features.permute(0, 2, 1, 3)  # [B, N, C, D]

        return features


class SimpleColumnEncoder(nn.Module):
    """
    Simplified column encoder without Set Transformer.

    Just projects scalar values to embeddings with learnable
    feature position embeddings.

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

        # Projection: scalar -> embedding
        self.proj = nn.Linear(1, d_model)

        # Feature position embeddings
        self.feature_embed = nn.Parameter(
            torch.randn(max_features, d_model) * 0.02
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C]

        Returns:
            Feature embeddings [B, N, C, D]
        """
        B, N, C = x.shape

        # Project each scalar: [B, N, C] -> [B, N, C, D]
        x = x.unsqueeze(-1)  # [B, N, C, 1]
        features = self.proj(x)  # [B, N, C, D]

        # Add feature position embeddings
        feat_pos = self.feature_embed[:C]  # [C, D]
        features = features + feat_pos.unsqueeze(0).unsqueeze(0)

        return self.norm(features)
