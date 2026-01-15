"""
Stage 1: Column-wise Encoder

Processes each column independently to capture distributional properties.
Uses Set Transformer with inducing points for O(n) complexity.
"""

import torch
import torch.nn as nn
from typing import Optional

from .components import SetTransformer, ValueEmbedding


class ColumnEncoder(nn.Module):
    """
    Column-wise embedding with Set Transformer.

    For each column/feature:
    1. Project scalar values to embeddings (via shared ValueEmbedding)
    2. Apply Set Transformer to capture column distribution

    The Set Transformer has residual connections, so original input
    information is preserved while learning distribution-aware representations.

    Args:
        d_model: Model dimension
        num_blocks: Number of Set Transformer blocks
        num_heads: Number of attention heads
        num_inducing: Number of inducing points
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        value_embedding: Optional shared ValueEmbedding instance
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_inducing: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        value_embedding: Optional[ValueEmbedding] = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Shared value embedding: scalar -> embedding
        self.value_embedding = value_embedding or ValueEmbedding(d_model)

        assert self.d_model == self.value_embedding.d_model, \
            "Incompatible value embedding dimension"
            
        # Set Transformer for distribution-aware processing
        # Note: SetTransformer output is already normalized (LayerNorm in each ISAB block)
        self.set_transformer = SetTransformer(
            d_model=d_model,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_inducing=num_inducing,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

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

        # 1. Reshape for column-wise processing: [B, N, C] -> [B, C, N]
        x = x.transpose(1, 2)

        # 2. Value embedding: [B, C, N] -> [B, C, N, D]
        features = self.value_embedding(x)

        # 3. Apply Set Transformer per column: [B*C, N, D]
        features = features.view(B * C, N, self.d_model)
        features = self.set_transformer(features, mask)

        # 4. Reshape: [B*C, N, D] -> [B, C, N, D] -> [B, N, C, D]
        features = features.view(B, C, N, self.d_model)
        features = features.permute(0, 2, 1, 3)

        return features


class SimpleColumnEncoder(nn.Module):
    """
    Simplified column encoder without Set Transformer.

    Just projects scalar values to embeddings with learnable
    feature position embeddings.

    Args:
        d_model: Model dimension
        max_features: Maximum number of features
        value_embedding: Optional shared ValueEmbedding instance
    """

    def __init__(
        self,
        d_model: int = 128,
        max_features: int = 200,
        value_embedding: Optional[ValueEmbedding] = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Shared value embedding: scalar -> embedding
        self.value_embedding = value_embedding or ValueEmbedding(d_model)

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
        C = x.shape[2]

        # Value embedding: [B, N, C] -> [B, N, C, D]
        features = self.value_embedding(x)

        # Add feature position embeddings
        feat_pos = self.feature_embed[:C]  # [C, D]
        features = features + feat_pos  # broadcasts to [B, N, C, D]

        return self.norm(features)
