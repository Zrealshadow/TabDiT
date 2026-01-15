"""
Value Embedding

Shared latent encoding that projects scalar values to embedding space.
Used as the first step in processing tabular data.
"""

import torch
import torch.nn as nn


class ValueEmbedding(nn.Module):
    """
    Projects scalar values to embedding space.

    This is the fundamental operation that maps each scalar feature value
    to a D-dimensional embedding. Shared across all features.

    Args:
        d_model: Embedding dimension
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Scalar values [..., 1] or [...] (will unsqueeze if needed)

        Returns:
            Embeddings [..., D]
        """
        if x.dim() >= 1 and x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        return self.proj(x)
