"""
Set Transformer with Induced Self-Attention

Reduces complexity from O(n²) to O(n) using learnable inducing points.
Used in Stage 1 (Column-wise Embedding) to process variable-length columns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InducedSelfAttention(nn.Module):
    """
    Induced Self-Attention Block (ISAB)

    Two-stage attention that reduces O(n²) to O(m*n) where m << n:
    1. Inducing points attend to all inputs (aggregate)
    2. Inputs attend to inducing points (distribute)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_inducing: Number of inducing points (m)
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_inducing: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_inducing = num_inducing

        # Learnable inducing points
        self.inducing_points = nn.Parameter(
            torch.randn(num_inducing, d_model) * 0.02
        )

        # Stage 1: Inducing points attend to inputs
        self.attn1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Stage 2: Inputs attend to inducing points
        self.attn2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask for Stage 1 (to prevent leakage)

        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape

        # Expand inducing points for batch
        inducing = self.inducing_points.unsqueeze(0).expand(B, -1, -1)
        # inducing: [B, M, D]

        # Stage 1: Aggregate - inducing points query from inputs
        # Q=inducing [B, M, D], K=V=x [B, N, D]
        h, _ = self.attn1(
            query=inducing,
            key=x,
            value=x,
            key_padding_mask=mask,
        )
        h = self.norm1(inducing + h)  # [B, M, D]

        # Stage 2: Distribute - inputs query from inducing points
        # Q=x [B, N, D], K=V=h [B, M, D]
        out, _ = self.attn2(
            query=x,
            key=h,
            value=h,
        )
        out = self.norm2(x + out)  # [B, N, D]

        # Feed-forward
        out = self.norm3(out + self.ffn(out))

        return out


class SetTransformer(nn.Module):
    """
    Set Transformer for distribution-aware column embedding.

    Processes each column independently to capture distributional properties.
    Uses induced self-attention for O(n) complexity.

    Args:
        d_model: Model dimension
        num_blocks: Number of ISAB blocks
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

        self.blocks = nn.ModuleList([
            InducedSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_inducing=num_inducing,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, D] or [B, C, N, D] for batched columns
            mask: Optional attention mask

        Returns:
            Output tensor with same shape as input
        """
        for block in self.blocks:
            x = block(x, mask)
        return x
