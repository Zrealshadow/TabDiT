"""
Stage 2: Row-wise Encoder

Captures feature interactions within each row using transformer
encoder with Rotary Position Encoding (RoPE).
"""

import torch
import torch.nn as nn
from typing import Optional

from .components import RoPEMultiheadAttention


class RowEncoderBlock(nn.Module):
    """
    Single row encoder block with RoPE attention.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for RoPE
        rope_base: Base for RoPE frequencies
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        rope_base: float = 100000.0,
    ):
        super().__init__()

        # RoPE attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C+K, D]
            attn_mask: Optional attention mask

        Returns:
            Output tensor [B, N, C+K, D]
        """
        B, N, L, D = x.shape

        # Reshape for attention: [B*N, L, D]
        x_flat = x.view(B * N, L, D)

        # Self-attention with RoPE
        attn_out = self.attn(x_flat, attn_mask)
        x_flat = self.norm1(x_flat + attn_out)

        # Feed-forward
        x_flat = self.norm2(x_flat + self.ffn(x_flat))

        # Reshape back: [B, N, L, D]
        return x_flat.view(B, N, L, D)


class RowEncoder(nn.Module):
    """
    Row-wise encoder with CLS tokens and RoPE.

    For each row:
    1. Prepend learnable CLS tokens
    2. Apply transformer encoder with RoPE
    3. Extract CLS tokens as row representation

    Args:
        d_model: Model dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        num_cls_tokens: Number of CLS tokens per row
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_features: Maximum number of features (for RoPE)
        rope_base: Base for RoPE frequencies
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 3,
        num_heads: int = 8,
        num_cls_tokens: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_features: int = 200,
        rope_base: float = 100000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cls_tokens = num_cls_tokens

        # Learnable CLS tokens
        self.cls_tokens = nn.Parameter(
            torch.randn(num_cls_tokens, d_model) * 0.02
        )

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            RowEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_features + num_cls_tokens,
                rope_base=rope_base,
            )
            for _ in range(num_blocks)
        ])

        # Output normalization
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, N, C, D] from Stage 1

        Returns:
            cls_output: CLS representations [B, N, K*D]
            full_output: Full sequence output [B, N, C+K, D] (for skip connection)
        """
        B, N, C, D = x.shape
        K = self.num_cls_tokens

        # 1. Expand CLS tokens for all rows
        cls = self.cls_tokens.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
        cls = cls.expand(B, N, -1, -1)  # [B, N, K, D]

        # 2. Prepend CLS tokens: [B, N, K+C, D]
        x = torch.cat([cls, x], dim=2)

        # 3. Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # 4. Apply output normalization
        x = self.out_norm(x)

        # 5. Extract and flatten CLS tokens
        cls_output = x[:, :, :K, :]  # [B, N, K, D]
        cls_output = cls_output.flatten(-2)  # [B, N, K*D]

        # 6. Return both CLS output and full sequence (without CLS for skip)
        feature_output = x[:, :, K:, :]  # [B, N, C, D]

        return cls_output, feature_output


class RowEncoderNoCLS(nn.Module):
    """
    Row-wise encoder without CLS tokens.

    Processes features directly and returns full feature embeddings.
    Simpler alternative when skip connections will use full features.

    Args:
        d_model: Model dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_features: Maximum number of features
        rope_base: Base for RoPE frequencies
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_features: int = 200,
        rope_base: float = 100000.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            RowEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_features,
                rope_base=rope_base,
            )
            for _ in range(num_blocks)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C, D]

        Returns:
            Feature embeddings [B, N, C, D]
        """
        for block in self.blocks:
            x = block(x)

        return self.out_norm(x)
