"""
Stage 2: Row-wise Encoder

Captures feature interactions within each row using transformer encoder.
Supports two position encoding methods:
- "subspace": Additive feature positional embeddings (default, like TabPFN)
- "rope": Rotary position encoding in attention
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from .components import RoPEMultiheadAttention, FeaturePositionalEmbedding


PositionEncoding = Literal["subspace", "rope"]


class RowEncoderBlock(nn.Module):
    """
    Single row encoder block with configurable attention.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        position_encoding: "subspace" for standard attention, "rope" for RoPE attention
        max_seq_len: Maximum sequence length (for RoPE)
        rope_base: Base for RoPE frequencies
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        position_encoding: PositionEncoding = "subspace",
        max_seq_len: int = 512,
        rope_base: float = 100000.0,
    ):
        super().__init__()
        self.position_encoding = position_encoding

        # Attention layer based on position encoding choice
        if position_encoding == "rope":
            self.attn = RoPEMultiheadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:  # subspace - use standard attention
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
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
            x: Input tensor [B, N, L, D]
            attn_mask: Optional attention mask

        Returns:
            Output tensor [B, N, L, D]
        """
        B, N, L, D = x.shape

        # Reshape for attention: [B*N, L, D]
        x_flat = x.view(B * N, L, D)

        # Self-attention (RoPE or standard based on config)
        if self.position_encoding == "rope":
            attn_out = self.attn(x_flat, attn_mask)
        else:
            attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=attn_mask)

        x_flat = self.norm1(x_flat + attn_out)

        # Feed-forward
        x_flat = self.norm2(x_flat + self.ffn(x_flat))

        # Reshape back: [B, N, L, D]
        return x_flat.view(B, N, L, D)


class RowEncoder(nn.Module):
    """
    Row-wise encoder with CLS tokens and configurable position encoding.

    For each row:
    1. Prepend learnable CLS tokens
    2. Add position encoding (subspace or RoPE)
    3. Apply transformer encoder
    4. Extract CLS tokens as row representation

    Args:
        d_model: Model dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        num_cls_tokens: Number of CLS tokens per row
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_features: Maximum number of features
        position_encoding: "subspace" (additive, default) or "rope" (rotary)
        rope_base: Base for RoPE frequencies (only used if position_encoding="rope")
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 3,
        num_heads: int = 8,
        num_cls_tokens: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_features: int = 500,
        position_encoding: PositionEncoding = "subspace",
        rope_base: float = 100000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cls_tokens = num_cls_tokens
        self.position_encoding = position_encoding

        # Feature positional embedding (only for subspace mode)
        if position_encoding == "subspace":
            self.feature_pos_embedding = FeaturePositionalEmbedding(
                d_model=d_model,
                max_features=max_features + num_cls_tokens,
            )
        else:
            self.feature_pos_embedding = None

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
                position_encoding=position_encoding,
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
            feature_output: Feature embeddings [B, N, C, D] (for skip connection)
        """
        B, N, C, D = x.shape
        K = self.num_cls_tokens

        # 1. Expand CLS tokens for all rows
        cls = self.cls_tokens.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
        cls = cls.expand(B, N, -1, -1)  # [B, N, K, D]

        # 2. Prepend CLS tokens: [B, N, K+C, D]
        x = torch.cat([cls, x], dim=2)

        # 3. Add feature positional embeddings (only for subspace mode)
        if self.feature_pos_embedding is not None:
            pos_emb = self.feature_pos_embedding(K + C)  # [K+C, D]
            x = x + pos_emb  # broadcasts to [B, N, K+C, D]

        # 4. Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # 5. Apply output normalization
        x = self.out_norm(x)

        # 6. Extract and flatten CLS tokens
        cls_output = x[:, :, :K, :]  # [B, N, K, D]
        cls_output = cls_output.flatten(-2)  # [B, N, K*D]

        # 7. Return both CLS output and feature embeddings (without CLS for skip)
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
        position_encoding: "subspace" (additive, default) or "rope" (rotary)
        rope_base: Base for RoPE frequencies (only used if position_encoding="rope")
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_features: int = 500,
        position_encoding: PositionEncoding = "subspace",
        rope_base: float = 100000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.position_encoding = position_encoding

        # Feature positional embedding (only for subspace mode)
        if position_encoding == "subspace":
            self.feature_pos_embedding = FeaturePositionalEmbedding(
                d_model=d_model,
                max_features=max_features,
            )
        else:
            self.feature_pos_embedding = None

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            RowEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                position_encoding=position_encoding,
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
        # Add feature positional embeddings (only for subspace mode)
        if self.feature_pos_embedding is not None:
            C = x.shape[2]
            pos_emb = self.feature_pos_embedding(C)  # [C, D]
            x = x + pos_emb  # broadcasts to [B, N, C, D]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.out_norm(x)
