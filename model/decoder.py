"""
Decoder: Feature Reconstruction

Reconstructs full feature tensor [B, N, C] from:
- CLS context [B, N, K*D] from diffusion transformer
- Skip connection [B, N, C, D] from column encoder
"""

import torch
import torch.nn as nn
from typing import Optional


class DecoderBlock(nn.Module):
    """
    Single decoder block with cross-attention and self-attention.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Cross-attention: features query CLS context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention: features attend to each other
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
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
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Feature embeddings [B*N, C, D]
            context: CLS context [B*N, K, D]

        Returns:
            Updated feature embeddings [B*N, C, D]
        """
        # Cross-attention: features attend to CLS context
        x_attn, _ = self.cross_attn(
            query=x,
            key=context,
            value=context,
        )
        x = self.norm1(x + x_attn)

        # Self-attention: features attend to each other
        x_self, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
        )
        x = self.norm2(x + x_self)

        # Feed-forward
        x = self.norm3(x + self.ffn(x))

        return x


class Decoder(nn.Module):
    """
    Feature reconstruction decoder.

    Combines skip connection features with CLS context via cross-attention
    to reconstruct the full feature tensor.

    Args:
        d_model: Feature embedding dimension
        d_context: CLS context dimension (num_cls * d_model)
        num_cls_tokens: Number of CLS tokens
        num_blocks: Number of decoder blocks
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        max_features: Maximum number of features (for position embedding)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        d_context: int = 512,
        num_cls_tokens: int = 4,
        num_blocks: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        max_features: int = 200,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cls_tokens = num_cls_tokens

        # Feature position embedding (handles variable C)
        self.feature_pos_embed = nn.Parameter(
            torch.randn(max_features, d_model) * 0.02
        )

        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Output projection: embedding -> scalar
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        context: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context: CLS representations from diffusion transformer [B, N, K*D]
            skip: Feature embeddings from column encoder [B, N, C, D]

        Returns:
            Reconstructed features [B, N, C]
        """
        B, N, C, D = skip.shape
        K = self.num_cls_tokens

        # 1. Add feature position embeddings to skip connection
        feat_pos = self.feature_pos_embed[:C]  # [C, D]
        skip = skip + feat_pos.unsqueeze(0).unsqueeze(0)  # [B, N, C, D]

        # 2. Reshape context from [B, N, K*D] to [B, N, K, D]
        context = context.view(B, N, K, D)

        # 3. Flatten batch and row dimensions for attention
        skip_flat = skip.view(B * N, C, D)        # [B*N, C, D]
        context_flat = context.view(B * N, K, D)  # [B*N, K, D]

        # 4. Apply decoder blocks
        h = skip_flat
        for block in self.blocks:
            h = block(h, context_flat)

        # 5. Output projection
        h = self.out_norm(h)      # [B*N, C, D]
        h = self.out_proj(h)      # [B*N, C, 1]
        h = h.squeeze(-1)         # [B*N, C]

        # 6. Reshape back to [B, N, C]
        output = h.view(B, N, C)

        return output


class SimpleDecoder(nn.Module):
    """
    Simplified decoder without cross-attention.

    Uses additive fusion of context and skip features,
    followed by MLP projection.

    Args:
        d_model: Feature embedding dimension
        d_context: CLS context dimension
        num_cls_tokens: Number of CLS tokens
        max_features: Maximum number of features
        hidden_mult: MLP hidden dimension multiplier
    """

    def __init__(
        self,
        d_model: int = 128,
        d_context: int = 512,
        num_cls_tokens: int = 4,
        max_features: int = 200,
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cls_tokens = num_cls_tokens

        # Feature position embedding
        self.feature_pos_embed = nn.Parameter(
            torch.randn(max_features, d_model) * 0.02
        )

        # Context projection: broadcast CLS to features
        self.context_proj = nn.Linear(d_context, d_model)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.LayerNorm(d_model),
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        context: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context: CLS representations [B, N, K*D]
            skip: Feature embeddings [B, N, C, D]

        Returns:
            Reconstructed features [B, N, C]
        """
        B, N, C, D = skip.shape

        # 1. Add feature position embeddings
        feat_pos = self.feature_pos_embed[:C]
        skip = skip + feat_pos.unsqueeze(0).unsqueeze(0)

        # 2. Project context and broadcast to all features
        ctx = self.context_proj(context)  # [B, N, D]
        ctx = ctx.unsqueeze(2).expand(-1, -1, C, -1)  # [B, N, C, D]

        # 3. Additive fusion
        h = skip + ctx

        # 4. MLP processing
        h = self.fusion(h)  # [B, N, C, D]

        # 5. Output projection
        output = self.out_proj(h).squeeze(-1)  # [B, N, C]

        return output
