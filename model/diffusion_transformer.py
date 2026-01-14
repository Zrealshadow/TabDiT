"""
Stage 3: Diffusion Transformer

Processes row representations with timestep conditioning.
This is the core diffusion component that learns the denoising process.
"""

import torch
import torch.nn as nn
from typing import Optional

from .components import TimestepEmbedder


class DiffusionTransformerBlock(nn.Module):
    """
    Single transformer block for diffusion.

    Standard transformer block with optional cross-attention
    for conditioning (not used in unconditional diffusion).

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
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

        # Adaptive LayerNorm for timestep conditioning (DiT-style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, D]
            t_embed: Timestep embedding [B, D]
            attn_mask: Optional attention mask

        Returns:
            Output tensor [B, N, D]
        """
        # Compute adaptive LayerNorm parameters from timestep
        # [B, D] -> [B, 6*D] -> 6 x [B, 1, D]
        modulation = self.adaLN_modulation(t_embed)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = \
            modulation.chunk(6, dim=-1)

        # Expand for broadcasting: [B, D] -> [B, 1, D]
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_ffn = shift_ffn.unsqueeze(1)
        scale_ffn = scale_ffn.unsqueeze(1)
        gate_ffn = gate_ffn.unsqueeze(1)

        # Self-attention with adaptive LayerNorm
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa) + shift_msa
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + gate_msa * attn_out

        # FFN with adaptive LayerNorm
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.ffn(x_norm)

        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT-style) for tabular data.

    Processes CLS representations with timestep conditioning
    using adaptive layer normalization.

    Args:
        d_model: Model dimension (K * d_feature for CLS)
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_timesteps: Maximum diffusion timesteps
    """

    def __init__(
        self,
        d_model: int = 512,
        num_blocks: int = 12,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model

        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(
            d_out=d_model,
            max_timesteps=max_timesteps,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Final layer normalization
        self.out_norm = nn.LayerNorm(d_model)

        # Final modulation
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: CLS representations [B, N, D]
            t: Timesteps [B]
            attn_mask: Optional attention mask

        Returns:
            Processed representations [B, N, D]
        """
        # Get timestep embeddings
        t_embed = self.timestep_embedder(t)  # [B, D]

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, t_embed, attn_mask)

        # Final adaptive LayerNorm
        shift, scale = self.final_adaLN(t_embed).chunk(2, dim=-1)
        x = self.out_norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return x


class SimpleDiffusionTransformer(nn.Module):
    """
    Simplified diffusion transformer without adaptive LayerNorm.

    Adds timestep embedding directly to input instead of using
    per-layer modulation. Lighter weight alternative.

    Args:
        d_model: Model dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_timesteps: Maximum diffusion timesteps
    """

    def __init__(
        self,
        d_model: int = 512,
        num_blocks: int = 12,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model

        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(
            d_out=d_model,
            max_timesteps=max_timesteps,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: CLS representations [B, N, D]
            t: Timesteps [B]
            attn_mask: Optional attention mask

        Returns:
            Processed representations [B, N, D]
        """
        # Get timestep embeddings and add to input
        t_embed = self.timestep_embedder(t)  # [B, D]
        x = x + t_embed.unsqueeze(1)  # Broadcast to all rows

        # Process through transformer
        x = self.transformer(x, mask=attn_mask)

        return self.out_norm(x)
