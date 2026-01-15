"""
Tabular Diffusion Model

Main model that combines all stages:
1. Column Encoder (Stage 1)
2. Row Encoder (Stage 2)
3. Diffusion Transformer (Stage 3)
4. Decoder

For unconditional generation of tabular data.
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from .column_encoder import ColumnEncoder, SimpleColumnEncoder
from .row_encoder import RowEncoder, RowEncoderNoCLS
from .diffusion_transformer import DiffusionTransformer, SimpleDiffusionTransformer
from .decoder import Decoder, SimpleDecoder


@dataclass
class TabularDiffusionConfig:
    """Configuration for TabularDiffusion model."""

    # Dimensions
    d_model: int = 128              # Feature embedding dimension
    num_cls_tokens: int = 4         # CLS tokens (d_context = num_cls * d_model)

    # Stage 1: Column Encoder
    column_blocks: int = 3
    column_heads: int = 4
    num_inducing: int = 128

    # Stage 2: Row Encoder
    row_blocks: int = 3
    row_heads: int = 8
    rope_base: float = 100000.0

    # Stage 3: Diffusion Transformer
    diffusion_blocks: int = 12
    diffusion_heads: int = 4

    # Decoder
    decoder_blocks: int = 3
    decoder_heads: int = 4

    # Limits
    max_features: int = 200
    max_timesteps: int = 1000

    # General
    dim_feedforward_mult: int = 2
    dropout: float = 0.0

    # Architecture variants
    use_simple_column_encoder: bool = False
    use_simple_diffusion: bool = False
    use_simple_decoder: bool = False


class TabularDiffusion(nn.Module):
    """
    Tabular Diffusion Model.

    Architecture:
        Input X_noisy [B, N, C] + timestep t
            ↓
        Stage 1: Column Encoder → [B, N, C, D]
            ↓
            ├──────────────────────────┐ skip
            ↓                          │
        Stage 2: Row Encoder           │
            ↓                          │
        [B, N, K*D] (CLS)              │
            ↓                          │
        Stage 3: Diffusion Transformer │
            ↓                          │
        [B, N, K*D]                    │
            ↓                          ↓
        Decoder (cross-attention with skip)
            ↓
        Output [B, N, C] (noise prediction)

    Args:
        config: TabularDiffusionConfig or None for defaults
        **kwargs: Override config parameters
    """

    def __init__(
        self,
        config: Optional[TabularDiffusionConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # Initialize config
        if config is None:
            config = TabularDiffusionConfig()

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.d_model = config.d_model
        self.d_context = config.num_cls_tokens * config.d_model

        # Computed dimensions
        dim_ff_column = config.d_model * config.dim_feedforward_mult
        dim_ff_row = config.d_model * config.dim_feedforward_mult
        dim_ff_diffusion = self.d_context * config.dim_feedforward_mult
        dim_ff_decoder = config.d_model * config.dim_feedforward_mult

        # Stage 1: Column Encoder
        if config.use_simple_column_encoder:
            self.column_encoder = SimpleColumnEncoder(
                d_model=config.d_model,
                max_features=config.max_features,
            )
        else:
            self.column_encoder = ColumnEncoder(
                d_model=config.d_model,
                num_blocks=config.column_blocks,
                num_heads=config.column_heads,
                num_inducing=config.num_inducing,
                dim_feedforward=dim_ff_column,
                dropout=config.dropout,
            )

        # Stage 2: Row Encoder
        self.row_encoder = RowEncoder(
            d_model=config.d_model,
            num_blocks=config.row_blocks,
            num_heads=config.row_heads,
            num_cls_tokens=config.num_cls_tokens,
            dim_feedforward=dim_ff_row,
            dropout=config.dropout,
            max_features=config.max_features,
            rope_base=config.rope_base,
        )

        # Stage 3: Diffusion Transformer
        if config.use_simple_diffusion:
            self.diffusion_transformer = SimpleDiffusionTransformer(
                d_model=self.d_context,
                num_blocks=config.diffusion_blocks,
                num_heads=config.diffusion_heads,
                dim_feedforward=dim_ff_diffusion,
                dropout=config.dropout,
                max_timesteps=config.max_timesteps,
            )
        else:
            self.diffusion_transformer = DiffusionTransformer(
                d_model=self.d_context,
                num_blocks=config.diffusion_blocks,
                num_heads=config.diffusion_heads,
                dim_feedforward=dim_ff_diffusion,
                dropout=config.dropout,
                max_timesteps=config.max_timesteps,
            )

        # Decoder
        # Note: Decoder doesn't need d_context or max_features
        # Skip from Row Encoder already has feature position embedding
        if config.use_simple_decoder:
            self.decoder = SimpleDecoder(
                d_model=config.d_model,
                d_context=self.d_context,
                num_cls_tokens=config.num_cls_tokens,
            )
        else:
            self.decoder = Decoder(
                d_model=config.d_model,
                num_cls_tokens=config.num_cls_tokens,
                num_blocks=config.decoder_blocks,
                num_heads=config.decoder_heads,
                dim_feedforward=dim_ff_decoder,
                dropout=config.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.

        Args:
            x: Noisy input [B, N, C]
            t: Timesteps [B]

        Returns:
            Predicted noise [B, N, C]
        """
        # Stage 1: Column-wise embedding
        # [B, N, C] -> [B, N, C, D]
        features = self.column_encoder(x)

        # Stage 2: Row-wise interaction
        # [B, N, C, D] -> ([B, N, K*D], [B, N, C, D])
        cls_repr, skip = self.row_encoder(features)

        # Stage 3: Diffusion transformer with timestep
        # [B, N, K*D] -> [B, N, K*D]
        context = self.diffusion_transformer(cls_repr, t)

        # Decoder: reconstruct features
        # ([B, N, K*D], [B, N, C, D]) -> [B, N, C]
        output = self.decoder(context, skip)

        return output

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: Exclude position embeddings if True

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract position embeddings
            if hasattr(self.column_encoder, 'feature_embed'):
                n_params -= self.column_encoder.feature_embed.numel()
            if hasattr(self.decoder, 'feature_pos_embed'):
                n_params -= self.decoder.feature_pos_embed.numel()

        return n_params


class TabularDiffusionSimple(nn.Module):
    """
    Simplified Tabular Diffusion without CLS compression.

    Keeps full feature embeddings throughout, simpler architecture
    but potentially more expensive for large C.

    Architecture:
        Input X_noisy [B, N, C] + timestep t
            ↓
        Column Encoder → [B, N, C, D]
            ↓
        Row Encoder (no CLS) → [B, N, C, D]
            ↓
        + Timestep Embedding
            ↓
        Feature Diffusion Transformer → [B, N, C, D]
            ↓
        Output Projection → [B, N, C]
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 6,
        num_heads: int = 8,
        max_features: int = 200,
        max_timesteps: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Simple column encoder
        self.column_encoder = SimpleColumnEncoder(
            d_model=d_model,
            max_features=max_features,
        )

        # Row encoder without CLS
        self.row_encoder = RowEncoderNoCLS(
            d_model=d_model,
            num_blocks=num_blocks // 2,
            num_heads=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            max_features=max_features,
        )

        # Timestep embedding
        from .components import TimestepEmbedder
        self.timestep_embedder = TimestepEmbedder(
            d_out=d_model,
            max_timesteps=max_timesteps,
        )

        # Feature-wise diffusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.diffusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
        )

        # Output projection
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input [B, N, C]
            t: Timesteps [B]

        Returns:
            Predicted noise [B, N, C]
        """
        B, N, C = x.shape

        # Column encoding: [B, N, C] -> [B, N, C, D]
        features = self.column_encoder(x)

        # Row encoding: [B, N, C, D] -> [B, N, C, D]
        features = self.row_encoder(features)

        # Add timestep embedding (broadcast)
        t_embed = self.timestep_embedder(t)  # [B, D]
        features = features + t_embed.view(B, 1, 1, -1)

        # Flatten for transformer: [B, N, C, D] -> [B, N*C, D]
        features = features.view(B, N * C, self.d_model)

        # Diffusion transformer
        features = self.diffusion_transformer(features)

        # Reshape and project: [B, N*C, D] -> [B, N, C, D] -> [B, N, C]
        features = features.view(B, N, C, self.d_model)
        features = self.out_norm(features)
        output = self.out_proj(features).squeeze(-1)

        return output
