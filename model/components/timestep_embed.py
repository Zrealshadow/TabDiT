"""
Timestep Embedding for Diffusion Models

Provides sinusoidal positional encoding for diffusion timesteps.
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding as used in DDPM.

    Maps scalar timesteps to high-dimensional embeddings using
    sinusoidal functions at different frequencies.

    Args:
        d_embed: Embedding dimension (should be even)
        max_timesteps: Maximum number of timesteps
    """

    def __init__(
        self,
        d_embed: int = 512,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.d_embed = d_embed
        self.max_timesteps = max_timesteps
        self.max_period = 10000
        # max_period: controls the minimum frequency of the embeddings
        # Precompute embedding table
        self._build_embedding_table()

    def _build_embedding_table(self):
        """Build sinusoidal embedding table."""
        half_dim = self.d_embed // 2

        # Frequency scaling
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim).float() / half_dim
        )

        # Timestep positions
        timesteps = torch.arange(self.max_timesteps).float()

        # Compute angles: [max_timesteps, half_dim]
        angles = timesteps.unsqueeze(1) * freqs.unsqueeze(0)

        # Interleave sin and cos: [max_timesteps, d_embed]
        embeddings = torch.zeros(self.max_timesteps, self.d_embed)
        embeddings[:, 0::2] = angles.sin()
        embeddings[:, 1::2] = angles.cos()

        self.register_buffer("embedding_table", embeddings, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for timesteps.

        Args:
            t: Timestep tensor [B] with integer values

        Returns:
            Embeddings [B, d_embed]
        """
        return self.embedding_table[t]


class TimestepMLP(nn.Module):
    """
    MLP for processing timestep embeddings.

    Projects sinusoidal embeddings through MLP layers
    for richer representations.

    Args:
        d_embed: Input embedding dimension
        d_out: Output dimension
        hidden_mult: Hidden layer multiplier
    """

    def __init__(
        self,
        d_embed: int = 512,
        d_out: int = 512,
        hidden_mult: int = 4,
    ):
        super().__init__()

        d_hidden = d_embed * hidden_mult

        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_embed: Timestep embeddings [B, d_embed]

        Returns:
            Processed embeddings [B, d_out]
        """
        return self.mlp(t_embed)


class TimestepEmbedder(nn.Module):
    """
    Complete timestep embedding module.

    Combines sinusoidal embedding with MLP processing.

    Args:
        d_out: Output embedding dimension
        max_timesteps: Maximum number of timesteps
    """

    def __init__(
        self,
        d_out: int = 512,
        max_timesteps: int = 1000,
    ):
        super().__init__()

        self.sinusoidal = SinusoidalTimestepEmbedding(
            d_embed=d_out,
            max_timesteps=max_timesteps,
        )
        self.mlp = TimestepMLP(
            d_embed=d_out,
            d_out=d_out,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor [B] with integer values

        Returns:
            Embeddings [B, d_out]
        """
        t_embed = self.sinusoidal(t)
        return self.mlp(t_embed)
