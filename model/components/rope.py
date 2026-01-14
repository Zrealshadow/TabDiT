"""
Rotary Position Encoding (RoPE)

Encodes position information through rotation in the embedding space.
Used in Stage 2 (Row-wise Interaction) to distinguish feature positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Encoding for transformer attention.

    Applies rotation to query and key vectors based on position,
    enabling relative position awareness without learned parameters.

    Args:
        d_model: Model dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)
    """

    def __init__(
        self,
        d_model: int = 128,
        max_seq_len: int = 1024,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        # Frequency for each dimension pair
        dim_pairs = self.d_model // 2
        freqs = 1.0 / (self.base ** (torch.arange(0, dim_pairs).float() / dim_pairs))

        # Position indices
        positions = torch.arange(seq_len).float()

        # Outer product: [seq_len, dim_pairs]
        angles = torch.outer(positions, freqs)

        # Cache sin and cos
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor [..., seq_len, d_model]
            k: Key tensor [..., seq_len, d_model]
            offset: Position offset (for incremental decoding)

        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[-2]

        # Extend cache if needed
        if offset + seq_len > self.cos_cache.shape[0]:
            self._build_cache(offset + seq_len)

        # Get relevant positions
        cos = self.cos_cache[offset:offset + seq_len]  # [seq_len, d/2]
        sin = self.sin_cache[offset:offset + seq_len]  # [seq_len, d/2]

        # Apply rotation
        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)

        return q_rot, k_rot


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position encoding to tensor.

    Rotation formula for each pair (x0, x1):
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos

    Args:
        x: Input tensor [..., seq_len, d_model]
        cos: Cosine values [seq_len, d_model/2]
        sin: Sine values [seq_len, d_model/2]

    Returns:
        Rotated tensor [..., seq_len, d_model]
    """
    # Split into pairs
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]

    # Broadcast cos/sin to match x shape
    # cos, sin: [seq_len, d/2] -> [..., seq_len, d/2]
    cos = cos.view(*([1] * (x.dim() - 2)), cos.shape[0], cos.shape[1])
    sin = sin.view(*([1] * (x.dim() - 2)), sin.shape[0], sin.shape[1])

    # Apply rotation
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return torch.cat([x1_rot, x2_rot], dim=-1)


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Encoding.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for RoPE
        rope_base: Base for RoPE frequencies
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        rope_base: float = 100000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE
        self.rope = RotaryPositionEncoding(
            d_model=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, seq_len, d_model]
            attn_mask: Attention mask [seq_len, seq_len]
            key_padding_mask: Key padding mask [B, seq_len]

        Returns:
            Output tensor [B, seq_len, d_model]
        """
        B, N, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: [B, N, H, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        # Transpose for attention: [B, H, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # [B, N] -> [B, 1, 1, N]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, N, head_dim]

        # Reshape back: [B, N, d_model]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return out
