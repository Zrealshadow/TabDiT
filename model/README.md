# Tabular Diffusion Model Architecture

## Overview

This module implements an unconditional diffusion model for tabular data generation. The architecture is inspired by TabICL but adapted for the diffusion paradigm.

### Target

Train a general unconditional diffusion model on tabular data:
- **Input**: Noisy data `X` with shape `[B, N, C]` where:
  - `B` = batch size (typically 1)
  - `N` = number of rows (samples)
  - `C` = number of columns (features)
- **Output**: Denoised version of `X` with shape `[B, N, C]`
- **Training Data**: Synthetically generated using Structural Causal Models (SCM)

## Architecture

```
Input: X_noisy [B, N, C] + timestep t
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Column-wise Embedding (ColumnEncoder)             │
│  ─────────────────────────────────────────────────────────  │
│  • Linear projection: [B, N, C] → [B, C, N, D]              │
│  • Set Transformer with inducing points (O(n) complexity)   │
│  • Distribution-aware weights & biases                      │
│  • Output: [B, N, C, D]                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Row-wise Interaction (RowEncoder)                 │
│  ─────────────────────────────────────────────────────────  │
│  • Prepend CLS tokens [B, N, K+C, D]                        │
│  • Transformer encoder + subspace position encoding         │
│  • Extract CLS: [B, N, K, D] → flatten: [B, N, K*D]         │
│  • Extract features: [B, N, C, D] (skip connection)         │
└───────────┬─────────────────────────────────┬───────────────┘
            │ CLS [B, N, K*D]                 │ skip [B, N, C, D]
            ▼                                 │
┌─────────────────────────────────────────┐   │
│  Stage 3: Diffusion Transformer  (DiT)  │   │
│  ─────────────────────────────────────  │   │
│  • Add timestep embedding (adaLN-Zero)  │   │
│  • Self-attention transformer           │   │
│  • Output: [B, N, K*D]                  │   │
└─────────────────────┬───────────────────┘   │
                      │                       │
                      ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Decoder: Feature Reconstruction                            │
│  ─────────────────────────────────────────────────────────  │
│  • Cross-attention: skip features query CLS context         │
│  • Self-attention: feature interactions                     │
│  • Output projection: [B, N, C, D] → [B, N, C]              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Output: noise_pred [B, N, C]
```

## Key Design Decisions

### Why Skip Connections?

The CLS tokens in Stage 2 compress per-feature information into a fixed-size representation. For diffusion (which needs to reconstruct exact feature values), we preserve Stage 1 embeddings via skip connections.

### Why Set Transformer for Columns?

- **O(n) complexity**: Uses inducing points instead of O(n²) self-attention
- **Distribution awareness**: Learns column statistics across all rows
- **Permutation invariance**: Row order doesn't affect column embeddings

### Why Subspace Position Encoding for Rows?

We use **subspace mode** (additive learned embeddings) by default for row-wise attention:

- **Column distinction**: Learned positional embeddings help the model distinguish different columns/features
- **Variable C support**: Embeddings are sliced to `[:C]` for any number of features
- **TabPFN-inspired**: Following the successful TabPFN approach for feature position encoding

Alternative: RoPE (Rotary Position Encoding) is also supported but subspace is preferred for tabular data where feature identity matters more than relative position.

### Why Cross-Attention in Decoder?

- **Query**: Skip features (what to reconstruct)
- **Key/Value**: CLS context (global information)
- Allows each feature to selectively retrieve relevant global patterns


## Problems to Solve

### Problem 1: Variable N (Number of Rows)

**Challenge**: Different datasets have different numbers of samples.

**Current Solutions**:
1. **Inducing points** (Set Transformer): Compress N to fixed size
2. **Chunked processing**: Process rows in batches with shared inducing points
3. **Efficient attention**: Use FlashAttention for large N

**Trade-offs**:
| Method | Pros | Cons |
|--------|------|------|
| Inducing points | O(n), proven | Information bottleneck |
| Chunking | Scales to any N | Complex implementation |
| FlashAttention | Preserves all info | O(n²) memory eventually |

### Problem 2: Variable C (Number of Features)

**Challenge**: Different datasets have different numbers of features.

**Current Solutions**:
1. **Shared projections**: Same linear layer for all features
2. **Learnable position embeddings**: `[C_max, D]` sliced to `[:C]`
3. **CLS aggregation**: Fixed number of CLS tokens regardless of C

**Implementation**:
```python
# Feature position embedding handles variable C
self.feature_pos_embed = nn.Parameter(torch.randn(C_max, D))

# At runtime, slice to actual C
feat_pos = self.feature_pos_embed[:C]  # Works for any C <= C_max
```

### Problem 3: Preserving Feature Correlations

**Challenge**: Generated features must maintain realistic correlations.

**Solutions**:
1. **Joint diffusion**: Diffuse all features together (current approach)
2. **Attention mechanisms**: Let features attend to each other
3. **SCM training data**: Learn correlation patterns from synthetic SCM data


## Configuration

Default hyperparameters (from `TabularDiffusionConfig`):

```python
@dataclass
class TabularDiffusionConfig:
    # Dimensions
    d_model: int = 128              # Feature embedding dimension
    num_cls_tokens: int = 4         # CLS tokens (d_context = num_cls * d_model)

    # Stage 1: Column Encoder
    column_blocks: int = 3
    column_heads: int = 4
    num_inducing: int = 128         # Set Transformer inducing points

    # Stage 2: Row Encoder
    row_blocks: int = 3
    row_heads: int = 8
    position_encoding: str = "subspace"  # "subspace" (default) or "rope"
    rope_base: float = 100000.0          # Only used if position_encoding="rope"

    # Stage 3: Diffusion Transformer (DiT)
    diffusion_blocks: int = 12
    diffusion_heads: int = 4

    # Decoder
    decoder_blocks: int = 3
    decoder_heads: int = 4

    # Limits
    max_features: int = 200         # Maximum C
    max_timesteps: int = 1000       # Diffusion timesteps

    # General
    dim_feedforward_mult: int = 2   # FFN hidden dim = d_model * mult
    dropout: float = 0.0

    # Architecture variants
    use_simple_column_encoder: bool = False
    use_simple_diffusion: bool = False
```

**Note**: This model currently focuses on **numerical data only**. Categorical features should be converted to numerical values via preprocessing (e.g., label encoding, one-hot encoding) before being passed to the model.

