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
                      ├──────────────────────────┐
                      ▼                          │ skip connection
┌─────────────────────────────────────────┐     │
│  Stage 2: Row-wise Interaction          │     │
│  (RowEncoder)                           │     │
│  ─────────────────────────────────────  │     │
│  • Prepend CLS tokens [B, N, C+K, D]    │     │
│  • RoPE Transformer encoder             │     │
│  • Extract CLS: [B, N, K, D]            │     │
│  • Flatten: [B, N, K*D]                 │     │
└─────────────────────┬───────────────────┘     │
                      │                          │
                      ▼                          │
┌─────────────────────────────────────────┐     │
│  Stage 3: Diffusion Transformer         │     │
│  ─────────────────────────────────────  │     │
│  • Add timestep embedding               │     │
│  • Self-attention transformer           │     │
│  • Output: [B, N, K*D]                  │     │
└─────────────────────┬───────────────────┘     │
                      │                          │
                      ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Decoder: Feature Reconstruction                             │
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

### Why RoPE for Rows?

- **Feature position awareness**: Distinguishes feature slots
- **Variable C support**: Generalizes to any number of features
- **No learned parameters**: Geometric encoding

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

### Problem 3: Diffusion on Mixed Data Types

**Challenge**: Tabular data contains continuous and categorical features.

**Potential Solutions**:
1. **Unified continuous space**: Embed categoricals, diffuse in embedding space
2. **Separate diffusion**: Different noise schedules for different types
3. **Score matching**: Categorical-aware score functions

### Problem 4: Preserving Feature Correlations

**Challenge**: Generated features must maintain realistic correlations.

**Solutions**:
1. **Joint diffusion**: Diffuse all features together (current approach)
2. **Attention mechanisms**: Let features attend to each other
3. **SCM training data**: Learn correlation patterns from synthetic SCM data

## Module Structure

```
model/
├── README.md                 # This file
├── __init__.py              # Module exports
├── column_encoder.py        # Stage 1: Column-wise embedding
├── row_encoder.py           # Stage 2: Row-wise interaction
├── diffusion_transformer.py # Stage 3: Diffusion with timestep
├── decoder.py               # Feature reconstruction decoder
├── tabular_diffusion.py     # Main model combining all stages
└── components/
    ├── __init__.py
    ├── set_transformer.py   # Induced self-attention
    ├── rope.py              # Rotary position encoding
    └── timestep_embed.py    # Sinusoidal timestep embedding
```

## Configuration

Default hyperparameters (following TabICL):

```python
config = {
    # Dimensions
    "d_model": 128,              # Feature embedding dimension
    "d_context": 512,            # CLS context dimension (num_cls * d_model)

    # Stage 1: Column Encoder
    "num_inducing_points": 128,  # Set Transformer inducing points
    "column_blocks": 3,          # Number of Set Transformer blocks
    "column_heads": 4,

    # Stage 2: Row Encoder
    "num_cls_tokens": 4,         # CLS tokens per row
    "row_blocks": 3,             # Number of RoPE transformer blocks
    "row_heads": 8,
    "rope_base": 100000,

    # Stage 3: Diffusion Transformer
    "diffusion_blocks": 12,      # Number of transformer blocks
    "diffusion_heads": 4,

    # Decoder
    "decoder_blocks": 3,
    "decoder_heads": 4,

    # Limits
    "max_features": 200,         # Maximum C
    "max_rows": 10000,           # Maximum N (soft limit)

    # Diffusion
    "timesteps": 1000,
}
```

## Usage

```python
from model import TabularDiffusion

# Create model
model = TabularDiffusion(
    d_model=128,
    num_cls_tokens=4,
    max_features=200,
)

# Forward pass (training)
x_noisy = torch.randn(1, 100, 50)  # [B=1, N=100, C=50]
t = torch.randint(0, 1000, (1,))   # timestep
noise_pred = model(x_noisy, t)     # [1, 100, 50]

# Loss
loss = F.mse_loss(noise_pred, actual_noise)
```

## References

- TabICL: [Architecture Overview](../tabICL_Architecture_Overview.md)
- Data Generation: [SCM Overview](../TabICL_Data_Generation_Overview.md)
- Diffusion Models: DDPM, Score-based models
