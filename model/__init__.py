"""
Tabular Diffusion Model

A diffusion model for unconditional tabular data generation,
inspired by the TabICL architecture.
"""

from .tabular_diffusion import (
    TabularDiffusion,
    TabularDiffusionSimple,
    TabularDiffusionConfig,
)
from .column_encoder import ColumnEncoder, SimpleColumnEncoder
from .row_encoder import RowEncoder, RowEncoderNoCLS
from .diffusion_transformer import DiffusionTransformer, SimpleDiffusionTransformer
from .decoder import Decoder, SimpleDecoder

__all__ = [
    # Main models
    "TabularDiffusion",
    "TabularDiffusionSimple",
    "TabularDiffusionConfig",
    # Encoders
    "ColumnEncoder",
    "SimpleColumnEncoder",
    "RowEncoder",
    "RowEncoderNoCLS",
    # Diffusion
    "DiffusionTransformer",
    "SimpleDiffusionTransformer",
    # Decoder
    "Decoder",
    "SimpleDecoder",
]
