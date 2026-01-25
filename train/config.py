"""
Training Configuration

Defines configuration classes for training the tabular diffusion model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Configuration for data generation."""

    # Prior dataset settings
    batch_size: int = 56
    batch_size_per_gp: int = 56
    prior_type: str = "mix_scm"  # 'mlp_scm', 'tree_scm', 'mix_scm', 'dummy'

    # Feature settings
    min_features: int = 10
    max_features: int = 50

    # Sequence length settings
    min_seq_len: Optional[int] = 256
    max_seq_len: int = 512
    log_seq_len: bool = False
    seq_len_per_gp: bool = False

    # Train/test split settings
    min_train_size: float = 0.1
    max_train_size: float = 0.9

    # Class settings
    max_classes: int = 10

    # Performance settings
    n_jobs: int = 1  # Avoid nested parallelism with DDP
    num_threads_per_generate: int = 1
    prefetch_factor: int = 4
    num_workers: int = 0

    # Other
    replay_small: bool = False


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""

    # Timesteps
    num_timesteps: int = 1000

    # Noise schedule
    schedule: str = "cosine"  # 'linear', 'cosine', 'sigmoid'
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Sampling
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # DDIM settings
    use_ddim: bool = False
    num_inference_steps: int = 50
    ddim_eta: float = 0.0


@dataclass
class OptimConfig:
    """Configuration for optimization."""

    # Learning rate
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 1000

    # Optimizer
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'linear', 'constant', 'cosine_with_restarts'
    num_cycles: int = 1  # For cosine_with_restarts

    # Gradient
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_after_step: int = 0


@dataclass
class TrainConfig:
    """Main training configuration."""

    # Training steps
    max_steps: int = 100000
    eval_every: int = 500
    save_every: int = 500
    log_every: int = 50

    # Batch settings
    micro_batch_size: int = 64  # Actual batch size per forward pass
    eval_batch_size: int = 64

    # Checkpointing
    output_dir: str = "./outputs"
    resume_from: Optional[str] = None
    save_total_limit: int = 3

    # Logging
    project_name: str = "tabular-diffusion"
    run_name: Optional[str] = None
    use_wandb: bool = False

    # Hardware
    device: str = "cuda"
    seed: int = 42
    mixed_precision: str = "no"  # 'no', 'fp16', 'bf16'
    detect_anomaly: bool = False  # Enable torch.autograd.set_detect_anomaly for NaN debugging

    # DDP settings
    # ddp: bool = False
    # ddp_backend: str = "nccl"

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def get_default_config() -> TrainConfig:
    """Get default training configuration."""
    return TrainConfig()


def get_debug_config() -> TrainConfig:
    """Get debug configuration for quick testing."""
    config = TrainConfig(
        max_steps=100,
        eval_every=50,
        save_every=100,
        log_every=10,
        micro_batch_size=8,
        eval_batch_size=8,
    )
    config.data.batch_size = 32
    config.data.max_features = 20
    config.data.min_seq_len = 32
    config.data.max_seq_len = 128
    config.data.prior_type = "dummy"
    config.diffusion.num_timesteps = 100
    return config
