"""
Training Script for Tabular Diffusion Model

Usage:
    python -m train.run --config default
    python -m train.run --config debug
    python -m train.run --max_steps 10000 --lr 1e-4
"""

from __future__ import annotations

import argparse
import torch

from .config import TrainConfig, get_default_config, get_debug_config
from .trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tabular Diffusion Model")

    # Config preset
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "debug"],
        help="Configuration preset",
    )

    # Training settings
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluation frequency")
    parser.add_argument("--save_every", type=int, default=None, help="Checkpoint frequency")
    parser.add_argument("--log_every", type=int, default=None, help="Logging frequency")

    # Batch settings
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for data generation")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Micro batch size per forward")

    # Data settings
    parser.add_argument("--max_features", type=int, default=None, help="Maximum number of features")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--prior_type", type=str, default=None, help="Prior type: mlp_scm, tree_scm, mix_scm, dummy")

    # Diffusion settings
    parser.add_argument("--num_timesteps", type=int, default=None, help="Number of diffusion timesteps")
    parser.add_argument("--schedule", type=str, default=None, help="Noise schedule: linear, cosine, sigmoid")

    # Optimization settings
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")

    # Hardware settings
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--mixed_precision", type=str, default=None, help="Mixed precision: no, fp16, bf16")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--detect_anomaly", action="store_true", help="Enable torch.autograd.set_detect_anomaly for NaN debugging")

    # Output settings
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project_name", type=str, default=None, help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    # Model settings (passed to model constructor)
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_cls_tokens", type=int, default=4, help="Number of CLS tokens")
    parser.add_argument("--diffusion_blocks", type=int, default=12, help="Number of diffusion blocks")

    return parser.parse_args()


def build_config(args) -> TrainConfig:
    """Build configuration from command line arguments."""

    # Start with preset
    if args.config == "debug":
        config = get_debug_config()
    else:
        config = get_default_config()

    # Override with command line arguments
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.log_every is not None:
        config.log_every = args.log_every
    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.device is not None:
        config.device = args.device
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.seed is not None:
        config.seed = args.seed
    if args.detect_anomaly:
        config.detect_anomaly = True
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.use_wandb:
        config.use_wandb = True
    if args.project_name is not None:
        config.project_name = args.project_name
    if args.run_name is not None:
        config.run_name = args.run_name

    # Data config
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.max_features is not None:
        config.data.max_features = args.max_features
    if args.max_seq_len is not None:
        config.data.max_seq_len = args.max_seq_len
    if args.prior_type is not None:
        config.data.prior_type = args.prior_type

    # Diffusion config
    if args.num_timesteps is not None:
        config.diffusion.num_timesteps = args.num_timesteps
    if args.schedule is not None:
        config.diffusion.schedule = args.schedule

    # Optimization config
    if args.lr is not None:
        config.optim.lr = args.lr
    if args.warmup_steps is not None:
        config.optim.warmup_steps = args.warmup_steps
    if args.weight_decay is not None:
        config.optim.weight_decay = args.weight_decay

    return config


def build_model(args, config: TrainConfig):
    """Build the model."""
    from model.tabular_diffusion import TabularDiffusion, TabularDiffusionConfig

    model_config = TabularDiffusionConfig(
        d_model=args.d_model,
        num_cls_tokens=args.num_cls_tokens,
        diffusion_blocks=args.diffusion_blocks,
        max_features=config.data.max_features,
        max_timesteps=config.diffusion.num_timesteps,
    )

    return TabularDiffusion(config=model_config)


def main():
    args = parse_args()
    config = build_config(args)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("=" * 60)
    print("Tabular Diffusion Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Max features: {config.data.max_features}")
    print(f"Max seq len: {config.data.max_seq_len}")
    print(f"Prior type: {config.data.prior_type}")
    print(f"Num timesteps: {config.diffusion.num_timesteps}")
    print(f"Schedule: {config.diffusion.schedule}")
    print(f"Learning rate: {config.optim.lr}")
    print("=" * 60)

    # Build model
    model = build_model(args, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(model=model, config=config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
