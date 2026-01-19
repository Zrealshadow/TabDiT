"""
Trainer for Tabular Diffusion Model

Implements the main training loop with:
- On-the-fly data generation using PriorDataset
- Diffusion training (noise prediction)
- Gradient accumulation
- Mixed precision training
- Checkpointing
- Logging
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .config import TrainConfig
from .diffusion import DiffusionScheduler, DDIMScheduler


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
    ):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.step = 0

        # Create shadow parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        self.step += 1
        if self.step <= self.update_after_step:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].lerp_(param.data, 1 - self.decay)

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters (not implemented, use backup)."""
        pass

    def state_dict(self):
        return {"shadow": self.shadow, "step": self.step}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.step = state_dict["step"]


class Trainer:
    """
    Main trainer class for tabular diffusion model.

    Parameters
    ----------
    model : nn.Module
        The tabular diffusion model to train
    config : TrainConfig
        Training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        model.to(self.device)

        # Wrap with DataParallel for multi-GPU training
        if torch.cuda.device_count() > 1 and config.device == "cuda":
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(model)
            self.model_unwrapped = model  # Keep reference to unwrapped model
        else:
            self.model = model
            self.model_unwrapped = model

        # Setup diffusion scheduler
        self._setup_diffusion()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Setup data loader
        self._setup_dataloader()

        # Setup EMA (use unwrapped model for EMA)
        self.ema = None
        if config.optim.use_ema:
            self.ema = EMA(
                self.model_unwrapped,
                decay=config.optim.ema_decay,
                update_after_step=config.optim.ema_update_after_step,
            )

        # Setup mixed precision
        self.scaler = None
        self.autocast_ctx = nullcontext()
        if config.mixed_precision == "fp16":
            self.scaler = GradScaler()
            self.autocast_ctx = autocast(dtype=torch.float16)
        elif config.mixed_precision == "bf16":
            self.autocast_ctx = autocast(dtype=torch.bfloat16)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Setup output directory with run_name subdirectory
        run_name = config.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(config.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()

    def _setup_diffusion(self):
        """Setup diffusion scheduler."""
        cfg = self.config.diffusion

        if cfg.use_ddim:
            self.scheduler = DDIMScheduler(
                num_timesteps=cfg.num_timesteps,
                num_inference_steps=cfg.num_inference_steps,
                eta=cfg.ddim_eta,
                schedule=cfg.schedule,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                clip_sample=cfg.clip_sample,
                clip_sample_range=cfg.clip_sample_range,
            )
        else:
            self.scheduler = DiffusionScheduler(
                num_timesteps=cfg.num_timesteps,
                schedule=cfg.schedule,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                clip_sample=cfg.clip_sample,
                clip_sample_range=cfg.clip_sample_range,
            )

        self.scheduler.to(self.device)

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        cfg = self.config.optim

        # Optimizer
        if cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        elif cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        elif cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        # Learning rate scheduler
        if cfg.scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps - cfg.warmup_steps,
                eta_min=cfg.min_lr,
            )
        elif cfg.scheduler == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=cfg.min_lr / cfg.lr,
                total_iters=self.config.max_steps - cfg.warmup_steps,
            )
        elif cfg.scheduler == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=self.config.max_steps,
            )
        else:
            self.lr_scheduler = None

    def _setup_dataloader(self):
        """Setup data loader with on-the-fly generation."""
        from prior.dataset import PriorDataset

        cfg = self.config.data

        self.dataset = PriorDataset(
            batch_size=cfg.batch_size,
            batch_size_per_gp=cfg.batch_size_per_gp,
            prior_type=cfg.prior_type,
            min_features=cfg.min_features,
            max_features=cfg.max_features,
            max_classes=cfg.max_classes,
            min_seq_len=cfg.min_seq_len,
            max_seq_len=cfg.max_seq_len,
            log_seq_len=cfg.log_seq_len,
            seq_len_per_gp=cfg.seq_len_per_gp,
            min_train_size=cfg.min_train_size,
            max_train_size=cfg.max_train_size,
            replay_small=cfg.replay_small,
            n_jobs=cfg.n_jobs,
            num_threads_per_generate=cfg.num_threads_per_generate,
            device="cpu",  # Generate on CPU, move to GPU in training loop
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=None,  # Dataset returns complete batches
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        )

        self.data_iter = iter(self.dataloader)

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__,
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            self.config.use_wandb = False

    def _get_lr(self) -> float:
        """Get current learning rate with warmup."""
        cfg = self.config.optim

        if self.global_step < cfg.warmup_steps:
            # Linear warmup
            return cfg.lr * (self.global_step + 1) / cfg.warmup_steps
        else:
            # Use scheduler
            if self.lr_scheduler is not None:
                return self.lr_scheduler.get_last_lr()[0]
            return cfg.lr

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_batch(self):
        """Get next batch from data loader.

        Returns
        -------
        tuple
            (X, y, d, seq_lens, train_sizes) tensors on device
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        X, y, d, seq_lens, train_sizes = batch

        # NOTE: there we assume one batch share same sampled hyperparameter
        # so for each element in batch, they share same d/seq_lens/train_sizes

        assert (d[0] == d).all(), f"Inconsistent feature counts across batch {str(d.tolist())}"

        # tailer X to actual_feature from max_features
        actual_feature = d[0].item()
        X = X[:, :, : actual_feature]
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # d = d.to(self.device)
        # seq_lens = seq_lens.to(self.device)
        # train_sizes = train_sizes.to(self.device)

        return X, y, d, seq_lens, train_sizes

    def compute_loss(
        self,
        x_0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Parameters
        ----------
        x_0 : torch.Tensor
            Clean data [B, N, C]
        mask : torch.Tensor, optional
            Mask for valid features [B, C]

        Returns
        -------
        torch.Tensor
            MSE loss for noise prediction
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0, self.scheduler.num_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Sample noise
        noise = torch.randn_like(x_0)

        # Add noise to clean data
        x_t = self.scheduler.add_noise(x_0, noise, t)

        # Predict noise
        with self.autocast_ctx:
            noise_pred = self.model(x_t, t)

        # Compute loss
        if mask is not None:
            # Mask out padded features
            loss = F.mse_loss(noise_pred * mask, noise * mask, reduction="none")
            loss = loss.sum() / mask.sum()
        else:
            loss = F.mse_loss(noise_pred, noise)

        return loss

    def train_step(self) -> Dict[str, float]:
        """Execute single training step with gradient accumulation.

        Returns
        -------
        Dict[str, float]
            Dictionary with loss and other metrics
        """
        self.model.train()

        total_loss = 0.0
        cfg = self.config

        # Gradient accumulation
        for _ in range(cfg.optim.gradient_accumulation_steps):
            # Get batch 
            # In our scenario, only X is used
            X, y, d, seq_lens, train_sizes = self.get_batch()

            # X shape: [B, N, C] where B=batch_size, N=seq_len, C=max_features
            # For diffusion, we treat the entire feature matrix as the data to denoise

            # Create feature mask based on actual number of features
            # d contains actual feature count for each sample
            # feature_mask = torch.arange(X.shape[-1], device=self.device).unsqueeze(0) < d.unsqueeze(-1)
            # feature_mask = feature_mask.unsqueeze(1).float()  # [B, 1, C]
            
            # no feature mask // since all element share same num_features
            feature_mask = None
            
            # Compute loss
            loss = self.compute_loss(X, mask=feature_mask)
            loss = loss / cfg.optim.gradient_accumulation_steps

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # Gradient clipping
        if cfg.optim.max_grad_norm > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                cfg.optim.max_grad_norm,
            )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        # Update learning rate
        lr = self._get_lr()
        self._set_lr(lr)

        if self.lr_scheduler is not None and self.global_step >= cfg.optim.warmup_steps:
            self.lr_scheduler.step()

        return {
            "loss": total_loss * cfg.optim.gradient_accumulation_steps,
            "lr": lr,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on generated samples.

        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        self.model.eval()

        # Use EMA weights for evaluation if available
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss = 0.0
        num_batches = 5

        for _ in range(num_batches):
            # in our scenairo,only X is used
            X, y, d, seq_lens, train_sizes = self.get_batch()

            # feature_mask = torch.arange(X.shape[-1], device=self.device).unsqueeze(0) < d.unsqueeze(-1)
            # feature_mask = feature_mask.unsqueeze(1).float()

            # No feature mask, since all element in batch share same num_features
            feature_mask = None
            
            loss = self.compute_loss(X, mask=feature_mask)
            total_loss += loss.item()

        return {"eval_loss": total_loss / num_batches}

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint.

        Parameters
        ----------
        name : str
            Checkpoint name
        """
        checkpoint = {
            "model_state_dict": self.model_unwrapped.state_dict(),  # Save unwrapped model
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.output_dir / f"{name}_step{self.global_step}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Keep only last N checkpoints
        checkpoints = sorted(self.output_dir.glob(f"{name}_step*.pt"))
        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model_unwrapped.load_state_dict(checkpoint["model_state_dict"])  # Load to unwrapped model
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint.get("epoch", 0)

        if "lr_scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {path} at step {self.global_step}")

    def train(self):
        """Main training loop."""
        cfg = self.config

        # Resume from checkpoint if specified
        if cfg.resume_from is not None:
            self.load_checkpoint(cfg.resume_from)

        print(f"Starting training from step {self.global_step}")
        print(f"Training for {cfg.max_steps} steps")
        # print(f"Model has {sum(p.numel() for p in self.model_unwrapped.parameters()):,} parameters")

        start_time = time.time()
        log_loss = 0.0

        while self.global_step < cfg.max_steps:
            # Training step
            metrics = self.train_step()
            self.global_step += 1

            log_loss += metrics["loss"]

            # Logging
            if self.global_step % cfg.log_every == 0:
                avg_loss = log_loss / cfg.log_every
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed

                print(
                    f"Step {self.global_step}/{cfg.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )

                if self.wandb_run is not None:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": metrics["lr"],
                        "train/steps_per_sec": steps_per_sec,
                    }, step=self.global_step)

                log_loss = 0.0

            # Evaluation
            if self.global_step % cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")

                if self.wandb_run is not None:
                    import wandb
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                    }, step=self.global_step)

            # Save checkpoint
            if self.global_step % cfg.save_every == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint("final")
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

        if self.wandb_run is not None:
            self.wandb_run.finish()
