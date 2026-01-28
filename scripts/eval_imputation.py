"""
Imputation Evaluation Script.

Evaluates imputation methods on a task defined by a config file.
All methods (baselines and diffusion) use the same interface.

Usage:
    # Generate task configs
    uv run python -m tasks.imputation \
        --prior_data_dir data/eval_imputation \
        --save_dir tasks/configs \
        --mode standard

    # Run evaluation with baselines only
    uv run python scripts/eval_imputation.py \
        --task_config tasks/configs/mcar_20.json \
        --methods mean median knn

    # Run evaluation with diffusion model
    uv run python scripts/eval_imputation.py \
        --task_config tasks/configs/mcar_20.json \
        --methods mean knn diffusion \
        --checkpoint outputs/checkpoint.pt
"""

import argparse
import time
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from prior.genload import LoadPriorDataset
from prior.missing import generate_missing_mask, apply_missing_mask
from tasks.imputation import ImputationTaskConfig


@dataclass
class TableData:
    """Single table with ground truth and missing values."""
    X_true: torch.Tensor      # [N, C]
    X_missing: torch.Tensor   # [N, C] with NaN
    mask: torch.Tensor        # [N, C] binary (1=observed, 0=missing)
    num_rows: int
    num_features: int


@dataclass
class ImputationResult:
    """Result of imputation for a single table."""
    method: str
    rmse: float
    mae: float
    r2: float
    num_missing: int
    total_values: int
    time_seconds: float


# =============================================================================
# Imputation Methods - Unified Interface
# =============================================================================

class Imputer(ABC):
    """Base class for all imputation methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        """Impute missing values. Input/output are numpy arrays."""
        pass


class MeanImputer(Imputer):
    name = "mean"

    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        return SimpleImputer(strategy='mean').fit_transform(X_missing)


class MedianImputer(Imputer):
    name = "median"

    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        return SimpleImputer(strategy='median').fit_transform(X_missing)


class ModeImputer(Imputer):
    name = "mode"

    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        return SimpleImputer(strategy='most_frequent').fit_transform(X_missing)


class KNNImputer_(Imputer):
    name = "knn"

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        return KNNImputer(n_neighbors=self.n_neighbors).fit_transform(X_missing)


class HotDeckImputer(Imputer):
    name = "hot_deck"

    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        import pandas as pd
        df = pd.DataFrame(X_missing)
        X_imputed = np.zeros_like(X_missing)
        for col in df.columns:
            sorted_df = df.sort_values(by=col, na_position='last')
            filled = sorted_df[col].ffill().bfill()
            X_imputed[:, col] = filled.reindex(df.index).values
        return X_imputed


class DiffusionImputer(Imputer):
    name = "diffusion"

    def __init__(self, model, num_timesteps: int, use_resampling: bool, device: str):
        self.model = model
        self.num_timesteps = num_timesteps
        self.use_resampling = use_resampling
        self.device = device

    def impute(self, X_missing: np.ndarray) -> np.ndarray:
        from inference import Imputer as DiffImputer, ImputationConfig

        X_tensor = torch.from_numpy(X_missing).float().unsqueeze(0).to(self.device)

        config = ImputationConfig(
            num_timesteps=self.num_timesteps,
            use_resampling=self.use_resampling,
            show_progress=False,
            device=self.device,
            normalize_data=True,
        )
        imputer = DiffImputer(self.model, config)
        X_imputed = imputer.impute(X_tensor)

        return X_imputed.squeeze(0).cpu().numpy()


BASELINE_IMPUTERS = {
    "mean": MeanImputer,
    "median": MedianImputer,
    "mode": ModeImputer,
    "knn": KNNImputer_,
    "hot_deck": HotDeckImputer,
}


# =============================================================================
# Data Loading
# =============================================================================

def load_tables(
    task_config: ImputationTaskConfig,
    max_batches: Optional[int] = None,
    max_tables: Optional[int] = None,
) -> List[TableData]:
    """Load prior data and apply missing mask based on task config."""
    # Apply seed from task config for reproducible masking
    if task_config.missing_config.seed is not None:
        torch.manual_seed(task_config.missing_config.seed)
        np.random.seed(task_config.missing_config.seed)

    dataset = LoadPriorDataset(task_config.prior_data_dir, max_batches=max_batches, verbose = True)
    tables = []
    table_idx = 0

    for X, _, d, seq_lens, _ in dataset:
        batch_size = X.shape[0]

        for i in range(batch_size):
            if max_tables is not None and len(tables) >= max_tables:
                break

            num_rows = seq_lens[i].item()
            num_features = d[i].item()
            X_true = X[i, :num_rows, :num_features]

            missing_config = task_config.get_missing_config(table_idx)
            mask = generate_missing_mask(X_true, missing_config)
            X_missing = apply_missing_mask(X_true, mask)

            tables.append(TableData(
                X_true=X_true,
                X_missing=X_missing,
                mask=mask,
                num_rows=num_rows,
                num_features=num_features,
            ))
            table_idx += 1

        if max_tables is not None and len(tables) >= max_tables:
            break

    return tables


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Evaluate imputation on missing values only."""
    missing = (mask == 0)
    if not missing.any():
        return {"rmse": 0.0, "mae": 0.0, "r2": 1.0, "num_missing": 0}

    y_true = X_true[missing]
    y_pred = X_imputed[missing]

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
        "num_missing": int(missing.sum()),
    }


def run_imputation(imputer: Imputer, table: TableData) -> ImputationResult:
    """Run a single imputer on a table."""
    X_missing = table.X_missing.numpy().copy()

    start = time.time()
    try:
        X_imputed = imputer.impute(X_missing)
    except Exception as e:
        print(f"  Warning: {imputer.name} failed: {e}")
        return None
    elapsed = time.time() - start

    metrics = evaluate(table.X_true.numpy(), X_imputed, table.mask.numpy())

    return ImputationResult(
        method=imputer.name,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        num_missing=metrics["num_missing"],
        total_values=table.num_rows * table.num_features,
        time_seconds=elapsed,
    )


def print_results(results_by_method: Dict[str, List[ImputationResult]]):
    """Print results table."""
    print(f"\n{'=' * 85}")
    print("RESULTS")
    print(f"{'=' * 85}")
    print(f"{'Method':<12} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Miss%':>8} {'Missing Ratio':>18} {'Time(s)':>10}")
    print("-" * 80)

    for method, results in results_by_method.items():
        if not results:
            continue
        total_missing = sum(r.num_missing for r in results)
        total_values = sum(r.total_values for r in results)
        missing_rate = total_missing / total_values * 100
        print(f"{method:<12} "
              f"{np.mean([r.rmse for r in results]):>10.4f} "
              f"{np.mean([r.mae for r in results]):>10.4f} "
              f"{np.mean([r.r2 for r in results]):>10.4f} "
              f"{missing_rate:>8.2f} "
              f"{total_missing:>8}/{total_values:<8} "
              f"{np.mean([r.time_seconds for r in results]):>10.4f}")


def load_diffusion_model(checkpoint_path: str, device: str):
    """Load trained diffusion model."""
    from model.tabular_diffusion import TabularDiffusion, TabularDiffusionConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", None)
    else:
        state_dict = checkpoint
        config = None

    if config is not None:
        if isinstance(config, dict):
            config = TabularDiffusionConfig(**config)
        model = TabularDiffusion(config)
    else:
        model = TabularDiffusion()

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Imputation Evaluation")
    parser.add_argument("--task_config", type=str, required=True,
                        help="Path to task config JSON file")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["mean", "median", "knn"],
                        help="Methods to evaluate: mean, median, mode, knn, hot_deck, diffusion")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Diffusion model checkpoint (required if 'diffusion' in methods)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_timesteps", type=int, default=100,
                        help="Diffusion timesteps")
    parser.add_argument("--use_resampling", action="store_true",
                        help="Use RePaint resampling for diffusion")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_tables", type=int, default=None)
    args = parser.parse_args()

    # Load task config
    task_config = ImputationTaskConfig.load(args.task_config)
    print(f"Task: {task_config}")

    # Load data
    tables = load_tables(task_config, args.max_batches, args.max_tables)
    print(f"Loaded {len(tables)} tables")

    # Build imputers
    imputers: List[Imputer] = []
    for method in args.methods:
        if method == "diffusion":
            if args.checkpoint is None:
                parser.error("--checkpoint required when using diffusion method")
            model = load_diffusion_model(args.checkpoint, args.device)
            imputers.append(DiffusionImputer(model, args.num_timesteps, args.use_resampling, args.device))
        elif method in BASELINE_IMPUTERS:
            imputers.append(BASELINE_IMPUTERS[method]())
        else:
            print(f"Unknown method: {method}, skipping")

    # Run evaluation
    results_by_method: Dict[str, List[ImputationResult]] = {imp.name: [] for imp in imputers}

    print(f"\nEvaluating {len(tables)} tables with methods: {[imp.name for imp in imputers]}")
    for i, table in enumerate(tables):
        for imputer in imputers:
            result = run_imputation(imputer, table)
            if result:
                results_by_method[imputer.name].append(result)

        if (i + 1) % 50 == 0 or i == len(tables) - 1:
            print(f"  Processed {i + 1}/{len(tables)} tables")

    # Print results
    print_results(results_by_method)

    # Summary
    if task_config.name:
        print(f"\nTask: {task_config.name}")
    print(f"Missing: {task_config.missing_type}, rate={task_config.missing_rate}")
    print(f"Tables: {len(tables)}")


if __name__ == "__main__":
    main()
