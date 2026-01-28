"""
Imputation Task Configuration.

Defines and manages imputation task configurations that combine:
1. Prior data source (directory path)
2. Missing value configuration (MissingConfig from prior.missing)

Uses pydantic for automatic serialization/deserialization.

Usage:
    from prior.missing import MissingConfig, MissingType

    task = ImputationTaskConfig(
        prior_data_dir="data/eval_imputation",
        missing_config=MissingConfig(
            missing_type=MissingType.MCAR,
            missing_rate=0.3,
            seed=42,
        ),
        name="mcar_30",
    )
    task.save("tasks/configs/mcar_30.yaml")

    # Load
    task = ImputationTaskConfig.load("tasks/configs/mcar_30.yaml")
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, List

import yaml
import numpy as np
from pydantic import BaseModel, field_validator, ConfigDict

from prior.missing import MissingType, MissingConfig


# Configuration space for random sampling
MISSING_TYPE_CHOICES = ["mcar", "mar", "mnar"]
MISSING_RATE_RANGE = (0.1, 0.5)
MAR_THRESHOLD_QUANTILE_RANGE = (0.3, 0.7)
MNAR_THRESHOLD_QUANTILE_RANGE = (0.2, 0.5)
MNAR_HIGH_MISSING_PROB_RANGE = (0.4, 0.7)
MNAR_LOW_MISSING_PROB_RANGE = (0.05, 0.2)


class ImputationTaskConfig(BaseModel):
    """
    Configuration for an imputation evaluation task.

    Combines prior data source with missing value configuration.
    Validates that prior_data_dir exists.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prior_data_dir: str
    missing_config: MissingConfig
    name: Optional[str] = None
    description: Optional[str] = None

    @field_validator("prior_data_dir")
    @classmethod
    def validate_prior_data_dir(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            raise ValueError(f"prior_data_dir does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"prior_data_dir is not a directory: {v}")
        return v

    def get_missing_config(self, table_idx: int = 0) -> MissingConfig:
        """
        Get MissingConfig for a specific table.

        Each table gets a unique seed derived from base seed + table_idx.
        """
        base_seed = self.missing_config.seed
        if base_seed is None:
            return self.missing_config

        return MissingConfig(
            missing_type=self.missing_config.missing_type,
            missing_rate=self.missing_config.missing_rate,
            mar_obs_cols=self.missing_config.mar_obs_cols,
            mar_threshold_quantile=self.missing_config.mar_threshold_quantile,
            mnar_threshold_quantile=self.missing_config.mnar_threshold_quantile,
            mnar_high_missing_prob=self.missing_config.mnar_high_missing_prob,
            mnar_low_missing_prob=self.missing_config.mnar_low_missing_prob,
            seed=base_seed + table_idx,
        )

    def save(self, path: str) -> None:
        """Save task config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use pydantic's model_dump for serialization
        data = self.model_dump(mode="json", exclude_none=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Saved task config to {path}")

    @classmethod
    def load(cls, path: str) -> "ImputationTaskConfig":
        """Load task config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    @property
    def missing_type(self) -> str:
        """Shortcut to missing_config.missing_type.value"""
        return self.missing_config.missing_type.value

    @property
    def missing_rate(self) -> float:
        """Shortcut to missing_config.missing_rate"""
        return self.missing_config.missing_rate

    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return (
            f"ImputationTaskConfig({name_str}, "
            f"type={self.missing_type}, rate={self.missing_rate}, "
            f"seed={self.missing_config.seed})"
        )


def sample_missing_config(seed: Optional[int] = None) -> MissingConfig:
    """Sample a random MissingConfig from configuration space."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    missing_type = random.choice(MISSING_TYPE_CHOICES)
    missing_rate = round(np.random.uniform(*MISSING_RATE_RANGE), 3)

    if missing_type == "mar":
        return MissingConfig(
            missing_type=MissingType.MAR,
            missing_rate=missing_rate,
            mar_threshold_quantile=round(np.random.uniform(*MAR_THRESHOLD_QUANTILE_RANGE), 3),
            seed=seed,
        )
    elif missing_type == "mnar":
        return MissingConfig(
            missing_type=MissingType.MNAR,
            missing_rate=missing_rate,
            mnar_threshold_quantile=round(np.random.uniform(*MNAR_THRESHOLD_QUANTILE_RANGE), 3),
            mnar_high_missing_prob=round(np.random.uniform(*MNAR_HIGH_MISSING_PROB_RANGE), 3),
            mnar_low_missing_prob=round(np.random.uniform(*MNAR_LOW_MISSING_PROB_RANGE), 3),
            seed=seed,
        )
    else:  # mcar
        return MissingConfig(
            missing_type=MissingType.MCAR,
            missing_rate=missing_rate,
            seed=seed,
        )


def sample_imputation_task(
    prior_data_dir: str,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> ImputationTaskConfig:
    """Sample a random imputation task configuration."""
    missing_config = sample_missing_config(seed)

    return ImputationTaskConfig(
        prior_data_dir=prior_data_dir,
        missing_config=missing_config,
        name=name,
    )


def sample_imputation_tasks(
    prior_data_dir: str,
    num_tasks: int,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> List[ImputationTaskConfig]:
    """Sample multiple random imputation task configurations."""
    tasks = []

    for i in range(num_tasks):
        task_seed = seed + i
        task = sample_imputation_task(
            prior_data_dir=prior_data_dir,
            seed=task_seed,
            name=f"task_{i:03d}",
        )
        tasks.append(task)

        if save_dir is not None:
            task.save(f"{save_dir}/task_{i:03d}.yaml")

    return tasks


def create_standard_tasks(
    prior_data_dir: str,
    save_dir: Optional[str] = None,
    seed: int = 42,
) -> List[ImputationTaskConfig]:
    """
    Create a standard set of imputation tasks for benchmarking.

    Creates:
    - MCAR at 5%, 10%, 20%, 30% missing rates
    - MAR at 10%, 20%, 30%
    - MNAR at 10%,20%, 30%
    """
    tasks = []

    # MCAR tasks
    for rate in [0.05, 0.1, 0.2, 0.3]:
        task = ImputationTaskConfig(
            prior_data_dir=prior_data_dir,
            missing_config=MissingConfig(
                missing_type=MissingType.MCAR,
                missing_rate=rate,
                seed=seed,
            ),
            name=f"mcar_{int(rate*100)}",
            description=f"MCAR with {int(rate*100)}% missing rate",
        )
        tasks.append(task)

    # MAR tasks
    for rate in [0.1, 0.2, 0.3]:
        task = ImputationTaskConfig(
            prior_data_dir=prior_data_dir,
            missing_config=MissingConfig(
                missing_type=MissingType.MAR,
                missing_rate=rate,
                mar_threshold_quantile=0.5,
                seed=seed,
            ),
            name=f"mar_{int(rate*100)}",
            description=f"MAR with {int(rate*100)}% missing rate",
        )
        tasks.append(task)

    # MNAR tasks
    # Effective rate ≈ threshold_quantile * high_prob + (1 - threshold_quantile) * low_prob
    # With threshold_quantile=0.3 and high/low ratio=5: rate ≈ 2.2 * low_prob
    mnar_configs = [
        (0.1, 0.23, 0.045),  # ~10% missing
        (0.2, 0.45, 0.09),   # ~20% missing
        (0.3, 0.68, 0.14),   # ~30% missing
    ]
    for idx, (rate, high_prob, low_prob) in enumerate(mnar_configs):
        task = ImputationTaskConfig(
            prior_data_dir=prior_data_dir,
            missing_config=MissingConfig(
                missing_type=MissingType.MNAR,
                missing_rate=rate,
                mnar_threshold_quantile=0.3,
                mnar_high_missing_prob=high_prob,
                mnar_low_missing_prob=low_prob,
                seed=seed,
            ),
            name=f"mnar_{idx+1}",
            description=f"MNAR with ~{int(rate*100)}% missing rate",
        )
        tasks.append(task)

    if save_dir is not None:
        for task in tasks:
            task.save(f"{save_dir}/{task.name}.yaml")

    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate imputation task configs")
    parser.add_argument("--prior_data_dir", type=str, required=True,
                        help="Directory containing prior data")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save task configs")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "random"],
                        help="Task generation mode")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of random tasks (for random mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    if args.mode == "standard":
        tasks = create_standard_tasks(
            prior_data_dir=args.prior_data_dir,
            save_dir=args.save_dir,
            seed=args.seed,
        )
        print(f"Created {len(tasks)} standard tasks")
    else:
        tasks = sample_imputation_tasks(
            prior_data_dir=args.prior_data_dir,
            num_tasks=args.num_tasks,
            seed=args.seed,
            save_dir=args.save_dir,
        )
        print(f"Sampled {len(tasks)} random tasks")

    for task in tasks:
        print(f"  {task}")
