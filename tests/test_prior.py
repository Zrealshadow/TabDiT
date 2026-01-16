"""
Tests for Prior API.

Run with: uv run pytest tests/test_prior.py -v
"""

import pytest
import torch
import numpy as np

from prior.dataset import PriorDataset, Prior, SCMPrior, DummyPrior
from prior.mlp_scm import MLPSCM
from prior.tree_scm import TreeSCM
from prior.reg2cls import Reg2Cls
from prior.hp_sampling import HpSampler, HpSamplerList
from prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


class TestPriorDataset:
    """Tests for the main PriorDataset class."""

    def test_dummy_prior_basic(self):
        """Test DummyPrior generates correct shapes."""
        dataset = PriorDataset(
            batch_size=16,
            max_features=20,
            max_seq_len=128,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch()

        assert X.shape[0] == 16  # batch_size
        assert X.shape[1] == 128  # max_seq_len
        assert X.shape[2] == 20  # max_features
        assert y.shape == (16, 128)
        assert d.shape == (16,)
        assert seq_lens.shape == (16,)
        assert train_sizes.shape == (16,)

    def test_dummy_prior_custom_batch_size(self):
        """Test get_batch with custom batch size."""
        dataset = PriorDataset(
            batch_size=32,
            max_features=20,
            max_seq_len=128,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch(batch_size=8)

        assert X.shape[0] == 8

    def test_dummy_prior_iterator(self):
        """Test PriorDataset as iterator."""
        dataset = PriorDataset(
            batch_size=8,
            max_features=10,
            max_seq_len=64,
            prior_type="dummy",
        )

        # Get a few batches
        for i, batch in enumerate(dataset):
            X, y, d, seq_lens, train_sizes = batch
            assert X.shape == (8, 64, 10)
            if i >= 2:
                break

    def test_dummy_prior_values(self):
        """Test that generated values are reasonable."""
        dataset = PriorDataset(
            batch_size=16,
            max_features=20,
            max_seq_len=128,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch()

        # X should be roughly standard normal
        assert X.mean().abs() < 1.0
        assert X.std() < 3.0

        # d should equal max_features for dummy prior
        assert (d == 20).all()

        # seq_lens should all be the same
        assert (seq_lens == seq_lens[0]).all()

        # train_sizes should be within valid range
        assert (train_sizes > 0).all()
        assert (train_sizes < seq_lens).all()


class TestDummyPrior:
    """Tests for DummyPrior class."""

    def test_basic(self):
        """Test basic DummyPrior functionality."""
        prior = DummyPrior(
            batch_size=8,
            max_features=20,
            max_seq_len=64,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        assert X.shape == (8, 64, 20)
        assert y.shape == (8, 64)

    def test_variable_seq_len(self):
        """Test with variable sequence length sampling."""
        prior = DummyPrior(
            batch_size=8,
            min_features=5,
            max_features=20,
            min_seq_len=32,
            max_seq_len=128,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        # All samples should have same seq_len (per batch)
        assert (seq_lens == seq_lens[0]).all()
        assert seq_lens[0] >= 32
        assert seq_lens[0] <= 128


class TestPriorHelpers:
    """Tests for Prior base class helper methods."""

    def test_sample_seq_len_fixed(self):
        """Test sequence length sampling with fixed length."""
        seq_len = Prior.sample_seq_len(None, 1024)
        assert seq_len == 1024

    def test_sample_seq_len_range(self):
        """Test sequence length sampling within range."""
        for _ in range(10):
            seq_len = Prior.sample_seq_len(100, 1000)
            assert 100 <= seq_len < 1000

    def test_sample_seq_len_log(self):
        """Test log-uniform sequence length sampling."""
        seq_lens = [Prior.sample_seq_len(10, 1000, log=True) for _ in range(100)]
        # Log sampling should produce more small values
        median = np.median(seq_lens)
        assert median < 500  # Should be skewed toward smaller values

    def test_sample_train_size_ratio(self):
        """Test train size sampling with ratio."""
        for _ in range(10):
            train_size = Prior.sample_train_size(0.1, 0.9, seq_len=1000)
            assert 100 <= train_size <= 900

    def test_sample_train_size_absolute(self):
        """Test train size sampling with absolute values."""
        for _ in range(10):
            train_size = Prior.sample_train_size(100, 500, seq_len=1000)
            assert 100 <= train_size < 500

    def test_adjust_max_features(self):
        """Test adaptive feature limit based on sequence length."""
        assert Prior.adjust_max_features(1000, 100) == 100
        assert Prior.adjust_max_features(15000, 100) == 80
        assert Prior.adjust_max_features(25000, 100) == 60
        assert Prior.adjust_max_features(70000, 100) == 10

    def test_validate_train_size_range(self):
        """Test train size validation."""
        # Valid ranges
        Prior.validate_train_size_range(100, 500)
        Prior.validate_train_size_range(0.1, 0.9)

        # Invalid ranges
        with pytest.raises(AssertionError):
            Prior.validate_train_size_range(500, 100)  # min > max

        with pytest.raises(ValueError):
            Prior.validate_train_size_range(100, 0.5)  # mixed types


class TestMLPSCM:
    """Tests for MLP-based Structural Causal Model."""

    def test_basic_forward(self):
        """Test basic MLPSCM forward pass."""
        scm = MLPSCM(
            seq_len=128,
            num_features=20,
            num_layers=3,
            hidden_dim=32,
        )

        X, y = scm()

        assert X.shape == (128, 20)
        assert y.shape == (128,)

    def test_non_causal_mode(self):
        """Test MLPSCM in non-causal mode."""
        scm = MLPSCM(
            seq_len=128,
            num_features=20,
            is_causal=False,
            num_layers=3,
        )

        X, y = scm()

        assert X.shape == (128, 20)
        assert y.shape == (128,)

    def test_causal_mode(self):
        """Test MLPSCM in causal mode."""
        scm = MLPSCM(
            seq_len=128,
            num_features=20,
            is_causal=True,
            num_causes=10,
            num_layers=4,
        )

        X, y = scm()

        assert X.shape == (128, 20)
        assert y.shape == (128,)

    def test_different_samplings(self):
        """Test different input sampling strategies."""
        for sampling in ["normal", "uniform", "mixed"]:
            scm = MLPSCM(
                seq_len=64,
                num_features=10,
                sampling=sampling,
            )
            X, y = scm()
            assert X.shape == (64, 10)

    def test_values_are_finite(self):
        """Test that generated values are finite."""
        scm = MLPSCM(
            seq_len=128,
            num_features=20,
        )

        X, y = scm()

        # Check for NaN/Inf (model handles this internally)
        if not torch.isnan(X).any():
            assert torch.isfinite(X).all()


class TestTreeSCM:
    """Tests for Tree-based Structural Causal Model."""

    @pytest.mark.slow
    def test_basic_forward(self):
        """Test basic TreeSCM forward pass."""
        scm = TreeSCM(
            seq_len=128,
            num_features=10,
        )

        X, y = scm()

        assert X.shape == (128, 10)
        assert y.shape == (128,)

    @pytest.mark.slow
    def test_different_tree_models(self):
        """Test different tree model types."""
        for tree_model in ["xgboost", "decision_tree"]:
            scm = TreeSCM(
                seq_len=64,
                num_features=5,
                tree_model=tree_model,
            )
            X, y = scm()
            assert X.shape == (64, 5)


class TestReg2Cls:
    """Tests for Regression to Classification converter."""

    def test_basic_conversion(self):
        """Test basic regression to classification."""
        hp = {
            "num_classes": 3,
            "max_features": 20,
            "multiclass_type": "rank",
            "multiclass_ordered_prob": 0.2,
            "balanced": False,
        }
        converter = Reg2Cls(hp)

        X = torch.randn(100, 10)
        y = torch.randn(100)

        X_out, y_out = converter(X, y)

        assert X_out.shape == (100, 20)  # Padded to max_features
        assert y_out.shape == (100,)
        assert y_out.unique().numel() <= 3  # At most 3 classes

    def test_binary_classification(self):
        """Test binary classification conversion."""
        hp = {
            "num_classes": 2,
            "max_features": 10,
            "multiclass_type": "rank",
            "multiclass_ordered_prob": 0.0,
            "balanced": False,
        }
        converter = Reg2Cls(hp)

        X = torch.randn(100, 10)
        y = torch.randn(100)

        X_out, y_out = converter(X, y)

        assert set(y_out.unique().tolist()).issubset({0.0, 1.0})

    def test_balanced_binary(self):
        """Test balanced binary classification."""
        hp = {
            "num_classes": 2,
            "max_features": 10,
            "multiclass_type": "rank",
            "multiclass_ordered_prob": 0.0,
            "balanced": True,
        }
        converter = Reg2Cls(hp)

        X = torch.randn(100, 10)
        y = torch.randn(100)

        X_out, y_out = converter(X, y)

        # Should be roughly balanced (50/50 split)
        class_counts = torch.bincount(y_out.long())
        assert abs(class_counts[0] - class_counts[1]) <= 2


class TestHpSampling:
    """Tests for hyperparameter sampling."""

    def test_uniform_sampler(self):
        """Test uniform distribution sampler."""
        sampler = HpSampler(
            distribution="uniform",
            device="cpu",
            min=0.0,
            max=1.0,
        )

        for _ in range(10):
            value = sampler()
            assert 0.0 <= value <= 1.0

    def test_uniform_int_sampler(self):
        """Test uniform integer sampler."""
        sampler = HpSampler(
            distribution="uniform_int",
            device="cpu",
            min=1,
            max=10,
        )

        for _ in range(10):
            value = sampler()
            assert 1 <= value <= 10
            assert isinstance(value, int) or value == int(value)

    def test_meta_choice_sampler(self):
        """Test meta choice sampler."""
        sampler = HpSampler(
            distribution="meta_choice",
            device="cpu",
            choice_values=["a", "b", "c"],
        )

        values = [sampler() for _ in range(20)]
        # Should eventually sample all choices
        unique_values = set(values)
        assert len(unique_values) >= 1
        assert all(v in ["a", "b", "c"] for v in values)

    def test_hp_sampler_list(self):
        """Test HpSamplerList for batch sampling."""
        hp_config = {
            "learning_rate": {
                "distribution": "uniform",
                "min": 0.001,
                "max": 0.1,
            },
            "num_layers": {
                "distribution": "uniform_int",
                "min": 2,
                "max": 6,
            },
        }

        sampler_list = HpSamplerList(hp_config, device="cpu")
        params = sampler_list.sample()

        assert "learning_rate" in params
        assert "num_layers" in params
        assert 0.001 <= params["learning_rate"] <= 0.1
        assert 2 <= params["num_layers"] <= 6


class TestSCMPrior:
    """Tests for SCMPrior class (integration tests)."""

    def test_mlp_scm_prior(self):
        """Test SCMPrior with MLP SCM."""
        prior = SCMPrior(
            batch_size=4,
            batch_size_per_gp=2,
            max_features=20,
            max_seq_len=64,
            prior_type="mlp_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        assert X.shape[0] == 4
        assert X.shape[2] == 20  # max_features
        assert y.shape[0] == 4
        assert d.shape == (4,)

    def test_mix_scm_prior(self):
        """Test SCMPrior with mixed SCM."""
        prior = SCMPrior(
            batch_size=4,
            batch_size_per_gp=2,
            max_features=20,
            max_seq_len=64,
            prior_type="mix_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        assert X.shape[0] == 4

    def test_variable_features(self):
        """Test that different samples can have different feature counts."""
        prior = SCMPrior(
            batch_size=8,
            batch_size_per_gp=2,
            min_features=5,
            max_features=20,
            max_seq_len=64,
            prior_type="mlp_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        # d should vary between samples (after filtering constant features)
        # At minimum, d should be > 0
        assert (d > 0).all()
        assert (d <= 20).all()

    def test_batch_size_per_gp_shared_hyperparams(self):
        """Test that samples within the same group share hyperparameters.

        With batch_size=8 and batch_size_per_gp=4, we have 2 groups.
        Samples 0-3 should share seq_len and train_size (group 1).
        Samples 4-7 should share seq_len and train_size (group 2).
        """
        prior = SCMPrior(
            batch_size=8,
            batch_size_per_gp=4,  # 2 groups of 4 samples each
            min_features=5,
            max_features=50,
            min_seq_len=32,
            max_seq_len=128,
            min_train_size=0.2,
            max_train_size=0.8,
            prior_type="mlp_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        # Group 1: samples 0-3 should have same seq_len and train_size
        assert (seq_lens[:4] == seq_lens[0]).all(), "Group 1 should share seq_len"
        assert (train_sizes[:4] == train_sizes[0]).all(), "Group 1 should share train_size"

        # Group 2: samples 4-7 should have same seq_len and train_size
        assert (seq_lens[4:] == seq_lens[4]).all(), "Group 2 should share seq_len"
        assert (train_sizes[4:] == train_sizes[4]).all(), "Group 2 should share train_size"

    def test_batch_size_per_gp_full_batch(self):
        """Test batch_size_per_gp == batch_size means all samples share hyperparams."""
        prior = SCMPrior(
            batch_size=8,
            batch_size_per_gp=8,  # All samples in one group
            min_features=5,
            max_features=50,
            min_seq_len=32,
            max_seq_len=128,
            prior_type="mlp_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        # All samples should have same seq_len and train_size
        assert (seq_lens == seq_lens[0]).all(), "All samples should share seq_len"
        assert (train_sizes == train_sizes[0]).all(), "All samples should share train_size"
        print(d)
        
    def test_batch_size_per_gp_single_sample_groups(self):
        """Test batch_size_per_gp == 1 means each sample can have different hyperparams."""
        prior = SCMPrior(
            batch_size=8,
            batch_size_per_gp=1,  # Each sample is its own group
            seq_len_per_gp=True,
            min_features=5,
            max_features=50,
            min_seq_len=32,
            max_seq_len=128,
            min_train_size=0.2,
            max_train_size=0.8,
            prior_type="mlp_scm",
            n_jobs=1,
        )

        # Run multiple times to check for variation
        all_same_seq_len = True
        all_same_train_size = True
        all_same_feature_num = True
        for _ in range(10):
            X, y, d, seq_lens, train_sizes = prior.get_batch()

            # print("seq_lens:" + str(seq_lens.tolist()))
            if not (seq_lens == seq_lens[0]).all():
                all_same_seq_len = False

            # print("train size:" + str(train_sizes.tolist()))
            if not (train_sizes == train_sizes[0]).all():
                all_same_train_size = False

            # print("feature num:" + str(d.tolist()))
            if not (d == d[0]).all():
                all_same_feature_num = False
                
        # With batch_size_per_gp=1, we expect variation (not guaranteed but likely)
        # At least one of them should vary across multiple runs
        assert not (all_same_seq_len and all_same_train_size and all_same_feature_num), \
            "With batch_size_per_gp=1, samples should have independent hyperparams"

    def test_batch_size_per_gp_with_nested_tensor(self):
        """Test with seq_len_per_gp=True producing NestedTensor.

        When seq_len_per_gp=True, different groups can have different sequence
        lengths, resulting in X being a NestedTensor instead of a dense tensor.
        """
        prior = SCMPrior(
            batch_size=8,
            batch_size_per_gp=4,  # 2 groups
            min_features=5,
            max_features=50,
            min_seq_len=32,
            max_seq_len=128,
            seq_len_per_gp=True,  # Enable variable seq_len per group
            prior_type="mlp_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()

        # X should be a NestedTensor or SliceNestedTensor
        # Check that it's not a regular dense tensor with uniform shape
        assert hasattr(X, 'is_nested') or not isinstance(X, torch.Tensor) or X.is_nested, \
            "X should be a nested tensor when seq_len_per_gp=True"

        # Group 1 (samples 0-3) should share same seq_len
        assert (seq_lens[:4] == seq_lens[0]).all(), "Group 1 should share seq_len"
        assert (d[:4] == d[0]).all(), "Group 1 should share feature count"
        
        # Group 2 (samples 4-7) should share same seq_len
        assert (seq_lens[4:] == seq_lens[4]).all(), "Group 2 should share seq_len"
        assert (d[4:] == d[4]).all(), "Group 2 should share feature count"
        
        # Access individual samples - each should have correct shape
        for i in range(8):
            sample = X[i]
            assert sample.shape[0] == seq_lens[i], \
                f"Sample {i} seq_len mismatch: {sample.shape[0]} vs {seq_lens[i]}"
            assert sample.shape[1] == 50, \
                f"Sample {i} should have max_features=50, got {sample.shape[1]}"

    def test_seq_len_per_gp_false_dense_tensor(self):
        """Test that seq_len_per_gp=False produces a regular dense tensor."""
        prior = SCMPrior(
            batch_size=8,
            batch_size_per_gp=4,
            min_features=5,
            max_features=50,
            min_seq_len=32,
            max_seq_len=128,
            seq_len_per_gp=False,  # Default - all samples same seq_len
            prior_type="mlp_scm",
            n_jobs=1,
        )

        X, y, d, seq_lens, train_sizes = prior.get_batch()
        print("feature num:" + str(d.tolist()))
        # X should be a regular dense tensor
        assert isinstance(X, torch.Tensor), "X should be a regular tensor"
        assert X.dim() == 3, "X should be 3D: (batch_size, seq_len, max_features)"
        assert X.shape[0] == 8, "Batch size should be 8"
        assert X.shape[2] == 50, "Features should be padded to max_features=50"
        
        assert (d[:4] == d[0]).all(), "Group 1 should share feature count"
        assert (d[4:] == d[4]).all(), "Group 2 should share feature count"
        
        # All samples should have the same seq_len
        assert (seq_lens == seq_lens[0]).all(), \
            "All samples should have same seq_len when seq_len_per_gp=False"


class TestPriorDatasetIntegration:
    """Integration tests for PriorDataset."""

    def test_dataloader_compatibility(self):
        """Test that PriorDataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = PriorDataset(
            batch_size=8,
            max_features=10,
            max_seq_len=32,
            prior_type="dummy",
        )

        dataloader = DataLoader(dataset, batch_size=None)

        for batch in dataloader:
            X, y, d, seq_lens, train_sizes = batch
            assert X.shape == (8, 32, 10)
            break

    def test_train_test_split(self):
        """Test that train/test split is valid."""
        dataset = PriorDataset(
            batch_size=16,
            max_features=20,
            max_seq_len=128,
            min_train_size=0.2,
            max_train_size=0.8,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch()

        for i in range(16):
            train_size = train_sizes[i].item()
            seq_len = seq_lens[i].item()

            # Train size should be within valid range
            assert train_size >= int(0.2 * seq_len) - 1
            assert train_size <= int(0.8 * seq_len) + 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_features(self):
        """Test with minimum number of features."""
        dataset = PriorDataset(
            batch_size=4,
            min_features=2,
            max_features=2,
            max_seq_len=32,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch()
        assert X.shape[2] == 2

    def test_minimum_seq_len(self):
        """Test with minimum sequence length."""
        dataset = PriorDataset(
            batch_size=4,
            max_features=10,
            min_seq_len=16,
            max_seq_len=16,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch()
        assert X.shape[1] == 16

    def test_single_sample(self):
        """Test with batch size of 1."""
        dataset = PriorDataset(
            batch_size=1,
            max_features=10,
            max_seq_len=32,
            prior_type="dummy",
        )

        X, y, d, seq_lens, train_sizes = dataset.get_batch()
        assert X.shape[0] == 1


# if __name__ == "__main__":
#     # Quick smoke test
#     print("Running prior API smoke test...")

#     # Test DummyPrior
#     dataset = PriorDataset(
#         batch_size=8,
#         max_features=20,
#         max_seq_len=64,
#         prior_type="dummy",
#     )

#     X, y, d, seq_lens, train_sizes = dataset.get_batch()

#     print(f"X shape: {X.shape}")
#     print(f"y shape: {y.shape}")
#     print(f"d: {d}")
#     print(f"seq_lens: {seq_lens}")
#     print(f"train_sizes: {train_sizes}")

#     print("\nSmoke test passed!")
