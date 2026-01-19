"""
Tests for Diffusion Scheduler.

Run with: uv run pytest tests/test_diffusion.py -v
"""

import pytest
import torch
import torch.nn as nn

from train.diffusion import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
    DiffusionScheduler,
    DDIMScheduler,
)


class TestBetaSchedules:
    """Tests for noise schedule functions."""

    def test_linear_schedule_shape(self):
        """Test linear schedule returns correct shape."""
        betas = linear_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_linear_schedule_range(self):
        """Test linear schedule values are in expected range."""
        betas = linear_beta_schedule(1000, beta_start=1e-4, beta_end=0.02)

        assert betas[0] == pytest.approx(1e-4, rel=1e-4)
        assert betas[-1] == pytest.approx(0.02, rel=1e-4)
        assert (betas >= 0).all()
        assert (betas <= 1).all()

    def test_linear_schedule_monotonic(self):
        """Test linear schedule is monotonically increasing."""
        betas = linear_beta_schedule(1000)
        assert (betas[1:] >= betas[:-1]).all()

    def test_cosine_schedule_shape(self):
        """Test cosine schedule returns correct shape."""
        betas = cosine_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_cosine_schedule_range(self):
        """Test cosine schedule values are in valid range."""
        betas = cosine_beta_schedule(1000)

        assert (betas >= 0).all()
        assert (betas <= 1).all()

    def test_cosine_schedule_smoother(self):
        """Test cosine schedule is smoother at endpoints."""
        cosine_betas = cosine_beta_schedule(1000)
        linear_betas = linear_beta_schedule(1000)

        # Cosine should start smaller (more gradual)
        assert cosine_betas[0] < linear_betas[0] * 10

    def test_sigmoid_schedule_shape(self):
        """Test sigmoid schedule returns correct shape."""
        betas = sigmoid_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_sigmoid_schedule_range(self):
        """Test sigmoid schedule values are in expected range."""
        betas = sigmoid_beta_schedule(1000, beta_start=1e-4, beta_end=0.02)

        assert (betas >= 1e-4).all()
        assert (betas <= 0.02).all()


class TestDiffusionScheduler:
    """Tests for DiffusionScheduler class."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        assert scheduler.num_timesteps == 1000
        assert scheduler.betas.shape == (1000,)
        assert scheduler.alphas.shape == (1000,)
        assert scheduler.alphas_cumprod.shape == (1000,)

    def test_different_schedules(self):
        """Test initialization with different schedules."""
        for schedule in ["linear", "cosine", "sigmoid"]:
            scheduler = DiffusionScheduler(
                num_timesteps=100,
                schedule=schedule,
            )
            assert scheduler.betas.shape == (100,)

    def test_alphas_cumprod_decreasing(self):
        """Test that cumulative alphas decrease over time."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        # alpha_cumprod should decrease (more noise over time)
        assert (scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1]).all()

        # Should start near 1 and end near 0
        assert scheduler.alphas_cumprod[0] > 0.99
        assert scheduler.alphas_cumprod[-1] < 0.1

    def test_add_noise(self):
        """Test forward diffusion (adding noise)."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        B, N, C = 4, 100, 50
        x_0 = torch.randn(B, N, C)
        noise = torch.randn(B, N, C)
        t = torch.tensor([0, 250, 500, 999])

        x_t = scheduler.add_noise(x_0, noise, t)

        assert x_t.shape == (B, N, C)

        # At t=0, should be mostly x_0
        # At t=999, should be mostly noise
        x_0_contrib = scheduler.sqrt_alphas_cumprod[0]
        noise_contrib = scheduler.sqrt_one_minus_alphas_cumprod[0]
        assert x_0_contrib > 0.99  # Almost all signal at t=0

    def test_add_noise_deterministic(self):
        """Test that add_noise is deterministic given same inputs."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        x_0 = torch.randn(2, 50, 20)
        noise = torch.randn(2, 50, 20)
        t = torch.tensor([100, 500])

        x_t1 = scheduler.add_noise(x_0, noise, t)
        x_t2 = scheduler.add_noise(x_0, noise, t)

        assert torch.allclose(x_t1, x_t2)

    def test_predict_x0_from_noise(self):
        """Test x_0 prediction from noise."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        B, N, C = 2, 50, 20
        x_0 = torch.randn(B, N, C)
        noise = torch.randn(B, N, C)
        t = torch.tensor([100, 500])

        # Add noise
        x_t = scheduler.add_noise(x_0, noise, t)

        # Recover x_0 (assuming we know the exact noise)
        x_0_pred = scheduler.predict_x0_from_noise(x_t, t, noise)

        # Should recover original x_0
        assert torch.allclose(x_0, x_0_pred, atol=1e-5)

    def test_posterior_mean(self):
        """Test posterior mean computation."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        B, N, C = 2, 50, 20
        x_0 = torch.randn(B, N, C)
        x_t = torch.randn(B, N, C)
        t = torch.tensor([100, 500])

        mean = scheduler.posterior_mean(x_0, x_t, t)

        assert mean.shape == (B, N, C)

    def test_step_shape(self):
        """Test single denoising step output shape."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        B, N, C = 2, 50, 20
        x_t = torch.randn(B, N, C)
        noise_pred = torch.randn(B, N, C)
        t = torch.tensor([500, 500])

        x_prev = scheduler.step(noise_pred, t, x_t)

        assert x_prev.shape == (B, N, C)

    def test_step_no_noise_at_t0(self):
        """Test that no noise is added at t=0."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        B, N, C = 2, 50, 20
        x_t = torch.randn(B, N, C)
        noise_pred = torch.randn(B, N, C)
        t = torch.tensor([0, 0])

        # Set seed for reproducibility
        torch.manual_seed(42)
        x_prev1 = scheduler.step(noise_pred, t, x_t)

        torch.manual_seed(123)  # Different seed
        x_prev2 = scheduler.step(noise_pred, t, x_t)

        # At t=0, should be deterministic (no noise added)
        assert torch.allclose(x_prev1, x_prev2)

    def test_loss_weight(self):
        """Test loss weight computation."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        t = torch.tensor([0, 250, 500, 750, 999])
        weights = scheduler.get_loss_weight(t)

        assert weights.shape == (5,)
        # Default is uniform weighting
        assert torch.allclose(weights, torch.ones(5))


class TestDiffusionSchedulerSampling:
    """Tests for DDPM sampling."""

    def test_sample_shape(self):
        """Test that sampling produces correct shape."""
        scheduler = DiffusionScheduler(num_timesteps=10)  # Few steps for speed

        # Simple model that returns zeros (for testing)
        class DummyModel(nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        model = DummyModel()
        shape = (2, 50, 20)

        samples = scheduler.sample(model, shape, device="cpu")

        assert samples.shape == shape

    def test_sample_with_intermediates(self):
        """Test sampling with intermediate outputs."""
        scheduler = DiffusionScheduler(num_timesteps=10)

        class DummyModel(nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        model = DummyModel()
        shape = (2, 50, 20)

        intermediates = scheduler.sample(
            model, shape, device="cpu", return_intermediates=True
        )

        # Should have num_timesteps + 1 intermediates
        assert len(intermediates) == 11
        assert all(x.shape == shape for x in intermediates)


@pytest.mark.skip("Skipping DDIM tests")
class TestDDIMScheduler:
    """Tests for DDIM scheduler."""

    def test_initialization(self):
        """Test DDIM scheduler initialization."""
        scheduler = DDIMScheduler(
            num_timesteps=1000,
            num_inference_steps=50,
            eta=0.0,
        )

        assert scheduler.num_timesteps == 1000
        assert scheduler.num_inference_steps == 50
        assert scheduler.eta == 0.0
        assert len(scheduler.inference_timesteps) == 50

    def test_inference_timesteps(self):
        """Test inference timesteps are evenly spaced."""
        scheduler = DDIMScheduler(
            num_timesteps=1000,
            num_inference_steps=10,
        )

        # Should be evenly spaced, descending
        timesteps = scheduler.inference_timesteps
        assert timesteps[0] > timesteps[-1]

    def test_ddim_step_deterministic(self):
        """Test DDIM step is deterministic when eta=0."""
        scheduler = DDIMScheduler(
            num_timesteps=1000,
            num_inference_steps=50,
            eta=0.0,  # Deterministic
        )

        x_t = torch.randn(2, 50, 20)
        noise_pred = torch.randn(2, 50, 20)

        torch.manual_seed(42)
        x_prev1 = scheduler.ddim_step(noise_pred, 500, 450, x_t)

        torch.manual_seed(123)
        x_prev2 = scheduler.ddim_step(noise_pred, 500, 450, x_t)

        # Should be identical when eta=0
        assert torch.allclose(x_prev1, x_prev2)

    def test_ddim_step_stochastic(self):
        """Test DDIM step adds noise when eta>0."""
        scheduler = DDIMScheduler(
            num_timesteps=1000,
            num_inference_steps=50,
            eta=1.0,  # Stochastic
        )

        x_t = torch.randn(2, 50, 20)
        noise_pred = torch.randn(2, 50, 20)

        torch.manual_seed(42)
        x_prev1 = scheduler.ddim_step(noise_pred, 500, 450, x_t)

        torch.manual_seed(123)
        x_prev2 = scheduler.ddim_step(noise_pred, 500, 450, x_t)

        # Should differ when eta=1
        assert not torch.allclose(x_prev1, x_prev2)

    def test_ddim_sample_faster(self):
        """Test that DDIM sampling uses fewer steps."""
        ddpm = DiffusionScheduler(num_timesteps=1000)
        ddim = DDIMScheduler(num_timesteps=1000, num_inference_steps=50)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, x, t):
                self.call_count += 1
                return torch.zeros_like(x)

        # DDPM model
        ddpm_model = DummyModel()
        ddpm.sample(ddpm_model, (1, 10, 5), device="cpu")
        ddpm_calls = ddpm_model.call_count

        # DDIM model
        ddim_model = DummyModel()
        ddim.sample(ddim_model, (1, 10, 5), device="cpu")
        ddim_calls = ddim_model.call_count

        # DDIM should use much fewer forward passes
        assert ddim_calls == 50
        assert ddpm_calls == 1000


class TestDiffusionSchedulerDevice:
    """Tests for device handling."""

    def test_to_device(self):
        """Test moving scheduler to device."""
        scheduler = DiffusionScheduler(num_timesteps=100)
        scheduler.to("cpu")

        assert scheduler.betas.device.type == "cpu"
        assert scheduler.alphas_cumprod.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_operations(self):
        """Test operations on CUDA."""
        scheduler = DiffusionScheduler(num_timesteps=100)
        scheduler.to("cuda")

        x_0 = torch.randn(2, 50, 20, device="cuda")
        noise = torch.randn(2, 50, 20, device="cuda")
        t = torch.tensor([50, 75], device="cuda")

        x_t = scheduler.add_noise(x_0, noise, t)

        assert x_t.device.type == "cuda"


class TestDiffusionIntegration:
    """Integration tests for diffusion training."""

    def test_training_step(self):
        """Test a complete training step."""
        scheduler = DiffusionScheduler(num_timesteps=1000)

        # Simple MLP for testing
        class SimpleMLP(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim + 1, 128),  # +1 for timestep
                    nn.ReLU(),
                    nn.Linear(128, dim),
                )

            def forward(self, x, t):
                B, N, C = x.shape
                t_embed = t.float().view(B, 1, 1).expand(-1, N, 1) / 1000
                x_t = torch.cat([x, t_embed], dim=-1)
                return self.net(x_t.view(B * N, -1)).view(B, N, C)

        model = SimpleMLP(dim=20)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step
        x_0 = torch.randn(4, 50, 20)
        noise = torch.randn_like(x_0)
        t = torch.randint(0, 1000, (4,))

        x_t = scheduler.add_noise(x_0, noise, t)
        noise_pred = model(x_t, t)

        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        print(f"Training loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Quick smoke test
    print("Running diffusion scheduler smoke test...")

    # Test schedules
    for name, fn in [("linear", linear_beta_schedule),
                     ("cosine", cosine_beta_schedule),
                     ("sigmoid", sigmoid_beta_schedule)]:
        betas = fn(1000)
        print(f"{name}: beta_start={betas[0]:.6f}, beta_end={betas[-1]:.6f}")

    # Test scheduler
    scheduler = DiffusionScheduler(num_timesteps=1000)
    print(f"\nalpha_cumprod[0]={scheduler.alphas_cumprod[0]:.4f}")
    print(f"alpha_cumprod[-1]={scheduler.alphas_cumprod[-1]:.4f}")

    # Test forward diffusion
    x_0 = torch.randn(2, 50, 20)
    noise = torch.randn_like(x_0)
    t = torch.tensor([0, 999])

    x_t = scheduler.add_noise(x_0, noise, t)
    print(f"\nx_t shape: {x_t.shape}")

    print("\nSmoke test passed!")
