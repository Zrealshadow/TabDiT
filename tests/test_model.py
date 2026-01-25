"""
Tests for TabularDiffusion model.

Run with: uv run pytest tests/test_model.py -v
"""

import pytest
import torch

from model import (
    TabularDiffusion,
    TabularDiffusionSimple,
    TabularDiffusionConfig,
    ColumnEncoder,
    RowEncoder,
    DiffusionTransformer,
    Decoder,
)


class TestTabularDiffusion:
    """Tests for the main TabularDiffusion model."""

    def test_forward_basic(self):
        """Test basic forward pass with default config."""
        model = TabularDiffusion()

        B, N, C = 1, 100, 50
        x = torch.randn(B, N, C)
        t = torch.randint(0, 1000, (B,))

        output = model(x, t)

        assert output.shape == (B, N, C)

    def test_forward_variable_dimensions(self):
        """Test forward pass with various N and C values."""
        model = TabularDiffusion()

        test_cases = [
            (1, 50, 20),    # Small dataset
            (1, 200, 100),  # Medium dataset
            (1, 500, 150),  # Larger dataset
            (2, 100, 50),   # Batch size > 1
        ]

        for B, N, C in test_cases:
            x = torch.randn(B, N, C)
            t = torch.randint(0, 1000, (B,))

            output = model(x, t)

            assert output.shape == (B, N, C), f"Failed for shape ({B}, {N}, {C})"

    def test_custom_config(self):
        """Test model with custom configuration."""
        config = TabularDiffusionConfig(
            d_model=64,
            num_cls_tokens=2,
            column_blocks=2,
            row_blocks=2,
            diffusion_blocks=4,
            decoder_blocks=2,
        )
        model = TabularDiffusion(config)

        B, N, C = 1, 50, 30
        x = torch.randn(B, N, C)
        t = torch.randint(0, 1000, (B,))

        output = model(x, t)

        assert output.shape == (B, N, C)

    def test_kwargs_override(self):
        """Test config override via kwargs."""
        model = TabularDiffusion(d_model=64, diffusion_blocks=4)

        assert model.config.d_model == 64
        assert model.config.diffusion_blocks == 4

    def test_simple_variants(self):
        """Test simplified architecture variants."""
        model = TabularDiffusion(
            use_simple_column_encoder=True,
            use_simple_diffusion=True,
        )

        B, N, C = 1, 100, 50
        x = torch.randn(B, N, C)
        t = torch.randint(0, 1000, (B,))

        output = model(x, t)

        assert output.shape == (B, N, C)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = TabularDiffusion(
            d_model=64,
            diffusion_blocks=2,
        )

        B, N, C = 1, 50, 20
        x = torch.randn(B, N, C, requires_grad=True)
        t = torch.randint(0, 1000, (B,))

        output = model(x, t)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_num_params(self):
        """Test parameter counting."""
        model = TabularDiffusion(d_model=64, diffusion_blocks=4)
        n_params = model.get_num_params()

        assert n_params > 0
        print(f"Model has {n_params:,} parameters")


class TestTabularDiffusionSimple:
    """Tests for the simplified TabularDiffusion model."""
    
    @pytest.mark.skip(reason="Simple variant tests are currently skipped")
    def test_forward_basic(self):
        """Test basic forward pass."""
        model = TabularDiffusionSimple(
            d_model=64,
            num_blocks=4,
        )

        B, N, C = 1, 100, 50
        x = torch.randn(B, N, C)
        t = torch.randint(0, 1000, (B,))

        output = model(x, t)

        assert output.shape == (B, N, C)


class TestColumnEncoder:
    """Tests for Stage 1: Column Encoder."""

    def test_forward(self):
        """Test column encoder forward pass."""
        encoder = ColumnEncoder(d_model=64, num_blocks=2)

        B, N, C = 1, 100, 50
        x = torch.randn(B, N, C)

        output = encoder(x)

        # Output should be [B, N, C, D]
        assert output.shape == (B, N, C, 64)

    def test_variable_columns(self):
        """Test with different number of columns."""
        encoder = ColumnEncoder(d_model=64, num_blocks=2)

        for C in [10, 50, 100]:
            x = torch.randn(1, 100, C)
            output = encoder(x)
            assert output.shape == (1, 100, C, 64)


class TestRowEncoder:
    """Tests for Stage 2: Row Encoder."""

    def test_forward(self):
        """Test row encoder forward pass."""
        encoder = RowEncoder(
            d_model=64,
            num_blocks=2,
            num_cls_tokens=4,
        )

        B, N, C, D = 1, 100, 50, 64
        x = torch.randn(B, N, C, D)

        cls_output, skip = encoder(x)

        # CLS output: [B, N, K*D]
        assert cls_output.shape == (B, N, 4 * 64)

        # Skip: [B, N, C, D]
        assert skip.shape == (B, N, C, D)


class TestDiffusionTransformer:
    """Tests for Stage 3: Diffusion Transformer."""

    def test_forward(self):
        """Test diffusion transformer forward pass."""
        transformer = DiffusionTransformer(
            d_model=256,
            num_blocks=4,
        )

        B, N, D = 1, 100, 256
        x = torch.randn(B, N, D)
        t = torch.randint(0, 1000, (B,))

        output = transformer(x, t)

        assert output.shape == (B, N, D)
    
    def test_timestep_conditioning(self):
        """Test that different timesteps produce different outputs."""
        transformer = DiffusionTransformer(
            d_model=256,
            num_blocks=4,
        )

        # With additive conditioning, different timesteps immediately produce
        # different outputs (no zero-init like adaLN-Zero)
        x = torch.randn(1, 100, 256)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])

        out1 = transformer(x, t1)
        out2 = transformer(x, t2)

        # Different timesteps should produce different outputs
        assert not torch.allclose(out1, out2)


class TestDecoder:
    """Tests for the Decoder."""

    def test_forward(self):
        """Test decoder forward pass."""
        decoder = Decoder(
            d_model=64,
            num_blocks=2,
        )

        B, N, C, D = 1, 100, 50, 64
        context = torch.randn(B, N, 256)  # [B, N, K*D]
        skip = torch.randn(B, N, C, D)    # [B, N, C, D]

        output = decoder(context, skip)

        # Output should be [B, N, C]
        assert output.shape == (B, N, C)

    def test_variable_features(self):
        """Test decoder with different number of features."""
        decoder = Decoder(
            d_model=64,
            num_blocks=2,
        )

        for C in [10, 50, 100, 150]:
            context = torch.randn(1, 100, 256)
            skip = torch.randn(1, 100, C, 64)

            output = decoder(context, skip)

            assert output.shape == (1, 100, C)

    def test_context_dimension_mismatch(self):
        """Test that mismatched context dimension raises an error."""
        decoder = Decoder(
            d_model=64,
            num_blocks=2,
        )

        B, N, C, D = 1, 100, 50, 64
        # Context dim 100 is not divisible by d_model 64
        context = torch.randn(B, N, 100)
        skip = torch.randn(B, N, C, D)

        with pytest.raises(AssertionError, match="Context dim .* must be divisible by d_model"):
            decoder(context, skip)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_training_step(self):
        """Simulate a training step."""
        model = TabularDiffusion(
            d_model=64,
            diffusion_blocks=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Simulate noisy input and target noise
        B, N, C = 1, 100, 50
        x_0 = torch.randn(B, N, C)  # Original data
        noise = torch.randn(B, N, C)  # Added noise
        t = torch.randint(0, 1000, (B,))

        # Simple noise schedule (linear)
        alpha = 1 - t.float() / 1000
        alpha = alpha.view(B, 1, 1)
        x_t = alpha * x_0 + (1 - alpha) * noise  # Noisy data

        # Forward pass
        noise_pred = model(x_t, t)

        # MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        print(f"Training loss: {loss.item():.4f}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test model on CUDA."""
        model = TabularDiffusion(d_model=64, diffusion_blocks=4).cuda()

        x = torch.randn(1, 100, 50).cuda()
        t = torch.randint(0, 1000, (1,)).cuda()

        output = model(x, t)

        assert output.device.type == "cuda"
        assert output.shape == (1, 100, 50)


if __name__ == "__main__":
    # Quick smoke test
    print("Running smoke test...")

    model = TabularDiffusion(
        d_model=64,
        diffusion_blocks=4,
    )

    x = torch.randn(1, 100, 50)
    t = torch.randint(0, 1000, (1,))

    output = model(x, t)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {model.get_num_params():,}")
    print("Smoke test passed!")
