"""Tests for Pure PINN model (8D input, no hard mask)."""
import torch
import pytest
from backend.core.pinn_model import PurePINN


def test_forward_shape():
    """Input (N, 8) -> Output (N, 2)."""
    model = PurePINN(hidden_dim=32, num_layers=2, num_freqs=16)
    coords = torch.randn(50, 8)
    coords[:, 0] = coords[:, 0].abs() * 504
    coords[:, 1] = coords[:, 1].abs() * 40
    coords[:, 4] = coords[:, 4].abs() * 15 + 5
    coords[:, 5] = coords[:, 5].abs() * 15 + 5
    out = model(coords)
    assert out.shape == (50, 2)


def test_no_nan():
    """Output should not contain NaN."""
    model = PurePINN(hidden_dim=32, num_layers=2, num_freqs=16)
    coords = torch.randn(100, 8)
    out = model(coords)
    assert not torch.isnan(out).any()


def test_gradient_flows():
    """Gradients should flow through the model."""
    model = PurePINN(hidden_dim=32, num_layers=2, num_freqs=16)
    coords = torch.randn(10, 8, requires_grad=True)
    out = model(coords)
    loss = out.sum()
    loss.backward()
    assert coords.grad is not None
    assert not torch.isnan(coords.grad).any()


def test_parameter_count():
    """Small model should have reasonable param count."""
    model = PurePINN(hidden_dim=64, num_layers=3, num_freqs=24)
    n = sum(p.numel() for p in model.parameters())
    assert 5000 < n < 50000, f"Param count {n} seems off"
