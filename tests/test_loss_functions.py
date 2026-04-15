"""Tests for loss functions (L_H, L_phase, L_BC)."""
import torch
import pytest
from backend.core.pinn_model import PurePINN
from backend.training.loss_functions import helmholtz_loss, phase_loss, bm_boundary_loss, ASMIncidentLUT
from backend.training.collocation_sampler import hierarchical_collocation


@pytest.fixture
def model():
    return PurePINN(hidden_dim=32, num_layers=2, num_freqs=16)


@pytest.fixture
def asm_lut():
    try:
        return ASMIncidentLUT("data/asm_luts/incident_z40.npz")
    except FileNotFoundError:
        pytest.skip("ASM LUT not generated yet")


def test_helmholtz_loss_finite(model):
    """L_H should be a finite positive scalar."""
    coords = hierarchical_collocation(100, torch.device("cpu"))
    loss = helmholtz_loss(model, coords)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_phase_loss_finite(model, asm_lut):
    """L_phase should be a finite positive scalar."""
    loss = phase_loss(model, asm_lut, 50, torch.device("cpu"))
    assert loss.dim() == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_bc_loss_finite(model):
    """L_BC should be a finite non-negative scalar."""
    loss = bm_boundary_loss(model, 50, torch.device("cpu"))
    assert loss.dim() == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_losses_backprop(model, asm_lut):
    """All losses should allow gradient backpropagation."""
    coords = hierarchical_collocation(50, torch.device("cpu"))
    L_H = helmholtz_loss(model, coords)
    L_ph = phase_loss(model, asm_lut, 30, torch.device("cpu"))
    L_bc = bm_boundary_loss(model, 30, torch.device("cpu"))
    total = L_H + L_ph + L_bc
    total.backward()
    for p in model.parameters():
        assert p.grad is not None
