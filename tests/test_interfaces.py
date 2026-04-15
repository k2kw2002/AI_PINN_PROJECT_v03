"""Tests for module interface contracts (v6 Section 3.5).

Validates unit consistency, dtype consistency, and
data flow between TMM -> ASM -> PINN -> PSF.
"""
import math
import numpy as np
import torch
import pytest
from backend.physics.tmm_calculator import GorillaDXTMM, TMMOutput
from backend.physics.asm_propagator import ASMPropagator
from backend.core.pinn_model import PurePINN
from backend.training.loss_functions import ASMIncidentLUT
from backend.physics.psf_metrics import compute_psf_7, compute_all_metrics
from backend.physics.boundary_conditions import compute_is_bm


def test_tmm_to_asm_interface():
    """TMM output -> ASM initial field: complex amplitude consistent."""
    tmm = GorillaDXTMM()
    asm = ASMPropagator()
    out = tmm.compute(30.0)

    # Round-trip: TMMOutput -> complex -> back
    c = out.to_complex()
    assert abs(abs(c) - out.t_amplitude) < 1e-6
    assert abs(math.degrees(math.atan2(c.imag, c.real)) - out.phase_shift_deg) < 1e-4

    # Initial field should have correct amplitude
    x = np.linspace(0, 504, 100)
    U_init = asm.make_initial_field(out, x)
    assert U_init.dtype == np.complex128
    amp = np.abs(U_init)
    assert abs(amp.mean() - out.t_amplitude) < 0.01


def test_asm_to_pinn_interface():
    """ASM LUT -> PINN collocation: dtype and shape compatible."""
    try:
        lut = ASMIncidentLUT("data/asm_luts/incident_z40.npz")
    except FileNotFoundError:
        pytest.skip("ASM LUT not generated")

    x = torch.linspace(0, 504, 100)
    sin_th = torch.zeros(100)
    U_re, U_im = lut.lookup(x, sin_th)

    assert U_re.dtype == torch.float32
    assert U_re.shape == (100,)
    assert not torch.isnan(U_re).any()
    assert not torch.isnan(U_im).any()


def test_pinn_to_psf_interface():
    """PINN output -> PSF 7 pixels: correct shape and non-negative."""
    model = PurePINN(hidden_dim=32, num_layers=2, num_freqs=16)
    psf = compute_psf_7(model, delta_bm1=0, delta_bm2=0, w1=10, w2=10, theta_deg=0)

    assert psf.shape == (7,)
    assert (psf >= 0).all()
    assert psf.sum() > 0


def test_psf_to_metrics_interface():
    """PSF -> metrics: all values finite."""
    psf = np.array([0.05, 0.1, 0.3, 1.0, 0.3, 0.1, 0.05])
    m = compute_all_metrics(psf)

    assert -1 <= m["mtf_ridge"] <= 1
    assert -1 <= m["skewness"] <= 1
    assert m["throughput"] > 0
    assert m["crosstalk"] >= 0


def test_boundary_condition_consistency():
    """BM regions: slit ratio matches expected geometry."""
    x = torch.linspace(0, 504, 10000)
    delta = torch.zeros(10000)
    w = torch.full((10000,), 10.0)

    is_bm = compute_is_bm(x, delta, w)
    bm_ratio = is_bm.float().mean().item()

    # BM ratio should be ~1 - 7*10/504 = ~0.861
    expected = 1 - 7 * 10 / 504
    assert abs(bm_ratio - expected) < 0.02, f"BM ratio {bm_ratio:.3f}, expected {expected:.3f}"


def test_full_pipeline_no_crash():
    """Full pipeline TMM -> ASM -> PINN -> PSF runs without error."""
    tmm = GorillaDXTMM()
    asm = ASMPropagator()
    model = PurePINN(hidden_dim=32, num_layers=2, num_freqs=16)

    # TMM
    tmm_out = tmm.compute(15.0)
    assert tmm_out.t_amplitude > 0

    # ASM
    x = np.linspace(0, 504, 256)
    U_init = asm.make_initial_field(tmm_out, x)
    U_prop = asm.propagate(U_init, x[1] - x[0])
    assert not np.isnan(U_prop).any()

    # PINN
    psf = compute_psf_7(model, 0, 0, 10, 10, 15)
    assert psf.shape == (7,)

    # Metrics
    metrics = compute_all_metrics(psf)
    assert all(np.isfinite(v) for v in [metrics["mtf_ridge"], metrics["skewness"], metrics["throughput"]])
