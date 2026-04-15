"""Tests for ASM propagator (Cover Glass)."""
import numpy as np
import pytest
from backend.physics.tmm_calculator import GorillaDXTMM
from backend.physics.asm_propagator import ASMPropagator, generate_incident_lut


@pytest.fixture
def asm():
    return ASMPropagator()


@pytest.fixture
def tmm():
    return GorillaDXTMM()


def test_plane_wave_uniform(asm, tmm):
    """theta=0 after propagation should be uniform amplitude."""
    x = np.linspace(0, 504, 1000)
    tmm_out = tmm.compute(0.0)
    U_init = asm.make_initial_field(tmm_out, x)
    U_prop = asm.propagate(U_init, x[1] - x[0])
    amp = np.abs(U_prop)
    assert amp.std() / amp.mean() < 0.01, "theta=0 should be uniform"


def test_energy_conservation(asm, tmm):
    """Energy should be conserved after propagation."""
    x = np.linspace(0, 504, 2048)
    dx = x[1] - x[0]
    for theta in [0, 15, 30]:
        tmm_out = tmm.compute(theta)
        U_init = asm.make_initial_field(tmm_out, x)
        U_prop = asm.propagate(U_init, dx)
        e_in = np.sum(np.abs(U_init) ** 2) * dx
        e_out = np.sum(np.abs(U_prop) ** 2) * dx
        ratio = e_out / e_in
        assert abs(ratio - 1.0) < 0.01, f"Energy ratio {ratio:.6f} at theta={theta}"


def test_lut_shapes(tmm):
    """LUT generation returns correct shapes."""
    theta = np.arange(-5, 6, 1.0)
    x = np.linspace(0, 504, 256)
    lut = generate_incident_lut(tmm, theta, x)
    assert lut["theta_values"].shape == (11,)
    assert lut["x_values"].shape == (256,)
    assert lut["U_re"].shape == (11, 256)
    assert lut["U_im"].shape == (11, 256)
    assert lut["U_re"].dtype == np.float32


def test_lut_symmetry(tmm):
    """LUT should be symmetric for +theta and -theta amplitude."""
    theta = np.array([-30.0, 0.0, 30.0])
    x = np.linspace(0, 504, 512)
    lut = generate_incident_lut(tmm, theta, x)
    amp_neg = np.sqrt(lut["U_re"][0] ** 2 + lut["U_im"][0] ** 2)
    amp_pos = np.sqrt(lut["U_re"][2] ** 2 + lut["U_im"][2] ** 2)
    assert abs(amp_neg.mean() - amp_pos.mean()) < 0.01
