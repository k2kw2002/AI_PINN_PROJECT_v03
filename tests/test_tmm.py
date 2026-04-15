"""Tests for TMM calculator (AR coating)."""
import math
import numpy as np
import pytest
from backend.physics.tmm_calculator import GorillaDXTMM, TMMOutput


def test_tmm_normal_incidence():
    """T(0) should be ~98.8% (power transmittance)."""
    tmm = GorillaDXTMM()
    out = tmm.compute(0.0)
    T_power = out.t_amplitude ** 2 * 1.52  # |t|^2 * n_glass/n_air
    assert 0.95 < T_power < 1.01, f"T(0) = {T_power:.4f}, expected ~0.988"


def test_tmm_symmetry():
    """T(+theta) == T(-theta)."""
    tmm = GorillaDXTMM()
    out_pos = tmm.compute(30.0)
    out_neg = tmm.compute(-30.0)
    assert abs(out_pos.t_amplitude - out_neg.t_amplitude) < 1e-6


def test_tmm_output_complex():
    """TMMOutput.to_complex() round-trips correctly."""
    out = TMMOutput(theta_deg=15, wavelength_nm=520, t_amplitude=0.8, phase_shift_deg=10.0)
    c = out.to_complex()
    assert abs(abs(c) - 0.8) < 1e-6
    assert abs(math.degrees(math.atan2(c.imag, c.real)) - 10.0) < 1e-4


def test_tmm_lut_shape():
    """LUT returns correct shapes."""
    tmm = GorillaDXTMM()
    theta = np.arange(-41, 42, 1.0)
    lut = tmm.compute_lut(theta)
    assert lut["t_amplitude"].shape == (83,)
    assert lut["phase_shift_deg"].shape == (83,)
    assert lut["t_complex"].shape == (83,)
