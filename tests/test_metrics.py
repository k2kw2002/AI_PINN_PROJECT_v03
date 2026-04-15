"""Tests for PSF metrics (MTF@ridge, skewness, throughput, crosstalk)."""
import numpy as np
import pytest
from backend.physics.psf_metrics import (
    compute_mtf_at_ridge, compute_skewness, compute_throughput,
    compute_crosstalk_ratio, compute_all_metrics,
)


def test_mtf_perfect():
    """Ideal PSF: all light in center pixel."""
    psf = np.array([0, 0, 0, 1.0, 0, 0, 0])
    assert compute_mtf_at_ridge(psf) == 1.0


def test_mtf_uniform():
    """Uniform PSF: no contrast."""
    psf = np.ones(7)
    assert abs(compute_mtf_at_ridge(psf)) < 0.01


def test_skewness_symmetric():
    """Symmetric PSF: skewness = 0."""
    psf = np.array([0.1, 0.2, 0.5, 1.0, 0.5, 0.2, 0.1])
    assert abs(compute_skewness(psf)) < 1e-6


def test_skewness_asymmetric():
    """Right-heavy PSF: positive skewness."""
    psf = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5])
    assert compute_skewness(psf) > 0


def test_crosstalk_no_leakage():
    """All light in center: zero crosstalk."""
    psf = np.array([0, 0, 0, 1.0, 0, 0, 0])
    assert compute_crosstalk_ratio(psf) == 0.0


def test_all_metrics_keys():
    """compute_all_metrics returns expected keys."""
    psf = np.array([0.1, 0.2, 0.3, 1.0, 0.3, 0.2, 0.1])
    m = compute_all_metrics(psf)
    assert "mtf_ridge" in m
    assert "skewness" in m
    assert "throughput" in m
    assert "crosstalk" in m
    assert "psf_center" in m
    assert len(m["psf_7"]) == 7
