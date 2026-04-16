"""Mock LightTools data generator for testing L_I pipeline.

Generates synthetic intensity profiles at z=0 (OPD plane) using
a simplified analytical model: ASM field at z=40 → geometric BM
shadow → Fraunhofer-like diffraction through slits → z=0 intensity.

This is NOT physically accurate — it's a stand-in for real LT data
to test the L_I training pipeline before LT is available.

When real LT data is obtained, replace these mock files in data/lt_results/.

Dependencies:
    pip install numpy
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _slit_diffraction_1d(
    x: np.ndarray,
    slit_center: float,
    slit_width: float,
    wavelength_um: float = 0.342,  # wavelength in medium
    z_prop: float = 20.0,          # propagation distance
) -> np.ndarray:
    """Simple single-slit Fresnel diffraction pattern."""
    # Fresnel number
    k = 2 * np.pi / wavelength_um
    dx = x - slit_center

    # Near-field approximation: geometric shadow + edge diffraction
    # Inside slit: intensity ~ 1 (with Fresnel oscillations)
    # Outside slit: intensity ~ 0 (with edge ringing)
    half_w = slit_width / 2

    # Fresnel integral approximation
    u1 = np.sqrt(2 * k / (np.pi * z_prop)) * (dx + half_w)
    u2 = np.sqrt(2 * k / (np.pi * z_prop)) * (dx - half_w)

    # Simplified: use tanh for smooth edge transition
    edge_sharpness = np.sqrt(k / z_prop) * 2
    inside = 0.5 * (np.tanh(edge_sharpness * (dx + half_w))
                     - np.tanh(edge_sharpness * (dx - half_w)))

    return np.clip(inside, 0, 1)


def generate_mock_lt_result(
    delta_bm1: float,
    delta_bm2: float,
    w1: float,
    w2: float,
    theta_deg: float,
    n_x: int = 1000,
    x_max: float = 504.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mock intensity profile at z=0 for given BM parameters.

    Model: BM2 aperture (z=40) → 20um propagation → BM1 aperture (z=20)
    → 20um propagation → OPD (z=0). Each BM acts as a slit array.

    Args:
        delta_bm1, delta_bm2: BM offsets (um).
        w1, w2: Slit widths (um).
        theta_deg: Incidence angle (degrees).
        n_x: Number of x samples.
        x_max: Domain width (um).

    Returns:
        (x_coords, intensity): each (n_x,) arrays.
    """
    pitch = 72.0
    x = np.linspace(0, x_max, n_x)

    # AR transmission intensity (from TMM)
    # Approximate: T decreases slightly with angle
    T_ar = 0.99 - 0.002 * (theta_deg / 40.0) ** 2

    # BM2 diffraction (z=40 → z=20, distance=20um)
    I_bm2 = np.zeros_like(x)
    for i in range(7):
        center = i * pitch + pitch / 2 + delta_bm2
        I_bm2 += _slit_diffraction_1d(x, center, w2, z_prop=20.0)

    # BM1 diffraction (z=20 → z=0, distance=20um)
    I_bm1 = np.zeros_like(x)
    for i in range(7):
        center = i * pitch + pitch / 2 + delta_bm1
        I_bm1 += _slit_diffraction_1d(x, center, w1, z_prop=20.0)

    # Combined: light must pass through both BM2 and BM1
    I_combined = I_bm2 * I_bm1 * T_ar

    # Angle effect: shift the pattern laterally
    shift_um = 40.0 * math.tan(math.radians(theta_deg))  # approx lateral shift
    if abs(shift_um) > 0.1:
        I_combined = np.interp(x, x + shift_um, I_combined, left=0, right=0)

    # Normalize to reasonable range
    I_combined = I_combined * 0.65  # scale to match ASM amplitude^2

    return x.astype(np.float32), I_combined.astype(np.float32)


def generate_mock_lt_dataset(
    configs: list[dict],
    output_dir: str = "data/lt_results",
) -> int:
    """Generate mock LT results for a batch of configurations.

    Args:
        configs: List of dicts with keys: sim_id, delta_bm1, delta_bm2, w1, w2, theta_deg.
        output_dir: Output directory.

    Returns:
        Number of files generated.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        x, I = generate_mock_lt_result(
            cfg["delta_bm1"], cfg["delta_bm2"],
            cfg["w1"], cfg["w2"], cfg["theta_deg"],
        )

        # PSF 7 pixels
        psf_7 = np.zeros(7, dtype=np.float32)
        for i in range(7):
            center = i * 72 + 36
            mask = (x >= center - 5) & (x <= center + 5)
            dx = x[1] - x[0]
            psf_7[i] = I[mask].sum() * dx

        np.savez(
            out_path / f"sim_{cfg['sim_id']:04d}.npz",
            intensity=I,
            x_coords=x,
            psf_7=psf_7,
            delta_bm1=np.float32(cfg["delta_bm1"]),
            delta_bm2=np.float32(cfg["delta_bm2"]),
            w1=np.float32(cfg["w1"]),
            w2=np.float32(cfg["w2"]),
            theta_deg=np.float32(cfg["theta_deg"]),
        )

    return len(configs)
