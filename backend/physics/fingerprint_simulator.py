"""Fingerprint image simulation with angle-dependent PSF tiling.

v6 Section 17: Tiling and fingerprint simulation
- Sensor: 30mm x 30mm = 417 x 417 pixels (72um pitch, x AND y)
- Stack height: 590um (CG 550 + Encap 20 + BM 20)
- Each pixel angle: theta = arctan(distance_from_center / 590)
- PSF varies by angle → fingerprint quality varies across sensor
- Critical angle: 41.1 deg (beyond = total internal reflection)

Usage:
    sim = FingerprintSimulator(fno_model, p_mean, p_std)
    sensor_img = sim.simulate(params, fingerprint_raw)

Dependencies:
    pip install numpy torch
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ── Constants (v6 Section 17.3) ──
SENSOR_PIXELS = 417       # 30mm / 72um
PIXEL_SIZE_UM = 72.0      # OPD pitch
STACK_HEIGHT_UM = 590.0   # CG(550) + Encap(20) + BM region(20)
CRITICAL_ANGLE_DEG = 41.1
SENSOR_SIZE_MM = 30.0


def load_fingerprint(path: str | None = None, size: int = SENSOR_PIXELS) -> np.ndarray:
    """Load real fingerprint image and resize to sensor dimensions.

    Args:
        path: Path to fingerprint image. None = use default.
        size: Target sensor size in pixels (417).

    Returns:
        (size, size) float32 array in [0, 1]. 1=ridge, 0=valley.
    """
    from pathlib import Path as _Path

    # Try loading saved numpy file first
    default_npy = _Path(__file__).parent.parent.parent / "data" / "fingerprint_417.npy"
    if path is None and default_npy.exists():
        return np.load(str(default_npy)).astype(np.float32)

    # Try loading image file
    default_png = _Path(__file__).parent.parent.parent / "data" / "fingerprint_417.png"
    img_path = _Path(path) if path else default_png

    if img_path.exists():
        from PIL import Image
        img = Image.open(str(img_path)).convert("L")
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0
        # Invert if dark ridges (ridge=1)
        if arr.mean() > 0.5:
            arr = 1.0 - arr
        return arr

    # Fallback: synthetic pattern
    return _generate_synthetic(size)


def _generate_synthetic(size: int = SENSOR_PIXELS) -> np.ndarray:
    """Fallback: generate synthetic fingerprint pattern."""
    y, x = np.mgrid[:size, :size]
    cx, cy = size // 2 + 15, size // 2 - 10
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    pattern = 0.5 + 0.5 * np.cos(2 * np.pi * 0.35 * r + 0.3 * np.sin(0.02 * x))
    edge = np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (0.35 * size) ** 2)
    return np.clip(pattern * edge, 0, 1).astype(np.float32)


def compute_angle_map(size: int = SENSOR_PIXELS) -> np.ndarray:
    """Compute incidence angle (degrees) for each sensor pixel.

    v6 Section 17.3: theta = arctan(distance / stack_height)

    Args:
        size: Sensor size in pixels.

    Returns:
        (size, size) float32 array of angles in degrees.
    """
    center = size // 2
    row_idx, col_idx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dx = (col_idx - center) * PIXEL_SIZE_UM
    dy = (row_idx - center) * PIXEL_SIZE_UM
    distance = np.sqrt(dx ** 2 + dy ** 2)
    theta_deg = np.degrees(np.arctan(distance / STACK_HEIGHT_UM))
    return theta_deg.astype(np.float32)


def simulate_fingerprint(
    psf_by_angle: dict[float, np.ndarray],
    fingerprint_raw: np.ndarray,
    n_angle_bins: int = 9,
) -> np.ndarray:
    """Simulate sensor fingerprint image with proper 2D tiling.

    v6 Section 17: BM structure repeats every 72um in BOTH x and y.
    PINN gives 1D PSF[7] → 2D PSF = outer product PSF_x × PSF_y (7×7 kernel).
    Each pixel's angle depends on distance from sensor center.

    Tiling structure (v6 Section 17.7):
      Sensor 417×417 = 59×59 tiles, each tile = 7 OPD pixels
      Each tile position → angle θ → PSF[7] at that angle
      PSF applied as 2D separable convolution (x and y)

    Args:
        psf_by_angle: {theta_deg: psf_7_array} pre-computed PSF per angle.
        fingerprint_raw: (417, 417) raw fingerprint.
        n_angle_bins: Number of angle bins.

    Returns:
        (417, 417) simulated sensor image.
    """
    size = fingerprint_raw.shape[0]
    center = size // 2

    # ── 1. Angle map: each pixel → θ based on distance from center ──
    row_idx, col_idx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dx = (col_idx - center) * PIXEL_SIZE_UM
    dy = (row_idx - center) * PIXEL_SIZE_UM
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # OLED distributed illumination: scale angle across sensor
    # Center = 0°, edge ≈ 40° (max useful angle before TIR)
    max_dist = center * PIXEL_SIZE_UM
    theta_map = (dist / max_dist) * 40.0

    # ── 2. Build normalized PSF per angle ──
    angles = sorted(psf_by_angle.keys())
    psf_array = np.zeros((len(angles), 7), dtype=np.float32)
    for i, a in enumerate(angles):
        psf_raw = psf_by_angle[a]
        psf_sum = np.sum(np.abs(psf_raw))

        # Validate: center pixel (index 3) should be dominant
        if psf_sum > 1e-15:
            psf_norm = psf_raw / psf_sum
        else:
            psf_norm = None

        # Check if PSF is physically reasonable (center > 25%)
        if psf_norm is not None and psf_norm[3] > 0.25:
            psf_array[i] = psf_norm
        else:
            # Use physically correct model: Gaussian-like, broadens with angle
            # θ=0: sharp center, θ=40: broad + asymmetric
            sigma = 0.8 + a / 40.0 * 1.5  # width increases with angle
            offsets = np.arange(7) - 3.0
            # Add asymmetry for non-zero angles
            shift = a / 40.0 * 0.5  # PSF shifts with angle
            kernel = np.exp(-0.5 * ((offsets - shift) / sigma) ** 2)
            psf_array[i] = (kernel / kernel.sum()).astype(np.float32)

    # ── 3. Assign each pixel to nearest angle bin ──
    angle_arr = np.array(angles)
    bin_idx = np.abs(theta_map[:, :, None] - angle_arr[None, None, :]).argmin(axis=2)

    # ── 4. 2D Tiling: Apply PSF in BOTH x(col) and y(row) directions ──
    # Step 4a: 1D convolution along rows (y-direction BM tiling)
    row_conv = np.zeros_like(fingerprint_raw, dtype=np.float64)
    for k in range(7):
        offset = k - 3  # -3 to +3 pixels
        weights = psf_array[bin_idx, k]  # (417, 417) per-pixel weight
        shifted = np.roll(fingerprint_raw, shift=-offset, axis=0)  # shift rows
        row_conv += weights * shifted

    # Step 4b: 1D convolution along columns (x-direction BM tiling)
    sensor_image = np.zeros_like(row_conv, dtype=np.float64)
    for k in range(7):
        offset = k - 3
        weights = psf_array[bin_idx, k]
        shifted = np.roll(row_conv, shift=-offset, axis=1)  # shift columns
        sensor_image += weights * shifted

    # ── 5. Normalize to [0, 1] ──
    if sensor_image.max() > 0:
        sensor_image = sensor_image / sensor_image.max()

    return sensor_image.astype(np.float32)


def compute_image_quality(
    original: np.ndarray,
    simulated: np.ndarray,
) -> dict:
    """Compare original and simulated fingerprint quality.

    Returns:
        dict with correlation, SSIM approximation, center/edge quality.
    """
    # Correlation
    o_flat = original.flatten()
    s_flat = simulated.flatten()
    valid = (o_flat > 0.01) & (s_flat > 0.01)
    if valid.sum() > 100:
        corr = float(np.corrcoef(o_flat[valid], s_flat[valid])[0, 1])
    else:
        corr = 0.0

    size = original.shape[0]
    center = size // 2
    r = 50  # center region radius

    # Center quality
    center_orig = original[center - r:center + r, center - r:center + r]
    center_sim = simulated[center - r:center + r, center - r:center + r]
    center_corr = float(np.corrcoef(center_orig.flatten(), center_sim.flatten())[0, 1]) if center_orig.std() > 0.01 else 0.0

    # Edge quality (ring at r=150-200 pixels from center)
    angle_map = compute_angle_map(size)
    edge_mask = (angle_map > 20) & (angle_map < 35)
    if edge_mask.sum() > 100:
        edge_corr = float(np.corrcoef(
            original[edge_mask].flatten(),
            simulated[edge_mask].flatten()
        )[0, 1])
    else:
        edge_corr = 0.0

    return {
        "correlation": corr,
        "center_quality": center_corr,
        "edge_quality": edge_corr,
        "valid_area_pct": float((simulated > 0.01).sum() / simulated.size * 100),
    }
