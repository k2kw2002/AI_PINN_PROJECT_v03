"""PSF calculation and metrics (MTF@ridge, skewness, throughput, crosstalk).

v6 Section 3.5.4, 17.1: PSF module
- Input: PINN model + design variables + angle
- Output: PSF 7 pixels, MTF@ridge, skewness, throughput, crosstalk_ratio

PSF pixel layout (v6):
  Index:  0   1   2   3   4   5   6
  Type:   R   V   R   V   R   V   R
  Center: 36  108 180 252 324 396 468  (um)
  Index 3 = center pixel

Dependencies:
    pip install torch numpy
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ── Constants (v6 Section 2.5) ──
OPD_PITCH = 72.0    # um
OPD_WIDTH = 10.0    # um
N_OPD = 7
N_SAMPLES_PER_OPD = 200  # integration samples per pixel


def compute_psf_7(
    model: torch.nn.Module,
    delta_bm1: float,
    delta_bm2: float,
    w1: float,
    w2: float,
    theta_deg: float,
    device: torch.device = torch.device("cpu"),
    n_samples: int = N_SAMPLES_PER_OPD,
) -> np.ndarray:
    """Compute PSF for 7 OPD pixels from PINN model (v6 Section 3.5.4).

    Args:
        model: Trained PurePINN model (eval mode).
        delta_bm1, delta_bm2: BM offsets (um).
        w1, w2: Slit aperture widths (um).
        theta_deg: Incidence angle (degrees).
        device: Torch device.
        n_samples: Integration samples per OPD pixel.

    Returns:
        psf_7: (7,) array, intensity at each OPD pixel.
    """
    model.eval()
    sin_t = math.sin(math.radians(theta_deg))
    cos_t = math.cos(math.radians(theta_deg))
    psf = np.zeros(N_OPD)

    with torch.no_grad():
        for i in range(N_OPD):
            center = i * OPD_PITCH + OPD_PITCH / 2
            x = torch.linspace(
                center - OPD_WIDTH / 2, center + OPD_WIDTH / 2,
                n_samples, device=device,
            )
            coords = torch.stack([
                x,
                torch.zeros(n_samples, device=device),
                torch.full((n_samples,), delta_bm1, device=device),
                torch.full((n_samples,), delta_bm2, device=device),
                torch.full((n_samples,), w1, device=device),
                torch.full((n_samples,), w2, device=device),
                torch.full((n_samples,), sin_t, device=device),
                torch.full((n_samples,), cos_t, device=device),
            ], dim=1)
            U = model(coords)
            intensity = U[:, 0] ** 2 + U[:, 1] ** 2
            psf[i] = intensity.mean().item() * OPD_WIDTH

    return psf


def compute_mtf_at_ridge(psf_7: np.ndarray) -> float:
    """Compute MTF@ridge from 7-pixel PSF.

    MTF@ridge measures how well the sensor resolves ridge/valley pattern.
    Higher is better (max = 1.0).

    Formula: MTF = (V_peak - V_neighbor) / (V_peak + V_neighbor)
    where V_peak = center pixel, V_neighbor = average of adjacent pixels.
    """
    center = psf_7[3]  # center pixel (index 3)
    neighbors = (psf_7[2] + psf_7[4]) / 2  # adjacent R pixels
    if center + neighbors < 1e-12:
        return 0.0
    return float((center - neighbors) / (center + neighbors))


def compute_skewness(psf_7: np.ndarray) -> float:
    """Compute PSF skewness (asymmetry measure).

    Skewness = 0 means perfectly symmetric PSF.
    Lower magnitude is better.

    Compares left (indices 0,1,2) vs right (indices 4,5,6) halves.
    """
    left = psf_7[:3].sum()
    right = psf_7[4:].sum()
    total = psf_7.sum()
    if total < 1e-12:
        return 0.0
    return float((right - left) / total)


def compute_throughput(psf_7: np.ndarray) -> float:
    """Compute throughput (total collected intensity).

    Sum of all 7 OPD pixel intensities, normalized by center pixel
    of a reference design.
    """
    return float(psf_7.sum())


def compute_crosstalk_ratio(psf_7: np.ndarray) -> float:
    """Compute crosstalk ratio.

    Crosstalk = sum of non-center pixels / center pixel.
    Lower is better.
    """
    center = psf_7[3]
    if center < 1e-12:
        return float("inf")
    others = psf_7.sum() - center
    return float(others / center)


def compute_all_metrics(psf_7: np.ndarray) -> dict:
    """Compute all PSF metrics at once.

    Returns:
        dict with keys: mtf_ridge, skewness, throughput, crosstalk,
                        psf_center, psf_7
    """
    return {
        "mtf_ridge": compute_mtf_at_ridge(psf_7),
        "skewness": compute_skewness(psf_7),
        "throughput": compute_throughput(psf_7),
        "crosstalk": compute_crosstalk_ratio(psf_7),
        "psf_center": float(psf_7[3]),
        "psf_7": psf_7.tolist(),
    }


def compute_psf_multi_angle(
    model: torch.nn.Module,
    delta_bm1: float = 0.0,
    delta_bm2: float = 0.0,
    w1: float = 10.0,
    w2: float = 10.0,
    theta_list: list[float] | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Compute PSF and metrics for multiple angles.

    Args:
        model: Trained PurePINN.
        delta_bm1, delta_bm2, w1, w2: Design variables.
        theta_list: List of angles (degrees). Default [0, 15, 30, 40].
        device: Torch device.

    Returns:
        dict: {theta: {metrics_dict}} for each angle.
    """
    if theta_list is None:
        theta_list = [0, 15, 30, 40]

    results = {}
    for theta in theta_list:
        psf = compute_psf_7(model, delta_bm1, delta_bm2, w1, w2, theta, device)
        metrics = compute_all_metrics(psf)
        metrics["theta_deg"] = theta
        results[theta] = metrics

    return results
