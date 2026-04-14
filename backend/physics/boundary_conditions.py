"""Boundary condition definitions for PINN domain.

v6 Section 4.3, 6.4: BM boundary conditions
- BM1 (z=20): U=0 at BM regions (outside slit aperture w1)
- BM2 (z=40): U=0 at BM regions (outside slit aperture w2)
- Slit center: i*72 + 36 + delta  (i=0..6)
- Slit half-width: w/2

v6 Section 8.3: Direct sampling for slit/BM regions

Dependencies:
    pip install torch
"""
from __future__ import annotations

import torch

OPD_PITCH = 72.0   # um
N_PITCHES = 7
SLIT_BASE_CENTER = 36.0  # center of pitch = 72/2


def compute_is_bm(
    x: torch.Tensor,
    delta: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Determine if x coordinates fall in BM (opaque) regions.

    A point is in BM if it is outside ALL 7 slit apertures.
    Slit i center: i*72 + 36 + delta, half-width: w/2.

    Args:
        x: (N,) x-coordinates in um.
        delta: (N,) BM offset in um (scalar broadcast OK).
        w: (N,) slit aperture width in um (scalar broadcast OK).

    Returns:
        (N,) bool tensor, True = BM region (opaque).
    """
    in_any_slit = torch.zeros_like(x, dtype=torch.bool)
    for i in range(N_PITCHES):
        center = i * OPD_PITCH + SLIT_BASE_CENTER + delta
        dist = torch.abs(x - center)
        in_any_slit = in_any_slit | (dist < w / 2)
    return ~in_any_slit


def sample_bm2_slit_direct(
    n_samples: int,
    delta_bm2: torch.Tensor,
    w2: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample x-coordinates directly inside BM2 slit apertures (v6 Section 8.3.2).

    Used for L_phase: only slit interior needs ASM matching at z=40.

    Args:
        n_samples: Total number of samples.
        delta_bm2: (n_samples,) or scalar BM2 offset.
        w2: (n_samples,) or scalar slit width.
        device: Torch device.

    Returns:
        (n_samples,) x-coordinates inside BM2 slits.
    """
    # Pick a random slit index per sample
    slit_idx = torch.randint(0, N_PITCHES, (n_samples,), device=device).float()
    center = slit_idx * OPD_PITCH + SLIT_BASE_CENTER + delta_bm2
    # Uniform within slit
    x = center + (torch.rand(n_samples, device=device) - 0.5) * w2
    return x


def sample_bm_region_direct(
    n_samples: int,
    delta: torch.Tensor,
    w: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample x-coordinates directly in BM (opaque) regions (v6 Section 8.3.2).

    Used for L_BC: BM regions where U=0.

    Args:
        n_samples: Total number of samples.
        delta: (n_samples,) or scalar BM offset.
        w: (n_samples,) or scalar slit width.
        device: Torch device.

    Returns:
        (n_samples,) x-coordinates in BM (opaque) regions.
    """
    # Pick a random pitch index
    slit_idx = torch.randint(0, N_PITCHES, (n_samples,), device=device).float()
    center = slit_idx * OPD_PITCH + SLIT_BASE_CENTER + delta
    pitch_start = slit_idx * OPD_PITCH
    pitch_end = (slit_idx + 1) * OPD_PITCH

    slit_left = center - w / 2
    slit_right = center + w / 2

    # BM widths on left/right of slit
    bm_left_w = (slit_left - pitch_start).clamp(min=0)
    bm_right_w = (pitch_end - slit_right).clamp(min=0)
    total_bm_w = bm_left_w + bm_right_w

    # Random position within total BM width
    r = torch.rand(n_samples, device=device) * total_bm_w

    # Map to left or right BM
    x = torch.where(
        r < bm_left_w,
        pitch_start + r,                    # left BM
        slit_right + (r - bm_left_w),       # right BM
    )
    return x.clamp(0, OPD_PITCH * N_PITCHES)
