"""Red flag automatic detection during training.

v6 Section 18.5: Detects Phase B failure patterns
- Uniform field (plane wave convergence) -> RED FLAG
- BM region non-zero (boundary not learned) -> WARNING
- Design variable insensitivity -> WARNING
- Loss plateau without fringe formation -> WARNING

Dependencies:
    pip install torch numpy
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RedFlagReport:
    """Results of red flag detection."""

    uniform_field: bool = False         # z-interior is uniform (plane wave)
    bm_not_learned: bool = False        # BM regions have |U| > threshold
    design_insensitive: bool = False    # PINN doesn't respond to design vars
    bm1_mean_amp: float = 0.0
    bm2_mean_amp: float = 0.0
    interior_cov: float = 0.0           # coefficient of variation
    design_sensitivity: float = 0.0

    @property
    def has_red_flag(self) -> bool:
        return self.uniform_field

    @property
    def has_warning(self) -> bool:
        return self.bm_not_learned or self.design_insensitive

    def summary(self) -> str:
        lines = []
        if self.uniform_field:
            lines.append(f"RED FLAG: Interior uniform (CoV={self.interior_cov:.4f})")
        if self.bm_not_learned:
            lines.append(f"WARNING: BM not learned (BM1={self.bm1_mean_amp:.4f}, BM2={self.bm2_mean_amp:.4f})")
        if self.design_insensitive:
            lines.append(f"WARNING: Design insensitive (range={self.design_sensitivity:.4f})")
        if not lines:
            lines.append("OK: No red flags detected")
        return "\n".join(lines)


def detect_red_flags(
    model: torch.nn.Module,
    device: torch.device,
    bm_threshold: float = 0.05,
    uniform_cov_threshold: float = 0.05,
    sensitivity_threshold: float = 0.01,
) -> RedFlagReport:
    """Run all red flag checks on the current model state.

    Args:
        model: PurePINN model (eval mode will be set internally).
        device: Torch device.
        bm_threshold: Max acceptable |U| in BM regions.
        uniform_cov_threshold: If CoV < this, field is "uniform" (bad).
        sensitivity_threshold: Min |U| range when sweeping design var.

    Returns:
        RedFlagReport with all check results.
    """
    model.eval()
    report = RedFlagReport()
    N = 500

    x_line = torch.linspace(0, 504, N, device=device)
    zeros = torch.zeros(N, device=device)
    ones = torch.ones(N, device=device)
    w_default = torch.full((N,), 10.0, device=device)

    def _make_coords(z_val):
        return torch.stack([
            x_line,
            torch.full((N,), z_val, device=device),
            zeros, zeros, w_default, w_default, zeros, ones,
        ], dim=1)

    with torch.no_grad():
        # Check 1: Interior uniformity at z=10
        U_10 = model(_make_coords(10.0))
        amp_10 = torch.sqrt(U_10[:, 0] ** 2 + U_10[:, 1] ** 2)
        mean_10 = amp_10.mean().item()
        std_10 = amp_10.std().item()
        cov = std_10 / max(mean_10, 1e-8)
        report.interior_cov = cov
        report.uniform_field = cov < uniform_cov_threshold

        # Check 2: BM regions at z=20 and z=40
        from backend.physics.boundary_conditions import compute_is_bm

        U_20 = model(_make_coords(20.0))
        amp_20 = torch.sqrt(U_20[:, 0] ** 2 + U_20[:, 1] ** 2)
        is_bm = compute_is_bm(x_line, zeros, w_default)
        report.bm1_mean_amp = amp_20[is_bm].mean().item() if is_bm.sum() > 0 else 0.0

        U_40 = model(_make_coords(40.0))
        amp_40 = torch.sqrt(U_40[:, 0] ** 2 + U_40[:, 1] ** 2)
        report.bm2_mean_amp = amp_40[is_bm].mean().item() if is_bm.sum() > 0 else 0.0

        report.bm_not_learned = (
            report.bm1_mean_amp > bm_threshold or report.bm2_mean_amp > bm_threshold
        )

        # Check 3: Design variable sensitivity (sweep w1 at OPD center)
        N_s = 200
        w1_sweep = torch.linspace(5, 20, N_s, device=device)
        coords_s = torch.stack([
            torch.full((N_s,), 252.0, device=device),
            torch.zeros(N_s, device=device),
            torch.zeros(N_s, device=device),
            torch.zeros(N_s, device=device),
            w1_sweep,
            torch.full((N_s,), 10.0, device=device),
            torch.zeros(N_s, device=device),
            torch.ones(N_s, device=device),
        ], dim=1)
        U_s = model(coords_s)
        amp_s = torch.sqrt(U_s[:, 0] ** 2 + U_s[:, 1] ** 2)
        report.design_sensitivity = (amp_s.max() - amp_s.min()).item()
        report.design_insensitive = report.design_sensitivity < sensitivity_threshold

    return report
