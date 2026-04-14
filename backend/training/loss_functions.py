"""Loss functions for PINN training.

v6 Section 6: 4 loss components
- L_H (Helmholtz): PDE residual, lambda_H = 1.0
- L_phase: ASM boundary matching at z=40, lambda_phase = 0.5
- L_BC: BM boundary condition (U=0 at BM), lambda_BC = 0.5
- L_I: Intensity matching (optional), lambda_I = 0.3

Rule: lambda_H >= max(lambda_phase, lambda_BC, lambda_I)

Dependencies:
    pip install torch numpy
"""
from __future__ import annotations

import math

import numpy as np
import torch

from backend.physics.boundary_conditions import (
    compute_is_bm,
    sample_bm2_slit_direct,
    sample_bm_region_direct,
)

K_MEDIUM = 18.37  # wavenumber in um^-1 (v6 Section 4.1: 2*pi*n/lambda)
SIN_MAX = math.sin(math.radians(41.1))


# ── v6 Section 6.2: L_Helmholtz ──────────────────────────────────────

def helmholtz_loss(
    model: torch.nn.Module,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Helmholtz PDE residual loss: nabla^2 U + k^2 U = 0.

    Computes second derivatives via autograd for both Re(U) and Im(U).
    Gradients are taken w.r.t. x (col 0) and z (col 1) only.

    Args:
        model: PurePINN model.
        coords: (N, 8) collocation points (requires_grad will be set internally).

    Returns:
        Scalar loss (mean squared PDE residual).
    """
    coords = coords.detach().requires_grad_(True)
    U = model(coords)
    U_re = U[:, 0:1]
    U_im = U[:, 1:2]

    # First derivatives w.r.t. all 8 inputs
    grads_re = torch.autograd.grad(U_re.sum(), coords, create_graph=True)[0]
    grads_im = torch.autograd.grad(U_im.sum(), coords, create_graph=True)[0]

    # x-derivatives (column 0) and z-derivatives (column 1)
    U_re_x = grads_re[:, 0:1]
    U_re_z = grads_re[:, 1:2]
    U_im_x = grads_im[:, 0:1]
    U_im_z = grads_im[:, 1:2]

    # Second derivatives
    U_re_xx = torch.autograd.grad(U_re_x.sum(), coords, create_graph=True)[0][:, 0:1]
    U_re_zz = torch.autograd.grad(U_re_z.sum(), coords, create_graph=True)[0][:, 1:2]
    U_im_xx = torch.autograd.grad(U_im_x.sum(), coords, create_graph=True)[0][:, 0:1]
    U_im_zz = torch.autograd.grad(U_im_z.sum(), coords, create_graph=True)[0][:, 1:2]

    # PDE residual: nabla^2 U + k^2 U = 0
    k2 = K_MEDIUM ** 2
    res_re = U_re_xx + U_re_zz + k2 * U_re
    res_im = U_im_xx + U_im_zz + k2 * U_im

    return torch.mean(res_re**2 + res_im**2)


# ── v6 Section 6.3: L_phase (ASM matching at z=40) ──────────────────

class ASMIncidentLUT:
    """PINN L_phase target lookup from precomputed ASM LUT (v6 Section 3.5.3).

    Loads incident_z40.npz and provides bilinear interpolation.
    """

    def __init__(self, filepath: str = "data/asm_luts/incident_z40.npz"):
        data = np.load(filepath)
        self.theta_deg = data["theta_values"]  # (N_theta,)
        self.x_um = data["x_values"]           # (N_x,)
        self._U_re = data["U_re"]              # (N_theta, N_x) float32
        self._U_im = data["U_im"]              # (N_theta, N_x) float32

        self._theta_min = self.theta_deg[0]
        self._theta_step = self.theta_deg[1] - self.theta_deg[0]
        self._x_min = self.x_um[0]
        self._x_step = self.x_um[1] - self.x_um[0]
        self._n_theta = len(self.theta_deg)
        self._n_x = len(self.x_um)

        # Pre-convert to torch tensors
        self._lut_re: torch.Tensor | None = None
        self._lut_im: torch.Tensor | None = None

    def _ensure_tensors(self, device: torch.device):
        if self._lut_re is None or self._lut_re.device != device:
            self._lut_re = torch.from_numpy(self._U_re).to(device)
            self._lut_im = torch.from_numpy(self._U_im).to(device)

    def lookup(
        self,
        x_query: torch.Tensor,
        sin_theta_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear interpolation of ASM LUT (v6 Section 3.5.3).

        Args:
            x_query: (N,) x-coordinates in um.
            sin_theta_query: (N,) sin(theta) values.

        Returns:
            (U_re, U_im): each (N,) float32 tensors.
        """
        device = x_query.device
        self._ensure_tensors(device)

        # theta from sin(theta)
        theta_deg = torch.asin(sin_theta_query.clamp(-1, 1)) * (180.0 / math.pi)

        # Theta axis indices
        t_idx = (theta_deg - self._theta_min) / self._theta_step
        t_lo = t_idx.long().clamp(0, self._n_theta - 2)
        t_hi = t_lo + 1
        t_frac = (t_idx - t_lo.float()).clamp(0, 1).unsqueeze(1)

        # X axis indices
        x_idx = (x_query - self._x_min) / self._x_step
        x_lo = x_idx.long().clamp(0, self._n_x - 2)
        x_hi = x_lo + 1
        x_frac = (x_idx - x_lo.float()).clamp(0, 1).unsqueeze(1)

        # Bilinear interpolation for both Re and Im
        def _bilinear(lut):
            v00 = lut[t_lo, x_lo].unsqueeze(1)
            v01 = lut[t_lo, x_hi].unsqueeze(1)
            v10 = lut[t_hi, x_lo].unsqueeze(1)
            v11 = lut[t_hi, x_hi].unsqueeze(1)
            return (
                (1 - t_frac) * (1 - x_frac) * v00
                + (1 - t_frac) * x_frac * v01
                + t_frac * (1 - x_frac) * v10
                + t_frac * x_frac * v11
            ).squeeze(1)

        return _bilinear(self._lut_re), _bilinear(self._lut_im)


def phase_loss(
    model: torch.nn.Module,
    asm_lut: ASMIncidentLUT,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """L_phase: match PINN output at z=40 to ASM result (v6 Section 6.3).

    Uses direct slit sampling (v6 Section 8.3) - only BM2 slit interior
    is matched. BM exterior at z=40 is handled by L_BC.

    Args:
        model: PurePINN model.
        asm_lut: Precomputed ASM incident LUT.
        n_samples: Number of sample points.
        device: Torch device.

    Returns:
        Scalar loss.
    """
    # Random design variables
    d1 = torch.rand(n_samples, device=device) * 20 - 10
    d2 = torch.rand(n_samples, device=device) * 20 - 10
    w1 = torch.rand(n_samples, device=device) * 15 + 5
    w2 = torch.rand(n_samples, device=device) * 15 + 5

    # Random angles
    sin_th = (torch.rand(n_samples, device=device) * 2 - 1) * SIN_MAX
    cos_th = torch.sqrt(1 - sin_th**2)

    # Sample x directly inside BM2 slits (efficient, v6 Section 8.3)
    x = sample_bm2_slit_direct(n_samples, d2, w2, device)
    z = torch.full((n_samples,), 40.0, device=device)

    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)

    # ASM target
    U_asm_re, U_asm_im = asm_lut.lookup(x, sin_th)

    # PINN prediction
    U = model(coords)

    return torch.mean((U[:, 0] - U_asm_re) ** 2 + (U[:, 1] - U_asm_im) ** 2)


# ── v6 Section 6.4: L_BC (BM boundary) ──────────────────────────────

def bm_boundary_loss(
    model: torch.nn.Module,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """L_BC: enforce U=0 at BM opaque regions (v6 Section 6.4).

    Samples BM regions directly at z=20 (BM1) and z=40 (BM2).

    Args:
        model: PurePINN model.
        n_samples: Number of samples per BM layer (total = 2 * n_samples).
        device: Torch device.

    Returns:
        Scalar loss.
    """
    def _bm_loss_at_z(z_val: float, delta_idx: int, w_idx: int) -> torch.Tensor:
        n = n_samples
        d1 = torch.rand(n, device=device) * 20 - 10
        d2 = torch.rand(n, device=device) * 20 - 10
        w1 = torch.rand(n, device=device) * 15 + 5
        w2 = torch.rand(n, device=device) * 15 + 5

        sin_th = (torch.rand(n, device=device) * 2 - 1) * SIN_MAX
        cos_th = torch.sqrt(1 - sin_th**2)

        delta = d1 if delta_idx == 1 else d2
        w = w1 if w_idx == 1 else w2

        # Direct BM region sampling (v6 Section 8.3)
        x = sample_bm_region_direct(n, delta, w, device)
        z = torch.full((n,), z_val, device=device)

        coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
        U = model(coords)
        return torch.mean(U[:, 0] ** 2 + U[:, 1] ** 2)

    loss_bm1 = _bm_loss_at_z(20.0, delta_idx=1, w_idx=1)
    loss_bm2 = _bm_loss_at_z(40.0, delta_idx=2, w_idx=2)
    return (loss_bm1 + loss_bm2) / 2
