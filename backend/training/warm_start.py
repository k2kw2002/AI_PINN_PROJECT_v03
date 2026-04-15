"""Warm Start with ASM approximation (v6 Section 7.5.2).

Initializes PINN weights by pre-training on ASM free-space propagation.
The PINN learns "approximate wave propagation" before the 3-stage curriculum.

Effect (v6):
  Cold Start: ~8 hours GPU
  Warm Start + Stage 1~3: ~4-6 hours (25-40% reduction)

  Cold: Stage 1 may fail to learn boundaries from random init
  Warm: boundaries already roughly correct -> Stage 1 completes fast

Caution:
  - Don't run too long (500 epochs max)
  - Warm start ignores BM -> PINN learns free-space only
  - Stage 1~3 then adds BM boundaries and PDE refinement

Dependencies:
    pip install torch numpy
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch

from backend.training.collocation_sampler import hierarchical_collocation
from backend.training.loss_functions import ASMIncidentLUT

logger = logging.getLogger(__name__)


def _compute_asm_target_at_z(
    x: torch.Tensor,
    z: torch.Tensor,
    sin_th: torch.Tensor,
    asm_lut: ASMIncidentLUT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ASM free-space propagation target at arbitrary (x, z).

    For warm start, we approximate the field at any z as the z=40 field
    propagated downward by (40-z) through free space. This is a simplified
    model that ignores BM boundaries.

    The free-space propagation from z=40 to z adds a phase factor:
      U(x, z) = U(x, 40) * exp(-i * kz * (40 - z))

    For intensity purposes and as an initialization hint, we use the
    z=40 field amplitude scaled by a simple propagation factor.

    Args:
        x: (N,) x-coordinates in um.
        z: (N,) z-coordinates in um.
        sin_th: (N,) sin(theta) values.
        asm_lut: ASM incident LUT at z=40.

    Returns:
        (U_re_target, U_im_target): each (N,) tensors.
    """
    # Get z=40 field from LUT
    U_re_40, U_im_40 = asm_lut.lookup(x, sin_th)

    # Simple propagation: add phase for free-space travel from z=40 to z
    # kz = k * cos(theta), propagation distance = 40 - z
    k = 18.37  # um^-1
    cos_th = torch.sqrt(1 - sin_th**2)
    kz = k * cos_th
    dz = 40.0 - z  # propagation distance (positive = downward)

    # Phase rotation: U(z) = U(40) * exp(-i * kz * dz)
    phase = -kz * dz
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)

    # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    U_re = U_re_40 * cos_phase - U_im_40 * sin_phase
    U_im = U_re_40 * sin_phase + U_im_40 * cos_phase

    return U_re, U_im


def warm_start(
    model: torch.nn.Module,
    asm_lut: ASMIncidentLUT,
    epochs: int = 500,
    n_points: int = 2000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    log_every: int = 50,
) -> torch.nn.Module:
    """Warm start PINN with ASM free-space approximation (v6 Section 7.5.2).

    Pre-trains the PINN to output approximate free-space propagation
    (ignoring BM boundaries). This provides a good initialization for
    the subsequent 3-stage curriculum training.

    Args:
        model: PurePINN model.
        asm_lut: Precomputed ASM incident LUT.
        epochs: Warm start epochs (default 500, don't go much higher).
        n_points: Collocation points per epoch.
        lr: Learning rate.
        device: Torch device.
        log_every: Log frequency.

    Returns:
        model: Warm-started model (same object, modified in-place).
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Warm Start: {epochs} epochs, {n_points} points, lr={lr}")

    for epoch in range(epochs):
        # Sample collocation points across full domain
        coords = hierarchical_collocation(n_points, device)
        x = coords[:, 0]
        z = coords[:, 1]
        sin_th = coords[:, 6]

        # ASM free-space target (no BM)
        U_re_target, U_im_target = _compute_asm_target_at_z(x, z, sin_th, asm_lut)

        # PINN prediction
        U_pinn = model(coords)

        # MSE loss
        loss = torch.mean(
            (U_pinn[:, 0] - U_re_target) ** 2
            + (U_pinn[:, 1] - U_im_target) ** 2
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            logger.info(f"  Warm-start epoch {epoch:4d}/{epochs}: loss={loss.item():.6f}")

    logger.info("Warm Start complete. Proceeding to Stage 1~3 curriculum.")
    return model
