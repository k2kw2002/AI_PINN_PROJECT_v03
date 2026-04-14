"""Curriculum learning strategy (3-stage).

v6 Section 7: Curriculum 3-Stage
- Stage 1 (0~20%): Boundary learning (L_phase + L_BC)
- Stage 2 (20~60%): PDE activation (+ L_H ramp 0.1 -> 1.0)
- Stage 3 (60~100%): Full loss (+ L_I if data available)

v6 Section 7.4: Optimizer
- First 70%: Adam with CosineAnnealing
- Last 30%: L-BFGS with strong Wolfe

Dependencies:
    pip install torch
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """Configuration for 3-stage curriculum (v6 Section 7)."""

    total_epochs: int = 500         # Total training epochs
    stage1_frac: float = 0.20      # Stage 1: boundary learning
    stage2_frac: float = 0.40      # Stage 2: PDE activation
    # Stage 3: remaining (0.40)

    # Loss weights (v6 Section 6.1)
    lambda_H: float = 1.0          # Helmholtz PDE
    lambda_phase: float = 0.5      # ASM boundary matching
    lambda_BC: float = 0.5         # BM boundary condition
    lambda_I: float = 0.0          # Intensity (disabled until LT data)

    # Stage 2 PDE ramp (v6 Section 7.2)
    lambda_H_start: float = 0.1
    lambda_H_end: float = 1.0

    # Optimizer (v6 Section 7.4)
    lr_adam: float = 1e-3
    lr_adam_min: float = 1e-5
    lbfgs_frac: float = 0.30       # Last 30% uses L-BFGS
    lr_lbfgs: float = 1.0

    @property
    def stage1_end(self) -> int:
        return int(self.total_epochs * self.stage1_frac)

    @property
    def stage2_end(self) -> int:
        return int(self.total_epochs * (self.stage1_frac + self.stage2_frac))

    @property
    def lbfgs_start(self) -> int:
        return int(self.total_epochs * (1.0 - self.lbfgs_frac))


def get_loss_weights(epoch: int, config: CurriculumConfig) -> dict[str, float]:
    """Get loss weights for a given epoch based on curriculum stage.

    Args:
        epoch: Current training epoch.
        config: Curriculum configuration.

    Returns:
        dict with keys: 'lambda_H', 'lambda_phase', 'lambda_BC', 'lambda_I'
    """
    s1_end = config.stage1_end
    s2_end = config.stage2_end

    if epoch < s1_end:
        # Stage 1: boundary learning only (v6 Section 7.1)
        return {
            "lambda_H": 0.0,
            "lambda_phase": config.lambda_phase,
            "lambda_BC": config.lambda_BC,
            "lambda_I": 0.0,
        }
    elif epoch < s2_end:
        # Stage 2: PDE ramp-up (v6 Section 7.2)
        progress = (epoch - s1_end) / (s2_end - s1_end)
        lam_H = config.lambda_H_start + progress * (config.lambda_H_end - config.lambda_H_start)
        return {
            "lambda_H": lam_H,
            "lambda_phase": config.lambda_phase,
            "lambda_BC": config.lambda_BC,
            "lambda_I": 0.0,
        }
    else:
        # Stage 3: full loss (v6 Section 7.3)
        return {
            "lambda_H": config.lambda_H,
            "lambda_phase": config.lambda_phase,
            "lambda_BC": config.lambda_BC,
            "lambda_I": config.lambda_I,
        }


def get_stage_name(epoch: int, config: CurriculumConfig) -> str:
    """Get human-readable stage name."""
    if epoch < config.stage1_end:
        return "Stage 1: Boundary Learning"
    elif epoch < config.stage2_end:
        return "Stage 2: PDE Activation"
    else:
        return "Stage 3: Full Training"
