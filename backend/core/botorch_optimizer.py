"""BoTorch qNEHVI multi-objective optimizer for inverse design.

v6 Section 13.7: Inverse Design
- Input: target specs (MTF_min, skewness_max, throughput_min)
- Uses FNO surrogate for fast evaluation (~1ms per query)
- Multi-objective: maximize MTF, maximize throughput, minimize skewness
- Output: Pareto-optimal design candidates

Pipeline:
  BoTorch proposes (delta1, delta2, w1, w2)
  → FNO predicts PSF[7] (0.8ms)
  → compute metrics (MTF, skew, T)
  → BoTorch updates GP model
  → repeat for N iterations
  → return Pareto front

Dependencies:
    pip install botorch gpytorch torch
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from backend.core.fno_model import FNOSurrogate
from backend.physics.psf_metrics import compute_all_metrics


# Design variable bounds (v6 Section 2.5)
BOUNDS = torch.tensor([
    [-10.0, -10.0, 5.0, 5.0],   # lower
    [10.0, 10.0, 20.0, 20.0],   # upper
], dtype=torch.float64)

# Reference point for hypervolume (worst acceptable values)
# Objectives: MTF (maximize), throughput (maximize), -skewness (maximize = minimize |skew|)
REF_POINT = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)


@dataclass
class InverseDesignResult:
    """Result of inverse design optimization."""
    best_params: np.ndarray       # (4,) best design variables
    best_metrics: dict             # MTF, skewness, throughput, etc.
    pareto_params: np.ndarray     # (n_pareto, 4)
    pareto_objectives: np.ndarray # (n_pareto, 3)
    all_params: np.ndarray        # (n_total, 4) all evaluated
    all_objectives: np.ndarray    # (n_total, 3) all objectives
    n_iterations: int
    elapsed_sec: float


def _eval_design(fno: FNOSurrogate, params_4d: torch.Tensor,
                 p_mean: torch.Tensor, p_std: torch.Tensor,
                 theta_deg: float = 0.0) -> torch.Tensor:
    """Evaluate design using FNO surrogate.

    Args:
        fno: FNO model.
        params_4d: (batch, 4) design variables [d1, d2, w1, w2].
        p_mean, p_std: normalization stats from FNO training.
        theta_deg: fixed angle for evaluation.

    Returns:
        objectives: (batch, 3) [MTF, throughput, -|skewness|]
    """
    batch_size = params_4d.shape[0]
    device = next(fno.parameters()).device

    # Add theta to make 5D input
    theta = torch.full((batch_size, 1), theta_deg, device=device, dtype=params_4d.dtype)
    params_5d = torch.cat([params_4d.to(device), theta], dim=1)

    # Normalize
    params_norm = (params_5d - p_mean.to(device)) / p_std.to(device)

    with torch.no_grad():
        psf_pred = fno(params_norm.float())  # (batch, 7)

    objectives = torch.zeros(batch_size, 3, dtype=torch.float64)
    for i in range(batch_size):
        psf = psf_pred[i].cpu().numpy()
        m = compute_all_metrics(psf)
        objectives[i, 0] = m["mtf_ridge"]
        objectives[i, 1] = m["throughput"]
        objectives[i, 2] = -abs(m["skewness"])  # minimize |skewness| = maximize -|skew|

    return objectives


def run_inverse_design(
    fno_checkpoint: str = "checkpoints/fno_surrogate.pt",
    n_initial: int = 20,
    n_iterations: int = 30,
    batch_size: int = 4,
    theta_deg: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> InverseDesignResult:
    """Run BoTorch multi-objective inverse design.

    Args:
        fno_checkpoint: Path to trained FNO model.
        n_initial: Number of initial Sobol samples.
        n_iterations: Number of BO iterations.
        batch_size: Candidates per iteration.
        theta_deg: Fixed angle for evaluation.
        device: Torch device.

    Returns:
        InverseDesignResult with Pareto-optimal designs.
    """
    t_start = time.time()

    # Load FNO
    ckpt = torch.load(fno_checkpoint, map_location=device, weights_only=False)
    fno = FNOSurrogate().to(device)
    fno.load_state_dict(ckpt["model_state_dict"])
    fno.eval()
    p_mean = ckpt["p_mean"]
    p_std = ckpt["p_std"]

    print(f"FNO loaded from {fno_checkpoint}")
    print(f"Inverse design: {n_initial} initial + {n_iterations} iterations x {batch_size} batch")

    # Initial Sobol samples
    sobol = draw_sobol_samples(bounds=BOUNDS, n=n_initial, q=1).squeeze(1)  # (n_initial, 4)
    train_x = sobol.to(torch.float64)
    train_obj = _eval_design(fno, train_x, p_mean, p_std, theta_deg)

    print(f"Initial {n_initial} samples evaluated")

    # BO iterations
    for it in range(n_iterations):
        # Fit GP models (one per objective)
        models = []
        for j in range(3):
            gp = SingleTaskGP(train_x, train_obj[:, j:j+1])
            models.append(gp)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # GP fitting sometimes fails, continue

        # Acquisition function
        try:
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=REF_POINT,
                X_baseline=train_x,
            )

            # Optimize acquisition
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=BOUNDS,
                q=batch_size,
                num_restarts=5,
                raw_samples=64,
            )

            # Evaluate new candidates
            new_obj = _eval_design(fno, candidates, p_mean, p_std, theta_deg)

            # Update training data
            train_x = torch.cat([train_x, candidates])
            train_obj = torch.cat([train_obj, new_obj])

        except Exception as e:
            print(f"  Iteration {it}: BO step failed ({e}), using random")
            random_x = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * torch.rand(batch_size, 4, dtype=torch.float64)
            new_obj = _eval_design(fno, random_x, p_mean, p_std, theta_deg)
            train_x = torch.cat([train_x, random_x])
            train_obj = torch.cat([train_obj, new_obj])

        if (it + 1) % 10 == 0:
            pareto_mask = is_non_dominated(train_obj)
            n_pareto = pareto_mask.sum().item()
            best_mtf = train_obj[:, 0].max().item()
            print(f"  Iteration {it+1}/{n_iterations}: "
                  f"{len(train_x)} total, {n_pareto} Pareto, best MTF={best_mtf:.4f}")

    elapsed = time.time() - t_start

    # Find Pareto front
    pareto_mask = is_non_dominated(train_obj)
    pareto_x = train_x[pareto_mask].numpy()
    pareto_obj = train_obj[pareto_mask].numpy()

    # Best by MTF
    best_idx = train_obj[:, 0].argmax().item()
    best_params = train_x[best_idx].numpy()
    best_psf_5d = np.append(best_params, theta_deg)

    # Get full metrics for best
    best_obj = train_obj[best_idx].numpy()
    best_metrics = {
        "mtf_ridge": float(best_obj[0]),
        "throughput": float(best_obj[1]),
        "skewness": float(-best_obj[2]),  # convert back
        "delta_bm1": float(best_params[0]),
        "delta_bm2": float(best_params[1]),
        "w1": float(best_params[2]),
        "w2": float(best_params[3]),
    }

    print(f"\nInverse design complete in {elapsed:.1f}s")
    print(f"  Total evaluations: {len(train_x)}")
    print(f"  Pareto front: {len(pareto_x)} designs")
    print(f"  Best MTF: {best_metrics['mtf_ridge']:.4f}")
    print(f"  Best design: d1={best_params[0]:.1f} d2={best_params[1]:.1f} "
          f"w1={best_params[2]:.1f} w2={best_params[3]:.1f}")

    return InverseDesignResult(
        best_params=best_params,
        best_metrics=best_metrics,
        pareto_params=pareto_x,
        pareto_objectives=pareto_obj,
        all_params=train_x.numpy(),
        all_objectives=train_obj.numpy(),
        n_iterations=n_iterations,
        elapsed_sec=elapsed,
    )
