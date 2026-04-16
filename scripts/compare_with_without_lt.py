"""Compare PINN training with vs without LightTools data (L_I).

Runs two training sessions:
  A: L_H + L_phase + L_BC only (no simulation data)
  B: L_H + L_phase + L_BC + L_I (with simulation data)

Then compares PSF quality, convergence, and metrics.

Usage:
    python scripts/compare_with_without_lt.py
    python scripts/compare_with_without_lt.py --epochs 500 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from backend.core.pinn_model import PurePINN
from backend.training.loss_functions import (
    ASMIncidentLUT, helmholtz_loss, phase_loss, bm_boundary_loss,
)
from backend.training.collocation_sampler import hierarchical_collocation
from backend.training.curriculum import CurriculumConfig, get_loss_weights
from backend.training.warm_start import warm_start
from backend.training.red_flag_detector import detect_red_flags
from backend.physics.psf_metrics import compute_psf_7, compute_all_metrics
from backend.data.lighttools_runner import LTResultDataset


def parse_args():
    p = argparse.ArgumentParser(description="Compare with/without LT data")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmstart-epochs", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def train_one(
    name: str, use_lt: bool, args, asm_lut, lt_dataset,
) -> dict:
    """Train a single model and return results."""
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"Training: {name} (L_I={'ON' if use_lt else 'OFF'})")
    print(f"{'='*60}")

    model = PurePINN(hidden_dim=args.hidden_dim, num_layers=3, num_freqs=24).to(device)

    # Warm start
    warm_start(model, asm_lut, epochs=args.warmstart_epochs, device=device, log_every=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    curriculum = CurriculumConfig(
        total_epochs=args.epochs,
        lambda_I=0.3 if use_lt else 0.0,
    )

    history = {"total": [], "L_H": [], "L_phase": [], "L_BC": [], "L_I": []}
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        w = get_loss_weights(epoch, curriculum)

        coords = hierarchical_collocation(1000, device)
        L_H = helmholtz_loss(model, coords) if w["lambda_H"] > 0 else torch.tensor(0.0, device=device)
        L_ph = phase_loss(model, asm_lut, 100, device)
        L_bc = bm_boundary_loss(model, 100, device)

        L_I = torch.tensor(0.0, device=device)
        if use_lt and w["lambda_I"] > 0 and lt_dataset is not None:
            import math
            idx = np.random.randint(0, lt_dataset.n_samples)
            cfg_lt, x_lt, I_lt = lt_dataset.get_target(idx)
            n_lt = len(x_lt)
            sin_t = math.sin(math.radians(cfg_lt["theta_deg"]))
            cos_t = math.cos(math.radians(cfg_lt["theta_deg"]))
            coords_lt = torch.stack([
                torch.from_numpy(x_lt).to(device),
                torch.zeros(n_lt, device=device),
                torch.full((n_lt,), cfg_lt["delta_bm1"], device=device),
                torch.full((n_lt,), cfg_lt["delta_bm2"], device=device),
                torch.full((n_lt,), cfg_lt["w1"], device=device),
                torch.full((n_lt,), cfg_lt["w2"], device=device),
                torch.full((n_lt,), sin_t, device=device),
                torch.full((n_lt,), cos_t, device=device),
            ], dim=1)
            U_lt = model(coords_lt)
            I_pinn = U_lt[:, 0] ** 2 + U_lt[:, 1] ** 2
            I_target = torch.from_numpy(I_lt).to(device)
            L_I = torch.mean((I_pinn - I_target) ** 2)

        L_total = w["lambda_H"] * L_H + w["lambda_phase"] * L_ph + w["lambda_BC"] * L_bc + w["lambda_I"] * L_I

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(L_total.item())
        history["L_H"].append(L_H.item())
        history["L_phase"].append(L_ph.item())
        history["L_BC"].append(L_bc.item())
        history["L_I"].append(L_I.item())

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:4d} | Total={L_total.item():.4e} H={L_H.item():.4e} "
                  f"Ph={L_ph.item():.4e} BC={L_bc.item():.4e} I={L_I.item():.4e}")

    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    report = detect_red_flags(model, device)
    psf_0 = compute_psf_7(model, 0, 0, 10, 10, 0, device)
    psf_30 = compute_psf_7(model, 0, 0, 10, 10, 30, device)
    metrics_0 = compute_all_metrics(psf_0)
    metrics_30 = compute_all_metrics(psf_30)

    return {
        "name": name,
        "use_lt": use_lt,
        "history": history,
        "elapsed": elapsed,
        "red_flag": {
            "interior_cov": report.interior_cov,
            "bm1_amp": report.bm1_mean_amp,
            "bm2_amp": report.bm2_mean_amp,
            "sensitivity": report.design_sensitivity,
        },
        "psf_0": metrics_0,
        "psf_30": metrics_30,
        "model": model,
    }


def main():
    args = parse_args()

    # ASM LUT
    asm_lut = ASMIncidentLUT(str(ROOT / "data" / "asm_luts" / "incident_z40.npz"))

    # Generate mock LT data
    print("Generating mock LightTools data...")
    from backend.data.lhs_sampler import generate_lhs_samples
    from backend.data.mock_lt_generator import generate_mock_lt_dataset

    plan = generate_lhs_samples(20, n_angles=3, seed=42)
    n_generated = generate_mock_lt_dataset(plan["all_configs"], str(ROOT / "data" / "lt_results"))
    print(f"Generated {n_generated} mock LT results")

    lt_dataset = LTResultDataset(str(ROOT / "data" / "lt_results"))

    # Train A: without L_I
    result_a = train_one("Without L_I", False, args, asm_lut, None)

    # Train B: with L_I
    result_b = train_one("With L_I", True, args, asm_lut, lt_dataset)

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("COMPARISON: Without L_I vs With L_I")
    print("=" * 60)

    print(f"\n{'Metric':<25s} {'Without L_I':>15s} {'With L_I':>15s}")
    print("-" * 55)

    comparisons = [
        ("Final L_total", result_a["history"]["total"][-1], result_b["history"]["total"][-1]),
        ("Final L_H", result_a["history"]["L_H"][-1], result_b["history"]["L_H"][-1]),
        ("Final L_phase", result_a["history"]["L_phase"][-1], result_b["history"]["L_phase"][-1]),
        ("Final L_BC", result_a["history"]["L_BC"][-1], result_b["history"]["L_BC"][-1]),
        ("Interior CoV", result_a["red_flag"]["interior_cov"], result_b["red_flag"]["interior_cov"]),
        ("BM1 |U|", result_a["red_flag"]["bm1_amp"], result_b["red_flag"]["bm1_amp"]),
        ("Design sensitivity", result_a["red_flag"]["sensitivity"], result_b["red_flag"]["sensitivity"]),
        ("MTF@ridge (0 deg)", result_a["psf_0"]["mtf_ridge"], result_b["psf_0"]["mtf_ridge"]),
        ("MTF@ridge (30 deg)", result_a["psf_30"]["mtf_ridge"], result_b["psf_30"]["mtf_ridge"]),
        ("Skewness (0 deg)", result_a["psf_0"]["skewness"], result_b["psf_0"]["skewness"]),
        ("Crosstalk (0 deg)", result_a["psf_0"]["crosstalk"], result_b["psf_0"]["crosstalk"]),
        ("Training time (s)", result_a["elapsed"], result_b["elapsed"]),
    ]

    for name, va, vb in comparisons:
        print(f"  {name:<25s} {va:>15.6f} {vb:>15.6f}")

    # Save results
    out_path = ROOT / "experiments" / "lt_comparison.json"
    save_data = {
        "without_lt": {
            k: v for k, v in result_a.items() if k != "model"
        },
        "with_lt": {
            k: v for k, v in result_b.items() if k != "model"
        },
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\nVisualize: open notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb")


if __name__ == "__main__":
    main()
