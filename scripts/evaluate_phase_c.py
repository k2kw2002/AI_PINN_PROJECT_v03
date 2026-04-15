"""Phase C final evaluation script.

Usage:
    python scripts/evaluate_phase_c.py
    python scripts/evaluate_phase_c.py --checkpoint checkpoints/phase_c_final.pt
    python scripts/evaluate_phase_c.py --checkpoint checkpoints/phase_c_epoch5000.pt --device cuda

Runs all validation checks from v6 Section 18:
1. BM boundary: |U| < 0.05 at BM regions
2. Interior fringe: std > 0.1 (not uniform)
3. Design variable sensitivity
4. PSF 7-pixel at multiple angles
5. MTF, skewness, throughput metrics
6. Red flag detection
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from backend.core.pinn_model import PurePINN
from backend.physics.psf_metrics import compute_psf_multi_angle, compute_all_metrics, compute_psf_7
from backend.training.red_flag_detector import detect_red_flags


def parse_args():
    parser = argparse.ArgumentParser(description="Phase C Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def find_checkpoint() -> Path:
    ckpt_dir = ROOT / "checkpoints"
    final = ckpt_dir / "phase_c_final.pt"
    if final.exists():
        return final
    ckpts = sorted(ckpt_dir.glob("phase_c_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError("No checkpoints found. Run train_phase_c.py first.")


def find_config() -> dict:
    exp_dirs = sorted((ROOT / "experiments").glob("*phase_c*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if exp_dirs and (exp_dirs[0] / "config.yaml").exists():
        with open(exp_dirs[0] / "config.yaml") as f:
            return yaml.safe_load(f)["model"]
    return {"hidden_dim": 128, "num_layers": 4, "num_freqs": 48, "omega_0": 30.0}


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    mcfg = find_config()
    model = PurePINN(**mcfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("PHASE C EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path.name} (epoch {ckpt['epoch']})")
    print(f"Model: {n_params:,} params")
    print(f"Device: {device}")
    print()

    # ── 1. Red Flag Detection ──
    print("--- 1. Red Flag Detection ---")
    report = detect_red_flags(model, device)
    print(f"  Interior CoV:       {report.interior_cov:.4f} (want > 0.05)")
    print(f"  BM1 mean |U|:       {report.bm1_mean_amp:.4f} (want < 0.05)")
    print(f"  BM2 mean |U|:       {report.bm2_mean_amp:.4f} (want < 0.05)")
    print(f"  Design sensitivity: {report.design_sensitivity:.4f} (want > 0.01)")
    print(f"  {report.summary()}")
    print()

    # ── 2. PSF Multi-Angle ──
    print("--- 2. PSF Multi-Angle ---")
    test_designs = [
        {"delta_bm1": 0, "delta_bm2": 0, "w1": 10, "w2": 10, "label": "default"},
        {"delta_bm1": 5, "delta_bm2": -5, "w1": 8, "w2": 15, "label": "asymmetric"},
    ]

    all_results = []
    for design in test_designs:
        label = design.pop("label")
        print(f"\n  Design: {label} ({design})")
        results = compute_psf_multi_angle(model, **design, device=device)
        for theta, metrics in results.items():
            print(f"    theta={theta:3.0f}: MTF={metrics['mtf_ridge']:.4f} "
                  f"skew={metrics['skewness']:+.4f} "
                  f"T={metrics['throughput']:.4f} "
                  f"XT={metrics['crosstalk']:.2f}")
            all_results.append({"design": label, "theta": theta, **metrics})

    # ── 3. Summary ──
    print()
    print("=" * 60)
    checks = [
        ("BM boundary", not report.bm_not_learned),
        ("Interior fringe", not report.uniform_field),
        ("Design sensitivity", not report.design_insensitive),
        ("No red flags", not report.has_red_flag),
    ]

    all_pass = True
    for name, passed in checks:
        icon = "+" if passed else "X"
        print(f"  [{icon}] {name:25s} {'PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed

    print("=" * 60)
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
        print("Next: Phase D (FNO distillation + BoTorch inverse design)")
    elif report.has_red_flag:
        print("RESULT: RED FLAG - Do NOT proceed. Diagnose and retrain.")
    else:
        print("RESULT: WARNINGS present. May need more training.")
    print("=" * 60)

    # Save results
    out_path = ROOT / "experiments" / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": str(ckpt_path.name),
            "epoch": ckpt["epoch"],
            "red_flag": {
                "interior_cov": report.interior_cov,
                "bm1_amp": report.bm1_mean_amp,
                "bm2_amp": report.bm2_mean_amp,
                "sensitivity": report.design_sensitivity,
                "has_red_flag": report.has_red_flag,
            },
            "psf_results": all_results,
            "all_pass": all_pass,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
