"""FNO distillation from trained PINN (Phase D).

Trains an FNO surrogate to predict PSF from design parameters,
using data generated from the PINN model.

Usage:
    python scripts/distill_fno.py
    python scripts/distill_fno.py --data data/fno_training/pinn_distill_data.npz
    python scripts/distill_fno.py --epochs 2000 --device cuda
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
import torch.nn as nn

from backend.core.fno_model import FNOSurrogate


def parse_args():
    p = argparse.ArgumentParser(description="FNO Distillation")
    p.add_argument("--data", type=str, default="data/fno_training/pinn_distill_data.npz")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load distillation data
    data = np.load(ROOT / args.data)
    params_np = data["params"]  # (N, 5)
    psfs_np = data["psfs"]      # (N, 7)
    N = len(params_np)
    print(f"Distillation data: {N} samples, params={params_np.shape}, psfs={psfs_np.shape}")

    # Train/val split
    split = int(N * 0.9)
    params_train = torch.from_numpy(params_np[:split]).to(device)
    psfs_train = torch.from_numpy(psfs_np[:split]).to(device)
    params_val = torch.from_numpy(params_np[split:]).to(device)
    psfs_val = torch.from_numpy(psfs_np[split:]).to(device)
    print(f"Train: {split}, Val: {N - split}")

    # Normalize params
    p_mean = params_train.mean(dim=0)
    p_std = params_train.std(dim=0) + 1e-8
    params_train_n = (params_train - p_mean) / p_std
    params_val_n = (params_val - p_mean) / p_std

    # Model
    model = FNOSurrogate(hidden_channels=32, modes=16, n_fourier_layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    # Training
    best_val_loss = float("inf")
    history = {"train": [], "val": []}
    t0 = time.time()

    print(f"\nTraining FNO: {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()

        # Mini-batch
        perm = torch.randperm(split, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, split, args.batch_size):
            idx = perm[i:i + args.batch_size]
            p_batch = params_train_n[idx]
            psf_batch = psfs_train[idx]

            pred = model(p_batch)
            loss = criterion(pred, psf_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(params_val_n)
            val_loss = criterion(val_pred, psfs_val).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:5d} | Train={train_loss:.6f} Val={val_loss:.6f} "
                  f"Best={best_val_loss:.6f} LR={scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    print(f"\nTraining complete: {elapsed:.1f}s")

    # Save best model
    model.load_state_dict(best_state)
    save_path = ROOT / "checkpoints" / "fno_surrogate.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "p_mean": p_mean.cpu(),
        "p_std": p_std.cpu(),
        "best_val_loss": best_val_loss,
        "history": history,
    }, save_path)
    print(f"Saved: {save_path}")

    # Inference speed test
    model.eval()
    with torch.no_grad():
        t_start = time.time()
        for _ in range(1000):
            _ = model(params_val_n[:1])
        t_per = (time.time() - t_start) / 1000 * 1000  # ms
    print(f"Inference speed: {t_per:.2f} ms/sample")

    # Accuracy test
    with torch.no_grad():
        pred_all = model(params_val_n)
        rel_err = torch.abs(pred_all - psfs_val) / (psfs_val.abs() + 1e-8)
        print(f"Validation relative error: {rel_err.mean()*100:.2f}%")

    # Save results
    results_path = ROOT / "experiments" / "fno_distillation.json"
    with open(results_path, "w") as f:
        json.dump({
            "n_train": split,
            "n_val": N - split,
            "best_val_loss": best_val_loss,
            "inference_ms": t_per,
            "val_rel_error_pct": float(rel_err.mean() * 100),
            "training_time_s": elapsed,
        }, f, indent=2)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
