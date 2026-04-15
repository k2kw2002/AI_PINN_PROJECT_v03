"""Phase C PINN training script (GPU long-running).

v6 Section 7: Curriculum 3-Stage training
v6 Section 15.5: Layer 3 - CLI script for production training

Usage:
    python scripts/train_phase_c.py                              # default config
    python scripts/train_phase_c.py --config configs/phase_c_full_gpu.yaml
    python scripts/train_phase_c.py --epochs 1000 --device cpu   # override

Saves:
    - checkpoints/phase_c_stage{1,2,3}.pt
    - checkpoints/phase_c_final.pt
    - experiments/YYYY-MM-DD_phase_c/training.log
    - experiments/YYYY-MM-DD_phase_c/red_flag_history.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from backend.core.pinn_model import PurePINN
from backend.training.loss_functions import (
    ASMIncidentLUT,
    helmholtz_loss,
    phase_loss,
    bm_boundary_loss,
)
from backend.training.collocation_sampler import hierarchical_collocation
from backend.training.curriculum import CurriculumConfig, get_loss_weights, get_stage_name
from backend.training.red_flag_detector import detect_red_flags


def parse_args():
    parser = argparse.ArgumentParser(description="Phase C PINN Training")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_colloc", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag")
    return parser.parse_args()


def load_config(args) -> dict:
    """Load config from YAML file, then apply CLI overrides."""
    # Defaults (v6 Section 7, configs/phase_c_full_gpu.yaml)
    config = {
        "model": {
            "hidden_dim": 128,
            "num_layers": 4,
            "num_freqs": 48,
            "omega_0": 30.0,
        },
        "training": {
            "epochs": 50000,
            "lr": 1e-3,
            "lr_min": 1e-5,
            "n_colloc": 20000,
            "n_phase": 2000,
            "n_bc": 2000,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "curriculum": {
            "stage1_frac": 0.20,
            "stage2_frac": 0.40,
            "lambda_H": 1.0,
            "lambda_phase": 0.5,
            "lambda_BC": 0.5,
            "lambda_I": 0.0,
        },
        "checkpoint": {
            "save_every": 1000,
            "dir": "checkpoints",
        },
        "red_flag": {
            "check_every": 500,
        },
    }

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        for section in yaml_cfg:
            if section in config and isinstance(config[section], dict):
                config[section].update(yaml_cfg[section])
            else:
                config[section] = yaml_cfg[section]

    # CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.n_colloc is not None:
        config["training"]["n_colloc"] = args.n_colloc
    if args.device is not None:
        config["training"]["device"] = args.device

    return config


def setup_experiment(tag: str) -> Path:
    """Create experiment directory with timestamp."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    name = f"{date_str}_phase_c"
    if tag:
        name += f"_{tag}"
    exp_dir = ROOT / "experiments" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def setup_logging(exp_dir: Path):
    """Configure logging to file and stdout."""
    log_path = exp_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("train")


def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, path: Path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_history": loss_history,
    }, path)


def train(config: dict, args):
    """Main training loop."""
    exp_dir = setup_experiment(args.tag)
    logger = setup_logging(exp_dir)

    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    device = torch.device(config["training"]["device"])
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ──
    mcfg = config["model"]
    model = PurePINN(
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        num_freqs=mcfg["num_freqs"],
        omega_0=mcfg["omega_0"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # ── Resume ──
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── Optimizer (v6 Section 7.4) ──
    tcfg = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"], eta_min=tcfg["lr_min"]
    )

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── ASM LUT ──
    lut_path = ROOT / "data" / "asm_luts" / "incident_z40.npz"
    asm_lut = ASMIncidentLUT(str(lut_path))
    logger.info(f"ASM LUT loaded: {lut_path}")

    # ── Curriculum ──
    ccfg = config["curriculum"]
    curriculum = CurriculumConfig(
        total_epochs=tcfg["epochs"],
        stage1_frac=ccfg["stage1_frac"],
        stage2_frac=ccfg["stage2_frac"],
        lambda_H=ccfg["lambda_H"],
        lambda_phase=ccfg["lambda_phase"],
        lambda_BC=ccfg["lambda_BC"],
        lambda_I=ccfg["lambda_I"],
    )
    logger.info(f"Curriculum: S1=[0,{curriculum.stage1_end}], "
                f"S2=[{curriculum.stage1_end},{curriculum.stage2_end}], "
                f"S3=[{curriculum.stage2_end},{tcfg['epochs']}]")

    # ── Checkpoint dir ──
    ckpt_dir = ROOT / config["checkpoint"]["dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = config["checkpoint"]["save_every"]
    rf_every = config["red_flag"]["check_every"]

    # ── Training loop ──
    loss_history = {"total": [], "L_H": [], "L_phase": [], "L_BC": []}
    red_flag_history = []
    prev_stage = ""

    logger.info(f"Starting training: {tcfg['epochs']} epochs, "
                f"n_colloc={tcfg['n_colloc']}, n_phase={tcfg['n_phase']}, n_bc={tcfg['n_bc']}")
    t_start = time.time()

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        weights = get_loss_weights(epoch, curriculum)
        stage = get_stage_name(epoch, curriculum)

        # Stage transition logging
        if stage != prev_stage:
            logger.info(f"{'='*60}")
            logger.info(f">>> {stage} (epoch {epoch})")
            logger.info(f"    Weights: H={weights['lambda_H']:.2f} "
                        f"Ph={weights['lambda_phase']:.2f} BC={weights['lambda_BC']:.2f}")
            logger.info(f"{'='*60}")
            prev_stage = stage

        # Sample collocation points
        coords = hierarchical_collocation(tcfg["n_colloc"], device)

        # Compute losses
        L_H = helmholtz_loss(model, coords) if weights["lambda_H"] > 0 else torch.tensor(0.0, device=device)
        L_ph = phase_loss(model, asm_lut, tcfg["n_phase"], device)
        L_bc = bm_boundary_loss(model, tcfg["n_bc"], device)

        L_total = (
            weights["lambda_H"] * L_H
            + weights["lambda_phase"] * L_ph
            + weights["lambda_BC"] * L_bc
        )

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        scheduler.step()

        # Record
        loss_history["total"].append(L_total.item())
        loss_history["L_H"].append(L_H.item())
        loss_history["L_phase"].append(L_ph.item())
        loss_history["L_BC"].append(L_bc.item())

        # Log every 100 epochs
        if epoch % 100 == 0 or epoch == tcfg["epochs"] - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (epoch - start_epoch + 1) * (tcfg["epochs"] - epoch - 1) if epoch > start_epoch else 0
            logger.info(
                f"Epoch {epoch:6d}/{tcfg['epochs']} | "
                f"Total={L_total.item():.4e} H={L_H.item():.4e} "
                f"Ph={L_ph.item():.4e} BC={L_bc.item():.4e} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"ETA={eta/60:.0f}min"
            )

        # Checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"phase_c_epoch{epoch+1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path.name}")

        # Stage checkpoints
        if epoch + 1 == curriculum.stage1_end:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history,
                            ckpt_dir / "phase_c_stage1.pt")
            logger.info("Stage 1 checkpoint saved")
        elif epoch + 1 == curriculum.stage2_end:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history,
                            ckpt_dir / "phase_c_stage2.pt")
            logger.info("Stage 2 checkpoint saved")

        # Red flag detection
        if (epoch + 1) % rf_every == 0:
            report = detect_red_flags(model, device)
            rf_entry = {
                "epoch": epoch,
                "has_red_flag": report.has_red_flag,
                "has_warning": report.has_warning,
                "interior_cov": report.interior_cov,
                "bm1_mean_amp": report.bm1_mean_amp,
                "bm2_mean_amp": report.bm2_mean_amp,
                "design_sensitivity": report.design_sensitivity,
            }
            red_flag_history.append(rf_entry)
            logger.info(f"Red Flag Check: {report.summary()}")

            if report.has_red_flag:
                logger.warning("RED FLAG DETECTED - consider stopping and diagnosing")

    # ── Final save ──
    total_time = time.time() - t_start
    save_checkpoint(model, optimizer, scheduler, tcfg["epochs"] - 1, loss_history,
                    ckpt_dir / "phase_c_final.pt")
    logger.info(f"Final checkpoint saved: phase_c_final.pt")

    # Save loss history
    with open(exp_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    # Save red flag history
    with open(exp_dir / "red_flag_history.json", "w") as f:
        json.dump(red_flag_history, f, indent=2)

    logger.info(f"Training complete in {total_time/3600:.1f} hours")
    logger.info(f"Experiment dir: {exp_dir}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    train(config, args)
