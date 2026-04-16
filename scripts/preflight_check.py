"""GPU training pre-flight check.

Verifies everything is ready before starting long GPU training.
Run this on the GPU machine before `train_phase_c.py`.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --config configs/phase_c_full_gpu.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(name: str, condition: bool, detail: str = ""):
    icon = "PASS" if condition else "FAIL"
    color_code = "" if condition else "  ← FIX THIS"
    print(f"  [{icon}] {name:40s} {detail}{color_code}")
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE C - GPU Training Pre-flight Check")
    print("=" * 60)
    all_ok = True

    # 1. Python packages
    print("\n--- 1. Python Packages ---")
    for pkg, imp in [("torch", "torch"), ("numpy", "numpy"), ("yaml", "yaml"),
                     ("tmm", "tmm"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]:
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "OK")
            all_ok &= check(pkg, True, f"v{ver}")
        except ImportError:
            all_ok &= check(pkg, False, "NOT INSTALLED")

    # 2. CUDA
    print("\n--- 2. GPU / CUDA ---")
    import torch
    cuda_ok = torch.cuda.is_available()
    all_ok &= check("CUDA available", cuda_ok)
    if cuda_ok:
        check("GPU name", True, torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        check("GPU memory", mem_gb > 2, f"{mem_gb:.1f} GB")
    else:
        print("  → CPU 학습은 가능하지만 50K epochs에 매우 오래 걸립니다")
        print("  → GPU 권장 (CUDA 설치: https://pytorch.org/get-started)")

    # 3. ASM LUT
    print("\n--- 3. Data Files ---")
    lut_path = ROOT / "data" / "asm_luts" / "incident_z40.npz"
    lut_exists = lut_path.exists()
    all_ok &= check("ASM LUT (incident_z40.npz)", lut_exists,
                     f"{lut_path.stat().st_size/1024:.0f} KB" if lut_exists else "MISSING")
    if not lut_exists:
        print("  → Run: notebooks/02_phase_c_development/01_asm_lut_generation.ipynb")

    lt_files = list((ROOT / "data" / "lt_results").glob("sim_*.npz"))
    check("LightTools results", True,  # optional for Phase C-1
          f"{len(lt_files)} files" if lt_files else "None (optional, L_I disabled in Phase C-1)")

    source_files = list((ROOT / "data" / "lt_source_files").glob("source_*.txt"))
    check("LT source files", len(source_files) > 0,
          f"{len(source_files)} files" if source_files else "None")

    # 4. Config
    print("\n--- 4. Configuration ---")
    configs = list((ROOT / "configs").glob("*.yaml"))
    check("YAML configs", len(configs) > 0, f"{len(configs)} files")

    if args.config:
        cfg_path = Path(args.config)
        cfg_exists = cfg_path.exists()
        all_ok &= check(f"Selected config ({cfg_path.name})", cfg_exists)
        if cfg_exists:
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "training" in cfg:
                check("  epochs", True, str(cfg["training"].get("epochs", "?")))
                check("  device", True, cfg["training"].get("device", "?"))
            if "warm_start" in cfg:
                check("  warm_start", True,
                      f"{'enabled' if cfg['warm_start'].get('enabled') else 'disabled'}, "
                      f"{cfg['warm_start'].get('epochs', '?')} epochs")

    # 5. Model test
    print("\n--- 5. Model Quick Test ---")
    try:
        from backend.core.pinn_model import PurePINN
        model = PurePINN(hidden_dim=64, num_layers=3, num_freqs=24)
        x = torch.randn(10, 8)
        y = model(x)
        all_ok &= check("PINN forward pass", y.shape == (10, 2), f"shape={y.shape}")

        if cuda_ok:
            model_gpu = model.to("cuda")
            x_gpu = x.to("cuda")
            y_gpu = model_gpu(x_gpu)
            all_ok &= check("PINN on GPU", y_gpu.shape == (10, 2))
    except Exception as e:
        all_ok &= check("PINN forward pass", False, str(e))

    # 6. Loss test
    print("\n--- 6. Loss Functions ---")
    try:
        from backend.training.loss_functions import helmholtz_loss, ASMIncidentLUT
        from backend.training.collocation_sampler import hierarchical_collocation

        coords = hierarchical_collocation(50, torch.device("cpu"))
        L_H = helmholtz_loss(model, coords)
        all_ok &= check("L_Helmholtz", L_H.item() > 0, f"value={L_H.item():.4f}")

        if lut_exists:
            lut = ASMIncidentLUT(str(lut_path))
            from backend.training.loss_functions import phase_loss
            L_ph = phase_loss(model, lut, 30, torch.device("cpu"))
            all_ok &= check("L_phase", L_ph.item() >= 0, f"value={L_ph.item():.4f}")
    except Exception as e:
        all_ok &= check("Loss functions", False, str(e))

    # 7. Disk space
    print("\n--- 7. Disk Space ---")
    import shutil
    total, used, free = shutil.disk_usage(ROOT)
    free_gb = free / 1e9
    all_ok &= check("Free disk space", free_gb > 1, f"{free_gb:.1f} GB free")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED - Ready for GPU training!")
        print()
        print("Start training:")
        if args.config:
            print(f"  python scripts/train_phase_c.py --config {args.config}")
        else:
            print("  python scripts/train_phase_c.py --config configs/phase_c_warmstart.yaml")
    else:
        print("SOME CHECKS FAILED - Fix issues above before training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
