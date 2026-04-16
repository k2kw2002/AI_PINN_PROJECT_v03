"""Resume Phase C training from checkpoint.

This is a convenience wrapper. Equivalent to:
    python scripts/train_phase_c.py --resume <checkpoint> --no-warmstart

Usage:
    python scripts/train_phase_c_resume.py checkpoints/phase_c_stage2.pt
    python scripts/train_phase_c_resume.py checkpoints/phase_c_epoch5000.pt --epochs 50000
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_phase_c_resume.py <checkpoint_path> [options]")
        print()
        print("Examples:")
        print("  python scripts/train_phase_c_resume.py checkpoints/phase_c_stage2.pt")
        print("  python scripts/train_phase_c_resume.py checkpoints/phase_c_epoch5000.pt --epochs 50000")
        sys.exit(1)

    # Forward to train_phase_c.py with --resume and --no-warmstart
    ckpt_path = sys.argv[1]
    extra_args = sys.argv[2:]

    import subprocess
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_phase_c.py"),
        "--resume", ckpt_path,
        "--no-warmstart",
    ] + extra_args

    print(f"Resuming from: {ckpt_path}")
    subprocess.run(cmd)
