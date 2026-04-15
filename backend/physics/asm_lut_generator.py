"""ASM LUT generator and LightTools source file exporter.

Generates:
1. incident_z40.npz - PINN L_phase target (complex field)
2. LightTools custom source files - for direct TMM+ASM injection into LT

The LT custom source ensures z=40 incident light is EXACTLY the same
as our TMM+ASM computation, guaranteeing L_phase and L_I consistency.

Dependencies:
    pip install numpy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def export_lt_source_files(
    lut_path: str = "data/asm_luts/incident_z40.npz",
    output_dir: str = "data/lt_source_files",
    theta_subset: list[float] | None = None,
):
    """Export TMM+ASM results as LightTools custom source files.

    For each angle theta, generates a text file containing the intensity
    profile I(x) = |U(x, z=40)|^2 that can be imported as a LightTools
    Grid Source or File-based Source at z=40.

    Args:
        lut_path: Path to incident_z40.npz (from 01_asm_lut_generation).
        output_dir: Directory to write LT source files.
        theta_subset: Specific angles to export. None = all angles in LUT.

    Returns:
        List of generated file paths.
    """
    data = np.load(lut_path)
    theta_values = data["theta_values"]
    x_values = data["x_values"]
    U_re = data["U_re"]
    U_im = data["U_im"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if theta_subset is not None:
        indices = []
        for t in theta_subset:
            idx = np.argmin(np.abs(theta_values - t))
            indices.append(idx)
    else:
        indices = range(len(theta_values))

    generated = []

    for idx in indices:
        theta = float(theta_values[idx])
        intensity = U_re[idx] ** 2 + U_im[idx] ** 2  # |U|^2

        # ── Format 1: Tab-separated text (LT Grid Source import) ──
        txt_path = out_dir / f"source_theta{theta:+06.1f}.txt"
        with open(txt_path, "w") as f:
            f.write(f"# TMM+ASM Intensity at z=40, theta={theta:.1f} deg\n")
            f.write(f"# Wavelength: 520 nm, CG: 550um n=1.52\n")
            f.write(f"# AR: SiO2(34.6)/TiO2(25.9)/SiO2(20.7)/TiO2(169.5) nm\n")
            f.write(f"# Columns: x_um\tintensity\n")
            f.write(f"# N_points: {len(x_values)}\n")
            for x, I in zip(x_values, intensity):
                f.write(f"{x:.4f}\t{I:.8f}\n")

        # ── Format 2: IES-like format (some LT versions) ──
        # (same data, different header)

        generated.append(str(txt_path))

    # ── Summary file ──
    summary_path = out_dir / "source_summary.txt"
    with open(summary_path, "w") as f:
        f.write("TMM+ASM Custom Source Files for LightTools\n")
        f.write("=" * 50 + "\n\n")
        f.write("Usage in LightTools:\n")
        f.write("  1. Insert > Source > Grid Source (or File Source)\n")
        f.write("  2. Position: z = 40 um\n")
        f.write("  3. Import file: source_theta+XX.X.txt\n")
        f.write("  4. Set angle: Tilt X = theta\n\n")
        f.write("Physical setup:\n")
        f.write("  - NO AR coating needed in LT model\n")
        f.write("  - NO Cover Glass needed in LT model\n")
        f.write("  - Only: Source(z=40) > BM2 > ILD > BM1 > Encap > OPD\n\n")
        f.write(f"Files generated: {len(generated)}\n")
        for theta_idx in indices:
            t = float(theta_values[theta_idx])
            I = U_re[theta_idx] ** 2 + U_im[theta_idx] ** 2
            f.write(f"  theta={t:+6.1f} deg  I_mean={I.mean():.6f}\n")

    generated.append(str(summary_path))
    return generated


def export_lt_source_npz(
    lut_path: str = "data/asm_luts/incident_z40.npz",
    output_path: str = "data/lt_source_files/lt_source_intensity.npz",
):
    """Export all angles as a single NPZ with intensity (no phase).

    This can be loaded and used to create LT sources programmatically.

    Saved arrays:
        theta_values: (N_theta,) degrees
        x_values: (N_x,) um
        intensity: (N_theta, N_x) |U|^2
    """
    data = np.load(lut_path)
    intensity = data["U_re"] ** 2 + data["U_im"] ** 2

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        theta_values=data["theta_values"],
        x_values=data["x_values"],
        intensity=intensity.astype(np.float32),
    )
    return str(out)
