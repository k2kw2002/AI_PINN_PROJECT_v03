"""Common visualization utilities for notebooks.

Shared plotting functions for:
- UDFPS stack structure (full + PINN domain zoom)
- BM slit geometry with design variables
- Pipeline stage mapping
- PSF, field distributions, loss curves
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PatchCollection


# ═══════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "finger": "#FFD9B3",
    "ar": "#A8D8EA",
    "cg": "#C5E1A5",
    "bm": "#424242",
    "bm_edge": "#757575",
    "ild": "#FFF9C4",
    "encap": "#F8BBD0",
    "opd": "#CE93D8",
    "slit": "#FFFFFF",
    "pinn_domain": "#E3F2FD",
    "tmm": "#42A5F5",
    "asm": "#66BB6A",
    "pinn": "#EF5350",
    "psf": "#AB47BC",
    "light": "#FFC107",
}


# ═══════════════════════════════════════════════════════════════════
# 1. Full UDFPS Stack (v6 Section 2.1)
# ═══════════════════════════════════════════════════════════════════

def plot_full_stack(figsize=(10, 12)):
    """Plot full UDFPS COE stack from Finger to OPD sensor.

    v6 Section 2.1, 2.2: Physical stack structure with z-coordinates.
    Shows layer thicknesses (not to scale) and pipeline stage assignments.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 16)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("UDFPS COE Stack Structure (v6 Section 2.1)", fontsize=14, fontweight="bold", pad=15)

    W = 6.0  # layer width
    x0 = 2.0  # left edge

    # Layer definitions: (label, height, color, z_label, detail)
    layers = [
        ("OPD Sensor", 0.8, COLORS["opd"], "z = 0 μm", "Photodiode array"),
        ("Encapsulation", 2.5, COLORS["encap"], "z = 0~20 μm", "20 μm, PINN L_H"),
        ("BM1", 0.3, COLORS["bm"], "z = 20 μm", "Aperture w₁, offset δ₁"),
        ("ILD", 2.5, COLORS["ild"], "z = 20~40 μm", "20 μm, PINN L_H"),
        ("BM2", 0.3, COLORS["bm"], "z = 40 μm", "Aperture w₂, offset δ₂"),
        ("Cover Glass", 5.0, COLORS["cg"], "z = 40~590 μm", "550 μm, n=1.52, ASM"),
        ("AR Coating", 0.5, COLORS["ar"], "z ≈ 590 μm", "~300 nm, Gorilla DX 4-layer, TMM"),
        ("Finger", 1.2, COLORS["finger"], "", "Contact surface"),
    ]

    y = 0
    layer_positions = {}
    for label, h, color, z_label, detail in layers:
        rect = FancyBboxPatch(
            (x0, y), W, h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333333", linewidth=1.2,
        )
        ax.add_patch(rect)

        # Label inside
        text_color = "white" if label.startswith("BM") else "#333333"
        fontw = "bold" if label.startswith("BM") else "normal"
        ax.text(x0 + W / 2, y + h / 2, label,
                ha="center", va="center", fontsize=10, fontweight=fontw, color=text_color)

        # z-label on the left
        if z_label:
            ax.text(x0 - 0.15, y + h / 2, z_label,
                    ha="right", va="center", fontsize=8, color="#555555")

        # detail on the right
        ax.text(x0 + W + 0.15, y + h / 2, detail,
                ha="left", va="center", fontsize=7.5, color="#777777", style="italic")

        layer_positions[label] = (y, y + h)
        y += h

    # ── PINN domain bracket ──
    pinn_bottom = layer_positions["OPD Sensor"][0]
    pinn_top = layer_positions["BM2"][1]
    bx = x0 - 1.8
    ax.annotate("", xy=(bx + 0.1, pinn_bottom), xytext=(bx + 0.1, pinn_top),
                arrowprops=dict(arrowstyle="<->", color=COLORS["pinn"], lw=2.5))
    ax.text(bx - 0.1, (pinn_bottom + pinn_top) / 2, "PINN\nDomain\nz=[0,40]μm",
            ha="center", va="center", fontsize=9, fontweight="bold", color=COLORS["pinn"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLORS["pinn"], alpha=0.9))

    # ── Pipeline stage labels on far right ──
    stage_info = [
        ("TMM", layer_positions["AR Coating"], COLORS["tmm"]),
        ("ASM", layer_positions["Cover Glass"], COLORS["asm"]),
        ("PINN", (pinn_bottom, pinn_top), COLORS["pinn"]),
    ]
    sx = x0 + W + 3.0
    for stage_label, (yb, yt), color in stage_info:
        mid = (yb + yt) / 2
        ax.text(sx, mid, stage_label, ha="center", va="center", fontsize=10, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor=color))

    # ── Light arrow ──
    ax.annotate("", xy=(x0 + W / 2, 0.4), xytext=(x0 + W / 2, y - 0.4),
                arrowprops=dict(arrowstyle="->", color=COLORS["light"], lw=2.5,
                                connectionstyle="arc3,rad=0"))
    ax.text(x0 + W / 2 + 0.4, y - 0.8, "Light ↓",
            fontsize=9, color=COLORS["light"], fontweight="bold")

    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# 2. PINN Domain Zoom (v6 Section 2.3, 4.2)
# ═══════════════════════════════════════════════════════════════════

def plot_pinn_domain(
    delta_bm1: float = 0.0,
    delta_bm2: float = 0.0,
    w1: float = 10.0,
    w2: float = 10.0,
    n_pitches: int = 7,
    pitch: float = 72.0,
    figsize=(14, 7),
):
    """Plot PINN domain (z=0~40) with BM slit geometry.

    v6 Section 2.3, 4.2, 4.3: Shows BM1, BM2 slit positions,
    design variables (delta, w), and the 7 OPD pixels.

    Args:
        delta_bm1, delta_bm2: BM offsets (μm).
        w1, w2: Slit aperture widths (μm).
        n_pitches: Number of OPD pitches (default 7).
        pitch: OPD pitch in μm (default 72).
    """
    x_max = n_pitches * pitch  # 504
    fig, ax = plt.subplots(figsize=figsize)

    # ── Background layers ──
    # Encap (z=0~20)
    ax.axhspan(0, 20, color=COLORS["encap"], alpha=0.2, label="Encap (z=0~20)")
    # ILD (z=20~40)
    ax.axhspan(20, 40, color=COLORS["ild"], alpha=0.2, label="ILD (z=20~40)")

    # ── BM layers ──
    for i in range(n_pitches):
        # BM1 at z=20
        center1 = i * pitch + pitch / 2 + delta_bm1
        bm1_left = (i * pitch, 19.5)
        bm1_right_start = center1 + w1 / 2
        bm1_left_end = center1 - w1 / 2

        # Left BM block
        if bm1_left_end > i * pitch:
            ax.add_patch(plt.Rectangle(
                (i * pitch, 19.5), bm1_left_end - i * pitch, 1.0,
                facecolor=COLORS["bm"], edgecolor=COLORS["bm_edge"], linewidth=0.5))
        # Right BM block
        if (i + 1) * pitch > bm1_right_start:
            ax.add_patch(plt.Rectangle(
                (bm1_right_start, 19.5), (i + 1) * pitch - bm1_right_start, 1.0,
                facecolor=COLORS["bm"], edgecolor=COLORS["bm_edge"], linewidth=0.5))

        # BM2 at z=40
        center2 = i * pitch + pitch / 2 + delta_bm2
        bm2_left_end = center2 - w2 / 2
        bm2_right_start = center2 + w2 / 2

        if bm2_left_end > i * pitch:
            ax.add_patch(plt.Rectangle(
                (i * pitch, 39.5), bm2_left_end - i * pitch, 0.5,
                facecolor=COLORS["bm"], edgecolor=COLORS["bm_edge"], linewidth=0.5))
        if (i + 1) * pitch > bm2_right_start:
            ax.add_patch(plt.Rectangle(
                (bm2_right_start, 39.5), (i + 1) * pitch - bm2_right_start, 0.5,
                facecolor=COLORS["bm"], edgecolor=COLORS["bm_edge"], linewidth=0.5))

    # ── OPD pixels at z=0 ──
    opd_width = 10.0
    for i in range(n_pitches):
        cx = i * pitch + pitch / 2
        ax.add_patch(plt.Rectangle(
            (cx - opd_width / 2, -1.5), opd_width, 1.5,
            facecolor=COLORS["opd"], edgecolor="#9C27B0", linewidth=0.8, alpha=0.7))
        ax.text(cx, -0.75, f"{i}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # ── Annotations ──
    # BM1 detail (zoom to one slit)
    i_anno = 3  # center pitch
    c1 = i_anno * pitch + pitch / 2 + delta_bm1
    c2 = i_anno * pitch + pitch / 2 + delta_bm2

    # w1 annotation
    ax.annotate("", xy=(c1 - w1 / 2, 18.5), xytext=(c1 + w1 / 2, 18.5),
                arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5))
    ax.text(c1, 17.8, f"w₁={w1:.0f}μm", ha="center", va="center", fontsize=8, color="blue", fontweight="bold")

    # w2 annotation
    ax.annotate("", xy=(c2 - w2 / 2, 41.5), xytext=(c2 + w2 / 2, 41.5),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(c2, 42.2, f"w₂={w2:.0f}μm", ha="center", va="center", fontsize=8, color="red", fontweight="bold")

    # delta annotations (if non-zero)
    if abs(delta_bm1) > 0.1:
        base = i_anno * pitch + pitch / 2
        ax.annotate("", xy=(base, 21.5), xytext=(c1, 21.5),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1, ls="--"))
        ax.text((base + c1) / 2, 22.2, f"δ₁={delta_bm1:.0f}", ha="center", fontsize=7, color="blue")

    if abs(delta_bm2) > 0.1:
        base = i_anno * pitch + pitch / 2
        ax.annotate("", xy=(base, 38.5), xytext=(c2, 38.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1, ls="--"))
        ax.text((base + c2) / 2, 37.8, f"δ₂={delta_bm2:.0f}", ha="center", fontsize=7, color="red")

    # ── z-axis labels ──
    ax.axhline(y=0, color="#9C27B0", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.axhline(y=20, color=COLORS["bm"], linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=40, color=COLORS["bm"], linestyle="--", linewidth=1, alpha=0.5)

    ax.text(-8, 0, "z=0\nOPD", ha="right", va="center", fontsize=9, fontweight="bold", color="#9C27B0")
    ax.text(-8, 20, "z=20\nBM1", ha="right", va="center", fontsize=9, fontweight="bold", color=COLORS["bm"])
    ax.text(-8, 40, "z=40\nBM2", ha="right", va="center", fontsize=9, fontweight="bold", color=COLORS["bm"])

    # ── Loss labels ──
    ax.text(x_max + 5, 10, "L_H\n(Helmholtz)", ha="left", va="center", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["encap"], alpha=0.5))
    ax.text(x_max + 5, 30, "L_H\n(Helmholtz)", ha="left", va="center", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["ild"], alpha=0.5))
    ax.text(x_max + 5, 20, "L_BC\n(U=0)", ha="left", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bm"]))
    ax.text(x_max + 5, 40, "L_phase + L_BC", ha="left", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["pinn"]))

    # ── Title and formatting ──
    ax.set_xlim(-15, x_max + 40)
    ax.set_ylim(-3, 44)
    ax.set_xlabel("x (μm)", fontsize=11)
    ax.set_ylabel("z (μm)", fontsize=11)
    ax.set_title(
        f"PINN Domain z=[0, 40]μm  |  δ₁={delta_bm1:.1f}, δ₂={delta_bm2:.1f}, "
        f"w₁={w1:.0f}, w₂={w2:.0f}μm  |  7 OPD pixels (pitch={pitch:.0f}μm)",
        fontsize=11, fontweight="bold",
    )
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.15)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["encap"], alpha=0.4, label="Encap"),
        mpatches.Patch(facecolor=COLORS["ild"], alpha=0.4, label="ILD"),
        mpatches.Patch(facecolor=COLORS["bm"], label="BM (opaque)"),
        mpatches.Patch(facecolor=COLORS["opd"], alpha=0.7, label="OPD pixel"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Slit (aperture)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# 3. Pipeline Diagram (v6 Section 3.1)
# ═══════════════════════════════════════════════════════════════════

def plot_pipeline(figsize=(14, 3.5)):
    """Plot the TMM → ASM → PINN → PSF pipeline diagram.

    v6 Section 3.1: Hybrid pipeline overview.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-0.5, 3)
    ax.axis("off")
    ax.set_title("Pipeline: TMM → ASM → PINN → PSF (v6 Section 3.1)", fontsize=13, fontweight="bold")

    stages = [
        ("TMM\n(AR Coating)", "~300nm\nd₁~d₄ → t(θ), Δφ(θ)", COLORS["tmm"], 0),
        ("ASM\n(Cover Glass)", "550μm\nFFT propagation", COLORS["asm"], 4),
        ("PINN\n(BM~OPD)", "40μm\n8D → U(x,z)", COLORS["pinn"], 8),
        ("PSF\n(Metrics)", "7 OPD pixels\nMTF, skew, T", COLORS["psf"], 12),
    ]

    for label, detail, color, x in stages:
        box = FancyBboxPatch(
            (x, 0.5), 2.8, 1.8,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(x + 1.4, 1.7, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x + 1.4, 0.85, detail, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.9)

    # Arrows
    for x_start in [2.8, 6.8, 10.8]:
        ax.annotate("", xy=(x_start + 1.2, 1.4), xytext=(x_start + 0.1, 1.4),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    # Interface labels
    interfaces = [
        (3.45, "t(θ)·e^{iΔφ}"),
        (7.45, "U(x, z=40, θ)"),
        (11.45, "U(x, z=0)"),
    ]
    for x, label in interfaces:
        ax.text(x, 0.2, label, ha="center", va="center", fontsize=7, color="#555", style="italic")

    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# 4. PINN Network Architecture (v6 Section 5)
# ═══════════════════════════════════════════════════════════════════

def plot_pinn_architecture(figsize=(14, 6)):
    """Plot PINN network architecture diagram.

    v6 Section 5.2: 8D Input -> Fourier -> SIREN -> 2D Output
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, 18)
    ax.set_ylim(-1.5, 8)
    ax.axis("off")
    ax.set_title("Pure PINN Architecture (v6 Section 5)", fontsize=14, fontweight="bold")

    def _box(x, y, w, h, label, sublabel, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white", lw=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w/2, y + h*0.25, sublabel, ha="center", va="center",
                    fontsize=7, color="white", alpha=0.85)

    def _arrow(x1, y, x2):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    # Input (8D)
    input_labels = ["x", "z", "δ₁", "δ₂", "w₁", "w₂", "sinθ", "cosθ"]
    for i, lbl in enumerate(input_labels):
        y = 7 - i * 0.85
        ax.add_patch(FancyBboxPatch((0, y), 1.2, 0.65, boxstyle="round,pad=0.05",
                                    facecolor="#78909C", edgecolor="white", lw=1))
        ax.text(0.6, y + 0.32, lbl, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")
    ax.text(0.6, -0.5, "Input 8D", ha="center", fontsize=9, fontweight="bold", color="#555")

    # Normalizer
    _box(2, 1.5, 1.8, 4, "Input\nNorm", "x/504, z/40\nδ/10, ...", "#607D8B")
    _arrow(1.3, 3.5, 2)

    # Fourier Embedding
    _box(4.5, 1.5, 2, 4, "Fourier\nEmbedding", "48 freq → 96D\nsin/cos(Bx)", "#FF7043")
    _arrow(3.8, 3.5, 4.5)

    # SIREN layers
    siren_x = [7.2, 9.2, 11.2, 13.2]
    for i, sx in enumerate(siren_x):
        _box(sx, 1.5, 1.5, 4, f"SIREN\n#{i+1}", f"128→128\nsin(ω₀·Wx)", "#5C6BC0")
        if i == 0:
            _arrow(6.5, 3.5, sx)
        else:
            _arrow(siren_x[i-1] + 1.5, 3.5, sx)

    # Output
    _box(15.5, 2.3, 1.5, 2.5, "Linear\nOutput", "128→2", "#AB47BC")
    _arrow(14.7, 3.5, 15.5)

    # Output labels
    out_labels = [("Re(U)", 4.2), ("Im(U)", 3.0)]
    for lbl, y in out_labels:
        ax.add_patch(FancyBboxPatch((17.3, y), 0.7, 0.7, boxstyle="round,pad=0.05",
                                    facecolor="#CE93D8", edgecolor="white", lw=1))
        ax.text(17.65, y + 0.35, lbl, ha="center", va="center", fontsize=7,
                fontweight="bold", color="white")
    ax.text(17.65, -0.5, "Output 2D", ha="center", fontsize=9, fontweight="bold", color="#555")
    _arrow(17, 3.5, 17.3)

    # Annotations
    ax.text(0.6, 7.8, "v6 §5.1\nNO slit_dist!", ha="center", fontsize=7,
            color=COLORS["pinn"], fontweight="bold")
    ax.text(5.5, 0.7, "ω₀ = 30", ha="center", fontsize=8, color="#FF7043", style="italic")
    ax.text(10.2, 0.7, "~100K params total", ha="center", fontsize=8,
            color="#5C6BC0", style="italic")

    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# 5. Loss Composition & Curriculum (v6 Section 6, 7)
# ═══════════════════════════════════════════════════════════════════

def plot_loss_and_curriculum(total_epochs=50000, figsize=(14, 8)):
    """Plot loss composition and curriculum schedule."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1.3])

    # ── Top: Loss composition ──
    ax = axes[0]
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")
    ax.set_title("Loss Composition (v6 Section 6)", fontsize=13, fontweight="bold")

    losses = [
        ("L_Helmholtz", "∇²U + k²U = 0\nPDE residual", "#1565C0", "λ_H = 1.0\n(44%)", 0),
        ("L_phase", "U(z=40) = U_ASM\nBM2 slit interior", "#C62828", "λ_ph = 0.5\n(22%)", 4),
        ("L_BC", "U = 0 at BM\nz=20, z=40", "#2E7D32", "λ_BC = 0.5\n(22%)", 8),
        ("L_I", "|U(z=0)|² = I_target\n(optional)", "#6A1B9A", "λ_I = 0.3\n(13%)", 12),
    ]
    for label, detail, color, weight, x in losses:
        box = FancyBboxPatch((x, 0.3), 3.2, 2.5, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white", lw=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 1.6, 2.2, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x + 1.6, 1.3, detail, ha="center", va="center",
                fontsize=7, color="white", alpha=0.9)
        ax.text(x + 1.6, 0.0, weight, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)

    ax.text(8, -0.5, "Rule: λ_H ≥ max(λ_phase, λ_BC, λ_I)",
            ha="center", fontsize=9, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825"))

    # ── Bottom: Curriculum timeline ──
    ax = axes[1]
    s1 = int(total_epochs * 0.2)
    s2 = int(total_epochs * 0.6)
    epochs = np.arange(total_epochs)

    lH = np.zeros(total_epochs)
    lH[s1:s2] = np.linspace(0.1, 1.0, s2 - s1)
    lH[s2:] = 1.0

    ax.fill_between(epochs, 0, lH, alpha=0.3, color="#1565C0", label="λ_H (Helmholtz)")
    ax.plot(epochs, lH, color="#1565C0", lw=2)
    ax.plot(epochs, np.full(total_epochs, 0.5), color="#C62828", lw=1.5, ls="--", label="λ_phase")
    ax.plot(epochs, np.full(total_epochs, 0.5), color="#2E7D32", lw=1.5, ls=":", label="λ_BC")

    ax.axvline(x=s1, color="#FF6F00", lw=2)
    ax.axvline(x=s2, color="#FF6F00", lw=2)

    for cx, lbl in [(s1/2, "Stage 1\nBoundary"), ((s1+s2)/2, "Stage 2\nPDE Ramp"), ((s2+total_epochs)/2, "Stage 3\nFull")]:
        ax.text(cx, 1.15, lbl, ha="center", fontsize=10, fontweight="bold", color="#FF6F00",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss Weight (λ)", fontsize=11)
    ax.set_title("Curriculum 3-Stage Schedule (v6 Section 7)", fontsize=13, fontweight="bold")
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(-0.05, 1.4)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
# 6. End-to-End Data Flow (v6 Section 3, 10)
# ═══════════════════════════════════════════════════════════════════

def plot_data_flow(figsize=(16, 10)):
    """Plot end-to-end data flow and system architecture."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, 18)
    ax.set_ylim(-0.5, 12)
    ax.axis("off")
    ax.set_title("End-to-End System Architecture & Data Flow",
                 fontsize=14, fontweight="bold", pad=15)

    def _box(x, y, w, h, label, sub, color, text_color="white"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor="white", lw=1.5, alpha=0.9)
        ax.add_patch(b)
        ax.text(x+w/2, y+h*0.65, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)
        if sub:
            ax.text(x+w/2, y+h*0.25, sub, ha="center", va="center",
                    fontsize=6.5, color=text_color, alpha=0.85)

    def _arr(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    # ── Row 1: Forward Pipeline ──
    y1 = 9.5
    _box(0, y1, 2.2, 1.6, "Design\nVariables", "δ₁,δ₂,w₁,w₂ + θ", "#78909C")
    _arr(2.2, y1+0.8, 2.8, y1+0.8)
    _box(2.8, y1, 2, 1.6, "TMM", "AR t(θ),Δφ(θ)", COLORS["tmm"])
    _arr(4.8, y1+0.8, 5.4, y1+0.8)
    _box(5.4, y1, 2.2, 1.6, "ASM", "CG 550μm FFT", COLORS["asm"])
    _arr(7.6, y1+0.8, 8.2, y1+0.8)
    _box(8.2, y1, 2.2, 1.6, "PINN", "8D→U(x,z)", COLORS["pinn"])
    _arr(10.4, y1+0.8, 11, y1+0.8)
    _box(11, y1, 2, 1.6, "PSF", "7 OPD pixels", COLORS["psf"])
    _arr(13, y1+0.8, 13.6, y1+0.8)
    _box(13.6, y1, 2.5, 1.6, "Metrics", "MTF, skew, T", "#795548")

    ax.text(9, y1+2, "Forward Pipeline (Phase C)", ha="center", fontsize=11,
            fontweight="bold", color="#333")

    # ── Row 2: Training Loop ──
    y2 = 6
    _box(0, y2, 2.5, 1.5, "ASM LUT", "incident_z40.npz", "#26A69A")
    _box(3, y2, 2.5, 1.5, "Collocation\nSampler", "Hierarchical z", "#FF8A65", text_color="#333")
    _box(6, y2, 3.5, 1.5, "Loss Function", "λ_H·L_H + λ_ph·L_ph\n+ λ_BC·L_BC", "#D32F2F")
    _box(10.5, y2, 2.5, 1.5, "Optimizer", "Adam → L-BFGS\nCosineAnnealing", "#E65100", text_color="#333")
    _box(14, y2, 2, 1.5, "Curriculum", "3-Stage λ", "#FF6F00", text_color="#333")

    _arr(2.5, y2+0.75, 6, y2+0.75, "#26A69A")
    _arr(5.5, y2+0.75, 6, y2+0.75, "#FF8A65")
    _arr(9.5, y2+0.75, 10.5, y2+0.75, "#D32F2F")
    _arr(14, y2+0.75, 13, y2+0.75, "#FF6F00")
    _arr(11.75, y2+1.5, 9.3, y1, "#E65100")  # backprop to PINN

    ax.text(9, y2+2, "Training Loop", ha="center", fontsize=11,
            fontweight="bold", color="#333")

    # ── Row 3: Outputs ──
    y3 = 3
    _box(0, y3, 2.5, 1.3, "Checkpoints", ".pt files", "#546E7A")
    _box(3.5, y3, 2.5, 1.3, "Red Flag\nDetector", "Auto Phase B check", "#F44336")
    _box(7, y3, 2.5, 1.3, "Experiment\nLogs", "JSON + .log", "#8D6E63")
    _box(10.5, y3, 2.5, 1.3, "Monitor\nNotebook", "Live plots", "#7B1FA2")

    ax.text(7, y3+1.8, "Validation & Monitoring", ha="center", fontsize=11,
            fontweight="bold", color="#333")

    # ── Row 4: Future Phases ──
    y4 = 0.5
    _box(0, y4, 3, 1.3, "FNO Surrogate", "PINN→FNO distill (Phase D)", "#00838F")
    _box(4, y4, 3.5, 1.3, "BoTorch Optimizer", "qNEHVI reverse design (D)", "#00695C")
    _box(8.5, y4, 3, 1.3, "Design Studio", "3-tab React UI (Phase E)", "#4527A0")
    _arr(3, y4+0.65, 4, y4+0.65, "#00838F")
    _arr(7.5, y4+0.65, 8.5, y4+0.65, "#00695C")

    ax.text(7, y4+1.8, "Future: Inverse Design Platform (Phase D-E)", ha="center",
            fontsize=11, fontweight="bold", color="#888")

    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# 7. Design Variable Space (v6 Section 2.5)
# ═══════════════════════════════════════════════════════════════════

def plot_design_space(figsize=(12, 5)):
    """Visualize the 4D design variable space and ranges."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    params = [
        ("δ_BM1", -10, 10, "μm", "BM1 offset\n(slit shift)", "#1565C0"),
        ("δ_BM2", -10, 10, "μm", "BM2 offset\n(slit shift)", "#C62828"),
        ("w₁", 5, 20, "μm", "BM1 aperture\n(slit width)", "#2E7D32"),
        ("w₂", 5, 20, "μm", "BM2 aperture\n(slit width)", "#6A1B9A"),
    ]

    for ax, (name, lo, hi, unit, desc, color) in zip(axes, params):
        ax.barh(0, hi - lo, left=lo, height=0.4, color=color, alpha=0.7,
                edgecolor="white", lw=2)
        ax.set_xlim(lo - 3, hi + 3)
        ax.set_ylim(-1, 1.5)
        ax.set_yticks([])
        ax.set_xlabel(unit, fontsize=9)
        ax.set_title(name, fontsize=13, fontweight="bold", color=color)
        ax.text((lo + hi) / 2, -0.6, desc, ha="center", fontsize=7.5, color="#555")
        ax.text(lo, 0.7, str(lo), ha="center", fontsize=9, fontweight="bold", color=color)
        ax.text(hi, 0.7, str(hi), ha="center", fontsize=9, fontweight="bold", color=color)
        ax.grid(True, alpha=0.15, axis="x")

    fig.suptitle("Design Variable Space (v6 Section 2.5) — Phase C: 4D",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
# 8. Project Structure (v6 Section 11.5)
# ═══════════════════════════════════════════════════════════════════

def plot_project_structure(figsize=(14, 7)):
    """Visualize the 3-layer hybrid project structure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1, 9)
    ax.axis("off")
    ax.set_title("Hybrid Project Structure (v6 Section 11.5)", fontsize=14, fontweight="bold")

    # Layer boxes
    layers = [
        (0, "Layer 1: notebooks/", ["Jupyter (.ipynb)", "Rapid experiment & viz",
         "Phase 1 workflow", "Executive reports"], "#FF7043", "Extract logic ->"),
        (5.5, "Layer 2: backend/", ["Python modules (.py)", "Class/function defs",
         "Reusable via import", "Test target (tests/)"], "#42A5F5", "Wrap as CLI/API ->"),
        (11, "Layer 3: scripts/ + api/", ["CLI: GPU training, batch",
         "FastAPI: production serve", "Docker: containerize", ""], "#66BB6A", None),
    ]

    for x, title, items, color, arrow in layers:
        box = FancyBboxPatch((x, 5), 4.8, 3.5, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="white", lw=2, alpha=0.12)
        ax.add_patch(box)
        ax.text(x + 0.3, 8.1, title, fontsize=11, fontweight="bold", color=color)
        for i, item in enumerate(items):
            if item:
                ax.text(x + 0.5, 7.3 - i * 0.5, f"• {item}",
                        fontsize=8, color="#444", family="monospace")
        if arrow:
            ax.annotate("", xy=(x + 5.2, 6.75), xytext=(x + 4.9, 6.75),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    # Data stores
    for label, desc, color, x in [("data/", "ASM LUT, LT results", "#78909C", 0.5),
                                   ("configs/", "YAML settings", "#8D6E63", 4),
                                   ("checkpoints/", "Models (.pt)", "#546E7A", 7.5),
                                   ("experiments/", "Logs, red flags", "#795548", 11)]:
        box = FancyBboxPatch((x, 1), 3, 1.2, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white", lw=1, alpha=0.7)
        ax.add_patch(box)
        ax.text(x+1.5, 1.8, label, ha="center", fontsize=9, fontweight="bold", color="white")
        ax.text(x+1.5, 1.25, desc, ha="center", fontsize=7, color="white", alpha=0.85)

    ax.text(7.5, 0.5, "Data & Artifacts", ha="center", fontsize=10, fontweight="bold", color="#555")

    plt.tight_layout()
    return fig, ax
