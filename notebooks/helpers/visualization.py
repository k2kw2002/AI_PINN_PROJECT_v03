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
