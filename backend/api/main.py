"""FastAPI application for UDFPS PINN Platform (Phase E).

v6 Section 14: API endpoints for Design Studio UI.

Usage:
    uvicorn backend.api.main:app --reload --port 8000

Endpoints:
    GET  /api/health          - System status
    POST /api/inference/psf   - PSF prediction from design params
    POST /api/design/run      - Run inverse design optimization
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.schemas import (
    BMDesignParams, PSFRequest, PSFResponse,
    DesignRequest, DesignResponse, DesignCandidate,
    HealthResponse,
)
from backend.core.pinn_model import PurePINN
from backend.core.fno_model import FNOSurrogate
from backend.physics.psf_metrics import compute_psf_7, compute_all_metrics

ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(
    title="UDFPS PINN Platform",
    description="BM Optical Inverse Design API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def serve_ui():
    """Serve Design Studio UI."""
    html_path = ROOT / "frontend" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "Frontend not found. Visit /docs for API docs."}

# ── Global model state ──
_state = {
    "pinn": None,
    "fno": None,
    "fno_p_mean": None,
    "fno_p_std": None,
    "device": torch.device("cpu"),
}


def _load_models():
    """Load PINN and FNO models at startup."""
    device = _state["device"]

    # PINN
    pinn_paths = [
        ROOT / "checkpoints" / "phase_c_final.pt",
        ROOT / "checkpoints" / "phase_c_cpu_10k.pt",
    ]
    for p in pinn_paths:
        if p.exists():
            ckpt = torch.load(p, map_location=device, weights_only=False)
            cfg = ckpt.get("config", {"hidden_dim": 64, "num_layers": 3, "num_freqs": 24, "omega_0": 30.0})
            model = PurePINN(**cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _state["pinn"] = model
            break

    # FNO
    fno_path = ROOT / "checkpoints" / "fno_surrogate.pt"
    if fno_path.exists():
        ckpt = torch.load(fno_path, map_location=device, weights_only=False)
        fno = FNOSurrogate().to(device)
        fno.load_state_dict(ckpt["model_state_dict"])
        fno.eval()
        _state["fno"] = fno
        _state["fno_p_mean"] = ckpt["p_mean"]
        _state["fno_p_std"] = ckpt["p_std"]


@app.on_event("startup")
async def startup():
    _load_models()


# ── Health ──

@app.get("/api/health", response_model=HealthResponse)
async def health():
    pinn = _state["pinn"]
    return HealthResponse(
        status="ok",
        pinn_loaded=pinn is not None,
        fno_loaded=_state["fno"] is not None,
        n_pinn_params=sum(p.numel() for p in pinn.parameters()) if pinn else 0,
        device=str(_state["device"]),
    )


# ── PSF Inference ──

@app.post("/api/inference/psf", response_model=PSFResponse)
async def inference_psf(req: PSFRequest):
    pinn = _state["pinn"]
    if pinn is None:
        raise HTTPException(status_code=503, detail="PINN model not loaded")

    p = req.params
    device = _state["device"]

    t0 = time.time()
    psf = compute_psf_7(pinn, p.delta_bm1, p.delta_bm2, p.w1, p.w2, p.theta_deg, device)
    elapsed_ms = (time.time() - t0) * 1000

    metrics = compute_all_metrics(psf)

    return PSFResponse(
        psf_7=metrics["psf_7"],
        mtf_ridge=metrics["mtf_ridge"],
        skewness=metrics["skewness"],
        throughput=metrics["throughput"],
        crosstalk=min(metrics["crosstalk"], 999),
        inference_ms=elapsed_ms,
    )


# ── Fingerprint Simulation ──

@app.post("/api/fingerprint/simulate")
async def fingerprint_simulate(req: PSFRequest):
    """Simulate fingerprint image with current design params."""
    pinn = _state["pinn"]
    if pinn is None:
        raise HTTPException(status_code=503, detail="PINN model not loaded")

    import base64, io
    from backend.physics.fingerprint_simulator import (
        load_fingerprint, simulate_fingerprint, compute_image_quality,
    )
    from backend.physics.psf_metrics import compute_psf_7

    p = req.params
    device = _state["device"]

    # Generate PSF at multiple angles
    angles = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    psf_by_angle = {}
    for theta in angles:
        psf = compute_psf_7(pinn, p.delta_bm1, p.delta_bm2, p.w1, p.w2, float(theta), device)
        psf_by_angle[float(theta)] = psf

    # Load real fingerprint image (417x417)
    fp_raw = load_fingerprint()
    fp_sim = simulate_fingerprint(psf_by_angle, fp_raw)
    quality = compute_image_quality(fp_raw, fp_sim)

    # Convert to base64 PNG for frontend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(fp_raw, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(fp_sim, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Simulated (corr={quality['correlation']:.3f})", fontsize=11)
    axes[1].axis("off")

    # Difference
    diff = np.abs(fp_raw - fp_sim)
    axes[2].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
    axes[2].set_title("Difference", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(f"d1={p.delta_bm1:.1f} d2={p.delta_bm2:.1f} w1={p.w1:.0f} w2={p.w2:.0f}",
                 fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "image_base64": img_b64,
        "quality": quality,
        "sensor_size_mm": 30.0,
        "sensor_pixels": 417,
        "pitch_um": 72.0,
    }


# ── Inverse Design ──

@app.post("/api/design/run", response_model=DesignResponse)
async def design_run(req: DesignRequest):
    """Fast inverse design using PINN direct evaluation."""
    pinn = _state["pinn"]
    if pinn is None:
        raise HTTPException(status_code=503, detail="PINN model not loaded")

    import time as _time

    device = _state["device"]
    t0 = _time.time()

    # Random search (fast, works without BoTorch/FNO)
    n_samples = 200
    np.random.seed(None)  # random each time

    candidates = []
    for _ in range(n_samples):
        d1 = float(np.random.uniform(-10, 10))
        d2 = float(np.random.uniform(-10, 10))
        w1 = float(np.random.uniform(5, 20))
        w2 = float(np.random.uniform(5, 20))

        psf = compute_psf_7(pinn, d1, d2, w1, w2, req.theta_deg, device, n_samples=50)
        m = compute_all_metrics(psf)

        candidates.append(DesignCandidate(
            params=BMDesignParams(
                delta_bm1=d1, delta_bm2=d2, w1=w1, w2=w2,
                theta_deg=req.theta_deg,
            ),
            mtf_ridge=m["mtf_ridge"],
            skewness=m["skewness"],
            throughput=m["throughput"],
            crosstalk=min(m["crosstalk"], 999),
        ))

    # Sort by MTF (descending) and pick top 5
    candidates.sort(key=lambda c: c.mtf_ridge, reverse=True)
    top5 = candidates[:5]
    elapsed = _time.time() - t0

    return DesignResponse(
        best=top5[0],
        pareto_front=top5,
        n_evaluations=n_samples,
        elapsed_sec=elapsed,
    )
