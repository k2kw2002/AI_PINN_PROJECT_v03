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

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

# Also load immediately for TestClient compatibility
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


# ── Inverse Design ──

@app.post("/api/design/run", response_model=DesignResponse)
async def design_run(req: DesignRequest):
    fno = _state["fno"]
    if fno is None:
        raise HTTPException(status_code=503, detail="FNO model not loaded")

    from backend.core.botorch_optimizer import run_inverse_design

    fno_path = str(ROOT / "checkpoints" / "fno_surrogate.pt")

    try:
        result = run_inverse_design(
            fno_checkpoint=fno_path,
            n_initial=20,
            n_iterations=req.n_iterations,
            batch_size=4,
            theta_deg=req.theta_deg,
            device=_state["device"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    # Build response
    best = DesignCandidate(
        params=BMDesignParams(
            delta_bm1=result.best_metrics["delta_bm1"],
            delta_bm2=result.best_metrics["delta_bm2"],
            w1=result.best_metrics["w1"],
            w2=result.best_metrics["w2"],
            theta_deg=req.theta_deg,
        ),
        mtf_ridge=result.best_metrics["mtf_ridge"],
        skewness=result.best_metrics["skewness"],
        throughput=result.best_metrics["throughput"],
        crosstalk=0.0,
    )

    pareto = []
    for i in range(len(result.pareto_params)):
        p = result.pareto_params[i]
        o = result.pareto_objectives[i]
        pareto.append(DesignCandidate(
            params=BMDesignParams(
                delta_bm1=float(p[0]), delta_bm2=float(p[1]),
                w1=float(p[2]), w2=float(p[3]), theta_deg=req.theta_deg,
            ),
            mtf_ridge=float(o[0]),
            skewness=float(-o[2]),
            throughput=float(o[1]),
            crosstalk=0.0,
        ))

    return DesignResponse(
        best=best,
        pareto_front=pareto,
        n_evaluations=len(result.all_params),
        elapsed_sec=result.elapsed_sec,
    )
