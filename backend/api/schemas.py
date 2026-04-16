"""API schemas (Pydantic models) for Phase E.

v6 Section 14: API specifications.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Design Parameters ──

class BMDesignParams(BaseModel):
    """BM design variables (v6 Section 2.5)."""
    delta_bm1: float = Field(0.0, ge=-10, le=10, description="BM1 offset (um)")
    delta_bm2: float = Field(0.0, ge=-10, le=10, description="BM2 offset (um)")
    w1: float = Field(10.0, ge=5, le=20, description="BM1 aperture width (um)")
    w2: float = Field(10.0, ge=5, le=20, description="BM2 aperture width (um)")
    theta_deg: float = Field(0.0, ge=0, le=41, description="Incidence angle (degrees)")


# ── Inference ──

class PSFRequest(BaseModel):
    """Request for PSF inference."""
    params: BMDesignParams


class PSFResponse(BaseModel):
    """PSF inference result."""
    psf_7: list[float]
    mtf_ridge: float
    skewness: float
    throughput: float
    crosstalk: float
    inference_ms: float


# ── Inverse Design ──

class DesignSpec(BaseModel):
    """Target specifications for inverse design."""
    mtf_ridge_min: float = Field(0.60, ge=0.1, le=0.95)
    skewness_max: float = Field(0.10, ge=0.01, le=0.50)
    throughput_min: float = Field(0.60, ge=0.1, le=0.95)


class DesignRequest(BaseModel):
    """Inverse design request."""
    spec: DesignSpec = DesignSpec()
    n_iterations: int = Field(30, ge=5, le=100)
    theta_deg: float = Field(0.0, ge=0, le=41)


class DesignCandidate(BaseModel):
    """A single design candidate from Pareto front."""
    params: BMDesignParams
    mtf_ridge: float
    skewness: float
    throughput: float
    crosstalk: float


class DesignResponse(BaseModel):
    """Inverse design result."""
    best: DesignCandidate
    pareto_front: list[DesignCandidate]
    n_evaluations: int
    elapsed_sec: float


# ── Health ──

class HealthResponse(BaseModel):
    """System health check."""
    status: str
    pinn_loaded: bool
    fno_loaded: bool
    n_pinn_params: int
    device: str
