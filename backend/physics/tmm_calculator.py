"""TMM calculator for AR coating (Gorilla DX 4-layer).

v6 Section 3.2, 3.5.2: TMM module
- Input: AR layer thicknesses (d1~d4 in nm), angle theta (degrees), wavelength (nm)
- Output: TMMOutput(theta_deg, wavelength_nm, t_amplitude, phase_shift_deg)
- Phase C: fixed Phase 1 optimal values (34.6/25.9/20.7/169.5 nm)

Dependencies:
    pip install tmm numpy
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from tmm import coh_tmm


# ── v6 Section 3.5.2: TMM Output interface ──────────────────────────

@dataclass
class TMMOutput:
    """TMM calculation result in standard format (v6 Section 3.5.2)."""

    theta_deg: float          # Incidence angle in external medium (degrees)
    wavelength_nm: float      # Vacuum wavelength (nm)
    t_amplitude: float        # Transmission amplitude |t|
    phase_shift_deg: float    # Phase shift arg(t) (degrees)

    def to_complex(self) -> complex:
        """Complex transmission coefficient t = |t| * exp(i * dphi)."""
        phase_rad = math.radians(self.phase_shift_deg)
        return self.t_amplitude * cmath.exp(1j * phase_rad)


# ── v6 Section 2.4: Gorilla DX 4-layer AR coating ───────────────────

# Phase 1 optimal values (fixed for Phase C)
PHASE1_AR_THICKNESSES_NM = [34.6, 25.9, 20.7, 169.5]

# Refractive indices at 520 nm
N_SIO2 = 1.46
N_TIO2 = 2.35
N_AIR = 1.0
N_GLASS = 1.52
WAVELENGTH_NM = 520.0


class GorillaDXTMM:
    """Gorilla DX 4-layer AR coating TMM calculator.

    Stack (top to bottom, light direction):
        Air (n=1.0) -> SiO2 -> TiO2 -> SiO2 -> TiO2 -> Glass (n=1.52)

    Args:
        ar_thicknesses_nm: [d1_SiO2, d2_TiO2, d3_SiO2, d4_TiO2] in nm.
            Defaults to Phase 1 optimal values.
        wavelength_nm: Vacuum wavelength in nm. Default 520.
        n_incident: Refractive index of incident medium. Default 1.0 (air).
        n_substrate: Refractive index of substrate (CG). Default 1.52.
    """

    def __init__(
        self,
        ar_thicknesses_nm: list[float] | None = None,
        wavelength_nm: float = WAVELENGTH_NM,
        n_incident: float = N_AIR,
        n_substrate: float = N_GLASS,
    ):
        if ar_thicknesses_nm is None:
            ar_thicknesses_nm = list(PHASE1_AR_THICKNESSES_NM)

        self.wavelength_nm = wavelength_nm
        self.n_incident = n_incident
        self.n_substrate = n_substrate

        # TMM layer stack: [incident, SiO2, TiO2, SiO2, TiO2, substrate]
        self.n_list = [
            n_incident,
            N_SIO2, N_TIO2, N_SIO2, N_TIO2,
            n_substrate,
        ]
        self.d_list = [
            np.inf,
            ar_thicknesses_nm[0],
            ar_thicknesses_nm[1],
            ar_thicknesses_nm[2],
            ar_thicknesses_nm[3],
            np.inf,
        ]

    def compute(self, theta_deg: float) -> TMMOutput:
        """Compute transmission through AR coating for a single angle.

        Args:
            theta_deg: Incidence angle in external medium (degrees).
                Must be within [-41.1, 41.1] (TIR limit).

        Returns:
            TMMOutput with amplitude and phase.
        """
        theta_rad = math.radians(theta_deg)

        # Average s and p polarizations for unpolarized illumination
        result_s = coh_tmm("s", self.n_list, self.d_list, theta_rad, self.wavelength_nm)
        result_p = coh_tmm("p", self.n_list, self.d_list, theta_rad, self.wavelength_nm)
        t_avg = (result_s["t"] + result_p["t"]) / 2

        return TMMOutput(
            theta_deg=theta_deg,
            wavelength_nm=self.wavelength_nm,
            t_amplitude=abs(t_avg),
            phase_shift_deg=math.degrees(cmath.phase(t_avg)),
        )

    def compute_lut(self, theta_array_deg: np.ndarray) -> dict:
        """Compute t(theta), dphi(theta) LUT for an array of angles.

        Args:
            theta_array_deg: 1D array of angles (degrees).

        Returns:
            dict with keys:
                'theta_deg': (N,) angles
                't_amplitude': (N,) transmission amplitudes
                'phase_shift_deg': (N,) phase shifts in degrees
                't_complex': (N,) complex transmission coefficients
        """
        n_theta = len(theta_array_deg)
        t_amp = np.zeros(n_theta)
        t_phase = np.zeros(n_theta)
        t_complex = np.zeros(n_theta, dtype=np.complex128)

        for i, theta in enumerate(theta_array_deg):
            out = self.compute(float(theta))
            t_amp[i] = out.t_amplitude
            t_phase[i] = out.phase_shift_deg
            t_complex[i] = out.to_complex()

        return {
            "theta_deg": np.array(theta_array_deg),
            "t_amplitude": t_amp,
            "phase_shift_deg": t_phase,
            "t_complex": t_complex,
        }
