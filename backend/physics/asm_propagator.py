"""ASM propagator for Cover Glass (550um free-space propagation).

v6 Section 3.2, 3.5.2, 3.5.3: ASM module
- Propagates complex field through a homogeneous medium using FFT-based
  Angular Spectrum Method (ASM).
- Input: plane wave after AR coating at z=590
- Output: complex field U(x, theta) at z=40 (PINN boundary)
- Propagation distance: 550 um through CG (n=1.52)

Dependencies:
    pip install numpy
"""
from __future__ import annotations

import math

import numpy as np

from backend.physics.tmm_calculator import TMMOutput


# ── v6 Section 2.2: Physical constants ───────────────────────────────

CG_THICKNESS_UM = 550.0  # Cover Glass thickness
N_CG = 1.52              # CG refractive index
WAVELENGTH_UM = 0.520     # 520 nm in um


class ASMPropagator:
    """Angular Spectrum Method propagator for homogeneous media.

    Propagates a 1D complex field through a uniform medium using FFT.

    Args:
        n_medium: Refractive index of propagation medium.
        wavelength_um: Vacuum wavelength in um.
        propagation_distance_um: Distance to propagate in um.
    """

    def __init__(
        self,
        n_medium: float = N_CG,
        wavelength_um: float = WAVELENGTH_UM,
        propagation_distance_um: float = CG_THICKNESS_UM,
    ):
        self.n = n_medium
        self.wavelength_um = wavelength_um
        self.distance_um = propagation_distance_um
        self.k = 2 * np.pi * n_medium / wavelength_um  # k in medium (um^-1)

    def propagate(self, U_in: np.ndarray, dx_um: float) -> np.ndarray:
        """Propagate 1D complex field using ASM.

        Args:
            U_in: (N,) complex128 input field.
            dx_um: Spatial sampling interval in um.

        Returns:
            U_out: (N,) complex128 propagated field.
        """
        N = len(U_in)

        # Spatial frequencies (cycles / um)
        fx = np.fft.fftfreq(N, d=dx_um)
        kx = 2 * np.pi * fx

        # Axial wavenumber: kz = sqrt(k^2 - kx^2)
        kz_sq = self.k**2 - kx**2

        # Propagating modes: real kz; evanescent modes: imaginary kz (decay)
        kz = np.where(
            kz_sq >= 0,
            np.sqrt(np.maximum(kz_sq, 0)),
            1j * np.sqrt(np.maximum(-kz_sq, 0)),
        )

        # Transfer function for forward propagation
        H = np.exp(1j * kz * self.distance_um)

        # Propagate via FFT
        U_out = np.fft.ifft(np.fft.fft(U_in) * H)

        return U_out

    def make_initial_field(
        self,
        tmm_out: TMMOutput,
        x_um: np.ndarray,
    ) -> np.ndarray:
        """Create initial plane wave field after AR coating (v6 Section 3.5.2).

        The field at the top of the Cover Glass (z=590) is a plane wave
        modulated by the TMM transmission coefficient.

        Args:
            tmm_out: TMM output for a given angle.
            x_um: (N,) spatial coordinates in um.

        Returns:
            U_init: (N,) complex128 initial field at z=590.
        """
        theta_rad = math.radians(tmm_out.theta_deg)
        phase_rad = math.radians(tmm_out.phase_shift_deg)

        # Transverse wavenumber (conserved across interfaces by Snell's law)
        # kx = k0 * n_incident * sin(theta_incident) = k0 * sin(theta_air)
        k0 = 2 * np.pi / self.wavelength_um
        kx = k0 * math.sin(theta_rad)  # n_air = 1.0

        # Plane wave with AR amplitude and phase
        U_init = tmm_out.t_amplitude * np.exp(1j * (phase_rad + kx * x_um))

        return U_init


def generate_incident_lut(
    tmm_calculator,
    theta_array_deg: np.ndarray,
    x_array_um: np.ndarray,
) -> dict:
    """Generate complete ASM LUT for PINN boundary at z=40 (v6 Section 3.5.3).

    For each angle, computes TMM -> initial field -> ASM propagation -> z=40 field.

    Args:
        tmm_calculator: GorillaDXTMM instance.
        theta_array_deg: (N_theta,) angles in degrees.
        x_array_um: (N_x,) x-coordinates in um (uniformly spaced).

    Returns:
        dict with LUT data ready for np.savez:
            'theta_values': (N_theta,) float32
            'x_values': (N_x,) float32
            'U_re': (N_theta, N_x) float32
            'U_im': (N_theta, N_x) float32
    """
    dx = x_array_um[1] - x_array_um[0]
    N_theta = len(theta_array_deg)
    N_x = len(x_array_um)

    asm = ASMPropagator()

    U_re = np.zeros((N_theta, N_x), dtype=np.float32)
    U_im = np.zeros((N_theta, N_x), dtype=np.float32)

    for i, theta in enumerate(theta_array_deg):
        # TMM: AR transmission
        tmm_out = tmm_calculator.compute(float(theta))

        # Initial field at z=590 (top of CG)
        U_init = asm.make_initial_field(tmm_out, x_array_um)

        # ASM: propagate 550 um through CG to z=40
        U_z40 = asm.propagate(U_init, dx)

        U_re[i] = U_z40.real.astype(np.float32)
        U_im[i] = U_z40.imag.astype(np.float32)

    return {
        "theta_values": theta_array_deg.astype(np.float32),
        "x_values": x_array_um.astype(np.float32),
        "U_re": U_re,
        "U_im": U_im,
    }
