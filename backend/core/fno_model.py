"""FNO Surrogate model for Phase D distillation.

v6 Section 13.6: Fourier Neural Operator
- Input: (delta_bm1, delta_bm2, w1, w2, theta) = 5D design parameters
- Output: PSF 7 pixels
- Purpose: Fast surrogate for BoTorch optimization (0.8ms/inference)

Architecture:
  5D params → Lift → SpectralConv1d × 4 → Project → 7 PSF pixels

Training: distill from trained PINN (10,000 pairs)

Dependencies:
    pip install torch numpy
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """Fourier domain learnable kernel (v6 Section 13.6)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.modes], self.weights,
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNOSurrogate(nn.Module):
    """FNO: 5D design params → 7 PSF pixels (v6 Section 13.6).

    Fast surrogate (~0.8ms) for BoTorch inverse design.
    Distilled from trained PINN.

    Args:
        hidden_channels: Width of spectral layers.
        modes: Number of Fourier modes.
        n_fourier_layers: Number of FNO layers.
        spatial_size: Internal spatial resolution.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        modes: int = 16,
        n_fourier_layers: int = 4,
        spatial_size: int = 128,
    ):
        super().__init__()
        self.spatial_size = spatial_size

        # 5D → hidden
        self.lift = nn.Linear(5, hidden_channels)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(hidden_channels, hidden_channels, modes)
            for _ in range(n_fourier_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1)
            for _ in range(n_fourier_layers)
        ])

        self.norm = nn.InstanceNorm1d(hidden_channels)
        self.act = nn.GELU()

        # hidden → 7 PSF
        self.project = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 7),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (batch, 5) design parameters [delta1, delta2, w1, w2, theta]
        Returns:
            psf: (batch, 7) predicted PSF
        """
        # Lift to spatial
        x = self.lift(p)  # (B, hidden)
        x = x.unsqueeze(-1).expand(-1, -1, self.spatial_size)  # (B, hidden, spatial)

        # Fourier layers
        for fourier, w in zip(self.fourier_layers, self.w_layers):
            x_f = fourier(x)
            x_w = w(x)
            x = self.norm(x_f + x_w)
            x = self.act(x)

        # Project to PSF
        x_mean = x.mean(dim=-1)  # (B, hidden)
        psf = self.project(x_mean)  # (B, 7)
        psf = torch.relu(psf)  # non-negative intensity

        return psf
