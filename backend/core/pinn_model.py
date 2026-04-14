"""Pure PINN model (8D input, SIREN+Fourier).

v6 Section 5: Network architecture
- Input: (x, z, delta_bm1, delta_bm2, w1, w2, sin_theta, cos_theta) = 8D
- Output: (Re(U), Im(U)) = 2D
- NO hard mask, NO slit_dist input

Architecture (v6 Section 5.2):
  Input 8D -> Fourier Embedding (48 freq -> 96 features)
  -> SIREN Layer 1 (96 -> hidden_dim, sin(omega_0 * x))
  -> SIREN Layer 2..N (hidden_dim -> hidden_dim, sin)
  -> Linear output (hidden_dim -> 2)

Dependencies:
    pip install torch
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── v6 Section 5.1: Input normalization constants ─────────────────

X_MAX = 504.0       # x / 504
Z_MAX = 40.0        # z / 40
DELTA_MAX = 10.0    # delta / 10
W_CENTER = 12.5     # (w - 12.5) / 7.5
W_SCALE = 7.5


class InputNormalizer(nn.Module):
    """Normalize raw 8D input to [-1, 1] range (v6 Section 5.1)."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 8) raw coordinates
                [x, z, delta_bm1, delta_bm2, w1, w2, sin_theta, cos_theta]
        Returns:
            (N, 8) normalized coordinates
        """
        x = coords[:, 0:1] / X_MAX                          # [0, 1]
        z = coords[:, 1:2] / Z_MAX                          # [0, 1]
        d1 = coords[:, 2:3] / DELTA_MAX                     # [-1, 1]
        d2 = coords[:, 3:4] / DELTA_MAX                     # [-1, 1]
        w1 = (coords[:, 4:5] - W_CENTER) / W_SCALE          # [-1, 1]
        w2 = (coords[:, 5:6] - W_CENTER) / W_SCALE          # [-1, 1]
        sin_th = coords[:, 6:7]                              # [-1, 1]
        cos_th = coords[:, 7:8]                              # [0, 1]
        return torch.cat([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier feature embedding for periodic structure learning.

    Maps each input dimension through sin/cos with learned frequencies.
    8D input -> 2 * num_freqs features per dimension -> select top num_freqs total.

    v6 Section 5.2: 48 frequencies -> 96 features.
    """

    def __init__(self, in_dim: int = 8, num_freqs: int = 48, sigma: float = 10.0):
        super().__init__()
        self.num_freqs = num_freqs
        # Random frequency matrix (fixed, not trained)
        B = torch.randn(in_dim, num_freqs) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_freqs  # sin + cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) normalized input
        Returns:
            (N, 2 * num_freqs) Fourier features
        """
        proj = x @ self.B  # (N, num_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


class SIRENLayer(nn.Module):
    """Single SIREN layer: Linear -> sin(omega_0 * x).

    Uses SIREN-specific initialization (Sitzmann et al. 2020).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features: int, is_first: bool):
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class PurePINN(nn.Module):
    """Pure PINN for UDFPS BM diffraction learning (v6 Section 5).

    NO hard mask. NO slit_dist input.
    Input 8D -> Fourier -> SIREN hidden layers -> Output 2D (Re, Im).

    Args:
        hidden_dim: Hidden layer width. Default 128.
        num_layers: Number of SIREN hidden layers. Default 4.
        num_freqs: Number of Fourier frequencies. Default 48.
        omega_0: SIREN frequency parameter. Default 30.0.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_freqs: int = 48,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.normalizer = InputNormalizer()
        self.embedding = FourierFeatureEmbedding(in_dim=8, num_freqs=num_freqs)

        embed_dim = self.embedding.out_dim  # 96

        # SIREN hidden layers
        layers = []
        layers.append(SIRENLayer(embed_dim, hidden_dim, omega_0=omega_0, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (no activation)
        self.output_layer = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 8) raw coordinates
                [x, z, delta_bm1, delta_bm2, w1, w2, sin_theta, cos_theta]
        Returns:
            (N, 2) complex field [Re(U), Im(U)]
        """
        x = self.normalizer(coords)
        x = self.embedding(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)
