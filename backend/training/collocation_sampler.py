"""Collocation point sampler for PINN training.

v6 Section 8: Hierarchical sampling strategy
- Domain: x in [0, 504], z in [0, 40] um
- Dense sampling near BM boundaries (z=20, z=40)
- Design variables: delta in [-10,10], w in [5,20]
- Angle: theta in [-41.1, 41.1] degrees

Dependencies:
    pip install torch
"""
from __future__ import annotations

import math

import torch

SIN_MAX = math.sin(math.radians(41.1))  # ~0.6574


def hierarchical_collocation(
    n_total: int,
    device: torch.device,
) -> torch.Tensor:
    """Hierarchical collocation sampling for PINN domain (v6 Section 8.2).

    z-distribution:
        15%: z in [39, 40] - inlet boundary + BM2
        25%: z in [21, 39] - ILD propagation
        15%: z in [19, 21] - BM1 boundary
        25%: z in [1, 19]  - Encap propagation
        10%: z in [0, 1]   - OPD outlet
        10%: z in [0, 40]  - uniform buffer

    Args:
        n_total: Total number of collocation points.
        device: Torch device.

    Returns:
        (n_total, 8) tensor: [x, z, d1, d2, w1, w2, sin_theta, cos_theta]
    """
    # z-region allocation (v6 Section 8.1)
    counts = {
        "inlet_bm2": int(n_total * 0.15),
        "ild": int(n_total * 0.25),
        "bm1": int(n_total * 0.15),
        "encap": int(n_total * 0.25),
        "outlet": int(n_total * 0.10),
    }
    counts["buffer"] = n_total - sum(counts.values())

    all_x, all_z = [], []

    # Inlet + BM2 (z=39~40)
    n = counts["inlet_bm2"]
    all_x.append(torch.rand(n, device=device) * 504)
    all_z.append(39.0 + torch.rand(n, device=device))

    # ILD (z=21~39)
    n = counts["ild"]
    all_x.append(torch.rand(n, device=device) * 504)
    all_z.append(21.0 + torch.rand(n, device=device) * 18.0)

    # BM1 (z=19~21)
    n = counts["bm1"]
    all_x.append(torch.rand(n, device=device) * 504)
    all_z.append(19.0 + torch.rand(n, device=device) * 2.0)

    # Encap (z=1~19)
    n = counts["encap"]
    all_x.append(torch.rand(n, device=device) * 504)
    all_z.append(1.0 + torch.rand(n, device=device) * 18.0)

    # Outlet (z=0~1)
    n = counts["outlet"]
    all_x.append(torch.rand(n, device=device) * 504)
    all_z.append(torch.rand(n, device=device))

    # Buffer (uniform over full domain)
    n = counts["buffer"]
    all_x.append(torch.rand(n, device=device) * 504)
    all_z.append(torch.rand(n, device=device) * 40.0)

    x = torch.cat(all_x)
    z = torch.cat(all_z)
    N = x.shape[0]

    # Design variables: uniform random (v6 Section 2.5)
    d1 = torch.rand(N, device=device) * 20 - 10       # [-10, 10]
    d2 = torch.rand(N, device=device) * 20 - 10       # [-10, 10]
    w1 = torch.rand(N, device=device) * 15 + 5        # [5, 20]
    w2 = torch.rand(N, device=device) * 15 + 5        # [5, 20]

    # Angle: uniform sin(theta) (v6 Section 2.6)
    sin_th = (torch.rand(N, device=device) * 2 - 1) * SIN_MAX
    cos_th = torch.sqrt(1 - sin_th**2)

    return torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
