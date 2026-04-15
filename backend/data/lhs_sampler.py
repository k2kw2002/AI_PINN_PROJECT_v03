"""Latin Hypercube Sampling for design variable space exploration.

v6 Section 2.5: Design variables for Phase C
- delta_bm1: [-10, 10] um
- delta_bm2: [-10, 10] um
- w1: [5, 20] um
- w2: [5, 20] um
- theta: [0, 40] degrees (positive only, symmetric)

Used for:
1. LightTools batch simulation planning
2. PINN training data generation
3. Evaluation grid

Dependencies:
    pip install numpy scipy
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc


# Design variable bounds (v6 Section 2.5)
DESIGN_BOUNDS = {
    "delta_bm1": (-10.0, 10.0),   # um
    "delta_bm2": (-10.0, 10.0),   # um
    "w1": (5.0, 20.0),            # um
    "w2": (5.0, 20.0),            # um
}

ANGLE_BOUNDS = (0.0, 40.0)  # degrees (positive only, use symmetry)


def generate_lhs_samples(
    n_samples: int,
    include_angles: bool = True,
    n_angles: int = 5,
    seed: int = 42,
) -> dict:
    """Generate Latin Hypercube samples for design variable space.

    Args:
        n_samples: Number of design variable combinations.
        include_angles: If True, cross with angle samples.
        n_angles: Number of angle values to cross with.
        seed: Random seed for reproducibility.

    Returns:
        dict with:
            'design_params': (n_samples, 4) array [delta1, delta2, w1, w2]
            'param_names': list of parameter names
            'theta_values': (n_angles,) array if include_angles
            'all_configs': list of dicts, each a full simulation config
            'n_total': total number of simulations
    """
    # LHS for 4D design space
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples_unit = sampler.random(n=n_samples)  # (n_samples, 4) in [0, 1]

    # Scale to physical bounds
    bounds = list(DESIGN_BOUNDS.values())
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    samples = qmc.scale(samples_unit, l_bounds, u_bounds)

    param_names = list(DESIGN_BOUNDS.keys())

    result = {
        "design_params": samples,
        "param_names": param_names,
    }

    # Generate angle values
    if include_angles:
        theta_values = np.linspace(0, ANGLE_BOUNDS[1], n_angles)
        result["theta_values"] = theta_values
    else:
        theta_values = np.array([0.0])
        result["theta_values"] = theta_values

    # Create all simulation configs
    configs = []
    for i in range(n_samples):
        for theta in theta_values:
            configs.append({
                "sim_id": len(configs),
                "delta_bm1": float(samples[i, 0]),
                "delta_bm2": float(samples[i, 1]),
                "w1": float(samples[i, 2]),
                "w2": float(samples[i, 3]),
                "theta_deg": float(theta),
            })

    result["all_configs"] = configs
    result["n_total"] = len(configs)

    return result


def save_simulation_plan(result: dict, filepath: str):
    """Save simulation plan to JSON for LightTools batch runner."""
    import json
    plan = {
        "n_designs": len(result["design_params"]),
        "n_angles": len(result["theta_values"]),
        "n_total": result["n_total"],
        "theta_values": result["theta_values"].tolist(),
        "param_names": result["param_names"],
        "configs": result["all_configs"],
    }
    with open(filepath, "w") as f:
        json.dump(plan, f, indent=2)
