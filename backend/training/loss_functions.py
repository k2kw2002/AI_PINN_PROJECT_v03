"""Loss functions for PINN training.

v6 Section 6: 4 loss components
- L_H (Helmholtz): PDE residual, weight >= 1.0
- L_phase: ASM boundary matching at z=40, weight ~0.5
- L_BC: BM boundary condition (U=0 at BM), weight ~0.5
- L_I: Intensity matching (optional), weight ~0.3

Rule: lambda_H >= max(lambda_phase, lambda_BC, lambda_I)
"""
