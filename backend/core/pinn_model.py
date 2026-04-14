"""Pure PINN model (8D input, SIREN+Fourier).

v6 Section 5: Network architecture
- Input: (x, z, delta_bm1, delta_bm2, w1, w2, sin_theta, cos_theta) = 8D
- Output: (Re(U), Im(U)) = 2D
- NO hard mask, NO slit_dist input
"""
