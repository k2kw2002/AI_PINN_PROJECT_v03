"""Collocation point sampler for PINN training.

v6 Section 8: Sampling strategy
- Domain: x in [0, 504], z in [0, 40] um
- Dense sampling near BM boundaries (z=20, z=40)
- BM slit/BM region direct sampling (Section 8.3)
"""
