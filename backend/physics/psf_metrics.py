"""PSF calculation and metrics (MTF@ridge, skewness, throughput, crosstalk).

v6 Section 3.2: PSF module
- Input: complex field U(x, z=0) at OPD plane
- Output: PSF[7], MTF@ridge, skewness, throughput, crosstalk_ratio
"""
