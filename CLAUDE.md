# CLAUDE.md - UDFPS PINN Platform Project Rules

## Project Overview
UDFPS (Under-Display Fingerprint Sensor) BM optical inverse design automation platform.
Phase C: Pure PINN implementation for BM diffraction + phase distortion learning.

## Master Guide
All decisions must reference: `docs/udfps_pinn_master_guide_v6.md`
Always cite section numbers when making decisions.

## Absolute Red Lines (v6 Section "절대 하지 말 것")

1. **NO hard mask** - `mask = sigmoid(sharpness * slit_dist)` is BANNED
2. **Input is 8D only** - `(x, z, d1, d2, w1, w2, sin_theta, cos_theta)`. NO slit_dist (9D)
3. **L_H weight >= 0.5** - PDE is primary learning signal. `lambda_H = 1.0`
4. **L_BC is mandatory** - BM=0 is learned via loss, never removed
5. **PINN domain z=[0, 40]um** - FIXED. No shrink, no expand
6. **No result normalization hacks** - Fix the learning, not outputs
7. **No "temporary" workarounds** - Do it right from the start

## 10 Commandments (v6)
1. PINN learns physics. Never force externally.
2. L_Helmholtz is the main learning signal. Weight >= 1.0.
3. L_BC is mandatory. BM=0 learned via loss.
4. Input is 8D. No hints.
5. PINN domain is z=[0, 40]um. BM~OPD.
6. No hard mask. No slit_dist input.
7. Don't normalize results to match. Fix learning.
8. Don't start with "temporarily". Start correctly.
9. Reject workaround proposals. Find root causes.
10. No "success" declaration without z-interior fringe verification.

## Pipeline
```
TMM(AR) -> ASM(CG 550um) -> PINN(BM~OPD 40um) -> PSF
```

## Unit Conventions
- Coordinates (x, z): um (micrometers)
- AR thickness: nm
- Angles: degrees (I/O), radians (internal calc)
- Wavelength: 520 nm
- Wavenumber k: 18.37 um^-1
- Complex fields: (Re, Im) split or torch.complex64

## Key Physical Constants
- OPD pitch: 72 um
- OPD pixel width: 10 um
- Cover Glass: 550 um, n=1.52
- BM1: z=20 um, BM2: z=40 um
- Encap: 0~20 um, ILD: 20~40 um
- AR Phase 1 optimal: SiO2(34.6)/TiO2(25.9)/SiO2(20.7)/TiO2(169.5) nm

## Design Variables (Phase C: 4D)
- delta_bm1: BM1 offset [-10, 10] um
- delta_bm2: BM2 offset [-10, 10] um
- w1: BM1 aperture width [5, 20] um
- w2: BM2 aperture width [5, 20] um

## Development Workflow (v6 Section 15.5)
1. Explore in notebooks/ (small data, CPU, visualize)
2. Extract confirmed logic to backend/ (reusable modules)
3. Wrap in scripts/ or api/ (CLI/API for production)

## Git Convention
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Commit after each completed task
- Branch strategy: feature branches off main

## Red Flags to Watch For
- Uniform field inside z domain (plane wave = failed learning)
- BM region |U| > 0.05 (boundary not learned)
- Design variable insensitivity (PINN not parametric)
- L_H weight being lowered below 0.5
- Any "workaround" or "temporary" solution proposals
