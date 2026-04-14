# AGENTS.md - Domain Rules for AI Coding Agents

## Project: UDFPS PINN Platform (Phase C)

### Absolute Rules (v6 Red Lines)
1. **NO hard mask** - BM=0 must be learned via L_BC loss, never forced
2. **Input is 8D** - (x, z, d1, d2, w1, w2, sin_theta, cos_theta). NO slit_dist
3. **L_H weight >= 0.5** - PDE is the primary learning signal
4. **L_BC is mandatory** - Never remove BM boundary loss
5. **PINN domain z=[0, 40]um** - Never shrink or expand
6. **No result normalization tricks** - Fix learning, not outputs
7. **No "temporary" workarounds** - Do it right from the start

### Unit Conventions
- Coordinates (x, z): um
- AR thickness (d1~d4): nm
- Angles: degrees (I/O), radians (internal)
- Wavelength: 520 nm
- Wavenumber k: um^-1 (18.37)
- Complex fields: (Re, Im) split or torch.complex64

### Pipeline
```
TMM(AR) -> ASM(CG 550um) -> PINN(BM~OPD 40um) -> PSF
```

### Key Constants
- OPD pitch: 72 um
- OPD pixel width: 10 um
- Cover Glass: 550 um, n=1.52
- BM1: z=20 um, BM2: z=40 um
- Encap: 0~20 um, ILD: 20~40 um
