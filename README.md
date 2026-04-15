# UDFPS PINN Platform

**Under-Display Fingerprint Sensor** BM Optical Inverse Design Automation Platform using Physics-Informed Neural Networks.

## North Star

> **PINN 기반의 역설계 자동화 플랫폼**
>
> Parametric PINN이 BM 회절/위상 왜곡을 학습하여, 목표 성능에서 최적 설계변수를 자동으로 찾아내는 시스템

## Pipeline

```
TMM (AR ~300nm)  →  ASM (CG 550um)  →  PINN (BM~OPD 40um)  →  PSF (7 pixels)
  t(θ), Δφ(θ)       U(x, z=40, θ)       U(x, z=0)              MTF, skew, T
```

## Quick Start

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate.bat          # Windows cmd
pip install -r requirements.txt

# 2. Generate ASM LUT (run once)
python -c "
import sys; sys.path.insert(0,'.')
from backend.physics.tmm_calculator import GorillaDXTMM
from backend.physics.asm_propagator import generate_incident_lut
import numpy as np
tmm = GorillaDXTMM()
theta = np.arange(-41, 42, 1.0)
x = np.linspace(0, 504.0, 4096)
lut = generate_incident_lut(tmm, theta, x)
np.savez('data/asm_luts/incident_z40.npz', **lut,
         z_inlet=np.float32(40), cg_thick=np.float32(550),
         wavelength_nm=np.float32(520), n_cg=np.float32(1.52))
"

# 3. CPU Validation
python scripts/train_phase_c.py --epochs 300 --hidden_dim 64 --num_layers 3 --device cpu --tag cpu_test

# 4. GPU Full Training
python scripts/train_phase_c.py --config configs/phase_c_full_gpu.yaml
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_project_overview` | Architecture diagrams (8 visualizations) |
| `01_exploration/00_pinn_concepts` | AI concepts for HW engineers |
| `01_exploration/01_stack_visualization` | Physical stack structure |
| `02_phase_c/01_asm_lut_generation` | TMM+ASM → LUT |
| `02_phase_c/02_pinn_cpu_validation` | CPU structure validation |
| `02_phase_c/03_pinn_training_monitor` | GPU training monitor |
| `02_phase_c/04_pinn_evaluation` | Post-training validation |
| `02_phase_c/07_lighttools_integration` | LT → L_I targets |

## Project Structure

```
backend/           Reusable Python modules
  core/            PINN model (8D SIREN+Fourier)
  physics/         TMM, ASM, boundary conditions
  training/        Loss functions, sampler, curriculum
  data/            LHS sampling, LightTools runner
notebooks/         Jupyter experiments & visualization
scripts/           CLI training & batch execution
configs/           YAML training configurations
data/              ASM LUT, LT results
checkpoints/       Trained model weights
```

## Key Rules (v6 Red Lines)

1. **NO hard mask** — BM=0 learned via L_BC loss
2. **Input is 8D** — (x, z, δ₁, δ₂, w₁, w₂, sinθ, cosθ). NO slit_dist
3. **λ_H ≥ 0.5** — PDE is primary learning signal
4. **PINN domain z=[0, 40]μm** — fixed
5. **No workarounds** — do it right from the start

## Reference

- Master guide: `docs/udfps_pinn_master_guide_v6.md`
- GitHub: https://github.com/k2kw2002/AI_PINN_PROJECT_v03
