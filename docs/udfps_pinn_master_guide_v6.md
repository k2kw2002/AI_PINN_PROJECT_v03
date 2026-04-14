# UDFPS BM 광학 역설계 플랫폼 — 정석 PINN 마스터 가이드 v6

> **이 문서 하나로 처음부터 끝까지 구현 가능한 완전한 가이드**
>
> **v6 핵심 철학: 하이브리드 개발 (Jupyter + Python)**
> Phase 1 노트북 성공 방식 계승 + Phase E 제품화 준비
>
> v6 추가 사항 (v5 대비):
> - **섹션 11.5**: 하이브리드 프로젝트 구조 (notebooks/ + backend/ + scripts/)
> - **섹션 15.5**: 개발 워크플로우 (실험→모듈→배포)
> - **섹션 17.5**: 노트북 템플릿 (Phase별 .ipynb 목차)
>
> v5 기반 확정 사항:
> - **파이프라인**: TMM(AR) → ASM(CG 550μm) → **PINN(BM~OPD, 40μm)** → PSF
> - **PINN 도메인**: z ∈ [0, 40]μm
> - **Phase B 우회 기법 절대 금지**
> - **모든 수치 실측/Phase 1 기반**
> - **인터페이스 규약, Buffer Zone, Warm Start, BM 직접 샘플링, LT 방어, Red Flag 감지**
>
> **개발 도구 유연성**:
> - 에이전트: Claude Code / Cline / Cursor / Aider / 수동 작성
> - IDE: Jupyter Lab / VS Code / PyCharm
> - 환경: 집 CPU + 회사 GPU (Git 동기화)

---

## ⚙ 툴 중립성 (Tool-Agnostic)

**이 문서는 어떤 AI 코딩 에이전트로도 구현 가능합니다.**

- Claude Code (Anthropic CLI)
- Cline (VS Code extension)
- Cursor (AI IDE)
- Aider (터미널)
- 또는 수동 작성

문서 내 "Claude Code" 언급은 예시일 뿐이며, 사용하는 에이전트에 맞게 해석하면 됩니다.
에이전트에 전달 시 이 문서 전체를 컨텍스트로 제공하고, 섹션 번호를 참조하여 지시하세요.

예시 프롬프트:
```
"첨부한 마스터 가이드 v4의 Phase C를 구현해줘.
섹션 5(Loss 구성)와 섹션 12(핵심 모듈)를 기준으로
backend/training/train_phase_c.py를 작성해줘."
```

Git으로 코드 관리하면 집→회사 이동도 자유롭습니다 (집에서 Cline → 회사에서 Cursor 가능).

---

## 🛑 절대 하지 말 것 (Phase B 실패 교훈)

**이 섹션은 프로젝트 생존을 위한 레드라인입니다. 어떤 이유로도 위반하면 Phase B 실패를 반복합니다.**

### 🚫 금지사항 #1: Hard Mask 우회 금지

```python
# ✗ 절대 금지 (Phase B 방식)
A_raw = network(coords)
mask = sigmoid(sharpness * slit_dist)
A_final = A_raw * mask  # BM=0 수학적 강제

# ✓ 정석 (Phase C)
A_final = network(coords)  # 순수 네트워크
L_BC = mean(|U|² at BM regions)  # loss로 학습
```

### 🚫 금지사항 #2: 입력 차원 부풀리기 금지

```python
# ✗ 절대 금지
input_9d = (x, z, d1, d2, w1, w2, sin_θ, cos_θ, slit_dist)  # 힌트 주입

# ✓ 정석
input_8d = (x, z, d1, d2, w1, w2, sin_θ, cos_θ)
```

### 🚫 금지사항 #3: L_Helmholtz 가중치 낮추기 금지

```python
# ✗ 절대 금지
lam_pde = 0.01  # "physics fine-tuning only"

# ✓ 정석
λ_H = 1.0      # PDE가 주 학습 신호
λ_phase = 0.5
λ_BC = 0.5
λ_I = 0.3
# 규칙: λ_H >= max(λ_phase, λ_BC, λ_I)
```

### 🚫 금지사항 #4: 도메인 축소 금지 (PINN 영역만)

PINN 도메인은 **z=[0, 40]μm** (BM~OPD). 크로스토크 및 회절 학습을 위해 필수.
```
✗ z=[0, 20] (BM1 아래만, BM2 회절 미학습)
✗ z=[0, 10] (Encap만, BM 회절 미학습)
✓ z=[0, 40] (BM2 아래~OPD, 완전한 BM 회절 학습)

파이프라인:
- AR → TMM (박막 이론, 정확)
- CG 550μm → ASM (자유공간 전파, 설계변수 무관)
- BM~OPD 40μm → PINN (회절+위상 결합, 설계변수 의존)
```

### 🚫 금지사항 #5: L_BC 없이 학습 시작 금지

```
PDE만으로는 유일해 없음 → 평면파(자명해)로 수렴
L_BC 필수 (BM1, BM2 경계에서 U=0)
```

### 🚫 금지사항 #6: 결과 맞추기 금지

```python
# ✗ 절대 금지
psf_pinn_corrected = psf_pinn / psf_pinn.sum() * psf_asm.sum()

# ✓ 정석
# PINN 학습 자체를 올바르게. 정규화로 가리지 말 것.
```

### 🚫 금지사항 #7: "PINN 이름만 빌리기" 금지

```
✗ ASM이 주 엔진, PINN은 mask
✓ PINN이 회절/위상/간섭 직접 학습
```

---

## 🔍 Red Flag 조기 경보

### 🚩 Red Flag #1: 우회 기법 등장
"SIREN이 학습 못하니까 hard mask 추가" → Phase B 실패 패턴

### 🚩 Red Flag #2: "일단" 사고
"일단 hard mask로..." → 영구 부채

### 🚩 Red Flag #3: 정당화 시도
"Hard constraint는 학계에서도..." → 과거 실수 재포장

### 🚩 Red Flag #4: PINN 본질 훼손
"PINN 어려우니 ASM으로 대체" → 북극성 흔들기

---

## ✅ 결정 체크리스트

새로운 기법 도입 전 자문:

1. Pure PINN 정의 유지?
2. Phase B 실패 패턴과 유사?
3. 북극성(PINN 기반 역설계 자동화) 부합?
4. "쉬운 길" 제안인가?
5. "진짜 학습 증거" 제시 가능?

**5개 중 하나라도 부정적 → 중단**

---

## 📋 Phase B 실패 타임라인 (반면교사)

```
1. 원본 v2 계획: 정석 PINN 설계 (올바른 방향)
2. SIREN의 slit indicator 학습 어려움 (갈림길)
3. 우회 선택 (실수 시작):
   - slit_dist 입력 추가 (힌트)
   - hard mask로 BM=0 강제 (학습 포기)
4. Loss 구성 훼손:
   - L_BC 제거
   - L_H 가중치 0.01 (fine-tuning only)
5. 도메인 혼란:
   - z=[0, 40]으로 축소 (의도 불명)
6. 결과:
   - Slit/BM: hard mask로 구분 OK
   - 회절 학습: 실패
   - z 내부: 평면파 (uniform 0.799)
7. 후속 피해:
   - "회절 학습" 잘못 주장
   - 보고서/세미나 자료 오기재
```

**근본 원인**: 어려움 만나자 우회 선택 → 우회를 "해결"로 오인

**Phase C 방지책**:
- 정석 loss 구성 (v2 원본 계획 + Phase 1 성공 구조)
- PINN 도메인 명확 (BM2~OPD 40μm)
- 검증 기준 사전 정의

---

## 🎯 10개 계명

```
1. PINN이 물리를 학습한다. 외부로 강제하지 않는다.
2. L_Helmholtz는 주 학습 신호다. 가중치 1.0 이상.
3. L_BC는 필수다. BM=0은 loss로 학습.
4. 입력은 8D다. 힌트 추가 금지.
5. PINN 도메인은 z=[0, 40]μm. BM~OPD.
6. Hard mask 금지. slit_dist 입력 금지.
7. 결과 정규화로 맞추지 않는다. 학습을 고친다.
8. "일단"으로 시작하지 않는다. 처음부터 정석.
9. 우회 제안은 거부한다. 근본 원인을 찾는다.
10. z 내부 fringe 확인 없이 "성공" 선언 금지.
```

---

## 목차

1. 프로젝트 개요 및 북극성
2. UDFPS COE 스택 (Phase 1 구조 기반)
3. 파이프라인 구조 (TMM → ASM → PINN → PSF)
   - 3.5 모듈 간 인터페이스 규약
   - 3.6 Buffer Zone 전략
4. PINN 수학적 문제 정의
5. PINN 네트워크 구조 (Pure, 8D)
6. Loss 구성 (4가지)
7. 학습 전략 (Curriculum 3-Stage)
   - 7.5 초기화 전략 (Warm Start)
8. Collocation 샘플링
   - 8.3 BM Slit/BM 영역 직접 샘플링
9. 실행 환경 (집 CPU + 회사 GPU)
10. 시스템 아키텍처
11. 프로젝트 파일 구조
    - **11.5 하이브리드 구조 (v6 신규)**
12. AGENTS.md 도메인 규칙
13. 핵심 모듈 구현 명세
    - 13.8 Robust LightTools Runner
14. API 명세
15. 구현 순서 (Phase A~E)
    - **15.5 개발 워크플로우 (v6 신규)**
16. Design Studio UI
17. 타일링 및 지문 이미지 시뮬레이션
    - **17.5 노트북 템플릿 (v6 신규)**
18. 성공 기준 및 검증
    - 18.5 Red Flag 자동 감지
19. 리스크 및 Fallback
20. Phase B 자산 관리

---

## 1. 프로젝트 개요 및 북극성

### 1.1 북극성

```
PINN 기반의 역설계 자동화 플랫폼

- PINN: 순수 Physics-Informed Neural Network (힌트 주입 없음)
- 역설계: 목표 성능 → 설계변수 역추출
- 자동화: 사람 개입 없이 end-to-end
- 플랫폼: 재사용 가능한 시스템
```

### 1.2 플랫폼 워크플로우

```
사용자 목표 입력 (MTF@ridge, skewness, throughput)
        ↓
BoTorch qNEHVI 역설계 (8초)
        ↓ p 후보 제안
FNO Surrogate (0.8ms/추론)
        ↓ PSF 7개
PSFMetrics (MTF, skew, T)
        ↓ 목적 함수
BoTorch 다음 iteration
        ↓ 수렴
Pareto Front → Top-5 → UI
```

### 1.3 Phase 1 성공 기반 확장

이 플랫폼은 Phase 1(이미 검증된 성공 구조)의 확장입니다:

```
Phase 1 결과 (이미 달성):
- MTF@ridge 99.78% (목표 50% 초과)
- PSF skewness 0.075 (24개월 목표 조기 달성)
- Hypervolume 0.464
- Streamlit Cloud 배포 완료

Phase C (본 프로젝트) 확장:
- PINN 영역 확대: 30μm → 40μm (BM2~OPD)
- 설계변수 확대: 5D → 7D (AR 4층 + δ_BM1, δ_BM2, w1, w2)
- 파라메트릭 PINN (설계변수 입력에 포함)
- 3탭 Design Studio UI
- FastAPI 프로덕션 전환
```

### 1.4 핵심 가치 (LightTools 단독 대비)

- **역설계 자동화**: 목표 → 최적 BM 구조 자동 도출
- **위상 왜곡 학습**: LightTools 불가 → PINN 가능
- **BO 연결**: LightTools 833시간 → FNO+BoTorch 8초
- **Sim-to-Real**: L_I target 교체로 fine-tuning
- **크로스토크 정량화**: 7피치 도메인

---

## 2. UDFPS COE 스택 (Phase 1 구조 기반)

### 2.1 물리적 스택 구조

```
       ═══════════════════════════════ ← Finger (지문 접촉면)
       ┌─────────────────────────────┐
       │       AR Coating            │  ~300 nm (Gorilla DX 4층)
       │  SiO2/TiO2 다층 박막        │  SiO2(34.6) / TiO2(25.9) /
       │                             │  SiO2(20.7) / TiO2(169.5) nm
       │                             │  → TMM으로 t(θ), Δφ(θ) 계산
       ├─────────────────────────────┤  ← z ≈ 590.3 μm
       │                             │
       │       Cover Glass           │  550 μm, n=1.52
       │       (n = 1.52)            │  → ASM 파동 전파
       │                             │
       │                             │
       ├─────────────────────────────┤  ← z = 40 μm (PINN 입사 경계)
       │ (PINN 도메인 시작)           │
       │                             │
       │       BM2 (Black Matrix)    │  z = 40 μm (두께 0.1 무시)
       │                             │  아퍼처 w2, 오프셋 δ_BM2
       │                             │
       │       ILD (Interlayer)      │  z = 20~40 μm (d=20μm)
       │                             │  회절 전파 영역
       │                             │
       │       BM1 (Black Matrix)    │  z = 20 μm (두께 0.1 무시)
       │                             │  아퍼처 w1, 오프셋 δ_BM1
       │                             │
       │       Encapsulation         │  z = 0~20 μm (20μm)
       │                             │  최종 전파 영역
       │                             │
       │ (PINN 도메인 끝)             │
       ├─────────────────────────────┤  ← z = 0 μm (OPD 평면)
       │       Sensor (OPD)          │  Photodiode
       └─────────────────────────────┘
       ═══════════════════════════════

파이프라인:
  Finger → [TMM: AR] → [ASM: CG 550μm] → [PINN: 40μm] → [PSF]
```

### 2.2 레이어별 두께 및 역할

| 레이어 | 두께 | z 범위 | 처리 방법 | 비고 |
|---|---|---|---|---|
| AR Coating | ~300 nm | 590.0~590.3 | **TMM** | Gorilla DX 4층 (Phase 1 최적값) |
| Cover Glass | 550 μm | 40~590 | **ASM** | n=1.52, FFT 전파 |
| (PINN 경계) | - | z = 40 | ASM→PINN | BC-1 target |
| BM2 | 0.1 μm → 0 | 40 | **PINN (L_BC)** | 아퍼처 w2 |
| ILD | 20 μm | 20~40 | **PINN (L_H)** | 회절 전파 |
| BM1 | 0.1 μm → 0 | 20 | **PINN (L_BC)** | 아퍼처 w1 |
| Encap | 20 μm | 0~20 | **PINN (L_H)** | 최종 전파 |
| OPD | - | 0 | BC-4 | PSF 측정면 |

### 2.3 z 좌표 정의 (확정)

```
[OPD = 0 기준, 위로 z 증가]

z = 0        : OPD 평면 (센서)
z = 0~20     : Encap (20μm, 고정)
z = 20       : BM1 평면 (두께 무시)
z = 20~40    : ILD (d=20μm, 고정)
z = 40       : BM2 평면 (두께 무시) ← PINN 입사 경계
z = 40~590   : Cover Glass (550μm, ASM 영역)
z = 590~590.3 : AR 코팅 (~300nm, TMM 영역)
z > 590.3    : 지문 접촉면

PINN 도메인: z ∈ [0, 40] μm
ASM 전파 거리: 550 μm (CG)
Stack Height (타일링용): 590 μm
```

### 2.4 AR 코팅 (Gorilla DX 4층)

**Phase 1 최적값** (Phase C 초기 고정):

| 층 | 소재 | 굴절률 n | 두께 (nm) |
|---|---|---|---|
| 1 | SiO2 | 1.46 | 34.6 |
| 2 | TiO2 | 2.35 | 25.9 |
| 3 | SiO2 | 1.46 | 20.7 |
| 4 | TiO2 | 2.35 | 169.5 |
| **총합** | | | **250.7 nm** |

**Phase C에서는 AR 고정** (Phase 1 최적값 사용), **Phase D에서 d1~d4 최적화 포함** (7D → 8D 확장).

**TMM 출력**:
```
t(θ): 투과 진폭 (θ=0°에서 약 0.99)
Δφ(θ): 위상 지연 (θ=30°에서 약 -7.5°)
```

### 2.5 설계변수 (Phase C: 4개, μm)

```python
class BMDesignParams(BaseModel):
    # ── BM 구조 4개 (μm) ──
    delta_bm1: float   # BM1 오프셋 [-10, 10]
    delta_bm2: float   # BM2 오프셋 [-10, 10]
    w1:        float   # BM1 아퍼처 폭 [5, 20]
    w2:        float   # BM2 아퍼처 폭 [5, 20]
    
    # ── 고정값 ──
    d_ild:     float = 20.0    # BM1-BM2 ILD
    z_encap:   float = 20.0    # Encap 두께
    z_bm1:     float = 20.0    # BM1 z 좌표
    z_bm2:     float = 40.0    # BM2 z 좌표
    opd_pitch: float = 72.0    # OPD 피치
    opd_width: float = 10.0    # OPD 픽셀 폭
    cg_thick:  float = 550.0   # Cover Glass
    n_cg:      float = 1.52    # 굴절률
    
    # ── AR 고정 (Phase 1 최적) ──
    ar_d1: float = 34.6    # SiO2 (nm)
    ar_d2: float = 25.9    # TiO2 (nm)
    ar_d3: float = 20.7    # SiO2 (nm)
    ar_d4: float = 169.5   # TiO2 (nm)
```

**Phase D 확장**: ar_d1~ar_d4를 설계변수로 포함 (8D 최적화)

### 2.6 입사각 범위

```
물리적 상한: θ_crit = arcsin(1/n_CG) ≈ 41.1°
방향:        ±방향 모두 (지문 난반사)
TMM 범위:    -41.1° ~ +41.1°
BM 수용각:   θ_eff = ±arctan(w1 / d_ILD) = ±arctan(w1/20)
            w1=10 → θ_eff ≈ ±26.6°
            w1=20 → θ_eff ≈ ±45° (초과, 제한 필요)

지문 이미지 시뮬레이션 (타일링):
  각 픽셀 θ = arctan(distance / stack_height)
  stack_height = 590 μm (CG 550 + Encap 20 + BM 영역 20)
  센서 중심: θ ≈ 0° → PSF 대칭
  센서 가장자리: θ → 41° → PSF 비대칭
  θ > 41.1° → 전반사 (OPD 도달 불가)
```

---

## 3. 파이프라인 구조 (TMM → ASM → PINN → PSF)

### 3.1 Hybrid 파이프라인 (전체)

```
[설계변수 입력]
  AR: d1~d4 (Phase C 고정, Phase D 최적화)
  BM: δ_BM1, δ_BM2, w1, w2 (Phase C 최적화)
        ↓
┌──────────────────────────────────────┐
│  AR Coating (TMM)                    │
│                                      │
│  입력: d1~d4, θ, λ                   │
│  출력: t(θ), Δφ(θ) LUT               │
│  설계변수 의존: d1~d4                 │
│                                      │
│  data-free, 수식 계산                 │
│  Phase C: Phase 1 최적값 고정 사용    │
└──────────────┬───────────────────────┘
               │ 복소 진폭: t(θ)·exp(i·Δφ_TMM(θ))
               ▼
┌──────────────────────────────────────┐
│  Cover Glass 550μm (ASM)             │
│                                      │
│  입력: 평면파 + AR 위상               │
│  출력: z=40 위치 복소장 U_CG(x,θ)     │
│  설계변수 의존: 없음 (상수)           │
│                                      │
│  FFT 기반 자유공간 전파               │
│  전파 거리: 550μm                    │
└──────────────┬───────────────────────┘
               │ U_CG(x, z=40, θ)
               ▼
═══════════════════════════════════════  PINN 입사 경계 (z=40)
               │ ← L_phase target
               ▼
┌──────────────────────────────────────┐
│  PINN 도메인 z ∈ [0, 40]μm           │
│                                      │
│  ★ BM 회절 + 위상 왜곡 결합 학습 ★    │
│                                      │
│  입력: (x, z, δ₁, δ₂, w₁, w₂, sin θ, cos θ) │
│  출력: 복소 U(x, z)                   │
│                                      │
│  BM2 (z=40) ← L_BC: U=0 at BM         │
│  ILD 20μm   ← L_Helmholtz             │
│  BM1 (z=20) ← L_BC: U=0 at BM         │
│  Encap 20μm ← L_Helmholtz             │
│                                      │
│  설계변수 의존: δ₁, δ₂, w₁, w₂        │
└──────────────┬───────────────────────┘
               │ U(x, z=0): OPD 평면 복소장
               ▼
═══════════════════════════════════════  OPD 평면 (z=0)
               │
               ▼
┌──────────────────────────────────────┐
│  PSF 계산                             │
│                                      │
│  |U(x, 0)|² → OPD 픽셀 적분           │
│  → PSF 7개                            │
│                                      │
│  인덱스: 0 1 2 3 4 5 6                │
│         R V R V R V R                 │
│         (인덱스 3 = 중심)              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  메트릭                                │
│                                      │
│  MTF@ridge, skewness, throughput,    │
│  crosstalk_ratio                     │
└──────────────────────────────────────┘
```

### 3.2 각 모듈의 역할 (명확히)

#### TMM (AR Coating)
```
역할: 입사파의 위상 왜곡 계산
설계변수 의존: AR 층 두께 (d1~d4)
출력: Δφ(θ), t(θ) LUT
라이브러리: tmm 패키지
PINN과의 연결: AR 적용 후 ASM 전파의 시작점
두께: ~300 nm (Phase 1 최적값)
```

#### ASM (Cover Glass)
```
역할: 자유공간 파동 전파
설계변수 의존: 없음 (상수)
입력: AR 통과 후 z=590 위치 복소장
출력: z=40 위치 복소장 (550μm 전파)
구현: FFT 기반
PINN과의 연결: L_phase의 target (PINN 입사 경계)
```

#### PINN (BM~OPD, 40μm)
```
역할: BM 회절 + 위상 왜곡 결합 학습
설계변수 의존: δ_BM1, δ_BM2, w1, w2
도메인: x ∈ [0, 504], z ∈ [0, 40]
구성:
  z = 40       : BM2 (L_BC 적용)
  z = 20~40    : ILD (L_H 적용)
  z = 20       : BM1 (L_BC 적용)
  z = 0~20     : Encap (L_H 적용)
  z = 0        : OPD (출력)
학습: L_Helmholtz + L_phase + L_BC + L_I
```

#### PSF 계산
```
역할: 복소장 → PSF 7개
구현: OPD 픽셀 적분
|U(x, 0)|² at 각 OPD 위치 적분
```

### 3.3 왜 이 구조가 맞는가

**물리적 정당성**:
- AR 위상: 박막 이론 (매우 얇음, ~300nm) → TMM 정확
- CG 자유공간 전파: 설계변수(δ, w) 무관 → ASM 충분 (설계변수 변화에도 불변)
- BM 회절 + ILD 간섭 + 2차 회절 + Encap 전파: 모두 결합 → **PINN 필요**

**학습 가능성**:
- PINN 도메인 40μm = 파장(520nm)의 약 77배
- Phase 1 30μm 성공 → 40μm 확장 (비슷한 스케일)
- Full 570μm(1000배)보다 14배 현실적

**역설계 효율**:
- PINN이 δ, w → PSF 직접 매핑 학습
- FNO 증류 가능
- BoTorch 탐색 공간 = PINN 학습 공간

### 3.4 Phase 1과의 차이

| 항목 | Phase 1 | Phase C (v5) |
|---|---|---|
| PINN 도메인 | BM 근방 30μm | BM~OPD 40μm |
| 통합 파이프라인 | TMM+ASM+단순 BM | TMM+ASM+**PINN** |
| 설계변수 | 5D (AR 4 + δ_BM) | Phase C: 4D (BM), Phase D: 8D |
| PINN 입력 | (x, z) 2D | (x,z,설계4,각도2) 8D |
| PINN 역할 | 연구용 (별도 노트북) | **파이프라인 핵심** |
| UI | Streamlit 10탭 | Design Studio 3탭 |

### 3.5 모듈 간 인터페이스 규약 (단위/dtype 일관성)

**Phase B 실패 원인 중 하나**: 단위 혼동, complex tensor 처리 실수. 이 섹션으로 방지.

#### 3.5.1 단위 일관성 규약

```
좌표 (x, z):          μm (모든 모듈 통일)
AR 두께 (d1~d4):      nm (TMM 전용)
각도 (θ):             degrees (입출력), radians (내부 계산)
파장 (λ):             nm (520)
파수 (k):             μm⁻¹ (18.37)
복소수:               (Re, Im) 분리 또는 torch.complex64

❌ 금지:
  - 단위 혼용 (μm와 nm 섞기)
  - dtype 혼용 (float64 + float32)
  - degrees와 radians 암묵적 변환
  
✓ 규약:
  - 함수 입력 시점에 단위 명시 (docstring)
  - 내부에서 통일 단위 사용
  - 변환은 명시적 (degrees → radians)
```

#### 3.5.2 TMM → ASM 인터페이스

```python
# TMM 출력 표준
from dataclasses import dataclass
import math
import cmath

@dataclass
class TMMOutput:
    """TMM 계산 결과의 표준 형식."""
    theta_deg: float              # 입사각 (도)
    wavelength_nm: float          # 파장 (nm)
    t_amplitude: float            # 투과 진폭 |t|
    phase_shift_deg: float        # 위상 지연 Δφ (도)
    
    def to_complex(self) -> complex:
        """복소 투과 계수 t = |t|·exp(i·Δφ)"""
        phase_rad = math.radians(self.phase_shift_deg)
        return self.t_amplitude * cmath.exp(1j * phase_rad)

# ASM 입력 예시
def asm_initial_field(tmm_out: TMMOutput, x_array: np.ndarray, k0: float) -> np.ndarray:
    """AR 통과 후 평면파 (ASM 초기 조건)."""
    sin_th = math.sin(math.radians(tmm_out.theta_deg))
    phase_rad = math.radians(tmm_out.phase_shift_deg)
    
    # 복소 평면파
    U = tmm_out.t_amplitude * np.exp(1j * (phase_rad + k0 * x_array * sin_th))
    return U  # np.complex128
```

#### 3.5.3 ASM → PINN 경계조건 인터페이스

```python
# ASM LUT 저장 형식 (표준)
"""
저장 파일: data/asm_luts/incident_z40.npz
구조:
  - theta_values: (N_theta,) float32, 도, -41~+41 간격 1도
  - x_values:     (N_x,) float32, μm, 0~504 간격 0.123μm (4096 포인트)
  - U_re:         (N_theta, N_x) float32 (실수부)
  - U_im:         (N_theta, N_x) float32 (허수부)
  - metadata:     dict
    - z_inlet: 40.0
    - cg_thick: 550.0
    - wavelength_nm: 520.0
    - n_cg: 1.52
"""

class ASMIncidentLUT:
    """PINN L_phase target 조회."""
    
    def __init__(self, filepath="data/asm_luts/incident_z40.npz"):
        data = np.load(filepath)
        self.theta = data['theta_values']   # (N_theta,) float32
        self.x = data['x_values']            # (N_x,) float32
        self.U_re = data['U_re']             # (N_theta, N_x) float32
        self.U_im = data['U_im']             # (N_theta, N_x) float32
    
    def lookup(self, x_query, sin_theta_query):
        """
        Bilinear interpolation으로 임의 (x, θ)에서 복소장 조회.
        
        Args:
            x_query: (N,) torch.float32, μm
            sin_theta_query: (N,) torch.float32, [-sin(41.1°), +sin(41.1°)]
        
        Returns:
            U_re_target: (N,) torch.float32
            U_im_target: (N,) torch.float32
        """
        theta_deg = torch.asin(sin_theta_query) * 180 / math.pi
        
        # theta axis: index 계산
        theta_idx_float = (theta_deg - self.theta[0]) / (self.theta[1] - self.theta[0])
        theta_idx_low = torch.floor(theta_idx_float).long().clamp(0, len(self.theta) - 2)
        theta_idx_high = theta_idx_low + 1
        theta_frac = theta_idx_float - theta_idx_low.float()
        
        # x axis: index 계산
        x_idx_float = (x_query - self.x[0]) / (self.x[1] - self.x[0])
        x_idx_low = torch.floor(x_idx_float).long().clamp(0, len(self.x) - 2)
        x_idx_high = x_idx_low + 1
        x_frac = x_idx_float - x_idx_low.float()
        
        # Bilinear interpolation (LUT tensor로 변환 후)
        lut_re = torch.from_numpy(self.U_re)
        lut_im = torch.from_numpy(self.U_im)
        
        # 4 corners
        v00_re = lut_re[theta_idx_low, x_idx_low]
        v01_re = lut_re[theta_idx_low, x_idx_high]
        v10_re = lut_re[theta_idx_high, x_idx_low]
        v11_re = lut_re[theta_idx_high, x_idx_high]
        
        # Bilinear
        U_re = ((1-theta_frac)*(1-x_frac)*v00_re + (1-theta_frac)*x_frac*v01_re
                + theta_frac*(1-x_frac)*v10_re + theta_frac*x_frac*v11_re)
        
        # 동일하게 U_im
        v00_im = lut_im[theta_idx_low, x_idx_low]
        v01_im = lut_im[theta_idx_low, x_idx_high]
        v10_im = lut_im[theta_idx_high, x_idx_low]
        v11_im = lut_im[theta_idx_high, x_idx_high]
        
        U_im = ((1-theta_frac)*(1-x_frac)*v00_im + (1-theta_frac)*x_frac*v01_im
                + theta_frac*(1-x_frac)*v10_im + theta_frac*x_frac*v11_im)
        
        return U_re, U_im
```

#### 3.5.4 PINN → PSF 인터페이스

```python
def compute_psf_7_opd(model, params: BMDesignParams, theta_deg: float, 
                      device='cuda') -> np.ndarray:
    """
    PINN의 z=0 출력 → PSF 7개 OPD 픽셀.
    
    Args:
        model: PurePINN
        params: 설계변수
        theta_deg: 입사각 (도)
        device: 'cuda' or 'cpu'
    
    Returns:
        psf_7: (7,) np.ndarray, 각 OPD 픽셀 강도
    """
    model.eval()
    
    # 샘플링 밀도 (OPD 픽셀 내부 적분용)
    N_SAMPLES_PER_OPD = 100
    PITCH = 72.0  # μm
    OPD_WIDTH = 10.0  # μm
    
    psf_7 = np.zeros(7)
    
    with torch.no_grad():
        for opd_idx in range(7):
            center = opd_idx * PITCH + PITCH / 2
            x = torch.linspace(
                center - OPD_WIDTH/2,
                center + OPD_WIDTH/2,
                N_SAMPLES_PER_OPD,
                device=device
            )
            z = torch.zeros(N_SAMPLES_PER_OPD, device=device)
            
            # 설계변수 broadcast
            d1 = torch.full((N_SAMPLES_PER_OPD,), params.delta_bm1, device=device)
            d2 = torch.full((N_SAMPLES_PER_OPD,), params.delta_bm2, device=device)
            w1 = torch.full((N_SAMPLES_PER_OPD,), params.w1, device=device)
            w2 = torch.full((N_SAMPLES_PER_OPD,), params.w2, device=device)
            
            sin_th = torch.full((N_SAMPLES_PER_OPD,), math.sin(math.radians(theta_deg)), device=device)
            cos_th = torch.sqrt(1 - sin_th**2)
            
            coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
            
            U = model(coords)
            intensity = U[:, 0]**2 + U[:, 1]**2  # |U|²
            
            # OPD 픽셀 적분 (평균 × 폭)
            psf_7[opd_idx] = intensity.mean().item() * OPD_WIDTH
    
    return psf_7
```

#### 3.5.5 인터페이스 검증 테스트 (필수)

```python
# tests/test_interfaces.py

def test_tmm_asm_interface():
    """TMM 출력 → ASM 입력 복원 가능."""
    tmm = GorillaDXTMM()
    t_amp, dphi_deg = tmm.compute_complex_t(theta_deg=30.0, wl_nm=520.0)
    
    tmm_out = TMMOutput(
        theta_deg=30.0, wavelength_nm=520.0,
        t_amplitude=t_amp, phase_shift_deg=dphi_deg
    )
    complex_t = tmm_out.to_complex()
    
    # 복원
    amp_back = abs(complex_t)
    dphi_back_deg = math.degrees(cmath.phase(complex_t))
    
    assert abs(amp_back - t_amp) < 1e-6
    assert abs(dphi_back_deg - dphi_deg) < 1e-6
    print("✓ TMM → ASM 인터페이스 검증 통과")


def test_asm_pinn_interface():
    """ASM LUT → PINN collocation 호환."""
    lut = ASMIncidentLUT()
    
    x = torch.linspace(0, 504, 100)
    sin_th = torch.zeros(100)
    U_re, U_im = lut.lookup(x, sin_th)
    
    assert U_re.dtype == torch.float32
    assert U_re.shape == (100,)
    assert not torch.isnan(U_re).any()
    assert not torch.isnan(U_im).any()
    
    # θ=0 수직 입사는 진폭 근사 일정
    assert abs(U_re.std().item()) < 0.5
    print("✓ ASM → PINN 인터페이스 검증 통과")


def test_pinn_psf_interface():
    """PINN → PSF 7개 변환."""
    model = PurePINN(hidden_dim=64, num_layers=3, num_freqs=24)
    params = BMDesignParams(
        delta_bm1=0, delta_bm2=0, w1=10, w2=10
    )
    
    psf = compute_psf_7_opd(model, params, theta_deg=0, device='cpu')
    
    assert psf.shape == (7,)
    assert (psf >= 0).all()  # 비음수
    print("✓ PINN → PSF 인터페이스 검증 통과")
```

### 3.6 Buffer Zone 전략 (경계 효과 방지)

#### 3.6.1 문제

```
센서 가장자리에서:
  크로스토크 광이 도메인 밖으로 나감
  측면 경계(x=0, x=504)에서 인공적 반사/흡수
  → 7 픽셀 중 가장자리(index 0, 6) PSF 부정확
  → 타일링 시 인접 타일과 불연속
```

#### 3.6.2 해결: Buffer Zone

```
계산 도메인: x ∈ [-72, 576] μm  (9 피치, buffer 1피치씩)
  ↑ PINN은 이 영역에서 학습
  
신뢰 영역:   x ∈ [0, 504] μm  (7 피치, 중심)
  ↑ PSF 계산, Pareto 평가에 사용
  
Buffer 영역:
  좌측: x ∈ [-72, 0]
  우측: x ∈ [504, 576]
  ↑ 경계 효과 흡수, PSF 계산 시 무시
```

#### 3.6.3 구현

```python
# Collocation 샘플링 시 (섹션 8 확장)
def hierarchical_collocation_with_buffer(n_total, device):
    """Buffer zone 포함 샘플링."""
    # 90%: 신뢰 영역 [0, 504]
    # 10%: Buffer [-72, 0] ∪ [504, 576]
    
    n_core = int(n_total * 0.9)
    n_buffer = n_total - n_core
    
    # Core 샘플링 (기존 계층적)
    coords_core = _sample_core_region(n_core, device)
    
    # Buffer 샘플링
    n_left = n_buffer // 2
    n_right = n_buffer - n_left
    
    x_left = -72 + torch.rand(n_left, device=device) * 72    # [-72, 0]
    x_right = 504 + torch.rand(n_right, device=device) * 72  # [504, 576]
    
    # Buffer에서 z는 균일 샘플링
    z_buffer = torch.rand(n_buffer, device=device) * 40
    
    x_buffer = torch.cat([x_left, x_right])
    
    # 설계변수는 core와 동일하게
    # ... (coords_buffer 구성)
    
    return torch.cat([coords_core, coords_buffer], dim=0)


# 측면 경계 처리 (Absorbing BC)
def side_boundary_loss(model, n_samples, device):
    """x=-72, x=576 경계에서 흡수."""
    # Sommerfeld 방사 조건: ∂U/∂x - i·k·U = 0
    # 간단화: |U(x=경계)|² 최소화 (흡수)
    
    x_left = torch.full((n_samples,), -72.0, device=device)
    x_right = torch.full((n_samples,), 576.0, device=device)
    
    loss_left = _compute_absorbing_residual(model, x_left, device)
    loss_right = _compute_absorbing_residual(model, x_right, device)
    
    return (loss_left + loss_right) / 2


# PSF 계산 시 신뢰 영역만
def compute_psf_7_opd_core_only(model, params, theta_deg, device='cuda'):
    """
    신뢰 영역 x ∈ [0, 504]의 7개 OPD만 계산.
    Buffer 영역 결과는 무시.
    """
    # 섹션 3.5.4와 동일하지만 x ∈ [0, 504] 범위만
    # OPD 인덱스 0~6 (센서 중심 ±3 피치)
    ...
```

#### 3.6.4 학습 영향

```
PINN 네트워크 입력 정규화:
  x_norm = (x - 252) / 324   ← [-72, 576] 범위 대응
  또는
  x_norm = x / 504            ← 기존 유지, buffer도 같은 스케일

권장: x_norm = x / 504
  - Buffer 영역: x_norm ∈ [-0.143, 1.143]
  - 신뢰 영역: x_norm ∈ [0, 1]
  - 네트워크는 모든 범위 학습 가능

Collocation 분포:
  z=40 평면 L_phase: Buffer 제외 (신뢰 영역만 target 있음)
  z=20 BM1, z=40 BM2 L_BC: Buffer 포함 (슬릿/BM 패턴 주기적)
  PDE L_H: 전 영역 (buffer 포함)
```

---

## 4. PINN 수학적 문제 정의

### 4.1 지배 방정식

**Scalar Helmholtz**:
```
∇²U(x, z) + k²·U(x, z) = 0

k = k₀·n = (2π/λ)·n
λ = 520 nm
n = 1.52
k ≈ 18.37 μm⁻¹
```

복소장: U = U_re + i·U_im

### 4.2 도메인

```
x ∈ [0, 504] μm    (7 피치 = 72×7)
z ∈ [0, 40]  μm    (BM~OPD)

z 좌표 (Phase C 기준):
  z = 0    : OPD 평면
  z = 20   : BM1 평면
  z = 40   : BM2 평면 (= PINN 입사 경계)
```

### 4.3 경계조건

#### BC-1: 입사 경계 (z = 40μm, ASM 결과)

```
U(x, 40, θ, p) = U_ASM(x, 40, θ)

여기서:
  U_ASM = [AR(TMM) + CG 550μm 전파(ASM)]의 결과
  = ASM 결과가 PINN 입력으로 들어옴
  
AR 고정 (Phase C): d1~d4 = Phase 1 최적값
  t(θ): AR 투과 진폭 (θ=0에서 ~0.99)
  Δφ_TMM(θ): AR 위상 왜곡 (θ=30°에서 ~-7.5°)

ASM 전파: 550μm (CG 두께)
  평면파 + AR 위상 → z=40 위치 복소장

θ 범위: -41.1° ~ +41.1°
```

**구현 방식**:
```
학습 전 미리 계산:
  각 (θ, x) 조합에 대해 U_ASM LUT 생성
  data/asm_luts/incident_z40.npz 저장
  
학습 중:
  L_phase에서 이 LUT 조회하여 target으로 사용
```

#### BC-2: BM 경계 (z = 20, z = 40 BM 영역)

```
U(x, 20, p) = 0  where x ∉ BM1_slit(δ₁, w₁)
U(x, 40, p) = 0  where x ∉ BM2_slit(δ₂, w₂)

BM1 slit 영역 (z = 20):
  slit_i = [i·72 + 36 + δ₁ - w₁/2, i·72 + 36 + δ₁ + w₁/2]
  (i = 0, 1, ..., 6)

BM2 slit 영역 (z = 40):
  slit_i = [i·72 + 36 + δ₂ - w₂/2, i·72 + 36 + δ₂ + w₂/2]
```

**중요**: BM2 경계조건은 BC-1 (입사 경계)과 같은 z=40 평면에 있음.
- BM2 slit 내부: L_phase target 적용 (ASM 결과)
- BM2 slit 외부 (BM 영역): L_BC U=0 적용
- 두 조건이 동일 평면에서 작용

#### BC-3: 측면 경계 (x = 0, x = 504)

```
권장: Absorbing BC (흡수)
  ∂U/∂x - i·k·U = 0 at x=0, 504

대안: Periodic BC
  U(0, z) = U(504, z)
  (구현 단순, 무한 주기 가정)

구현은 한 가지만 선택 (일관성)
```

#### BC-4: 출사 경계 (z = 0, OPD 면)

```
옵션 A: 방사 조건
  ∂U/∂z - i·k·U = 0 at z=0

옵션 B: L_I로 매칭
  |U(x, 0)|² = I_target(x, p, θ)
  target = LT 측정 또는 ASM 근사

Phase C 초기: 옵션 A (흡수 경계)
LT 확보 후: 옵션 B 추가 (L_I)
```

### 4.4 굴절률 분포

```
n(x, z) = 1.52 (모든 z, 모든 x)

BM은 별도 매질이 아닌 "경계":
  L_BC로 U=0 강제 (z=20, z=40 BM 영역)
```

### 4.5 물리적 해석

```
L_Helmholtz:
  PINN 도메인 z ∈ [0, 40] 모든 점에서 PDE 만족
  ILD (z=20~40), Encap (z=0~20)에서 자유 전파
  BM 경계도 포함되지만 L_BC가 지배

L_phase:
  z=40에서 ASM 결과와 일치
  AR(TMM) + CG(ASM) 결과를 PINN이 이어받음
  BM2 slit 내부만 적용 (BM 외부는 L_BC)

L_BC:
  BM1 (z=20), BM2 (z=40) 불투명 영역 U=0
  Hard mask 없이 loss로 학습

L_I (optional):
  z=0 OPD 평면에서 측정/ASM과 매칭
  Sim-to-real 보정
```

---

## 5. PINN 네트워크 구조 (Pure, 8D)

### 5.1 입력 (8D)

```
(x, z, δ₁, δ₂, w₁, w₂, sin θ, cos θ)

정규화:
  x_norm = x / 504
  z_norm = z / 40        ← PINN 도메인 z=[0, 40]
  δ_norm = δ / 10
  w_norm = (w - 12.5) / 7.5
  sin, cos: [-1, 1]

힌트 없음:
  ✗ slit_dist 입력 금지
  ✗ pre-computed feature 금지
```

### 5.2 네트워크 아키텍처

```
Input (8D)
  ↓
Fourier Feature Embedding (48 frequencies)
  2·48 = 96 features
  ↓
SIREN Layer 1 (Linear 96→128, sin(ω₀·x), ω₀=30)
  ↓
SIREN Layer 2 (Linear 128→128, sin)
  ↓
SIREN Layer 3 (Linear 128→128, sin)
  ↓
SIREN Layer 4 (Linear 128→128, sin)
  ↓
Output Linear 128→2
  ↓
U = (U_re, U_im)

총 파라미터: ~100K
```

### 5.3 Phase 1 대비 변경

```
Phase 1 (30μm):
  입력 2D (x, z)
  SIREN 4층 × 128
  파라미터 50K
  도메인: BM 근방 30μm

Phase C (40μm):
  입력 8D (x, z + 설계변수 4개 + 각도 2개)
  SIREN 4층 × 128
  Fourier embedding 추가 (주기 구조 학습)
  파라미터 ~100K (입력 차원 증가)
  도메인: BM~OPD 40μm (확장)
```

---

## 6. Loss 구성 (4가지)

### 6.1 총 손실

```
L_total = λ_H · L_Helmholtz + λ_phase · L_phase + λ_BC · L_BC + λ_I · L_I

λ_H     = 1.0   ← 주 학습 신호 (PDE)
λ_phase = 0.5   ← 입사 BC (z=40, ASM 결과)
λ_BC    = 0.5   ← BM BC (z=20, z=40의 BM 영역)
λ_I     = 0.3   ← 측정 (optional, z=0)

비중:
  L_H: ~44% (주 신호)
  L_phase: ~22%
  L_BC: ~22%
  L_I: ~13%

규칙:
  λ_H >= max(λ_phase, λ_BC, λ_I)
```

### 6.2 L_Helmholtz 구현

```python
def helmholtz_loss(model, coords):
    """∇²U + k²U = 0"""
    coords = coords.detach().requires_grad_(True)
    
    U = model(coords)
    U_re = U[:, 0:1]
    U_im = U[:, 1:2]
    
    # First derivatives
    grads_re = torch.autograd.grad(U_re.sum(), coords, create_graph=True)[0]
    grads_im = torch.autograd.grad(U_im.sum(), coords, create_graph=True)[0]
    
    U_re_x = grads_re[:, 0:1]
    U_re_z = grads_re[:, 1:2]
    U_im_x = grads_im[:, 0:1]
    U_im_z = grads_im[:, 1:2]
    
    # Second derivatives
    U_re_xx = torch.autograd.grad(U_re_x.sum(), coords, create_graph=True)[0][:, 0:1]
    U_re_zz = torch.autograd.grad(U_re_z.sum(), coords, create_graph=True)[0][:, 1:2]
    U_im_xx = torch.autograd.grad(U_im_x.sum(), coords, create_graph=True)[0][:, 0:1]
    U_im_zz = torch.autograd.grad(U_im_z.sum(), coords, create_graph=True)[0][:, 1:2]
    
    k = 18.37
    res_re = U_re_xx + U_re_zz + k**2 * U_re
    res_im = U_im_xx + U_im_zz + k**2 * U_im
    
    return torch.mean(res_re**2 + res_im**2)
```

### 6.3 L_phase 구현 (ASM 결과 매칭)

```python
def phase_loss(model, asm_lut, n_samples, device):
    """z=40에서 ASM 결과와 매칭 (BM2 slit 내부만)"""
    x = torch.rand(n_samples, device=device) * 504
    z = torch.full((n_samples,), 40.0, device=device)  # ← z=40
    
    # 설계변수 (BM만, AR 고정)
    d1 = torch.rand(n_samples, device=device) * 20 - 10
    d2 = torch.rand(n_samples, device=device) * 20 - 10
    w1 = torch.rand(n_samples, device=device) * 15 + 5
    w2 = torch.rand(n_samples, device=device) * 15 + 5
    
    # 입사각
    sin_max = math.sin(math.radians(41.1))
    sin_th = torch.rand(n_samples, device=device) * 2 * sin_max - sin_max
    cos_th = torch.sqrt(1 - sin_th**2)
    
    # BM2 slit 내부만 필터 (BM 외부는 L_BC가 담당)
    is_slit = ~compute_is_bm(x, d2, w2)  # BM2 slit 내부
    
    if is_slit.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    x_slit = x[is_slit]
    z_slit = z[is_slit]
    d1_slit = d1[is_slit]
    d2_slit = d2[is_slit]
    w1_slit = w1[is_slit]
    w2_slit = w2[is_slit]
    sin_slit = sin_th[is_slit]
    cos_slit = cos_th[is_slit]
    
    coords = torch.stack([
        x_slit, z_slit, d1_slit, d2_slit,
        w1_slit, w2_slit, sin_slit, cos_slit
    ], dim=1)
    
    # ASM LUT 조회 (z=40 위치 복소장)
    U_asm_re, U_asm_im = asm_lut.lookup(x_slit, sin_slit)
    
    # PINN 출력
    U = model(coords)
    
    return torch.mean((U[:, 0] - U_asm_re)**2 + (U[:, 1] - U_asm_im)**2)
```

### 6.4 L_BC 구현 (BM1, BM2)

```python
def bm_boundary_loss(model, n_samples, device):
    """BM1 (z=20), BM2 (z=40) 영역에서 U=0"""
    
    def compute_bm_loss(z_val, delta_idx, w_idx):
        x = torch.rand(n_samples, device=device) * 504
        z = torch.full((n_samples,), z_val, device=device)
        d1 = torch.rand(n_samples, device=device) * 20 - 10
        d2 = torch.rand(n_samples, device=device) * 20 - 10
        w1 = torch.rand(n_samples, device=device) * 15 + 5
        w2 = torch.rand(n_samples, device=device) * 15 + 5
        
        sin_max = math.sin(math.radians(41.1))
        sin_th = torch.rand(n_samples, device=device) * 2 * sin_max - sin_max
        cos_th = torch.sqrt(1 - sin_th**2)
        
        delta = d1 if delta_idx == 1 else d2
        w = w1 if w_idx == 1 else w2
        
        is_bm = compute_is_bm(x, delta, w)
        
        coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
        U = model(coords)
        
        if is_bm.sum() > 0:
            return torch.mean(U[is_bm, 0]**2 + U[is_bm, 1]**2)
        return torch.tensor(0.0, device=device)
    
    loss_bm1 = compute_bm_loss(20.0, 1, 1)  # BM1 at z=20
    loss_bm2 = compute_bm_loss(40.0, 2, 2)  # BM2 at z=40
    
    return (loss_bm1 + loss_bm2) / 2


def compute_is_bm(x, delta, w):
    """x가 BM 영역(slit 바깥)인지"""
    pitch = 72.0
    pitch_idx = torch.floor(x / pitch).clamp(0, 6)
    center = pitch_idx * pitch + pitch/2 + delta
    dist = torch.abs(x - center)
    is_slit = dist < w/2
    return ~is_slit
```

### 6.5 L_I 구현 (optional)

```python
def intensity_loss(model, target_source, n_samples, device):
    """z=0에서 |U|² = target"""
    x = torch.rand(n_samples, device=device) * 504
    z = torch.zeros(n_samples, device=device)
    
    # 설계변수 (dataset에서)
    d1, d2, w1, w2, sin_th, cos_th = target_source.sample(n_samples)
    
    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
    U = model(coords)
    intensity = U[:, 0]**2 + U[:, 1]**2
    
    I_target = target_source.get_target(x, d1, d2, w1, w2, sin_th, cos_th)
    
    return torch.mean((intensity - I_target)**2)
```

**target_source**:
- Phase C-1: ASM 근사 (자유공간 전파, BM 효과 제외)
- Phase C-2: LightTools (실측)
- Phase E: 실 센서 데이터

---

## 7. 학습 전략 (Curriculum 3-Stage)

### 7.1 Stage 1: 경계조건 학습 (0 ~ 20% epochs)

```
목표: 경계 먼저 맞추기
Loss: L_phase + L_BC

예상:
  - z=40 BM2 slit 내부에서 U = U_ASM (입사파 정확히 재현)
  - z=20 (BM1), z=40 (BM2) BM 영역에서 U ≈ 0
  - 내부(ILD, Encap)는 아직 undefined
```

### 7.2 Stage 2: PDE 활성화 (20% ~ 60% epochs)

```
목표: 내부 파동 전파 학습
Loss: L_phase + L_BC + λ_H_ramp · L_Helmholtz
  λ_H_ramp: 0.1 → 1.0 (선형 증가)

예상:
  - ILD (z=20~40) 회절 fringe 발생
  - Encap (z=0~20) 전파 학습
  - 경계조건 유지
```

### 7.3 Stage 3: 데이터 매칭 (60% ~ 100%)

```
목표: 센서 면 정확도 향상
Loss: 전체 + L_I
  LT 데이터 확보 시
  없으면 Stage 2 연장
```

### 7.4 Optimizer

```python
# 처음 70%: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs_adam, eta_min=1e-5)

# 마지막 30%: L-BFGS (Phase 1 성공 경험)
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0, max_iter=20,
    line_search_fn='strong_wolfe'
)
```

### 7.5 초기화 전략 (Warm Start)

#### 7.5.1 Cold Start (기본, 안전)

```
랜덤 초기화 (SIREN 권장 분포)
Curriculum 3-Stage 순차 학습
시간: GPU 4-8시간
성공률: 중-높

장점:
  - 편향 없음
  - 표준 방식
단점:
  - 초기 수렴 느림
  - BM 경계 학습 어려움 (초기 예측이 랜덤)
```

#### 7.5.2 Warm Start with ASM (권장)

**아이디어**: PINN의 초기 weights를 "ASM 근사 → PINN 출력" 학습으로 시작.

```python
def warm_start_with_asm(model, asm_propagator, epochs=500, device='cuda'):
    """
    Stage 0: ASM 근사로 PINN weights 초기화.
    
    목적: PINN이 "대략적인 파동 전파" 먼저 학습
    그 후 Stage 1~3에서 BC + PDE 정밀화
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # 전체 도메인에서 샘플링 (BM 효과 없이 자유 전파)
        coords = hierarchical_collocation(n_total=2000, device=device)
        
        # ASM 근사 target (BM 무시, 자유공간 전파)
        # z=40 입사 → z 값에 따라 propagate
        x = coords[:, 0]
        z = coords[:, 1]
        sin_th = coords[:, 6]
        
        # 각 (x, z) 위치에서 ASM 예측
        U_asm_re, U_asm_im = asm_propagator.predict_at(x, z, sin_th)
        
        # PINN 출력
        U_pinn = model(coords)
        
        # MSE loss (복소 장)
        loss = torch.mean((U_pinn[:, 0] - U_asm_re)**2 + 
                         (U_pinn[:, 1] - U_asm_im)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Warm-start epoch {epoch}: loss={loss.item():.6f}")
    
    print("Warm start 완료. Stage 1~3 정석 학습으로 진행.")
    return model
```

**효과**:
```
Cold Start: 8시간
Warm Start + Stage 1~3: 4-6시간 (25~40% 단축)

수렴 안정성:
  Cold: Stage 1에서 경계 학습 실패 시 Stage 2로 못 감
  Warm: 경계가 이미 대략 맞음 → Stage 1 빠르게 완료

주의:
  - Warm start 너무 오래 하면 BM 효과 학습 방해
  - 500 epoch 내외가 적절
  - 그 후 바로 정석 Curriculum
```

#### 7.5.3 Phase 1 Checkpoint 활용 (최고 효율, 조건부)

```python
def transfer_from_phase1(model, phase1_checkpoint_path):
    """
    Phase 1 (30μm, 2D 입력) 체크포인트에서 Transfer.
    
    주의: 입력 차원이 다름 (2D → 8D)
    전체 weights 복사 불가, hidden layers만 부분 복사.
    """
    # Phase 1 모델 로드
    phase1_state = torch.load(phase1_checkpoint_path)
    
    # Hidden layers만 복사 (입력 레이어 제외)
    current_state = model.state_dict()
    
    for key in current_state:
        if 'hidden_layers' in key and key in phase1_state:
            if current_state[key].shape == phase1_state[key].shape:
                current_state[key] = phase1_state[key]
    
    model.load_state_dict(current_state)
    print("Phase 1 hidden layers 복사 완료. 입력/출력 레이어는 랜덤.")
    return model
```

**사용 시점**:
- Phase 1 노트북 03의 체크포인트가 있을 때만
- 없으면 skip (Warm Start만으로 충분)

#### 7.5.4 전략 선택 기준

```
상황 A: 시간 충분 (8시간 이상)
  → Cold Start
  → 편향 없는 표준 학습

상황 B: 시간 제약 (4-6시간)
  → Warm Start with ASM
  → 빠른 수렴

상황 C: Phase 1 체크포인트 있음
  → Transfer + Warm Start + Stage 1~3
  → 최대 효율

Phase C 권장: 상황 B (Warm Start)
  이유: 4월 30일 데모 시간 제약
```

---

## 8. Collocation 샘플링

### 8.1 계층적 분포

```
전체 N개 중 z 축 분포:

- 15% : z ∈ [39, 40]  (입사 경계, BC-1 + BM2)
- 25% : z ∈ [21, 39]  (ILD, 전파)
- 15% : z ∈ [19, 21]  (BM1 근처)
- 25% : z ∈ [1, 19]   (Encap, 전파)
- 10% : z ∈ [0, 1]    (OPD 경계, BC-4)
- 10% : 나머지        (buffer)

x: uniform [0, 504]
δ₁, δ₂: uniform [-10, 10]
w₁, w₂: uniform [5, 20]
θ: uniform [-41.1°, +41.1°]

집중 영역:
  z=40 (15%): 입사 + BM2 경계 (가장 중요)
  z=20 (15%): BM1 경계
  z=0 (10%): OPD 출사
  나머지 60%: 자유 전파 영역
```

### 8.2 샘플링 코드

```python
def hierarchical_collocation(n_total, device):
    """
    PINN 도메인 z ∈ [0, 40] 계층적 샘플링
    """
    counts = {
        'inlet_bm2':   int(n_total * 0.15),   # z=39~40 (입사+BM2)
        'ild':         int(n_total * 0.25),   # z=21~39 (ILD)
        'bm1':         int(n_total * 0.15),   # z=19~21 (BM1)
        'encap':       int(n_total * 0.25),   # z=1~19 (Encap)
        'outlet':      int(n_total * 0.10),   # z=0~1 (OPD)
        'buffer':      int(n_total * 0.10),   # 나머지
    }
    
    all_x, all_z = [], []
    
    # Inlet + BM2 (z=39~40)
    all_x.append(torch.rand(counts['inlet_bm2'], device=device) * 504)
    all_z.append(39 + torch.rand(counts['inlet_bm2'], device=device))
    
    # ILD (z=21~39)
    all_x.append(torch.rand(counts['ild'], device=device) * 504)
    all_z.append(21 + torch.rand(counts['ild'], device=device) * 18)
    
    # BM1 (z=19~21)
    all_x.append(torch.rand(counts['bm1'], device=device) * 504)
    all_z.append(19 + torch.rand(counts['bm1'], device=device) * 2)
    
    # Encap (z=1~19)
    all_x.append(torch.rand(counts['encap'], device=device) * 504)
    all_z.append(1 + torch.rand(counts['encap'], device=device) * 18)
    
    # Outlet (z=0~1)
    all_x.append(torch.rand(counts['outlet'], device=device) * 504)
    all_z.append(torch.rand(counts['outlet'], device=device))
    
    # Buffer (uniform)
    all_x.append(torch.rand(counts['buffer'], device=device) * 504)
    all_z.append(torch.rand(counts['buffer'], device=device) * 40)
    
    x = torch.cat(all_x)
    z = torch.cat(all_z)
    
    n = x.shape[0]
    d1 = torch.rand(n, device=device) * 20 - 10
    d2 = torch.rand(n, device=device) * 20 - 10
    w1 = torch.rand(n, device=device) * 15 + 5
    w2 = torch.rand(n, device=device) * 15 + 5
    
    sin_max = math.sin(math.radians(41.1))
    sin_th = torch.rand(n, device=device) * 2 * sin_max - sin_max
    cos_th = torch.sqrt(1 - sin_th**2)
    
    return torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
```

### 8.3 BM Slit/BM 영역 직접 샘플링 (효율 향상)

#### 8.3.1 문제: 균일 샘플링의 비효율

```python
# 균일 샘플링 방식 (비효율)
x = torch.rand(n, device=device) * 504  # [0, 504] 균일
is_slit = ~compute_is_bm(x, delta, w)
x_slit = x[is_slit]

# 예: w=10μm, pitch=72μm
# slit 비율 = 10/72 ≈ 14%
# 1000개 샘플링 → 약 140개만 slit 내부
# 860개는 L_phase 계산 무관 (낭비)
```

#### 8.3.2 해결: 직접 샘플링

```python
def sample_bm2_slit_direct(n_samples, delta_bm2, w2, device):
    """
    BM2 slit 내부만 직접 샘플링 (L_phase용).
    
    7개 slit 중심: x = i*72 + 36 + δ_BM2 (i=0..6)
    각 slit 범위: center ± w2/2
    """
    # 7개 slit에 균등 분배
    n_per_slit = n_samples // 7
    remainder = n_samples - n_per_slit * 7
    
    x_list = []
    for i in range(7):
        center = i * 72 + 36 + delta_bm2
        count = n_per_slit + (1 if i < remainder else 0)
        x_slit = center - w2/2 + torch.rand(count, device=device) * w2
        x_list.append(x_slit)
    
    return torch.cat(x_list)


def sample_bm_region_direct(n_samples, delta, w, device):
    """
    BM 영역 (slit 외부) 직접 샘플링 (L_BC용).
    
    BM 영역: 각 pitch에서 slit 제외 부분
    pitch 범위 [i*72, (i+1)*72]
    slit 범위 [center - w/2, center + w/2]
    BM 범위 1: [i*72, center - w/2]
    BM 범위 2: [center + w/2, (i+1)*72]
    """
    n_per_pitch = n_samples // 7
    remainder = n_samples - n_per_pitch * 7
    
    x_list = []
    for i in range(7):
        center = i * 72 + 36 + delta
        pitch_start = i * 72
        pitch_end = (i + 1) * 72
        
        bm_left_start = pitch_start
        bm_left_end = center - w/2
        bm_right_start = center + w/2
        bm_right_end = pitch_end
        
        bm_left_width = max(0, bm_left_end - bm_left_start)
        bm_right_width = max(0, bm_right_end - bm_right_start)
        total_bm_width = bm_left_width + bm_right_width
        
        count = n_per_pitch + (1 if i < remainder else 0)
        
        if total_bm_width > 0:
            # 좌우 BM에 비례 분배
            n_left = int(count * bm_left_width / total_bm_width)
            n_right = count - n_left
            
            if n_left > 0 and bm_left_width > 0:
                x_left = bm_left_start + torch.rand(n_left, device=device) * bm_left_width
                x_list.append(x_left)
            if n_right > 0 and bm_right_width > 0:
                x_right = bm_right_start + torch.rand(n_right, device=device) * bm_right_width
                x_list.append(x_right)
    
    if len(x_list) > 0:
        return torch.cat(x_list)
    return torch.empty(0, device=device)
```

#### 8.3.3 개선된 Loss 함수

```python
def phase_loss_direct(model, asm_lut, n_samples, device):
    """BM2 slit 내부 직접 샘플링 L_phase."""
    # 설계변수 랜덤 선택
    d1 = torch.rand(n_samples, device=device) * 20 - 10
    d2 = torch.rand(n_samples, device=device) * 20 - 10
    w1 = torch.rand(n_samples, device=device) * 15 + 5
    w2 = torch.rand(n_samples, device=device) * 15 + 5
    
    # Slit 직접 샘플링 (batch에서 각 샘플마다 다른 δ, w)
    # 간단화: 같은 batch의 δ, w로 공통 slit
    x = sample_bm2_slit_direct(n_samples, d2[0].item(), w2[0].item(), device)
    z = torch.full((n_samples,), 40.0, device=device)
    
    sin_max = math.sin(math.radians(41.1))
    sin_th = torch.rand(n_samples, device=device) * 2 * sin_max - sin_max
    cos_th = torch.sqrt(1 - sin_th**2)
    
    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
    
    U_asm_re, U_asm_im = asm_lut.lookup(x, sin_th)
    U = model(coords)
    
    return torch.mean((U[:, 0] - U_asm_re)**2 + (U[:, 1] - U_asm_im)**2)


def bm_boundary_loss_direct(model, n_samples, device):
    """BM1, BM2 BM 영역 직접 샘플링 L_BC."""
    d1 = torch.rand(n_samples, device=device) * 20 - 10
    d2 = torch.rand(n_samples, device=device) * 20 - 10
    w1 = torch.rand(n_samples, device=device) * 15 + 5
    w2 = torch.rand(n_samples, device=device) * 15 + 5
    
    sin_max = math.sin(math.radians(41.1))
    sin_th = torch.rand(n_samples, device=device) * 2 * sin_max - sin_max
    cos_th = torch.sqrt(1 - sin_th**2)
    
    # BM1 영역 (z=20)
    x_bm1 = sample_bm_region_direct(n_samples, d1[0].item(), w1[0].item(), device)
    z_bm1 = torch.full_like(x_bm1, 20.0)
    
    coords_bm1 = torch.stack([
        x_bm1, z_bm1, d1[:len(x_bm1)], d2[:len(x_bm1)],
        w1[:len(x_bm1)], w2[:len(x_bm1)],
        sin_th[:len(x_bm1)], cos_th[:len(x_bm1)]
    ], dim=1)
    U_bm1 = model(coords_bm1)
    loss_bm1 = torch.mean(U_bm1[:, 0]**2 + U_bm1[:, 1]**2)
    
    # BM2 영역 (z=40)
    x_bm2 = sample_bm_region_direct(n_samples, d2[0].item(), w2[0].item(), device)
    z_bm2 = torch.full_like(x_bm2, 40.0)
    
    coords_bm2 = torch.stack([
        x_bm2, z_bm2, d1[:len(x_bm2)], d2[:len(x_bm2)],
        w1[:len(x_bm2)], w2[:len(x_bm2)],
        sin_th[:len(x_bm2)], cos_th[:len(x_bm2)]
    ], dim=1)
    U_bm2 = model(coords_bm2)
    loss_bm2 = torch.mean(U_bm2[:, 0]**2 + U_bm2[:, 1]**2)
    
    return (loss_bm1 + loss_bm2) / 2
```

#### 8.3.4 효율 개선

```
균일 샘플링 vs 직접 샘플링:

L_phase:
  균일: 1000 samples → ~140 slit 샘플 (14% 활용)
  직접: 1000 samples → 1000 slit 샘플 (100% 활용)
  → 7배 효율 향상

L_BC:
  균일: 1000 samples → ~860 BM 샘플 (86% 활용)
  직접: 1000 samples → 1000 BM 샘플 (100% 활용)
  → 1.16배 효율 향상 (이미 대부분 BM)

실제 학습 시간:
  균일: 8시간
  직접: 5-6시간 (약 25% 단축)
```

#### 8.3.5 주의사항

```
설계변수 일괄 처리:
  한 batch 내에서 같은 δ, w로 sample.
  → 각 batch마다 다른 설계 조건
  
Batch 다양성:
  Outer loop: 여러 batch로 설계 공간 커버
  Inner loop (1 batch): 공간 샘플 다양성

Gradient 흐름:
  δ, w가 x 샘플링 위치에 영향 (non-differentiable)
  하지만 loss 계산은 x, z, δ, w 모두에 differentiable
  → 역전파 정상 작동
```

---

## 9. 실행 환경 (집 CPU + 회사 GPU)

### 9.1 이중 실행 전략

**동일한 코드, config만 다름**:

```bash
# 집 (CPU) - 구조 검증
python train_phase_c.py --config small --device cpu

# 회사 (GPU) - 본격 학습
python train_phase_c.py --config full --device cuda
```

### 9.2 Small config (CPU, 30분~1시간)

```python
CONFIG_SMALL = {
    'hidden_dim':    64,
    'num_layers':    3,
    'num_freqs':     24,
    'n_colloc':      2000,
    'n_phase':       200,
    'n_bc':          500,
    'epochs':        500,
    'stage1_end':    100,
    'stage2_end':    400,
    'stage3_end':    500,
    'lr':            1e-3,
    'device':        'cpu',
}
```

**검증 목표**:
- Loss 발산 없이 감소
- Slit/BM 분화 조짐
- z 내부 변화 (uniform 아님)
- NaN/Inf 없음

### 9.3 Full config (GPU, 4-8시간)

```python
CONFIG_FULL = {
    'hidden_dim':    128,
    'num_layers':    4,
    'num_freqs':     48,
    'n_colloc':      30000,
    'n_phase':       500,
    'n_bc':          2000,
    'epochs':        10000,
    'stage1_end':    2000,
    'stage2_end':    6000,
    'stage3_end':    10000,
    'lr':            1e-3,
    'device':        'cuda',
}
```

**본격 학습 목표**:
- BM 영역 |U| < 0.05 (hard mask 없이)
- Slit 영역 |U| ≈ t(θ) × ASM 결과 (±10%)
- z 내부 회절 fringe 명확
- 설계변수 반응

### 9.4 실행 흐름

```
[집, 새벽]
  1. train_phase_c.py 작성
  2. python train_phase_c.py --config small --device cpu
  3. 결과 확인 → 구조 OK면 commit + push
  
[회사, 아침]
  4. git pull
  5. python train_phase_c.py --config full --device cuda
  6. 백그라운드 실행 (4-8시간)
  7. LightTools 수집 병행
  
[회사, 저녁]
  8. 학습 완료 확인
  9. 검증 스크립트
  10. phase_c_final.pt 확보
```

---

## 10. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    UI Layer (Frontend)                    │
│  Summary / Detail / Explore (3 tabs)                      │
│  Zustand + API client                                     │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP REST
┌────────────────────┴─────────────────────────────────────┐
│                   API Layer (FastAPI)                     │
│  design | inference | candidates | training               │
└──────────┬──────────────────────────────────┬────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────┐              ┌──────────────────────┐
│   Harness Layer   │              │   Engine Layer       │
│   ├─ Validator    │              │   ├─ PINN (Pure)     │
│   ├─ AGENTS.md    │              │   ├─ FNO Surrogate   │
│   └─ Drift Det.   │              │   ├─ BoTorch qNEHVI  │
└──────────────────┘              │   └─ UQ Filter       │
                                   └──────────┬───────────┘
                                              │
                        ┌─────────────────────┼─────────────────┐
                        ▼                     ▼                 ▼
                ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
                │ Physics      │    │ Data          │   │ Flywheel     │
                │ ├─ TMM       │    │ ├─ LHS        │   │ └─ Active    │
                │ ├─ ASM       │    │ ├─ LT Runner  │   │    Learning  │
                │ ├─ Helmholtz │    │ └─ Dataset    │   │              │
                │ └─ Metrics   │    └──────────────┘   └──────────────┘
                └──────────────┘
```

---

## 11. 프로젝트 파일 구조

```
udfps-pinn-platform/
│
├── notebooks/                          ★ v6 신규: Jupyter 탐색/실험
│   ├── 01_exploration/                 # Phase 1 계승 탐색
│   │   ├── 01_tmm_exploration.ipynb
│   │   ├── 02_asm_exploration.ipynb
│   │   └── 03_phase_b_analysis.ipynb
│   │
│   ├── 02_phase_c_development/         # Phase C 핵심 (이번 목표)
│   │   ├── 01_asm_lut_generation.ipynb       # ★ 여기부터 시작
│   │   ├── 02_pinn_cpu_validation.ipynb
│   │   ├── 03_pinn_training_monitor.ipynb
│   │   ├── 04_pinn_evaluation.ipynb
│   │   ├── 05_red_flag_detection.ipynb
│   │   └── 06_phase_c_report.ipynb           # 경영진 보고서
│   │
│   ├── 03_phase_d_fno/                 # Phase D FNO + 역설계
│   │   ├── 01_fno_distillation.ipynb
│   │   ├── 02_fno_validation.ipynb
│   │   └── 03_botorch_optimization.ipynb
│   │
│   ├── 04_phase_e_integration/         # Phase E 통합 테스트
│   │   ├── 01_e2e_pipeline.ipynb
│   │   └── 02_fingerprint_simulation.ipynb
│   │
│   └── helpers/                        # 공통 시각화 유틸
│       ├── visualization.py
│       └── notebook_utils.py
│
├── backend/                            ★ 재사용 모듈 (확정된 로직)
│   ├── core/
│   │   ├── pinn_model.py               # Pure PINN (8D, SIREN+Fourier)
│   │   ├── fno_model.py                # FNO Surrogate
│   │   ├── botorch_optimizer.py        # qNEHVI 역설계
│   │   ├── uq_filter.py                # MC Dropout UQ
│   │   └── active_learning.py
│   │
│   ├── physics/
│   │   ├── tmm_calculator.py           # AR TMM
│   │   ├── asm_propagator.py           # CG ASM
│   │   ├── asm_lut_generator.py        # ★ z=40 입사 경계 LUT 로직
│   │   ├── psf_metrics.py
│   │   └── boundary_conditions.py
│   │
│   ├── training/
│   │   ├── loss_functions.py
│   │   ├── collocation_sampler.py
│   │   ├── curriculum.py
│   │   └── red_flag_detector.py        # v5 자동 감지
│   │
│   ├── harness/
│   │   ├── physical_validator.py
│   │   ├── agents_config.py
│   │   └── drift_detector.py
│   │
│   ├── data/
│   │   ├── lhs_sampler.py
│   │   ├── lighttools_runner.py        # v5 Robust Runner
│   │   ├── dataset_manager.py
│   │   └── flywheel.py
│   │
│   ├── api/                            # Phase E FastAPI
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── design.py
│   │       ├── inference.py
│   │       ├── candidates.py
│   │       ├── training.py
│   │       └── export.py
│   │
│   ├── AGENTS.md
│   └── __init__.py
│
├── scripts/                            ★ v6 신규: CLI 실행 스크립트
│   ├── generate_asm_lut.py             # LUT 생성 (노트북의 스크립트 버전)
│   ├── train_phase_c.py                # GPU 장시간 학습 (백그라운드)
│   ├── train_phase_c_resume.py         # 체크포인트 재개
│   ├── distill_fno.py                  # FNO 증류 (Phase D)
│   ├── run_lightttools_batch.py        # LT 야간 자동화
│   └── evaluate_phase_c.py             # 최종 검증
│
├── tests/                              ★ v6 신규: 유닛 테스트
│   ├── test_tmm.py
│   ├── test_asm.py
│   ├── test_pinn_model.py
│   ├── test_loss_functions.py
│   ├── test_interfaces.py              # 인터페이스 규약 검증
│   └── test_metrics.py
│
├── frontend/                           # Phase E React UI
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── store/designStore.js
│       ├── api/client.js
│       └── components/
│           ├── SpecInput/
│           ├── ParetoPlot/
│           ├── PsfChart/
│           ├── MtfChart/
│           ├── FingerprintView/
│           ├── DesignVarTable/
│           ├── CandidateCard/
│           ├── KpiBanner/
│           └── drawStack/
│
├── checkpoints/                        # 학습된 모델
│   ├── phase_c_warmstart.pt
│   ├── phase_c_stage1.pt
│   ├── phase_c_stage2.pt
│   ├── phase_c_final.pt                ← Pure PINN 완성
│   └── fno_surrogate.pt
│
├── data/                               # 데이터
│   ├── asm_luts/                       # ASM 입사 경계 LUT
│   │   └── incident_z40.npz
│   ├── lt_results/                     # LT 수집 (200 한도)
│   ├── lt_checkpoint/                  # LT 배치 진행 상황
│   └── fno_training/                   # PINN→FNO 증류용
│
├── experiments/                        # 실험 로그
│   ├── YYYY-MM-DD_phase_c_attempt_N/
│   │   ├── config.yaml
│   │   ├── training.log
│   │   ├── validation.json
│   │   └── red_flag_history.json
│   └── ...
│
├── configs/                            # 학습 config
│   ├── phase_c_small_cpu.yaml
│   ├── phase_c_full_gpu.yaml
│   └── phase_c_warmstart.yaml
│
├── docs/                               # 문서
│   ├── master_guide_v6.md              ← 이 문서
│   ├── development_workflow.md
│   └── api_spec.md
│
├── requirements.txt
├── pyproject.toml                      # 패키지 설정
├── docker-compose.yml
├── .gitignore
└── README.md
```

### 11.5 하이브리드 구조 (v6 신규)

#### 11.5.1 3단계 코드 분리 원칙

**각 코드의 역할이 명확히 분리됨**:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: notebooks/  (실험/탐색/분석)                        │
│  ────────────────────────────────────                        │
│  - Jupyter Notebook (.ipynb)                                  │
│  - 빠른 반복, 시각화, 디버깅                                  │
│  - Phase 1 계승 흐름                                          │
│  - 경영진 보고서 겸용                                         │
└──────────────────┬──────────────────────────────────────────┘
                   │ 확정된 로직 추출
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: backend/  (재사용 모듈)                             │
│  ────────────────────────────────                            │
│  - Python 모듈 (.py)                                          │
│  - 클래스, 함수 정의                                          │
│  - import로 재사용 가능                                       │
│  - 테스트 대상                                                │
└──────────────────┬──────────────────────────────────────────┘
                   │ CLI/API로 감싸기
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: scripts/ + backend/api/  (실행/배포)                │
│  ──────────────────────────────────────────                  │
│  - CLI 스크립트: 장시간 학습, 야간 자동화                      │
│  - FastAPI: Production 서비스                                 │
│  - Docker 배포                                                │
└─────────────────────────────────────────────────────────────┘
```

#### 11.5.2 코드 흐름 예시

**Step 1: 노트북에서 실험**

```python
# notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb

# 셀 1: Import
import torch
import numpy as np
import matplotlib.pyplot as plt

# 셀 2: 실험용 간단한 PINN 클래스 정의
class PINN_v1(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.net(x)

# 셀 3: 작은 데이터로 테스트
model = PINN_v1(hidden_dim=64)
coords = torch.randn(100, 8)
U = model(coords)
print(U.shape)  # 바로 결과 확인

# 셀 4: 시각화
plt.plot(U[:, 0].detach().numpy())
plt.show()

# ... 여러 실험 후 구조 확정 ...
```

**Step 2: 확정된 코드를 backend/로 추출**

```python
# backend/core/pinn_model.py

import torch
import torch.nn as nn
import math

class PurePINN(nn.Module):
    """
    노트북에서 확정된 모델을 재사용 가능한 클래스로 변환.
    
    노트북의 PINN_v1을 기반으로 개선:
    - SIREN 초기화 추가
    - Fourier feature embedding
    - 타입 힌트
    - docstring
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_freqs: int = 48,
        omega_0: float = 30.0
    ):
        super().__init__()
        # ... (섹션 13.4의 완성된 코드)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # ...
        return U
```

**Step 3: 노트북에서 모듈 재사용**

```python
# notebooks/02_phase_c_development/03_pinn_training_monitor.ipynb

# 셀 1
from backend.core.pinn_model import PurePINN
from backend.training.loss_functions import (
    helmholtz_loss, phase_loss, bm_boundary_loss
)

# 셀 2 - 모듈 import로 재사용
model = PurePINN(hidden_dim=128, num_layers=4)

# 셀 3 - 학습 (노트북의 강점: 실시간 시각화)
loss_history = []
for epoch in range(100):
    loss = training_step(model, ...)
    loss_history.append(loss)
    
    if epoch % 10 == 0:
        plt.clf()
        plt.plot(loss_history)
        plt.yscale('log')
        plt.draw()
        plt.pause(0.1)
```

**Step 4: 확정 후 scripts/로 CLI 버전**

```python
# scripts/train_phase_c.py

"""
GPU 장시간 학습 스크립트.
노트북의 로직을 CLI로 감쌈.

사용법:
    python scripts/train_phase_c.py --config configs/phase_c_full_gpu.yaml
    
백그라운드 실행:
    nohup python scripts/train_phase_c.py \\
        --config configs/phase_c_full_gpu.yaml \\
        > training.log 2>&1 &
"""

import argparse
import yaml
from backend.core.pinn_model import PurePINN  # 같은 모듈!
from backend.training.loss_functions import *
from backend.training.red_flag_detector import RedFlagDetector

def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = PurePINN(**config['model'])
    detector = RedFlagDetector(device=config['device'])
    
    for epoch in range(config['epochs']):
        # 학습
        loss = training_step(model, ...)
        
        # Red flag 체크 (v5 섹션 18.5)
        if epoch % 100 == 0:
            alerts = detector.check_all(model, epoch)
            # ...
        
        # 체크포인트
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
```

**Step 5: Phase E에서 API로 감싸기**

```python
# backend/api/main.py

from fastapi import FastAPI
from backend.core.pinn_model import PurePINN  # 같은 모듈!
from backend.core.botorch_optimizer import BMOptimizer

app = FastAPI()

# 학습된 모델 로드
model = PurePINN(...)
model.load_state_dict(torch.load('checkpoints/phase_c_final.pt'))
optimizer = BMOptimizer(model)

@app.post("/api/design/run")
async def run_design(spec: BMDesignSpec):
    result = optimizer.optimize(spec)
    return result

# 실행:
# uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

#### 11.5.3 각 레이어의 책임

**notebooks/ 책임**:

```
✓ 탐색적 데이터 분석 (EDA)
✓ 가설 검증 (작은 모델, 작은 데이터)
✓ 결과 시각화
✓ 디버깅
✓ 경영진/동료 공유용 리포트
✓ Phase 1 흐름 계승

✗ Production 로직 정의 (모듈로 추출)
✗ 장시간 학습 (scripts로)
✗ 자동화 (scripts로)
```

**backend/ 책임**:

```
✓ 재사용 가능한 클래스/함수
✓ 독립적 모듈 (타 파일에서 import)
✓ 타입 힌트 + docstring
✓ 유닛 테스트 대상
✓ FastAPI에서 import

✗ main 함수 (scripts가 담당)
✗ CLI 파싱 (scripts가 담당)
✗ 시각화 (노트북/helpers가 담당)
```

**scripts/ 책임**:

```
✓ CLI 파싱 (argparse)
✓ 설정 파일 로드 (yaml)
✓ 장시간 실행 (GPU 학습)
✓ 백그라운드 실행
✓ 로깅
✓ 체크포인트 저장
✓ 자동화 (CRON, Task Scheduler)

✗ 로직 정의 (backend/ import)
✗ 상세 분석 (노트북으로)
```

#### 11.5.4 실제 워크플로우 — Phase C 예시

```
[월요일 - 실험]
notebooks/02_phase_c_development/01_asm_lut_generation.ipynb
  ↓ ASM LUT 설계 완료 + 시각화로 검증
  ↓ 결과 저장: data/asm_luts/incident_z40.npz

notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb
  ↓ PurePINN 클래스 실험 (셀에서 정의)
  ↓ 작은 모델 학습 (CPU, 30분)
  ↓ 구조 확정

[화요일 - 모듈 추출]
노트북의 PurePINN 클래스 → backend/core/pinn_model.py
노트북의 Loss 함수 → backend/training/loss_functions.py
노트북의 샘플링 → backend/training/collocation_sampler.py

테스트 작성:
  tests/test_pinn_model.py
  tests/test_loss_functions.py

[수요일 - 스크립트 작성]
scripts/train_phase_c.py 작성 (CLI)
  - backend/ 모듈 import
  - argparse로 config 받음
  - 로깅, 체크포인트

[수요일 밤 - 학습 시작]
nohup python scripts/train_phase_c.py \
    --config configs/phase_c_full_gpu.yaml \
    > experiments/2026-04-15_phase_c_attempt1/training.log 2>&1 &

[목요일 아침 - 분석]
notebooks/02_phase_c_development/04_pinn_evaluation.ipynb
  ↓ 학습된 모델 로드
  ↓ z 내부 fringe 시각화 (여러 단면)
  ↓ 설계변수 반응 테스트
  ↓ Red flag 체크

문제 발견 시:
  ↓ 노트북에서 원인 분석
  ↓ backend/ 모듈 수정
  ↓ 재학습

[금요일 - 보고서]
notebooks/02_phase_c_development/06_phase_c_report.ipynb
  ↓ 결과 정리
  ↓ 시각화 포함
  ↓ 경영진/팀 공유용
  ↓ Jupyter → PDF 또는 HTML export
```

#### 11.5.5 Import 구조

**pyproject.toml 또는 setup.py 설정**:

```toml
# pyproject.toml
[project]
name = "udfps-pinn-platform"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["backend*"]
```

**설치**:
```bash
# 개발 모드 설치 (한 번만)
pip install -e .

# 이제 어디서든 import 가능
# - 노트북: from backend.core.pinn_model import PurePINN
# - 스크립트: from backend.core.pinn_model import PurePINN
# - FastAPI: from backend.core.pinn_model import PurePINN
```

**경로 문제 방지**:
```python
# 노트북 맨 위 셀 (pip install -e . 안 했을 때)
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))

# 이후
from backend.core.pinn_model import PurePINN  # 정상 작동
```

#### 11.5.6 Git 버전 관리

```
# .gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store

# 학습 결과 (대용량)
checkpoints/*.pt
!checkpoints/.gitkeep

# 실험 로그
experiments/*/
!experiments/.gitkeep

# LT 결과
data/lt_results/
data/lt_checkpoint/

# 모델 출력
*.pkl

# Jupyter 출력 정리 (선택)
# nbstripout을 pre-commit에 설정
```

**노트북 Git 관리 팁**:

```bash
# nbstripout 설치 (노트북 출력 제거, diff 깔끔)
pip install nbstripout
nbstripout --install

# 이제 git commit 시 출력 자동 제거
# 코드만 버전 관리됨
```

#### 11.5.7 장점 요약

```
Jupyter-only 대비:
  ✓ 재사용성 (backend/ import)
  ✓ 테스트 가능
  ✓ 장시간 학습 (백그라운드)
  ✓ Production 배포
  ✓ Git 관리

Python-only 대비:
  ✓ 실험 속도 (셀 반복)
  ✓ 시각화 직관적
  ✓ 디버깅 편함
  ✓ 경영진 보고 (Jupyter 스토리텔링)
  ✓ Phase 1 계승

요약: 두 방식의 장점만 조합
```

---

## 12. AGENTS.md — 도메인 규칙

```markdown
# UDFPS BM 광학 설계 도메인 규칙 (AGENTS.md)

## 시스템 고정값 (절대 변경 불가)

### COE 스택
- Cover Glass + OCR: 550μm (합산), n=1.52
- AR 코팅: Gorilla DX (4층, Phase 1 최적값)
  - SiO2 34.6nm / TiO2 25.9nm / SiO2 20.7nm / TiO2 169.5nm
  - 총 두께 ~300nm
- BM1: z = 20μm (두께 0.1μm 무시)
- BM2: z = 40μm (두께 0.1μm 무시)
- ILD: d = 20μm (BM1-BM2, 고정)
- Encap: 20μm (BM1-OPD)
- OPD: 폭 10μm, 피치 72μm, z = 0μm

### 광학 상수
- 평가 파장: 520nm
- k₀ ≈ 12.08 μm⁻¹
- k = k₀·n_CG ≈ 18.37 μm⁻¹
- CG 임계각: 41.1°
- 크로스토크: 22.5° → 191μm (2.65 피치)

### PINN 도메인
- 가로 x: 504μm (7 피치)
- 깊이 z: 40μm (BM~OPD)
- 입사각: -41.1° ~ +41.1°

### Stack Height (타일링)
- Stack Height = 590μm (CG 550 + Encap 20 + BM 영역 20)
- 각 픽셀 입사각 = arctan(distance / 590)

## 최적화 설계변수 (4개, μm) — Phase C 초기

| 변수 | 범위 | 설명 |
|---|---|---|
| δ_BM1 | -10 ~ +10 | BM1 오프셋 |
| δ_BM2 | -10 ~ +10 | BM2 오프셋 |
| w₁    |  5 ~ 20   | BM1 아퍼처 폭 |
| w₂    |  5 ~ 20   | BM2 아퍼처 폭 |

**Phase D 확장**: AR 4층 (d1~d4) 포함 → 7D

## 물리 하드 제약

- w₁ > 0, w₂ > 0
- d_int = 20μm 고정
- |δ_BM1| ≤ w₁/2
- |δ_BM2| ≤ w₂/2
- θ_eff = arctan(w₁/20) ≤ 41.1°
- PINN 도메인 외삽 금지

## PINN 구조 규칙 (정석)

- 입력: 8D (힌트 금지)
- 출력: 복소 U (U_re, U_im)
- Hard mask 금지
- L_Helmholtz 가중치 >= 0.5
- L_BC 가중치 >= 0.3

## 🛑 절대 금지 사항 (Phase B 재발 방지)

### 1. 우회 기법 금지
- ✗ Hard mask
- ✗ slit_dist 9D 입력
- ✗ 네트워크 출력 수학적 강제

### 2. Loss 가중치 금지
- ✗ lam_pde < 0.5
- ✗ lam_phase > lam_pde
- ✗ L_BC 제거 또는 0.1 미만

### 3. 도메인 금지
- ✗ z 범위 축소 (반드시 [0, 40])
- ✗ z 범위 확장 (570μm 등, CG 포함 시 PINN 학습 불가능)
- ✗ x 2피치 이하

### 4. 검증 우회 금지
- ✗ z 내부 fringe 확인 없이 "성공"
- ✗ 결과 정규화 강제

### 5. 근본 회피 금지
- ✗ "ASM으로 대체" 제안 수용
- ✗ "일단" 접근

### 위반 시 조치
1. 즉시 중단
2. 마스터 가이드 🛑 섹션 재확인
3. 근본 원인 분석
4. 북극성 재확인
5. 정석 재시작

## 최적화 목표 (3목적 qNEHVI)

- MTF@ridge  ≥ 60%
- skewness   ≤ 0.10
- 광량 T     ≥ 60%

## BoTorch 규칙

- LHS 20점 초기
- qNEHVI (qLogNEHVI fallback)
- HV 개선 < 0.1% × 5회 → 수렴
- σ > 0.05 → LT 강제 검증

## Evaluator 채점 (각 20점)

1. MTF 달성도
2. skewness 달성도
3. T 달성도
4. BO 수렴
5. 물리 제약 마진
- 70점 미만 → 재탐색

## Active Learning

- LT 결과 → 재학습 편입
- PINN 오차 > 5% → 재학습 트리거
- σ 상위 20% → LT 우선 검증
- LT 200개 한도

## 출력 형식

- 설계변수: μm 단위
- PSF: skewness, FWHM, peak, crosstalk
- 불확실도 σ 항상 출력
```

---

## 13. 핵심 모듈 구현 명세

### 13.1 BMPhysicalValidator (Phase 1 동일)

```python
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationResult:
    passed: bool
    reason: Optional[str] = None
    fix_hint: Optional[str] = None

class BMPhysicalValidator:
    BOUNDS = {
        "delta_bm1": (-10.0, 10.0),
        "delta_bm2": (-10.0, 10.0),
        "w1":        (5.0,   20.0),
        "w2":        (5.0,   20.0),
    }
    D_INT      = 20.0
    THETA_CRIT = 41.1

    def validate(self, p):
        for f, (lo, hi) in self.BOUNDS.items():
            v = getattr(p, f)
            if not (lo <= v <= hi):
                return ValidationResult(False, f"{f}={v:.2f} 범위 초과")
        
        if p.w1 <= 0 or p.w2 <= 0:
            return ValidationResult(False, "w 양수")
        
        if abs(p.delta_bm1) > p.w1 / 2:
            return ValidationResult(False, 
                f"|δ_BM1|={abs(p.delta_bm1):.2f} > w₁/2={p.w1/2:.2f}")
        if abs(p.delta_bm2) > p.w2 / 2:
            return ValidationResult(False,
                f"|δ_BM2|={abs(p.delta_bm2):.2f} > w₂/2={p.w2/2:.2f}")
        
        theta_eff = math.degrees(math.atan(p.w1 / (2 * self.D_INT)))
        if theta_eff > self.THETA_CRIT:
            return ValidationResult(False,
                f"θ_eff={theta_eff:.1f}° > {self.THETA_CRIT}°")
        
        return ValidationResult(True)
```

### 13.2 TMM 계산기 (Phase 1 계승)

```python
import numpy as np
from tmm import coh_tmm

class GorillaDXTMM:
    """Gorilla DX (4층) AR 코팅 TMM"""
    
    N_LIST = [1.0, 1.46, 2.35, 1.46, 2.35, 1.52]
    D_LIST_NOMINAL = [np.inf, 100, 70, 120, 50, np.inf]  # nm
    
    def compute_complex_t(self, theta_deg, wl_nm=520.0, d_list=None):
        """복소 투과 계수 t(θ) = |t|·exp(i·Δφ)"""
        if d_list is None:
            d_list = self.D_LIST_NOMINAL
        
        th_rad = np.radians(abs(theta_deg))
        res = coh_tmm('p', self.N_LIST, d_list, th_rad, wl_nm)
        ref = coh_tmm('p', self.N_LIST, d_list, 0.0, wl_nm)
        
        t_amp = np.abs(res['t'])
        dphi_deg = np.angle(res['t'], deg=True) - np.angle(ref['t'], deg=True)
        
        if theta_deg < 0:
            dphi_deg = -dphi_deg
        elif theta_deg == 0:
            dphi_deg = 0.0
        
        return float(t_amp), float(dphi_deg)
    
    def compute_lut(self, theta_range=np.arange(-41, 42, 1), wl_nm=520.0, d_list=None):
        return {
            float(th): self.compute_complex_t(th, wl_nm, d_list)
            for th in theta_range
        }
```

### 13.3 ASM Propagator (Phase 1 계승)

```python
import numpy as np

class ASMPropagator:
    """
    Angular Spectrum Method.
    CG 550μm 자유공간 전파.
    """
    
    def __init__(self, wl_um=0.520, n=1.52, N_grid=4096, x_extent_um=504.0):
        self.wl = wl_um
        self.n = n
        self.N = N_grid
        self.x_extent = x_extent_um
        self.dx = x_extent_um / N_grid
        self.k0 = 2 * np.pi / wl_um
        self.k = self.k0 * n
    
    def propagate(self, U_in, z_propagate_um):
        """복소장 U_in을 z_propagate_um만큼 전파"""
        # FFT
        A = np.fft.fft(U_in)
        kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        
        # Transfer function
        kz_sq = self.k**2 - kx**2
        kz = np.sqrt(kz_sq.astype(complex))
        H = np.exp(1j * kz * z_propagate_um)
        
        # Evanescent 제거
        H[kz_sq < 0] = 0
        
        # IFFT
        U_out = np.fft.ifft(A * H)
        return U_out
    
    def compute_incident_lut(self, tmm_lut, cg_thick=550.0, z_pinn_inlet=40.0):
        """
        z=40 PINN 입사 경계 LUT 생성.
        
        과정:
          AR 통과 후 평면파 (z=590) → CG 550μm 전파 → z=40 도달
          → 해당 위치 복소장을 LUT로 저장
        
        Args:
            tmm_lut: {theta_deg: (t_amp, dphi_deg)}
            cg_thick: CG 두께 (μm)
            z_pinn_inlet: PINN 입사 경계 z 좌표 (μm)
        """
        z_propagate = cg_thick  # 550μm 전파
        
        lut = {}
        for theta_deg, (t_amp, dphi_deg) in tmm_lut.items():
            # 초기 평면파 at z=590 (AR 직후, CG 상단)
            x = np.linspace(0, self.x_extent, self.N, endpoint=False)
            sin_th = np.sin(np.radians(theta_deg))
            dphi_rad = np.radians(dphi_deg)
            
            # AR 투과 후 평면파
            U_initial = t_amp * np.exp(1j * (dphi_rad + self.k0 * x * sin_th))
            
            # CG 전파 (550μm, z=590→z=40)
            U_at_40 = self.propagate(U_initial, z_propagate)
            
            lut[theta_deg] = {
                'x': x,
                'U_re': U_at_40.real,
                'U_im': U_at_40.imag,
            }
        
        return lut
```

### 13.4 Pure PINN Model

```python
import torch
import torch.nn as nn
import math

class PurePINN(nn.Module):
    """
    Pure Physics-Informed Neural Network.
    
    입력: 8D (x, z, δ₁, δ₂, w₁, w₂, sin θ, cos θ)
    출력: 복소 U (U_re, U_im)
    
    Hard mask 없음. slit_dist 입력 없음.
    BM 경계는 L_BC로 학습.
    
    도메인: x ∈ [0, 504]μm, z ∈ [0, 40]μm
    """
    
    def __init__(self, hidden_dim=128, num_layers=4, num_freqs=48, omega_0=30.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.omega_0 = omega_0
        
        # Fourier feature embedding
        self.B = nn.Parameter(
            torch.randn(8, num_freqs) * 10.0,
            requires_grad=False
        )
        
        input_dim = 2 * num_freqs
        
        # SIREN layers
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layer = nn.Linear(in_dim, hidden_dim)
            if i == 0:
                nn.init.uniform_(layer.weight, -1/in_dim, 1/in_dim)
            else:
                bound = math.sqrt(6/in_dim) / omega_0
                nn.init.uniform_(layer.weight, -bound, bound)
            layers.append(layer)
            in_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 2)
    
    def normalize_input(self, coords):
        """입력 정규화 (z=40 기준)"""
        x_norm = coords[:, 0:1] / 504.0
        z_norm = coords[:, 1:2] / 40.0        # ← Phase C: /40
        d1_norm = coords[:, 2:3] / 10.0
        d2_norm = coords[:, 3:4] / 10.0
        w1_norm = (coords[:, 4:5] - 12.5) / 7.5
        w2_norm = (coords[:, 5:6] - 12.5) / 7.5
        sin_th = coords[:, 6:7]
        cos_th = coords[:, 7:8]
        
        return torch.cat([
            x_norm, z_norm, d1_norm, d2_norm,
            w1_norm, w2_norm, sin_th, cos_th
        ], dim=1)
    
    def fourier_embed(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def forward(self, coords):
        x_norm = self.normalize_input(coords)
        h = self.fourier_embed(x_norm)
        
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if i == 0:
                h = torch.sin(self.omega_0 * h)
            else:
                h = torch.sin(h)
        
        U = self.output(h)
        return U
```

### 13.5 PSFMetrics (Phase 1 동일)

```python
import numpy as np

class PSFMetrics:
    PITCH = 72.0
    OPD_W = 10.0
    N_PIXELS = 7
    
    def compute(self, psf_7):
        psf_7 = np.asarray(psf_7)
        
        ridge_vals = psf_7[[0, 2, 4, 6]]
        valley_vals = psf_7[[1, 3, 5]]
        
        r_mean = ridge_vals.mean()
        v_mean = valley_vals.mean()
        mtf = (r_mean - v_mean) / (r_mean + v_mean + 1e-8)
        
        xs = np.arange(self.N_PIXELS) * self.PITCH
        norm = psf_7 / (psf_7.sum() + 1e-8)
        mu = np.sum(xs * norm)
        sigma = np.sqrt(np.sum((xs - mu)**2 * norm) + 1e-8)
        skew = np.sum(((xs - mu)/sigma)**3 * norm)
        
        T = float(psf_7.sum())
        
        center = psf_7[3]
        xtalk = (psf_7[2] + psf_7[4]) / 2.0
        xtalk_ratio = xtalk / (center + 1e-8)
        
        return dict(
            mtf_ridge=float(np.clip(mtf, 0, 1)),
            skewness=float(skew),
            throughput=T,
            crosstalk_ratio=float(xtalk_ratio),
        )
```

### 13.6 FNO Surrogate

```python
import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    """Fourier 도메인 학습 가능 kernel"""
    
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.modes], self.weights
        )
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOSurrogate(nn.Module):
    """
    4D 설계변수 → PSF 7개 OPD
    
    PINN 증류용.
    """
    
    def __init__(self, hidden_channels=32, modes=16, n_fourier_layers=4):
        super().__init__()
        self.lift = nn.Linear(4, hidden_channels)  # 4D → hidden
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(hidden_channels, hidden_channels, modes)
            for _ in range(n_fourier_layers)
        ])
        
        self.w_layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1)
            for _ in range(n_fourier_layers)
        ])
        
        self.instance_norm = nn.InstanceNorm1d(hidden_channels)
        self.activation = nn.GELU()
        
        self.project = nn.Linear(hidden_channels, 7)  # hidden → 7 OPD
    
    def forward(self, p):
        """
        Args:
            p: (batch, 4) 설계변수
        Returns:
            psf: (batch, 7) PSF
        """
        # p → spatial representation (임시)
        # 실제로는 더 복잡한 lifting 필요
        x = self.lift(p).unsqueeze(-1).expand(-1, -1, 128)  # (B, hidden, 128)
        
        for fourier, w in zip(self.fourier_layers, self.w_layers):
            x_fourier = fourier(x)
            x_local = w(x)
            x = self.instance_norm(x_fourier + x_local)
            x = self.activation(x)
        
        # Project to 7 OPD
        x_mean = x.mean(dim=-1)  # (B, hidden)
        psf = self.project(x_mean)  # (B, 7)
        psf = torch.relu(psf)  # 비음수
        
        return psf
```

### 13.7 BoTorch Optimizer

```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHyperVolumeImprovement
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

class BMOptimizer:
    BOUNDS_RAW = {
        "delta_bm1": (-10.0, 10.0),
        "delta_bm2": (-10.0, 10.0),
        "w1":        (5.0,   20.0),
        "w2":        (5.0,   20.0),
    }
    REF_POINT = torch.tensor([0.0, 0.0, -1.0])
    
    def __init__(self, fno_surrogate, validator):
        self.fno = fno_surrogate
        self.val = validator
        lo = [v[0] for v in self.BOUNDS_RAW.values()]
        hi = [v[1] for v in self.BOUNDS_RAW.values()]
        self.bounds = torch.tensor([lo, hi], dtype=torch.float64)
    
    def _eval(self, p_batch):
        results = []
        for p in p_batch:
            params = self._to_params(p)
            v = self.val.validate(params)
            if not v.passed:
                results.append(torch.tensor([0.0, 0.0, -1.0]))
                continue
            psf7 = self.fno(p.unsqueeze(0)).squeeze().detach().numpy()
            m = PSFMetrics().compute(psf7)
            results.append(torch.tensor([
                m["mtf_ridge"],
                m["throughput"],
                -m["skewness"],
            ]))
        return torch.stack(results)
    
    def optimize(self, n_iter=50):
        X = draw_sobol_samples(bounds=self.bounds, n=1, q=20).squeeze(0)
        Y = self._eval(X)
        
        for i in range(n_iter):
            models = [SingleTaskGP(X, Y[:, j:j+1]) for j in range(3)]
            model = ModelListGP(*models)
            acqf = qLogNoisyExpectedHyperVolumeImprovement(
                model=model, ref_point=self.REF_POINT, X_baseline=X)
            cands, _ = optimize_acqf(
                acqf, bounds=self.bounds, q=4,
                num_restarts=10, raw_samples=512)
            Y_new = self._eval(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, Y_new])
        
        mask = is_non_dominated(Y)
        top5 = Y[mask].sum(dim=1).topk(min(5, mask.sum())).indices
        return dict(
            pareto_X=X[mask],
            pareto_Y=Y[mask],
            top5=[{"p": X[mask][i], "Y": Y[mask][i]} for i in top5]
        )
    
    def _to_params(self, t):
        k = list(self.BOUNDS_RAW.keys())
        return BMDesignParams(**{k[i]: float(t[i]) for i in range(4)})
```

### 13.8 Robust LightTools Runner (야간 자동화 방어 로직)

#### 13.8.1 왜 필수인가

```
LightTools COM API의 알려진 문제:
1. 메모리 누수 (수 시간 연속 실행 시 크래시)
2. 라이선스 서버 타임아웃
3. 팝업 다이얼로그로 스크립트 정지
4. Windows 절전 모드 진입
5. COM 객체 해제 실패

증상:
- 야간 120개 연속 실행 중 20~30번째에 멈춤
- 스크립트 전체 재실행 필요
- 이미 완료된 것도 재계산
```

#### 13.8.2 방어 로직 구현

```python
# backend/data/lighttools_runner.py

import subprocess
import multiprocessing as mp
import time
import gc
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


@dataclass
class LTSimulationParams:
    """LightTools 시뮬레이션 파라미터."""
    delta_bm1: float
    delta_bm2: float
    w1: float
    w2: float
    theta_deg: float
    
    def hash_id(self) -> str:
        """고유 ID 생성 (체크포인트용)."""
        import hashlib
        s = f"{self.delta_bm1:.3f}_{self.delta_bm2:.3f}_{self.w1:.3f}_{self.w2:.3f}_{self.theta_deg:.2f}"
        return hashlib.md5(s.encode()).hexdigest()[:12]


@dataclass
class LTResult:
    """LightTools 결과."""
    params: LTSimulationParams
    psf_7: List[float]
    runtime_sec: float
    success: bool
    error: Optional[str] = None


class RobustLightToolsRunner:
    """
    야간 자동화에 강건한 LightTools 실행기.
    
    기능:
    - multiprocessing 격리 (메모리 누수 방지)
    - Timeout 처리 (멈춤 감지)
    - 재시도 로직
    - 체크포인트 (중단 후 재개)
    - COM 객체 명시적 해제
    - LightTools 강제 종료 (hang 시)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        timeout_sec: int = 600,
        checkpoint_dir: str = "data/lt_checkpoint",
        results_dir: str = "data/lt_results"
    ):
        self.max_retries = max_retries
        self.timeout = timeout_sec
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / "completed.json"
        self.completed: set = self._load_checkpoint()
    
    def _load_checkpoint(self) -> set:
        """이전 진행 상황 로드."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return set(json.load(f))
        return set()
    
    def _save_checkpoint(self):
        """진행 상황 저장."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.completed), f)
    
    def run_single(self, params: LTSimulationParams) -> Optional[LTResult]:
        """
        단일 시뮬레이션 실행 (방어 로직 포함).
        
        Returns:
            LTResult 또는 None (모든 재시도 실패 시)
        """
        params_id = params.hash_id()
        
        # 이미 완료됐으면 skip
        if params_id in self.completed:
            log.info(f"Skip {params_id} (already completed)")
            return self._load_result(params_id)
        
        for attempt in range(self.max_retries):
            log.info(f"Attempt {attempt+1}/{self.max_retries} for {params_id}")
            
            try:
                # multiprocessing으로 격리
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
                    async_result = pool.apply_async(
                        _lt_worker, (asdict(params),)
                    )
                    
                    try:
                        psf_7, runtime = async_result.get(timeout=self.timeout)
                        
                        result = LTResult(
                            params=params,
                            psf_7=psf_7,
                            runtime_sec=runtime,
                            success=True
                        )
                        
                        self._save_result(params_id, result)
                        self.completed.add(params_id)
                        self._save_checkpoint()
                        
                        log.info(f"Success: {params_id} in {runtime:.1f}s")
                        return result
                    
                    except mp.TimeoutError:
                        log.error(f"Timeout ({self.timeout}s) on {params_id}")
                        pool.terminate()
                        self._kill_lighttools()
                        time.sleep(30)  # LT 종료 대기
            
            except Exception as e:
                log.error(f"Attempt {attempt+1} failed: {e}")
                self._kill_lighttools()
                time.sleep(30)
        
        log.error(f"All retries failed for {params_id}")
        return None
    
    def run_batch(
        self,
        param_list: List[LTSimulationParams],
        resume: bool = True
    ) -> Dict[str, LTResult]:
        """
        배치 실행 (체크포인트 지원).
        
        Args:
            param_list: 시뮬레이션할 파라미터 리스트
            resume: True면 이전 중단 지점에서 재개
        
        Returns:
            {params_id: LTResult}
        """
        if not resume:
            self.completed = set()
            self._save_checkpoint()
        
        total = len(param_list)
        results = {}
        
        log.info(f"Starting batch: {total} simulations")
        log.info(f"Already completed: {len(self.completed)}")
        
        for i, params in enumerate(param_list):
            log.info(f"Progress: {i+1}/{total}")
            result = self.run_single(params)
            if result is not None:
                results[params.hash_id()] = result
            
            # 10개마다 체크포인트 저장
            if (i + 1) % 10 == 0:
                self._save_checkpoint()
                log.info(f"Checkpoint saved: {len(self.completed)}/{total}")
        
        return results
    
    def _save_result(self, params_id: str, result: LTResult):
        """결과 저장."""
        filepath = self.results_dir / f"{params_id}.json"
        with open(filepath, 'w') as f:
            json.dump({
                'params': asdict(result.params),
                'psf_7': result.psf_7,
                'runtime_sec': result.runtime_sec,
                'success': result.success,
            }, f, indent=2)
    
    def _load_result(self, params_id: str) -> Optional[LTResult]:
        """저장된 결과 로드."""
        filepath = self.results_dir / f"{params_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            data = json.load(f)
        return LTResult(
            params=LTSimulationParams(**data['params']),
            psf_7=data['psf_7'],
            runtime_sec=data['runtime_sec'],
            success=data['success']
        )
    
    def _kill_lighttools(self):
        """LightTools 프로세스 강제 종료."""
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', 'LightTools.exe'],
                capture_output=True,
                timeout=30
            )
            log.info("LightTools 프로세스 종료")
        except Exception as e:
            log.warning(f"LT 종료 실패: {e}")


def _lt_worker(params_dict: dict) -> tuple:
    """
    자식 프로세스에서 실행.
    
    매번 새 프로세스로 실행되어 메모리 누수 방지.
    """
    import pythoncom
    import win32com.client
    import time
    
    params = LTSimulationParams(**params_dict)
    
    pythoncom.CoInitialize()
    lt = None
    
    try:
        t_start = time.time()
        
        # LightTools COM 객체 생성
        lt = win32com.client.Dispatch("LightTools.LTAPI4")
        
        # 파라미터 설정
        _setup_simulation(lt, params)
        
        # 시뮬레이션 실행
        lt.Cmd("BeginCommandGroup; TraceRays; EndCommandGroup")
        
        # 결과 추출
        psf_7 = _extract_psf(lt)
        
        runtime = time.time() - t_start
        return psf_7, runtime
    
    finally:
        # 명시적 해제
        if lt is not None:
            try:
                lt.Cmd("Close")
            except:
                pass
            del lt
        pythoncom.CoUninitialize()
        gc.collect()


def _setup_simulation(lt, params: LTSimulationParams):
    """LightTools 파라미터 설정."""
    # BM1 오프셋
    lt.Cmd(f"Edit BM1_Aperture[Offset]={params.delta_bm1}")
    lt.Cmd(f"Edit BM1_Aperture[Width]={params.w1}")
    
    # BM2 오프셋
    lt.Cmd(f"Edit BM2_Aperture[Offset]={params.delta_bm2}")
    lt.Cmd(f"Edit BM2_Aperture[Width]={params.w2}")
    
    # 입사각
    lt.Cmd(f"Edit Source[Angle]={params.theta_deg}")


def _extract_psf(lt) -> List[float]:
    """OPD 센서에서 PSF 7개 추출."""
    psf_7 = []
    for i in range(7):
        value = lt.Cmd(f"Get OPD_Pixel_{i}[Intensity]")
        psf_7.append(float(value))
    return psf_7


# 사용 예시
if __name__ == "__main__":
    runner = RobustLightToolsRunner(
        max_retries=3,
        timeout_sec=600
    )
    
    # 80개 LHS 파라미터 생성
    from backend.data.lhs_sampler import generate_lhs_params
    param_list = generate_lhs_params(n=80)
    
    # 배치 실행 (야간)
    results = runner.run_batch(param_list, resume=True)
    
    print(f"Total completed: {len(results)}/{len(param_list)}")
```

#### 13.8.3 실행 방식

```bash
# 야간 자동 실행 (회사 Windows)
nohup python -m backend.data.lighttools_runner > lt_batch.log 2>&1 &

# 또는 Windows Task Scheduler 등록
schtasks /create /tn "LT_Nightly" /sc once /st 22:00 ^
  /tr "python -m backend.data.lighttools_runner"

# 진행 상황 모니터링
tail -f lt_batch.log

# 중간 중단 후 재개
python -m backend.data.lighttools_runner  # resume=True 자동
```

#### 13.8.4 체크포인트 구조

```
data/
├── lt_checkpoint/
│   └── completed.json       # 완료된 params_id 리스트
└── lt_results/
    ├── a1b2c3d4e5f6.json   # 각 시뮬레이션 결과
    ├── f6e5d4c3b2a1.json
    └── ...
```

#### 13.8.5 Windows 설정 (필수)

```
절전 모드 해제:
  제어판 → 전원 옵션 → 고급 설정
  절전 모드: 안 함
  디스플레이 끄기: 안 함
  하드 디스크 끄기: 안 함

Windows Update 연기:
  야간 실행 중 업데이트 재시작 방지
  설정 → Windows Update → 일시 중지

알림 차단:
  집중 지원 켜기
  팝업 방지
```

```
POST /api/design/run
     body: { spec, weights }
     response: { job_id }

GET  /api/design/status/{job_id}
     response: { status, progress, iteration, pareto_front, pinn_loss }

GET  /api/design/candidates/{job_id}
     response: { candidates: [BMCandidate] }

GET  /api/design/compare?candidate_ids[]=A&candidate_ids[]=B
POST /api/design/export/{candidate_id}

POST /api/inference/psf
     body: { params: BMDesignParams }
     response: { psf7, metrics }

POST /api/training/add_gt
     body: { p, psf7_measured }

GET  /api/fingerprint/simulate
     body: { params, fingerprint_id }
     response: { image_baseline, image_current, image_optimal }
     ← 지문 이미지 시뮬레이션 (섹션 17)

GET  /api/health
```

### 스키마

```python
from pydantic import BaseModel, Field

class BMDesignSpec(BaseModel):
    mtf_ridge_min:  float = Field(0.60, ge=0.10, le=0.95)
    skewness_max:   float = Field(0.10, ge=0.01, le=0.50)
    throughput_min: float = Field(0.60, ge=0.10, le=0.95)

class ParetoWeights(BaseModel):
    mtf:        float = 0.4
    throughput: float = 0.3
    skewness:   float = 0.3

class BMCandidate(BaseModel):
    id:                str
    label:             str
    params:            BMDesignParams
    mtf_ridge:         float
    skewness:          float
    throughput:        float
    crosstalk_ratio:   float
    evaluator_score:   float
    pareto_rank:       int
    uncertainty_sigma: float
    constraint_ok:     bool
```

---

## 15. 구현 순서

### Phase A — 기반 (1~2일) ✓ 완료
- 프로젝트 구조
- AGENTS.md
- BMPhysicalValidator
- Pydantic 스키마
- FastAPI
- TMM 계산기
- AR LUT
- ASM propagator

### Phase B — 실패 (교훈 획득) ✓ 완료
- Hard mask + slit_dist 우회
- L_H 가중치 0.01
- 회절 학습 실패
- **교훈**: 정석 PINN 필수

### Phase C — Pure PINN 재학습 ★ 지금

**Step 1: ASM LUT 생성 (1시간)**
```bash
python backend/physics/asm_lut_generator.py
# → data/asm_luts/incident_z40.npz 생성
# z=40 PINN 입사 경계 target
```

**Step 2: 집 CPU 검증 (30분~1시간)**
```bash
python backend/training/train_phase_c.py --config small --device cpu
```
성공 기준:
- Loss 발산 없음
- Slit/BM 분화 조짐
- z 내부 변화

**Step 3: 회사 GPU 본격 학습 (4-8시간)**
```bash
python backend/training/train_phase_c.py --config full --device cuda
```
성공 기준:
- Slit |A| ≈ ASM target (±10%)
- BM |A| < 0.05 (z=20, z=40 BM 영역)
- z 내부 회절 fringe (ILD, Encap)
- 설계변수 반응
- `phase_c_final.pt`

### Phase D — FNO + 역설계 (3~5일)
- PINN → FNO 증류 (10,000쌍)
- BoTorch qNEHVI
- 통합 테스트 (8초 역설계)

### Phase E — 플랫폼 (3~5일)
- Design Studio 3탭
- Active Learning
- LightTools 연동
- 지문 이미지 시뮬레이션 (Phase 1 기능 계승)
- E2E 배포

### 15.5 개발 워크플로우 (v6 신규)

#### 15.5.1 3단계 개발 패턴

모든 Phase는 아래 3단계를 거칩니다:

```
[1단계: 탐색] Jupyter Notebook
  ↓
[2단계: 모듈화] Python Module (backend/)
  ↓
[3단계: 자동화/배포] CLI Script + API (scripts/, backend/api/)
```

#### 15.5.2 탐색 단계 (Jupyter)

**목적**: 빠른 실험, 시각화, 가설 검증

**원칙**:
- 작은 데이터, 작은 모델
- 실시간 시각화 (matplotlib inline)
- 여러 접근법 비교
- 실패 기록 남기기

**노트북 구성 권장**:

```python
# 셀 1: 환경 설정
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))  # backend/ import 가능하게

import numpy as np
import torch
import matplotlib.pyplot as plt
%matplotlib inline

# 셀 2: 설정
CONFIG = {
    'hidden_dim': 64,      # 작게 시작
    'num_layers': 3,
    'n_colloc': 1000,      # 적게 시작
    'epochs': 100,         # 짧게 시작
    'device': 'cpu',       # CPU에서 검증
}

# 셀 3: 데이터/모델 준비
# (backend/ 모듈 import 또는 셀에서 프로토타입)

# 셀 4: 학습 루프 (시각화 포함)
loss_history = []
for epoch in range(CONFIG['epochs']):
    loss = ...
    loss_history.append(loss)
    
    if epoch % 10 == 0:
        # 실시간 loss curve
        plt.clf()
        plt.plot(loss_history)
        plt.yscale('log')
        plt.title(f'Epoch {epoch}')
        plt.show()

# 셀 5: 결과 분석
# z 내부 fringe, 설계변수 반응 등 시각화

# 셀 6: 결론
# 다음 단계로 가져갈 것 / 버릴 것 정리
```

**탐색 체크리스트**:

- [ ] 가설이 명확한가?
- [ ] 작은 scale에서 검증 가능한가?
- [ ] 시각화로 결과 확인했는가?
- [ ] Red flag가 없는가? (v5 섹션 18.5)
- [ ] backend/로 추출할 로직이 확정됐는가?

#### 15.5.3 모듈화 단계 (backend/)

**목적**: 노트북의 확정된 로직을 재사용 가능한 모듈로 변환

**언제 이동하는가**:

```
노트북에서:
- 같은 클래스/함수를 여러 셀에서 반복 사용
- 다른 노트북에서도 쓰고 싶음
- 확정된 로직 (더 이상 수정 안 함)

→ backend/ 모듈로 추출
```

**추출 절차**:

```python
# Before: 노트북의 셀
# notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb

class PurePINN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        # ...
    def forward(self, coords):
        # ...
```

```python
# After: 모듈로 추출
# backend/core/pinn_model.py

"""
Pure PINN model for UDFPS BM optimization.

Derived from: notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb
Verified in: notebooks/02_phase_c_development/03_pinn_training_monitor.ipynb
"""

import torch
import torch.nn as nn
from typing import Optional


class PurePINN(nn.Module):
    """
    Physics-Informed Neural Network for UDFPS BM.
    
    Input: 8D (x, z, δ₁, δ₂, w₁, w₂, sin θ, cos θ)
    Output: Complex U (U_re, U_im)
    
    Domain: x ∈ [0, 504]μm, z ∈ [0, 40]μm
    
    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Number of SIREN layers
        num_freqs: Fourier feature dimensions
        omega_0: SIREN frequency parameter
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_freqs: int = 48,
        omega_0: float = 30.0
    ):
        super().__init__()
        # ... (완성된 구현)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 8) tensor of input coordinates
        
        Returns:
            U: (N, 2) tensor of complex field (real, imag)
        """
        # ...
```

```python
# 노트북에서 재사용
# notebooks/02_phase_c_development/03_pinn_training_monitor.ipynb

from backend.core.pinn_model import PurePINN  # 추출된 모듈

model = PurePINN(hidden_dim=128, num_layers=4)
# ... 학습 및 실험
```

**모듈화 체크리스트**:

- [ ] Docstring 작성 (사용법, 입력/출력)
- [ ] Type hints 추가
- [ ] Import 정리 (불필요한 것 제거)
- [ ] 유닛 테스트 작성 (tests/)
- [ ] 상호 의존성 최소화

#### 15.5.4 자동화 단계 (scripts/ + API)

**목적**: 장시간 실행, 자동화, 배포

**scripts/ 작성 패턴**:

```python
# scripts/train_phase_c.py

"""
Phase C PINN 학습 스크립트.

Usage:
    python scripts/train_phase_c.py --config configs/phase_c_full_gpu.yaml

Background execution:
    nohup python scripts/train_phase_c.py \
        --config configs/phase_c_full_gpu.yaml \
        --output experiments/$(date +%Y-%m-%d_%H%M%S) \
        > training.log 2>&1 &
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from backend.core.pinn_model import PurePINN
from backend.training.loss_functions import (
    helmholtz_loss, phase_loss, bm_boundary_loss
)
from backend.training.red_flag_detector import RedFlagDetector
from backend.training.collocation_sampler import hierarchical_collocation_with_buffer


def setup_logging(output_dir: Path) -> logging.Logger:
    """로거 설정."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(config_path: str, output_dir: Path):
    """메인 학습 루프."""
    # Config 로드
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    log = setup_logging(output_dir)
    log.info(f"Config: {config}")
    
    # Config 복사 (재현성)
    import shutil
    shutil.copy(config_path, output_dir / 'config.yaml')
    
    # 모델
    device = config['device']
    model = PurePINN(**config['model']).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Red Flag Detector
    detector = RedFlagDetector(device=device)
    
    # 학습 루프
    for epoch in range(config['epochs']):
        # Collocation 샘플링
        coords = hierarchical_collocation_with_buffer(
            n_total=config['n_colloc'],
            device=device
        )
        
        # Loss 계산
        loss_h = helmholtz_loss(model, coords)
        loss_p = phase_loss(model, ...)
        loss_bc = bm_boundary_loss(model, ...)
        
        loss = (config['lambda_h'] * loss_h 
                + config['lambda_phase'] * loss_p
                + config['lambda_bc'] * loss_bc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 로깅
        if epoch % 100 == 0:
            log.info(
                f"Epoch {epoch}: L_H={loss_h:.6f}, "
                f"L_phase={loss_p:.6f}, L_BC={loss_bc:.6f}"
            )
            
            # Red flag 체크
            alerts = detector.check_all(model, epoch)
            for alert in alerts:
                log.warning(f"[{alert.severity}] {alert.flag_name}: {alert.message}")
        
        # 체크포인트
        if epoch % 500 == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path}")
    
    # 최종 저장
    final_path = output_dir / 'phase_c_final.pt'
    torch.save(model.state_dict(), final_path)
    log.info(f"Training complete. Model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config YAML path')
    parser.add_argument('--output', default=None, help='Output directory')
    args = parser.parse_args()
    
    if args.output:
        output_dir = Path(args.output)
    else:
        from datetime import datetime
        output_dir = Path(f'experiments/phase_c_{datetime.now():%Y%m%d_%H%M%S}')
    
    main(args.config, output_dir)
```

**Config 파일 패턴**:

```yaml
# configs/phase_c_full_gpu.yaml

device: cuda
seed: 42

model:
  hidden_dim: 128
  num_layers: 4
  num_freqs: 48
  omega_0: 30.0

training:
  lr: 1.0e-3
  epochs: 10000
  n_colloc: 30000
  
  # Loss weights
  lambda_h: 1.0
  lambda_phase: 0.5
  lambda_bc: 0.5
  lambda_i: 0.3
  
  # Curriculum
  stage1_end: 2000
  stage2_end: 6000
  stage3_end: 10000

validation:
  check_interval: 100
  asm_lut_path: "data/asm_luts/incident_z40.npz"
```

**배포 (Phase E)**:

```python
# backend/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.pinn_model import PurePINN
from backend.core.botorch_optimizer import BMOptimizer
from backend.api.routes import design, inference, candidates

app = FastAPI(title="UDFPS PINN Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우트 등록
app.include_router(design.router, prefix="/api/design")
app.include_router(inference.router, prefix="/api/inference")
app.include_router(candidates.router, prefix="/api/candidates")

# 모델 로드 (startup)
@app.on_event("startup")
async def load_models():
    app.state.pinn = PurePINN(...)
    app.state.pinn.load_state_dict(torch.load('checkpoints/phase_c_final.pt'))
    app.state.pinn.eval()

# 실행:
# uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

#### 15.5.5 전체 Phase C 타임라인 예시

```
Day 1 (월요일): 탐색
─────────────────────
오전: notebooks/02_phase_c_development/01_asm_lut_generation.ipynb
  - TMM LUT 시각화 (위상 vs 각도)
  - ASM 전파 테스트 (θ=0, 30 비교)
  - z=40 경계 복소장 확인
  - NPZ 저장

오후: notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb
  - PurePINN 프로토타입 (셀에서 정의)
  - Loss 함수 4개 각각 테스트
  - 100 epoch CPU 학습
  - Loss 감소 확인

저녁: 확정 로직 → backend/ 모듈 추출
  - backend/core/pinn_model.py
  - backend/training/loss_functions.py
  - backend/training/collocation_sampler.py


Day 2 (화요일): 모듈화 + 스크립트
─────────────────────
오전: backend/ 모듈 테스트 작성
  - tests/test_pinn_model.py
  - tests/test_loss_functions.py
  - pytest 통과 확인

오후: scripts/train_phase_c.py 작성
  - CLI 파싱
  - Config 로드
  - 학습 루프
  - 로깅, 체크포인트

저녁: configs/phase_c_full_gpu.yaml 작성
  - GPU 설정
  - 학습 파라미터
  - Loss 가중치


Day 3 (수요일): 학습 시작
─────────────────────
오전: 집 CPU에서 작은 config 테스트
  nohup python scripts/train_phase_c.py \
    --config configs/phase_c_small_cpu.yaml &
  - 5 epoch만 돌려서 오류 없는지 확인

오후: 회사 GPU로 이동
  git pull
  nohup python scripts/train_phase_c.py \
    --config configs/phase_c_full_gpu.yaml \
    > training.log 2>&1 &
  - 4-8시간 학습 시작
  - Terminal 떠나도 계속 실행

저녁: LightTools 야간 수집 병행
  nohup python scripts/run_lighttools_batch.py --n 80 &


Day 4 (목요일): 분석 + 반복
─────────────────────
오전: notebooks/02_phase_c_development/04_pinn_evaluation.ipynb
  - 학습된 모델 로드 (checkpoints/phase_c_final.pt)
  - z=10 단면 시각화 (내부 fringe)
  - z=40 BM2 경계 확인 (BM 영역 U≈0?)
  - 설계변수 변화 → PSF 변화 테스트
  - Red flag 체크

오후: 문제 발견 시 반복
  - 노트북에서 원인 분석
  - backend/ 모듈 수정 (예: L_BC 가중치)
  - configs/ 수정
  - scripts/ 재실행


Day 5 (금요일): 보고
─────────────────────
notebooks/02_phase_c_development/06_phase_c_report.ipynb
  - Executive Summary
  - 파이프라인 설명 + 다이어그램
  - 학습 Loss curves
  - z 내부 fringe 이미지
  - 설계변수 sensitivity
  - LT 대비 검증 결과
  - 결론 + 다음 단계 (Phase D)

Export: Jupyter → PDF 또는 HTML
  → 경영진/팀 공유
```

#### 15.5.6 개발 효율 비교

```
                    Jupyter Only    Hybrid (v6)
─────────────────────────────────────────────────
실험 속도           ★★★★★          ★★★★★ (동일)
시각화              ★★★★★          ★★★★★ (동일)
재사용성             ★                ★★★★★
테스트 가능성        ★                ★★★★★
장시간 실행          ★★                ★★★★★
Production 배포     ✗                ★★★★★
경영진 보고         ★★★★★          ★★★★★ (동일)
Git 협업            ★★                ★★★★★

→ Hybrid가 Jupyter의 장점은 유지하면서 단점 해결
```

---

## 16. Design Studio UI

### 16.1 탭 구조

```
Nav: Summary | Detail | Explore

Summary (임원)
  - Executive Banner 5 KPI
  - Pareto front
  - Top-5 카드

Detail (엔지니어)
  - 역설계 실행
  - 후보 선택
  - COE 단면
  - 지표 상세

Explore (수동)
  - 4 슬라이더
  - 실시간 PSF
  - 물리 제약
```

### 16.2 Zustand State

```javascript
const useDesignStore = create((set, get) => ({
  spec: { mtf_min: 0.60, skew_max: 0.10, T_min: 0.60 },
  weights: { mtf: 0.4, T: 0.3, skew: 0.3 },
  baseline: { mtf: 0.42, skew: 0.28, T: 0.48 },
  explore_params: { delta_bm1: 0, delta_bm2: 0, w1: 10, w2: 10 },
  candidates: [],
  pareto_points: [],
  selected_id: null,
  hypervolume: null,
  pinn_loss: { helm: null, phase: null, BC: null, I: null },
  iter_current: 0,
  iter_total: 50,
  fno_ready: false,
  
  setSpec: (s) => set({ spec: {...get().spec, ...s} }),
  setExploreParams: (p) => set({ explore_params: {...get().explore_params, ...p} }),
  selectCandidate: (id) => set({ selected_id: id }),
}));
```

### 16.3 Executive Banner

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ FRR 개선  │ 선명도    │ 광량      │ 설계시간  │ Pareto   │
│ 7.1%     │ 0.047    │ 74%      │ 8초      │ 3/5      │
│ ↓-42%    │ ↓-83%    │ ↑+26pp   │ vs 833h  │ 달성     │
└──────────┴──────────┴──────────┴──────────┴──────────┘

Baseline:
  MTF = 0.42, skew = 0.28, T = 0.48
```

### 16.4 KPI 색상

| 지표 | 녹색 | 노란 | 빨간 |
|---|---|---|---|
| MTF@ridge | ≥ 60% | 45~60% | < 45% |
| Skewness | ≤ 0.10 | 0.10~0.20 | > 0.20 |
| 광량 T | ≥ 60% | 45~60% | < 45% |
| 크로스토크 | ≤ 10% | 10~20% | > 20% |

### 16.5 Explore 탭 주의

```javascript
// ✓ 올바른 바인딩
document.addEventListener('DOMContentLoaded', () => {
  ['esl0','esl1','esl2','esl3'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', exploreUpdate);
  });
});
```

---

## 17. 타일링 및 지문 이미지 시뮬레이션

### 17.1 PINN 출력 구조

```
PINN 도메인: 504μm × 40μm × 입사각 범위

단일 추론 = 7개 OPD PSF (중심 ±3 피치)
  입력: (δ₁, δ₂, w₁, w₂, θ)
  출력: PSF[7] (각 OPD 픽셀 강도)
```

### 17.2 Phase 1의 타일링 방식 (보고서 섹션 5.0 기반)

**Phase 1 원문**:
> "실제 센서 이미지 = Σ (각 행의 지문 × 해당 입사각의 BM PSF)"
> "각도별(θ=0°, 10°, 20°, 30°) 개별 PSF 적용 이미지 비교"

**핵심 아이디어**:
- 전체 센서를 동일 패턴 복사하는 단순 타일링이 아님
- **각 픽셀의 센서 중심으로부터 거리에 따라 입사각이 다름**
- **각 픽셀마다 해당 입사각의 PSF를 적용**

### 17.3 센서 기하학

```
센서: 30mm × 30mm = 417 × 417 픽셀 (72μm 피치)
스택 높이 (지문 접촉면 ~ OPD):
  AR:    ~300nm (무시)
  CG:    550μm
  Encap: 20μm
  BM+ILD+BM: 20μm (BM 두께 0.1 무시)
  ─────────────
  총합:  ~590μm

각 픽셀의 입사각:
  θ_pixel = arctan(distance / stack_height)
  
  stack_height = 590μm
  
  센서 중심 (row=208, col=208): 
    distance = 0 → θ = 0°
  
  센서 모서리 (row=0, col=0):
    distance = √((208·72)² + (208·72)²) ≈ 21.2mm = 21200μm
    θ = arctan(21200 / 590) ≈ 88° (전반사 영역)
  
실제 유효 영역:
  θ ≤ 41.1° 픽셀만 OPD 도달
  θ > 41.1°: 전반사 (CG 임계각)
  
  유효 반경 = stack_height × tan(41.1°)
           = 590 × 0.875
           ≈ 516μm
           ≈ 7 피치
  
  → 7 픽셀 타일링 구조의 물리적 근거
```

### 17.4 구현 알고리즘

```python
def simulate_fingerprint_image(
    params,
    fingerprint_raw,
    fno_surrogate,
    precompute_angles=[0, 5, 10, 15, 20, 25, 30, 35, 40],
):
    """
    Phase 1 방식 지문 이미지 시뮬레이션.
    
    Args:
        params: BMDesignParams
        fingerprint_raw: (417, 417) 원본 지문 이미지
        fno_surrogate: FNO 모델 (params, theta) → PSF[7]
        precompute_angles: 미리 계산할 입사각 리스트 (도)
    
    Returns:
        sensor_image: (417, 417) 시뮬레이션 이미지
    """
    import numpy as np
    
    sensor_size = 417
    pixel_size = 72.0  # μm
    stack_height = 590.0  # μm (CG 550 + Encap 20 + BM 영역 20)
    
    # 1. 입사각별 PSF 미리 계산 (FNO 호출 수 최소화)
    psf_by_angle = {}
    for theta_deg in precompute_angles:
        psf_7 = fno_surrogate(params, theta_deg)  # (7,)
        psf_by_angle[theta_deg] = psf_7
    
    # 2. 센서 중심 좌표
    center = sensor_size // 2  # 208
    
    # 3. 각 픽셀에 해당 PSF 적용
    sensor_image = np.zeros((sensor_size, sensor_size))
    
    for row in range(sensor_size):
        for col in range(sensor_size):
            # 센서 중심으로부터 거리
            dx = (col - center) * pixel_size
            dy = (row - center) * pixel_size
            distance = np.sqrt(dx**2 + dy**2)
            
            # 입사각 (도)
            theta_rad = np.arctan(distance / stack_height)
            theta_deg = np.degrees(theta_rad)
            
            # 임계각 초과 → 전반사 (신호 없음)
            if theta_deg > 41.1:
                sensor_image[row, col] = 0
                continue
            
            # 가장 가까운 precomputed 각도 선택
            closest_angle = min(
                precompute_angles,
                key=lambda a: abs(a - theta_deg)
            )
            psf = psf_by_angle[closest_angle]
            
            # 원본 지문에 PSF 합성곱 (7 OPD = 7 방향)
            # 중심 픽셀 기준 ±3 OPD 범위
            weighted_sum = 0.0
            for k in range(7):
                offset = k - 3  # -3, -2, -1, 0, 1, 2, 3
                src_row = row + offset
                if 0 <= src_row < sensor_size:
                    weighted_sum += fingerprint_raw[src_row, col] * psf[k]
            
            sensor_image[row, col] = weighted_sum
    
    return sensor_image
```

### 17.5 최적화된 구현 (Vectorized)

```python
def simulate_fingerprint_image_fast(
    params,
    fingerprint_raw,
    fno_surrogate,
    n_angle_bins=8,
):
    """
    Vectorized 구현 (빠름).
    
    각도를 N개 빈으로 나누고, 각 빈의 픽셀은 동일한 PSF.
    """
    import numpy as np
    
    sensor_size = 417
    pixel_size = 72.0
    stack_height = 590.0  # μm
    
    # 1. 각 픽셀의 입사각 계산 (vectorized)
    row_idx, col_idx = np.meshgrid(
        np.arange(sensor_size), np.arange(sensor_size), indexing='ij'
    )
    dx = (col_idx - sensor_size // 2) * pixel_size
    dy = (row_idx - sensor_size // 2) * pixel_size
    distance = np.sqrt(dx**2 + dy**2)
    theta_deg_map = np.degrees(np.arctan(distance / stack_height))
    
    # 2. 각도 binning
    max_theta = min(theta_deg_map.max(), 41.1)
    angle_edges = np.linspace(0, max_theta, n_angle_bins + 1)
    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    
    # 3. 각 bin의 대표 각도에서 PSF 계산
    psf_array = np.stack([
        fno_surrogate(params, theta_deg)
        for theta_deg in angle_centers
    ])  # (n_angle_bins, 7)
    
    # 4. 각 픽셀을 해당 bin에 할당
    bin_idx = np.digitize(theta_deg_map, angle_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_angle_bins - 1)
    
    # 5. 전반사 영역 mask
    valid_mask = theta_deg_map <= 41.1
    
    # 6. PSF 합성곱 적용
    sensor_image = np.zeros_like(fingerprint_raw, dtype=float)
    
    for k in range(7):
        offset = k - 3
        # 각 픽셀의 PSF 가중치
        weights = psf_array[bin_idx, k]  # (417, 417)
        
        # 지문 이미지 shift
        shifted = np.roll(fingerprint_raw, shift=offset, axis=0)
        
        # 누적
        sensor_image += weights * shifted
    
    sensor_image *= valid_mask
    
    return sensor_image
```

### 17.6 Design Studio에서의 표시

```
Detail/Explore 탭:
  - 원본 지문 (baseline, 블러 없음)
  - 현재 설계 적용 지문
  - 최적 설계 적용 지문
  - 3장 나란히 비교
  - 상관계수 정량 표시:
    corr_current = correlation(baseline, current)
    corr_optimal = correlation(baseline, optimal)
```

### 17.7 "왜 타일링이라 부르는가"

```
PINN 출력: 504μm × 7 OPD PSF (1개 단위)
  ↓
이게 "타일"
  ↓
전체 센서 = 타일 × 417 행 × 417 열
  ↓
BUT 각 타일의 PSF는 위치(=입사각)에 따라 다름

→ "각도별 PSF 타일링"
→ 단순 복사 아닌 "각도 맞춤 적용"
```

### 17.8 Phase 2 확장 (2D PSF)

```
Phase C: 1D PSF (x 방향만, 7개 OPD)
Phase 2: 2D PSF (x, y 모두, 7×7 OPD)
  - PINN 2D 확장 필요
  - 계산 비용 ~7배 증가
  - 실제 광학계 완전 재현
```

### 17.9 구현 시 주의사항

**성능**:
- 단순 loop: O(N²) × FNO 호출 = 417² × N ≈ 매우 느림
- Vectorized: 각도 bin 8~16개로 축소 → 수십 ms 내 가능

**경계 효과**:
- 센서 가장자리 픽셀: 이웃 없어서 PSF 적용 시 경계 처리
- 옵션: 0 padding, reflection padding, 또는 무시

**지문 데이터**:
- Baseline: 블러 없는 원본 지문 이미지
- 시뮬레이션 결과: 센서가 보는 이미지 (블러/비대칭 적용)
- 비교 metric: 상관계수, SSIM, MSE

### 17.10 노트북 템플릿 (v6 신규)

이 섹션은 Phase 1 노트북 방식을 계승하여, Phase C~E에서 사용할 **표준 노트북 템플릿**을 제공합니다.

#### 17.10.1 Phase C 개발 노트북 (5개)

**01. ASM LUT 생성**

```
notebooks/02_phase_c_development/01_asm_lut_generation.ipynb

목적: PINN L_phase target 생성 (z=40 경계)

셀 구조:
  Cell 1: Import & 환경 설정
  Cell 2: TMM 계산 (AR Gorilla DX 4층)
    - 입력: Phase 1 최적값 (34.6/25.9/20.7/169.5 nm)
    - 출력: t(θ), Δφ(θ) LUT
    - 시각화: 투과율 곡선, 위상 지연 곡선
  Cell 3: ASM 전파 (CG 550μm)
    - z=590 (AR 직후) → z=40 (PINN 입사)
    - 전파 거리: 550μm
  Cell 4: 복소장 시각화
    - |U|(x) at z=40, θ=0 (대칭 기대)
    - |U|(x) at z=40, θ=30 (비대칭 기대)
    - Re/Im heatmap (θ vs x)
  Cell 5: 검증
    - θ=0에서 평면파인지
    - 에너지 보존 확인
  Cell 6: NPZ 저장
    - data/asm_luts/incident_z40.npz

예상 시간: 1-2시간
주요 의존성:
  - backend/physics/tmm_calculator.py
  - backend/physics/asm_propagator.py

출력:
  - data/asm_luts/incident_z40.npz
  - notebooks/figures/asm_lut_verification.png
```

**02. PINN CPU 검증**

```
notebooks/02_phase_c_development/02_pinn_cpu_validation.ipynb

목적: 작은 모델로 구조 검증 (집 CPU)

셀 구조:
  Cell 1: Import
    - backend.core.pinn_model (또는 노트북에서 프로토타입)
    - backend.training.loss_functions
  Cell 2: Config (small)
    hidden_dim: 64, num_layers: 3, n_colloc: 2000, epochs: 500
  Cell 3: 모델 & optimizer
  Cell 4: 학습 루프 (실시간 시각화)
    - Loss curve (L_H, L_phase, L_BC)
    - 매 50 epoch마다 plt.draw()
  Cell 5: 중간 결과 시각화
    - z=10 단면 |U|(x)
    - z=40 단면 |U|(x) - ASM 매칭
    - z=20, z=40 BM 영역 |U| 값
  Cell 6: Red Flag 체크
    - 평면파 수렴?
    - BM 학습?
    - 설계변수 반응?
  Cell 7: 결론 & 다음 단계

예상 시간: 30분-1시간 (학습 포함)

성공 기준:
  ✓ Loss 발산 없이 감소
  ✓ Slit/BM 분화 조짐
  ✓ z 내부 약간의 fringe (aftermath Stage 2)

실패 시:
  → Loss 가중치 재조정
  → 네트워크 크기 증가
  → Collocation 밀집
```

**03. PINN 학습 모니터링**

```
notebooks/02_phase_c_development/03_pinn_training_monitor.ipynb

목적: GPU 장시간 학습 모니터링 (학습은 scripts/train_phase_c.py가 담당)

전제:
  - scripts/train_phase_c.py가 백그라운드 실행 중
  - 체크포인트가 checkpoints/ 또는 experiments/*/에 저장됨

셀 구조:
  Cell 1: 학습 로그 읽기
    !tail -50 experiments/latest/training.log
  Cell 2: Loss history 시각화
    - experiments/latest/training.log 파싱
    - plot loss curves (L_H, L_phase, L_BC, total)
  Cell 3: 최신 체크포인트 로드
    latest_ckpt = find_latest_checkpoint('experiments/latest/')
    model.load_state_dict(torch.load(latest_ckpt))
  Cell 4: 현재 학습 상태 시각화
    - z=10, z=20, z=40 단면 비교
    - 설계변수 변경 시 반응
  Cell 5: Red flag 히스토리
    - experiments/latest/red_flag_history.json 로드
    - 시계열 차트

갱신 빈도: 매 30분 (학습 진행 확인)

활용:
  - 학습 중단 여부 판단
  - 파라미터 조정 필요성 판단
  - 실시간 디버깅
```

**04. PINN 최종 평가**

```
notebooks/02_phase_c_development/04_pinn_evaluation.ipynb

목적: 학습 완료 후 정밀 검증

셀 구조:
  Cell 1: 최종 모델 로드
    model.load_state_dict(torch.load('checkpoints/phase_c_final.pt'))
    model.eval()
  
  Cell 2: BM 경계 검증
    - z=20 BM1 영역 |U| < 0.05 확인
    - z=40 BM2 영역 |U| < 0.05 확인
    - 통과율 계산
  
  Cell 3: z=40 ASM 매칭 검증
    - BM2 slit 내부에서 PINN vs ASM 비교
    - 상대 오차 < 10% 확인
  
  Cell 4: z 내부 회절 검증
    - z=10, 20, 30, 40 단면 |U|(x)
    - std > 0.1 (uniform 아님)
    - Fringe 패턴 시각적 확인
  
  Cell 5: 설계변수 sensitivity
    - δ_BM1: [-5, 0, 5] → PSF 7개 비교
    - w_1: [7, 10, 15] → PSF 7개 비교
    - 반응성 정량화
  
  Cell 6: PSF 계산 (7개 OPD)
    - 여러 입사각 θ=[0, 15, 30, 40]
    - compute_psf_7_opd() 호출
    - Skewness, MTF, throughput 계산
  
  Cell 7: LightTools 비교 (있다면)
    - 같은 파라미터 LT 실행
    - PSF 7개 비교
    - MSE 계산
  
  Cell 8: 최종 평가 보고서
    - 합격/불합격 판정
    - 다음 단계 (Phase D로 진행 가능?)

예상 시간: 2-3시간
의존성:
  - checkpoints/phase_c_final.pt
  - data/asm_luts/incident_z40.npz
  - (옵션) data/lt_results/
```

**05. Phase C 보고서**

```
notebooks/02_phase_c_development/06_phase_c_report.ipynb

목적: 경영진/팀 공유용 보고서 (Phase 1 방식 계승)

셀 구조 (스토리텔링 중심):
  [Executive Summary]
  Cell 1: Markdown - 한 페이지 요약
    - 목표
    - 달성 사항
    - 주요 지표 (MTF, Skewness, Throughput)
    - 다음 단계
  
  [문제 정의]
  Cell 2: Markdown - UDFPS 문제
    - FRR 저하 원인
    - BM 크로스토크 + 위상 왜곡
    - Phase 1 기반 확장 필요성
  
  [기술 접근]
  Cell 3: Markdown - PINN 아키텍처
  Cell 4: Figure - 파이프라인 다이어그램
    - AR → CG → PINN → PSF
  Cell 5: Markdown - Phase B 실패 교훈
  Cell 6: Markdown - Phase C 정석 접근
    - 4가지 Loss
    - BM~OPD 40μm
    - Buffer Zone
  
  [결과]
  Cell 7: Figure - Loss 수렴 곡선
  Cell 8: Figure - z 내부 회절 (4단면)
  Cell 9: Figure - 설계변수 sensitivity
  Cell 10: Figure - PSF 7개 (입사각별)
  Cell 11: Table - 지표 달성도
    - MTF@ridge: 목표 60% → 달성 ?
    - Skewness: 목표 0.10 → 달성 ?
    - Throughput: 목표 60% → 달성 ?
  
  [검증]
  Cell 12: Figure - LightTools 비교
  Cell 13: Figure - 지문 이미지 시뮬레이션
    - 기준 vs 현재 설계
  
  [결론]
  Cell 14: Markdown - 달성 요약
  Cell 15: Markdown - Phase D 계획
    - FNO 증류
    - BoTorch 역설계
    - 8초 설계 목표
  
  [Export]
  # Jupyter → PDF 또는 HTML
  # File > Download as > PDF via LaTeX

대상: CTO, 팀장, 경영진
포맷: PDF 또는 HTML
재사용: 기술보고서 작성 기반
```

#### 17.10.2 Phase D 노트북 (3개)

```
notebooks/03_phase_d_fno/

01_fno_distillation.ipynb
  - PINN에서 10,000 (params, PSF) 쌍 생성
  - FNO 학습 (SpectralConv1d)
  - 0.8ms/추론 달성
  - MSE < 1% (PINN 대비)

02_fno_validation.ipynb
  - FNO vs PINN 비교
  - FNO vs LightTools 비교
  - Speedup 측정

03_botorch_optimization.ipynb
  - qNEHVI 역설계
  - Pareto front 탐색
  - 8초 역설계 달성 확인
```

#### 17.10.3 Phase E 노트북 (2개)

```
notebooks/04_phase_e_integration/

01_e2e_pipeline.ipynb
  - 전체 파이프라인 통합 테스트
  - TMM → ASM → PINN → FNO → BoTorch
  - API 명세 검증

02_fingerprint_simulation.ipynb
  - Phase 1 지문 시뮬레이션 계승
  - 타일링 적용 (섹션 17)
  - Baseline vs 최적 설계 비교
```

#### 17.10.4 노트북 helpers

```python
# notebooks/helpers/visualization.py

"""
모든 노트북에서 재사용하는 시각화 함수.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_u_field_slice(model, z_values: list, device='cpu'):
    """
    여러 z 단면에서 |U|(x) 시각화.
    
    Args:
        model: PurePINN
        z_values: [0, 10, 20, 30, 40] 등
        device: 'cpu' or 'cuda'
    """
    fig, axes = plt.subplots(len(z_values), 1, figsize=(12, 2*len(z_values)))
    
    x = torch.linspace(0, 504, 500, device=device)
    
    with torch.no_grad():
        for i, z_val in enumerate(z_values):
            z = torch.full_like(x, z_val)
            d1 = torch.zeros_like(x)
            d2 = torch.zeros_like(x)
            w1 = torch.full_like(x, 10.0)
            w2 = torch.full_like(x, 10.0)
            sin_th = torch.zeros_like(x)
            cos_th = torch.ones_like(x)
            
            coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
            U = model(coords)
            amp = torch.sqrt(U[:, 0]**2 + U[:, 1]**2)
            
            axes[i].plot(x.cpu().numpy(), amp.cpu().numpy())
            axes[i].set_title(f'z = {z_val} μm')
            axes[i].set_xlabel('x (μm)')
            axes[i].set_ylabel('|U|')
    
    plt.tight_layout()
    return fig


def plot_design_sensitivity(model, param_name: str, values: list, device='cpu'):
    """
    설계변수 변화에 따른 PSF 변화 시각화.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for val in values:
        # ... (PSF 계산)
        # ax.plot(opd_indices, psf_7, label=f'{param_name}={val}')
    
    ax.legend()
    ax.set_xlabel('OPD Pixel Index')
    ax.set_ylabel('PSF Intensity')
    return fig


def plot_training_log(log_path: str):
    """
    training.log 파싱 & 시각화.
    """
    import re
    
    pattern = r'Epoch (\d+): L_H=([\d.e-]+), L_phase=([\d.e-]+), L_BC=([\d.e-]+)'
    
    epochs, L_H, L_phase, L_BC = [], [], [], []
    
    with open(log_path) as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                epochs.append(int(m.group(1)))
                L_H.append(float(m.group(2)))
                L_phase.append(float(m.group(3)))
                L_BC.append(float(m.group(4)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, L_H, label='L_Helmholtz', alpha=0.8)
    ax.plot(epochs, L_phase, label='L_phase', alpha=0.8)
    ax.plot(epochs, L_BC, label='L_BC', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
```

#### 17.10.5 Jupyter ↔ Python 변환 도구

**jupytext 사용** (선택):

```bash
pip install jupytext

# 노트북 ↔ .py 자동 동기화
jupytext --set-formats ipynb,py notebook.ipynb

# 이제 .ipynb 수정하면 .py 자동 업데이트
# Git에는 .py만 커밋 (깔끔한 diff)
```

**nbconvert** (내보내기):

```bash
# HTML 보고서
jupyter nbconvert --to html 06_phase_c_report.ipynb

# PDF (LaTeX 필요)
jupyter nbconvert --to pdf 06_phase_c_report.ipynb

# Python 스크립트 (참고용)
jupyter nbconvert --to python 02_pinn_cpu_validation.ipynb
```

---

## 18. 성공 기준 및 검증

### 18.1 Phase C 필수 기준

```
[1] BM 영역 |U| < 0.05
    → L_BC로 학습됨 (hard mask 없이)
    → z=20 (BM1), z=40 (BM2) 모두 확인

[2] Slit 영역 |U| ≈ ASM 결과 (±10%)
    → L_phase 정확히 매칭
    → z=40 BM2 slit 내부에서 ASM과 일치

[3] z 내부 회절 fringe
    → ILD (z=21~39), Encap (z=1~19)
    → std > 0.1 (uniform 아님)

[4] PDE residual < 0.1
    → Helmholtz 방정식 만족
    → 전체 도메인 [0, 40] 평균
```

### 18.2 권장 기준

```
[5] 설계변수 반응
    (δ_BM1, w_1) 변화 → PSF 변화
    → 파라메트릭 학습 작동

[6] LT 매칭 MSE < 20%
    L_I 사용 시
    → Sim-to-real 가능

[7] ASM 대비 차별화
    단순 ASM과 구별
    → 진짜 회절 학습
```

### 18.3 Phase 1 PoC 완료 기준

```
[ ] skewness < 0.10 (Phase 1 이미 달성 0.075)
[ ] 크로스토크 포착 (7피치 도메인)
[ ] PINN ↔ LightTools MSE < 10%
[ ] Pure PINN (hard mask 없음)
[ ] 역설계 8초
[ ] Design Studio 3탭
[ ] 지문 이미지 시뮬레이션
```

### 18.4 검증 스크립트

```python
# validate_phase_c.py
import torch
import numpy as np
from backend.core.pinn_model import PurePINN

def validate():
    model = PurePINN(hidden_dim=128, num_layers=4, num_freqs=48)
    ckpt = torch.load('checkpoints/phase_c_final.pt')
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    # 1. BM 영역 U≈0
    validate_bm_boundary(model)
    
    # 2. ASM 매칭
    validate_phase_boundary(model)
    
    # 3. z 내부 fringe
    validate_interior_fringe(model)
    
    # 4. PDE residual
    validate_pde_residual(model)
    
    # 5. 설계변수 반응
    validate_parametric_response(model)
    
    print("All validations passed!" if all_ok else "FAILED")
```

### 18.5 Red Flag 자동 감지 (Phase B 실패 재발 방지)

#### 18.5.1 목적

**Phase B는 학습 중 이상 신호를 놓쳐서 실패**:
- 평면파 수렴을 "정상 수렴"으로 오인
- Hard mask 효과를 "BM 학습 성공"으로 착각
- z 내부 uniform을 "매끄러운 해"로 합리화

**Phase C에서는 매 N epoch마다 자동 감지 → 조기 개입**

#### 18.5.2 Red Flag 체크 함수

```python
# backend/training/red_flag_detector.py

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RedFlagAlert:
    flag_name: str
    severity: str  # "warning" or "critical"
    value: float
    threshold: float
    message: str
    suggested_action: str


class RedFlagDetector:
    """
    학습 중 주기적으로 이상 신호 감지.
    Phase B 실패 패턴 조기 발견.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.history = []
    
    def check_all(self, model, epoch: int) -> List[RedFlagAlert]:
        """모든 red flag 검사. 감지된 alert 반환."""
        alerts = []
        
        alerts.extend(self._check_flat_wave_convergence(model))
        alerts.extend(self._check_bm_learning(model))
        alerts.extend(self._check_phase_boundary(model))
        alerts.extend(self._check_pde_residual(model))
        alerts.extend(self._check_design_response(model))
        
        # 로그 저장
        self.history.append({
            'epoch': epoch,
            'alerts': [a.__dict__ for a in alerts]
        })
        
        return alerts
    
    def _check_flat_wave_convergence(self, model) -> List[RedFlagAlert]:
        """
        [Red Flag 1] 평면파 수렴 감지
        
        증상: z 내부에서 |U| 거의 uniform
        원인: L_H 가중치 부족, PDE 학습 실패
        """
        model.eval()
        alerts = []
        
        with torch.no_grad():
            # z=10 (Encap 내부) 단면 확인
            x = torch.linspace(0, 504, 200, device=self.device)
            z = torch.full((200,), 10.0, device=self.device)
            d1 = torch.zeros(200, device=self.device)
            d2 = torch.zeros(200, device=self.device)
            w1 = torch.full((200,), 10.0, device=self.device)
            w2 = torch.full((200,), 10.0, device=self.device)
            sin_th = torch.zeros(200, device=self.device)
            cos_th = torch.ones(200, device=self.device)
            
            coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
            U = model(coords)
            
            amp = torch.sqrt(U[:, 0]**2 + U[:, 1]**2)
            std_over_mean = (amp.std() / (amp.mean() + 1e-8)).item()
        
        # 임계: std/mean < 0.01 → uniform 의심
        if std_over_mean < 0.01:
            alerts.append(RedFlagAlert(
                flag_name="flat_wave_convergence",
                severity="critical",
                value=std_over_mean,
                threshold=0.01,
                message=f"z=10에서 |U| uniform (std/mean={std_over_mean:.5f})",
                suggested_action="L_PDE 가중치 상향 (1.0 → 2.0), Curriculum Stage 2 확인"
            ))
        elif std_over_mean < 0.05:
            alerts.append(RedFlagAlert(
                flag_name="flat_wave_convergence",
                severity="warning",
                value=std_over_mean,
                threshold=0.05,
                message=f"z=10에서 |U| 변동 약함 (std/mean={std_over_mean:.5f})",
                suggested_action="L_PDE 가중치 점검"
            ))
        
        return alerts
    
    def _check_bm_learning(self, model) -> List[RedFlagAlert]:
        """
        [Red Flag 2] BM 학습 실패 감지
        
        증상: BM 영역 |U|가 slit 영역과 비슷
        원인: L_BC 가중치 부족 또는 학습 실패
        """
        model.eval()
        alerts = []
        
        with torch.no_grad():
            # BM2 평면 (z=40)에서 BM 영역 vs slit 영역
            x_slit = torch.full((50,), 36.0, device=self.device)  # 중심 slit
            x_bm = torch.full((50,), 0.0, device=self.device)     # BM 영역
            z = torch.full((50,), 40.0, device=self.device)
            
            d1 = torch.zeros(50, device=self.device)
            d2 = torch.zeros(50, device=self.device)
            w1 = torch.full((50,), 10.0, device=self.device)
            w2 = torch.full((50,), 10.0, device=self.device)
            sin_th = torch.zeros(50, device=self.device)
            cos_th = torch.ones(50, device=self.device)
            
            coords_slit = torch.stack([x_slit, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
            coords_bm = torch.stack([x_bm, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
            
            U_slit = model(coords_slit)
            U_bm = model(coords_bm)
            
            amp_slit = torch.sqrt(U_slit[:, 0]**2 + U_slit[:, 1]**2).mean().item()
            amp_bm = torch.sqrt(U_bm[:, 0]**2 + U_bm[:, 1]**2).mean().item()
            
            ratio = amp_bm / (amp_slit + 1e-8)
        
        # 임계: BM/slit 비율 > 0.3 → BM 학습 실패
        if ratio > 0.3:
            alerts.append(RedFlagAlert(
                flag_name="bm_learning_failure",
                severity="critical",
                value=ratio,
                threshold=0.3,
                message=f"BM 영역 |U|={amp_bm:.4f} vs slit |U|={amp_slit:.4f} (ratio={ratio:.3f})",
                suggested_action="L_BC 가중치 상향 (0.5 → 1.0), BM collocation 증가"
            ))
        elif ratio > 0.1:
            alerts.append(RedFlagAlert(
                flag_name="bm_learning_failure",
                severity="warning",
                value=ratio,
                threshold=0.1,
                message=f"BM 영역 |U| 부분 학습 (ratio={ratio:.3f})",
                suggested_action="학습 더 진행 또는 L_BC 확인"
            ))
        
        return alerts
    
    def _check_phase_boundary(self, model, asm_lut=None) -> List[RedFlagAlert]:
        """
        [Red Flag 3] 입사 경계 부정확
        
        증상: z=40 slit 내부에서 PINN이 ASM과 크게 다름
        원인: L_phase 가중치 부족
        """
        if asm_lut is None:
            return []
        
        model.eval()
        alerts = []
        
        with torch.no_grad():
            # z=40, slit 중심 샘플링
            x = torch.linspace(31, 41, 50, device=self.device)  # slit 범위
            z = torch.full((50,), 40.0, device=self.device)
            d1 = torch.zeros(50, device=self.device)
            d2 = torch.zeros(50, device=self.device)
            w1 = torch.full((50,), 10.0, device=self.device)
            w2 = torch.full((50,), 10.0, device=self.device)
            sin_th = torch.zeros(50, device=self.device)
            cos_th = torch.ones(50, device=self.device)
            
            coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
            
            U_pinn = model(coords)
            U_asm_re, U_asm_im = asm_lut.lookup(x, sin_th)
            
            diff_re = (U_pinn[:, 0] - U_asm_re).abs().mean().item()
            diff_im = (U_pinn[:, 1] - U_asm_im).abs().mean().item()
            
            asm_magnitude = torch.sqrt(U_asm_re**2 + U_asm_im**2).mean().item()
            rel_error = (diff_re + diff_im) / (asm_magnitude + 1e-8)
        
        if rel_error > 0.2:
            alerts.append(RedFlagAlert(
                flag_name="phase_boundary_inaccurate",
                severity="critical",
                value=rel_error,
                threshold=0.2,
                message=f"z=40에서 PINN vs ASM 상대 오차 {rel_error:.3f}",
                suggested_action="L_phase 가중치 상향 (0.5 → 1.0)"
            ))
        elif rel_error > 0.1:
            alerts.append(RedFlagAlert(
                flag_name="phase_boundary_inaccurate",
                severity="warning",
                value=rel_error,
                threshold=0.1,
                message=f"z=40 경계 오차 {rel_error:.3f}",
                suggested_action="학습 진행 상황 확인"
            ))
        
        return alerts
    
    def _check_pde_residual(self, model) -> List[RedFlagAlert]:
        """
        [Red Flag 4] PDE 잔차 과대
        
        증상: Helmholtz 방정식 불만족
        원인: PDE 학습 실패 또는 collocation 부족
        """
        alerts = []
        
        # 샘플 collocation에서 PDE residual 계산
        coords = hierarchical_collocation(500, self.device)
        coords.requires_grad_(True)
        
        U = model(coords)
        U_re = U[:, 0:1]
        U_im = U[:, 1:2]
        
        # Gradients
        grads_re = torch.autograd.grad(U_re.sum(), coords, create_graph=False)[0]
        grads_im = torch.autograd.grad(U_im.sum(), coords, create_graph=False)[0]
        
        U_re_xx = torch.autograd.grad(grads_re[:, 0].sum(), coords, create_graph=False)[0][:, 0]
        U_re_zz = torch.autograd.grad(grads_re[:, 1].sum(), coords, create_graph=False)[0][:, 1]
        U_im_xx = torch.autograd.grad(grads_im[:, 0].sum(), coords, create_graph=False)[0][:, 0]
        U_im_zz = torch.autograd.grad(grads_im[:, 1].sum(), coords, create_graph=False)[0][:, 1]
        
        k = 18.37
        res_re = U_re_xx + U_re_zz + k**2 * U_re.squeeze()
        res_im = U_im_xx + U_im_zz + k**2 * U_im.squeeze()
        
        residual_norm = torch.mean(res_re**2 + res_im**2).sqrt().item()
        
        if residual_norm > 10.0:
            alerts.append(RedFlagAlert(
                flag_name="pde_residual_high",
                severity="critical",
                value=residual_norm,
                threshold=10.0,
                message=f"PDE residual norm={residual_norm:.3f}",
                suggested_action="L_PDE 가중치 상향, collocation 증가"
            ))
        elif residual_norm > 1.0:
            alerts.append(RedFlagAlert(
                flag_name="pde_residual_high",
                severity="warning",
                value=residual_norm,
                threshold=1.0,
                message=f"PDE residual norm={residual_norm:.3f}",
                suggested_action="추가 학습으로 감소 확인"
            ))
        
        return alerts
    
    def _check_design_response(self, model) -> List[RedFlagAlert]:
        """
        [Red Flag 5] 설계변수 반응 부재
        
        증상: δ₁, w₁ 변경해도 PSF 변화 없음
        원인: 설계변수가 학습에 반영 안 됨
        """
        model.eval()
        alerts = []
        
        with torch.no_grad():
            # 두 가지 설계 비교
            x = torch.linspace(0, 504, 100, device=self.device)
            z = torch.zeros(100, device=self.device)
            w1_case = torch.full((100,), 10.0, device=self.device)
            w2_case = torch.full((100,), 10.0, device=self.device)
            
            # Case A: δ₁ = 0
            d1_a = torch.zeros(100, device=self.device)
            coords_a = torch.stack([
                x, z, d1_a, d1_a, w1_case, w2_case,
                torch.zeros(100, device=self.device),
                torch.ones(100, device=self.device)
            ], dim=1)
            U_a = model(coords_a)
            amp_a = torch.sqrt(U_a[:, 0]**2 + U_a[:, 1]**2)
            
            # Case B: δ₁ = 5
            d1_b = torch.full((100,), 5.0, device=self.device)
            coords_b = torch.stack([
                x, z, d1_b, d1_b, w1_case, w2_case,
                torch.zeros(100, device=self.device),
                torch.ones(100, device=self.device)
            ], dim=1)
            U_b = model(coords_b)
            amp_b = torch.sqrt(U_b[:, 0]**2 + U_b[:, 1]**2)
            
            response = (amp_a - amp_b).abs().mean().item()
            baseline = amp_a.mean().item()
            rel_response = response / (baseline + 1e-8)
        
        if rel_response < 0.01:
            alerts.append(RedFlagAlert(
                flag_name="no_design_response",
                severity="critical",
                value=rel_response,
                threshold=0.01,
                message=f"δ 변경 시 PSF 변화 없음 (rel={rel_response:.5f})",
                suggested_action="설계변수가 학습에 반영 안 됨. 네트워크 입력 확인"
            ))
        
        return alerts


# 사용 예시 (학습 루프 내)
def training_loop_with_red_flags(model, optimizer, config):
    detector = RedFlagDetector(device=config['device'])
    
    for epoch in range(config['epochs']):
        # ... 학습 ...
        
        # 매 100 epoch마다 체크
        if epoch % 100 == 0:
            alerts = detector.check_all(model, epoch)
            
            for alert in alerts:
                log.warning(
                    f"[{alert.severity.upper()}] {alert.flag_name}: "
                    f"{alert.message} → {alert.suggested_action}"
                )
            
            critical_count = sum(1 for a in alerts if a.severity == "critical")
            if critical_count >= 2:
                log.error(f"Critical alerts: {critical_count}. Consider intervention.")
                # 옵션: 학습 중단, Fallback 전략 실행
```

#### 18.5.3 학습 스크립트 통합

```python
# train_phase_c.py에서

from backend.training.red_flag_detector import RedFlagDetector

def train_phase_c(config):
    model = PurePINN(...)
    detector = RedFlagDetector(device=config['device'])
    
    for epoch in range(config['epochs']):
        # 학습 step
        loss = training_step(model, ...)
        
        # 100 epoch마다 검증
        if epoch % 100 == 0:
            alerts = detector.check_all(model, epoch)
            
            # Critical alert 2개 이상 → 자동 조정
            critical = [a for a in alerts if a.severity == 'critical']
            if len(critical) >= 2:
                log.error(f"Critical alerts at epoch {epoch}")
                for a in critical:
                    log.error(f"  - {a.flag_name}: {a.message}")
                
                # 자동 가중치 조정
                if any(a.flag_name == "flat_wave_convergence" for a in critical):
                    config['lambda_pde'] *= 2.0
                if any(a.flag_name == "bm_learning_failure" for a in critical):
                    config['lambda_bc'] *= 2.0
                
                log.info(f"Adjusted: lam_pde={config['lambda_pde']}, lam_bc={config['lambda_bc']}")
    
    # 최종 검증
    final_alerts = detector.check_all(model, config['epochs'])
    
    # 히스토리 저장
    import json
    with open('red_flag_history.json', 'w') as f:
        json.dump(detector.history, f, indent=2)
```

#### 18.5.4 알림 기준 요약

| Red Flag | Critical 임계 | Warning 임계 | 조치 |
|---|---|---|---|
| Flat wave convergence | std/mean < 0.01 | < 0.05 | L_PDE 상향 |
| BM learning failure | BM/slit > 0.3 | > 0.1 | L_BC 상향 |
| Phase boundary | rel_error > 0.2 | > 0.1 | L_phase 상향 |
| PDE residual | norm > 10 | > 1 | collocation 증가 |
| No design response | rel < 0.01 | - | 입력 구조 점검 |

**Critical 2개 이상 → 자동 가중치 조정 또는 학습 중단**

### 19.1 리스크

| 리스크 | 확률 | 영향 | 대응 |
|---|---|---|---|
| 40μm 수렴 실패 | 중 | 높 | L_BC 상향, 네트워크 확장, 30μm 축소 |
| 학습 시간 초과 | 중 | 중 | Epoch 조정 |
| BM 경계 불완전 | 중 | 중 | Collocation 밀집 |
| LT 확보 실패 | 높 | 낮 | ASM 임시 L_I |
| 내부 fringe 실패 | 낮 | 높 | PDE 가중치 증가 |

### 19.2 Fallback

**Level 1**: L_BC 강화 (가중치 상향, 0.5 → 1.0)

**Level 2**: 네트워크 확장 (hidden_dim 256, layers 6)

**Level 3**: 도메인 축소 (40 → 30μm, Phase 1 구조로 복귀)
- BM 근방만 PINN
- Encap + BM 전파만 ASM 추가
- Phase 1 검증된 스케일로 복귀

**Level 4**: Phase 1 구조 복원 (30μm + ASM)
- PINN: BM 근방 30μm만
- 나머지 ASM + 고전 회절 모델
- 안전한 복귀

**Level 5**: Soft mask 재도입 (sharpness 0.5, L_BC 병행)
- 최후의 수단
- "Hybrid constraint" 포지셔닝

---

## 20. Phase B 자산 관리

### 유지 (Phase C 재사용)

```
✓ backend/physics/tmm_calculator.py
✓ backend/physics/ar_coating/
✓ backend/physics/asm_propagator.py
✓ backend/physics/psf_metrics.py
✓ backend/harness/physical_validator.py
✓ backend/api/ (서버 구조)
✓ backend/data/
```

### 폐기

```
✗ checkpoints/phase_b_stage2.pt
✗ train_phase_b.py
✗ parametric_pinn.py의 use_bm_mask
```

### 신규 작성

```
+ backend/core/pinn_model.py (Pure 8D)
+ backend/training/train_phase_c.py ★
+ backend/training/loss_functions.py
+ backend/training/collocation_sampler.py
+ backend/training/curriculum.py
+ backend/training/validate_phase_c.py
+ backend/physics/asm_lut_generator.py (z=40 LUT)
+ backend/physics/boundary_conditions.py
```

---

## 21. 지금 바로 시작하기

### 21.1 집 CPU에서 (새벽)

```bash
cd udfps-pinn-platform

# 1. ASM LUT 생성 (PINN L_phase target)
python backend/physics/asm_lut_generator.py
# → data/asm_luts/incident_z40.npz

# 2. Pure PINN 모델 작성 지시
# (AI 에이전트에게 v4 섹션 13.4 기반으로 작성 요청)

# 3. 학습 스크립트 작성 지시
# (v4 섹션 6~9 기반)

# 4. Small config로 실행
python backend/training/train_phase_c.py --config small --device cpu

# 5. 검증
python backend/training/validate_phase_c.py

# 6. OK면 commit
git add backend/core/pinn_model.py backend/training/ backend/physics/asm_lut_generator.py
git commit -m "feat: Pure PINN Phase C structure (v4, BM~OPD 40μm)"
git push
```

### 21.2 회사 GPU에서 (아침)

```bash
git pull

# Full config 실행 (백그라운드)
nohup python backend/training/train_phase_c.py \
    --config full --device cuda > training.log 2>&1 &

tail -f training.log

# 병행: LightTools 수집
python backend/data/collect_lt_data.py --n 80

# 저녁에 검증
python backend/training/validate_phase_c.py

# 성공 시
git add checkpoints/phase_c_final.pt
git commit -m "feat: Phase C Pure PINN trained"
git push
```

### 21.3 다음 단계 (Phase D)

```bash
# FNO 증류
python backend/training/distill_fno.py \
    --pinn-checkpoint checkpoints/phase_c_final.pt \
    --n-samples 10000

# BoTorch 테스트
python backend/training/test_botorch.py

# 통합
uvicorn backend.api.main:app --reload --port 8000
```

---

## 22. 문서 요약 (한 페이지)

```
목표: PINN 기반 역설계 자동화 플랫폼

Phase 1 기반 확장:
  Phase 1: BM 근방 30μm PINN (성공, MTF 99.78%)
  Phase C: BM~OPD 40μm PINN (BM2부터 OPD까지)

파이프라인:
  AR(TMM) → CG 550μm(ASM) → PINN 40μm(BM+ILD+BM+Encap) → PSF 7개

PINN 구조 (정석):
  입력 8D (x, z, δ₁, δ₂, w₁, w₂, sin θ, cos θ)
  출력 복소 U
  Hard mask 없음, slit_dist 없음

Loss 4가지:
  L_Helmholtz (1.0) ← PDE, 주 신호
  L_phase (0.5) ← z=40, ASM 결과 매칭 (BM2 slit 내부)
  L_BC (0.5) ← BM1(z=20), BM2(z=40) U=0
  L_I (0.3) ← z=0, 측정/ASM (optional)

도메인:
  x [0, 504] μm (7피치, 크로스토크)
  z [0, 40] μm (BM~OPD)

z 좌표:
  OPD = 0
  BM1 = 20 (두께 무시)
  ILD = 20~40
  BM2 = 40 (= PINN 입사 경계)
  CG = 40~590
  AR = 590~590.3 (~300nm)

AR 코팅 (Phase 1 최적값 고정):
  SiO2 34.6nm / TiO2 25.9nm / SiO2 20.7nm / TiO2 169.5nm

Stack Height (타일링):
  590μm (CG 550 + Encap 20 + BM 영역 20)
  유효 반경 = 590 × tan(41.1°) ≈ 516μm ≈ 7 피치

실행:
  집 CPU (small, 30분) → 구조 검증
  회사 GPU (full, 4-8시간) → 본격 학습

성공:
  BM |U| < 0.05 (z=20, z=40)
  Slit |U| ≈ ASM (z=40)
  z 내부 fringe (ILD, Encap)
  설계변수 반응

금지:
  Hard mask, slit_dist, L_H < 0.5
  도메인 축소/확장, L_BC 제거
  결과 정규화
```

---

**v4 최종본: Phase 1 성공 기반 + 정석 PINN + Phase B 방지 + 실측 물리 기반**

이 문서 하나로 Phase C부터 Phase E까지 끝까지 구현 가능.

**지금 할 일**: 집 CPU에서 ASM LUT(z=40) 생성 → Small config 학습 → 아침 검토.
