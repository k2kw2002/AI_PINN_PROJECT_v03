# ============================================================
# UDFPS PINN 플랫폼 v03 - 환경 설정 스크립트
# ============================================================
# 사용법:
#   .\setup_env.ps1           # 기본 (CPU 버전 PyTorch)
#   .\setup_env.ps1 -GPU      # GPU 버전 (CUDA 12.1)
#   .\setup_env.ps1 -GPU -CudaVersion 118   # CUDA 11.8
#
# PowerShell 실행 정책 오류 시:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# ============================================================

param(
    [switch]$GPU,
    [string]$CudaVersion = "121"
)

# 색상 함수
function Write-Section {
    param([string]$Message)
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "  $Message" -ForegroundColor Gray
}

# ============================================================
# 시작
# ============================================================
Write-Section "UDFPS PINN 플랫폼 v03 - 환경 설정 시작"

$startTime = Get-Date
Write-Info "시작 시간: $startTime"
Write-Info "작업 경로: $(Get-Location)"

if ($GPU) {
    Write-Info "모드: GPU (CUDA $CudaVersion)"
} else {
    Write-Info "모드: CPU"
}

# ============================================================
# 1. Python 버전 확인
# ============================================================
Write-Step "1. Python 버전 확인"

try {
    $pythonVersion = python --version 2>&1
    Write-Info $pythonVersion
    
    # Python 3.10+ 확인
    $versionNum = [version]($pythonVersion -replace 'Python ', '')
    if ($versionNum -lt [version]"3.10.0") {
        Write-Host "✗ Python 3.10 이상 필요 (현재: $pythonVersion)" -ForegroundColor Red
        Write-Host "  https://www.python.org/downloads/ 에서 최신 버전 설치" -ForegroundColor Red
        exit 1
    }
    Write-Success "Python 버전 OK"
}
catch {
    Write-Host "✗ Python이 설치되지 않았습니다" -ForegroundColor Red
    Write-Host "  https://www.python.org/downloads/ 에서 설치 후 재실행" -ForegroundColor Red
    exit 1
}

# ============================================================
# 2. venv 생성
# ============================================================
Write-Step "2. venv 가상환경 생성"

if (Test-Path "venv") {
    Write-Info "기존 venv 발견"
    $response = Read-Host "  덮어쓸까요? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Info "기존 venv 삭제 중..."
        Remove-Item -Recurse -Force venv
        python -m venv venv
        Write-Success "venv 재생성 완료"
    } else {
        Write-Info "기존 venv 사용"
    }
} else {
    python -m venv venv
    Write-Success "venv 생성 완료"
}

# ============================================================
# 3. venv 활성화
# ============================================================
Write-Step "3. venv 활성화"

try {
    .\venv\Scripts\Activate.ps1
    Write-Success "venv 활성화 완료"
    Write-Info "Python 경로: $(Get-Command python | Select-Object -ExpandProperty Source)"
}
catch {
    Write-Host "✗ venv 활성화 실패" -ForegroundColor Red
    Write-Host "  PowerShell 실행 정책 오류일 수 있습니다." -ForegroundColor Red
    Write-Host "  관리자 권한 PowerShell에서 실행:" -ForegroundColor Yellow
    Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}

# ============================================================
# 4. pip 업그레이드
# ============================================================
Write-Step "4. pip 업그레이드"

python -m pip install --upgrade pip --quiet
Write-Success "pip 업그레이드 완료"

# ============================================================
# 5. PyTorch 설치
# ============================================================
Write-Step "5. PyTorch 설치"

if ($GPU) {
    Write-Info "GPU 버전 설치 (CUDA $CudaVersion)..."
    Write-Info "다운로드 시간: 5-10분 (약 2GB)"
    
    switch ($CudaVersion) {
        "121" {
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
        "118" {
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        }
        default {
            Write-Host "✗ 지원하지 않는 CUDA 버전: $CudaVersion" -ForegroundColor Red
            Write-Info "지원: 121 (CUDA 12.1), 118 (CUDA 11.8)"
            exit 1
        }
    }
} else {
    Write-Info "CPU 버전 설치..."
    Write-Info "다운로드 시간: 3-5분 (약 200MB)"
    pip install torch torchvision torchaudio
}

Write-Success "PyTorch 설치 완료"

# ============================================================
# 6. 과학 계산 패키지
# ============================================================
Write-Step "6. 과학 계산 패키지 설치"

pip install numpy scipy --quiet
Write-Success "NumPy, SciPy 설치 완료"

# ============================================================
# 7. 물리 계산 (TMM)
# ============================================================
Write-Step "7. 물리 계산 패키지"

pip install tmm --quiet
Write-Success "TMM 설치 완료 (AR 코팅 계산)"

# ============================================================
# 8. 시각화
# ============================================================
Write-Step "8. 시각화 패키지"

pip install matplotlib seaborn --quiet
Write-Success "Matplotlib, Seaborn 설치 완료"

# ============================================================
# 9. Jupyter
# ============================================================
Write-Step "9. Jupyter 환경"

pip install jupyter jupyterlab ipywidgets --quiet
Write-Success "Jupyter 설치 완료"

# ============================================================
# 10. 최적화 (BoTorch)
# ============================================================
Write-Step "10. 최적화 패키지 (Phase D)"

pip install botorch gpytorch --quiet
Write-Success "BoTorch, GPyTorch 설치 완료"

# ============================================================
# 11. 웹 서비스 (FastAPI, Phase E)
# ============================================================
Write-Step "11. 웹 서비스 패키지 (Phase E)"

pip install fastapi uvicorn pydantic --quiet
Write-Success "FastAPI, Uvicorn, Pydantic 설치 완료"

# ============================================================
# 12. 설정 관리 (YAML)
# ============================================================
Write-Step "12. 설정 관리"

pip install pyyaml --quiet
Write-Success "PyYAML 설치 완료"

# ============================================================
# 13. 테스트 도구
# ============================================================
Write-Step "13. 테스트 도구"

pip install pytest pytest-cov --quiet
Write-Success "pytest 설치 완료"

# ============================================================
# 14. 개발 도구 (선택)
# ============================================================
Write-Step "14. 개발 도구"

pip install nbstripout --quiet
Write-Success "nbstripout 설치 완료 (Jupyter Git 관리)"

# ============================================================
# 설치 확인
# ============================================================
Write-Section "설치 확인"

$packages = @(
    @{Name="PyTorch"; Import="torch"; Version="torch.__version__"},
    @{Name="NumPy"; Import="numpy"; Version="numpy.__version__"},
    @{Name="SciPy"; Import="scipy"; Version="scipy.__version__"},
    @{Name="TMM"; Import="tmm"; Version="'OK'"},
    @{Name="Matplotlib"; Import="matplotlib"; Version="matplotlib.__version__"},
    @{Name="Jupyter"; Import="jupyter"; Version="'OK'"},
    @{Name="BoTorch"; Import="botorch"; Version="botorch.__version__"},
    @{Name="FastAPI"; Import="fastapi"; Version="fastapi.__version__"},
    @{Name="PyYAML"; Import="yaml"; Version="yaml.__version__"},
    @{Name="pytest"; Import="pytest"; Version="pytest.__version__"}
)

$allOk = $true
foreach ($pkg in $packages) {
    try {
        $version = python -c "import $($pkg.Import); print($($pkg.Version))" 2>&1
        Write-Host "  ✓ $($pkg.Name): $version" -ForegroundColor Green
    }
    catch {
        Write-Host "  ✗ $($pkg.Name): 실패" -ForegroundColor Red
        $allOk = $false
    }
}

# CUDA 확인 (GPU 모드인 경우)
if ($GPU) {
    Write-Host "`n  GPU/CUDA 상태:" -ForegroundColor Yellow
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
    if ($cudaAvailable -eq "True") {
        $cudaVersion = python -c "import torch; print(torch.version.cuda)" 2>&1
        $deviceName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
        Write-Host "    ✓ CUDA available: True" -ForegroundColor Green
        Write-Host "    ✓ CUDA version: $cudaVersion" -ForegroundColor Green
        Write-Host "    ✓ GPU: $deviceName" -ForegroundColor Green
    } else {
        Write-Host "    ✗ CUDA 사용 불가" -ForegroundColor Red
        Write-Host "    nvidia-smi 명령으로 드라이버 확인 필요" -ForegroundColor Yellow
        $allOk = $false
    }
}

# ============================================================
# 완료
# ============================================================
Write-Section "환경 설정 완료"

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Info "소요 시간: $($duration.ToString('mm\:ss'))"

if ($allOk) {
    Write-Host "`n✓ 모든 패키지 설치 완료!" -ForegroundColor Green
    
    Write-Host "`n다음 단계:" -ForegroundColor Cyan
    Write-Host "  1. v6 문서 배치:" -ForegroundColor White
    Write-Host "     mkdir docs" -ForegroundColor Gray
    Write-Host "     (udfps_pinn_master_guide_v6.md를 docs/ 로 복사)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Git 초기화:" -ForegroundColor White
    Write-Host "     git init" -ForegroundColor Gray
    Write-Host "     git remote add origin https://github.com/k2kw2002/AI_PINN_PROJECT_v03.git" -ForegroundColor Gray
    Write-Host "     git branch -M main" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. VS Code 열기:" -ForegroundColor White
    Write-Host "     code ." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  4. Claude Code 실행 후 첫 메시지 전달" -ForegroundColor White
} else {
    Write-Host "`n✗ 일부 패키지 설치 실패" -ForegroundColor Red
    Write-Host "  오류 메시지 확인 후 개별 재시도" -ForegroundColor Yellow
}

Write-Host ""
