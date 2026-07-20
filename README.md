# IronFlow Model_LAB

IronFlow Model_LAB은 로컬 PC에서 비전 모델 실험을 구성하고 원격 GPU/Vast.ai에서 실행·수집하기 위한 실험 플랫폼입니다.

이 저장소에는 소스 코드, 설정, 스크립트와 문서만 포함됩니다. 사전학습 모델, 데이터셋, 실험 결과, SSH 키와 사용자별 서버 접속 정보는 Git 저장소에 포함하지 않습니다.

## 가장 빠른 시작

사전학습 모델까지 포함된 완성형 배포본이 필요하면 GitHub의 **Releases**에서 `Model_LAB_portable_20260715.zip`을 내려받으세요.

1. ZIP을 짧은 영문 경로에 압축 해제합니다.
2. `setup_first.bat`을 실행합니다.
3. `run_gui.bat --check`로 설치 상태를 확인합니다.
4. `run_gui.bat`으로 GUI를 실행합니다.

자세한 내용은 [처음 실행 및 Vast.ai 실험 가이드](docs/README_FIRST_RUN_KO.md)를 참고하세요.

## 소스 저장소에서 실행

요구 사항:

- Windows 10/11
- Python 3.12 권장
- NVIDIA GPU 작업은 호환되는 CUDA/PyTorch 환경 필요

```powershell
git clone <repository-url>
cd Model_LAB
.\setup.bat
.\run_gui.bat --check
.\run_gui.bat
```

소스 저장소에는 사전학습 가중치가 없으므로 모델에 따라 별도 준비가 필요합니다. 완성형 실행 환경은 Release 배포본을 사용하는 것이 가장 간단합니다.

## 주요 기능

- Tkinter 기반 로컬 GUI
- 설정 기반 detection/classification 실험 구성
- 로컬 및 SSH 원격 실행 수명주기 관리
- Vast.ai 준비·업로드·실행·상태 확인·결과 수집 스크립트
- SQLite 기반 실행 이력과 표준 결과 산출물
- 모델별 adapter와 호환성 설정

## 저장소 구조

```text
configs/                 실험, 모델, 서버 설정 템플릿
docs/                    사용자·개발 문서
models/                  모델 adapter 관련 저장 위치
runs/                    로컬 실행 디렉터리(내용은 Git 제외)
scripts/                 데이터 및 원격 실행 보조 스크립트
src/ironflow_exp/        애플리케이션 소스
pyproject.toml           Python 패키지 및 의존성 정의
setup.bat                Windows 환경 설치
run_gui.bat              GUI 실행 및 점검
```

## 저장소에 올리면 안 되는 항목

- `models/checkpoints/` 아래의 `.pt`, `.pth` 등 모델 가중치
- `runs/` 아래의 실험 결과, 로그, SQLite 파일
- 데이터셋과 사용자 이미지
- `.env`, API 키, SSH 개인키
- 사용자별 서버 프로필과 실제 접속 정보
- `.venv`, Python 캐시와 IDE 개인 설정

커밋 전에는 다음 명령으로 포함 파일을 확인하세요.

```powershell
git status
git diff --cached
```

## 문서

- [처음 실행 및 Vast.ai 실험 가이드](docs/README_FIRST_RUN_KO.md)
- [휴대용 배포본 구성 안내](docs/DISTRIBUTION_GUIDE_KO.md)
- [사용자 설명서](docs/user_manual_ko.md)
- [모델 가중치/체크포인트 정책](docs/model_weight_checkpoint_policy_ko.md)
- [아키텍처](docs/architecture.md)
- [설정 스키마](docs/config_schema.md)

## 보안

실제 자격 증명이나 개인키는 이슈나 커밋에 첨부하지 마세요. 보안 문제를 발견한 경우 공개 이슈 대신 저장소 소유자에게 비공개로 알려주세요.

## 라이선스

현재 이 저장소에는 명시적인 오픈소스 라이선스가 부여되지 않았습니다. 따라서 별도 허가 없이 복제, 수정 또는 재배포할 수 없습니다. 공개 오픈소스로 운영하려면 소유자가 MIT, Apache-2.0 등 적절한 라이선스를 선택해 `LICENSE` 파일을 추가해야 합니다.

사전학습 모델과 외부 프레임워크에는 각각의 원저작자 라이선스가 별도로 적용될 수 있습니다.
