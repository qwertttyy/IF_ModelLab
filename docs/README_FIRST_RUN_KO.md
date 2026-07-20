# Model_LAB 처음 실행 및 Vast.ai 실험 가이드

배포본: `Model_LAB_portable_20260715`  
대상: Windows 로컬 PC + Vast.ai Linux GPU 인스턴스  
기준일: 2026-07-15

이 문서 하나로 설치, 데이터셋 복원, Vast.ai 연결, 실험 실행과 결과 수집까지 진행할 수 있다.

## 1. 전체 작업 흐름

```text
배포 폴더 압축 해제
-> 최초 설치 및 점검
-> 로컬 SSH 공개키 준비
-> Vast.ai 인스턴스 생성 및 공개키 등록
-> 데이터셋을 Vast 서버에 다운로드/복원
-> GUI에 SSH 명령과 Remote Root 입력
-> Experiments에서 실험을 queue에 추가
-> Preview 확인 후 실행
-> collected 확인 후 결과 검토
```

| 시점 | 해야 하는 일 |
|---|---|
| 새 PC에서 한 번 | `setup_first.bat`, SSH 키 준비 |
| 새 Vast 인스턴스마다 | 공개키 등록, 새 SSH 명령 적용, 데이터 복원 |
| 실험마다 | 실험 선택, 경로/조건 확인, Preview, 실행, 결과 확인 |

## 2. 폴더 구성

```text
Model_LAB_portable_20260715/
  README_FIRST_RUN_KO.md
  setup_first.bat
  run_gui.bat
  vision_experiment_platform/
    configs/                 # 실험, 모델, 서버 설정
    docs/                    # 상세 운영 문서
    models/                  # 사전학습 모델과 manifest
    scripts/                 # 데이터 복원, 배포 점검 도구
    src/                     # 프로그램 소스
    runs/e/                  # 수집된 실험 결과
    runs/w/                  # 실행 작업 공간
    runs/user_datasets/      # 로컬 사용자 데이터 공간
    setup.bat
    run_gui.bat
    pyproject.toml
```

기존 사용자의 결과, 데이터셋, SQLite 이력, `.venv`, SSH 키는 포함되지 않는다. 새 결과는 `vision_experiment_platform/runs`에 생성된다.

## 3. 권장 환경

### 로컬 PC

- Windows 10/11 64-bit
- 인터넷 연결
- 배포본 약 1.3GB 외에 Python 환경과 결과 저장 여유 공간
- 권장 경로: `D:\Model_LAB_portable_20260715`

긴 경로, OneDrive 동기화 폴더, 한글/특수문자가 많은 경로는 피하는 것이 안전하다.

### Vast.ai

- NVIDIA GPU와 CUDA/PyTorch 계열 이미지
- SSH 접속 가능
- `/workspace`에 데이터, 환경, 결과를 저장할 충분한 디스크
- balanced 기본 설정 기준 장비: RTX 5090 32GB

데이터와 산출물을 고려해 50GB 이상 여유 공간을 권장한다.

## 4. 최초 설치

파일 탐색기에서 `setup_first.bat`을 실행하거나 PowerShell에서 다음을 실행한다.

```powershell
Set-Location -LiteralPath 'D:\Model_LAB_portable_20260715'
.\setup_first.bat
```

설치 스크립트의 역할:

- Python 3.12 탐색 또는 설치
- Conda가 있으면 `IRONFLOW_VISION` 환경 생성
- Conda가 없으면 `vision_experiment_platform/.venv` 생성
- detection/classification/timm/YOLO/augmentation 의존성 설치
- `gdown`, `polars`, `torchvision`, `albumentations` 등 import 검증

패키지가 손상됐거나 재설치가 필요할 때만 사용한다.

```powershell
.\setup_first.bat --force
```

설치 점검:

```powershell
.\run_gui.bat --check
```

정상 출력:

```text
[IronFlow] GUI environment check passed.
```

## 5. GUI 실행과 탭

```powershell
.\run_gui.bat
```

| 탭 | 역할 |
|---|---|
| `Experiments` | detector/classifier/조합 실험 선택 |
| `Run` | 최대 12개 queue, Vast SSH, 데이터 경로, 실행 옵션 관리 |
| `Results` | 수집된 결과와 지표 조회 |
| `Advanced` | 세부 실행 및 검증 기능 |

처음 실행했을 때 Run queue가 비어 있는 것은 정상이다. `Experiments`에서 실험을 선택하고 `Add Selected to Queue`를 눌러야 한다.

## 6. SSH 키 준비

Vast.ai에는 공개키만 등록하고 개인키는 사용자 PC에만 둔다.

### GUI 권장 방법

1. `Run` 탭에서 `Copy Public Key`를 누른다.
2. 키가 없으면 프로그램 안내에 따라 ed25519 키를 만든다.
3. 복사된 `ssh-ed25519 ...` 한 줄을 Vast.ai에 등록한다.

기본 위치:

```text
C:\Users\<사용자>\.ssh\id_ed25519      # 개인키: 공유 금지
C:\Users\<사용자>\.ssh\id_ed25519.pub  # 공개키: Vast에 등록
```

수동 생성:

```powershell
ssh-keygen -t ed25519 -C "ironflow-vast"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

개인키를 배포 폴더, Drive, Git, 메신저에 넣지 않는다.

## 7. Vast.ai 인스턴스 연결

### 공개키 등록

1. Vast.ai에서 인스턴스를 생성한다.
2. `Manage SSH Keys for Instance`를 연다.
3. `New SSH Key`에 공개키를 붙여넣고 `ADD SSH KEY`를 누른다.
4. `Instance SSH Keys`에 표시되는지 확인한다.

### GUI에 Direct SSH 적용

Vast의 `Direct ssh connect` 명령을 복사한다.

```text
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080
```

`Run` 탭에서:

1. `Vast SSH`에 명령 전체를 붙여넣는다.
2. `Use Command`를 누른다.
3. Host, Port, User가 현재 인스턴스 값인지 확인한다.

새 인스턴스마다 IP/port가 바뀔 수 있으므로 이전 명령을 재사용하지 않는다. Direct가 불가능할 때만 Proxy SSH를 사용한다.

연결 테스트:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" `
  -o IdentitiesOnly=yes `
  -o StrictHostKeyChecking=accept-new `
  -p <PORT> root@<HOST> "echo SSH_OK"
```

`SSH_OK`가 출력되면 정상이다. 인스턴스 재생성 후 host key 충돌이 나면:

```powershell
ssh-keygen -R "[<HOST>]:<PORT>"
```

`Permission denied (publickey)`는 현재 PC의 공개키가 인스턴스에 없거나 잘못된 개인키를 사용한 경우가 대부분이다.

## 8. 데이터셋 규칙

데이터셋은 배포본에 포함하지 않는다. Google Drive의 tar-parts release를 Vast가 받아 다음 위치에 복원한다.

```text
/workspace/ironflow/prestaged/<DATASET_ID>
```

Drive release 표준 구성:

```text
manifest.json
restore_commands.txt
part-0000.bin 또는 part-000.bin
part-0001.bin                  # 필요한 경우
README.md                      # release에 따라 존재
```

`part-*.bin`은 zip이 아니라 압축하지 않은 tar를 분할한 조각이다. 각 part를 따로 `tar -xf` 하면 안 된다.

### Split 오염 방지

- 학습: `train`
- 모델 선택과 early stopping: `val`
- 최종 예측과 성능 평가: `test`
- `test`를 train/val에 섞지 않는다.

Classification 구조:

```text
classifier_mbt 또는 classifier_av 또는 classifier_all/
  images 또는 crops/
    train/<class_name>/*
    val/<class_name>/*
    test/<class_name>/*
```

Detection 구조:

```text
detector.../
  images/{train,val,test}/*
  labels/{train,val,test}/*.txt
  coco/annotations/instances_{train,val,test}.json
```

- YOLO detector는 YOLO txt 라벨을 사용한다.
- D-FINE/DETR/ViT detector의 native wrapper는 COCO JSON을 사용한다.
- classifier는 클래스별 폴더를 사용한다.
- 실험마다 요구하는 dataset ID와 `images`/`crops` variant가 다를 수 있다.

## 9. 데이터 다운로드와 복원

### 권장: SHA256 검증 포함 스크립트

현재 통합 원본 데이터셋 예시:

```text
DATASET_ID=tank_armor_prepared_v20260703_original_classifier_all_v1
DRIVE_URL=https://drive.google.com/drive/folders/YOUR_PREPARED_DATASET_FOLDER_ID
```

PowerShell에서 복원 스크립트를 Vast로 전송한다.

```powershell
Set-Location -LiteralPath 'D:\Model_LAB_portable_20260715'

scp -i "$env:USERPROFILE\.ssh\id_ed25519" `
  -o IdentitiesOnly=yes `
  -o StrictHostKeyChecking=accept-new `
  -P <PORT> `
  ".\vision_experiment_platform\scripts\restore_drive_dataset_release.py" `
  root@<HOST>:/workspace/restore_drive_dataset_release.py
```

복원 실행:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" `
  -o IdentitiesOnly=yes `
  -o StrictHostKeyChecking=accept-new `
  -p <PORT> root@<HOST> `
  "python3 /workspace/restore_drive_dataset_release.py --release-folder-url 'https://drive.google.com/drive/folders/YOUR_PREPARED_DATASET_FOLDER_ID' --download-dir '/workspace/ironflow/drive_dataset_parts/tank_armor_prepared_v20260703_original_classifier_all_v1' --extract-to '/workspace/ironflow/prestaged'"
```

스크립트가 하는 일:

- `gdown` 설치와 Drive 폴더 다운로드
- manifest 순서대로 part 결합
- part별 및 전체 tar SHA256 검증
- tar 경로 이탈 방지 검사 후 복원
- GUI에 넣을 `expected_remote_root` 출력

이미 part를 모두 받았다면 Vast 안에서 `--skip-download`를 사용할 수 있다. 기존 다운로드 디렉터리를 지우고 새로 받을 때만 `--overwrite-download-dir`를 사용한다.

### 수동 복원

다음은 SSH로 접속한 Vast Linux shell에서 실행한다. Windows PowerShell 프롬프트에 직접 입력하지 않는다.

```bash
DATASET_ID='tank_armor_prepared_v20260703_original_classifier_all_v1'
DRIVE_URL='https://drive.google.com/drive/folders/YOUR_PREPARED_DATASET_FOLDER_ID'
PARTS_DIR="/workspace/ironflow/drive_dataset_parts/$DATASET_ID"

python3 -m pip install -q gdown
mkdir -p "$PARTS_DIR" /workspace/ironflow/prestaged
gdown --folder "$DRIVE_URL" -O "$PARTS_DIR"
cd "$PARTS_DIR"
ls -lh part-*.bin manifest.json
cat part-*.bin > "ironflow_dataset_${DATASET_ID}.tar"
tar -xf "ironflow_dataset_${DATASET_ID}.tar" -C /workspace/ironflow/prestaged
ls "/workspace/ironflow/prestaged/$DATASET_ID"
```

수동 방식은 자동 checksum 검증이 없으므로 권장 스크립트보다 안전성이 낮다.

### 복원 확인

Vast shell에서:

```bash
DATASET_ROOT='/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1'
test -d "$DATASET_ROOT" && echo DATASET_ROOT_OK || echo DATASET_ROOT_MISSING
find "$DATASET_ROOT" -maxdepth 3 -type d | sort | head -n 80
du -sh "$DATASET_ROOT"
```

GUI 설정:

| 항목 | 값 |
|---|---|
| `Dataset Source` | `Remote pre-staged path` |
| `Remote Root` | `/workspace/ironflow/prestaged/<DATASET_ID>` |

## 10. 실험 실행

### Queue 추가

1. `Experiments`에서 실험을 선택한다.
2. `Add Selected to Queue`를 누른다.
3. `Run` 탭에서 Config, Model, Options를 확인한다.

queue는 최대 12개다. 반복 실험은 고유 experiment ID와 DB 경로를 사용해야 덮어쓰기를 피할 수 있다.

### 실행 전 체크

```text
[ ] Vast 인스턴스가 Running이다.
[ ] 현재 Direct SSH 명령을 Use Command로 적용했다.
[ ] Native wrappers와 Native Params가 대상 모델/GPU에 맞다.
[ ] Dataset Source가 Remote pre-staged path다.
[ ] Remote Root가 실제 복원 경로와 정확히 같다.
[ ] Config와 DB가 해당 실험용이다.
[ ] Run Options가 의도한 비교 조건이다.
[ ] Preview에서 test가 train 입력으로 섞이지 않는다.
```

Run Options를 비우면 locked config와 native defaults가 적용된다. 값을 입력하면 config를 override할 수 있으므로 비교 실험에서는 임의로 채우지 않는다.

`Replace existing`은 같은 experiment ID의 작업을 교체하는 재실행 옵션이다. 사용 전 기존 결과를 확인하고 Preview로 대상 경로를 점검한다.

### 실행 버튼

- `Preview`: GPU 작업 없이 effective config와 실행 계획 확인
- `Run Selected`: 선택한 queue 실행
- `Run All`: pending queue 순차 실행
- `Stop`: 현재 작업 중단 요청

첫 PC와 첫 인스턴스에서는 작은 smoke 또는 빠른 모델 하나로 전체 흐름을 검증한 뒤 긴 queue를 실행하는 것이 안전하다.

### 실행 단계

```text
install deps -> prepare -> bootstrap -> upload -> run -> collect
```

| 단계 | 역할 |
|---|---|
| `install deps` | 원격 Python 의존성 설치/확인 |
| `prepare` | effective config와 실행 계획 생성 |
| `bootstrap` | 원격 workspace 준비 |
| `upload` | 코드, config, 필요한 weight 전송 |
| `run` | 학습, test 예측, 지표 생성 |
| `collect` | 결과, summary, preview, checkpoint를 로컬로 수집 |

새 인스턴스의 첫 실행은 dependency 설치와 upload 때문에 이후보다 오래 걸릴 수 있다.

## 11. 상태와 결과

정상 상태 흐름:

```text
pending -> running -> collected
```

`task execution completed` 뒤에도 collect가 남아 있을 수 있다. 반드시 queue가 `collected`가 될 때까지 인스턴스를 유지한다.

로컬 결과 위치:

```text
vision_experiment_platform/runs/e/<experiment_id>/
```

| 버튼 | 용도 |
|---|---|
| `Open Result` | 결과 폴더 열기 |
| `Open Summary` | 요약 보고서 열기 |
| `Open Previews` | test prediction preview 열기 |
| `Copy Predictions` | prediction 산출물 위치 복사 |

주요 산출물은 `summary.md`, `metrics.json/csv`, `tasks/*/task.log`, predictions, `best.pt`, `last.pt`다. adapter에 따라 이름이 다를 수 있으므로 `Open Result`를 우선 사용한다.

| Task | 주요 test 지표 |
|---|---|
| Detection | mAP50, mAP50-95, precision, recall, latency |
| Classification | accuracy, macro F1, class별 precision/recall/F1, latency |

## 12. Vast 인스턴스 종료 기준

```text
[ ] 필요한 모든 queue가 collected다.
[ ] Open Result에서 summary와 metrics가 열린다.
[ ] 필요한 best.pt가 로컬에 있다.
[ ] 실패 분석에 필요한 원격 로그가 더 이상 없다.
```

원격 산출물 수집 전에 인스턴스를 destroy하면 복구할 수 없다.

## 13. 문제 해결

### `Permission denied (publickey)`

- 현재 PC의 `.pub` 키가 인스턴스에 등록됐는지 확인한다.
- GUI와 명령이 올바른 `id_ed25519`를 가리키는지 확인한다.
- `IdentitiesOnly=yes`를 사용한다.

### `Connection refused` 또는 timeout

- 인스턴스가 Running인지 확인한다.
- Vast에서 새 Direct SSH 명령을 복사해 `Use Command`를 다시 누른다.
- 이전 인스턴스의 host/port를 재사용하지 않는다.

### `gdown` 404 또는 folder contents 실패

- Drive 권한이 `링크가 있는 모든 사용자`인지 확인한다.
- 올바른 폴더 링크인지 확인한다.
- 다운로드 폴더에 `manifest.json`과 `part-*.bin`이 있는지 확인한다.

### `cat: part-*.bin: No such file or directory`

현재 디렉터리가 release 폴더가 아니거나 다운로드가 실패했다.

```bash
find /workspace/ironflow/drive_dataset_parts -name 'part-*.bin'
```

### `This does not look like a tar archive`

빈 tar를 만들었거나 일부 part가 누락된 경우다. manifest의 parts 목록과 실제 파일을 비교하고 권장 스크립트로 checksum부터 다시 검증한다.

### dataset path not found

- GUI Remote Root와 실제 prestaged 경로를 비교한다.
- config가 `images`를 요구하는데 release에 `crops`만 있는지 확인한다.
- detector/classifier와 데이터 variant가 일치하는지 확인한다.

### queue가 비어 있음

오류가 아니다. `Experiments`에서 선택 후 `Add Selected to Queue`를 누른다.

### install deps가 오래 걸림

새 인스턴스 첫 실행에서는 정상일 수 있다. 원격 로그가 계속 갱신되는지 확인한다. 같은 인스턴스의 다음 실행은 cache로 빨라질 수 있다.

### 학습 완료 후 전체 status가 failed

test prediction, manifest 검증, collect가 실패할 수 있다. 재실행 전에 실패한 `tasks/*/task.log`, test split, 이미지 경로를 확인한다.

### `polars` 등 로컬 package 누락

```powershell
.\setup_first.bat --force
```

원격 package는 GUI `install deps` 단계의 로그를 확인한다.

## 14. 전달과 보안

- 배포할 때 루트 전체를 압축하고 일부 폴더만 떼어 보내지 않는다.
- `models/checkpoints/pretrained`, `configs`, `scripts`, `src`를 유지한다.
- 데이터셋, runs, SQLite, 보고서, SSH 개인키를 Git에 넣지 않는다.
- `.local.yaml` Vast profile과 계정 정보도 공유하지 않는다.
- `SHA256SUMS.txt`는 최초 전달 직후 파일 무결성 확인용이다.

특정 파일 해시 확인:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\run_gui.bat'
```

## 15. 실험 일관성 원칙

- 데이터 split, seed, epoch, 평가 split을 고정한다.
- test는 최종 평가에만 사용한다.
- GUI option override를 사용했다면 보고서에 기록한다.
- 새 config는 full run 전에 Preview와 smoke로 검증한다.
- 성공 여부는 학습 로그만이 아니라 `collected`와 산출물 존재로 판단한다.

## 16. 최종 체크리스트

```text
[ ] 폴더 전체를 짧은 경로에 압축 해제했다.
[ ] setup_first.bat이 성공했다.
[ ] run_gui.bat --check가 통과했다.
[ ] Copy Public Key로 현재 PC의 공개키를 준비했다.
[ ] 새 Vast 인스턴스에 공개키를 등록했다.
[ ] 현재 Direct SSH 명령을 Use Command로 적용했다.
[ ] SSH_OK 테스트가 통과했다.
[ ] 올바른 Drive release를 다운로드하고 checksum을 검증했다.
[ ] /workspace/ironflow/prestaged에 데이터가 복원됐다.
[ ] Dataset Source와 Remote Root를 확인했다.
[ ] Experiments에서 queue를 추가했다.
[ ] Preview에서 config와 데이터 경로를 확인했다.
[ ] 실행 후 collected와 best.pt를 확인했다.
```

## 17. 핵심 명령

```powershell
.\setup_first.bat
.\run_gui.bat --check
.\run_gui.bat

ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> "echo SSH_OK"
```

실패하면 Vast를 바로 destroy하지 말고 현재 stage, GUI 로그, `task.log`, 실제 Remote Root부터 확인한다.
