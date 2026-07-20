# IronFlow Vision 처음 실행 가이드

작성일: 2026-06-26 KST

이 문서는 IronFlow Vision Experiment Platform을 처음 받은 팀원이 폴더를 복사한 순간부터 Vast GPU에서 실험을 실행하고 결과를 수집할 때까지 따라 할 수 있는 사용 설명서다.

## 1. 전체 흐름

처음 사용하는 흐름은 아래 순서다.

```text
폴더 받기
-> setup.bat 실행
-> run_gui.bat 실행
-> SSH key 준비
-> Vast 인스턴스 생성
-> Vast SSH 정보 GUI에 등록
-> Vast 서버에 데이터셋 pre-stage
-> 실험 선택
-> Run Selected 또는 Run All
-> collect 완료 후 결과 확인
```

처음에는 단계가 많아 보이지만, 매번 새로 해야 하는 것은 아니다.

매번 새 컴퓨터에서 한 번만 하는 일:

- `setup.bat` 실행
- 로컬 SSH key 준비

매번 새 Vast 인스턴스에서 한 번만 하는 일:

- Vast SSH 정보 등록
- 데이터셋 pre-stage
- 첫 `install deps` 또는 bootstrap

실험마다 하는 일:

- 실험 선택
- `Run Selected` 또는 `Run All`
- 결과 확인

## 2. 폴더 받기

전달받은 폴더는 원하는 위치에 놓아도 된다. 예시는 다음과 같다.

```text
<복사한_폴더>\vision_experiment_platform
```

권장 사항:

- 경로는 너무 길지 않게 둔다.
- 가능하면 한글/특수문자가 많은 경로는 피한다.
- 폴더 안의 `models/checkpoints/pretrained/`는 지우지 않는다.
- `runs/`는 비어 있어도 정상이다. 실행하면서 자동으로 채워진다.

중요한 폴더:

| 경로 | 의미 |
|---|---|
| `src/` | 프로그램 코드 |
| `scripts/` | 데이터 복원, weight 확인, 배포 점검 스크립트 |
| `configs/engine/top10_balanced/` | 추천 Top10 실험 config |
| `configs/engine/detector_only_balanced/` | detector 단독 실험 config |
| `configs/engine/classifier_only_balanced/` | classifier 단독 실험 config |
| `models/checkpoints/pretrained/` | 미리 준비된 pretrained weight |
| `runs/` | 실행 결과, DB, server profile이 생성되는 위치 |

## 3. 최초 설치

PowerShell 또는 파일 탐색기에서 `setup.bat`을 실행한다.

```powershell
Set-Location -LiteralPath '<복사한_폴더>\vision_experiment_platform'
.\setup.bat
```

`setup.bat`이 자동으로 하는 일:

- Conda가 있으면 `IRONFLOW_VISION` 환경을 만든다.
- Conda가 없으면 폴더 내부 `.venv`를 만든다.
- 필요한 Python 패키지를 설치한다.
- `gdown`을 설치한다.
- 기본 import가 되는지 확인한다.

설치가 끝나면 아래 명령으로 GUI 실행 가능 여부를 확인한다.

```powershell
.\run_gui.bat --check
```

정상 출력 예:

```text
[IronFlow] GUI environment check passed.
```

## 4. GUI 실행

아래 파일을 실행한다.

```powershell
.\run_gui.bat
```

GUI의 주요 탭:

| 탭 | 용도 |
|---|---|
| `Experiments` | 실행할 추천 실험이나 단독 실험을 고르는 곳 |
| `Run` | Vast SSH, 데이터 경로, batch/epoch, 실행 queue를 관리하는 곳 |
| `Results` | 수집된 결과를 DB 기준으로 보는 곳 |
| `Advanced` | 후보/검증/세부 기능 확인용 |

처음 쓰는 사람은 보통 `Experiments`와 `Run`만 알면 된다.

## 5. SSH key 준비

Vast는 SSH key로 접속한다. GUI는 팀원 각자의 컴퓨터에 맞춰 key를 만들거나 public key를 복사해준다.

1. GUI의 `Run` 탭으로 간다.
2. `Vast Details` 영역에서 `Key` 값을 확인한다.
   - 기본값은 보통 `C:\Users\<사용자>\.ssh\id_ed25519`다.
3. `Copy Public Key`를 누른다.

`Copy Public Key`가 자동으로 하는 일:

- private key가 이미 있으면 public key를 생성해서 복사한다.
- private key가 없으면 새 ed25519 key pair를 만든 뒤 public key를 복사한다.
- 복사된 public key는 클립보드에 들어간다.

Vast 웹사이트에서 해야 하는 일:

1. 인스턴스의 `Manage SSH Keys` 또는 SSH key 관리 화면을 연다.
2. `New SSH Key`에 방금 복사된 public key를 붙여넣는다.
3. `ADD SSH KEY`를 누른다.
4. 해당 key가 `Instance SSH Keys`에 들어갔는지 확인한다.

주의:

- private key는 본인 PC에만 있어야 한다.
- 팀원에게 폴더를 전달해도 SSH key는 같이 전달되지 않는다.
- 팀원은 자기 PC에서 `Copy Public Key`를 눌러 자기 key를 등록하면 된다.

## 6. Vast 인스턴스 만들기

권장 GPU:

- RTX 4090 이상
- RTX 5090 32GB면 현재 balanced 기본값에 잘 맞는다.

권장 조건:

- PyTorch/CUDA 계열 template
- 디스크 여유 공간 50GB 이상
- SSH 접속 가능 상태

인스턴스를 만든 뒤 Vast 화면에서 SSH 접속 명령을 복사한다.

예:

```text
ssh -p 40022 root@203.0.113.10 -L 8080:localhost:8080
```

## 7. Vast SSH 정보를 GUI에 등록

GUI의 `Run` 탭에서 진행한다.

1. `Vast SSH` 입력칸에 Vast에서 복사한 SSH 명령을 붙여넣는다.
2. `Use Command`를 누른다.

`Use Command`가 자동으로 하는 일:

- Host를 채운다.
- Port를 채운다.
- User를 채운다.
- GUI가 SSH 접속에 쓸 server profile 정보를 준비한다.

확인할 값:

| 항목 | 권장값 |
|---|---|
| `Server` 또는 `Name` | `vast_5090` |
| `User` | `root` |
| `Workspace` | `/workspace/ironflow` |
| `Runtime` | `Native Python` |
| `Native wrappers` | 체크 |
| `Native Params` | `configs/engine/native_params/rtx5090_balanced_defaults.yaml` |

server profile은 실행 시 자동으로 `runs/server_profiles/<name>.local.yaml`에 저장된다. 이 파일은 팀원마다 다르므로 배포 폴더에는 포함하지 않는다.

## 8. Vast 서버에 데이터셋 준비

현재 권장 방식은 Google Drive에 올려둔 tar-parts release를 Vast 서버가 직접 내려받는 방식이다.

GUI에서 사용할 데이터 경로:

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1
```

### 8.1 Vast 터미널 열기

Windows PowerShell에서 Vast SSH 명령을 실행한다.

```powershell
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080
```

접속되면 프롬프트가 대략 아래처럼 바뀐다.

```text
root@...:/workspace#
```

이제 Windows PowerShell이 아니라 Vast 서버 안의 Linux shell이다.

### 8.2 데이터 복원 명령 실행

새 Vast 인스턴스에서는 아래 명령을 한 번 실행한다.

```bash
python3 -m pip install -q gdown
mkdir -p /workspace/ironflow/drive_dataset_parts/tank_armor_prepared_v20260703_original_classifier_all_v1
gdown --folder "https://drive.google.com/drive/folders/YOUR_PREPARED_DATASET_FOLDER_ID" -O /workspace/ironflow/drive_dataset_parts/tank_armor_prepared_v20260703_original_classifier_all_v1
cd /workspace/ironflow/drive_dataset_parts/tank_armor_prepared_v20260703_original_classifier_all_v1
cat part-*.bin > ironflow_dataset_tank_armor_prepared_v20260703_original_classifier_all_v1.tar
# 선택: manifest.json의 tar_sha256 값과 아래 출력값을 비교한다.
sha256sum ironflow_dataset_tank_armor_prepared_v20260703_original_classifier_all_v1.tar
mkdir -p /workspace/ironflow/prestaged
tar -xf ironflow_dataset_tank_armor_prepared_v20260703_original_classifier_all_v1.tar -C /workspace/ironflow/prestaged
ls /workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1
```

마지막 `ls`에서 detection/classification 데이터 폴더가 보이면 준비 완료다.

GUI의 `Run` 탭에서는 아래처럼 둔다.

| 항목 | 값 |
|---|---|
| `Dataset Source` | `Remote pre-staged path` |
| `Remote Root` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1` |

`Remote pre-staged path`를 쓰면 GUI는 로컬 데이터셋을 업로드하지 않는다. 대신 config 안의 `data_variants` 경로를 Vast 서버의 데이터 경로로 자동 변환한다.

## 9. 실험 선택

1. GUI의 `Experiments` 탭으로 간다.
2. 추천 목록에서 원하는 실험을 선택한다.
3. `Add Selected to Queue`를 누른다.
4. `Run` 탭으로 이동한다.

처음 테스트 추천:

| 목적 | 추천 |
|---|---|
| 가장 빠른 end-to-end 확인 | Top10 #1 `yolo11n + mobilenet_v3_small` |
| YOLO detector 단독 확인 | detector-only `yolo11n` |
| classifier 단독 확인 | classifier-only `mobilenet_v3_small` 또는 `efficientnet_b0` |

처음부터 RF-DETR, D-FINE, RT-DETRv2, LW-DETR 같은 후보를 고르면 GPU 서버 dependency 영향이 더 크다. 첫 성공 확인은 YOLO/torchvision/timm 계열로 하는 것이 좋다.

## 10. Run 탭에서 자동 설정되는 것

`Experiments`에서 실험을 queue에 넣으면 `Run` 탭에 행이 생긴다.

자동으로 채워지는 값:

| 항목 | 자동 설정 내용 |
|---|---|
| `Config` | 선택한 실험의 locked config |
| `DB` | 해당 실험용 SQLite DB 경로 |
| `Experiment` | 실험 ID |
| `Server` | 현재 Vast profile 이름 |
| `Native wrappers` | 기본 체크 |
| `Native Params` | RTX 5090 balanced 기본값 |
| `Output` | CSV, Summary, Previews, Weights 수집 옵션 |
| `Collect Mode` | 결과와 weight 수집 정책 |

사용자가 확인하거나 바꿀 수 있는 값:

| 항목 | 설명 |
|---|---|
| `Epochs` | 학습 epoch 수 |
| `Det Batch` | detector batch size |
| `Cls Batch` | classifier batch size |
| `Det Img` | detector image size |
| `Cls Img` | classifier image size |
| `Timeout` | 전체 stage timeout |
| `Replace existing` | 같은 experiment id를 덮어쓸지 여부 |

배포 기본값은 RTX 5090 32GB 기준 balanced 설정이다.

## 11. 실행 버튼

가장 쉬운 방법:

1. queue에서 실행할 행을 선택한다.
2. `Run Selected`를 누른다.

여러 개를 순서대로 돌리려면:

1. 여러 실험을 queue에 넣는다.
2. `Run All`을 누른다.

중지하려면:

- `Stop`을 누른다.

실행 전에 설정만 보고 싶으면:

- `Preview`를 누른다.

`Preview`는 실제 GPU 작업을 시작하지 않고 현재 설정으로 어떤 config/명령이 만들어지는지 확인하는 용도다.

## 12. 실행 stage 설명

GUI는 내부적으로 아래 stage를 순서대로 진행한다.

```text
install deps -> prepare -> bootstrap -> upload -> run -> collect
```

| Stage | 하는 일 | 오래 걸리는 경우 |
|---|---|---|
| `install deps` | Vast 서버의 Python dependency를 확인/설치 | 새 인스턴스의 첫 실행 |
| `prepare` | effective config 생성, DB 기록, 원격 실행 계획 생성 | 보통 짧음 |
| `bootstrap` | 원격 workspace와 실행 환경 준비 | 새 인스턴스 첫 실행 |
| `upload` | 코드, config, pretrained weight를 Vast로 전송 | 첫 실행 또는 cache가 없을 때 |
| `run` | 실제 학습/추론 실행 | epoch, 모델 크기, 데이터 크기에 따라 다름 |
| `collect` | 결과, metric, summary, preview, weight를 로컬로 가져옴 | 결과/weight가 클 때 |

`Remote pre-staged path`를 쓰는 경우 `upload`는 데이터셋 전체를 올리지 않는다. 코드와 필요한 weight 중심으로 전송한다.

## 13. 진행 상황 보는 법

`Run` 탭의 queue table에서 `Current` 컬럼을 본다.

예:

```text
Upload: cache check -> transfer
Run: train detector -> predict detector
Collect: remote scan -> download
```

아래 log 영역에는 현재 stage, timeout, stdout/stderr 요약이 표시된다.

실패했을 때는 먼저 확인할 것:

1. log 영역의 `[stderr]`
2. 현재 stage 이름
3. `Status`가 `failed`인지 `running`인지
4. Vast 인스턴스가 아직 켜져 있는지

## 14. 결과 수집 후 확인

`collect`가 끝나면 결과는 로컬 `runs/` 아래에 저장된다.

주요 버튼:

| 버튼 | 용도 |
|---|---|
| `Open Result` | 실험 결과 폴더 열기 |
| `Open Summary` | `summary.md` 열기 |
| `Open Previews` | preview 이미지/결과 열기 |
| `Copy Predictions` | prediction 결과 경로 또는 내용을 복사 |

주요 결과 파일:

```text
runs/e/<experiment_id>/result/
runs/e/<experiment_id>/result/summary.md
runs/e/<experiment_id>/result/tasks/*/metrics.csv
runs/e/<experiment_id>/result/tasks/*/predictions/
runs/e/<experiment_id>/result/tasks/*/checkpoints/best.pt
runs/e/<experiment_id>/result/tasks/*/checkpoints/last.pt
```

실제 경로는 실행 방식과 experiment id에 따라 조금 다를 수 있다. 가장 안전한 방법은 GUI의 `Open Result`를 누르는 것이다.

## 15. Results 탭 사용

1. `Results` 탭으로 간다.
2. 필요한 DB를 선택하거나 현재 실행 DB를 사용한다.
3. `Refresh`를 누른다.
4. 실험 목록에서 보고 싶은 실험을 선택한다.
5. metric, summary, artifact를 확인한다.

비교할 때 주로 보는 값:

| Task | 주요 지표 |
|---|---|
| detection | `map50`, `map50_95`, precision, recall |
| classification | accuracy, macro F1, class recall |
| end-to-end | 전체 accuracy, class별 recall, task별 시간 |

## 16. 자주 나는 문제

### SSH 접속 실패

증상:

```text
Permission denied (publickey)
```

확인:

- GUI에서 `Copy Public Key`를 눌렀는지 확인한다.
- Vast instance에 public key가 등록되어 있는지 확인한다.
- `Key` 경로가 본인 PC의 private key를 가리키는지 확인한다.

### Connection refused

확인:

- Vast 인스턴스가 실행 중인지 확인한다.
- Vast의 SSH port가 바뀌지 않았는지 확인한다.
- Vast 화면에서 SSH command를 다시 복사해서 GUI `Vast SSH`에 붙여넣고 `Use Command`를 누른다.

### 데이터 경로 오류

증상:

```text
dataset path not found
```

확인:

- Vast 터미널에서 아래 명령을 실행한다.

```bash
ls /workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1
```

- GUI의 `Dataset Source`가 `Remote pre-staged path`인지 확인한다.
- GUI의 `Remote Root`가 `/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1`인지 확인한다.

### upload timeout

확인:

- `Dataset Source`가 `Local upload`가 아니라 `Remote pre-staged path`인지 확인한다.
- 처음 실행이면 weight 전송 때문에 시간이 걸릴 수 있다.
- 같은 Vast 인스턴스에서는 cache가 남아 이후 upload가 줄어든다.

### install deps가 오래 걸림

새 Vast 인스턴스 첫 실행에서는 정상적으로 오래 걸릴 수 있다.

확인:

- Vast 서버 네트워크가 느릴 수 있다.
- pip 설치가 진행 중일 수 있다.
- timeout이 너무 짧으면 `Timeout`을 늘린다.

### RF-DETR/D-FINE/RT-DETR 계열 실패

이 계열은 YOLO보다 native dependency 영향이 크다.

처음 성공 확인은 아래 계열부터 하는 것을 권장한다.

- YOLO detector
- torchvision classifier
- timm classifier

## 17. 새 Vast 인스턴스마다 반복해야 하는 것

새 인스턴스를 만들면 아래는 다시 해야 한다.

1. Vast SSH command를 새로 복사한다.
2. GUI `Vast SSH`에 붙여넣고 `Use Command`를 누른다.
3. public key가 새 인스턴스에 등록되어 있는지 확인한다.
4. 데이터셋을 `/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1`에 복원한다.
5. 첫 실행에서 `install deps/bootstrap/upload`가 다시 오래 걸릴 수 있다.

같은 인스턴스를 계속 쓰면 데이터셋과 일부 cache가 남아 있어서 다음 실험은 더 빠를 수 있다.

## 18. 배포 전 점검 명령

폴더를 다른 팀원에게 전달하기 전 또는 전달받은 뒤 아래 명령을 실행하면 좋다.

```powershell
Set-Location -LiteralPath '<복사한_폴더>\vision_experiment_platform'
.\run_gui.bat --check
python scripts\check_team_deployment_readiness.py
python scripts\download_pretrained_weights.py --check-only --include-timm --include-rtdetr
```

정상 기준:

- GUI environment check passed
- `blocker_count`가 `0`
- pretrained weight의 `ok`가 모두 `true`

## 19. 처음 실행 추천 체크리스트

처음 받은 팀원은 아래 순서만 따라 하면 된다.

```text
[ ] 폴더를 D 드라이브 등 짧은 경로에 복사했다.
[ ] setup.bat을 실행했다.
[ ] run_gui.bat --check가 통과했다.
[ ] run_gui.bat으로 GUI를 열었다.
[ ] Copy Public Key를 눌러 public key를 복사했다.
[ ] Vast 인스턴스에 public key를 등록했다.
[ ] Vast SSH command를 GUI에 붙여넣고 Use Command를 눌렀다.
[ ] Vast 터미널에서 데이터셋 restore 명령을 실행했다.
[ ] GUI Dataset Source가 Remote pre-staged path다.
[ ] Remote Root가 /workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1다.
[ ] Experiments 탭에서 첫 실험을 queue에 넣었다.
[ ] Run 탭에서 Run Selected를 눌렀다.
[ ] collect 완료 후 Open Summary 또는 Open Result로 결과를 확인했다.
```

## 20. 핵심 요약

팀원이 직접 바꿔야 하는 것은 세 가지다.

1. 자기 컴퓨터의 SSH key
2. 새 Vast 인스턴스의 SSH command
3. 새 Vast 인스턴스의 데이터셋 pre-stage

나머지 config, model weight, native params, metric 수집, result 저장 구조는 프로그램이 자동으로 맞춘다.
