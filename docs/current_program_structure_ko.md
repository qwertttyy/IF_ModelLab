# IronFlow 현재 프로그램 구성 설명

작성일: 2026-06-22

이 문서는 IronFlow Vision Experiment Platform이 현재 어떤 구성으로 되어 있고, 각 폴더와 모듈이 어떤 역할을 하는지 설명한다. 팀원이 코드를 처음 받았을 때 “어디를 봐야 하는지”를 빠르게 파악하는 것이 목적이다.

## 1. 한 줄 요약

IronFlow는 여러 vision 모델 조합을 config로 선택하고, 로컬/WSL/SSH GPU 환경에서 학습, 예측, 결과 수집, 지표 비교를 같은 구조로 실행하기 위한 실험 플랫폼이다.

현재 중심 흐름은 다음과 같다.

```text
데이터셋
-> engine config
-> task plan
-> task adapter 실행
-> artifacts/metrics/timings 저장
-> collect/analyze/compare
-> GUI 또는 CLI에서 확인
```

## 2. 최상위 구성

주요 폴더는 다음과 같다.

| 경로 | 역할 |
|---|---|
| `src/ironflow_exp/` | 실제 프로그램 코드 |
| `configs/engine/` | 실행 가능한 실험 config |
| `configs/experiments/` | 추천 조합 matrix와 실험 후보 정의 |
| `configs/model_catalog/` | 모델 catalog, adapter, readiness 정의 |
| `scripts/` | Top10 config 생성, pretrained 다운로드, GPU bootstrap, 외부 wrapper |
| `docs/` | 설계, 사용법, 작업 기록 문서 |
| `tests/` | 회귀 테스트 |
| `runs/` | 실험 결과, DB, 사용자 데이터셋, 원격 수집 결과 |
| `models/checkpoints/pretrained/` | 로컬 pretrained weight cache |

## 3. 실행 방식

IronFlow는 크게 세 가지 방식으로 실행된다.

| 방식 | 설명 | 주 사용처 |
|---|---|---|
| GUI | Tkinter 기반 UI | 팀원이 Top10 조합을 선택하고 실행 |
| CLI | `python -m ironflow_exp.engine.cli.main ...` | 디버깅, 자동화, SSH 세부 단계 실행 |
| Script | `scripts/*.py`, `scripts/*.sh` | Top10 config 생성, pretrained 준비, GPU 서버 bootstrap |

GUI는 모델 로직을 직접 실행하지 않는다. GUI는 config를 고르고 CLI/engine 실행을 호출하는 얇은 조작 계층이다.

## 4. Engine 구성

Engine 코드는 `src/ironflow_exp/engine/` 아래에 있다.

| 모듈 | 역할 |
|---|---|
| `configs/` | YAML config를 dataclass로 읽고 검증 |
| `core/` | task plan, artifact schema, prediction schema, compatibility 검사 |
| `tasks/` | 실제 task adapter 실행 |
| `runners/` | local runner, SSH runner, finalizer |
| `server/` | SSH 서버 등록, check, upload, bootstrap, submit, collect |
| `storage/` | SQLite 저장소 |
| `collector/` | 실행 결과 파일 수집 |
| `analyzer/` | metric/result summary 생성 |
| `datasets/` | 입력 데이터셋 형식 해석 |
| `transforms/` | detection 결과를 classification crop으로 변환 |
| `ui/` | Tkinter GUI, 추천 실험 UI, readiness/result compare UI |
| `cli/` | CLI entrypoint |

## 5. Task Adapter 구조

IronFlow는 실험을 task 단위로 쪼개서 실행한다.

대표 task type:

| Task | 예시 adapter | 역할 |
|---|---|---|
| detection | `ultralytics_yolo`, `rf_detr_detection`, `d_fine_detection`, `rt_detr_detection` | 이미지에서 객체 bbox 예측 |
| preprocessing | `detection_to_classification_crop` | detection bbox를 crop 이미지로 변환 |
| classification | `torchvision_classifier`, `timm_classifier`, `ultralytics_yolo_classifier` | 원본 또는 crop 이미지 분류 |
| segmentation | `sam_promptable_segmentation` | detection 결과 기반 mask 생성 |
| embedding | `clip_embedding`, `dinov3_embedding` | crop/image embedding 생성 |
| tracking | `bytetrack` 등 | 영상/sequence 기반 객체 연결, 현재 Top10 이미지 실험에서는 제외 |

Task 실행 결과는 `predictions/*.json`, `metrics.csv`, `timings.csv`, `artifacts.json`, `summary.md` 같은 공통 artifact 형식으로 저장된다.

## 6. 모델 구성

모델 후보는 `configs/model_catalog/default_model_catalog.yaml`에 등록된다.

모델 catalog는 다음 정보를 가진다.

- task 종류
- model id
- adapter key
- train/predict 가능 여부
- readiness 상태
- 필요한 dependency
- 입력/출력 artifact 계약

실제 adapter class 매핑은 `src/ironflow_exp/models/default_specs.py`와 `src/ironflow_exp/engine/tasks/transform_adapters.py` 쪽에서 연결된다.

현재 실전 학습 준비의 핵심 adapter는 다음이다.

| Adapter | 모델 예시 | 상태 |
|---|---|---|
| `ultralytics_yolo` | `yolo11n`, `yolo12n`, `yolo26n` | detection 학습/예측 가능 |
| `ultralytics_yolo_classifier` | `yolo26n-cls` | classification 학습/예측 가능 |
| `torchvision_classifier` | `mobilenet_v3_small`, `efficientnet_b0`, `efficientnet_v2_s` | classification 학습/예측 가능 |
| `timm_classifier` | `convnext_v2_tiny` | classification 학습/예측 가능 |
| `rf_detr_detection` | `rf_detr` | 외부 native wrapper 경로 준비 |
| `d_fine_detection` | `d_fine` | 외부 native wrapper 경로 준비 |
| `rt_detr_detection` | `rt_detr` | Ultralytics RT-DETR 경로 준비 |

## 7. Top10 실험 구성

Top10 추천 조합의 원본은 다음 파일이다.

```text
configs/experiments/recommended_model_combinations.yaml
```

실제 GUI/SSH 실행에 쓰는 잠긴 config 묶음은 다음 폴더에 있다.

```text
configs/engine/top10_balanced/
```

현재 Top10 중 실전 supervised 학습 준비 상태는 1~10번 전체다. GroundingDINO/DINOv3 audit 경로는 Top10 밖 deferred backlog로 이동했다.

특히 5번은 YOLO26 계열 단독 chain이다.

```text
YOLO26n detection
-> detection_to_classification_crop
-> YOLO26n-cls classification
```

## 8. Pretrained Cache 구성

로컬 pretrained weight cache는 다음 위치에 둔다.

```text
models/checkpoints/pretrained/
```

현재 주요 cache:

| 계열 | 파일 예시 |
|---|---|
| YOLO | `ultralytics/yolo11n.pt`, `yolo12n.pt`, `yolo26n.pt`, `yolo26n-cls.pt` |
| RT-DETR | `ultralytics/rtdetr-l.pt` |
| Torchvision | `torchvision/efficientnet_b0_imagenet.pt` 등 |
| timm | `timm/convnext_v2_tiny_pretrained.pt` |
| RF-DETR | `rf_detr/rf_detr_base.pth` |
| D-FINE | `d_fine/dfine_hgnetv2_n_coco.pth` |

이 폴더는 git에 포함되지 않는다. 팀원에게 배포할 때는 별도 압축 또는 공유 드라이브로 같이 전달해야 한다.

cache 상태 점검은 다음 명령으로 한다.

```powershell
python scripts\download_pretrained_weights.py --check-only --include-timm --include-rtdetr
python scripts\check_team_deployment_readiness.py
```

## 9. 데이터 구성

현재 주요 실험 데이터셋은 `imported_20260617` 기준이다.

대표 경로:

```text
runs/user_datasets/tank14_prepared_v20260629/combined_model_name_detection/detection
runs/user_datasets/tank14_prepared_v20260629/combined_model_name_classification/crops
```

detection 데이터는 YOLO/COCO/manifest 계열로 해석되고, classification 데이터는 image-folder 또는 manifest/crop 입력으로 해석된다.

## 10. GUI 구성

GUI 코드는 다음에 있다.

```text
src/ironflow_exp/engine/ui/tk_app.py
```

실행:

```powershell
python -m ironflow_exp.engine.ui.tk_app
```

주요 탭:

| 탭 | 역할 |
|---|---|
| Local Mock | 로컬 smoke/기본 검증 |
| Experiments | Top10 조합 선택 및 locked config 적용 |
| Run | WSL 또는 SSH 기반 원격 실행 |
| Results | 실험 결과 DB/metric 비교 |
| Advanced | candidate, local smoke, readiness 확인 |

팀원 실험 흐름은 보통 `Experiments`에서 조합을 고르고 `Apply to Run`을 누른 뒤 `Run` 실행 경로에서 `Preview`와 SSH stage 버튼을 쓰는 방식이다.

## 11. SSH/GPU 구성

GPU 서버 실행은 engine의 SSH runner 계층이 담당한다.

주요 모듈:

| 모듈 | 역할 |
|---|---|
| `server/ssh_plan.py` | 원격 실행 계획 생성 |
| `server/code_package.py` | 코드/데이터/weight 패키징 |
| `server/ssh_transfer.py` | 업로드/다운로드 |
| `server/ssh_remote_executor.py` | 원격 명령 실행 |
| `server/checker.py` | 서버 dependency/GPU/probe 확인 |

Vast AI 같은 GPU 서버에서는 다음 순서가 기본이다.

```text
server add/check
-> bootstrap/install deps
-> upload code/data/pretrained cache
-> submit
-> status/logs
-> collect
-> result compare
```

## 12. 결과 저장 구조

실험 결과는 보통 `runs/` 아래에 저장된다.

| 위치 | 설명 |
|---|---|
| `runs/e/` | 원격/GPU 실험 결과 |
| `runs/w/` | 원격 workspace |
| `runs/remote_collected/` | 원격에서 회수한 결과 |
| `runs/*.sqlite3` | 실험/metric/task 기록 DB |
| `runs/user_datasets/` | import된 사용자 데이터셋 |

각 task result에는 대체로 다음 파일들이 생긴다.

```text
predictions/*.json
metrics.csv
timings.csv
artifacts.json
summary.md
status.marker
```

## 13. 테스트 구성

주요 테스트는 다음 영역을 검증한다.

| 테스트 | 검증 내용 |
|---|---|
| `test_engine_ui_recommended_experiments.py` | Top10 matrix/config 생성/GUI locked path |
| `test_team_deployment_readiness.py` | 팀 배포 가능 상태와 pretrained cache |
| `test_task_adapter_executor.py` | task adapter 실행 계약 |
| `test_default_model_specs.py` | 모델 catalog와 adapter registry |
| `test_remote_task_adapter.py` | 원격 task adapter 실행 경로 |
| `test_code_package_service.py` | SSH 업로드 패키징 |

현재 Top10/YOLO26-only 관련 변경 후 확인한 핵심 테스트는 다음이다.

```powershell
python -m pytest -q tests\test_engine_ui_recommended_experiments.py tests\test_team_deployment_readiness.py tests\test_ultralytics_yolo_classifier_adapter.py
```

## 14. 현재 상태 요약

- GUI/CLI/SSH 기반 실험 실행 구조가 있다.
- Top10 추천 조합과 locked config가 있다.
- Top10 1~10번은 supervised 학습 준비 상태다.
- 5번은 YOLO26 계열 단독 detection/classification chain이다.
- pretrained cache는 로컬에 준비되어 있고, 팀 배포 시 별도 전달해야 한다.
- GroundingDINO/DINOv3 audit 경로는 Top10 밖 deferred backlog로 이동했다.
- 결과는 SQLite, metrics, predictions, timings, summary로 저장된다.

## 15. 관련 문서

| 문서 | 설명 |
|---|---|
| `docs/team_training_user_manual_ko.md` | 팀원이 실제로 프로그램을 사용하는 방법 |
| `docs/top10_real_training_combinations.md` | Top10 선정 및 실험 조합 |
| `docs/model_registry_readiness_matrix.md` | 모델 registry/readiness matrix |
| `docs/engine_cli_usage.md` | CLI 사용법 |
| `docs/local_wsl_mock_workflow.md` | Local/WSL workflow |
| `docs/work_log.md` | 작업 기록 |
