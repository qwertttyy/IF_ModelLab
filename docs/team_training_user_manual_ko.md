# IronFlow 팀 실전 학습 사용 설명서

작성일: 2026-06-22

이 문서는 IronFlow Vision Experiment Platform을 팀원이 사용해서 준비된 모델 조합을 학습하고 결과를 비교하는 방법을 설명한다. 기준 작업 위치는 다음 폴더다.

```powershell
cd /d <복사한_폴더>\vision_experiment_platform
```

## 1. 현재 가능한 일

현재 프로그램으로 가능한 작업은 다음과 같다.

- 이미지 detection 데이터셋으로 detection 모델 학습
- classification crop 데이터셋으로 crop classifier 학습
- detection 결과를 crop으로 변환한 뒤 classifier 예측 실행
- RF-DETR, D-FINE, RT-DETR, YOLO 계열 detector 비교
- MobileNetV3, EfficientNet, ConvNeXt V2 Tiny classifier 비교
- SAM2/OpenCLIP 기반 보조 추론 및 리뷰 workflow 실행
- 실험별 metric, timing, checkpoint, summary, prediction artifact 저장
- SQLite DB 기반 결과 비교 GUI 조회
- Vast AI 같은 GPU 서버로 코드/데이터/weight 업로드 후 SSH 실행

현재 실전 학습 준비 상태:

- 실전 학습 준비 완료: Top10 1~10번 전체
- 보류: `open_vocab_grounding_dino_sam2_dinov3_audit`는 Top10에서 제외하고 deferred audit backlog로 이동
- 보류 이유: GroundingDINO는 현재 supervised training path가 아니고, DINOv3 pretrained weight는 접근 이슈가 있어 보류 중

## 2. 준비된 Top10 모델 조합

실행 config 위치:

```text
configs/engine/top10_balanced/
```

현재 팀 실험에 사용할 기본 조합은 다음과 같다.

| 번호 | 상태 | 조합 | 역할 |
|---:|---|---|---|
| 1 | 준비 | YOLO11n + MobileNetV3 Small | 가장 빠른 baseline |
| 2 | 준비 | YOLO11n + EfficientNet-B0 | 균형형 baseline |
| 3 | 준비 | YOLO11n + ConvNeXt V2 Tiny | classifier 정확도 비교 |
| 4 | 준비 | YOLO12n + EfficientNet-B0 | YOLO 계열 detector 비교 |
| 5 | 준비 | YOLO26n + YOLO26n-cls | YOLO26 계열만 쓰는 detection + classification 검증 |
| 6 | 준비 | RF-DETR + ConvNeXt V2 Tiny | transformer detector 정확도 후보 |
| 7 | 준비 | D-FINE + EfficientNet-V2-S | real-time DETR 계열 후보 |
| 8 | 준비 | RT-DETR + EfficientNet-B0 | Ultralytics RT-DETR 후보 |
| 9 | 준비 | YOLO26n + EfficientNet-B0 | YOLO26 detector + 검증된 EfficientNet crop classifier |
| 10 | 준비 | YOLO26n + SAM2 + OpenCLIP | YOLO26 학습 + mask/embedding review |

주의: 10번은 YOLO26 detector 학습은 가능하지만, SAM2/OpenCLIP은 학습이 아니라 추론/리뷰 보조 단계다.

## 3. 준비된 pretrained weight

pretrained cache 위치:

```text
models/checkpoints/pretrained/
```

준비된 주요 weight:

- YOLO: `yolo11n.pt`, `yolo12n.pt`, `yolo26n.pt`, `yolo26n-cls.pt`
- RT-DETR: `rtdetr-l.pt`
- RF-DETR: `rf_detr_base.pth`
- D-FINE: `dfine_hgnetv2_n_coco.pth`
- Torchvision: MobileNetV3, EfficientNet-B0/B3/V2-S, ResNet50
- timm: ConvNeXt V2 Tiny

중요: `models/checkpoints/pretrained/`는 `.gitignore` 대상이다. 팀원에게 git만 공유하면 weight가 빠진다. 배포 시 이 폴더를 별도 압축 파일이나 공유 드라이브로 반드시 같이 전달해야 한다.

## 4. 배포 전 확인

팀원에게 배포하기 전, 아래 명령으로 준비 상태를 확인한다.

```powershell
python scripts\download_pretrained_weights.py --check-only --include-timm --include-rtdetr
python scripts\download_foundation_detector_weights.py --check-only
python scripts\check_team_deployment_readiness.py
```

정상 기준:

- `ready_for_supervised_training_count`가 10
- `blocker_count`가 0
- RF-DETR/D-FINE/RT-DETR missing checkpoint blocker가 없어야 함

## 5. GUI 실행

GUI 실행 명령:

```powershell
python -m ironflow_exp.engine.ui.tk_app
```

또는:

```powershell
python scripts\launch_model_lab_gui.py
```

GUI 주요 탭:

- `Experiments`: Top10 조합을 고르는 탭
- `Run`: GPU 서버에 실험을 실행하는 메인 탭
- `Results`: 실험 결과 비교 탭
- `Advanced`: candidate, local smoke, readiness 확인용 탭

팀 실전 학습은 기본적으로 `Experiments`, `Run`, `Results` 탭을 사용한다.

## 6. GUI에서 실험 실행 순서

1. GUI를 실행한다.
2. `Experiments` 탭으로 간다.
3. `Template` 목록에서 실행할 Top10 조합을 선택한다.
4. `Apply to Run` 버튼을 누른다.
5. 자동으로 열린 `Run` 탭에서 적용된 Top10 번호를 확인한다.
6. `Server`가 `vast_5090`인지 확인한다.
7. `Config`가 `configs/engine/top10_balanced/NN_...yaml` 형태인지 확인한다.
8. `Native wrappers`가 체크되어 있는지 확인한다.
9. `Native Params`가 `configs/engine/native_params/rtx5090_balanced_defaults.yaml`인지 확인한다.
10. `Experiment` 이름을 팀원/모델 번호가 구분되게 수정한다.
11. 새 실험이면 `Replace existing`는 끈다.
12. 아래 버튼 순서로 진행한다.

권장 버튼 순서:

```text
Check -> Bootstrap -> Upload -> Submit -> Status/Logs -> Collect
```

## 7. Run 탭 버튼 설명

| 버튼 | 의미 | 언제 누르나 |
|---|---|---|
| `Check` | SSH 서버 연결, 기본 상태 확인 | 서버를 새로 열었을 때 가장 먼저 |
| `Deps` | dependency 상태 확인 | 설치가 의심될 때 |
| `Preview` | 현재 선택값으로 effective config와 단계별 SSH 명령을 미리 표시 | GPU 비용이 발생하는 실행 전에 계획 확인 |
| `Prepare` | 실행 전 준비 단계 | 통합 준비가 필요할 때 |
| `Execute` | 단일 명령 방식 실행 | 단계별 실행 대신 한 번에 돌릴 때 |
| `Bootstrap` | GPU 서버 dependency 설치 | 새 GPU 서버를 빌린 직후 |
| `Upload` | 코드/데이터/weight 업로드 | Bootstrap 후, Submit 전 |
| `Submit` | 원격 학습 job 시작 | Upload 후 |
| `Status` | job 상태 확인 | 실행 중/완료 여부 확인 |
| `Logs` | 원격 로그 tail 확인 | 학습 진행/에러 확인 |
| `Collect` | 결과를 로컬로 수집 | job 완료 후 |
| `Clear` | GUI 로그창 지우기 | 화면 정리용 |

일반적인 Vast AI 사용 순서:

```text
서버 생성 -> SSH 정보 등록 -> Check -> Preview -> Bootstrap -> Upload -> Submit -> Logs/Status 반복 -> Collect
```

`Preview`는 SSH 연결, 업로드, 실행을 하지 않는다. 현재 GUI 선택값으로 생성될 effective config 경로, dependency profile, native wrapper 상태, `Check/Deps/Prepare/Bootstrap/Upload/Submit/Status/Logs/Collect` 명령을 확인하는 용도다. Vast AI처럼 비용이 발생하는 GPU 서버에서는 `Preview`로 실험 id, DB, config, server 이름을 먼저 확인한 뒤 실행한다.

## 8. Vast AI 서버 준비

권장 서버:

- GPU: RTX 4090 24GB 이상 권장
- Disk/container: 실행 검증은 50GB 가능, 장기 학습은 더 크게 잡는 것이 좋음
- Template: PyTorch 또는 CUDA 기반 template
- SSH key: IronFlow에서 사용하는 key를 Vast instance에 추가

서버 profile 이름:

```text
vast_5090
```

서버 정보가 바뀌면 `configs/servers/vast_manual_template.yaml` 또는 현재 사용하는 server profile을 갱신해야 한다.

## 9. 팀원이 모델을 맡아 실험하는 방법

팀 단위로 나눌 때는 다음처럼 배정하면 된다.

| 담당자 | 추천 배정 |
|---|---|
| A | 1번, 2번 baseline |
| B | 3번 classifier 강화 |
| C | 4번, 5번 YOLO 계열 |
| D | 6번 RF-DETR |
| E | 7번 D-FINE |
| F | 8번 RT-DETR |
| G | 9번 YOLO26 + EfficientNet |
| H | 10번 YOLO26 + review |

각 담당자는 자기 번호 config를 선택해서 실행한다. 예를 들어 RF-DETR 담당자는 `Experiments`에서 6번을 고른 뒤 `Apply to Run`을 누른다.

실험 이름 예시:

```text
team_a_top01_yolo11n_mobilenet_3ep
team_d_top06_rf_detr_convnextv2_3ep
```

## 10. 1 epoch smoke와 실제 학습의 차이

현재 `configs/engine/top10_balanced/`는 기본적으로 실행 검증과 빠른 비교를 위한 1 epoch 성격이다. 실제 성능 비교를 하려면 epoch를 늘린 config를 생성해야 한다.

예시:

```powershell
python scripts\generate_top10_engine_configs.py --epochs 3 --batch-size 2 --use-local-pretrained-cache
```

장기 학습 예시:

```powershell
python scripts\generate_top10_engine_configs.py --epochs 30 --batch-size 4 --max-seconds 86400 --use-local-pretrained-cache
```

주의:

- DINOv3 audit workflow는 Top10에서 제외하고 deferred backlog로 이동했다.
- DINOv3 실전 pretrained 실험을 하려면 별도 weight 경로가 필요하다.
- 장기 학습 config를 만들면 기존 `top10_balanced`를 덮어쓸 수 있으므로, 필요한 경우 `--output-dir`를 따로 지정한다.

예:

```powershell
python scripts\generate_top10_engine_configs.py --output-dir configs/engine/top10_native_3epoch --epochs 3 --batch-size 2 --use-local-pretrained-cache
```

## 11. 결과 확인 방법

GUI에서:

1. `Results` 탭으로 이동한다.
2. `Use WSL DB`를 누른다.
3. `Refresh`를 누른다.
4. 실험 목록에서 원하는 실험을 선택한다.
5. `Open Result` 또는 `Open Summary`를 누른다.

주요 결과 파일:

```text
runs/experiments/<experiment_id>/
```

자주 보는 파일:

- `summary.md`
- `metrics.csv`
- `timings.csv`
- `task_results.json`
- `artifacts.json`
- `tasks/*/predictions/*.json`
- `tasks/*/checkpoints/best.pt`

결과 DB:

```text
runs/top10_<번호>_<조합명>_experiments.sqlite3
```

## 12. 기록되는 지표

실험 결과에는 가능한 범위에서 다음 지표가 기록된다.

- detection: `map50`, `map50_95`, precision, recall
- classification: accuracy, macro precision/recall/F1, class recall
- latency: `latency_ms_per_image`, `p95_latency_ms`
- runtime: task별 실행 시간, 전체 실행 시간
- GPU memory: wrapper가 제공하는 경우 기록
- prediction count, ground truth count
- embedding/mask 계열: mask count, embedding count, embedding dim 등

모델별 native library가 제공하지 않는 지표는 빈 값일 수 있다.

## 13. Local Smoke 탭은 언제 쓰나

`Local Smoke`는 실전 학습용이 아니라 로컬에서 프로그램 흐름을 확인하는 용도다.

사용 예:

- GUI가 정상 실행되는지 확인
- config 생성/결과 저장 흐름 확인
- 작은 mock 데이터로 collect/result 버튼 확인
- GPU 서버 없이 UI 동작만 확인

실전 모델 학습은 `Run` 탭에서 한다.

## 14. 자주 나는 문제

### Check 실패

원인:

- Vast 서버가 꺼짐
- SSH port 변경
- SSH key 미등록
- server profile 정보 불일치

해결:

- Vast instance의 SSH 명령을 다시 확인
- server profile 갱신
- `Check` 재시도

### Bootstrap 실패

원인:

- pip/network 문제
- CUDA/PyTorch template 문제
- disk 부족

해결:

- template이 PyTorch/CUDA 기반인지 확인
- container disk 여유 확인
- `Logs` 또는 GUI 출력 확인

### Submit 후 바로 실패

원인:

- 데이터 업로드 누락
- pretrained cache 업로드 누락
- dependency 누락
- config 경로 오류

해결:

- `Upload`를 다시 실행
- `scripts\check_team_deployment_readiness.py` 로컬에서 확인
- `Logs` 확인

### 결과가 Results 탭에 안 보임

원인:

- 아직 `Collect`를 누르지 않음
- DB 경로가 다름
- experiment id가 다름

해결:

- job 완료 후 `Collect`
- `Use WSL DB`
- `Refresh`

## 15. 배포 체크리스트

팀원에게 전달할 것:

- 프로젝트 코드
- `runs/user_datasets/tank14_prepared_v20260629/` 데이터셋
- `models/checkpoints/pretrained/` pretrained cache
- Vast/SSH server profile 설정 방법
- 이 문서

배포 전 로컬에서 실행:

```powershell
python scripts\download_pretrained_weights.py --check-only --include-timm --include-rtdetr
python scripts\download_foundation_detector_weights.py --check-only
python scripts\check_team_deployment_readiness.py
```

GPU 서버에서 처음 실행:

```text
Check -> Bootstrap -> Upload -> Submit -> Status/Logs -> Collect
```

## 16. 현재 결론

현재 상태는 다음과 같다.

- Top10 10개 조합은 실전 supervised training 준비 완료
- GroundingDINO/SAM2/DINOv3 audit 조합은 Top10 밖 deferred backlog로 이동
- pretrained cache는 준비되어 있음
- 팀원 배포 시 weight 폴더를 git 외 별도 방식으로 반드시 포함해야 함
- GPU 서버에서는 bootstrap과 upload를 먼저 실행해야 함
