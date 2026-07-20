# IronFlow Vision Experiment Platform 상세 설명서

작성일: 2026-06-25

이 문서는 IronFlow Vision Experiment Platform이 어떤 프로그램인지, 어떤 기능을 제공하는지, 어떤 구조와 메커니즘으로 실험을 실행하는지, 그리고 팀 실험/배포 관점에서 어떤 장점이 있는지 설명한다.

## 1. 프로그램 개요

IronFlow Vision Experiment Platform은 군용 차량 이미지 데이터셋을 대상으로 여러 vision 모델을 동일한 조건에서 학습, 예측, 평가, 수집, 비교하기 위한 실험 플랫폼이다.

초기에는 detection 모델과 classification 모델을 조합해 end-to-end 성능을 비교하는 방향으로 출발했지만, 현재는 실험 설계가 더 명확하게 분리되어 있다.

- detector 단독 후보 비교
- classifier 단독 후보 비교
- 필요 시 detector -> crop -> classifier 조합 비교
- 최종 후보 고도화 전략 비교
- Vast AI 같은 외부 GPU 서버에서 원격 학습 실행
- 결과 파일, 성능 지표, checkpoint, preview artifact 자동 수집

즉 이 프로그램은 단순히 모델 하나를 실행하는 스크립트가 아니라, 팀원이 여러 모델 후보를 같은 방식으로 돌리고 결과를 비교할 수 있게 해주는 실험 운영 도구다.

## 2. 이 프로그램이 해결하는 문제

모델 실험을 수동으로 하면 다음 문제가 반복된다.

- 모델마다 실행 명령어가 다르다.
- 데이터셋 경로와 weight 경로를 매번 직접 맞춰야 한다.
- GPU 서버가 바뀔 때마다 SSH, dependency, upload, collect 과정을 반복해야 한다.
- 결과가 사람마다 다른 폴더 구조로 저장되어 비교가 어렵다.
- accuracy만 보고 판단하면 어떤 클래스에서 실패했는지 알기 어렵다.
- 학습이 멈춘 것인지, 진행 중인지 GUI에서 판단하기 어렵다.
- 여러 모델을 순차 실행할 때 사람이 계속 기다렸다가 다음 실험을 눌러야 한다.

IronFlow는 이 문제를 다음 방식으로 줄인다.

- 모델 실행을 YAML config로 표준화한다.
- GUI에서 후보 모델을 선택하면 실행 config와 DB 경로가 자동으로 채워진다.
- Vast AI SSH 명령을 붙여넣으면 서버 profile을 자동 등록한다.
- Run 버튼으로 dependency 확인, upload, remote run, collect까지 연결한다.
- 최대 5개 모델을 queue에 넣고 Run All로 순차 실행할 수 있다.
- 결과는 공통 artifact 구조로 저장된다.
- metric, prediction, timing, checkpoint, confusion matrix를 같은 위치에서 확인할 수 있다.

## 3. 핵심 개념

### 3.1 Experiment

Experiment는 하나의 실험 단위다.

예:

```text
exp_top10_06_classifier_efficientnet_b3
exp_top10_08_detector_yolo11s
exp_top10_09_yolo26n_effb0_6class_exports_5ep
```

하나의 experiment는 하나 이상의 task로 구성된다.

### 3.2 Task

Task는 실제 실행되는 최소 작업 단위다.

예:

```text
00_train_classify_efficientnet_b3
01_predict_classify_efficientnet_b3
00_train_detector_yolo11s
01_predict_detector_yolo11s
00_detect_original
01_make_crops
02_classify_crops
```

Task는 detection, classification, preprocessing, segmentation, embedding 등으로 나뉜다.

### 3.3 Adapter

Adapter는 특정 모델 라이브러리를 IronFlow 공통 실행 규격에 맞춰 연결하는 코드다.

예:

| Adapter | 역할 |
|---|---|
| `ultralytics_yolo` | YOLO 계열 detector 실행 |
| `ultralytics_yolo_classifier` | YOLO classification 모델 실행 |
| `torchvision_classifier` | Torchvision classifier 실행 |
| `timm_classifier` | timm classifier 실행 |
| `rf_detr_detection` | RF-DETR 계열 detector 실행 |
| `rt_detr_detection` | RT-DETR 계열 detector 실행 |
| `d_fine_detection` | D-FINE detector 실행 |
| `detection_to_classification_crop` | detection bbox를 classifier crop 입력으로 변환 |

프로그램은 모델명을 직접 실행하지 않는다. 모델명은 config에 들어가고, 실제 실행은 adapter가 담당한다.

### 3.4 Artifact

Artifact는 실험 중 생성되는 결과 파일이다.

대표 artifact:

- `metrics.csv`
- `predictions.json`
- `classification_predictions.json`
- `detection_predictions.json`
- `timings.csv`
- `summary.md`
- `task.json`
- `artifacts.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `previews/detection_bbox/*.jpg`
- `metrics/classification_confusion_matrix.csv`

Artifact 구조를 통일했기 때문에 실험 결과를 수집하고 비교하기 쉽다.

## 4. 전체 동작 흐름

프로그램의 큰 흐름은 다음과 같다.

```text
모델 후보 선택
-> GUI/CLI가 effective config 생성
-> experiment DB 등록
-> task plan 생성
-> 로컬 또는 원격 GPU 서버에서 task 실행
-> task별 metrics/predictions/checkpoints 저장
-> collect 단계에서 결과를 로컬로 수집
-> summary/Results 탭에서 비교
```

원격 GPU 실행의 경우 조금 더 자세히 보면 다음 흐름이다.

```text
Vast AI 인스턴스 생성
-> SSH key 등록
-> GUI에 Vast SSH command 붙여넣기
-> Use Command
-> Register Vast
-> Run 또는 Run All
-> install deps 또는 dependency probe
-> prepare
-> upload
-> remote run
-> collect
-> local result folder 확인
```

## 5. GUI 구성

현재 GUI는 크게 네 탭으로 구성된다.

| 탭 | 역할 |
|---|---|
| `Experiments` | 실행할 모델 후보를 선택한다. |
| `Run` | 선택된 실험의 실행 옵션, Vast SSH, dataset source, runtime, collect 옵션을 설정하고 실행한다. |
| `Results` | SQLite DB 또는 result folder를 기반으로 결과를 확인한다. |
| `Advanced` | 고급 설정, readiness, smoke 검증 등 보조 기능을 확인한다. |

### 5.1 Experiments 탭

Experiments 탭은 모델 후보를 고르는 곳이다.

현재는 detector-only, classifier-only, 조합형 후보들이 config로 준비되어 있다. 사용자는 1개에서 최대 5개까지 선택할 수 있고, `Apply to Run`을 누르면 Run 탭의 queue에 들어간다.

### 5.2 Run 탭

Run 탭은 실제 실행을 조작하는 중심 화면이다.

주요 영역:

- Scheduled Runs queue
- Applied Experiment 정보
- Flow 정보
- Models table
- Config path
- DB path
- Server name
- Vast SSH command
- Native Params
- Dataset Source
- Run Options
- Output options
- Collect options
- Weights options
- Vast Details
- 실행 로그/진행 상태

### 5.3 Queue 기능

팀 실험에서는 한 사람이 여러 모델을 맡을 수 있다. 그래서 Run 탭에는 queue 기능이 들어가 있다.

- 최소 1개, 최대 5개 모델 선택 가능
- `Run Selected`: 선택된 실험 하나만 실행
- `Run All`: queue에 있는 실험을 순차 실행
- 하나의 실험이 collect까지 끝난 뒤 다음 실험으로 넘어감
- 실패하면 해당 item은 failed가 되고 queue 실행은 중단
- GUI를 다시 켜도 queue 상태를 복구
- 오래된 running 상태는 interrupted로 복구해 사용자가 상태를 다시 판단할 수 있게 함

이 구조 덕분에 팀원은 밤새 3개 모델을 순차 실행시키는 식의 운영이 가능하다.

## 6. 실행 옵션

Run Options는 실험마다 조정 가능한 주요 학습/예측 조건이다.

| 옵션 | 의미 |
|---|---|
| `Epochs` | 학습 epoch 수 |
| `Det Batch` | detector 학습 batch size |
| `Cls Batch` | classifier 학습 batch size |
| `Det Img` | detector 입력 이미지 크기 |
| `Cls Img` | classifier 입력 이미지 크기 |
| `Timeout` | 원격 실행 timeout |
| `LR` | learning rate override |
| `Patience` | early stopping patience |
| `Min Delta` | early stopping 최소 개선 폭 |

빈칸으로 두면 config 또는 native params에 정의된 기본값이 사용된다. 사용자가 값을 입력하면 해당 값이 effective config에 override된다.

detector와 classifier는 학습 특성이 다르기 때문에 batch size와 image size를 분리했다. 예를 들어 detector는 큰 이미지가 필요할 수 있고, classifier는 crop 이미지 기준으로 상대적으로 작은 입력을 사용할 수 있다.

## 7. Dataset Source

현재 GUI는 데이터셋 source를 선택할 수 있다.

| Source | 의미 |
|---|---|
| `Remote pre-staged path` | Vast 서버에 이미 풀어둔 데이터셋 경로를 사용 |
| `Local package/upload` | 로컬 데이터셋을 패키징해 서버로 업로드 |

현재 운영에서는 `Remote pre-staged path`가 기본값에 가깝다. 이유는 전체 데이터셋을 매번 로컬에서 Vast로 올리면 upload 시간이 오래 걸리기 때문이다.

대표 remote root:

```text
/workspace/ironflow/prestaged/imported_20260617
```

데이터셋은 크게 detection용과 classification용으로 나뉜다.

```text
combined_model_name_detection/detection
combined_model_name_classification/crops
```

detector 실험은 detection 데이터셋을 사용하고, classifier 실험은 crop classification 데이터셋을 사용한다.

## 8. 모델 후보 구조

현재 프로그램은 크게 세 종류의 실험 후보를 가진다.

### 8.1 Detector-only 후보

detector만 단독으로 학습/예측해서 detection 성능을 비교한다.

예:

- YOLO11n
- YOLO11s
- YOLO12s
- YOLO26s
- RT-DETR 계열
- LW-DETR 계열
- RF-DETR 계열
- D-FINE 계열

주요 성능 지표:

- precision
- recall
- mAP50
- mAP50-95
- class AP
- prediction count
- GT count
- latency/timing

### 8.2 Classifier-only 후보

crop 이미지 classification 데이터셋으로 classifier를 단독 학습/예측한다.

예:

- MobileNetV3 Small
- EfficientNet-B0
- ConvNeXt V2 Tiny
- EfficientNet-V2-S
- YOLO26n-cls
- EfficientNet-B3
- MobileNetV3 Large
- ResNet50
- ResNeXt50 32x4d
- ConvNeXt Small
- Swin Tiny

주요 성능 지표:

- accuracy
- macro precision
- macro recall
- macro F1
- minimum class recall
- label error rate
- confusion matrix

### 8.3 Chain 후보

detection 결과를 crop으로 변환한 뒤 classifier에 넣는 조합형 실험이다.

예:

```text
YOLO26n detection
-> detection_to_classification_crop
-> EfficientNet-B0 classification
```

이 구조는 최종 프로그램에 가까운 형태를 검증할 때 중요하다. 다만 detector와 classifier를 각각 단독으로 비교하는 실험이 선행되면, 어떤 부분이 병목인지 더 명확하게 해석할 수 있다.

## 9. Native Wrappers의 의미

GUI의 `Native wrappers`는 IronFlow 내부의 mock/contract 경로가 아니라 실제 모델 라이브러리 또는 외부 모델 wrapper를 통해 실행하겠다는 의미다.

쉽게 말하면 다음 차이다.

| 모드 | 의미 |
|---|---|
| Native wrappers off | 테스트용 계약 실행 또는 mock/smoke 성격이 강함 |
| Native wrappers on | 실제 모델 라이브러리 기반 학습/예측 실행 |

실전 학습에서는 기본적으로 `Native wrappers`를 켜야 한다.

## 10. Weight 정책

Weights 옵션은 모델 시작 weight를 어떻게 가져올지 정한다.

| 옵션 | 의미 |
|---|---|
| `Preset config` | config/native params에 정의된 준비된 pretrained checkpoint 사용 |
| `Architecture only` | pretrained 없이 모델 구조만 사용 |
| `Pretrained download` | 라이브러리 기본 pretrained 다운로드 허용 |
| `Checkpoint` | 사용자가 지정한 checkpoint 사용 |

팀 실험에서는 기본적으로 `Preset config`를 권장한다.

이유:

- 같은 weight에서 시작해야 모델 비교가 공정하다.
- 외부 다운로드 실패 가능성을 줄인다.
- Vast 서버가 바뀌어도 package include 또는 prestaged weight 정책으로 재현성을 높일 수 있다.

## 11. Remote Runtime

Vast AI 서버에서는 두 가지 runtime 방식이 가능하다.

| Runtime | 의미 |
|---|---|
| `Native Python` | Vast template에 있는 Python 환경 위에 dependency를 설치/확인하며 실행 |
| `Docker image` | 미리 빌드한 Docker 이미지 환경에서 실행 |

현재 Docker runtime scaffold도 준비되어 있다.

기본 Docker image 이름:

```text
ironflow-gpu:20260625
```

배포형으로는 GHCR 같은 container registry에 이미지를 올리고, Vast에서 해당 이미지를 template으로 사용하는 방식이 장기적으로 안정적이다.

단, Docker image가 데이터셋 업로드 속도를 직접 개선하지는 않는다. Docker는 dependency 설치 시간과 template 차이 문제를 줄이기 위한 장치다.

## 12. Upload와 Collect 메커니즘

원격 실행에서 upload와 collect는 가장 시간이 오래 걸릴 수 있는 단계다.

### 12.1 Upload

upload 단계에서는 다음 항목을 원격 서버로 올린다.

- 실행에 필요한 코드 package
- effective config
- 필요한 weight/checkpoint
- 필요한 경우 데이터셋 package

현재는 shared asset cache와 package include 정책을 사용해 중복 전송을 줄이려는 구조가 들어가 있다.

하지만 GPU 서버가 새로 생성되면 서버에는 기존 cache가 없으므로 첫 upload는 오래 걸릴 수 있다.

### 12.2 Collect

collect 단계에서는 원격 result directory에서 로컬로 필요한 artifact를 내려받는다.

collect mode:

| Mode | 의미 |
|---|---|
| `Quick` | 가벼운 결과 위주, preview/checkpoint를 줄임 |
| `Standard` | 기본 결과와 필요한 checkpoint 수집 |
| `Weights` | weight 수집을 중시 |
| `Full Debug` | task directory 전체에 가까운 디버그 수집 |

checkpoint collect:

| 옵션 | 의미 |
|---|---|
| `Off` | weight를 내려받지 않음 |
| `Best only` | best checkpoint만 수집 |
| `Best + Last` | best와 last 모두 수집 |

실전 학습 검증에서는 `Best + Last`를 권장한다.

## 13. 결과 폴더 구조

실험 결과는 보통 다음 위치에 저장된다.

```text
runs/e/<experiment_id>/
```

예:

```text
runs/e/exp_top10_06_classifier_efficientnet_b3/
```

대표 구조:

```text
<experiment_id>/
  status.marker
  summary.md
  metrics.csv
  task_results.json
  timings.csv
  tasks/
    00_train_.../
      task.json
      task.log
      metrics.csv
      predictions.json
      artifacts.json
      timings.csv
      checkpoints/
        best.pt
        last.pt
      predictions/
        classification_predictions.json
      metrics/
        classification_confusion_matrix.csv
        classification_confusion_matrix_normalized.csv
        classification_confusion_pairs.csv
    01_predict_.../
      ...
```

root의 `metrics.csv`는 보통 실험 대표 metric을 모은 것이다. 세부 분석은 task별 `metrics.csv`, `predictions/*.json`, `metrics/classification_confusion_*.csv`를 함께 봐야 한다.

## 14. 성능 지표

### 14.1 Classification 지표

classifier 계열에서 주로 보는 지표:

| 지표 | 의미 |
|---|---|
| `accuracy` | 전체 샘플 중 맞춘 비율 |
| `macro_precision` | 클래스별 precision의 단순 평균 |
| `macro_recall` | 클래스별 recall의 단순 평균 |
| `macro_f1` | 클래스별 F1의 단순 평균 |
| `class_recall` | 현재는 minimum class recall 성격으로 사용 |
| `num_predictions` | 예측 샘플 수 |
| `num_gt` | GT label 수 |
| `label_error_rate` | `1 - accuracy` |

클래스 불균형이 있을 수 있으므로 accuracy만 보지 않고 macro F1과 macro recall을 함께 본다.

### 14.2 Confusion Matrix

classifier 결과에는 confusion matrix artifact가 생성된다.

```text
metrics/classification_confusion_matrix.csv
metrics/classification_confusion_matrix_normalized.csv
metrics/classification_confusion_pairs.csv
```

이 파일들은 어떤 클래스가 어떤 클래스로 헷갈렸는지 확인하기 위한 것이다. 예를 들어 `leopard_2`가 `leclerc`로 자주 예측된다면, 단순 accuracy보다 훨씬 직접적으로 데이터/모델 문제를 파악할 수 있다.

주의:

- 과거에 confusion matrix 패치 전 실행한 결과에는 이 파일이 없을 수 있다.
- 새 코드로 다시 실행하거나 원격 서버에 최신 코드가 upload된 이후의 결과부터 생성된다.

### 14.3 Detection 지표

detector 계열에서 주로 보는 지표:

| 지표 | 의미 |
|---|---|
| `precision` | 예측 bbox 중 실제 정답인 비율 |
| `recall` | 실제 객체 중 찾아낸 비율 |
| `map50` | IoU 0.5 기준 mAP |
| `map50_95` | IoU 0.5~0.95 평균 mAP |
| `class_ap50` | 클래스별 AP50 |
| `num_predictions` | 예측 bbox 수 |
| `num_gt` | GT 객체 수 |

detection은 classification과 달리 bbox 위치까지 평가하므로 mAP가 중요하다.

## 15. Preview 기능

detection 결과는 bbox preview 이미지로 확인할 수 있다.

예:

```text
tasks/<detector_task>/previews/detection_bbox/*.jpg
```

이 preview는 모델이 실제로 어디를 잡았는지 사람이 직접 확인하기 위한 것이다.

실무적으로는 전체 이미지를 모두 내려받기보다 50~100장 정도 샘플링하는 방식이 효율적이다. 너무 많은 preview를 collect하면 다운로드 시간이 길어진다.

## 16. 학습 진행 표시

GUI는 원격 실행 중 현재 단계와 로그를 보여준다.

대표 단계:

```text
install deps
prepare
upload
run
collect
```

run 단계에서는 원격 `task.log` 또는 train log를 tail하는 구조가 들어가 있어, 사용자는 학습이 멈춘 것인지 실제로 진행 중인지 판단할 수 있다.

## 17. 배포 관점의 장점

### 17.1 팀원이 같은 방식으로 실험 가능

각 팀원이 다른 컴퓨터에서 실행해도 다음 요소가 통일된다.

- config 구조
- dataset source
- pretrained weight 정책
- metric schema
- artifact schema
- result directory 구조
- Vast SSH 등록 방식

### 17.2 실험 재현성이 높다

GUI에서 만든 effective config가 저장되므로, 나중에 어떤 설정으로 실험했는지 추적할 수 있다.

```text
runs/configs/gui_effective/
```

### 17.3 결과 비교가 쉽다

SQLite DB, metrics.csv, summary.md, task_results.json이 함께 남기 때문에 결과를 비교하거나 실패 원인을 추적하기 쉽다.

### 17.4 GPU 운영 부담을 줄인다

Vast AI 서버를 새로 빌릴 때 필요한 작업을 GUI 흐름에 넣었다.

- SSH key 복사
- Vast SSH command 적용
- profile 등록
- dependency 확인
- upload
- remote run
- collect

### 17.5 모델 추가가 구조화되어 있다

새 모델을 추가할 때는 대체로 다음 순서를 따른다.

1. model catalog에 후보 등록
2. adapter가 이미 있으면 기존 adapter 사용
3. adapter가 없으면 새 adapter 구현
4. pretrained checkpoint 정책 정의
5. engine config YAML 추가
6. recommendation matrix 또는 GUI candidate 목록에 연결
7. readiness/test 추가

즉 모델 추가가 완전히 자동은 아니지만, 어디를 수정해야 하는지 구조가 분리되어 있다.

## 18. 현재 주의해야 할 점

### 18.1 실험 결과 해석

1 epoch smoke 결과는 실행 가능성 확인에 가깝다. 최종 성능 판단에는 충분하지 않다.

특히 큰 모델은 1 epoch에서 성능이 낮게 나올 수 있다. 이것이 모델 구조가 잘못되었다는 뜻은 아니다.

### 18.2 원격 서버와 로컬 코드 동기화

GUI에서 Run을 누르면 upload 단계에서 코드가 서버로 올라간다. 다만 이미 실행 중인 실험은 이전 코드로 돌고 있을 수 있다.

예:

- confusion matrix 패치 전에 시작한 실험은 결과에 matrix 파일이 없다.
- 패치 후 새로 실행한 실험부터 matrix 파일이 생긴다.

### 18.3 Vast 서버가 새로 바뀌면 cache가 사라질 수 있음

서버를 새로 만들면 원격 `/workspace`에는 기존 업로드 cache가 없다. 따라서 첫 실행은 느릴 수 있다.

### 18.4 Docker는 dependency 안정화를 위한 것

Docker image를 사용하면 dependency 설치와 template 차이 문제를 줄일 수 있다. 하지만 데이터셋 업로드/다운로드 속도를 직접 해결하지는 않는다.

### 18.5 Weight 파일은 git에 포함되지 않을 수 있음

pretrained/fine-tuned weight는 용량이 크기 때문에 git으로 관리하지 않는 경우가 많다. 배포 시에는 Google Drive, 별도 archive, Vast pre-stage, registry 정책을 함께 정해야 한다.

## 19. 주요 경로 요약

프로젝트 루트:

```text
<복사한_폴더>\vision_experiment_platform
```

주요 경로:

| 경로 | 역할 |
|---|---|
| `src/ironflow_exp/engine/` | 실험 엔진 본체 |
| `src/ironflow_exp/models/` | 모델 wrapper/spec |
| `configs/engine/` | 실행 YAML config |
| `configs/experiments/` | 추천 후보 matrix |
| `configs/model_catalog/` | 모델 catalog |
| `configs/servers/` | SSH/Vast server profile |
| `configs/engine/native_params/` | 모델별 native parameter defaults |
| `models/checkpoints/pretrained/` | pretrained weight cache |
| `runs/e/` | 실험 결과 |
| `runs/configs/gui_effective/` | GUI가 만든 effective config |
| `runs/gui_queue/` | GUI queue 상태 |
| `docker/` | GPU Docker runtime 정의 |
| `docs/` | 설계/운영 문서 |
| `tests/` | 회귀 테스트 |

## 20. 한 문장 요약

IronFlow Vision Experiment Platform은 여러 detector/classifier 후보를 GUI와 YAML config 기반으로 표준화해, 로컬 또는 Vast AI GPU 서버에서 학습하고, metric/checkpoint/prediction/preview/confusion matrix를 같은 구조로 수집해 팀 단위 모델 선택을 가능하게 하는 실험 운영 플랫폼이다.
