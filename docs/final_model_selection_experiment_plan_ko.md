# 최종 모델 선택 실험 계획

Date: 2026-06-22

목표: 최종 프로그램에 넣을 모델 또는 모델 체인 1개를 고른다.

이 문서는 논문식 전체 조합 비교가 아니라, IronFlow 프로그램에 실제로 넣을 수 있는 최종 후보를 단계적으로 줄이는 실험 계획이다.

## 현재 기준 상태

이미 완료된 것으로 보는 항목:

- Top10 config readiness:
  - `config_count: 10`
  - `ready_for_supervised_training_count: 10`
  - `blocker_count: 0`
- native wrapper smoke:
  - RT-DETR
  - RF-DETR
  - D-FINE
  - SAM2
  - OpenCLIP
  - DINOv3 fallback architecture smoke
- Vast GPU 연결 및 실행 흐름:
  - SSH 연결 확인
  - GPU probe 확인
  - upload, submit, status, logs, collect 흐름 확인
- Top10 중 end-to-end 1epoch smoke 완료:
  - #1 `baseline_fast_yolo11n_mobilenetv3_small`
  - #4 `attention_yolo12n_efficientnet_b0`
  - #5 `yolo26_only_detection_classification`
  - #9 `next_yolo_family_yolo26n_efficientnet_b0`
- collector timing summary 복구:
  - 위 네 실험 모두 `task_count=5`로 집계됨

따라서 다음 단계는 실행 가능성 검증이 아니라, 전체 데이터셋 기준 성능 선발 실험이다.

## Top10 조합 선정 근거

이 절은 `docs_2/model_md/ironflow_model_final_integrated_complete.md`의 모델 후보 정리와, 각 모델의 공식 문서/논문에서 확인되는 사실만 근거로 작성한다. 아래 내용은 특정 조합이 최종 성능이 좋을 것이라는 예측이 아니라, Top10 screening에서 검증할 비교축을 명확히 하기 위한 근거다.

### 근거 제한

- 공식 문서, 공식 GitHub, 논문/arXiv, 공식 모델 문서에 나온 사실만 사용한다.
- 논문/공식 benchmark 수치는 해당 논문 또는 문서의 실험 환경에서 나온 값이며, IronFlow 전체 데이터셋 성능을 보장하지 않는다.
- 최종 선택은 Phase 2 이후 IronFlow 전체 데이터셋의 동일 split, 동일 resize 조건, 동일 metric으로 다시 비교한다.

### Top10 선정 근거 표

| # | 조합 | 공식/논문 기반 비교축 | 근거 |
|---:|---|---|---|
| 1 | `YOLO11n -> MobileNetV3 Small` | 빠른 supervised detection + 저자원 crop classifier 기준선 | Ultralytics YOLO11 공식 문서는 YOLO11이 detection, segmentation, classification, pose, OBB를 지원하며, YOLO11n detection COCO benchmark와 T4 TensorRT latency를 제공한다. MobileNetV3 논문은 MobileNetV3-Small/Large를 low/high resource use case에 맞춰 제시하고, mobile phone CPU에 맞춘 hardware-aware NAS와 NetAdapt 기반 설계를 설명한다. 따라서 이 조합은 YOLO11 detector와 경량 classifier의 기준선이다. |
| 2 | `YOLO11n -> EfficientNet-B0` | 동일 YOLO11 detector에서 classifier를 EfficientNet 계열로 바꾼 균형 비교 | YOLO11은 위 #1과 같은 공식 detection 기준선이다. EfficientNet 논문은 depth, width, resolution을 compound scaling으로 함께 조절하는 EfficientNet 계열을 제시하고, B0를 기준 baseline network로 둔 뒤 계열을 확장한다. 따라서 이 조합은 detector를 고정하고 MobileNetV3 대비 EfficientNet-B0 crop classifier를 비교하는 축이다. |
| 3 | `YOLO11n -> ConvNeXt V2 Tiny` | 동일 YOLO11 detector에서 modern ConvNet crop classifier 비교 | YOLO11은 위 #1과 같은 공식 detection 기준선이다. ConvNeXt V2 논문은 ConvNeXt 구조와 self-supervised FCMAE, GRN layer를 함께 설계해 ImageNet classification, COCO detection, ADE20K segmentation benchmark 개선을 보고하고, 여러 크기의 pretrained ConvNeXt V2 모델을 제시한다. 따라서 이 조합은 동일 detector 아래에서 modern ConvNet classifier를 비교하는 축이다. |
| 4 | `YOLO12n -> EfficientNet-B0` | attention-centric YOLO detector와 EfficientNet-B0 crop classifier 비교 | Ultralytics YOLO12 공식 문서는 YOLO12를 attention-centric object detection 모델로 설명하고, Area Attention, R-ELAN, FlashAttention 사용 등을 핵심 특징으로 제시한다. 동시에 YOLO12는 community-driven release로 training instability, elevated memory consumption, slower CPU throughput 가능성이 있어 production에는 YOLO11 또는 YOLO26을 권장한다고 명시한다. 따라서 이 조합은 YOLO 계열 안에서 attention-centric detector의 이득과 운영 리스크를 함께 확인하는 비교축이다. |
| 5 | `YOLO26n -> YOLO26n-cls` | YOLO26 단일 family의 detection/classification task-head 비교 | Ultralytics YOLO26 공식 문서와 논문은 YOLO26이 detection, segmentation, pose, classification, OBB 등 task-specific heads를 제공하고, native end-to-end NMS-free inference, DFL 제거, Progressive Loss, STAL을 포함한다고 설명한다. 따라서 이 조합은 같은 YOLO26 family 안에서 detector와 classification head를 모두 사용하는 구조 비교축이다. |
| 6 | `RF-DETR -> ConvNeXt V2 Tiny` | DETR 계열 detector + modern ConvNet classifier 비교 | RF-DETR 공식 benchmark 문서는 RF-DETR의 COCO/RF100-VL detection benchmark와 T4 TensorRT FP16 latency를 제시하고, RF-DETR paper는 target dataset에 대해 accuracy-latency Pareto curve를 찾는 real-time detection transformer로 설명한다. ConvNeXt V2 근거는 #3과 같다. 따라서 이 조합은 YOLO 계열이 아닌 DETR detector와 modern ConvNet crop classifier의 비교축이다. |
| 7 | `D-FINE -> EfficientNetV2-S` | real-time DETR localization detector + 빠른 학습/효율 classifier 비교 | D-FINE 논문과 공식 GitHub는 D-FINE이 DETR의 bbox regression을 Fine-grained Distribution Refinement로 재정의하고, GO-LSD를 도입한 real-time object detector라고 설명한다. 공식 GitHub model zoo는 COCO AP, parameter 수, latency와 Objects365+COCO pretrained 결과를 제공한다. EfficientNetV2 논문은 training-aware NAS와 Fused-MBConv 등을 통해 이전 모델보다 faster training과 parameter efficiency를 목표로 한다고 설명한다. 따라서 이 조합은 DETR 계열 localization 후보와 EfficientNetV2 classifier의 비교축이다. |
| 8 | `RT-DETR -> EfficientNet-B0` | Ultralytics API로 다룰 수 있는 real-time NMS-free Transformer detector 비교 | RT-DETR 논문은 RT-DETR을 real-time end-to-end object detector로 제시하고, efficient hybrid encoder와 query selection을 사용한다고 설명한다. Ultralytics RT-DETR 공식 문서는 RT-DETR이 NMS-free framework, anchor-free detection, decoder layer 조절을 통한 inference speed trade-off를 지원한다고 설명한다. EfficientNet-B0 근거는 #2와 같다. 따라서 이 조합은 Ultralytics 경로에서 사용할 수 있는 Transformer detector와 EfficientNet crop classifier의 비교축이다. |
| 9 | `YOLO26n -> EfficientNet-B0` | YOLO26 detector와 검증 대상 EfficientNet-B0 classifier의 교차 비교 | YOLO26 detector 근거는 #5와 같고, EfficientNet-B0 근거는 #2와 같다. 이 조합은 #5의 YOLO26n-cls classifier와 #2/#4/#8의 EfficientNet-B0 classifier 축을 연결하여, detector는 YOLO26으로 두고 classifier만 EfficientNet-B0로 바꿨을 때의 차이를 확인하는 비교축이다. |
| 10 | `YOLO26n -> SAM2 -> OpenCLIP` | closed-set 최종 classifier가 아니라 mask/embedding review 보조 경로 | YOLO26 detector 근거는 #5와 같다. SAM 2 논문과 Meta 공식 GitHub는 SAM 2를 image/video promptable segmentation foundation model로 설명하고, box/point prompt 기반 image prediction과 video propagation API를 제공한다. OpenCLIP 공식 GitHub는 OpenAI CLIP의 open-source implementation이며 LAION-400M, LAION-2B, DataComp-1B 등으로 학습된 pretrained model과 zero-shot 평가 결과를 제공한다고 설명한다. CLIP 논문은 400M image-text pairs 기반 natural language supervision과 zero-shot transfer를 제시한다. 따라서 이 조합은 최종 closed-set classifier 후보가 아니라 mask, embedding, retrieval, review queue 보조 경로를 검증하는 축이다. |

### 출처

- Ultralytics YOLO11 공식 문서: https://docs.ultralytics.com/models/yolo11/
- Ultralytics YOLO12 공식 문서: https://docs.ultralytics.com/models/yolo12/
- Ultralytics YOLO26 공식 문서: https://docs.ultralytics.com/models/yolo26/
- Ultralytics YOLO26 논문: https://arxiv.org/abs/2606.03748
- Ultralytics RT-DETR 공식 문서: https://docs.ultralytics.com/models/rtdetr/
- RT-DETR 논문: https://arxiv.org/abs/2304.08069
- RF-DETR 공식 문서/benchmark: https://rfdetr.roboflow.com/latest/learn/benchmarks/
- RF-DETR 논문: https://arxiv.org/abs/2511.09554
- D-FINE 논문: https://arxiv.org/abs/2410.13842
- D-FINE 공식 GitHub: https://github.com/Peterande/D-FINE
- MobileNetV3 논문: https://arxiv.org/abs/1905.02244
- EfficientNet 논문: https://arxiv.org/abs/1905.11946
- EfficientNetV2 논문: https://arxiv.org/abs/2104.00298
- ConvNeXt V2 논문: https://arxiv.org/abs/2301.00808
- SAM 2 논문: https://arxiv.org/abs/2408.00714
- SAM 2 공식 GitHub: https://github.com/facebookresearch/sam2
- CLIP 논문: https://arxiv.org/abs/2103.00020
- OpenCLIP 공식 GitHub: https://github.com/mlfoundations/open_clip

## 핵심 원칙

1. Top10은 최종 후보군이지, 모두를 끝까지 튜닝할 대상이 아니다.
2. 초반에는 탈락 기준을 빠르게 적용한다.
3. 최종 test set은 마지막까지 잠근다.
4. 성능 수치뿐 아니라 속도, VRAM, artifact 안정성, GUI/운영 가능성을 같이 본다.
5. detector 단독 성능이 아니라 `Detection -> Crop -> Classification` end-to-end 결과를 우선한다.

## Resize와 Augmentation 배치 결정

원칙적으로는 Top10 전체에 resize와 augmentation을 모두 걸어 비교하는 것이 가장 넓은 탐색이다. 하지만 현재 프로젝트는 GPU 시간, 전체 데이터셋 학습 시간, 팀 운영 비용을 제한해야 하므로 초기 단계에서 모든 조합을 곱하지 않는다.

이번 계획에서는 다음 순서로 고정한다.

```text
앞단: 모델/resize 비교
뒷단: 최종 Top1 고도화
```

즉, Phase 1/2의 모델 선별 단계에서는 augmentation을 적용하지 않고, standard preprocessing만 사용한다. Augmentation은 최종 Top1 모델이 정해진 뒤 고도화 ablation으로만 비교한다.

### 왜 앞단에서 resize를 보는가

전차/군용 객체 탐지에서는 원거리, 작은 객체, 낮은 해상도 객체가 중요하다. 이 경우 입력 resize는 detector 성능과 small-object recall에 직접적인 영향을 줄 수 있다.

초기 screening에서 default resize 하나만 보면 다음 후보를 놓칠 수 있다.

```text
default resize에서는 낮은 성능
high resize에서는 small-object recall이 크게 상승하는 모델
```

이런 후보는 Top10 단계에서 탈락시키면 뒤에서 복구할 수 없다. 따라서 Top10 전체 screening에서는 최소한 `default`와 `high` 두 resize를 비교한다.

### 왜 앞단에서 augmentation을 제외하는가

Resize와 augmentation은 모두 모델 구조 자체는 아니지만 성격이 다르다. Resize는 모델이 입력을 어떤 해상도로 볼지 정하는 입력 해상도 조건이고, augmentation은 학습 데이터에 어떤 변형을 적용할지 정하는 학습 데이터 변환 정책이다.

초기 screening의 목적은 모델 후보와 입력 해상도 조건을 비교하는 것이다. 이 단계에 augmentation까지 포함하면 모델 구조, 입력 해상도, 데이터 변환 정책의 효과가 동시에 섞여 결과 해석이 어려워질 수 있다.

또한 Top10 전체에 augmentation 조건까지 넣으면 실험 수가 급격히 늘어난다.

예:

```text
Top10 x 2 resize x 3 augmentation = 60 experiments
```

여기에 epoch, seed, threshold까지 더하면 초기 screening의 목적을 벗어난다. 앞단의 목적은 최종 튜닝이 아니라 후보 선별이다.

따라서 Phase 1/2에서는 augmentation을 사용하지 않는다. 이는 augmentation 효과를 무시하기 위한 결정이 아니라, 모델/resize 선별과 augmentation 고도화 실험을 분리하기 위한 결정이다.

### Phase 1/2 입력 처리

Phase 1/2에서는 다음만 사용한다.

```text
standard preprocessing only
```

포함:

```text
- normalize
- resize
- letterbox/padding 또는 모델별 필수 입력 정렬
- tensor conversion
```

제외:

```text
- horizontal flip
- brightness/contrast jitter
- random scale/translation
- blur/noise
- cutout/occlusion
- 상하 flip
- 과도한 rotation
- 비현실적인 perspective
```

### Top1 augmentation 비교

augmentation은 최종 Top1 모델이 정해진 뒤 수행한다.

```text
Top1 x augmentation ablation
```

비교할 augmentation 수준:

| 수준 | 목적 |
|---|---|
| No Aug | 모델과 데이터의 기본 성능 확인 |
| Light Aug | 안정적인 일반화 향상 여부 확인 |
| Medium Aug | 더 강한 조명/품질/부분가림 변화 대응 여부 확인 |

이때 모델과 resize는 이미 선택된 Top1 조건으로 고정한다. 즉 고도화 단계에서는 모델, resize, augmentation을 동시에 바꾸지 않는다.

## Classification 입력 포맷 결정

이번 최종 모델 선택 실험에서는 classification 입력을 `crop image`로 고정한다.

`crop vs original` 비교가 완전히 의미 없는 질문은 아니다. 다만 이번 프로젝트에서는 다른 핵심 실험에 비해 우선순위가 낮고, 선행 연구 및 현재 문제 구조상 crop 입력이 가장 타당한 방향으로 정리되었기 때문에 정식 비교 실험에서는 제외한다.

검토한 선택지는 다음과 같다.

| 선택지 | 구조 | 판단 |
|---|---|---|
| Crop image classification | `crop image -> classifier -> class` | 최종 classification 입력으로 채택 |
| Original + bbox classification | `original image + bbox -> ROI/crop -> classifier -> class` | runtime crop과 같으므로 별도 성능 실험 가치가 낮음 |
| Original-only classification | `original image -> classifier -> class` | classifier에게 위치 탐색까지 암묵적으로 요구하므로 현재 구조와 맞지 않음 |

### 1. Crop image classification

이 방식은 현재 프로젝트에서 가장 명확한 구조다.

```text
Detection:
original image에서 객체 위치를 찾고 bbox를 만든다.

Crop:
bbox 기준으로 필요한 객체 영역만 자른다.

Classification:
crop image만 보고 세부 class를 판단한다.
```

장점:

- detector와 classifier의 역할이 분리된다.
- classifier는 객체 위치를 찾을 필요 없이 class 구분에 집중한다.
- 배경, 주변 객체, 이미지 전체 context가 classification을 방해할 가능성이 줄어든다.
- 이미 crop 된 데이터셋을 사용하는 현재 운영 방향과 맞다.
- 선행 연구와 일반적인 2-stage 인식 구조에서도 crop/ROI 기반 classification이 더 안정적인 방향으로 알려져 있다.

따라서 최종 모델 선택 실험에서 classification 입력은 crop image로 고정한다.

### 2. Original + bbox classification

이 방식은 겉으로는 original image를 classifier에 넣는 것처럼 보이지만, 실제로는 bbox를 이용해 필요한 영역을 다시 잘라 보게 된다.

```text
Detection -> bbox
Classification -> original image + bbox 입력
Classifier 내부 또는 직전 단계에서 ROI/crop 수행
```

이 경우 모델이 보는 정보는 결국 crop image와 거의 같다.

즉 차이는 다음 정도다.

```text
미리 crop 파일을 만들어 classifier에 넣는가
실행 중 original image와 bbox로 즉석 crop해서 넣는가
```

이는 모델 성능 비교라기보다 구현 방식 비교에 가깝다. 따라서 최종 모델 선발 실험의 별도 축으로 두지 않는다. 필요하면 운영 구현 단계에서 "precomputed crop"과 "runtime crop"의 I/O 비용만 따로 비교한다.

### 3. Original-only classification

이 방식은 bbox 없이 original image 전체를 classifier에 넣는 방식이다.

```text
original image -> classifier -> class
```

이론적으로는 실험할 수 있지만, 현재 문제에서는 실험 가치가 낮다.

이유:

- classifier가 객체 위치를 모른다.
- 객체가 작거나 배경이 큰 이미지에서는 class 판단보다 위치 탐색 문제가 먼저 생긴다.
- 이미지 안에 여러 객체가 있으면 어떤 객체의 class를 맞혀야 하는지 label 의미가 애매해진다.
- 성능이 낮게 나오면 예상 가능한 결과라 얻는 정보가 적다.
- 성능이 높게 나오더라도 데이터셋 편향, 중앙 객체 편향, 배경 단서 학습 가능성을 먼저 의심해야 한다.
- 최종 프로그램 구조인 `detector -> crop -> classifier`와 직접 연결되지 않는다.

따라서 original-only classification은 정식 실험에서 제외한다. 필요하다면 논문/발표용 참고 실험으로 소규모 sanity check를 할 수는 있지만, 최종 모델 선택 phase에는 넣지 않는다.

### 관련 구조: Detector-only

`original image -> detector -> bbox + class` 구조는 crop vs original classification 비교가 아니라, 아예 classifier 단계를 제거하는 다른 구조다.

따라서 이는 다음 비교로 다룬다.

```text
Detection-only
vs
Detection + Crop Classification
```

이 비교는 의미가 있지만, "classification에 original을 넣을지 crop을 넣을지"와는 다른 질문이다. Top10 안의 YOLO-only 후보는 이 구조 비교를 위한 후보로 볼 수 있다.

## 평가 우선순위

전차/군용 객체 탐지 프로그램 기준 우선순위:

| 우선순위 | 기준 | 이유 |
|---:|---|---|
| 1 | Recall | 실제 객체를 놓치는 것이 가장 치명적 |
| 2 | Small-object recall | 원거리/작은 객체 대응이 중요 |
| 3 | mAP50 | 전체 탐지 품질의 대표 지표 |
| 4 | Precision | 오탐이 많으면 분석관이 쓰기 어려움 |
| 5 | End-to-end object accuracy | crop classification까지 포함한 실사용 성능 |
| 6 | p95 latency | 프로그램에서 체감되는 속도 |
| 7 | VRAM/운영성 | 실제 GPU/로컬 배포 가능성 |
| 8 | 구현 안정성 | adapter, checkpoint, collect, GUI 흐름 안정성 |

최종 점수 예시:

| 항목 | 가중치 |
|---|---:|
| Recall | 25 |
| Small-object recall | 20 |
| mAP50 | 15 |
| End-to-end object accuracy | 15 |
| Precision | 10 |
| p95 latency | 5 |
| VRAM/운영성 | 5 |
| 구현 안정성 | 5 |

## Phase 0. 전체 데이터셋 고정

목표: 최종 모델 선발에 사용할 데이터 기준을 잠근다.

해야 할 일:

- 전체 데이터셋 위치 확정
- train/val/test split 고정
- class label mapping 고정
- 중복 이미지와 깨진 annotation 점검
- small-object 기준 정의
- test set은 최종 평가 전까지 사용하지 않기

권장 split:

```text
train: 학습
val: 모델 선택, resize/고도화/threshold 튜닝
test: 최종 1회 평가
```

주의:

- 지금까지의 1epoch smoke 결과는 실행 검증 결과로 본다.
- 최종 모델 순위 판단은 전체 데이터셋 train/val 기준 screening부터 시작한다.

## Phase 1. Top10 전체 1차 Screening

목표: Top10 전체에서 최종 후보가 될 가능성이 낮은 조합을 거른다.

실험 조건:

```text
Top10 x 2 resize x 10~20epoch
```

초기 추천:

```text
Top10 x 2 resize x 10epoch
```

10epoch 결과가 너무 불안정하면 같은 설정으로 20epoch까지 확장한다.

Augmentation:

```text
사용하지 않음
standard preprocessing only
```

Resize 조건:

| 모델군 | Default | High |
|---|---:|---:|
| YOLO 계열 | 640 | 960 |
| RT-DETR | 640 또는 config 기본 | 960 또는 704 |
| RF-DETR | wrapper 기본 | 한 단계 높은 resolution |
| D-FINE | 공식 config 기본 | `resize_size` / `eval_spatial_size` 상향 |
| SAM2/OpenCLIP review path | detector resize를 따른다 | detector high resize를 따른다 |

왜 resize 2개를 보는가:

- Default만 보면 작은 객체에 민감한 후보가 탈락할 수 있다.
- Top10 전체에 3개 이상 resize를 거는 것은 비용이 크다.
- 따라서 1차 screening에서는 `default`와 `high` 두 개만 본다.

선별 기준:

- val recall
- val small-object recall
- val mAP50
- precision 하한선
- end-to-end object accuracy
- p95 latency
- task failure, missing artifact, collect failure 여부

결과물:

- Top10 x 2 resize 결과표
- 후보별 best resize 후보
- Top5 선정

## Phase 2. Top5 Resize Ablation

목표: Top5 후보의 resize 민감도를 더 자세히 확인하고 최종 고도화 대상 Top1을 고른다.

실험 조건:

```text
Top5 x 3 resize x 20~30epoch
```

Resize 예시:

| 모델군 | Resize A | Resize B | Resize C |
|---|---:|---:|---:|
| YOLO 계열 | 640 | 960 | 1280 |
| RT-DETR | 640 | 960 | 1280 또는 704 |
| RF-DETR | default | high | higher |
| D-FINE | default | high | higher |

고정할 것:

- augmentation은 사용하지 않음
- train/val split 고정
- pretrained 사용 정책 고정
- threshold는 기본값으로 시작

결과물:

- Top5별 best resize
- resize 상승폭 분석
- small-object recall 상승 후보 표시
- Top1 선정

## Phase 3. Top1 고도화 Ablation

목표: Phase 2에서 선정된 Top1 모델을 대상으로, 최종 장기학습에 사용할 고도화 recipe를 정한다.

이 단계는 장기학습이 아니다. 선택된 Top1 모델에 대해 제한된 epoch 또는 고정 checkpoint 기반으로 augmentation, segmentation, threshold의 효과를 확인하는 ablation 단계다.

실험 조건:

```text
Top1 x augmentation ablation
Top1 x segmentation on/off ablation
Top1 threshold sweep
```

Augmentation 후보:

| 수준 | 의미 |
|---|---|
| No Aug | 모델 기본 성능 확인 |
| Light Aug | 밝기, 대비, 약한 crop/scale 등 안정적인 증강 |
| Medium Aug | 더 강한 scale, blur, noise, occlusion 등 |

Segmentation 후보:

| 조건 | 의미 |
|---|---|
| Segmentation off | Phase 2에서 선정된 기본 Top1 pipeline |
| Segmentation on | mask-assisted preprocessing 또는 review를 추가한 고도화 pipeline |

실험 원칙:

- 모델과 resize는 Phase 2에서 고른 Top1/best resize로 고정
- augmentation, segmentation, threshold를 모델 선택 변수로 섞지 않는다
- segmentation은 학습 재실행이 아니라 가능하면 validation inference/review 단계의 on/off 비교로 먼저 확인한다
- threshold sweep은 학습 후 val prediction에 대해 수행한다
- test set은 아직 보지 않는다

결과물:

- Top1 best augmentation
- segmentation 적용 여부
- 추천 threshold
- Top1 장기학습 recipe 확정

## Phase 4. Top1 장기 학습

목표: Phase 3에서 확정한 best recipe로 최종 배포 후보 checkpoint를 만든다.

실험 조건:

```text
Top1 x best resize x best augmentation x selected segmentation policy x 충분한 epoch
```

권장:

- pretrained=true
- early stopping 사용
- 동일 train/val split
- 가능하면 seed 반복은 별도 여유가 있을 때만 수행
- checkpoint, metrics, predictions, preview artifact 저장

확인할 것:

- 성능 평균
- seed 간 변동성
- p95 latency
- VRAM
- collect 안정성
- 예측 이미지 품질

결과물:

- 최종 test set에 올릴 Top1 checkpoint
- 장기학습 summary
- 실패 사례 검토용 predictions/previews

## Phase 5. 최종 Test 평가

목표: 잠가둔 test set으로 최종 선택을 검증한다.

조건:

- test set은 이 단계에서 처음 사용한다.
- test 결과를 보고 다시 튜닝하지 않는다.
- 수치와 예측 이미지를 같이 본다.

반드시 사람이 직접 볼 이미지:

- 전차 미탐
- 민수차량 오탐
- 장갑차/자주포 혼동
- 작은 객체
- 가려진 객체
- 원거리 객체
- 저품질/야간 이미지
- bbox 위치가 애매한 이미지

결과물:

- 최종 test metric
- 실패 사례 이미지 목록
- 최종 모델 1개 선택

## 전체 실행 수 예상

기본 계획:

```text
Phase 1: Top10 x 2 resize = 20
Phase 2: Top5 x 3 resize = 15
Phase 3: Top1 고도화 ablation = 약 5~8
Phase 4: Top1 장기 학습 = 1
Phase 5: 최종 test 평가 = 1
```

대략 총 42~45개 실험이다.

비용 절약 옵션:

```text
Phase 1을 Top10 x 2 resize x 10epoch로 먼저 돌린다.
결과가 불안정한 후보만 20epoch로 재실행한다.
```

하지 말아야 할 것:

```text
Top10 x 3 resize x 3 augmentation x 여러 epoch
```

이 방식은 실험 수가 너무 커지고, 어떤 요인이 성능을 바꿨는지 해석하기 어렵다.

초기 선별 단계에서 augmentation on/off를 함께 비교하는 것도 기본 계획에서는 제외한다.

예:

```text
Top10 x 2 resize x 2 augmentation = 40
Top5 x 3 resize x 2 augmentation = 30
```

이렇게 하면 Phase 1/2만 70개 실험이 되므로, 현재 GPU 1개 운영 조건에서는 모델/resize 선별보다 실험 수 증가가 더 큰 문제가 된다.

## Top5 선정 규칙

Top5는 단순 metric 순위 1~5가 아니라 아래 규칙으로 정한다.

포함 규칙:

- default resize 기준 상위권
- high resize에서 크게 상승한 후보
- small-object recall이 좋은 후보
- 속도가 실사용 가능한 후보
- artifact와 GUI/collect 흐름이 안정적인 후보

제외 규칙:

- recall이 낮은 후보
- precision이 지나치게 낮아 분석관 사용이 어려운 후보
- p95 latency가 운영 기준을 넘는 후보
- checkpoint, prediction, metrics, collect가 반복적으로 불안정한 후보
- weight 접근/라이선스/배포 문제가 해결되지 않은 후보

## 다음 즉시 작업

1. 전체 데이터셋 위치와 포맷 확정
2. train/val/test split 생성 및 고정
3. Top10 config들이 전체 데이터셋 split을 보도록 config 생성
4. Phase 1용 resize 2종 config 생성
5. GUI에서 Phase 1 실험을 선택/실행하기 쉽게 표시
6. Top10 x 2 resize x 10epoch screening 시작

## 최종 한 줄 원칙

최종 모델은 단순히 점수가 가장 높은 모델이 아니라, 전체 데이터셋에서 잘 잡고, 작고 먼 객체를 놓치지 않으며, 프로그램 안에서 안정적으로 돌아가는 모델이어야 한다.

## 2026-06-24 개정: detector/classifier 분리 선별 전략

기존 계획은 Top10 detector-classifier 조합을 end-to-end로 먼저 비교하는 방식이었다. 이후 논의 결과, 최종 프로그램이 `detector -> crop -> classifier` 구조를 쓰더라도 초기 선별 단계에서는 detector와 classifier를 각각 단독으로 평가한 뒤, 상위 후보만 재조합해 end-to-end 검증하는 방식이 더 해석 가능하고 실험 비용도 줄일 수 있다고 판단했다.

개정된 흐름은 다음과 같다.

```text
Phase 1A: detector-only screening
Phase 1B: classifier-only screening
Phase 2: selected detector x selected classifier end-to-end recombination
Phase 3: Top1 고도화 ablation
Phase 4: Top1 장기학습
Phase 5: 최종 test 평가
```

### Phase 1A. Detector-only Screening

목표는 원본 이미지와 bbox annotation만 사용해 탐지 모델 자체의 성능을 비교하는 것이다. 이 단계에서는 classifier를 붙이지 않는다.

후보군은 다음 locked config 폴더에서 관리한다.

```text
configs/engine/detector_only_balanced/
```

현재 detector-only 후보는 다음과 같다.

```text
01 detector_yolo11n
02 detector_yolo12n
03 detector_yolo26n
04 detector_rf_detr
05 detector_d_fine
06 detector_rt_detr
```

주요 지표는 `mAP50`, `mAP50-95`, recall, class별 AP/recall, latency다. 이 단계에서 좋은 detector는 localization 품질과 작은 객체 recall을 기준으로 선별한다.

### Phase 1B. Classifier-only Screening

목표는 이미 crop된 classification dataset만 사용해 crop classifier 자체의 성능을 비교하는 것이다. 이 단계에서는 detector가 만든 crop 품질 변수를 섞지 않는다.

후보군은 다음 locked config 폴더에서 관리한다.

```text
configs/engine/classifier_only_balanced/
```

현재 classifier-only 후보는 다음과 같다.

```text
01 classifier_mobilenet_v3_small
02 classifier_efficientnet_b0
03 classifier_convnext_v2_tiny
04 classifier_efficientnet_v2_s
05 classifier_yolo26n_cls
```

주요 지표는 `macro_f1`, `macro_recall`, `min_class_recall`, class별 precision/recall/F1, latency다. 클래스 불균형과 특정 class의 낮은 recall을 놓치지 않기 위해 단순 accuracy만으로 판단하지 않는다.

### Phase 2. End-to-end Recombination

Phase 1A/1B에서 선별된 detector 상위 후보와 classifier 상위 후보만 조합해 `detector -> crop -> classifier` 전체 성능을 검증한다.

이 단계의 목적은 "단독 detector 최고 + 단독 classifier 최고"가 실제 프로그램에서도 최고인지 확인하는 것이다. detector의 bbox 스타일과 crop 품질이 classifier 입력 분포에 영향을 주므로, 최종 선택 전 end-to-end 검증은 반드시 필요하다.

기존 Top10 end-to-end config는 폐기하지 않고, 재조합 검증 및 기존 결과와의 비교 기준으로 유지한다. 단, `YOLO26n -> SAM2 -> OpenCLIP` review path는 closed-set classifier 후보가 아니므로 detector-classifier ranking pool에는 넣지 않고 고도화/분석 경로로 분리한다.
