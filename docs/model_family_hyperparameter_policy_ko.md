# 모델 계열별 하이퍼파라미터 운영안

작성일: 2026-07-01  
대상 데이터셋: `tank_armor_prepared_v20260630`  
적용 범위: 전차/장갑차 detection 모델, 전차 model-name classification 모델

## 1. 기본 원칙

이번 실험에서는 모델 간 비교를 위해 `epochs`는 동일하게 맞춘다.

다만 batch size, optimizer, learning rate, augmentation, image size 등은 모델 계열마다 일반적으로 쓰이는 안정적인 값을 사용한다. YOLO, DETR/DFINE, CNN classifier, Transformer classifier는 메모리 사용량과 학습 안정성이 다르기 때문에 모든 값을 강제로 통일하지 않는다.

공통 고정값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| train split | `train` |
| validation split | `val` |
| final prediction/evaluation split | `test` |
| epochs | `30` |
| early stopping patience | `5` |
| early stopping min delta | `0.0` |
| pretrained | 사용 |
| random seed | 가능하면 동일 seed 사용 |
| detection class | `tank`, `armored_vehicle` |
| classification class | 전차 모델명 class |

공통 batch size 정책은 다음과 같이 고정한다.

| 계열 | batch size |
|---|---:|
| YOLO detection | `16` |
| DETR/DFINE/VIT 계열 detection | `8` |
| classification 전체 | `32` |

보고서 표현은 다음 문장을 사용한다.

> 전체 모델은 동일한 train/val/test split과 동일한 epoch 조건에서 비교하였다. 단, 모델 계열별 메모리 요구량과 권장 학습 설정이 다르므로 batch size, optimizer, learning rate 등은 각 모델 계열의 일반적인 안정값을 사용하였다. Early stopping은 validation 성능 기준 patience 5로 동일하게 적용하였다.

## 2. Detection 모델 권장값

### 2.1 YOLO 계열

대상 모델:

- `yolov8n`
- `yolo11n`
- `yolo11s`
- `yolo12n`
- `yolo12s`
- `yolo26n`
- `yolo26s`

권장값:

| 항목 | n 모델 | s 모델 | 비고 |
|---|---:|---:|---|
| image size | `640` | `640` | 공통 |
| batch size | `16` | `8` 또는 `16` | GPU 여유 있으면 n은 32 가능 |
| optimizer | `auto` 또는 `SGD/AdamW` | `auto` 또는 `SGD/AdamW` | Ultralytics 기본값 우선 |
| initial lr | framework default | framework default | 무리하게 통일하지 않음 |
| weight decay | framework default | framework default | Ultralytics 기본값 우선 |
| augmentation | YOLO 기본 augmentation | YOLO 기본 augmentation | mosaic/mixup 등 기본 정책 |
| patience | `5` | `5` | early stopping |
| AMP | `True` | `True` | 가능하면 사용 |
| workers | `4`-`8` | `4`-`8` | 서버 CPU 상황에 따라 조정 |

권장 시작값:

```yaml
image_size: 640
batch_size: 16
patience: 5
early_stopping_patience: 5
early_stopping_min_delta: 0.0
pretrained: true
```

YOLO `s` 모델에서 OOM이 나면 `batch_size: 8`로 낮춘다.

### 2.2 DETR / DFINE 계열

대상 모델:

- `rf_detr`
- `d_fine`
- `rt_detr`
- `rt_detr_v2`
- `lw_detr`

권장값:

| 항목 | 권장값 | 비고 |
|---|---:|---|
| data format | COCO JSON | `instances_train/val/test.json` |
| image size | `640` | 모델 wrapper가 허용하는 경우 |
| batch size | `2`-`4` | 4090급이면 4부터 시도 |
| optimizer | `AdamW` | Transformer 계열 표준 |
| initial lr | `1e-4` | 불안정하면 `5e-5` |
| backbone lr | `1e-5` | 지원되는 wrapper에서만 |
| weight decay | `1e-4` | 일반적인 AdamW 설정 |
| gradient clipping | `0.1` | 지원되면 사용 |
| augmentation | resize, horizontal flip 중심 | 강한 mosaic은 피함 |
| patience | `5` | early stopping |
| AMP | `True` | 가능하면 사용 |
| workers | `4` | 메모리/IO 상황에 따라 조정 |

권장 시작값:

```yaml
image_size: 640
batch_size: 4
learning_rate: 0.0001
weight_decay: 0.0001
patience: 5
early_stopping_patience: 5
early_stopping_min_delta: 0.0
pretrained: true
```

OOM이 나면 순서대로 조정한다.

1. `batch_size: 2`
2. `image_size: 512`
3. AMP 사용 확인

## 3. Classification 모델 권장값

classification은 전차 detector/classifier 분리 흐름에서 전차 crop/image-folder 데이터셋을 사용한다.

기본 입력 구조:

```text
classifier_mbt/crops/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

### 3.1 MobileNet / EfficientNet-B0 경량 CNN

대상 모델:

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`

권장값:

| 항목 | 권장값 | 비고 |
|---|---:|---|
| image size | `224` | 공통 |
| batch size | `32` 또는 `64` | GPU 여유 있으면 64 |
| optimizer | `AdamW` | 안정적인 기본값 |
| initial lr | `1e-3` | pretrained fine-tuning 기준 |
| weight decay | `1e-4` | 일반값 |
| scheduler | cosine 또는 step | 지원되는 경우 |
| augmentation | random resized crop, horizontal flip, color jitter 약하게 | 과한 증강은 피함 |
| patience | `5` | val accuracy 기준 |
| AMP | `True` | 가능하면 사용 |

권장 시작값:

```yaml
image_size: 224
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
early_stopping_patience: 5
early_stopping_min_delta: 0.0
pretrained: true
```

### 3.2 EfficientNet-B3 / EfficientNet-V2-S / ResNet / ResNeXt

대상 모델:

- `efficientnet_b3`
- `efficientnet_v2_s`
- `resnet50`
- `resnext50_32x4d`

권장값:

| 항목 | 권장값 | 비고 |
|---|---:|---|
| image size | `224` | 필요 시 EfficientNet-B3는 300도 가능하나 비교에서는 224 권장 |
| batch size | `16` 또는 `32` | 4090급이면 32부터 시도 |
| optimizer | `AdamW` |
| initial lr | `5e-4` |
| weight decay | `1e-4` |
| scheduler | cosine |
| augmentation | random resized crop, horizontal flip, color jitter 약하게 |
| patience | `5` |
| AMP | `True` |

권장 시작값:

```yaml
image_size: 224
batch_size: 32
learning_rate: 0.0005
weight_decay: 0.0001
early_stopping_patience: 5
early_stopping_min_delta: 0.0
pretrained: true
```

OOM이 나면 `batch_size: 16`으로 낮춘다.

### 3.3 ConvNeXt / Swin / ViT 계열 classifier

대상 모델:

- `convnext_v2_tiny`
- `convnext_small`
- `swin_tiny`
- 향후 ViT 계열 classifier

권장값:

| 항목 | 권장값 | 비고 |
|---|---:|---|
| image size | `224` |
| batch size | `16` |
| optimizer | `AdamW` |
| initial lr | `5e-5` 또는 `1e-4` | fine-tuning 안정성 우선 |
| weight decay | `0.05` | Transformer/ConvNeXt 계열에서 자주 쓰는 값 |
| scheduler | cosine |
| warmup | 전체 epoch의 5%-10% | 지원되는 경우 |
| augmentation | random resized crop, horizontal flip, color jitter 약하게 |
| label smoothing | `0.1` | 지원되는 경우 |
| patience | `5` |
| AMP | `True` |

권장 시작값:

```yaml
image_size: 224
batch_size: 16
learning_rate: 0.0001
weight_decay: 0.05
early_stopping_patience: 5
early_stopping_min_delta: 0.0
pretrained: true
```

validation accuracy가 흔들리면 `learning_rate: 0.00005`로 낮춘다.

### 3.4 YOLO classification 계열

대상 모델:

- `yolo26n-cls`

권장값:

| 항목 | 권장값 | 비고 |
|---|---:|---|
| image size | `224` |
| batch size | `32` |
| optimizer | Ultralytics 기본값 |
| initial lr | framework default |
| weight decay | framework default |
| augmentation | Ultralytics classification 기본 augmentation |
| patience | `5` |
| AMP | `True` |

권장 시작값:

```yaml
image_size: 224
batch_size: 32
patience: 5
early_stopping_patience: 5
early_stopping_min_delta: 0.0
pretrained: true
```

## 4. 모델별 요약표

### 4.1 Detection

| 모델 | 계열 | format | image size | batch | optimizer/lr | patience |
|---|---|---|---:|---:|---|---:|
| `yolov8n` | YOLO n | YOLO txt | 640 | 16 | Ultralytics default | 5 |
| `yolo11n` | YOLO n | YOLO txt | 640 | 16 | Ultralytics default | 5 |
| `yolo12n` | YOLO n | YOLO txt | 640 | 16 | Ultralytics default | 5 |
| `yolo26n` | YOLO n | YOLO txt | 640 | 16 | Ultralytics default | 5 |
| `yolo11s` | YOLO s | YOLO txt | 640 | 8-16 | Ultralytics default | 5 |
| `yolo12s` | YOLO s | YOLO txt | 640 | 8-16 | Ultralytics default | 5 |
| `yolo26s` | YOLO s | YOLO txt | 640 | 8-16 | Ultralytics default | 5 |
| `rf_detr` | DETR | COCO JSON | 640 | 2-4 | AdamW, lr 1e-4 | 5 |
| `d_fine` | DETR/DFINE | COCO JSON | 640 | 2-4 | AdamW, lr 1e-4 | 5 |
| `rt_detr` | RT-DETR | COCO JSON | 640 | 2-4 | AdamW, lr 1e-4 | 5 |
| `rt_detr_v2` | RT-DETR v2 | COCO JSON | 640 | 2-4 | AdamW, lr 1e-4 | 5 |
| `lw_detr` | DETR | COCO JSON | 640 | 2-4 | AdamW, lr 1e-4 | 5 |

### 4.2 Classification

| 모델 | 계열 | image size | batch | optimizer/lr | weight decay | patience |
|---|---|---:|---:|---|---:|---:|
| `mobilenet_v3_small` | 경량 CNN | 224 | 32-64 | AdamW, lr 1e-3 | 1e-4 | 5 |
| `mobilenet_v3_large` | 경량 CNN | 224 | 32-64 | AdamW, lr 1e-3 | 1e-4 | 5 |
| `efficientnet_b0` | 경량 CNN | 224 | 32-64 | AdamW, lr 1e-3 | 1e-4 | 5 |
| `efficientnet_b3` | CNN | 224 | 16-32 | AdamW, lr 5e-4 | 1e-4 | 5 |
| `efficientnet_v2_s` | CNN | 224 | 16-32 | AdamW, lr 5e-4 | 1e-4 | 5 |
| `resnet50` | CNN | 224 | 16-32 | AdamW, lr 5e-4 | 1e-4 | 5 |
| `resnext50_32x4d` | CNN | 224 | 16-32 | AdamW, lr 5e-4 | 1e-4 | 5 |
| `convnext_v2_tiny` | ConvNeXt | 224 | 16 | AdamW, lr 1e-4 | 0.05 | 5 |
| `convnext_small` | ConvNeXt | 224 | 16 | AdamW, lr 1e-4 | 0.05 | 5 |
| `swin_tiny` | Transformer | 224 | 16 | AdamW, lr 1e-4 | 0.05 | 5 |
| `yolo26n-cls` | YOLO cls | 224 | 32 | Ultralytics default | default | 5 |

## 5. OOM 발생 시 조정 규칙

OOM이 발생하면 다음 순서로 조정한다.

1. batch size를 절반으로 줄인다.
2. AMP가 꺼져 있으면 켠다.
3. DETR/DFINE 계열만 image size를 640에서 512로 낮춘다.
4. 그래도 실패하면 gradient accumulation을 사용해 effective batch를 보완한다.

권장 fallback:

| 계열 | 1차 batch | OOM fallback |
|---|---:|---:|
| YOLO n detection | 16 | 8 |
| YOLO s detection | 8-16 | 8 |
| DETR/DFINE detection | 4 | 2 |
| 경량 CNN classifier | 32-64 | 32 |
| 중형 CNN classifier | 32 | 16 |
| ConvNeXt/Swin classifier | 16 | 8 |

## 6. 최종 실험 기록 시 남길 항목

각 실험 결과에는 최소한 다음 값을 기록한다.

| 항목 | 기록 이유 |
|---|---|
| model id | 모델 구분 |
| adapter | YOLO/COCO/classifier 처리 경로 구분 |
| dataset path | detection/classification 데이터셋 구분 |
| data format | YOLO txt 또는 COCO JSON |
| epochs | 공통 비교 조건 |
| batch size | 계열별 실제 학습 조건 |
| image size | 입력 해상도 |
| optimizer | 모델별 학습 설정 |
| learning rate | 모델별 학습 설정 |
| weight decay | 모델별 학습 설정 |
| patience | 모두 5 |
| train/val/test split | 데이터 오염 방지 확인 |
| best epoch | early stopping 해석 |
| test metric | 최종 비교 기준 |

## 7. 요약

이번 실험에서 통일할 항목:

- `epochs`
- `train/val/test split`
- `test` 기준 최종 평가
- `patience: 5`
- 동일 dataset version
- 동일 metric 체계

모델 계열별로 다르게 둘 항목:

- batch size
- optimizer
- learning rate
- weight decay
- augmentation 세부값
- DETR/DFINE 계열의 gradient clipping/warmup 등 모델 특화 설정

이 방식이 모델별 대표 학습 조건을 해치지 않으면서도, 보고서에서 비교 기준을 명확히 설명하기 가장 적절하다.
