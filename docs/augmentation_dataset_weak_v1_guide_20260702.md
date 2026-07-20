# IronFlow Weak Augmentation Dataset Guide

Generated: 2026-07-02

## 목적

Top 후보 모델 추가 실험에서 사용할 `weak_v1` 사전 증강 데이터셋을 만들었다.

이번 증강은 모델 학습 코드 안에서 매 epoch마다 실시간으로 적용하는 방식이 아니라, **학습 전에 증강된 이미지와 라벨을 실제 파일로 만들어 둔 뒤 그 폴더를 모델 input으로 넣는 방식**이다.

즉 흐름은 다음과 같다.

```text
원본 prepared dataset
-> train split만 증강 파일 생성
-> 원본 train + 증강 train을 포함한 새 dataset root 생성
-> val/test는 원본 그대로 유지
-> YOLO/COCO/ImageFolder 구조를 모두 다시 맞춤
-> 모델 학습 input으로 사용
```

## 생성된 데이터셋

원본 데이터셋:

```text
D:\FinalProject\IronFlow\datawork\tank_armor_prepared_v20260630
```

증강 데이터셋:

```text
D:\FinalProject\IronFlow\datawork\tank_armor_prepared_v20260630_aug_weak_v1
```

Vast AI 서버에 올린 뒤 사용할 remote root:

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1
```

UI에서 이 증강 데이터셋으로 실험하려면 `Remote pre-staged path`를 위 remote root로 맞춘다.

## 전체 구조

원본 prepared dataset과 같은 구조를 유지했다.

```text
tank_armor_prepared_v20260630_aug_weak_v1/
  detector_tank_av/
    detection/
      images/train
      images/val
      images/test
      labels/train
      labels/val
      labels/test
      data.yaml
      manifest.json
      coco/
        coco_dataset.json
        annotations/instances_train.json
        annotations/instances_val.json
        annotations/instances_test.json
  classifier_mbt/
    crops/
      train/<class_name>/
      val/<class_name>/
      test/<class_name>/
      manifest.json
  classifier_av/
    crops/
      train/<class_name>/
      val/<class_name>/
      test/<class_name>/
      manifest.json
  augmentation_report_weak_v1.json
```

원본 파일은 가능하면 하드링크로 복제했고, 증강된 파일만 새로 생성했다. 그래서 같은 디스크 안에서는 복사 시간과 저장 공간을 줄일 수 있다. 압축/tar로 배포하면 최종 아카이브에는 일반 파일처럼 포함된다.

## Split 정책

증강은 **train split에만 적용**했다.

```text
train: 원본 이미지 + 증강 이미지
val: 원본 그대로
test: 원본 그대로
```

`val/test`에 증강 이미지를 넣지 않은 이유는 성능 평가 오염을 막기 위해서다. 검증과 테스트는 원본 분포에서 수행해야 모델 간 성능 비교가 방어 가능하다.

검증 결과 `val/test`에는 `__aug_`가 붙은 증강 파일이 0개였다.

## Detection 증강 방식

Detection은 이미지뿐 아니라 bbox 라벨도 함께 변환해야 하므로 `Albumentations`를 사용했다.

적용 정책:

```python
HorizontalFlip(p=0.5)
RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8)
HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=8, p=0.35)
```

Detection에서 `HorizontalFlip`이 적용되면 이미지가 좌우 반전될 뿐 아니라 YOLO bbox 좌표도 같이 변환된다. 이 때문에 `torchvision.transforms`만 쓰지 않고 bbox 처리가 가능한 `Albumentations`를 사용했다.

Detection 라벨 처리:

- 원본 YOLO label을 읽음
- bbox format은 YOLO normalized format 사용
- Albumentations에서 이미지와 bbox를 함께 변환
- 유효한 bbox만 남김
- 새 `.txt` label 파일 생성
- `manifest.json`에 증강 이미지 record 추가
- COCO annotation도 다시 materialize

생성된 detection 포맷:

- YOLO용 `data.yaml`
- YOLO labels
- IronFlow `manifest.json`
- COCO `instances_train.json`, `instances_val.json`, `instances_test.json`

따라서 YOLO 계열과 D-FINE/RF-DETR/RT-DETR 계열 모두 같은 증강 dataset root를 input으로 사용할 수 있다.

## Classification 증강 방식

Classification은 bbox가 없고 이미지 단위 변환만 필요하므로 팀원이 사용한 방식과 같은 `torchvision.transforms`를 사용했다.

적용 정책:

```python
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)
```

Classification 처리:

- `classifier_mbt/crops/train/<class_name>/` 이미지에만 적용
- `classifier_av/crops/train/<class_name>/` 이미지에만 적용
- 증강 이미지는 원본과 같은 class 폴더 아래 저장
- `val/test` class 폴더는 원본 그대로 유지
- `manifest.json`에 증강 이미지 record 추가

## 라이브러리가 다른 이유

Detection과 classification은 라벨 구조가 다르기 때문에 같은 라이브러리를 쓰는 것보다 **라벨 일관성과 증강 강도를 맞추는 것**이 더 중요하다. 이번 실험에서는 두 task 모두 `weak_v1`이라는 같은 정책 수준을 적용하되, 구현 라이브러리는 task 성격에 맞게 분리했다.

| 구분 | 사용 라이브러리 | 적용 대상 | 선택 근거 |
|---|---|---|
| Detection | Albumentations | 이미지 + bbox | 좌우반전이나 색상 변화 시 bbox 좌표와 class label을 이미지와 함께 동기화해야 함 |
| Classification | torchvision.transforms | 이미지 | bbox가 없고 class label은 변하지 않으므로 이미지 단위 변환만으로 충분함 |

Detection에서 `HorizontalFlip`을 적용하면 이미지의 좌우 위치가 바뀌므로 bbox의 x 좌표도 함께 바뀌어야 한다. 이미지 전용 transform만 적용하면 객체 위치와 라벨 좌표가 어긋나 학습 라벨 오류가 생긴다. `Albumentations`는 YOLO bbox format을 입력받아 이미지와 bbox를 함께 변환할 수 있고, 변환 후 유효하지 않은 bbox를 걸러낼 수 있어 detection 증강에 적합하다.

Classification에서는 각 crop 이미지 전체가 하나의 class label을 가진다. 좌우반전이나 약한 색상 변화가 적용되어도 class label 자체는 변하지 않으므로 bbox 동기화가 필요 없다. 따라서 팀원이 사용한 방식과 맞춰 `torchvision.transforms.RandomHorizontalFlip`, `ColorJitter`를 사용했다.

따라서 이번 설계의 기준은 "동일 라이브러리 사용"이 아니라 **동일한 weak augmentation 의도와 라벨 안정성 유지**다. 두 task 모두 train split에만 적용했고, val/test는 원본 그대로 유지했다.

보고서에는 다음처럼 정리할 수 있다.

> Detection과 classification은 라벨 구조가 다르므로 task별로 적합한 라이브러리를 사용했다. Detection은 bbox 좌표 동기화가 필요해 Albumentations를 사용했고, classification은 이미지 단위 변환만 필요하며 팀원 실험 조건과 맞추기 위해 torchvision.transforms를 사용했다. 두 경우 모두 train split에만 동일한 수준의 weak augmentation을 적용하고 val/test는 원본으로 유지해 평가 오염을 방지했다.

## 파일 이름 규칙

증강 파일에는 `__aug_weak_v1_01` 형태의 suffix를 붙였다.

예시:

```text
original_name.jpg
original_name__aug_weak_v1_01.jpg
original_name__aug_weak_v1_01.txt
```

이 규칙 덕분에 나중에 검증할 때 train/val/test 중 어디에 증강 파일이 들어갔는지 쉽게 확인할 수 있다.

## 생성 개수

| Branch | Split | Original | Augmented Dataset | Augmented Files |
|---|---:|---:|---:|---:|
| detection | train | 4,335 | 8,669 | 4,334 |
| detection | val | 548 | 548 | 0 |
| detection | test | 552 | 552 | 0 |
| classifier_mbt | train | 2,463 | 4,926 | 2,463 |
| classifier_mbt | val | 309 | 309 | 0 |
| classifier_mbt | test | 309 | 309 | 0 |
| classifier_av | train | 2,113 | 4,226 | 2,113 |
| classifier_av | val | 264 | 264 | 0 |
| classifier_av | test | 265 | 265 | 0 |

Detection train에서 1장은 원본 label이 비어 있어서 증강 대상에서 제외됐다. 이는 오류가 아니라 빈 label 이미지를 증강해 학습 라벨을 만들지 않도록 한 방어 처리다.

## 확인 완료 파일

아래 파일이 생성되어 있음을 확인했다.

- `detector_tank_av/detection/data.yaml`
- `detector_tank_av/detection/manifest.json`
- `detector_tank_av/detection/coco/coco_dataset.json`
- `detector_tank_av/detection/coco/annotations/instances_train.json`
- `detector_tank_av/detection/coco/annotations/instances_val.json`
- `detector_tank_av/detection/coco/annotations/instances_test.json`
- `classifier_mbt/crops/manifest.json`
- `classifier_av/crops/manifest.json`
- `augmentation_report_weak_v1.json`

## 재생성 명령

```powershell
python D:\Model_LAB\vision_experiment_platform\scripts\materialize_augmented_dataset.py `
  --input-root D:\FinalProject\IronFlow\datawork\tank_armor_prepared_v20260630 `
  --output-root D:\FinalProject\IronFlow\datawork\tank_armor_prepared_v20260630_aug_weak_v1 `
  --policy weak_v1 `
  --copies-per-train-image 1 `
  --overwrite
```

## 실험에 사용할 때

증강 데이터셋을 tar/chunk해서 Google Drive에 올리고 Vast 서버에서 복원한 뒤, UI의 dataset root를 아래로 맞춘다.

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1
```

이 root를 사용하면:

- detection 모델은 `detector_tank_av/detection`을 사용
- classification 모델은 `classifier_mbt/crops`를 사용
- 장갑차 classifier 실험이 필요하면 `classifier_av/crops`도 사용 가능
