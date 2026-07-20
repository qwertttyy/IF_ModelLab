# Tank + Armored Vehicle 데이터셋 준비 가이드

이 문서는 기존 전차 14개 클래스 데이터셋을 다시 다운로드하지 않고, 장갑차 데이터만 추가로 받아 통합 데이터셋을 만드는 방법을 정리한다.

## 목표

- Detection 모델은 객체를 `tank` 또는 `armored_vehicle`로 탐지한다.
- Classification 모델은 기존처럼 세부 모델명 class를 분류한다.
- 기존 전차 데이터셋은 재사용한다.
- 장갑차 Google Drive 데이터만 새로 내려받는다.

## 입력 데이터

기존 전차 데이터셋:

```text
/workspace/ironflow/prestaged/tank14_prepared_v20260629
```

장갑차 Google Drive 폴더:

```text
https://drive.google.com/drive/folders/YOUR_SOURCE_DATASET_FOLDER_ID
```

현재 장갑차 class:

```text
k806
m2_bradley
btr_80
bmp_2
```

## 생성되는 데이터셋

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260630
```

구조:

```text
tank_armor_prepared_v20260630/
  detector_tank_av/
    detection/
      data.yaml
      images/train,val,test
      labels/train,val,test
      coco/annotations/instances_train,val,test.json
  classifier_mbt/
    crops/
      train/<tank_model_name>/
      val/<tank_model_name>/
      test/<tank_model_name>/
  classifier_av/
    crops/
      train/<armored_vehicle_model_name>/
      val/<armored_vehicle_model_name>/
      test/<armored_vehicle_model_name>/
  merge_report.json
```

## Detection 라벨 정책

Detection은 세부 모델명 분류가 아니라 큰 객체 범주 탐지용으로 구성한다.

```yaml
names:
  0: tank
  1: armored_vehicle
```

- 기존 전차 bbox label은 class id를 모두 `0`으로 재작성한다.
- 장갑차 bbox label은 class id를 모두 `1`로 재작성한다.
- bbox 좌표값은 유지한다.

## Classification 라벨 정책

Classification은 세부 모델명 class를 유지한다.

예:

```text
k2
m1_abrams
t72
k806
m2_bradley
btr_80
bmp_2
```

## Vast 실행 방법

Vast 터미널에서 프로그램 폴더로 이동한 뒤 실행한다.

```bash
cd /workspace/ironflow
bash scripts/prepare_vast_tank_armor_dataset.sh
```

기존 전차 데이터셋 위치가 다르면 `TANK_ROOT`를 지정한다.

```bash
TANK_ROOT=/workspace/ironflow/prestaged/tank14_prepared_v20260629 \
bash scripts/prepare_vast_tank_armor_dataset.sh
```

출력 위치를 바꾸려면 `DATASET_ROOT`를 지정한다.

```bash
DATASET_ROOT=/workspace/ironflow/prestaged/tank_armor_prepared_v20260630 \
bash scripts/prepare_vast_tank_armor_dataset.sh
```

## GUI 설정

통합 데이터셋을 쓰려면 GUI에서 아래 값을 사용한다.

```text
Dataset Source: Remote pre-staged path
Remote Root: /workspace/ironflow/prestaged/tank_armor_prepared_v20260630
```

## 효율성

이 스크립트는 전차 14개 데이터를 Google Drive에서 다시 받지 않는다. 기존 pre-stage 데이터를 재사용하고, 장갑차 데이터만 `gdown`으로 다운로드한다.

이미지는 기본적으로 hardlink를 시도한다. hardlink가 실패하면 copy로 fallback한다. detection label 파일은 class id 재매핑이 필요하므로 새로 생성한다.

## 확인 명령

```bash
ls /workspace/ironflow/prestaged/tank_armor_prepared_v20260630
cat /workspace/ironflow/prestaged/tank_armor_prepared_v20260630/detector_tank_av/detection/data.yaml
ls /workspace/ironflow/prestaged/tank_armor_prepared_v20260630/classifier_mbt/crops/train
cat /workspace/ironflow/prestaged/tank_armor_prepared_v20260630/merge_report.json
```


## GUI에서 자동 준비

새 Vast 인스턴스를 만든 뒤에는 PowerShell에서 직접 SSH 접속 후 명령을 입력하지 않아도 된다.

1. Vast의 direct SSH command를 GUI의 `Vast SSH` 칸에 붙여넣는다.
2. `Use Command`를 눌러 Host/Port/User 값을 채운다.
3. `Key`에 로컬 private key 경로가 들어 있는지 확인한다.
4. `Workspace`는 기본값 `/workspace/ironflow`를 사용한다.
5. Advanced 탭의 `SSH Stages`에서 `Prepare Data`를 누른다.

`Prepare Data`는 로컬의 데이터 준비 스크립트를 Vast 서버에 업로드한 뒤 실행한다. 기존 전차 데이터셋이 아직 없으면 `prepare_vast_tank14_dataset.sh`를 먼저 실행하고, 이후 장갑차 데이터를 받아 통합 데이터셋을 생성한다.
