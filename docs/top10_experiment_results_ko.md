# Top10 실험 결과 기록

업데이트: 2026-06-24 KST

이 문서는 `runs/` 아래 실험 결과가 `.gitignore` 대상이라 push에 포함되지 않는 문제를 보완하기 위해 만든 결과 요약본이다. 원본 결과 파일, sqlite DB, checkpoint는 로컬 `runs/`에 남아 있고, 이 문서에는 push 가능한 형태로 핵심 지표와 실행 이력을 정리한다.

## 전체 요약

| 순위 | 실험 | 상태 | 최종 accuracy | macro F1 | min class recall | task 시간 합계 | 비고 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `exp_top10_05_yolo26_only_detection_classification_8class` | collected | 0.8893 | 0.8913 | 0.8194 | 5분 29초 | 현재 로컬 기준 최고 |
| 2 | `exp_top10_04_attention_yolo12n_efficientnet_b0` | collected | 0.8712 | 0.8739 | 0.7000 | 8분 13초 | 최근 Vast/Drive pre-stage 정상 실행 |
| 3 | `exp_top10_09_next_yolo_family_yolo26n_efficientnet_b0` | collected | 0.8011 | 기록 없음 | 기록 없음 | 1분 26초 | 초기 smoke 성격 |
| 4 | `exp_top10_05_yolo26_only_detection_classification` | collected | 0.7527 | 기록 없음 | 기록 없음 | 1분 16초 | 초기 smoke 성격 |
| 5 | `exp_top10_01_baseline_fast_yolo11n_mobilenetv3_small` | collected | 0.6764 | 기록 없음 | 기록 없음 | 1분 20초 | 빠른 baseline |

주의:

- `_8class`가 붙은 결과와 붙지 않은 초기 smoke 결과는 조건이 완전히 같지 않을 수 있다.
- `exp_top10_04_attention_yolo12n_efficientnet_b0`는 Google Drive에서 Vast 인스턴스로 pre-stage한 8-class dataset 경로를 사용한 정상 실행 결과다.
- `exp_top10_04_attention_yolo12n_efficientnet_b0_8class`, `exp_top10_09_next_yolo_family_yolo26n_efficientnet_b0_8class`는 prepare/upload 흔적은 있으나 로컬 `runs/e` 기준 수집된 최종 결과가 없다.

## 실험별 기록

## 1. Top10 #1 - YOLO11n + MobileNetV3 Small

실험 ID:

```text
exp_top10_01_baseline_fast_yolo11n_mobilenetv3_small
```

상태:

- status: `collected`
- runner: `ssh`
- 성격: 빠른 baseline smoke

최종 지표:

| metric | value |
|---|---:|
| accuracy | 0.6764 |
| best epoch | 1 |

task별 시간:

| task | type | time | 주요 지표 |
|---|---|---:|---|
| `train_detect_yolo11n` | detection train | 34.05초 | precision 0.5655, recall 0.4042, mAP50 0.3171 |
| `predict_detect_yolo11n` | detection predict | 23.41초 | predictions 686 |
| `make_model_name_detector_crops` | preprocessing | 7.71초 | crop 생성 |
| `train_classify_mobilenet_v3_small` | classification train | 7.77초 | val accuracy 0.6500 |
| `predict_classify_mobilenet_v3_small` | classification predict | 7.52초 | accuracy 0.6764 |

task 시간 합계:

```text
80.47초 = 1분 20초
```

해석:

- 가장 빠른 baseline 역할은 한다.
- 정확도는 현재 비교군 중 가장 낮다.
- 빠른 sanity check용으로는 유용하지만 최종 후보로 보기는 어렵다.

## 2. Top10 #4 - YOLO12n + EfficientNet-B0

실험 ID:

```text
exp_top10_04_attention_yolo12n_efficientnet_b0
```

상태:

- status: `collected`
- runner: `ssh`
- server: `vast_4090`
- dataset source: remote pre-staged path
- remote dataset root:

```text
/workspace/ironflow/prestaged/imported_20260617
```

최종 지표:

| metric | value |
|---|---:|
| accuracy | 0.8712 |
| macro precision | 0.8902 |
| macro recall | 0.8681 |
| macro F1 | 0.8739 |
| min class recall | 0.7000 |
| min class recall class | `challenger_2` |
| label error rate | 0.1288 |
| predictions | 497 |
| ground truth | 497 |

task별 시간:

| task | type | time | 주요 지표 |
|---|---|---:|---|
| `train_detect_yolo12n` | detection train | 237.42초 | precision 0.9420, recall 0.8426, mAP50 0.9546, mAP50-95 0.9509 |
| `predict_detect_yolo12n` | detection predict | 30.41초 | predictions 497 |
| `make_model_name_detector_crops` | preprocessing | 5.15초 | crop 생성 |
| `train_classify_efficientnet_b0` | classification train | 212.23초 | val accuracy 0.8957, macro F1 0.8980 |
| `predict_classify_efficientnet_b0` | classification predict | 7.83초 | test accuracy 0.8712, macro F1 0.8739 |

task 시간 합계:

```text
493.04초 = 8분 13초
```

전체 SSH lifecycle 시간:

| stage | time |
|---|---:|
| prepare + bootstrap | 11.0초 |
| upload | 1분 30.7초 |
| run launch gap | 4.9초 |
| remote run | 8분 20.1초 |
| collect/download | 5분 15.5초 |
| finalize | 0.2초 |
| prepare 시작 -> collect 완료 | 15분 22.4초 |
| run 시작 -> collect 완료 | 13분 35.8초 |

클래스별 지표:

| class | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| `altay` | 46 | 38 | 0.9474 | 0.7826 | 0.8571 |
| `challenger_2` | 60 | 50 | 0.8400 | 0.7000 | 0.7636 |
| `k2` | 81 | 85 | 0.8706 | 0.9136 | 0.8916 |
| `leclerc` | 81 | 91 | 0.8242 | 0.9259 | 0.8721 |
| `leopard_2` | 61 | 81 | 0.7037 | 0.9344 | 0.8028 |
| `m1_abrams` | 73 | 61 | 1.0000 | 0.8356 | 0.9104 |
| `merkava_mk4` | 41 | 39 | 0.9744 | 0.9268 | 0.9500 |
| `type_10` | 54 | 52 | 0.9615 | 0.9259 | 0.9434 |

해석:

- detection 성능은 매우 좋다.
- 최종 classification은 0.8712로 상위권이지만 현재 로컬 최고는 아니다.
- 약점은 `challenger_2` recall 0.7000, `leopard_2` precision 0.7037이다.
- collect/download가 5분 이상 걸려 전체 시간에서 꽤 큰 비중을 차지한다.

## 3. Top10 #5 - YOLO26n + YOLO26n-cls

실험 ID:

```text
exp_top10_05_yolo26_only_detection_classification
```

상태:

- status: `collected`
- runner: `ssh`
- 성격: 초기 smoke

최종 지표:

| metric | value |
|---|---:|
| accuracy | 0.7527 |
| best epoch | 1 |

task별 시간:

| task | type | time | 주요 지표 |
|---|---|---:|---|
| `train_detect_yolo26n` | detection train | 29.64초 | precision 0.7419, recall 0.0809, mAP50 0.2276 |
| `predict_detect_yolo26n` | detection predict | 24.73초 | predictions 372 |
| `make_model_name_detector_crops` | preprocessing | 5.34초 | crop 생성 |
| `train_classify_yolo26n_cls` | classification train | 11.87초 | val accuracy 0.7375 |
| `predict_classify_yolo26n_cls` | classification predict | 4.35초 | accuracy 0.7527 |

task 시간 합계:

```text
75.93초 = 1분 16초
```

해석:

- 초기 smoke 기준으로는 #1보다 높고 #9보다 낮다.
- detection recall이 낮아 full 조건에서는 별도 검증이 필요했다.
- `_8class` 재실행 결과에서는 같은 계열이 가장 좋은 후보로 올라왔다.

## 4. Top10 #5 8-class - YOLO26n + YOLO26n-cls

실험 ID:

```text
exp_top10_05_yolo26_only_detection_classification_8class
```

상태:

- status: `collected`
- runner: `ssh`
- 현재 로컬 결과 기준 최고 성능

최종 지표:

| metric | value |
|---|---:|
| accuracy | 0.8893 |
| macro precision | 0.8944 |
| macro recall | 0.8908 |
| macro F1 | 0.8913 |
| min class recall | 0.8194 |
| min class recall class | `challenger_2` |
| predictions | 587 |

task별 시간:

| task | type | time | 주요 지표 |
|---|---|---:|---|
| `train_detect_yolo26n` | detection train | 252.47초 | precision 0.8186, recall 0.7833, mAP50 0.8778, mAP50-95 0.8738 |
| `predict_detect_yolo26n` | detection predict | 22.11초 | predictions 587 |
| `make_model_name_detector_crops` | preprocessing | 5.91초 | crop 생성 |
| `train_classify_yolo26n_cls` | classification train | 40.99초 | val accuracy 0.9174 |
| `predict_classify_yolo26n_cls` | classification predict | 7.40초 | test accuracy 0.8893, macro F1 0.8913 |

task 시간 합계:

```text
328.87초 = 5분 29초
```

클래스별 지표:

| class | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| `altay` | 57 | 61 | 0.9016 | 0.9649 | 0.9322 |
| `challenger_2` | 72 | 69 | 0.8551 | 0.8194 | 0.8369 |
| `k2` | 100 | 97 | 0.9278 | 0.9000 | 0.9137 |
| `leclerc` | 86 | 95 | 0.8000 | 0.8837 | 0.8398 |
| `leopard_2` | 79 | 83 | 0.8313 | 0.8734 | 0.8519 |
| `m1_abrams` | 83 | 73 | 1.0000 | 0.8795 | 0.9359 |
| `merkava_mk4` | 48 | 44 | 0.9318 | 0.8542 | 0.8913 |
| `type_10` | 62 | 65 | 0.9077 | 0.9516 | 0.9291 |

해석:

- 현재 로컬 기준 가장 좋은 최종 accuracy와 macro F1을 기록했다.
- `challenger_2`가 여전히 최저 recall이지만 0.8194로 #4보다 훨씬 안정적이다.
- classifier train 시간이 EfficientNet-B0 조합보다 짧아서 전체 task 합계도 더 좋다.
- 다음 phase에서 우선 검증할 강한 후보로 볼 수 있다.

## 5. Top10 #9 - YOLO26n + EfficientNet-B0

실험 ID:

```text
exp_top10_09_next_yolo_family_yolo26n_efficientnet_b0
```

상태:

- status: `collected`
- runner: `ssh`
- 성격: 초기 smoke

최종 지표:

| metric | value |
|---|---:|
| accuracy | 0.8011 |
| best epoch | 1 |

task별 시간:

| task | type | time | 주요 지표 |
|---|---|---:|---|
| `train_detect_yolo26n` | detection train | 43.63초 | precision 0.7419, recall 0.0809, mAP50 0.2299 |
| `predict_detect_yolo26n` | detection predict | 20.16초 | predictions 372 |
| `make_model_name_detector_crops` | preprocessing | 5.70초 | crop 생성 |
| `train_classify_efficientnet_b0` | classification train | 11.21초 | val accuracy 0.7625, macro F1 0.7501 |
| `predict_classify_efficientnet_b0` | classification predict | 5.55초 | accuracy 0.8011 |

task 시간 합계:

```text
86.26초 = 1분 26초
```

해석:

- 초기 smoke 기준에서는 #5 초기 결과보다 높았다.
- 다만 이후 8-class 조건에서 #5 계열과 #4 계열이 더 구체적으로 검증되면서 우선순위는 내려갔다.
- EfficientNet-B0 classifier 자체는 안정적이지만, 현재 최고 성능은 YOLO26n-cls 조합이다.

## 결과 없는 prepare/upload 항목

아래 항목은 로컬 `runs/e` 기준 수집된 `metrics.json` 또는 `summary.md`가 없다.

```text
exp_top10_04_attention_yolo12n_efficientnet_b0_8class
exp_top10_09_next_yolo_family_yolo26n_efficientnet_b0_8class
```

상태 해석:

- prepare/upload 또는 queue 흔적은 있으나 최종 run/collect 결과로 판단할 수 없다.
- 성능 비교 표에는 포함하지 않는다.

## Google Drive -> Vast pre-stage 운영 기록

목적:

- 로컬 PC에서 Vast로 매번 대용량 dataset을 SCP 업로드하는 병목을 줄이기 위해 Google Drive에 dataset split archive를 올리고, Vast 인스턴스가 직접 다운로드하도록 변경했다.

Drive folder:

```text
https://drive.google.com/drive/folders/YOUR_EXPERIMENT_RESULTS_FOLDER_ID
```

Vast 새 인스턴스에서 최초 1회 실행:

```bash
python3 -m pip install -q gdown

mkdir -p /workspace/ironflow/drive_dataset_parts
gdown --folder "https://drive.google.com/drive/folders/YOUR_EXPERIMENT_RESULTS_FOLDER_ID" -O /workspace/ironflow/drive_dataset_parts

cd /workspace/ironflow/drive_dataset_parts
cat part-*.bin > ironflow_dataset_imported_20260617_8class.tar

echo "298DED110AF6684E58D1199024BBC750604E8189AD773323CF19DFEF2787E49B  ironflow_dataset_imported_20260617_8class.tar" | sha256sum -c -

mkdir -p /workspace/ironflow/prestaged
tar -xf ironflow_dataset_imported_20260617_8class.tar -C /workspace/ironflow/prestaged

ls /workspace/ironflow/prestaged/imported_20260617
```

성공 기준:

```text
/workspace/ironflow/prestaged/imported_20260617/combined_model_name_detection
/workspace/ironflow/prestaged/imported_20260617/combined_model_name_classification
```

GUI 설정:

| 항목 | 값 |
|---|---|
| Dataset Source | `Remote pre-staged path` |
| Remote Root | `/workspace/ironflow/prestaged/imported_20260617` |

운영 기준:

- 같은 Vast 인스턴스를 계속 쓰면 pre-stage dataset은 재사용 가능하다.
- 인스턴스를 삭제하고 새로 만들면 새 인스턴스에서 위 다운로드 과정을 다시 1회 수행해야 한다.
- stop/start만 하고 디스크가 유지되는 경우에는 보통 재다운로드가 필요 없다.

## 최근 안정화 작업 기록

이번 결과 산출 전후로 아래 안정화 작업을 진행했다.

1. GUI progress 보강

- queue `Current` 컬럼에 단계별 현재 작업을 표시하도록 변경했다.
- upload/collect/run 중 세부 진행 메시지를 streaming stdout 기반으로 갱신하도록 보강했다.
- 너무 긴 로그를 current 컬럼에 그대로 쌓지 않고, `이전 작업 -> 현재 작업` 형태로 줄이는 방향으로 정리했다.

2. Dataset Source 선택지 추가

- GUI에 `Local upload/cache`와 `Remote pre-staged path` 선택지를 추가했다.
- remote pre-stage 선택 시 config 생성 단계에서 `data_variants` 경로를 `/workspace/ironflow/prestaged/imported_20260617/...`로 바꾸도록 했다.
- remote pre-stage 선택 시 local dataset package include를 제거해 대용량 dataset 업로드를 피한다.
- 현재 기본값은 `Remote pre-staged path`로 조정했다.

3. SSH/upload 안정화

- OpenSSH 명령에 keepalive 옵션을 추가했다.
- SSH command stdin wedge 방지를 위해 `ssh -n`을 적용했다.
- upload cache check, remote prepare, archive extract timeout을 더 세분화했다.
- bootstrap은 단순 `mkdir -p` 단계이므로 timeout을 짧게 조정했다.

4. stale workspace prepare 실패 수정

- `replace_existing=True`로 같은 experiment id를 재실행할 때, 기존 workspace 안의 긴 Windows 경로 파일 때문에 `shutil.rmtree`가 실패하는 문제가 있었다.
- workspace 삭제 실패 시 기존 폴더를 `.stale_...`로 옆에 치우고 새 workspace를 만들도록 보강했다.
- 기존 `code_package` 삭제 실패 시 fallback package dir을 쓰도록 추가 보강했다.

검증:

```text
ssh prepare 재현 성공
stale workspace fallback 테스트 통과
code_package fallback 테스트 통과
remote-prestaged GUI 관련 테스트 통과
```

## 현재 판단

- 현재 성능만 보면 `exp_top10_05_yolo26_only_detection_classification_8class`가 가장 유력하다.
- `exp_top10_04_attention_yolo12n_efficientnet_b0`도 detection 성능이 강하고 정상적으로 완료됐지만, 최종 classification에서 `challenger_2`, `leopard_2` 약점이 있다.
- 다음 실험은 #5 8-class 조건을 기준으로 resize, epoch, augmentation, segmentation ablation을 단계적으로 붙이는 것이 합리적이다.
