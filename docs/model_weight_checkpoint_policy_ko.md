# 모델 weight/checkpoint 정책

이 문서는 IronFlow Vision Experiment Platform에서 모델 weight와 checkpoint를 어떻게 보관하고 실행 config에 연결할지 정리한다.

## checkpoint와 savepoint 차이

여기서 말하는 checkpoint는 모델 파일이다. 예를 들면 `best.pt`, `last.pt`, `model.safetensors`, `pytorch_model.bin`처럼 학습된 weight나 중간 학습 상태를 담은 파일이다. 추론을 시작하거나, fine-tuning을 재개하거나, 결과를 재현할 때 쓴다.

savepoint는 보통 DB transaction이나 긴 실행 흐름에서 되돌아갈 수 있는 지점을 뜻한다. 넓게 보면 "저장된 상태"라는 점은 비슷하지만, 이 프로젝트의 4번 작업은 savepoint가 아니라 모델 weight/checkpoint 관리 정책이다.

## 기본 원칙

- Git에는 큰 weight 파일을 올리지 않는다.
- Git에는 config, manifest, README, 출처/라이선스 기록처럼 작은 텍스트 파일만 둔다.
- 실제 weight는 로컬 디스크, 원격 서버, 외부 artifact 저장소 중 하나에 둔다.
- adapter는 암묵적인 pretrained download를 기본적으로 막는다.
- 다운로드를 허용해야 할 때는 config에 `allow_pretrained_download: true`를 명시한다.
- task 결과의 `task.json` metadata에는 weight 정책 상태가 남는다.

## 권장 경로

아래 경로는 자리만 Git에 남기고, 실제 파일은 `.gitignore`로 제외한다.

```text
models/checkpoints/
local_remote_simulator/models/
```

실험 실행 결과로 생기는 checkpoint는 보통 아래에 생긴다.

```text
runs/checkpoints/
runs/model_assets/
runs/experiments/<experiment_id>/tasks/<task_id>/checkpoints/
```

`runs/` 전체도 기본적으로 Git 제외 대상이다.

## config 작성 방식

네트워크 다운로드 없이 architecture만 띄워 smoke 실행을 할 때:

```yaml
params:
  pretrained: false
```

이미 준비한 checkpoint를 명시해서 쓸 때:

```yaml
params:
  checkpoint: models/checkpoints/yolo11n/best.pt
  pretrained: false
```

checkpoint가 명시되면 현재 runnable adapter는 아래처럼 동작한다.

| adapter | checkpoint support |
| --- | --- |
| `ultralytics_yolo` | inference/train에서 checkpoint path를 YOLO model reference로 사용한다. |
| `torchvision_classifier` | inference/train에서 `state_dict` 또는 raw state dict checkpoint를 로드한다. |
| `timm_classifier` | `torchvision_classifier`와 같은 checkpoint 로더를 공유한다. |
| prepared skeleton adapters | checkpoint 요구사항과 metadata는 남기지만, 실제 로드는 아직 fail-closed 상태다. |

pretrained weight 다운로드를 명시적으로 허용할 때:

```yaml
params:
  pretrained: true
  allow_pretrained_download: true
```

다운로드 허용 config는 네트워크, 원격 서버 권한, 모델 라이선스, 캐시 위치 영향을 받는다. 실무 실행에서는 한 번 받은 weight를 명시 checkpoint asset으로 고정하는 편이 더 재현성이 좋다.

## metadata 필드

real/planned model task adapter는 contract metadata에 아래 필드를 기록한다.

| field | meaning |
| --- | --- |
| `weight_policy_version` | weight 정책 스키마 버전 |
| `weight_source` | `explicit_checkpoint`, `pretrained_download_allowed`, `pretrained_download_blocked`, `architecture_only` 중 하나 |
| `checkpoint_path` | config에 들어온 checkpoint 경로 |
| `checkpoint_exists` | 명시 checkpoint가 현재 파일 시스템에 존재하는지 |
| `allow_pretrained_download` | 다운로드 허용 여부 |
| `recommended_weight_roots` | 권장 weight 보관 경로 목록 |

## recommended workflow

1. 모델을 고른다.
2. 해당 모델의 라이선스와 출처를 확인한다.
3. weight를 `models/checkpoints/<model_id>/` 또는 원격 서버의 동일한 의미 경로에 둔다.
4. config에는 `checkpoint`를 명시한다.
5. 실행 후 `task.json` metadata에서 `weight_source=explicit_checkpoint`인지 확인한다.
6. `runs/`나 checkpoint 파일을 Git에 올리지 않는다.

## checkpoint payload

Torchvision/timm train mode가 저장하는 checkpoint payload는 아래 값을 포함한다.

```text
model_id
classes
epoch
metrics
state_dict
optimizer_state_dict
```

Inference는 `state_dict` 또는 `model_state_dict` key를 가진 payload를 읽을 수 있고, raw state dict 파일도 읽을 수 있다. Classifier head는 manifest의 class 수에 맞춰 교체한 뒤 checkpoint를 로드한다.
