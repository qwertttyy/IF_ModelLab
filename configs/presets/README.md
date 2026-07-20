# Preset Configs

이 폴더는 GUI/CLI/runner가 공유할 실험 config 예시를 보관한다.

## v0.1 preset

| 파일 | 목적 |
|---|---|
| `original_yolo11n_mobilenetv3_small.yaml` | 사람이 수정하기 쉬운 기본 실험 config |
| `original_yolo11n_mobilenetv3_small.json` | GUI/worker round-trip 확인용 동일 config |

이 preset은 최종 모델 조합 범위를 제한하지 않는다. v0.1에서는 platform 흐름을 검증하기 위해 YOLO11n detection과 MobileNetV3 Small classification 조합을 첫 실행 대상으로 둔다.

## 실행 예정 형태

```powershell
python -m ironflow_exp.runners.cli --config configs/presets/original_yolo11n_mobilenetv3_small.yaml
```
