# IronFlow 실험 종료 후 시각화 리포트 생성 가이드

## 목적

모든 후보 실험이 끝난 뒤, Colab YOLO 노트북처럼 결과 시각화를 한 번에 모으기 위한 후처리 절차다.

이 도구는 학습 파이프라인에 끼어들지 않는다. 이미 수집된 `runs/e/<experiment_id>` 결과 폴더를 읽어서 별도 리포트 폴더를 만든다.

## 실행 명령

PowerShell에서 프로그램 폴더로 이동한 뒤 실행한다.

```powershell
Set-Location -LiteralPath "D:\Model_LAB\vision_experiment_platform"
python scripts\generate_visual_report.py --runs-root runs\e --output-dir reports\visual_reports\latest --force
```

생성 위치:

```text
D:\Model_LAB\vision_experiment_platform\reports\visual_reports\latest
```

## 생성되는 주요 산출물

```text
reports/visual_reports/latest/
  index.md
  comparison/
    experiment_summary.csv
    detection_map50_95.png
    classification_macro_f1.png
    latency_ms_per_image.png
    fps_by_model.png
    score_vs_latency.png
  detection/<experiment_id>/
    summary.md
    loss_curve.png
    detection_map_curve.png
    precision_recall_curve_by_epoch.png
    assets/
  classification/<experiment_id>/
    summary.md
    loss_curve.png
    classification_score_curve.png
    assets/
```

## 항목별 생성 방식

| 요청 항목 | 생성 방식 |
|---|---|
| epoch별 loss 곡선 | train task의 `metrics.csv`에서 생성 |
| epoch별 mAP 곡선 | detection train task의 `metrics.csv`에서 생성 |
| confusion matrix | YOLO/프레임워크가 만든 이미지가 있으면 `assets/`로 복사 |
| PR/F1/P/R curve | 프레임워크가 만든 이미지가 있으면 `assets/`로 복사 |
| validation 정답-예측 비교 이미지 | `val_batch*_labels`, `val_batch*_pred` 이미지가 있으면 복사 |
| test prediction preview | `test`, `preview`, `pred` 이름의 이미지가 있으면 복사 |
| latency(ms/img) 비교 | root `metrics.csv`의 `latency_ms_per_image` 사용 |
| FPS 비교 | `1000 / latency_ms_per_image`로 계산 |
| speed-accuracy trade-off | primary score와 latency를 scatter plot으로 생성 |

## 주의 사항

- YOLO 계열처럼 프레임워크가 이미 그림을 만든 경우는 원본 그림을 복사한다.
- RF-DETR, D-FINE, classifier처럼 그림 형식이 일정하지 않은 경우는 공통 `metrics.csv` 기반 그래프를 만든다.
- confusion matrix와 PR curve를 직접 재계산하려면 각 모델의 raw prediction과 class별 GT 매칭 포맷이 더 필요하다. 현재 준비 버전은 “있는 그림은 수집하고, 공통 metric 곡선은 재생성”하는 안정형이다.
- 최종 발표용으로 부족한 항목이 있으면, 이 리포트 결과를 보고 raw prediction 포맷이 있는 모델부터 재계산 기능을 추가하면 된다.

## 최종 실험 종료 후 체크

1. 모든 run이 `finished` 또는 분석 가능한 상태인지 확인한다.
2. GUI에서 collect가 끝났는지 확인한다.
3. 위 실행 명령을 실행한다.
4. `reports/visual_reports/latest/index.md`를 열어 전체 결과를 확인한다.
5. `comparison/experiment_summary.csv`를 기준으로 보고서 표를 만든다.
