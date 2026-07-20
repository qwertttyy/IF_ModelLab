# classifier_refinement_2_2_best_metric_smoke

Stage 2-2 best-metric patch verification config.

- Purpose: verify that `best_metric: macro_f1` and `early_stopping_metric: macro_f1` are connected through train, checkpoint selection, early stopping metadata, predict/test, and collection.
- Base dataset: `tank_armor_prepared_v20260630_aug_weak_v1`
- Input path: `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1/classifier_mbt/crops`
- Representative model: EfficientNetV2-S
- Scope: smoke verification, not a report-grade performance comparison.

| File | Model | Target | Epochs | Batch | Image size | LR | Metric |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `01_mbt_efficientnet_v2_s_best_metric_smoke.yaml` | EfficientNetV2-S | MBT | 3 | 16 | 384 | 0.0002 | macro_f1 |
