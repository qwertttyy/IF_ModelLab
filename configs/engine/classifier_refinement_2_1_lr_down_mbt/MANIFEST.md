# classifier_refinement_2_1_lr_down_mbt

Stage 2-1 MBT classifier LR-down fine-tuning configs.

- Base dataset: `tank_armor_prepared_v20260630_aug_weak_v1`
- Input path: `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1/classifier_mbt/crops`
- Base configs: `classifier_stage_1_4_crop_aug_mbt_model_opt`
- Change: keep image size/batch/epochs/patience/scheduler/checkpoint, lower only learning rate.

| File | Model | Base lr | 2-1 lr |
| --- | --- | ---: | ---: |
| `01_mbt_efficientnet_v2_s_lr_down.yaml` | EfficientNetV2-S | 0.0003 | 0.0002 |
| `02_mbt_swin_tiny_lr_down.yaml` | Swin Tiny | 0.0001 | 0.00005 |
| `03_mbt_convnext_v2_tiny_lr_down.yaml` | ConvNeXt V2 Tiny | 0.0001 | 0.00005 |
