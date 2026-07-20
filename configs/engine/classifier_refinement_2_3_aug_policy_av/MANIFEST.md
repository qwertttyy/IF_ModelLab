# Refinement 2-3 AV Augmentation Policy Configs

Model-specific hyperparameters are copied from refinement 2-1.
Only dataset root, experiment id, and stage metadata are changed.

| order | policy | model | config | remote dataset root |
|---:|---|---|---|---|
| 1 | `aug_none` | `convnext_small` | `configs/engine/classifier_refinement_2_3_aug_policy_av/01_av_aug_none_convnext_small.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_none_v1` |
| 2 | `aug_none` | `convnext_v2_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_av/02_av_aug_none_convnext_v2_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_none_v1` |
| 3 | `aug_none` | `efficientnet_v2_s` | `configs/engine/classifier_refinement_2_3_aug_policy_av/03_av_aug_none_efficientnet_v2_s.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_none_v1` |
| 4 | `aug_current_weak` | `convnext_small` | `configs/engine/classifier_refinement_2_3_aug_policy_av/04_av_aug_current_weak_convnext_small.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1` |
| 5 | `aug_current_weak` | `convnext_v2_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_av/05_av_aug_current_weak_convnext_v2_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1` |
| 6 | `aug_current_weak` | `efficientnet_v2_s` | `configs/engine/classifier_refinement_2_3_aug_policy_av/06_av_aug_current_weak_efficientnet_v2_s.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1` |
| 7 | `aug_color_safe` | `convnext_small` | `configs/engine/classifier_refinement_2_3_aug_policy_av/07_av_aug_color_safe_convnext_small.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_color_safe_v1` |
| 8 | `aug_color_safe` | `convnext_v2_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_av/08_av_aug_color_safe_convnext_v2_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_color_safe_v1` |
| 9 | `aug_color_safe` | `efficientnet_v2_s` | `configs/engine/classifier_refinement_2_3_aug_policy_av/09_av_aug_color_safe_efficientnet_v2_s.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_color_safe_v1` |
