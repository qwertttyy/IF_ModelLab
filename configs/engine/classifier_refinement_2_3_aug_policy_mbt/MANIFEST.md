# Refinement 2-3 MBT Augmentation Policy Configs

Model-specific hyperparameters are copied from refinement 2-1.
Only dataset root, experiment id, and stage metadata are changed.

| order | policy | model | config | remote dataset root |
|---:|---|---|---|---|
| 1 | `aug_none` | `efficientnet_v2_s` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/01_mbt_aug_none_efficientnet_v2_s.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_none_v1` |
| 2 | `aug_none` | `swin_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/02_mbt_aug_none_swin_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_none_v1` |
| 3 | `aug_none` | `convnext_v2_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/03_mbt_aug_none_convnext_v2_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_none_v1` |
| 4 | `aug_current_weak` | `efficientnet_v2_s` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/04_mbt_aug_current_weak_efficientnet_v2_s.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1` |
| 5 | `aug_current_weak` | `swin_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/05_mbt_aug_current_weak_swin_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1` |
| 6 | `aug_current_weak` | `convnext_v2_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/06_mbt_aug_current_weak_convnext_v2_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1` |
| 7 | `aug_color_safe` | `efficientnet_v2_s` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/07_mbt_aug_color_safe_efficientnet_v2_s.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_color_safe_v1` |
| 8 | `aug_color_safe` | `swin_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/08_mbt_aug_color_safe_swin_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_color_safe_v1` |
| 9 | `aug_color_safe` | `convnext_v2_tiny` | `configs/engine/classifier_refinement_2_3_aug_policy_mbt/09_mbt_aug_color_safe_convnext_v2_tiny.yaml` | `/workspace/ironflow/prestaged/tank_armor_prepared_v20260708_crop_color_safe_v1` |
