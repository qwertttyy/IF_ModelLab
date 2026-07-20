# AV Crop Classifier Top5 Weak Augmentation

Generated for dataset: `tank_armor_prepared_v20260705_av_crop_aug_weak_v1`

Purpose: run the same top5 classifier models from the previous MBT crop weak-augmentation experiment on AV-only crop data.

Dataset path on Vast:

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260705_av_crop_aug_weak_v1/classifier_av/crops
```

Model order follows `reports/aug_top5_visual_report_20260704/classifier_aug_top5_report_20260704.md`:

1. EfficientNet-B3
2. EfficientNet-B0
3. MobileNetV3-Large
4. EfficientNetV2-S
5. ResNet50

Common settings:

- epochs=50
- batch_size=32
- image_size=224
- scheduler=cosine
- early_stopping_patience=10
- prediction_split=test
- artifact_collection_mode=full

Model-specific learning rates:

- EfficientNet-B3: learning_rate=0.0005, min_learning_rate=0.000001
- EfficientNet-B0: learning_rate=0.0007, min_learning_rate=0.00001
- MobileNetV3-Large: learning_rate=0.0008, min_learning_rate=0.00001
- EfficientNetV2-S: learning_rate=0.0003, min_learning_rate=0.000001
- ResNet50: learning_rate=0.0005, min_learning_rate=0.00001
