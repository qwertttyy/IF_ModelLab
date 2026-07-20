# classifier_stage_1_2_original_aug_mbt_model_opt

Stage 1-2 MBT classifier on original images with weak augmentation in train only.

- Dataset ID: `tank_armor_prepared_v20260706_original_split_mbt_av_aug_weak_v1`
- Input path: `/workspace/ironflow/prestaged/tank_armor_prepared_v20260706_original_split_mbt_av_aug_weak_v1/classifier_mbt/images`
- Input variant: `model_name_classification_images`
- Prediction split: `test`
- Output collection: `full`

## Model-specific parameters

| # | Model | Adapter | Epochs | Batch | Image size | LR | Min LR | Patience | Workers | Basis |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `mobilenet_v3_small` | `torchvision_classifier` | 80 | 64 | 224 | 0.001 | 1e-05 | 12 | 4 | TorchVision MobileNetV3 pretrained weights; mobile/efficient CNN fine-tuning with AdamW + cosine. |
| 2 | `efficientnet_b0` | `torchvision_classifier` | 80 | 48 | 224 | 0.0007 | 1e-06 | 12 | 4 | TorchVision EfficientNet-B0 weights and EfficientNet/timm recipes; moderate LR for fine-tuning. |
| 3 | `convnext_v2_tiny` | `timm_classifier` | 90 | 24 | 224 | 0.0001 | 1e-06 | 14 | 4 | timm ConvNeXt V2 Tiny checkpoint; conservative LR because this model was previously unstable. |
| 4 | `efficientnet_v2_s` | `torchvision_classifier` | 90 | 16 | 384 | 0.0003 | 1e-06 | 14 | 4 | TorchVision EfficientNetV2-S weights; official weight family uses larger evaluation resolution, so use 384. |
| 5 | `yolo26n-cls` | `ultralytics_yolo_classifier` | 100 | 64 | 224 | 0.01 |  | 15 | 2 | Ultralytics classification guidance recommends pretrained YOLO-cls and 100-epoch train examples. |
| 6 | `efficientnet_b3` | `torchvision_classifier` | 90 | 24 | 300 | 0.0005 | 1e-06 | 14 | 4 | TorchVision EfficientNet-B3 weights; larger native input size and lower LR than B0. |
| 7 | `mobilenet_v3_large` | `torchvision_classifier` | 80 | 64 | 224 | 0.001 | 1e-05 | 12 | 4 | TorchVision MobileNetV3 Large pretrained weights; efficient CNN fine-tuning. |
| 8 | `resnet50` | `torchvision_classifier` | 80 | 48 | 224 | 0.001 | 1e-05 | 12 | 4 | TorchVision ResNet-50 pretrained weights; classic CNN baseline with cosine fine-tuning. |
| 9 | `resnext50_32x4d` | `torchvision_classifier` | 90 | 32 | 224 | 0.0007 | 1e-06 | 12 | 4 | TorchVision ResNeXt-50 weights and timm ResNeXt recipe family; slightly longer schedule. |
| 10 | `convnext_small` | `timm_classifier` | 90 | 24 | 224 | 0.0002 | 1e-06 | 14 | 4 | timm ConvNeXt Small checkpoint; modern ConvNet fine-tuning with conservative LR. |
| 11 | `swin_tiny` | `timm_classifier` | 100 | 24 | 224 | 0.0001 | 1e-06 | 15 | 4 | Swin Transformer/timm checkpoint; transformer-style classifier gets longer patience and low LR. |
