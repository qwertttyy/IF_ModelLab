from ironflow_exp.engine.tasks.augmentation.albumentations_crop import AlbumentationsClassificationCropTaskAdapter
from ironflow_exp.engine.tasks.augmentation.bbox import (
    AlbumentationsDetectionBBoxTaskAdapter,
    DetectionBBoxAugmentationSmokeTaskAdapter,
)
from ironflow_exp.engine.tasks.augmentation.crop import ClassificationCropAugmentationSmokeTaskAdapter
from ironflow_exp.engine.tasks.augmentation.policy import AugmentationPolicySmokeTaskAdapter
from ironflow_exp.engine.tasks.augmentation.segmentation_mask import AlbumentationsSegmentationMaskTaskAdapter

__all__ = [
    'AlbumentationsClassificationCropTaskAdapter',
    'AlbumentationsDetectionBBoxTaskAdapter',
    'AlbumentationsSegmentationMaskTaskAdapter',
    'AugmentationPolicySmokeTaskAdapter',
    'ClassificationCropAugmentationSmokeTaskAdapter',
    'DetectionBBoxAugmentationSmokeTaskAdapter',
]
