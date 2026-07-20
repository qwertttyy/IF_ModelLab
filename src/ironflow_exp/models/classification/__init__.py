"""Classification model adapters"""

from ironflow_exp.models.classification.torchvision_adapter import (
    SUPPORTED_TORCHVISION_CLASSIFIER_MODEL_IDS,
    TORCHVISION_CLASSIFIER_SPECS,
    MissingTorchvisionDependencyError,
    TorchvisionClassificationAdapter,
    UnsupportedTorchvisionClassifierError,
)
from ironflow_exp.models.classification.timm_adapter import (
    SUPPORTED_TIMM_CLASSIFIER_MODEL_IDS,
    TIMM_CLASSIFIER_SPECS,
    MissingTimmDependencyError,
    TimmClassificationAdapter,
    TimmClassifierReference,
    UnsupportedTimmClassifierError,
)
from ironflow_exp.models.classification.ultralytics_yolo_classifier_adapter import (
    SUPPORTED_ULTRALYTICS_YOLO_CLASSIFIER_MODEL_IDS,
    ULTRALYTICS_YOLO_CLASSIFIER_FAMILIES,
    ULTRALYTICS_YOLO_CLASSIFIER_SCALES,
    UltralyticsYoloClassificationAdapter,
    UltralyticsYoloClassifierReference,
    UnsupportedUltralyticsYoloClassifierError,
)


__all__ = [
    'MissingTorchvisionDependencyError',
    'MissingTimmDependencyError',
    'SUPPORTED_TIMM_CLASSIFIER_MODEL_IDS',
    'SUPPORTED_TORCHVISION_CLASSIFIER_MODEL_IDS',
    'SUPPORTED_ULTRALYTICS_YOLO_CLASSIFIER_MODEL_IDS',
    'TIMM_CLASSIFIER_SPECS',
    'TorchvisionClassificationAdapter',
    'TORCHVISION_CLASSIFIER_SPECS',
    'TimmClassificationAdapter',
    'TimmClassifierReference',
    'ULTRALYTICS_YOLO_CLASSIFIER_FAMILIES',
    'ULTRALYTICS_YOLO_CLASSIFIER_SCALES',
    'UltralyticsYoloClassificationAdapter',
    'UltralyticsYoloClassifierReference',
    'UnsupportedTimmClassifierError',
    'UnsupportedTorchvisionClassifierError',
    'UnsupportedUltralyticsYoloClassifierError',
]
