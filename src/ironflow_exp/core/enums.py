from enum import Enum


class TaskType(str, Enum):
    DETECTION = 'detection'
    TRACKING = 'tracking'
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'
    EMBEDDING = 'embedding'
    PREPROCESSING = 'preprocessing'


class PreprocessingMode(str, Enum):
    ORIGINAL = 'original'
    GT_CROP = 'gt_crop'
    DETECTOR_CROP = 'detector_crop'
    AUGMENTATION = 'augmentation'
    SEGMENTATION_MASK = 'segmentation_mask'
    SEG_CROP = 'seg_crop'
    MULTI_TASK = 'multi_task'


class TaskInputSource(str, Enum):
    NONE = 'none'
    ORIGINAL = 'original'
    GT_CROP = 'gt_crop'
    DETECTOR_CROP = 'detector_crop'
    AUGMENTATION = 'augmentation'
    SEGMENTATION_MASK = 'segmentation_mask'
    SEG_CROP = 'seg_crop'


class CompatibilityStatus(str, Enum):
    ALLOWED = 'allowed'
    WARNING = 'warning'
    BLOCKED = 'blocked'


class JobStatus(str, Enum):
    CREATED = 'created'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILED = 'failed'
    CANCELED = 'canceled'
    SKIPPED = 'skipped'


class ArtifactRole(str, Enum):
    CHECKPOINT = 'checkpoint'
    CROP = 'crop'
    MASK = 'mask'
    VISUALIZATION = 'visualization'
    METRIC = 'metric'
    REPORT = 'report'
    BUNDLE = 'bundle'
    MANIFEST = 'manifest'
    PREDICTION = 'prediction'
