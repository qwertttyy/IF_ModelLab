from dataclasses import dataclass, field

from ironflow_exp.core.enums import PreprocessingMode, TaskInputSource


@dataclass(frozen=True, slots=True)
class TaskInputConfig:
    detection: str = TaskInputSource.NONE.value
    classification: str = TaskInputSource.NONE.value
    segmentation: str = TaskInputSource.NONE.value
    embedding: str = TaskInputSource.NONE.value


@dataclass(frozen=True, slots=True)
class CropConfig:
    enabled: bool = False
    source: str = 'gt_bbox'
    padding_ratio: float = 0.08


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    enabled: bool = False
    target_split: str = 'train'
    policy_id: str = 'none'


@dataclass(frozen=True, slots=True)
class SegmentationPreprocessConfig:
    enabled: bool = False
    source: str = 'none'
    output_mode: str = 'mask'


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    mode: str = PreprocessingMode.ORIGINAL.value
    task_inputs: TaskInputConfig = field(default_factory=TaskInputConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    segmentation: SegmentationPreprocessConfig = field(default_factory=SegmentationPreprocessConfig)
