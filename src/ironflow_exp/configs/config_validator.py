from dataclasses import dataclass, field
from pathlib import Path

from ironflow_exp.configs.experiment_config import ExperimentConfig
from ironflow_exp.configs.model_config import ModelConfig
from ironflow_exp.core.enums import PreprocessingMode, TaskInputSource


ALLOWED_SCHEMA_VERSIONS = {'0.1'}
ALLOWED_RUNNERS = {'local', 'cloud'}
ALLOWED_DEVICES = {'cpu', 'cuda', 'mps', 'auto'}
ALLOWED_PRECISIONS = {'fp32', 'fp16', 'bf16', 'int8'}
ALLOWED_DATASET_FORMATS = {'class_nested_yolo', 'yolo_detection', 'image_folder', 'coco'}
ALLOWED_LABEL_FORMATS = {
    'yolo_bbox',
    'yolo_seg',
    'coco_detection',
    'coco_segmentation',
    'image_folder',
}
ALLOWED_SPLIT_STRATEGIES = {'existing', 'create', 'existing_or_create', 'group_create'}
ALLOWED_SPLIT_GROUP_KEYS = {
    'sample_id',
    'image_id',
    'source_id',
    'source_type',
    'split_group_id',
    'parent_sample_id',
    'source_class',
}
ALLOWED_PREPROCESSING_MODES = {mode.value for mode in PreprocessingMode}
ALLOWED_TASK_INPUTS = {source.value for source in TaskInputSource}
ALLOWED_CROP_SOURCES = {'gt_bbox', 'detector_bbox'}
ALLOWED_SEGMENTATION_SOURCES = {'none', 'gt_mask', 'pred_mask', 'prompt_mask'}
ALLOWED_SEGMENTATION_OUTPUT_MODES = {'mask', 'foreground', 'seg_crop'}


@dataclass(frozen=True, slots=True)
class ConfigValidationIssue:
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class ConfigValidationResult:
    is_valid: bool
    errors: list[ConfigValidationIssue] = field(default_factory=list)
    warnings: list[ConfigValidationIssue] = field(default_factory=list)


class ConfigValidator:
    def validate(
        self,
        config: ExperimentConfig,
        check_paths: bool = False,
    ) -> ConfigValidationResult:
        errors: list[ConfigValidationIssue] = []
        warnings: list[ConfigValidationIssue] = []

        self._validate_root(config=config, errors=errors)
        self._validate_runtime(config=config, errors=errors)
        self._validate_dataset(config=config, errors=errors, check_paths=check_paths)
        self._validate_preprocessing(config=config, errors=errors)
        self._validate_models(config=config, errors=errors)
        self._validate_evaluation(config=config, warnings=warnings)
        self._validate_export(config=config, errors=errors)

        return ConfigValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    def _validate_root(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        if config.schema_version not in ALLOWED_SCHEMA_VERSIONS:
            self._add_error(errors=errors, field='schema_version', message='unsupported schema version')

        if not config.experiment.name.strip():
            self._add_error(errors=errors, field='experiment.name', message='experiment name is required')

    def _validate_runtime(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        runtime = config.runtime

        if runtime.runner not in ALLOWED_RUNNERS:
            self._add_error(errors=errors, field='runtime.runner', message='unsupported runner')

        if runtime.device not in ALLOWED_DEVICES:
            self._add_error(errors=errors, field='runtime.device', message='unsupported device')

        if runtime.precision not in ALLOWED_PRECISIONS:
            self._add_error(errors=errors, field='runtime.precision', message='unsupported precision')

        if runtime.num_workers < 0:
            self._add_error(errors=errors, field='runtime.num_workers', message='num_workers must be 0 or greater')

    def _validate_dataset(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
        check_paths: bool,
    ) -> None:
        dataset = config.dataset
        split = dataset.split

        if not dataset.dataset_id.strip():
            self._add_error(errors=errors, field='dataset.dataset_id', message='dataset_id is required')

        if not dataset.source_root.strip():
            self._add_error(errors=errors, field='dataset.source_root', message='source_root is required')
        elif check_paths and not Path(dataset.source_root).exists():
            self._add_error(errors=errors, field='dataset.source_root', message='source_root does not exist')

        if dataset.dataset_format not in ALLOWED_DATASET_FORMATS:
            self._add_error(errors=errors, field='dataset.dataset_format', message='unsupported dataset format')

        if dataset.label_format not in ALLOWED_LABEL_FORMATS:
            self._add_error(errors=errors, field='dataset.label_format', message='unsupported label format')

        if split.strategy not in ALLOWED_SPLIT_STRATEGIES:
            self._add_error(errors=errors, field='dataset.split.strategy', message='unsupported split strategy')

        if split.strategy == 'group_create' and not split.group_key:
            self._add_error(errors=errors, field='dataset.split.group_key', message='group_create split requires group_key')
        elif split.group_key is not None and split.group_key not in ALLOWED_SPLIT_GROUP_KEYS:
            self._add_error(errors=errors, field='dataset.split.group_key', message='unsupported split group_key')

        ratio_sum = split.train_ratio + split.val_ratio + split.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            self._add_error(errors=errors, field='dataset.split', message='split ratios must sum to 1.0')

        if min(split.train_ratio, split.val_ratio, split.test_ratio) < 0:
            self._add_error(errors=errors, field='dataset.split', message='split ratios must be 0 or greater')

        if not dataset.class_map:
            self._add_error(errors=errors, field='dataset.class_map', message='class_map is required')

    def _validate_preprocessing(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        preprocessing = config.preprocessing

        if preprocessing.mode not in ALLOWED_PREPROCESSING_MODES:
            self._add_error(errors=errors, field='preprocessing.mode', message='unsupported preprocessing mode')

        self._validate_task_inputs(config=config, errors=errors)

        if preprocessing.crop.source not in ALLOWED_CROP_SOURCES:
            self._add_error(errors=errors, field='preprocessing.crop.source', message='unsupported crop source')

        if preprocessing.crop.padding_ratio < 0:
            self._add_error(errors=errors, field='preprocessing.crop.padding_ratio', message='padding_ratio must be 0 or greater')

        if preprocessing.augmentation.enabled and preprocessing.augmentation.target_split != 'train':
            self._add_error(errors=errors, field='preprocessing.augmentation.target_split', message='target_split must be train')

        if preprocessing.segmentation.source not in ALLOWED_SEGMENTATION_SOURCES:
            self._add_error(errors=errors, field='preprocessing.segmentation.source', message='unsupported segmentation source')

        if preprocessing.segmentation.output_mode not in ALLOWED_SEGMENTATION_OUTPUT_MODES:
            self._add_error(errors=errors, field='preprocessing.segmentation.output_mode', message='unsupported segmentation output mode')

        self._validate_preprocessing_mode_contract(config=config, errors=errors)

    def _validate_task_inputs(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        task_inputs = config.preprocessing.task_inputs
        input_by_field = {
            'preprocessing.task_inputs.detection': task_inputs.detection,
            'preprocessing.task_inputs.classification': task_inputs.classification,
            'preprocessing.task_inputs.segmentation': task_inputs.segmentation,
            'preprocessing.task_inputs.embedding': task_inputs.embedding,
        }

        for field, input_source in input_by_field.items():
            if input_source not in ALLOWED_TASK_INPUTS:
                self._add_error(errors=errors, field=field, message='unsupported task input source')

    def _validate_preprocessing_mode_contract(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        preprocessing = config.preprocessing

        if preprocessing.mode == PreprocessingMode.ORIGINAL.value:
            if preprocessing.crop.enabled or preprocessing.augmentation.enabled or preprocessing.segmentation.enabled:
                self._add_error(errors=errors, field='preprocessing.mode', message='original mode cannot enable crop, augmentation, or segmentation')

        if preprocessing.mode == PreprocessingMode.GT_CROP.value:
            if not preprocessing.crop.enabled or preprocessing.crop.source != 'gt_bbox':
                self._add_error(errors=errors, field='preprocessing.crop', message='gt_crop mode requires crop.source=gt_bbox')

        if preprocessing.mode == PreprocessingMode.DETECTOR_CROP.value:
            if not preprocessing.crop.enabled or preprocessing.crop.source != 'detector_bbox':
                self._add_error(errors=errors, field='preprocessing.crop', message='detector_crop mode requires crop.source=detector_bbox')

        if preprocessing.mode == PreprocessingMode.AUGMENTATION.value and not preprocessing.augmentation.enabled:
            self._add_error(errors=errors, field='preprocessing.augmentation.enabled', message='augmentation mode requires augmentation.enabled=true')

        if preprocessing.mode == PreprocessingMode.SEGMENTATION_MASK.value and not preprocessing.segmentation.enabled:
            self._add_error(errors=errors, field='preprocessing.segmentation.enabled', message='segmentation_mask mode requires segmentation.enabled=true')

        if preprocessing.mode == PreprocessingMode.SEG_CROP.value:
            if not preprocessing.crop.enabled or not preprocessing.segmentation.enabled:
                self._add_error(errors=errors, field='preprocessing.mode', message='seg_crop mode requires crop and segmentation')

    def _validate_models(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        model_by_task = {
            'detection': config.models.detection,
            'classification': config.models.classification,
            'segmentation': config.models.segmentation,
            'embedding': config.models.embedding,
        }

        for task, model in model_by_task.items():
            self._validate_model(task=task, model=model, errors=errors)

        if config.models.embedding.output_dim is not None and config.models.embedding.output_dim <= 0:
            self._add_error(errors=errors, field='models.embedding.output_dim', message='output_dim must be greater than 0')

    def _validate_model(
        self,
        task: str,
        model: ModelConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        if model.enabled:
            if not model.model_id:
                self._add_error(errors=errors, field=f'models.{task}.model_id', message='enabled model requires model_id')

            if not model.adapter:
                self._add_error(errors=errors, field=f'models.{task}.adapter', message='enabled model requires adapter')

        if model.train.epochs is not None and model.train.epochs <= 0:
            self._add_error(errors=errors, field=f'models.{task}.train.epochs', message='epochs must be greater than 0')

        if model.train.image_size is not None and model.train.image_size <= 0:
            self._add_error(errors=errors, field=f'models.{task}.train.image_size', message='image_size must be greater than 0')

        if model.train.batch_size is not None and model.train.batch_size <= 0:
            self._add_error(errors=errors, field=f'models.{task}.train.batch_size', message='batch_size must be greater than 0')

        if model.train.learning_rate is not None and model.train.learning_rate <= 0:
            self._add_error(errors=errors, field=f'models.{task}.train.learning_rate', message='learning_rate must be greater than 0')

        self._validate_predict_config(task=task, model=model, errors=errors)

    def _validate_predict_config(
        self,
        task: str,
        model: ModelConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        predict = model.predict

        if predict.confidence_threshold is not None and not 0 <= predict.confidence_threshold <= 1:
            self._add_error(errors=errors, field=f'models.{task}.predict.confidence_threshold', message='confidence_threshold must be between 0 and 1')

        if predict.iou_threshold is not None and not 0 <= predict.iou_threshold <= 1:
            self._add_error(errors=errors, field=f'models.{task}.predict.iou_threshold', message='iou_threshold must be between 0 and 1')

        if predict.top_k is not None and predict.top_k <= 0:
            self._add_error(errors=errors, field=f'models.{task}.predict.top_k', message='top_k must be greater than 0')

    def _validate_evaluation(
        self,
        config: ExperimentConfig,
        warnings: list[ConfigValidationIssue],
    ) -> None:
        metric_by_task = {
            'detection': config.evaluation.detection,
            'classification': config.evaluation.classification,
            'segmentation': config.evaluation.segmentation,
        }

        for task, metric_config in metric_by_task.items():
            if metric_config.enabled and not metric_config.metrics:
                warnings.append(
                    ConfigValidationIssue(
                        field=f'evaluation.{task}.metrics',
                        message='enabled evaluation has no metric',
                    ),
                )

        if config.evaluation.latency.warmup_runs < 0:
            warnings.append(
                ConfigValidationIssue(
                    field='evaluation.latency.warmup_runs',
                    message='warmup_runs is below 0 and will be treated as invalid by runners',
                ),
            )

    def _validate_export(
        self,
        config: ExperimentConfig,
        errors: list[ConfigValidationIssue],
    ) -> None:
        if not config.export.output_root.strip():
            self._add_error(errors=errors, field='export.output_root', message='output_root is required')

    def _add_error(
        self,
        errors: list[ConfigValidationIssue],
        field: str,
        message: str,
    ) -> None:
        errors.append(ConfigValidationIssue(field=field, message=message))
