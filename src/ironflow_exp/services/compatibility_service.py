from pathlib import Path

from ironflow_exp.configs import ExperimentConfig
from ironflow_exp.core.enums import CompatibilityStatus, PreprocessingMode, TaskInputSource
from ironflow_exp.domain import CompatibilityResult


DETECTOR_CROP_ARTIFACTS = [
    'predictions/detection_predictions.json',
    'manifests/object_manifest.json',
    'artifacts/crops/',
]

SEG_CROP_ARTIFACTS = [
    'manifests/object_manifest.json',
    'predictions/segmentation_predictions.json',
    'artifacts/masks/',
    'artifacts/crops/',
]

TASK_INPUT_ALLOWLIST = {
    'detection': {
        TaskInputSource.NONE.value,
        TaskInputSource.ORIGINAL.value,
        TaskInputSource.AUGMENTATION.value,
        TaskInputSource.SEGMENTATION_MASK.value,
    },
    'classification': {
        TaskInputSource.NONE.value,
        TaskInputSource.ORIGINAL.value,
        TaskInputSource.GT_CROP.value,
        TaskInputSource.DETECTOR_CROP.value,
        TaskInputSource.SEG_CROP.value,
        TaskInputSource.AUGMENTATION.value,
    },
    'segmentation': {
        TaskInputSource.NONE.value,
        TaskInputSource.ORIGINAL.value,
        TaskInputSource.GT_CROP.value,
        TaskInputSource.DETECTOR_CROP.value,
    },
    'embedding': {
        TaskInputSource.NONE.value,
        TaskInputSource.ORIGINAL.value,
        TaskInputSource.GT_CROP.value,
        TaskInputSource.DETECTOR_CROP.value,
        TaskInputSource.SEGMENTATION_MASK.value,
        TaskInputSource.SEG_CROP.value,
    },
}


class CompatibilityService:
    def validate(
        self,
        config: ExperimentConfig,
        artifact_root: str | Path | None = None,
    ) -> CompatibilityResult:
        errors: list[str] = []
        warnings: list[str] = []
        required_artifacts: list[str] = []

        self._validate_task_inputs(config=config, errors=errors)
        self._validate_enabled_tasks(config=config, errors=errors)
        self._validate_preprocessing(config=config, errors=errors, warnings=warnings)
        self._validate_artifact_requirements(
            config=config,
            artifact_root=artifact_root,
            errors=errors,
            required_artifacts=required_artifacts,
        )
        self._validate_metric_interpretation(config=config, warnings=warnings)

        if errors:
            return CompatibilityResult(
                status=CompatibilityStatus.BLOCKED.value,
                code='COMPATIBILITY_BLOCKED',
                message='; '.join(errors),
                required_artifacts=required_artifacts,
                warnings=warnings,
            )

        if warnings or required_artifacts:
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING.value,
                code='COMPATIBILITY_WARNING',
                message='configuration is runnable with warnings',
                required_artifacts=required_artifacts,
                warnings=warnings,
            )

        return CompatibilityResult()

    def _validate_task_inputs(
        self,
        config: ExperimentConfig,
        errors: list[str],
    ) -> None:
        task_inputs = config.preprocessing.task_inputs
        input_by_task = {
            'detection': task_inputs.detection,
            'classification': task_inputs.classification,
            'segmentation': task_inputs.segmentation,
            'embedding': task_inputs.embedding,
        }

        for task, input_source in input_by_task.items():
            allowed_inputs = TASK_INPUT_ALLOWLIST[task]
            if input_source not in allowed_inputs:
                errors.append(f'{task} input source {input_source} is not supported')

    def _validate_enabled_tasks(
        self,
        config: ExperimentConfig,
        errors: list[str],
    ) -> None:
        task_inputs = config.preprocessing.task_inputs

        if config.models.detection.enabled and task_inputs.detection == TaskInputSource.NONE.value:
            errors.append('detection is enabled but task_inputs.detection is none')

        if config.models.classification.enabled and task_inputs.classification == TaskInputSource.NONE.value:
            errors.append('classification is enabled but task_inputs.classification is none')

        if config.models.segmentation.enabled and task_inputs.segmentation == TaskInputSource.NONE.value:
            errors.append('segmentation is enabled but task_inputs.segmentation is none')

        if config.models.embedding.enabled and task_inputs.embedding == TaskInputSource.NONE.value:
            errors.append('embedding is enabled but task_inputs.embedding is none')

    def _validate_preprocessing(
        self,
        config: ExperimentConfig,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        preprocessing = config.preprocessing

        if preprocessing.mode == PreprocessingMode.MULTI_TASK.value:
            warnings.append('multi_task mode requires task-level artifact validation')

        if preprocessing.task_inputs.classification == TaskInputSource.GT_CROP.value:
            if not preprocessing.crop.enabled or preprocessing.crop.source != 'gt_bbox':
                errors.append('GT crop classification requires crop.enabled=true and crop.source=gt_bbox')

        if preprocessing.task_inputs.classification == TaskInputSource.DETECTOR_CROP.value:
            if not config.models.detection.enabled:
                errors.append('detector crop classification requires detection task')

        if preprocessing.task_inputs.classification == TaskInputSource.SEG_CROP.value:
            if not preprocessing.segmentation.enabled or preprocessing.segmentation.output_mode != 'seg_crop':
                errors.append('seg crop classification requires segmentation.enabled=true and output_mode=seg_crop')

        if preprocessing.augmentation.enabled and preprocessing.augmentation.target_split != 'train':
            errors.append('augmentation target_split must be train')

    def _validate_artifact_requirements(
        self,
        config: ExperimentConfig,
        artifact_root: str | Path | None,
        errors: list[str],
        required_artifacts: list[str],
    ) -> None:
        classification_input = config.preprocessing.task_inputs.classification

        if classification_input == TaskInputSource.DETECTOR_CROP.value:
            self._collect_missing_artifacts(
                artifact_root=artifact_root,
                artifact_paths=DETECTOR_CROP_ARTIFACTS,
                errors=errors,
                required_artifacts=required_artifacts,
                error_message='detector crop classification requires detection artifacts',
            )

        if classification_input == TaskInputSource.SEG_CROP.value:
            self._collect_missing_artifacts(
                artifact_root=artifact_root,
                artifact_paths=SEG_CROP_ARTIFACTS,
                errors=errors,
                required_artifacts=required_artifacts,
                error_message='seg crop classification requires segmentation artifacts',
            )

    def _collect_missing_artifacts(
        self,
        artifact_root: str | Path | None,
        artifact_paths: list[str],
        errors: list[str],
        required_artifacts: list[str],
        error_message: str,
    ) -> None:
        if artifact_root is None:
            required_artifacts.extend(artifact_paths)
            return

        root_path = Path(artifact_root)
        missing_paths = [
            artifact_path
            for artifact_path in artifact_paths
            if not (root_path / artifact_path).exists()
        ]

        if missing_paths:
            required_artifacts.extend(missing_paths)
            errors.append(error_message)

    def _validate_metric_interpretation(
        self,
        config: ExperimentConfig,
        warnings: list[str],
    ) -> None:
        classification_input = config.preprocessing.task_inputs.classification
        segmentation_source = config.preprocessing.segmentation.source
        segmentation_metrics = set(config.evaluation.segmentation.metrics)

        if classification_input == TaskInputSource.ORIGINAL.value:
            warnings.append('original image classification must be interpreted separately from object-level classification')

        if classification_input == TaskInputSource.DETECTOR_CROP.value:
            warnings.append('detector crop classification includes detector localization error')

        if classification_input == TaskInputSource.SEG_CROP.value:
            warnings.append('seg crop classification depends on mask quality')

        if segmentation_source == 'prompt_mask' and {'iou', 'dice', 'mask_map'} & segmentation_metrics:
            warnings.append('prompt segmentation metrics require GT mask or validated reference mask')
