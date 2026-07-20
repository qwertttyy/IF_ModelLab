from dataclasses import dataclass, field

from ironflow_exp.engine.core.task_orchestrator import TaskOrchestrationPlan
from ironflow_exp.engine.domain import TaskExecutionRecord
from ironflow_exp.models import (
    ModelRegistry,
    ModelSpec,
    UnknownModelSpecError,
    create_default_model_registry,
)


DETECTION_TO_CLASSIFICATION_CROP_ADAPTER = 'detection_to_classification_crop'
CLASSIFICATION_CROP_AUGMENTATION_ADAPTER = 'classification_crop_augmentation_smoke'
DETECTION_BBOX_AUGMENTATION_ADAPTER = 'detection_bbox_augmentation_smoke'
ALBUMENTATIONS_DETECTION_BBOX_ADAPTER = 'albumentations_detection_bbox'
ALBUMENTATIONS_CLASSIFICATION_CROP_ADAPTER = 'albumentations_classification_crop'
ALBUMENTATIONS_SEGMENTATION_MASK_ADAPTER = 'albumentations_segmentation_mask'
TASK_PRIMARY_OUTPUT_ARTIFACTS = {
    'detection': ('detection_predictions',),
    'tracking': ('tracking_predictions',),
    'classification': ('classification_predictions',),
    'segmentation': ('segmentation_predictions', 'mask_artifacts'),
    'embedding': ('embedding_predictions', 'embeddings', 'embeddings_meta'),
    'augmentation': ('augmentation_manifest',),
}
ADAPTER_OUTPUT_ARTIFACTS = {
    'manifest_detection_smoke': ('detection_predictions',),
    'manifest_segmentation_smoke': ('segmentation_predictions',),
    'ultralytics_yolo': ('detection_predictions',),
    DETECTION_TO_CLASSIFICATION_CROP_ADAPTER: (
        'detection_predictions',
        'crop_region_manifest',
        'classification_input_manifest',
        'crop_images',
    ),
    'manifest_classification_smoke': ('classification_predictions',),
    'torchvision_classifier': ('classification_predictions',),
    'ultralytics_yolo_classifier': ('classification_predictions',),
    'dinov2_embedding': ('embedding_predictions', 'embeddings', 'embeddings_meta'),
    'dinov3_embedding': ('embedding_predictions', 'embeddings', 'embeddings_meta'),
    'clip_embedding': ('embedding_predictions', 'embeddings', 'embeddings_meta'),
    'siglip_embedding': ('embedding_predictions', 'embeddings', 'embeddings_meta'),
    'botsort_tracker': ('tracking_predictions',),
    'ocsort_tracker': ('tracking_predictions',),
    'augmentation_policy_smoke': ('augmentation_manifest',),
    CLASSIFICATION_CROP_AUGMENTATION_ADAPTER: (
        'augmentation_manifest',
        'classification_input_manifest',
        'augmented_images',
    ),
    ALBUMENTATIONS_CLASSIFICATION_CROP_ADAPTER: (
        'augmentation_manifest',
        'classification_input_manifest',
        'augmented_images',
    ),
    DETECTION_BBOX_AUGMENTATION_ADAPTER: (
        'augmentation_manifest',
        'detection_input_manifest',
        'augmented_images',
    ),
    ALBUMENTATIONS_DETECTION_BBOX_ADAPTER: (
        'augmentation_manifest',
        'detection_input_manifest',
        'augmented_images',
    ),
    ALBUMENTATIONS_SEGMENTATION_MASK_ADAPTER: (
        'augmentation_manifest',
        'segmentation_input_manifest',
        'augmented_images',
        'augmented_masks',
    ),
}
RUNNABLE_READINESS_STATES = frozenset({'adapter_ready', 'validated'})
PREPARED_BUT_UNVALIDATED_STATES = frozenset({'prepared'})


@dataclass(frozen=True, slots=True)
class TaskPlanCompatibilityIssue:
    task_id: str
    message: str


@dataclass(frozen=True, slots=True)
class TaskPlanCompatibilityResult:
    errors: list[TaskPlanCompatibilityIssue] = field(default_factory=list)
    warnings: list[TaskPlanCompatibilityIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def error_messages(self) -> list[str]:
        return [
            f'{issue.task_id}: {issue.message}'
            for issue in self.errors
        ]

    def warning_messages(self) -> list[str]:
        return [
            f'{issue.task_id}: {issue.message}'
            for issue in self.warnings
        ]


class TaskPlanCompatibilityValidator:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or create_default_model_registry()

    def validate(self, plan: TaskOrchestrationPlan) -> TaskPlanCompatibilityResult:
        errors: list[TaskPlanCompatibilityIssue] = []
        warnings: list[TaskPlanCompatibilityIssue] = []
        records_by_id = {record.task_id: record for record in plan.records}

        for record in plan.records:
            self._validate_dependencies_exist(record=record, records_by_id=records_by_id, errors=errors)
            self._validate_model_contract(record=record, errors=errors, warnings=warnings)
            self._validate_required_upstream_artifacts(record=record, records_by_id=records_by_id, errors=errors)
            self._validate_detection_crop_transform(record=record, records_by_id=records_by_id, errors=errors)
            self._validate_classification_input(record=record, records_by_id=records_by_id, errors=errors, warnings=warnings)
            self._validate_augmentation_policy(record=record, errors=errors, warnings=warnings)

        return TaskPlanCompatibilityResult(errors=errors, warnings=warnings)

    def _validate_dependencies_exist(
        self,
        *,
        record: TaskExecutionRecord,
        records_by_id: dict[str, TaskExecutionRecord],
        errors: list[TaskPlanCompatibilityIssue],
    ) -> None:
        for dependency_id in record.depends_on:
            if dependency_id not in records_by_id:
                errors.append(
                    TaskPlanCompatibilityIssue(
                        task_id=record.task_id,
                        message=f'unknown dependency: {dependency_id}',
                    ),
                )

    def _validate_model_contract(
        self,
        *,
        record: TaskExecutionRecord,
        errors: list[TaskPlanCompatibilityIssue],
        warnings: list[TaskPlanCompatibilityIssue],
    ) -> None:
        spec = self._model_spec(record=record)
        if spec is None:
            if record.model_id:
                warnings.append(
                    TaskPlanCompatibilityIssue(
                        task_id=record.task_id,
                        message=f'model is not registered in the default compatibility catalog: {record.model_id}',
                    ),
                )
            return

        if record.adapter and not self.model_registry.adapter_matches(
            configured_adapter=record.adapter,
            spec_adapter=spec.adapter_key,
        ):
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message=(
                        f'adapter does not match registered model contract: '
                        f'configured={record.adapter}, expected={spec.adapter_key}'
                    ),
                ),
            )

        primary_outputs = TASK_PRIMARY_OUTPUT_ARTIFACTS.get(record.task_type, ())
        spec_outputs = self._artifact_ids(spec.output_artifacts)
        missing_outputs = [
            artifact
            for artifact in primary_outputs
            if artifact not in spec_outputs
        ]
        if missing_outputs:
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='model contract is missing task output artifacts: ' + ', '.join(missing_outputs),
                ),
            )

        if spec.readiness_status == 'needs_adapter':
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message=f'model contract is registered but runnable adapter is missing: {record.model_id}',
                ),
            )
        elif spec.readiness_status in PREPARED_BUT_UNVALIDATED_STATES:
            warnings.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message=f'model contract is prepared but not adapter-validated yet: {record.model_id}',
                ),
            )

    def _validate_required_upstream_artifacts(
        self,
        *,
        record: TaskExecutionRecord,
        records_by_id: dict[str, TaskExecutionRecord],
        errors: list[TaskPlanCompatibilityIssue],
    ) -> None:
        required = self._required_upstream_artifacts(record=record)
        if not required:
            return

        available = self._available_dependency_artifacts(record=record, records_by_id=records_by_id)
        missing = [
            artifact
            for artifact in required
            if artifact not in available
        ]
        if missing:
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='missing required upstream artifacts: ' + ', '.join(missing),
                ),
            )

    def _validate_detection_crop_transform(
        self,
        *,
        record: TaskExecutionRecord,
        records_by_id: dict[str, TaskExecutionRecord],
        errors: list[TaskPlanCompatibilityIssue],
    ) -> None:
        if record.adapter != DETECTION_TO_CLASSIFICATION_CROP_ADAPTER:
            return
        if not record.depends_on:
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='detection-to-classification crop transform requires a detection dependency',
                ),
            )
            return

        source_task_id = record.params.get('source_task_id')
        dependency_ids = [source_task_id] if isinstance(source_task_id, str) and source_task_id else record.depends_on
        if len(dependency_ids) != 1:
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='detection-to-classification crop transform requires exactly one source detection task',
                ),
            )
            return

        source = records_by_id.get(dependency_ids[0])
        if source is None:
            return
        if source.task_type != 'detection':
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message=f'detection-to-classification crop transform requires detection output, got {source.task_type}',
                ),
            )

    def _required_upstream_artifacts(self, *, record: TaskExecutionRecord) -> tuple[str, ...]:
        if record.adapter == DETECTION_TO_CLASSIFICATION_CROP_ADAPTER:
            return ('detection_predictions',)

        if record.adapter in {CLASSIFICATION_CROP_AUGMENTATION_ADAPTER, ALBUMENTATIONS_CLASSIFICATION_CROP_ADAPTER}:
            return ('classification_input_manifest',)

        if record.task_type == 'detection' and record.input_variant_kind == 'augmentation' and record.depends_on:
            return ('detection_input_manifest',)

        if record.task_type == 'classification' and record.input_variant_kind == 'detector_crop':
            return ('classification_input_manifest',)

        if record.task_type == 'tracking':
            return ('detection_predictions',)

        if record.task_type == 'segmentation' and record.depends_on:
            return ('detection_predictions',)

        if record.task_type == 'embedding' and record.input_variant_kind in {'detector_crop', 'crop'}:
            return ('classification_input_manifest',)

        return ()

    def _validate_augmentation_policy(
        self,
        *,
        record: TaskExecutionRecord,
        errors: list[TaskPlanCompatibilityIssue],
        warnings: list[TaskPlanCompatibilityIssue],
    ) -> None:
        if record.task_type != 'augmentation':
            return

        target_split = record.params.get('target_split', 'train')
        if target_split != 'train':
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='augmentation task target_split must be train',
                ),
            )

        required_true_flags = {
            'preserve_original_sample_id': 'augmentation task must preserve original_sample_id lineage',
            'preserve_split': 'augmentation task must preserve split lineage',
            'record_augmentation_id': 'augmentation task must record augmentation_id',
            'record_augmentation_recipe': 'augmentation task must record augmentation_recipe',
        }
        for key, message in required_true_flags.items():
            if record.params.get(key) is not True:
                errors.append(TaskPlanCompatibilityIssue(task_id=record.task_id, message=message))

        if record.params.get('record_label_transform') is not True:
            warnings.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='augmentation task should record label_transform for bbox/mask/class lineage',
                ),
            )

    def _available_dependency_artifacts(
        self,
        *,
        record: TaskExecutionRecord,
        records_by_id: dict[str, TaskExecutionRecord],
    ) -> set[str]:
        available: set[str] = set()
        for dependency_id in record.depends_on:
            dependency = records_by_id.get(dependency_id)
            if dependency is None:
                continue
            available.update(self._output_artifacts(record=dependency))

        return available

    def _output_artifacts(self, *, record: TaskExecutionRecord) -> tuple[str, ...]:
        adapter_outputs = ADAPTER_OUTPUT_ARTIFACTS.get(record.adapter or '')
        if adapter_outputs is not None:
            return adapter_outputs

        spec = self._model_spec(record=record)
        if spec is not None:
            return self._artifact_ids(spec.output_artifacts)

        return TASK_PRIMARY_OUTPUT_ARTIFACTS.get(record.task_type, ())

    def _artifact_ids(self, artifacts: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self._artifact_id(artifact) for artifact in artifacts)

    def _artifact_id(self, artifact: str) -> str:
        for suffix in ('.json', '.npy', '.csv'):
            if artifact.endswith(suffix):
                return artifact.removesuffix(suffix)

        return artifact

    def _model_spec(self, *, record: TaskExecutionRecord) -> ModelSpec | None:
        if not record.model_id:
            return None
        try:
            return self.model_registry.get_spec(task=record.task_type, model_id=record.model_id)
        except UnknownModelSpecError:
            return None

    def _validate_classification_input(
        self,
        *,
        record: TaskExecutionRecord,
        records_by_id: dict[str, TaskExecutionRecord],
        errors: list[TaskPlanCompatibilityIssue],
        warnings: list[TaskPlanCompatibilityIssue],
    ) -> None:
        if record.task_type != 'classification':
            return

        input_kind = record.input_variant_kind
        if input_kind in {'original', 'crop'}:
            if input_kind == 'original':
                warnings.append(
                    TaskPlanCompatibilityIssue(
                        task_id=record.task_id,
                        message='original image classification should be interpreted separately from object-level crop classification',
                    ),
                )
            return

        if input_kind == 'detector_crop':
            if not self._depends_on_crop_manifest_provider(record=record, records_by_id=records_by_id):
                errors.append(
                    TaskPlanCompatibilityIssue(
                        task_id=record.task_id,
                        message=(
                            'detector_crop classification requires an upstream '
                            'detection_to_classification_crop transform task; '
                            'direct detection -> classification is not supported'
                        ),
                    ),
                )
            return

        if input_kind == 'segmentation_crop':
            errors.append(
                TaskPlanCompatibilityIssue(
                    task_id=record.task_id,
                    message='segmentation_crop classification is not wired yet; add a segmentation-to-classification transform first',
                ),
            )
            return

        errors.append(
            TaskPlanCompatibilityIssue(
                task_id=record.task_id,
                message=f'classification input variant kind is not supported: {input_kind}',
            ),
        )

    def _depends_on_crop_manifest_provider(
        self,
        *,
        record: TaskExecutionRecord,
        records_by_id: dict[str, TaskExecutionRecord],
    ) -> bool:
        for dependency_id in record.depends_on:
            dependency = records_by_id.get(dependency_id)
            if dependency is None:
                continue
            if dependency.adapter == DETECTION_TO_CLASSIFICATION_CROP_ADAPTER:
                return True
            if 'classification_input_manifest' in self._output_artifacts(record=dependency):
                return True

        return False
