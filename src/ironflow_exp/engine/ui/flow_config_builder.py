from dataclasses import dataclass, replace

from ironflow_exp.engine.configs import (
    EngineDataVariantConfig,
    EngineExperimentConfig,
    EngineTaskConfig,
)
from ironflow_exp.engine.ui.augmentation_policy_catalog import (
    AUGMENTATION_POLICY_NONE,
    augmentation_policy_by_id,
)
from ironflow_exp.engine.ui.model_adapter_catalog import (
    READINESS_INVALID_CHAIN,
    READINESS_NEEDS_ADAPTER,
    ModelAdapterCandidate,
    candidate_by_id,
)


@dataclass(frozen=True, slots=True)
class DetectCropClassifySelection:
    detector_candidate_id: str
    crop_adapter_candidate_id: str
    classifier_candidate_id: str
    source_variant_id: str = 'original'
    source_variant_path: str | None = None
    detector_crop_variant_id: str = 'detector_crop'
    augmented_source_variant_id: str | None = None
    augmented_detector_crop_variant_id: str | None = None
    detector_task_id: str = 'detect_original'
    crop_task_id: str = 'make_detector_crops'
    classifier_task_id: str = 'classify_detector_crops'
    augmentation_policy_id: str = AUGMENTATION_POLICY_NONE
    augmentation_task_id: str | None = None
    allow_planned: bool = False


def build_detect_crop_classify_config(
    *,
    base_config: EngineExperimentConfig,
    selection: DetectCropClassifySelection,
) -> EngineExperimentConfig:
    detector = _candidate(
        candidate_id=selection.detector_candidate_id,
        expected_task_type='detection',
        expected_input_kind='original',
        allow_planned=selection.allow_planned,
    )
    crop_adapter = _candidate(
        candidate_id=selection.crop_adapter_candidate_id,
        expected_task_type='preprocessing',
        expected_input_kind='original',
        allow_planned=selection.allow_planned,
    )
    classifier = _candidate(
        candidate_id=selection.classifier_candidate_id,
        expected_task_type='classification',
        expected_input_kind='detector_crop',
        allow_planned=selection.allow_planned,
    )
    augmentation_policy = augmentation_policy_by_id(selection.augmentation_policy_id)
    augmented_source_variant_id = selection.augmented_source_variant_id or f'{selection.source_variant_id}_augmented'
    augmented_detector_crop_variant_id = (
        selection.augmented_detector_crop_variant_id
        or f'{selection.detector_crop_variant_id}_augmented'
    )

    data_variants = [
        EngineDataVariantConfig(
            id=selection.source_variant_id,
            kind='original',
            path=selection.source_variant_path,
            enabled=True,
        ),
        EngineDataVariantConfig(
            id=selection.detector_crop_variant_id,
            kind='detector_crop',
            source=selection.source_variant_id,
            enabled=True,
        ),
    ]
    detector_input_variant_id = selection.source_variant_id
    detector_depends_on: list[str] = []
    crop_depends_on = [selection.detector_task_id]
    classifier_input_variant_id = selection.detector_crop_variant_id
    classifier_depends_on = [selection.crop_task_id]
    augmentation_tasks: list[EngineTaskConfig] = []
    if augmentation_policy.mode == 'detection_bbox':
        if augmentation_policy.adapter is None:
            raise ValueError('detection bbox augmentation policy must define an adapter')
        data_variants.append(
            EngineDataVariantConfig(
                id=augmented_source_variant_id,
                kind='augmentation',
                source=selection.source_variant_id,
                enabled=True,
            ),
        )
        augmentation_task_id = selection.augmentation_task_id or 'augment_detection_bboxes'
        augmentation_tasks.append(
            EngineTaskConfig(
                id=augmentation_task_id,
                task_type='augmentation',
                input_variant=selection.source_variant_id,
                adapter=augmentation_policy.adapter,
                enabled=True,
                params=dict(augmentation_policy.task_params),
            ),
        )
        detector_input_variant_id = augmented_source_variant_id
        detector_depends_on = [augmentation_task_id]
        crop_depends_on = [selection.detector_task_id, augmentation_task_id]
    elif augmentation_policy.mode == 'classification_crop':
        if augmentation_policy.adapter is None:
            raise ValueError('classification crop augmentation policy must define an adapter')
        data_variants.append(
            EngineDataVariantConfig(
                id=augmented_detector_crop_variant_id,
                kind='detector_crop',
                source=selection.detector_crop_variant_id,
                enabled=True,
            ),
        )
        augmentation_task_id = selection.augmentation_task_id or 'augment_detector_crops'
        augmentation_tasks.append(
            EngineTaskConfig(
                id=augmentation_task_id,
                task_type='augmentation',
                input_variant=selection.detector_crop_variant_id,
                adapter=augmentation_policy.adapter,
                enabled=True,
                depends_on=[selection.crop_task_id],
                params=dict(augmentation_policy.task_params),
            ),
        )
        classifier_input_variant_id = augmented_detector_crop_variant_id
        classifier_depends_on = [augmentation_task_id]
    elif not augmentation_policy.is_noop:
        raise ValueError(f'unsupported augmentation policy mode: {augmentation_policy.mode}')

    tasks: list[EngineTaskConfig] = []
    if augmentation_policy.mode == 'detection_bbox':
        tasks.extend(augmentation_tasks)
    tasks.extend([
        EngineTaskConfig(
            id=selection.detector_task_id,
            task_type='detection',
            input_variant=detector_input_variant_id,
            model_id=detector.model_id,
            adapter=detector.adapter,
            enabled=True,
            depends_on=detector_depends_on,
            params=dict(detector.task_params),
        ),
        EngineTaskConfig(
            id=selection.crop_task_id,
            task_type='preprocessing',
            input_variant=detector_input_variant_id,
            adapter=crop_adapter.adapter,
            enabled=True,
            depends_on=crop_depends_on,
            params={
                'source_task_id': selection.detector_task_id,
                'input_source': 'detector_crop',
                'image_format': 'jpg',
                'padding_ratio': 0.0,
                'min_score': 0.0,
                'use_source_labels': True,
                'classification_dataset_id': f'{selection.source_variant_id}_detector_crops',
            },
        ),
    ])
    if augmentation_policy.mode == 'classification_crop':
        tasks.extend(augmentation_tasks)
    tasks.append(
        EngineTaskConfig(
            id=selection.classifier_task_id,
            task_type='classification',
            input_variant=classifier_input_variant_id,
            model_id=classifier.model_id,
            adapter=classifier.adapter,
            enabled=True,
            depends_on=classifier_depends_on,
            params=dict(classifier.task_params),
        ),
    )

    return replace(
        base_config,
        data_variants=data_variants,
        tasks=tasks,
    )


def _candidate(
    *,
    candidate_id: str,
    expected_task_type: str,
    expected_input_kind: str,
    allow_planned: bool,
) -> ModelAdapterCandidate:
    candidate = candidate_by_id(candidate_id)
    if candidate.task_type != expected_task_type:
        raise ValueError(
            f'candidate {candidate_id} must be task_type={expected_task_type}, got {candidate.task_type}',
        )
    if expected_input_kind not in candidate.supported_input_kinds:
        raise ValueError(
            f'candidate {candidate_id} does not support input kind: {expected_input_kind}',
        )
    if not allow_planned and not candidate.is_runnable:
        raise ValueError(
            f'candidate {candidate_id} is not currently runnable: '
            f'availability={candidate.availability}; readiness={candidate.readiness_label}',
        )
    if candidate.readiness_status in {READINESS_NEEDS_ADAPTER, READINESS_INVALID_CHAIN}:
        raise ValueError(
            f'candidate {candidate_id} is not currently runnable: readiness={candidate.readiness_label}',
        )

    return candidate
