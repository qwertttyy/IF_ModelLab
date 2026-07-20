from dataclasses import dataclass

from ironflow_exp.engine.configs import EngineDataVariantConfig, EngineExperimentConfig, EngineTaskConfig
from ironflow_exp.engine.ui.augmentation_policy_catalog import (
    AUGMENTATION_POLICY_NONE,
    AugmentationPolicyCandidate,
    all_augmentation_policies,
    augmentation_policy_by_id,
)
from ironflow_exp.engine.ui.flow_config_builder import (
    DetectCropClassifySelection,
    build_detect_crop_classify_config,
)
from ironflow_exp.engine.ui.model_adapter_catalog import (
    RUNTIME_LOCAL,
    ModelAdapterCandidate,
    candidate_by_id,
    candidates_for,
)


@dataclass(frozen=True, slots=True)
class CandidateChoice:
    label: str
    candidate_id: str


@dataclass(frozen=True, slots=True)
class DetectCropClassifyChoices:
    detectors: tuple[CandidateChoice, ...]
    crop_adapters: tuple[CandidateChoice, ...]
    classifiers: tuple[CandidateChoice, ...]
    augmentation_policies: tuple[CandidateChoice, ...]


@dataclass(frozen=True, slots=True)
class DetectCropClassifyCandidateSelection:
    detector_candidate_id: str
    crop_adapter_candidate_id: str
    classifier_candidate_id: str
    augmentation_policy_id: str = AUGMENTATION_POLICY_NONE


def available_detect_crop_classify_choices(
    *,
    runtime_target: str = RUNTIME_LOCAL,
) -> DetectCropClassifyChoices:
    return DetectCropClassifyChoices(
        detectors=_candidate_choices(
            task_type='detection',
            input_kind='original',
            runtime_target=runtime_target,
        ),
        crop_adapters=_candidate_choices(
            task_type='preprocessing',
            input_kind='original',
            runtime_target=runtime_target,
        ),
        classifiers=_candidate_choices(
            task_type='classification',
            input_kind='detector_crop',
            runtime_target=runtime_target,
        ),
        augmentation_policies=_augmentation_policy_choices(),
    )


def default_detect_crop_classify_selection(
    *,
    choices: DetectCropClassifyChoices | None = None,
) -> DetectCropClassifyCandidateSelection:
    choices = choices or available_detect_crop_classify_choices()
    if not choices.detectors:
        raise ValueError('no runnable detector candidates are available')
    if not choices.crop_adapters:
        raise ValueError('no runnable crop adapter candidates are available')
    if not choices.classifiers:
        raise ValueError('no runnable detector-crop classifier candidates are available')

    return DetectCropClassifyCandidateSelection(
        detector_candidate_id=choices.detectors[0].candidate_id,
        crop_adapter_candidate_id=choices.crop_adapters[0].candidate_id,
        classifier_candidate_id=choices.classifiers[0].candidate_id,
        augmentation_policy_id=choices.augmentation_policies[0].candidate_id,
    )


def choice_map(choices: tuple[CandidateChoice, ...]) -> dict[str, str]:
    return {
        choice.label: choice.candidate_id
        for choice in choices
    }


def label_for_candidate(candidate: ModelAdapterCandidate) -> str:
    return f'{candidate.label} [{candidate.readiness_label}] ({candidate.candidate_id})'


def label_for_candidate_id(candidate_id: str) -> str:
    return label_for_candidate(candidate_by_id(candidate_id))


def label_for_augmentation_policy(policy: AugmentationPolicyCandidate) -> str:
    return f'{policy.label} ({policy.policy_id})'


def label_for_augmentation_policy_id(policy_id: str) -> str:
    return label_for_augmentation_policy(augmentation_policy_by_id(policy_id))


def describe_detect_crop_classify_selection(selection: DetectCropClassifyCandidateSelection) -> str:
    detector = candidate_by_id(selection.detector_candidate_id)
    crop_adapter = candidate_by_id(selection.crop_adapter_candidate_id)
    classifier = candidate_by_id(selection.classifier_candidate_id)
    augmentation_policy = augmentation_policy_by_id(selection.augmentation_policy_id)

    parts = [
        f'detection/original: {detector.model_id or detector.adapter}'
    ]
    if augmentation_policy.mode == 'detection_bbox':
        parts = [
            f'augmentation/original: {augmentation_policy.adapter}',
            f'detection/augmentation: {detector.model_id or detector.adapter}',
        ]
    crop_input_kind = 'augmentation' if augmentation_policy.mode == 'detection_bbox' else 'original'
    parts.append(f'preprocessing/{crop_input_kind}: {crop_adapter.adapter}')
    if augmentation_policy.mode == 'classification_crop':
        parts.append(f'augmentation/detector_crop: {augmentation_policy.adapter}')
    parts.append(f'classification/detector_crop: {classifier.model_id or classifier.adapter}')

    return ' -> '.join(parts)


def build_detect_crop_classify_config_from_selection(
    *,
    base_config: EngineExperimentConfig,
    selection: DetectCropClassifyCandidateSelection,
    source_variant_path_override: str | None = None,
) -> EngineExperimentConfig:
    source_variant = _first_variant(config=base_config, kind='original')
    detector_crop_variant = _first_variant(config=base_config, kind='detector_crop')
    detector_task = _first_task(config=base_config, task_type='detection')
    crop_task = _first_task(config=base_config, task_type='preprocessing')
    classifier_task = _first_task(config=base_config, task_type='classification')

    return build_detect_crop_classify_config(
        base_config=base_config,
        selection=DetectCropClassifySelection(
            detector_candidate_id=selection.detector_candidate_id,
            crop_adapter_candidate_id=selection.crop_adapter_candidate_id,
            classifier_candidate_id=selection.classifier_candidate_id,
            augmentation_policy_id=selection.augmentation_policy_id,
            source_variant_id=source_variant.id,
            source_variant_path=source_variant_path_override or source_variant.path,
            detector_crop_variant_id=detector_crop_variant.id,
            detector_task_id=detector_task.id,
            crop_task_id=crop_task.id,
            classifier_task_id=classifier_task.id,
        ),
    )


def _candidate_choices(
    *,
    task_type: str,
    input_kind: str,
    runtime_target: str,
) -> tuple[CandidateChoice, ...]:
    return tuple(
        CandidateChoice(
            label=label_for_candidate(candidate),
            candidate_id=candidate.candidate_id,
        )
        for candidate in candidates_for(
            task_type=task_type,
            input_kind=input_kind,
            runtime_target=runtime_target,
        )
        if candidate.is_runnable
    )


def _augmentation_policy_choices() -> tuple[CandidateChoice, ...]:
    return tuple(
        CandidateChoice(
            label=label_for_augmentation_policy(policy),
            candidate_id=policy.policy_id,
        )
        for policy in all_augmentation_policies()
    )


def _first_variant(*, config: EngineExperimentConfig, kind: str) -> EngineDataVariantConfig:
    for variant in config.data_variants:
        if variant.enabled and variant.kind == kind:
            return variant

    raise ValueError(f'base config must include an enabled {kind} data variant')


def _first_task(*, config: EngineExperimentConfig, task_type: str) -> EngineTaskConfig:
    for task in config.tasks:
        if task.enabled and task.task_type == task_type:
            return task

    raise ValueError(f'base config must include an enabled {task_type} task')
