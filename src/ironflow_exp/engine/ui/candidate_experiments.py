from __future__ import annotations

from dataclasses import replace

from ironflow_exp.engine.configs import (
    EngineAnalysisConfig,
    EngineCodeConfig,
    EngineDataVariantConfig,
    EngineExperimentConfig,
    EngineExperimentMetaConfig,
    EngineRepositoryConfig,
    EngineTaskConfig,
)
from ironflow_exp.engine.ui.model_adapter_catalog import (
    AVAILABILITY_PLANNED_REAL,
    READINESS_ADAPTER_READY,
    READINESS_PREPARED,
    READINESS_VALIDATED,
    RUNTIME_GPU,
    RUNTIME_WSL,
    ModelAdapterCandidate,
    all_candidates,
    candidate_by_id,
)
from ironflow_exp.engine.ui.native_params import NativeParamOverrides, merge_native_param_overrides
from ironflow_exp.engine.ui.recommended_experiments import (
    CLASSIFICATION_BATCH_SIZE,
    COCO_DETECTION_BATCH_SIZE,
    DEFAULT_MODEL_NAME_CLASSIFICATION_CROPS,
    DEFAULT_MODEL_NAME_DETECTION_DATASET,
    EXTERNAL_DETECTION_WRAPPER_COMMANDS,
    EXTERNAL_EMBEDDING_WRAPPER_COMMANDS,
    EXTERNAL_SEGMENTATION_WRAPPER_COMMANDS,
    YOLO_DETECTION_BATCH_SIZE,
)


EXPERIMENT_CANDIDATE_READINESS = frozenset({
    READINESS_VALIDATED,
    READINESS_ADAPTER_READY,
    READINESS_PREPARED,
})
CLASSIFICATION_TRAIN_PARAMS = {
    'epochs': 30,
    'image_size': 224,
    'batch_size': CLASSIFICATION_BATCH_SIZE,
    'top_k': 3,
    'patience': 5,
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.0,
}


def gpu_wsl_experiment_candidates() -> tuple[ModelAdapterCandidate, ...]:
    candidates = [
        candidate
        for candidate in all_candidates()
        if candidate.availability == AVAILABILITY_PLANNED_REAL
        and candidate.readiness_status in EXPERIMENT_CANDIDATE_READINESS
        and (RUNTIME_GPU in candidate.runtime_targets or RUNTIME_WSL in candidate.runtime_targets)
    ]

    return tuple(sorted(candidates, key=lambda candidate: (candidate.task_type, candidate.label, candidate.model_id or '')))


def candidate_option_label(candidate: ModelAdapterCandidate) -> str:
    model = candidate.model_id or candidate.adapter
    targets = '/'.join(candidate.runtime_targets)

    return f'{candidate.task_type} | {model} | {candidate.readiness_label} | {targets}'


def candidate_option_map() -> dict[str, str]:
    return {
        candidate_option_label(candidate): candidate.candidate_id
        for candidate in gpu_wsl_experiment_candidates()
    }


def candidate_compact_flow(*, candidate: ModelAdapterCandidate) -> str:
    mode = _execution_mode_for(candidate=candidate)

    return f'{candidate.task_type.title()} candidate: {candidate.model_id or candidate.adapter} ({candidate.readiness_label}, {mode})'


def candidate_detail_rows(*, candidate: ModelAdapterCandidate) -> tuple[tuple[str, str, str, str], ...]:
    if candidate.task_type == 'tracking':
        return (
            ('Detection', 'yolo11n', 'original', 'ultralytics_yolo'),
            ('Tracking', candidate.model_id or candidate.adapter, 'detection_sequence', candidate.adapter),
        )

    return (
        (
            candidate.task_type.replace('_', ' ').title(),
            candidate.model_id or candidate.adapter,
            _input_variant_kind_for(candidate=candidate),
            candidate.adapter,
        ),
    )


def build_candidate_experiment_config(
    *,
    base_config: EngineExperimentConfig,
    candidate: ModelAdapterCandidate,
    detection_dataset_path: str = DEFAULT_MODEL_NAME_DETECTION_DATASET,
    classification_crop_dataset_path: str = DEFAULT_MODEL_NAME_CLASSIFICATION_CROPS,
    external_command_mode: str = 'dry_run_contract',
    native_param_overrides: NativeParamOverrides = None,
) -> EngineExperimentConfig:
    data_variants = _candidate_data_variants(
        candidate=candidate,
        detection_dataset_path=detection_dataset_path,
        classification_crop_dataset_path=classification_crop_dataset_path,
    )
    tasks = [
        replace(
            task,
            params=merge_native_param_overrides(
                params=task.params,
                task_id=task.id,
                task_type=task.task_type,
                adapter=task.adapter,
                model_id=task.model_id,
                overrides=native_param_overrides,
            ),
        )
        for task in _candidate_tasks(candidate=candidate, external_command_mode=external_command_mode)
    ]
    experiment_name = f'candidate_{_safe_id(candidate.task_type)}_{_safe_id(candidate.model_id or candidate.adapter)}'

    return replace(
        base_config,
        experiment=EngineExperimentMetaConfig(
            name=experiment_name,
            description=f'{candidate.label} experiment candidate from model_md/catalog parity.',
            tags=sorted({
                *base_config.experiment.tags,
                'model_candidate',
                candidate.task_type,
                candidate.readiness_status,
            }),
        ),
        code=_with_candidate_package_includes(
            code=base_config.code,
            detection_dataset_path=detection_dataset_path,
            classification_crop_dataset_path=classification_crop_dataset_path,
        ),
        data_variants=data_variants,
        tasks=tasks,
        analysis=_analysis_for(candidate=candidate),
        repository=EngineRepositoryConfig(
            sqlite_path=f'../../runs/{_safe_id(experiment_name)}_experiments.sqlite3',
        ),
    )


def build_candidate_experiment_config_by_id(
    *,
    base_config: EngineExperimentConfig,
    candidate_id: str,
) -> EngineExperimentConfig:
    return build_candidate_experiment_config(
        base_config=base_config,
        candidate=candidate_by_id(candidate_id),
    )


def _candidate_data_variants(
    *,
    candidate: ModelAdapterCandidate,
    detection_dataset_path: str,
    classification_crop_dataset_path: str,
) -> list[EngineDataVariantConfig]:
    original = EngineDataVariantConfig(
        id='model_name_detection',
        kind='original',
        path=detection_dataset_path,
        enabled=True,
    )
    classification = EngineDataVariantConfig(
        id='model_name_classification_crops',
        kind='original',
        path=classification_crop_dataset_path,
        enabled=True,
    )
    detection_sequence = EngineDataVariantConfig(
        id='model_name_detection_sequence',
        kind='detection_sequence',
        source=original.id,
        enabled=True,
    )

    if candidate.task_type == 'classification':
        return [classification]
    if candidate.task_type == 'tracking':
        return [original, detection_sequence]

    return [original]


def _candidate_tasks(*, candidate: ModelAdapterCandidate, external_command_mode: str) -> list[EngineTaskConfig]:
    if candidate.task_type == 'tracking':
        detector_task_id = 'detect_yolo11n_for_tracking'
        return [
            EngineTaskConfig(
                id=detector_task_id,
                task_type='detection',
                input_variant='model_name_detection',
                model_id='yolo11n',
                adapter='ultralytics_yolo',
                params={
                    'execution_mode': 'inference_smoke',
                    'pretrained': True,
                    'allow_pretrained_download': True,
                    'image_size': 640,
                    'batch_size': YOLO_DETECTION_BATCH_SIZE,
                    'prediction_split': 'test',
                },
            ),
            EngineTaskConfig(
                id=_task_id(prefix='track', candidate=candidate),
                task_type='tracking',
                input_variant='model_name_detection_sequence',
                model_id=candidate.model_id,
                adapter=candidate.adapter,
                depends_on=[detector_task_id],
                params={
                    **candidate.task_params,
                    'source_task_id': detector_task_id,
                },
            ),
        ]

    return [
        EngineTaskConfig(
            id=_task_id(prefix=candidate.task_type, candidate=candidate),
            task_type=candidate.task_type,
            input_variant=_input_variant_id_for(candidate=candidate),
            model_id=candidate.model_id,
            adapter=candidate.adapter,
            params=_task_params_for(candidate=candidate, external_command_mode=external_command_mode),
        ),
    ]


def _task_params_for(*, candidate: ModelAdapterCandidate, external_command_mode: str) -> dict[str, object]:
    params = {
        **candidate.task_params,
        'execution_mode': _execution_mode_for(candidate=candidate),
        'pretrained': True,
        'allow_pretrained_download': True,
    }
    if params.get('execution_mode') == 'train' and candidate.task_type in {'detection', 'classification'}:
        params.setdefault('epochs', 30)
    if candidate.task_type in {'detection', 'segmentation'}:
        params.setdefault('image_size', 640)
        if candidate.task_type == 'detection':
            params.setdefault(
                'batch_size',
                YOLO_DETECTION_BATCH_SIZE
                if candidate.adapter == 'ultralytics_yolo'
                else COCO_DETECTION_BATCH_SIZE,
            )
        params.setdefault('patience', 5)
        params.setdefault('early_stopping_patience', 5)
        params.setdefault('early_stopping_min_delta', 0.0)
    if candidate.task_type == 'classification':
        params.update(CLASSIFICATION_TRAIN_PARAMS)
    if params.get('execution_mode') == 'inference_smoke' and candidate.task_type in {'detection', 'classification'}:
        params.setdefault('prediction_split', 'test')
    if candidate.task_type == 'embedding':
        params.setdefault('image_size', 224)

    wrapper_command = _wrapper_command_for(candidate=candidate)
    if wrapper_command is not None:
        command = _wrapper_command(template=wrapper_command, external_command_mode=external_command_mode)
        params.setdefault('external_train_command', command)
        params.setdefault('external_inference_command', command)
        params.setdefault('external_command_contract_mode', external_command_mode)

    return params


def _wrapper_command_for(*, candidate: ModelAdapterCandidate) -> str | None:
    if candidate.task_type == 'detection':
        return EXTERNAL_DETECTION_WRAPPER_COMMANDS.get(candidate.adapter)
    if candidate.task_type == 'segmentation':
        return EXTERNAL_SEGMENTATION_WRAPPER_COMMANDS.get(candidate.adapter)
    if candidate.task_type == 'embedding':
        return EXTERNAL_EMBEDDING_WRAPPER_COMMANDS.get(candidate.adapter)

    return None


def _wrapper_command(*, template: str, external_command_mode: str) -> str:
    if external_command_mode == 'dry_run_contract':
        return f'{template} --request "{{request_json}}" --dry-run-contract'
    if external_command_mode == 'native':
        return f'{template} --request "{{request_json}}"'

    raise ValueError(f'unsupported external_command_mode: {external_command_mode}')


def _execution_mode_for(*, candidate: ModelAdapterCandidate) -> str:
    if candidate.readiness_status in {READINESS_VALIDATED, READINESS_ADAPTER_READY} and candidate.task_type in {
        'classification',
        'detection',
    }:
        return 'train'

    return 'inference_smoke'


def _input_variant_id_for(*, candidate: ModelAdapterCandidate) -> str:
    if candidate.task_type == 'classification':
        return 'model_name_classification_crops'

    return 'model_name_detection'


def _input_variant_kind_for(*, candidate: ModelAdapterCandidate) -> str:
    if candidate.task_type == 'classification':
        return 'classification_crops'
    if candidate.task_type == 'tracking':
        return 'detection_sequence'

    return 'original'


def _analysis_for(*, candidate: ModelAdapterCandidate) -> EngineAnalysisConfig:
    if candidate.task_type == 'classification':
        return EngineAnalysisConfig(primary_metric='accuracy', higher_is_better=True)
    if candidate.task_type in {'detection', 'segmentation'}:
        return EngineAnalysisConfig(primary_metric='map50', higher_is_better=True)
    if candidate.task_type == 'tracking':
        return EngineAnalysisConfig(primary_metric='idf1', higher_is_better=True)
    if candidate.task_type == 'embedding':
        return EngineAnalysisConfig(primary_metric='retrieval_map', higher_is_better=True)

    return EngineAnalysisConfig()


def _with_candidate_package_includes(
    *,
    code: EngineCodeConfig,
    detection_dataset_path: str,
    classification_crop_dataset_path: str,
) -> EngineCodeConfig:
    includes = [
        'pyproject.toml',
        'configs/model_catalog/**',
        'scripts/external_wrapper_common.py',
        'scripts/external_detection_wrappers/**',
        'scripts/external_segmentation_wrappers/**',
        'scripts/external_embedding_wrappers/**',
        'src/**',
        _package_pattern(detection_dataset_path),
        _package_pattern(classification_crop_dataset_path),
    ]

    return replace(
        code,
        package_include=list(dict.fromkeys([*code.package_include, *includes])),
    )


def _package_pattern(path: str) -> str:
    normalized = path.replace('\\', '/').lstrip('./')
    while normalized.startswith('../'):
        normalized = normalized[3:]

    return f'{normalized.rstrip("/")}/**'


def _task_id(*, prefix: str, candidate: ModelAdapterCandidate) -> str:
    return _safe_id(f'{prefix}_{candidate.model_id or candidate.adapter}')


def _safe_id(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value.lower()).strip('_')
