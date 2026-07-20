from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ironflow_exp.engine.configs import (
    EngineAnalysisConfig,
    EngineCodeConfig,
    EngineDataVariantConfig,
    EngineExperimentConfig,
    EngineExperimentMetaConfig,
    EngineRepositoryConfig,
    EngineTaskConfig,
)
from ironflow_exp.engine.ui.native_params import NativeParamOverrides, merge_native_param_overrides


RECOMMENDED_MATRIX_PATH = Path('configs/experiments/recommended_model_combinations.yaml')
DEFAULT_MODEL_NAME_DETECTION_DATASET = (
    '../../runs/user_datasets/tank_armor_prepared_v20260703_original_classifier_all_v1/detector_tank_av/detection'
)
DEFAULT_MODEL_NAME_CLASSIFICATION_CROPS = (
    '../../runs/user_datasets/tank_armor_prepared_v20260703_original_classifier_all_v1/classifier_all/images'
)
TASK_ADAPTER_ALIASES = {
    'bytetrack_tracker': 'bytetrack',
    'ultralytics_yolo_detection': 'ultralytics_yolo',
}
REAL_MODEL_TASK_TYPES = frozenset({'detection', 'classification', 'segmentation', 'embedding'})
EXTERNAL_DETECTION_WRAPPER_COMMANDS = {
    'd_fine_detection': 'python scripts/external_detection_wrappers/d_fine_detection_wrapper.py',
    'rf_detr_detection': 'python scripts/external_detection_wrappers/rf_detr_detection_wrapper.py',
    'rt_detr_detection': 'python scripts/external_detection_wrappers/rt_detr_detection_wrapper.py',
    'rt_detr_v2_detection': 'python scripts/external_detection_wrappers/rt_detr_v2_detection_wrapper.py',
    'lw_detr_detection': 'python scripts/external_detection_wrappers/lw_detr_detection_wrapper.py',
    'mmdet_detection': 'python scripts/external_detection_wrappers/mmdet_detection_wrapper.py',
    'yolo_world_open_vocab_detection': 'python scripts/external_detection_wrappers/yolo_world_detection_wrapper.py',
    'grounding_dino_open_vocab_detection': 'python scripts/external_detection_wrappers/grounding_dino_detection_wrapper.py',
    'owlv2_open_vocab_detection': 'python scripts/external_detection_wrappers/owlv2_detection_wrapper.py',
    'dino_x_open_world_detection': 'python scripts/external_detection_wrappers/dino_x_detection_wrapper.py',
    'sahi_slicing_detection': 'python scripts/external_detection_wrappers/sahi_slicing_detection_wrapper.py',
    'asahi_slicing_detection': 'python scripts/external_detection_wrappers/asahi_slicing_detection_wrapper.py',
    'yoloe_open_vocab_detection': 'python scripts/external_detection_wrappers/yoloe_detection_wrapper.py',
}
EXTERNAL_SEGMENTATION_WRAPPER_COMMANDS = {
    'sam_promptable_segmentation': 'python scripts/external_segmentation_wrappers/sam_promptable_segmentation_wrapper.py',
}
EXTERNAL_EMBEDDING_WRAPPER_COMMANDS = {
    'dinov2_embedding': 'python scripts/external_embedding_wrappers/dinov2_embedding_wrapper.py',
    'dinov3_embedding': 'python scripts/external_embedding_wrappers/dinov3_embedding_wrapper.py',
    'clip_embedding': 'python scripts/external_embedding_wrappers/clip_embedding_wrapper.py',
    'siglip_embedding': 'python scripts/external_embedding_wrappers/siglip_embedding_wrapper.py',
}
YOLO_DETECTION_BATCH_SIZE = 16
COCO_DETECTION_BATCH_SIZE = 8
CLASSIFICATION_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class RecommendedPipelineStep:
    task_type: str
    adapter_key: str
    model_id: str | None = None
    input_kind: str = 'original'
    readiness: str | None = None

    @property
    def task_adapter(self) -> str:
        return TASK_ADAPTER_ALIASES.get(self.adapter_key, self.adapter_key)


@dataclass(frozen=True, slots=True)
class RecommendedCombination:
    combination_id: str
    priority: int
    stage: str
    experiment_type: str
    objective: str
    tradeoff: str
    runtime_targets: tuple[str, ...]
    metrics_focus: tuple[str, ...]
    pipeline: tuple[RecommendedPipelineStep, ...]


@dataclass(frozen=True, slots=True)
class RecommendedModelMatrix:
    schema_version: str
    combinations: tuple[RecommendedCombination, ...]
    standalone_experiments: tuple[RecommendedCombination, ...] = ()

    def by_id(self, combination_id: str) -> RecommendedCombination:
        for combination in (*self.combinations, *self.standalone_experiments):
            if combination.combination_id == combination_id:
                return combination

        raise KeyError(f'unknown recommended combination: {combination_id}')

    @property
    def all_experiments(self) -> tuple[RecommendedCombination, ...]:
        return (*self.combinations, *self.standalone_experiments)


def load_recommended_model_matrix(project_dir: Path | None = None) -> RecommendedModelMatrix:
    root = project_dir or Path.cwd()
    path = root / RECOMMENDED_MATRIX_PATH
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'recommended model matrix must be a YAML mapping: {path}')

    return RecommendedModelMatrix(
        schema_version=str(data.get('schema_version') or ''),
        combinations=tuple(
            _combination_from_mapping(item)
            for item in data.get('recommended_combinations') or []
            if isinstance(item, dict)
        ),
        standalone_experiments=tuple(
            _combination_from_mapping(item)
            for item in data.get('standalone_experiments') or []
            if isinstance(item, dict)
        ),
    )


def recommended_combination_label(combination: RecommendedCombination) -> str:
    return f'{combination.priority}. {combination.combination_id} [{combination.stage}]'


def recommended_combination_summary(combination: RecommendedCombination) -> str:
    pipeline = ' -> '.join(
        f'{step.task_type}/{step.input_kind}: {step.model_id or step.task_adapter}'
        for step in combination.pipeline
    )
    targets = ', '.join(combination.runtime_targets)
    metrics = ', '.join(combination.metrics_focus)

    return (
        f'{pipeline} | stage={combination.stage}; targets={targets}; '
        f'metrics={metrics}; objective={combination.objective}'
    )


def build_recommended_combination_config(
    *,
    base_config: EngineExperimentConfig,
    combination: RecommendedCombination,
    detection_dataset_path: str = DEFAULT_MODEL_NAME_DETECTION_DATASET,
    classification_crop_dataset_path: str = DEFAULT_MODEL_NAME_CLASSIFICATION_CROPS,
    test_detection_dataset_path: str | None = None,
    test_classification_crop_dataset_path: str | None = None,
    execution_mode: str = 'train',
    external_command_mode: str = 'dry_run_contract',
    native_param_overrides: NativeParamOverrides = None,
) -> EngineExperimentConfig:
    original_variant_id = 'model_name_detection'
    prediction_original_variant_id = 'model_name_detection_test' if test_detection_dataset_path else original_variant_id
    detector_crop_variant_id = 'model_name_detector_crop'
    detection_sequence_variant_id = 'model_name_detection_sequence'
    classification_variant_id = 'model_name_classification_images'
    prediction_classification_variant_id = (
        'model_name_classification_images_test'
        if test_classification_crop_dataset_path
        else classification_variant_id
    )
    tasks: list[EngineTaskConfig] = []
    last_detection_task_id: str | None = None
    last_detection_input_variant_id = original_variant_id
    last_crop_task_id: str | None = None
    needs_detection_dataset = any(
        step.task_type in {'detection', 'segmentation'} or step.input_kind in {'original', 'detection_predictions'}
        for step in combination.pipeline
    )
    needs_classification_dataset = any(
        step.task_type == 'classification' or step.input_kind == 'classification_crop'
        for step in combination.pipeline
    )
    needs_detector_crop_variant = any(
        step.input_kind == 'detector_crop'
        for step in combination.pipeline
    )
    needs_detection_sequence_variant = any(
        step.task_type == 'tracking'
        for step in combination.pipeline
    )

    for step in combination.pipeline:
        if step.task_type == 'detection':
            if execution_mode == 'train' and _should_create_train_predict_pair(step=step):
                train_task_id = _task_id(prefix='train_detect', model_id=step.model_id, adapter=step.task_adapter)
                predict_task_id = _task_id(prefix='predict_detect', model_id=step.model_id, adapter=step.task_adapter)
                tasks.append(
                    EngineTaskConfig(
                        id=train_task_id,
                        task_type='detection',
                        input_variant=original_variant_id,
                        model_id=step.model_id,
                        adapter=step.task_adapter,
                        params=_real_model_params(
                            step=step,
                            execution_mode='train',
                            external_command_mode=external_command_mode,
                        ),
                    ),
                )
                tasks.append(
                    EngineTaskConfig(
                        id=predict_task_id,
                        task_type='detection',
                        input_variant=prediction_original_variant_id,
                        model_id=step.model_id,
                        adapter=step.task_adapter,
                        depends_on=[train_task_id],
                        params={
                            **_real_model_params(
                                step=step,
                                execution_mode='inference_smoke',
                                external_command_mode=external_command_mode,
                            ),
                            'checkpoint_from_task_id': train_task_id,
                        },
                    ),
                )
                last_detection_task_id = predict_task_id
                last_detection_input_variant_id = prediction_original_variant_id
                continue

            task_id = _task_id(prefix='detect', model_id=step.model_id, adapter=step.task_adapter)
            tasks.append(
                EngineTaskConfig(
                    id=task_id,
                    task_type='detection',
                    input_variant=prediction_original_variant_id,
                    model_id=step.model_id,
                    adapter=step.task_adapter,
                    params=_real_model_params(
                        step=step,
                        execution_mode=execution_mode,
                        external_command_mode=external_command_mode,
                    ),
                ),
            )
            last_detection_task_id = task_id
            last_detection_input_variant_id = prediction_original_variant_id
            continue

        if step.task_type == 'preprocessing':
            if last_detection_task_id is None:
                raise ValueError('recommended preprocessing step requires a preceding detection step')
            task_id = 'make_model_name_detector_crops'
            tasks.append(
                EngineTaskConfig(
                    id=task_id,
                    task_type='preprocessing',
                    input_variant=last_detection_input_variant_id,
                    adapter=step.task_adapter,
                    depends_on=[last_detection_task_id],
                    params={
                        'source_task_id': last_detection_task_id,
                        'input_source': 'detector_crop',
                        'image_format': 'jpg',
                        'padding_ratio': 0.08,
                        'min_score': 0.0,
                        'use_source_labels': True,
                        'classification_dataset_id': 'model_name_detector_crops',
                    },
                ),
            )
            last_crop_task_id = task_id
            continue

        if step.task_type == 'classification':
            has_detector_crop_input = step.input_kind == 'detector_crop' and last_crop_task_id is not None
            if execution_mode == 'train':
                train_task_id = _task_id(prefix='train_classify', model_id=step.model_id, adapter=step.task_adapter)
                predict_task_id = _task_id(prefix='predict_classify', model_id=step.model_id, adapter=step.task_adapter)
                tasks.append(
                    EngineTaskConfig(
                        id=train_task_id,
                        task_type='classification',
                        input_variant=classification_variant_id,
                        model_id=step.model_id,
                        adapter=step.task_adapter,
                        params=_real_model_params(
                            step=step,
                            execution_mode='train',
                            external_command_mode=external_command_mode,
                        ),
                    ),
                )
                tasks.append(
                        EngineTaskConfig(
                            id=predict_task_id,
                            task_type='classification',
                            input_variant=detector_crop_variant_id
                            if has_detector_crop_input
                            else prediction_classification_variant_id,
                        model_id=step.model_id,
                        adapter=step.task_adapter,
                        depends_on=[last_crop_task_id, train_task_id] if has_detector_crop_input else [train_task_id],
                        params={
                            **_real_model_params(
                                step=step,
                                execution_mode='inference_smoke',
                                external_command_mode=external_command_mode,
                            ),
                            'checkpoint_from_task_id': train_task_id,
                        },
                    ),
                )
                continue

            task_id = _task_id(prefix='classify', model_id=step.model_id, adapter=step.task_adapter)
            tasks.append(
                EngineTaskConfig(
                    id=task_id,
                    task_type='classification',
                    input_variant=detector_crop_variant_id
                    if has_detector_crop_input
                    else prediction_classification_variant_id,
                    model_id=step.model_id,
                    adapter=step.task_adapter,
                    depends_on=[last_crop_task_id] if has_detector_crop_input else [],
                    params=_real_model_params(
                        step=step,
                        execution_mode=execution_mode,
                        external_command_mode=external_command_mode,
                    ),
                ),
            )
            continue

        if step.task_type == 'tracking':
            if last_detection_task_id is None:
                raise ValueError('recommended tracking step requires a preceding detection step')
            task_id = _task_id(prefix='track', model_id=step.model_id, adapter=step.task_adapter)
            tasks.append(
                EngineTaskConfig(
                    id=task_id,
                    task_type='tracking',
                    input_variant=detection_sequence_variant_id,
                    model_id=step.model_id,
                    adapter=step.task_adapter,
                    depends_on=[last_detection_task_id],
                    params={
                        'source_task_id': last_detection_task_id,
                        'iou_threshold': 0.3,
                    },
                ),
            )
            continue

        depends_on: list[str] = []
        input_variant = original_variant_id
        params = _real_model_params(
            step=step,
            execution_mode='inference_smoke',
            external_command_mode=external_command_mode,
        )
        if step.task_type == 'segmentation' and step.input_kind == 'detection_predictions':
            if last_detection_task_id is None:
                raise ValueError('recommended segmentation step requires a preceding detection step')
            depends_on = [last_detection_task_id]
            input_variant = last_detection_input_variant_id
            params['source_task_id'] = last_detection_task_id
        if step.task_type == 'embedding' and step.input_kind == 'detector_crop':
            if last_crop_task_id is None:
                raise ValueError('recommended embedding step requires a preceding crop step')
            input_variant = detector_crop_variant_id
            depends_on = [last_crop_task_id]

        tasks.append(
            EngineTaskConfig(
                id=_task_id(prefix=step.task_type, model_id=step.model_id, adapter=step.task_adapter),
                task_type=step.task_type,
                input_variant=input_variant,
                model_id=step.model_id,
                adapter=step.task_adapter,
                depends_on=depends_on,
                params=params,
            ),
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
        for task in tasks
    ]

    data_variants: list[EngineDataVariantConfig] = []
    if needs_detection_dataset:
        data_variants.append(
            EngineDataVariantConfig(
                id=original_variant_id,
                kind='original',
                path=detection_dataset_path,
                enabled=True,
            ),
        )
    if needs_detector_crop_variant:
        data_variants.append(
            EngineDataVariantConfig(
                id=detector_crop_variant_id,
                kind='detector_crop',
                source=prediction_original_variant_id,
                enabled=True,
            ),
        )
    if needs_detection_sequence_variant:
        data_variants.append(
            EngineDataVariantConfig(
                id=detection_sequence_variant_id,
                kind='detection_sequence',
                source=original_variant_id,
                enabled=True,
            ),
        )
    if needs_classification_dataset:
        data_variants.append(
            EngineDataVariantConfig(
                id=classification_variant_id,
                kind='original',
                path=classification_crop_dataset_path,
                enabled=True,
            ),
        )
    if test_detection_dataset_path:
        data_variants.append(
            EngineDataVariantConfig(
                id=prediction_original_variant_id,
                kind='original',
                path=test_detection_dataset_path,
                enabled=True,
            ),
        )
    if test_classification_crop_dataset_path:
        data_variants.append(
            EngineDataVariantConfig(
                id=prediction_classification_variant_id,
                kind='original',
                path=test_classification_crop_dataset_path,
                enabled=True,
            ),
        )
    experiment_name = f'recommended_{combination.combination_id}'

    return replace(
        base_config,
        experiment=EngineExperimentMetaConfig(
            name=experiment_name,
            description=combination.objective,
            tags=sorted({
                *base_config.experiment.tags,
                'recommended_matrix',
                combination.stage,
                combination.experiment_type,
            }),
        ),
        code=_with_recommended_package_includes(
            code=base_config.code,
            pipeline=combination.pipeline,
            detection_dataset_path=detection_dataset_path,
            classification_crop_dataset_path=classification_crop_dataset_path,
            test_detection_dataset_path=test_detection_dataset_path,
            test_classification_crop_dataset_path=test_classification_crop_dataset_path,
            include_detection_dataset=needs_detection_dataset,
            include_classification_dataset=needs_classification_dataset,
        ),
        data_variants=data_variants,
        tasks=tasks,
        analysis=EngineAnalysisConfig(
            primary_metric=_primary_metric_for_combination(combination=combination),
            higher_is_better=True,
            failure_rules_enabled=base_config.analysis.failure_rules_enabled,
        ),
        repository=EngineRepositoryConfig(
            sqlite_path=f'../../runs/{_safe_id(experiment_name)}_experiments.sqlite3',
        ),
    )


def _combination_from_mapping(item: dict[str, Any]) -> RecommendedCombination:
    return RecommendedCombination(
        combination_id=str(item['id']),
        priority=int(item['priority']),
        stage=str(item.get('stage') or ''),
        experiment_type=str(item.get('experiment_type') or 'end_to_end'),
        objective=str(item.get('objective') or ''),
        tradeoff=str(item.get('tradeoff') or ''),
        runtime_targets=tuple(str(value) for value in item.get('runtime_targets') or []),
        metrics_focus=tuple(str(value) for value in item.get('metrics_focus') or []),
        pipeline=tuple(
            _step_from_mapping(step)
            for step in item.get('pipeline') or []
            if isinstance(step, dict)
        ),
    )


def _step_from_mapping(item: dict[str, Any]) -> RecommendedPipelineStep:
    return RecommendedPipelineStep(
        task_type=str(item['task_type']),
        adapter_key=str(item['adapter_key']),
        model_id=str(item['model_id']) if item.get('model_id') is not None else None,
        input_kind=str(item.get('input') or 'original'),
        readiness=str(item['readiness']) if item.get('readiness') is not None else None,
    )


def _real_model_params(
    *,
    step: RecommendedPipelineStep,
    execution_mode: str,
    external_command_mode: str = 'dry_run_contract',
) -> dict[str, object]:
    params: dict[str, object] = {}
    if step.task_type in REAL_MODEL_TASK_TYPES:
        params.update({
            'execution_mode': execution_mode,
            'pretrained': True,
            'allow_pretrained_download': True,
        })
        if execution_mode == 'train' and step.task_type in {'detection', 'classification'}:
            params.setdefault('epochs', 30)
        if execution_mode == 'inference_smoke' and step.task_type in {'detection', 'classification'}:
            params.setdefault('prediction_split', 'test')
    if step.task_type == 'detection':
        params.setdefault('image_size', 640)
        params.setdefault(
            'batch_size',
            YOLO_DETECTION_BATCH_SIZE
            if step.task_adapter == 'ultralytics_yolo'
            else COCO_DETECTION_BATCH_SIZE,
        )
        params.setdefault('patience', 5)
        params.setdefault('early_stopping_patience', 5)
        params.setdefault('early_stopping_min_delta', 0.0)
        wrapper_command = EXTERNAL_DETECTION_WRAPPER_COMMANDS.get(step.task_adapter)
        if wrapper_command is not None:
            command = _wrapper_command(template=wrapper_command, external_command_mode=external_command_mode)
            params.setdefault('external_train_command', command)
            params.setdefault('external_inference_command', command)
            params.setdefault('external_command_contract_mode', external_command_mode)
    if step.task_type == 'classification':
        params.setdefault('image_size', 224)
        params.setdefault('batch_size', CLASSIFICATION_BATCH_SIZE)
        params.setdefault('top_k', 3)
        params.setdefault('patience', 5)
        params.setdefault('early_stopping_patience', 5)
        params.setdefault('early_stopping_min_delta', 0.0)
    if step.task_type == 'segmentation':
        params.setdefault('image_size', 1024)
        wrapper_command = EXTERNAL_SEGMENTATION_WRAPPER_COMMANDS.get(step.task_adapter)
        if wrapper_command is not None:
            command = _wrapper_command(template=wrapper_command, external_command_mode=external_command_mode)
            params.setdefault('external_inference_command', command)
            params.setdefault('external_command_contract_mode', external_command_mode)
    if step.task_type == 'embedding':
        params.setdefault('image_size', 224)
        wrapper_command = EXTERNAL_EMBEDDING_WRAPPER_COMMANDS.get(step.task_adapter)
        if wrapper_command is not None:
            command = _wrapper_command(template=wrapper_command, external_command_mode=external_command_mode)
            params.setdefault('external_inference_command', command)
            params.setdefault('external_command_contract_mode', external_command_mode)

    return params


def _wrapper_command(*, template: str, external_command_mode: str) -> str:
    command = f'{template} --request "{{request_json}}"'
    if external_command_mode == 'dry_run_contract':
        return f'{command} --dry-run-contract'
    if external_command_mode == 'native':
        return command

    raise ValueError(f'unsupported external_command_mode: {external_command_mode}')


def _should_create_train_predict_pair(*, step: RecommendedPipelineStep) -> bool:
    if step.readiness == 'prepared':
        return False
    if step.task_adapter in {'grounding_dino_open_vocab_detection', 'yolo_world_open_vocab_detection'}:
        return False

    return True


def _with_recommended_package_includes(
    *,
    code: EngineCodeConfig,
    pipeline: tuple[RecommendedPipelineStep, ...],
    detection_dataset_path: str,
    classification_crop_dataset_path: str,
    test_detection_dataset_path: str | None = None,
    test_classification_crop_dataset_path: str | None = None,
    include_detection_dataset: bool = True,
    include_classification_dataset: bool = True,
) -> EngineCodeConfig:
    includes = [
        'pyproject.toml',
        'configs/model_catalog/**',
        'scripts/external_wrapper_common.py',
        'scripts/external_detection_wrappers/**',
        'scripts/external_segmentation_wrappers/**',
        'scripts/external_embedding_wrappers/**',
        *_pretrained_package_patterns(pipeline=pipeline),
        'src/**',
    ]
    if include_detection_dataset:
        includes.append(_package_pattern(path=detection_dataset_path))
    if include_classification_dataset:
        includes.append(_package_pattern(path=classification_crop_dataset_path))
    if include_detection_dataset and test_detection_dataset_path:
        includes.append(_package_pattern(path=test_detection_dataset_path))
    if include_classification_dataset and test_classification_crop_dataset_path:
        includes.append(_package_pattern(path=test_classification_crop_dataset_path))
    return replace(
        code,
        package_include=list(dict.fromkeys([*code.package_include, *includes])),
    )


def _pretrained_package_patterns(*, pipeline: tuple[RecommendedPipelineStep, ...]) -> list[str]:
    patterns: list[str] = []
    for step in pipeline:
        model_id = step.model_id
        if step.task_adapter == 'ultralytics_yolo' and model_id in {
            'yolo11n',
            'yolo11s',
            'yolo12n',
            'yolo12s',
            'yolo26n',
            'yolo26s',
            'yolov8n',
        }:
            patterns.append(f'models/checkpoints/pretrained/ultralytics/{model_id}.pt')
        elif step.task_adapter == 'ultralytics_yolo_classifier' and model_id in {'yolo26n-cls'}:
            patterns.append(f'models/checkpoints/pretrained/ultralytics/{model_id}.pt')
        elif step.task_adapter == 'rt_detr_detection' and model_id == 'rt_detr':
            patterns.append('models/checkpoints/pretrained/ultralytics/rtdetr-l.pt')
        elif step.task_adapter == 'rf_detr_detection' and model_id == 'rf_detr':
            patterns.append('models/checkpoints/pretrained/rf_detr/rf_detr_base.pth')
        elif step.task_adapter == 'd_fine_detection' and model_id == 'd_fine':
            patterns.append('models/checkpoints/pretrained/d_fine/dfine_hgnetv2_n_coco.pth')
        elif step.task_adapter == 'torchvision_classifier' and model_id in {
            'mobilenet_v3_small',
            'mobilenet_v3_large',
            'efficientnet_b0',
            'efficientnet_b3',
            'efficientnet_v2_s',
            'resnet50',
            'resnext50_32x4d',
        }:
            patterns.append(f'models/checkpoints/pretrained/torchvision/{model_id}_imagenet.pt')
        elif step.task_adapter == 'timm_classifier' and model_id in {
            'convnext_v2_tiny',
            'convnext_small',
            'swin_tiny',
        }:
            patterns.append(f'models/checkpoints/pretrained/timm/{model_id}_pretrained.pt')

    return patterns


def _primary_metric_for_combination(*, combination: RecommendedCombination) -> str:
    if combination.experiment_type == 'detector_only':
        return 'map50'
    if combination.experiment_type == 'classifier_only':
        return 'macro_f1'
    if combination.experiment_type in {'review_path', 'embedding_review'}:
        return 'review_hit_rate'

    return 'accuracy'


def _package_pattern(path: str) -> str:
    normalized = path.replace('\\', '/').lstrip('./')
    while normalized.startswith('../'):
        normalized = normalized[3:]
    return f'{normalized.rstrip("/")}/**'


def _task_id(*, prefix: str, model_id: str | None, adapter: str) -> str:
    return _safe_id(f'{prefix}_{model_id or adapter}')


def _safe_id(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value.lower()).strip('_')
