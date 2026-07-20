from pathlib import Path
from typing import Any

# *************************
# External Library
#   - PyYAML : ver 6.0.3
# *************************
import yaml

from ironflow_exp.core.enums import TaskType
from ironflow_exp.models.base import BaseModelAdapter
from ironflow_exp.models.classification import TorchvisionClassificationAdapter
from ironflow_exp.models.classification import TimmClassificationAdapter
from ironflow_exp.models.classification import UltralyticsYoloClassificationAdapter
from ironflow_exp.models.tracking import ByteTrackModelAdapter
from ironflow_exp.models.detection import UltralyticsYoloDetectionAdapter
from ironflow_exp.models.placeholder_adapters import (
    PlannedClassificationAdapter,
    PlannedDetectionAdapter,
    PlannedEmbeddingAdapter,
    PlannedSegmentationAdapter,
    PlannedTrackingAdapter,
)
from ironflow_exp.models.registry import ModelRegistry
from ironflow_exp.models.spec import ModelSpec


DEFAULT_MODEL_CATALOG_PATH = Path(__file__).resolve().parents[3] / 'configs' / 'model_catalog' / 'default_model_catalog.yaml'
ADAPTER_CLASSES: dict[str, type[BaseModelAdapter]] = {
    'planned_classification': PlannedClassificationAdapter,
    'planned_detection': PlannedDetectionAdapter,
    'planned_embedding': PlannedEmbeddingAdapter,
    'planned_segmentation': PlannedSegmentationAdapter,
    'planned_tracking': PlannedTrackingAdapter,
    'bytetrack_tracker': ByteTrackModelAdapter,
    'timm_classifier': TimmClassificationAdapter,
    'torchvision_classifier': TorchvisionClassificationAdapter,
    'ultralytics_yolo_classifier': UltralyticsYoloClassificationAdapter,
    'ultralytics_yolo_detection': UltralyticsYoloDetectionAdapter,
}


def create_default_model_registry(catalog_path: str | Path | None = None) -> ModelRegistry:
    registry = ModelRegistry()

    for spec in default_model_specs(catalog_path=catalog_path):
        registry.register(spec=spec)

    return registry


def default_model_specs(catalog_path: str | Path | None = None) -> list[ModelSpec]:
    return [
        _spec_from_row(row=row)
        for row in _load_catalog_rows(catalog_path=catalog_path)
    ]


def default_detection_specs(catalog_path: str | Path | None = None) -> list[ModelSpec]:
    return _filter_specs_by_task(task=TaskType.DETECTION.value, catalog_path=catalog_path)


def default_classification_specs(catalog_path: str | Path | None = None) -> list[ModelSpec]:
    return _filter_specs_by_task(task=TaskType.CLASSIFICATION.value, catalog_path=catalog_path)


def default_segmentation_specs(catalog_path: str | Path | None = None) -> list[ModelSpec]:
    return _filter_specs_by_task(task=TaskType.SEGMENTATION.value, catalog_path=catalog_path)


def default_embedding_specs(catalog_path: str | Path | None = None) -> list[ModelSpec]:
    return _filter_specs_by_task(task=TaskType.EMBEDDING.value, catalog_path=catalog_path)


def _filter_specs_by_task(task: str, catalog_path: str | Path | None) -> list[ModelSpec]:
    return [
        spec
        for spec in default_model_specs(catalog_path=catalog_path)
        if spec.task == task
    ]


def _load_catalog_rows(catalog_path: str | Path | None) -> list[dict[str, Any]]:
    resolved_path = Path(catalog_path) if catalog_path is not None else DEFAULT_MODEL_CATALOG_PATH

    with resolved_path.open(mode='r', encoding='utf-8') as file:
        catalog = yaml.safe_load(file)

    if not isinstance(catalog, dict):
        raise ValueError('model catalog must contain a mapping object')

    specs = catalog.get('specs')
    if not isinstance(specs, list):
        raise ValueError('model catalog must contain a specs list')

    rows: list[dict[str, Any]] = []
    for row in specs:
        if not isinstance(row, dict):
            raise ValueError('each model catalog spec must be a mapping object')
        rows.append(row)

    return rows


def _spec_from_row(row: dict[str, Any]) -> ModelSpec:
    adapter_type = str(row['adapter_type'])
    adapter_class = ADAPTER_CLASSES.get(adapter_type)

    if adapter_class is None:
        raise ValueError(f'unknown model catalog adapter_type: {adapter_type}')

    return ModelSpec(
        task=str(row['task']),
        model_id=str(row['model_id']),
        adapter_key=str(row['adapter_key']),
        adapter_class=adapter_class,
        family=str(row['family']),
        display_name=str(row['display_name']),
        source=str(row['source']),
        supports_train=bool(row.get('supports_train', False)),
        supports_predict=bool(row.get('supports_predict', False)),
        implementation_status=str(row.get('implementation_status', 'planned')),
        readiness_status=str(row.get('readiness_status', 'prepared')),
        runtime_targets=_tuple(row=row, key='runtime_targets'),
        input_formats=_tuple(row=row, key='input_formats'),
        input_artifacts=_tuple(row=row, key='input_artifacts'),
        output_artifacts=_tuple(row=row, key='output_artifacts'),
        required_dependencies=_tuple(row=row, key='required_dependencies'),
        tags=_tuple(row=row, key='tags'),
        notes=str(row.get('notes', '')),
    )


def _tuple(row: dict[str, Any], key: str) -> tuple[str, ...]:
    values = row.get(key, [])

    if not isinstance(values, list):
        raise ValueError(f'model catalog {key} must be a list')

    return tuple(str(value) for value in values)
