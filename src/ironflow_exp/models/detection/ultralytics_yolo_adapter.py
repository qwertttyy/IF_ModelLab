import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ironflow_exp.configs import ModelConfig
from ironflow_exp.core.enums import ArtifactRole, TaskType
from ironflow_exp.domain import ArtifactRecord, DetectionPredictionRecord, ExperimentContext
from ironflow_exp.models.base import DetectionModelAdapter, PredictionRecord


class MissingUltralyticsDependencyError(RuntimeError):
    pass


class UnsupportedUltralyticsYoloModelError(ValueError):
    pass


ULTRALYTICS_YOLO_DETECTION_FAMILIES = ('yolov8', 'yolo11', 'yolo12', 'yolo26')
ULTRALYTICS_YOLO_DETECTION_SCALES = ('n', 's', 'm', 'l', 'x')
SUPPORTED_ULTRALYTICS_YOLO_DETECTION_MODEL_IDS = frozenset(
    f'{family}{scale}'
    for family in ULTRALYTICS_YOLO_DETECTION_FAMILIES
    for scale in ULTRALYTICS_YOLO_DETECTION_SCALES
)


@dataclass(frozen=True, slots=True)
class UltralyticsYoloModelReference:
    model_id: str
    reference: str
    source: str
    pretrained: bool
    checkpoint: str | None


class UltralyticsYoloDetectionAdapter(DetectionModelAdapter):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config=model_config)
        self.model: Any | None = None

    def load(self, context: ExperimentContext) -> None:
        yolo_class = self._load_yolo_class(context=context)
        self.model = yolo_class(self._model_reference())

    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        model = self._loaded_model(context=context)
        data_yaml = self._data_yaml_path(context=context)
        train_result = model.train(**self._train_kwargs(context=context, data_yaml=data_yaml))
        checkpoint_path = self._best_checkpoint_path(train_result=train_result)

        if checkpoint_path is None:
            return []

        return [
            ArtifactRecord(
                artifact_id='artifact_detection_checkpoint_00000001',
                task=TaskType.DETECTION.value,
                role=ArtifactRole.CHECKPOINT.value,
                path=self._relative_to_output(path=checkpoint_path, context=context),
                format='pt',
                model_id=self.model_config.model_id,
                sample_id=None,
                object_id=None,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                metadata={
                    'adapter': 'ultralytics_yolo_detection',
                    'source_path': str(checkpoint_path),
                    'model_reference': self._model_reference(),
                },
            ),
        ]

    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        model = self._loaded_model(context=context)
        samples = self._load_manifest_samples(context=context)
        source_root = Path(context.config.dataset.source_root)
        predictions: list[PredictionRecord] = []

        for sample in samples:
            image_path = source_root / str(sample['image_path'])
            started_at = perf_counter()
            results = model.predict(**self._predict_kwargs(context=context, image_path=image_path))
            latency_ms = (perf_counter() - started_at) * 1000
            predictions.extend(
                self._prediction_records(
                    results=results,
                    sample=sample,
                    start_index=len(predictions),
                    latency_ms=latency_ms,
                ),
            )

        return predictions

    def _loaded_model(self, context: ExperimentContext) -> Any:
        if self.model is None:
            self.load(context=context)

        return self.model

    def _load_yolo_class(self, context: ExperimentContext) -> type[Any]:
        self._ensure_yolo_config_dir(context=context)

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise MissingUltralyticsDependencyError(
                'ultralytics is required for UltralyticsYoloDetectionAdapter',
            ) from error

        return YOLO

    def _ensure_yolo_config_dir(self, context: ExperimentContext) -> Path:
        config_dir = Path(context.output_dir) / 'runtime' / 'ultralytics'
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(config_dir)

        return config_dir

    def _model_reference(self) -> str:
        return self._model_reference_contract().reference

    def _model_reference_contract(self) -> UltralyticsYoloModelReference:
        model_id = self._validated_model_id()
        if self.model_config.checkpoint:
            return UltralyticsYoloModelReference(
                model_id=model_id,
                reference=self.model_config.checkpoint,
                source='checkpoint',
                pretrained=False,
                checkpoint=self.model_config.checkpoint,
            )

        suffix = '.pt' if self.model_config.pretrained else '.yaml'

        return UltralyticsYoloModelReference(
            model_id=model_id,
            reference=f'{model_id}{suffix}',
            source='pretrained_weights' if self.model_config.pretrained else 'architecture_yaml',
            pretrained=self.model_config.pretrained,
            checkpoint=None,
        )

    def _validated_model_id(self) -> str:
        model_id = self.model_config.model_id
        if not model_id:
            raise ValueError('model_id is required for Ultralytics YOLO detection')
        if model_id not in SUPPORTED_ULTRALYTICS_YOLO_DETECTION_MODEL_IDS:
            supported = ', '.join(sorted(SUPPORTED_ULTRALYTICS_YOLO_DETECTION_MODEL_IDS))
            raise UnsupportedUltralyticsYoloModelError(
                f'unsupported Ultralytics YOLO detection model_id: {model_id}; supported: {supported}',
            )

        return model_id

    def _data_yaml_path(self, context: ExperimentContext) -> Path:
        data_yaml = context.metadata.get('detection_data_yaml') or context.metadata.get('yolo_data_yaml')

        if data_yaml is None:
            raise ValueError('context.metadata must include detection_data_yaml for YOLO training')

        data_yaml_path = Path(str(data_yaml))
        if not data_yaml_path.exists():
            raise FileNotFoundError(f'YOLO data yaml not found: {data_yaml_path}')

        return data_yaml_path

    def _train_kwargs(self, context: ExperimentContext, data_yaml: Path) -> dict[str, object]:
        train_config = self.model_config.train
        kwargs: dict[str, object] = {
            'data': str(data_yaml),
            'project': str(Path(context.output_dir) / 'artifacts' / 'checkpoints'),
            'name': self._validated_model_id(),
            'exist_ok': True,
            'device': context.config.runtime.device,
            'verbose': False,
        }

        if train_config.epochs is not None:
            kwargs['epochs'] = train_config.epochs
        if train_config.image_size is not None:
            kwargs['imgsz'] = train_config.image_size
        if train_config.batch_size is not None:
            kwargs['batch'] = train_config.batch_size
        if train_config.learning_rate is not None:
            kwargs['lr0'] = train_config.learning_rate

        return kwargs

    def _predict_kwargs(self, context: ExperimentContext, image_path: Path) -> dict[str, object]:
        predict_config = self.model_config.predict
        kwargs: dict[str, object] = {
            'source': str(image_path),
            'device': context.config.runtime.device,
            'verbose': False,
        }

        if predict_config.confidence_threshold is not None:
            kwargs['conf'] = predict_config.confidence_threshold
        if predict_config.iou_threshold is not None:
            kwargs['iou'] = predict_config.iou_threshold
        if self.model_config.train.image_size is not None:
            kwargs['imgsz'] = self.model_config.train.image_size

        return kwargs

    def _best_checkpoint_path(self, train_result: Any) -> Path | None:
        save_dir = getattr(train_result, 'save_dir', None)
        if save_dir is None:
            return None

        checkpoint_path = Path(save_dir) / 'weights' / 'best.pt'
        if not checkpoint_path.exists():
            return None

        return checkpoint_path

    def _load_manifest_samples(self, context: ExperimentContext) -> list[dict[str, object]]:
        if context.dataset_manifest_path is None:
            raise ValueError('context.dataset_manifest_path is required for YOLO prediction')

        manifest_path = Path(context.dataset_manifest_path)
        with manifest_path.open(mode='r', encoding='utf-8') as file:
            manifest = json.load(file)

        samples = manifest.get('samples')
        if not isinstance(samples, list):
            raise ValueError('dataset manifest must contain a samples list')

        rows: list[dict[str, object]] = []
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError('dataset manifest sample must be a mapping object')
            rows.append(sample)

        return rows

    def _prediction_records(
        self,
        results: Any,
        sample: dict[str, object],
        start_index: int,
        latency_ms: float,
    ) -> list[DetectionPredictionRecord]:
        predictions: list[DetectionPredictionRecord] = []

        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue

            names = getattr(result, 'names', {})
            class_ids = self._sequence(getattr(boxes, 'cls', []))
            confidences = self._sequence(getattr(boxes, 'conf', []))
            xyxy_rows = self._rows(getattr(boxes, 'xyxy', []))
            yolo_rows = self._rows(getattr(boxes, 'xywhn', []))

            for box_index, class_id_value in enumerate(class_ids):
                class_id = int(class_id_value)
                predictions.append(
                    DetectionPredictionRecord(
                        prediction_id=f'pred_det_{start_index + len(predictions) + 1:08d}',
                        sample_id=str(sample['sample_id']),
                        image_id=str(sample['image_id']),
                        object_id=f'pred_obj_{start_index + len(predictions) + 1:08d}',
                        split=str(sample['split']),
                        class_id=class_id,
                        class_name=self._class_name(names=names, class_id=class_id),
                        confidence=self._optional_float_at(values=confidences, index=box_index),
                        bbox_xyxy=self._optional_row_at(rows=xyxy_rows, index=box_index),
                        bbox_yolo=self._optional_row_at(rows=yolo_rows, index=box_index),
                        latency_ms=latency_ms if box_index == 0 else None,
                        metadata={
                            'adapter': 'ultralytics_yolo_detection',
                            'source_model_id': self.model_config.model_id,
                        },
                    ),
                )

        return predictions

    def _sequence(self, value: Any) -> list[float]:
        converted = self._to_python(value=value)

        if converted is None:
            return []
        if isinstance(converted, list):
            return [float(item) for item in converted]

        return [float(converted)]

    def _rows(self, value: Any) -> list[list[float]]:
        converted = self._to_python(value=value)

        if converted is None:
            return []
        if not isinstance(converted, list):
            return []
        if not converted:
            return []
        if all(isinstance(item, int | float) for item in converted):
            return [[float(item) for item in converted]]

        return [
            [float(item) for item in row]
            for row in converted
            if isinstance(row, list)
        ]

    def _to_python(self, value: Any) -> Any:
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'tolist'):
            return value.tolist()

        return value

    def _optional_float_at(self, values: list[float], index: int) -> float | None:
        if index >= len(values):
            return None

        return float(values[index])

    def _optional_row_at(self, rows: list[list[float]], index: int) -> list[float] | None:
        if index >= len(rows):
            return None

        return rows[index]

    def _class_name(self, names: object, class_id: int) -> str | None:
        if isinstance(names, dict):
            value = names.get(class_id) or names.get(str(class_id))
            return str(value) if value is not None else None

        if isinstance(names, list) and class_id < len(names):
            return str(names[class_id])

        return None

    def _relative_to_output(self, path: Path, context: ExperimentContext) -> str:
        output_dir = Path(context.output_dir)

        try:
            return path.relative_to(output_dir).as_posix()
        except ValueError:
            return path.as_posix()
