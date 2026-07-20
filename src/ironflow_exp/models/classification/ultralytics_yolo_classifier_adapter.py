from dataclasses import dataclass
from typing import Any

from ironflow_exp.configs import ModelConfig
from ironflow_exp.domain import ArtifactRecord, ExperimentContext
from ironflow_exp.models.base import ClassificationModelAdapter, PredictionRecord
from ironflow_exp.models.detection import MissingUltralyticsDependencyError


class UnsupportedUltralyticsYoloClassifierError(ValueError):
    pass


ULTRALYTICS_YOLO_CLASSIFIER_FAMILIES = ('yolov8', 'yolo11', 'yolo12', 'yolo26')
ULTRALYTICS_YOLO_CLASSIFIER_SCALES = ('n', 's', 'm', 'l', 'x')
SUPPORTED_ULTRALYTICS_YOLO_CLASSIFIER_MODEL_IDS = frozenset(
    f'{family}{scale}-cls'
    for family in ULTRALYTICS_YOLO_CLASSIFIER_FAMILIES
    for scale in ULTRALYTICS_YOLO_CLASSIFIER_SCALES
)


@dataclass(frozen=True, slots=True)
class UltralyticsYoloClassifierReference:
    model_id: str
    reference: str
    source: str
    pretrained: bool
    checkpoint: str | None


class UltralyticsYoloClassificationAdapter(ClassificationModelAdapter):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config=model_config)
        self.model: Any | None = None

    def load(self, context: ExperimentContext) -> None:
        yolo_class = self._load_yolo_class()
        self.model = yolo_class(self._model_reference())

    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        raise NotImplementedError('Ultralytics YOLO classification training is implemented by the engine task adapter')

    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        raise NotImplementedError('Ultralytics YOLO classification prediction is implemented by the engine task adapter')

    def _load_yolo_class(self) -> type[Any]:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise MissingUltralyticsDependencyError(
                'ultralytics is required for UltralyticsYoloClassificationAdapter',
            ) from error

        return YOLO

    def _model_reference(self) -> str:
        return self._model_reference_contract().reference

    def _model_reference_contract(self) -> UltralyticsYoloClassifierReference:
        model_id = self._validated_model_id()
        if self.model_config.checkpoint:
            return UltralyticsYoloClassifierReference(
                model_id=model_id,
                reference=self.model_config.checkpoint,
                source='checkpoint',
                pretrained=False,
                checkpoint=self.model_config.checkpoint,
            )

        suffix = '.pt' if self.model_config.pretrained else '.yaml'

        return UltralyticsYoloClassifierReference(
            model_id=model_id,
            reference=f'{model_id}{suffix}',
            source='pretrained_weights' if self.model_config.pretrained else 'architecture_yaml',
            pretrained=self.model_config.pretrained,
            checkpoint=None,
        )

    def _validated_model_id(self) -> str:
        model_id = self.model_config.model_id
        if not model_id:
            raise ValueError('model_id is required for Ultralytics YOLO classification')
        if model_id not in SUPPORTED_ULTRALYTICS_YOLO_CLASSIFIER_MODEL_IDS:
            supported = ', '.join(sorted(SUPPORTED_ULTRALYTICS_YOLO_CLASSIFIER_MODEL_IDS))
            raise UnsupportedUltralyticsYoloClassifierError(
                f'unsupported Ultralytics YOLO classifier model_id: {model_id}; supported: {supported}',
            )

        return model_id
