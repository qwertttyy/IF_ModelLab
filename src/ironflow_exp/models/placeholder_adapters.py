from ironflow_exp.configs import ModelConfig
from ironflow_exp.domain import ArtifactRecord, ExperimentContext
from ironflow_exp.models.base import (
    ClassificationModelAdapter,
    DetectionModelAdapter,
    EmbeddingModelAdapter,
    PredictionRecord,
    SegmentationModelAdapter,
    TrackingModelAdapter,
)


class PlannedAdapterMixin:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def load(self, context: ExperimentContext) -> None:
        raise NotImplementedError('model adapter is registered but not implemented yet')

    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        raise NotImplementedError('model adapter is registered but not implemented yet')

    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        raise NotImplementedError('model adapter is registered but not implemented yet')


class PlannedDetectionAdapter(PlannedAdapterMixin, DetectionModelAdapter):
    pass


class PlannedTrackingAdapter(PlannedAdapterMixin, TrackingModelAdapter):
    pass


class PlannedClassificationAdapter(PlannedAdapterMixin, ClassificationModelAdapter):
    pass


class PlannedSegmentationAdapter(PlannedAdapterMixin, SegmentationModelAdapter):
    pass


class PlannedEmbeddingAdapter(PlannedAdapterMixin, EmbeddingModelAdapter):
    pass
