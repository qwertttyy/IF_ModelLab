from ironflow_exp.configs import ModelConfig
from ironflow_exp.domain import ArtifactRecord, ExperimentContext
from ironflow_exp.models.base import PredictionRecord, TrackingModelAdapter


class UnsupportedByteTrackModelError(ValueError):
    pass


SUPPORTED_BYTETRACK_MODEL_IDS = frozenset({'bytetrack'})


class ByteTrackModelAdapter(TrackingModelAdapter):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config=model_config)
        self._validated_model_id()

    def load(self, context: ExperimentContext) -> None:
        return None

    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        raise NotImplementedError(
            'ByteTrackModelAdapter.train is not applicable; ByteTrack is a tracking-by-detection adapter',
        )

    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        raise NotImplementedError(
            'ByteTrackModelAdapter.predict is executed by the engine task adapter against detection_predictions.json',
        )

    def _validated_model_id(self) -> str:
        model_id = self.model_config.model_id
        if model_id not in SUPPORTED_BYTETRACK_MODEL_IDS:
            supported = ', '.join(sorted(SUPPORTED_BYTETRACK_MODEL_IDS))
            raise UnsupportedByteTrackModelError(
                f'unsupported ByteTrack model_id: {model_id}; supported: {supported}',
            )

        return str(model_id)
