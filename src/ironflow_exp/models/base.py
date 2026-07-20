from abc import ABC, abstractmethod
from typing import TypeAlias

from ironflow_exp.configs import ModelConfig
from ironflow_exp.domain import (
    ArtifactRecord,
    ClassificationPredictionRecord,
    DetectionPredictionRecord,
    EmbeddingPredictionRecord,
    ExperimentContext,
    SegmentationPredictionRecord,
)


PredictionRecord: TypeAlias = (
    DetectionPredictionRecord
    | ClassificationPredictionRecord
    | SegmentationPredictionRecord
    | EmbeddingPredictionRecord
)


class BaseModelAdapter(ABC):
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    @abstractmethod
    def load(self, context: ExperimentContext) -> None:
        """
        모델 checkpoint 또는 pretrained weight 적재

        Args:
            context: 실행 run metadata

        Returns:
            None
        """

    @abstractmethod
    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        """
        task별 학습 실행 후 생성 artifact 반환

        Args:
            context: 실행 run metadata

        Returns:
            생성된 checkpoint 또는 학습 artifact 목록
        """

    @abstractmethod
    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        """
        task별 추론 실행 후 표준 prediction record 반환

        Args:
            context: 실행 run metadata

        Returns:
            표준 prediction record 목록
        """


class DetectionModelAdapter(BaseModelAdapter, ABC):
    pass


class TrackingModelAdapter(BaseModelAdapter, ABC):
    pass


class ClassificationModelAdapter(BaseModelAdapter, ABC):
    pass


class SegmentationModelAdapter(BaseModelAdapter, ABC):
    pass


class EmbeddingModelAdapter(BaseModelAdapter, ABC):
    pass
