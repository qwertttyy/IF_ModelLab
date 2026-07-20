from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TrainConfig:
    enabled: bool = False
    epochs: int | None = None
    image_size: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None


@dataclass(frozen=True, slots=True)
class PredictConfig:
    enabled: bool = True
    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    enabled: bool = False
    model_id: str | None = None
    adapter: str | None = None
    checkpoint: str | None = None
    pretrained: bool = True
    output_dim: int | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)


@dataclass(frozen=True, slots=True)
class ModelGroupConfig:
    detection: ModelConfig = field(default_factory=ModelConfig)
    classification: ModelConfig = field(default_factory=ModelConfig)
    segmentation: ModelConfig = field(default_factory=ModelConfig)
    embedding: ModelConfig = field(default_factory=ModelConfig)
