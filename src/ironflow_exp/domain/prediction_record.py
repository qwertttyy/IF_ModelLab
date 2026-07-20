from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DetectionPredictionRecord:
    prediction_id: str
    sample_id: str
    image_id: str
    object_id: str
    split: str
    class_id: int | None
    class_name: str | None
    confidence: float | None
    bbox_xyxy: list[float] | None
    bbox_yolo: list[float] | None
    matched_object_id: str | None = None
    match_iou: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClassificationTopKRecord:
    class_id: int
    class_name: str
    score: float


@dataclass(frozen=True, slots=True)
class ClassificationPredictionRecord:
    prediction_id: str
    sample_id: str
    image_id: str
    object_id: str | None
    input_source: str
    split: str
    true_class_id: int | None
    true_class_name: str | None
    pred_class_id: int | None
    pred_class_name: str | None
    confidence: float | None
    top_k: list[ClassificationTopKRecord] = field(default_factory=list)
    source_prediction_id: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SegmentationPredictionRecord:
    prediction_id: str
    sample_id: str
    image_id: str
    object_id: str | None
    split: str
    prompt_type: str | None
    prompt_source: str | None
    mask_path: str
    bbox_xyxy: list[float] | None
    gt_mask_path: str | None = None
    matched_object_id: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingPredictionRecord:
    prediction_id: str
    sample_id: str
    image_id: str
    object_id: str | None
    embedding_id: str
    embedding_index: int
    embedding_dim: int
    source_path: str
    latency_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
