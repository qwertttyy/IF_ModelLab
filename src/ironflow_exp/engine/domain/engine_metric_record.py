from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineMetricRecord:
    experiment_id: str
    epoch: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    class_recall: float | None = None
    class_ap50: float | None = None
    object_accuracy: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    num_predictions: float | None = None
    num_gt: float | None = None
    mask_count: float | None = None
    mask_coverage: float | None = None
    embedding_count: float | None = None
    embedding_dim: float | None = None
    retrieval_map: float | None = None
    neighbor_purity: float | None = None
    review_hit_rate: float | None = None
    label_error_rate: float | None = None
    latency_ms_per_image: float | None = None
    p95_latency_ms: float | None = None
    gpu_memory_mb: float | None = None
    lr: float | None = None
    created_at: str = ''
