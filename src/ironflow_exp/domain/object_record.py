from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    object_id: str
    sample_id: str
    image_id: str
    split: str
    class_id: int | None
    class_name: str | None
    bbox_xyxy: list[float] | None
    bbox_yolo: list[float] | None
    crop_path: str | None
    mask_path: str | None
    source: str
    parent_object_id: str | None = None
    source_prediction_id: str | None = None
    annotation_id: str | None = None
    is_gt: bool = True
    crop_params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
