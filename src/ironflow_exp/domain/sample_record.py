from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str
    image_id: str
    split: str
    image_path: str
    label_path: str | None
    source_id: str = ''
    source_type: str = ''
    split_group_id: str = ''
    width: int | None = None
    height: int | None = None
    parent_sample_id: str | None = None
    frame_index: int | None = None
    timestamp_sec: float | None = None
    file_hash_sha256: str | None = None
    phash: str | None = None
    label_source: str | None = None
    label_confidence: float | None = None
    annotation_version: str | None = None
    source_class: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
