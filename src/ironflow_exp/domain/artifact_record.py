from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    task: str
    role: str
    path: str
    format: str
    model_id: str | None
    sample_id: str | None
    object_id: str | None
    created_at: str
    checksum_sha256: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
