from dataclasses import dataclass, field

from ironflow_exp.configs import ExperimentConfig


@dataclass(slots=True)
class ExperimentContext:
    run_id: str
    config: ExperimentConfig
    output_dir: str
    dataset_manifest_path: str | None = None
    object_manifest_path: str | None = None
    artifact_manifest_path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
