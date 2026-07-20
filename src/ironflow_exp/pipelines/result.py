from dataclasses import dataclass, field

from ironflow_exp.datasets import DatasetManifest, ObjectManifest
from ironflow_exp.domain import ArtifactRecord


@dataclass(frozen=True, slots=True)
class PipelineStageResult:
    name: str
    status: str
    message: str = ''
    metadata: dict[str, object] = field(default_factory=dict)


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        stages: list[PipelineStageResult],
    ) -> None:
        super().__init__(message)
        self.stages = stages


@dataclass(frozen=True, slots=True)
class ExperimentPipelineResult:
    dataset_manifest: DatasetManifest
    object_manifest: ObjectManifest | None = None
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    stages: list[PipelineStageResult] = field(default_factory=list)

    @property
    def stage_statuses(self) -> dict[str, str]:
        return {
            stage.name: stage.status
            for stage in self.stages
        }
