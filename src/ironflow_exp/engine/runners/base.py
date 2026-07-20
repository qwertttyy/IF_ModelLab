from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ironflow_exp.engine.domain import ExperimentStatus


@dataclass(frozen=True, slots=True)
class ExperimentRunnerResult:
    experiment_id: str
    status: ExperimentStatus
    workspace_dir: Path | None = None
    message: str = ''
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in {
            ExperimentStatus.FINISHED,
            ExperimentStatus.COLLECTED,
        }


class BaseExperimentRunner(ABC):
    """Common runner contract for local, simulated remote, and SSH execution."""

    @abstractmethod
    def prepare(
        self,
        config: object,
        experiment_id: str,
        replace_existing: bool = False,
    ) -> ExperimentRunnerResult:
        """Prepare workspace, inputs, and command metadata for one experiment."""

    @abstractmethod
    def run(self, experiment_id: str) -> ExperimentRunnerResult:
        """Start or execute the prepared experiment."""

    @abstractmethod
    def status(self, experiment_id: str) -> ExperimentRunnerResult:
        """Return the latest known experiment status."""

    @abstractmethod
    def logs(self, experiment_id: str, tail: int | None = None) -> str:
        """Return experiment logs, optionally limited to the last lines."""

    @abstractmethod
    def collect(self, experiment_id: str, output_dir: Path | None = None) -> ExperimentRunnerResult:
        """Collect metrics, logs, checkpoints, and status artifacts."""
