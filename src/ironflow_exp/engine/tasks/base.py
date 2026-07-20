from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ironflow_exp.engine.domain import TaskExecutionRecord


@dataclass(frozen=True, slots=True)
class TaskDependencyContext:
    task_id: str
    task_type: str
    status: str
    result_dir: Path
    artifacts: list[dict[str, object]] = field(default_factory=list)

    def artifact_path(self, name: str) -> Path:
        for artifact in self.artifacts:
            if artifact.get('name') != name:
                continue
            raw_path = artifact.get('path')
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f'dependency artifact {name!r} has no path')
            path = Path(raw_path)
            root = self.result_dir.resolve()
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f'dependency artifact path escapes task result directory: task_id={self.task_id}, name={name}',
                ) from exc

            return resolved

        raise KeyError(f'dependency artifact not found: task_id={self.task_id}, name={name}')


@dataclass(frozen=True, slots=True)
class TaskExecutionContext:
    record: TaskExecutionRecord
    result_dir: Path
    dependency_results: dict[str, TaskDependencyContext] = field(default_factory=dict)

    def dependency(self, task_id: str) -> TaskDependencyContext:
        try:
            return self.dependency_results[task_id]
        except KeyError as exc:
            raise KeyError(f'dependency result not found: {task_id}') from exc

    def dependency_artifact_path(self, task_id: str, artifact_name: str) -> Path:
        return self.dependency(task_id).artifact_path(artifact_name)

    def first_dependency_artifact_path(self, artifact_name: str) -> Path:
        for dependency in self.dependency_results.values():
            try:
                return dependency.artifact_path(artifact_name)
            except KeyError:
                continue

        raise KeyError(f'dependency artifact not found: name={artifact_name}')

@dataclass(frozen=True, slots=True)
class TaskAdapterResult:
    success: bool
    status: str
    message: str
    metrics: list[dict[str, object]] = field(default_factory=list)
    predictions: list[dict[str, object]] = field(default_factory=list)
    artifacts: list[dict[str, object]] = field(default_factory=list)
    timings: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskExecutionResult:
    record: TaskExecutionRecord
    success: bool
    status: str
    message: str
    metrics: list[dict[str, object]] = field(default_factory=list)
    artifacts: list[dict[str, object]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timings: list[dict[str, object]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data['record'] = self.record.to_dict()

        return data


@dataclass(frozen=True, slots=True)
class TaskExecutionPlanResult:
    experiment_id: str
    success: bool
    task_results: list[TaskExecutionResult] = field(default_factory=list)
    message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'experiment_id': self.experiment_id,
            'success': self.success,
            'message': self.message,
            'task_results': [
                result.to_dict()
                for result in self.task_results
            ],
        }


class BaseTaskAdapter(ABC):
    @abstractmethod
    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        """Run one task and return task-level metrics, predictions, and artifacts."""
