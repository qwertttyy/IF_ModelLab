import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TASK_ARTIFACT_SCHEMA_VERSION = '0.1'
TASK_ARTIFACT_REQUIRED_FILES = (
    'task.json',
    'task.log',
    'metrics.csv',
    'predictions.json',
    'artifacts.json',
    'status.marker',
    'summary.md',
    'timings.csv',
)
TASK_JSON_REQUIRED_FIELDS = frozenset({'schema_version', 'message', 'task'})
TASK_RECORD_REQUIRED_FIELDS = frozenset({
    'experiment_id',
    'task_id',
    'task_type',
    'order_index',
    'input_variant_id',
    'input_variant_kind',
    'status',
})
TASK_PREDICTIONS_REQUIRED_FIELDS = frozenset({
    'schema_version',
    'task_id',
    'task_type',
    'success',
    'records',
})
TASK_ARTIFACTS_REQUIRED_FIELDS = frozenset({'schema_version', 'artifacts'})
TASK_ARTIFACT_ITEM_REQUIRED_FIELDS = frozenset({'name', 'path', 'kind', 'required'})
TASK_METRICS_FIELDNAMES = (
    'epoch',
    'train_loss',
    'val_loss',
    'accuracy',
    'precision',
    'recall',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'class_recall',
    'class_ap50',
    'object_accuracy',
    'map50',
    'map50_95',
    'num_predictions',
    'num_gt',
    'mask_count',
    'mask_coverage',
    'embedding_count',
    'embedding_dim',
    'retrieval_map',
    'neighbor_purity',
    'review_hit_rate',
    'label_error_rate',
    'latency_ms_per_image',
    'p95_latency_ms',
    'gpu_memory_mb',
    'lr',
)
TASK_TIMING_FIELDNAMES = ('stage', 'elapsed_seconds')
TASK_ALLOWED_STATUSES = frozenset({'pending', 'running', 'finished', 'failed', 'skipped'})


@dataclass(frozen=True, slots=True)
class TaskArtifactValidationIssue:
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class TaskArtifactValidationResult:
    errors: list[TaskArtifactValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def error_messages(self) -> list[str]:
        return [
            f'{issue.path}: {issue.message}'
            for issue in self.errors
        ]


class TaskArtifactValidator:
    def validate_task_dir(self, task_dir: str | Path) -> TaskArtifactValidationResult:
        task_path = Path(task_dir)
        errors: list[TaskArtifactValidationIssue] = []

        if not task_path.exists():
            return TaskArtifactValidationResult(
                errors=[TaskArtifactValidationIssue(path=str(task_path), message='task directory does not exist')],
            )

        for file_name in TASK_ARTIFACT_REQUIRED_FILES:
            if not (task_path / file_name).exists():
                errors.append(TaskArtifactValidationIssue(path=file_name, message='required artifact is missing'))

        if errors:
            return TaskArtifactValidationResult(errors=errors)

        errors.extend(self._validate_task_json(task_path / 'task.json'))
        errors.extend(self._validate_predictions_json(task_path / 'predictions.json'))
        errors.extend(self._validate_artifacts_json(task_path / 'artifacts.json'))
        errors.extend(self._validate_csv_header(task_path / 'metrics.csv', TASK_METRICS_FIELDNAMES))
        errors.extend(self._validate_csv_header(task_path / 'timings.csv', TASK_TIMING_FIELDNAMES))
        errors.extend(self._validate_status(task_path / 'status.marker'))

        return TaskArtifactValidationResult(errors=errors)

    def _validate_task_json(self, path: Path) -> list[TaskArtifactValidationIssue]:
        data, errors = self._read_json(path)
        if errors:
            return errors

        errors.extend(self._missing_fields(path.name, data, TASK_JSON_REQUIRED_FIELDS))
        task = data.get('task')
        if not isinstance(task, dict):
            errors.append(TaskArtifactValidationIssue(path=path.name, message='task must be an object'))
            return errors

        errors.extend(self._missing_fields(f'{path.name}.task', task, TASK_RECORD_REQUIRED_FIELDS))
        self._check_string_field(errors, f'{path.name}.task.task_id', task.get('task_id'))
        self._check_string_field(errors, f'{path.name}.task.task_type', task.get('task_type'))
        self._check_string_field(errors, f'{path.name}.task.input_variant_id', task.get('input_variant_id'))
        self._check_string_field(errors, f'{path.name}.task.status', task.get('status'))
        if task.get('status') not in TASK_ALLOWED_STATUSES:
            errors.append(TaskArtifactValidationIssue(path=f'{path.name}.task.status', message='status is not allowed'))

        return errors

    def _validate_predictions_json(self, path: Path) -> list[TaskArtifactValidationIssue]:
        data, errors = self._read_json(path)
        if errors:
            return errors

        errors.extend(self._missing_fields(path.name, data, TASK_PREDICTIONS_REQUIRED_FIELDS))
        if not isinstance(data.get('records'), list):
            errors.append(TaskArtifactValidationIssue(path=f'{path.name}.records', message='records must be a list'))
        if not isinstance(data.get('success'), bool):
            errors.append(TaskArtifactValidationIssue(path=f'{path.name}.success', message='success must be a boolean'))

        return errors

    def _validate_artifacts_json(self, path: Path) -> list[TaskArtifactValidationIssue]:
        data, errors = self._read_json(path)
        if errors:
            return errors

        errors.extend(self._missing_fields(path.name, data, TASK_ARTIFACTS_REQUIRED_FIELDS))
        artifacts = data.get('artifacts')
        if not isinstance(artifacts, list):
            errors.append(TaskArtifactValidationIssue(path=f'{path.name}.artifacts', message='artifacts must be a list'))
            return errors

        for index, artifact in enumerate(artifacts):
            item_path = f'{path.name}.artifacts[{index}]'
            if not isinstance(artifact, dict):
                errors.append(TaskArtifactValidationIssue(path=item_path, message='artifact entry must be an object'))
                continue
            errors.extend(self._missing_fields(item_path, artifact, TASK_ARTIFACT_ITEM_REQUIRED_FIELDS))
            if 'required' in artifact and not isinstance(artifact.get('required'), bool):
                errors.append(TaskArtifactValidationIssue(path=f'{item_path}.required', message='required must be boolean'))

        return errors

    def _validate_csv_header(self, path: Path, expected: tuple[str, ...]) -> list[TaskArtifactValidationIssue]:
        with path.open(mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                return [TaskArtifactValidationIssue(path=path.name, message='CSV file is empty')]

        if tuple(header) != expected:
            return [
                TaskArtifactValidationIssue(
                    path=path.name,
                    message=f'CSV header mismatch: expected {list(expected)}, got {header}',
                ),
            ]

        return []

    def _validate_status(self, path: Path) -> list[TaskArtifactValidationIssue]:
        status = path.read_text(encoding='utf-8').strip()
        if status not in TASK_ALLOWED_STATUSES:
            return [TaskArtifactValidationIssue(path=path.name, message='status marker is not allowed')]

        return []

    def _read_json(self, path: Path) -> tuple[dict[str, Any], list[TaskArtifactValidationIssue]]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            return {}, [TaskArtifactValidationIssue(path=path.name, message=f'invalid JSON: {exc.msg}')]

        if not isinstance(data, dict):
            return {}, [TaskArtifactValidationIssue(path=path.name, message='JSON root must be an object')]

        return data, []

    def _missing_fields(
        self,
        path: str,
        data: dict[str, Any],
        required_fields: frozenset[str],
    ) -> list[TaskArtifactValidationIssue]:
        missing = sorted(required_fields.difference(data.keys()))
        return [
            TaskArtifactValidationIssue(path=path, message=f'missing required field: {field_name}')
            for field_name in missing
        ]

    def _check_string_field(
        self,
        errors: list[TaskArtifactValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not isinstance(value, str) or not value:
            errors.append(TaskArtifactValidationIssue(path=path, message='field must be a non-empty string'))
