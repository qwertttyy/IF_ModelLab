import csv
import json
from pathlib import Path

from ironflow_exp.engine.core import (
    TASK_ARTIFACT_SCHEMA_VERSION,
    TASK_METRICS_FIELDNAMES,
    TASK_TIMING_FIELDNAMES,
)
from ironflow_exp.engine.domain import TaskExecutionRecord
from ironflow_exp.engine.tasks.base import TaskAdapterResult


class TaskArtifactWriter:
    def write(
        self,
        result_dir: Path,
        record: TaskExecutionRecord,
        adapter_result: TaskAdapterResult,
        elapsed_seconds: float,
        timing_rows: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        result_dir.mkdir(parents=True, exist_ok=True)
        artifacts = [
            *self.base_artifact_manifest(),
            *adapter_result.artifacts,
        ]

        self._write_task_json(
            result_dir=result_dir,
            record=record,
            message=adapter_result.message,
            metadata=adapter_result.metadata,
        )
        self._write_task_log(result_dir=result_dir, record=record, message=adapter_result.message)
        self._write_task_metrics(result_dir=result_dir, rows=adapter_result.metrics)
        self._write_predictions(result_dir=result_dir, record=record, adapter_result=adapter_result)
        self._write_status(result_dir=result_dir, status=adapter_result.status)
        self._write_artifacts(result_dir=result_dir, artifacts=artifacts)
        self._write_summary(result_dir=result_dir, record=record, adapter_result=adapter_result)
        self._write_timings(
            result_dir=result_dir,
            elapsed_seconds=elapsed_seconds,
            rows=timing_rows,
        )

        return artifacts

    def base_artifact_manifest(self) -> list[dict[str, object]]:
        return [
            {'name': 'task_json', 'path': 'task.json', 'kind': 'metadata', 'required': True},
            {'name': 'task_log', 'path': 'task.log', 'kind': 'log', 'required': True},
            {'name': 'metrics', 'path': 'metrics.csv', 'kind': 'metrics', 'required': True},
            {'name': 'predictions', 'path': 'predictions.json', 'kind': 'prediction', 'required': False},
            {'name': 'status', 'path': 'status.marker', 'kind': 'status', 'required': True},
            {'name': 'summary', 'path': 'summary.md', 'kind': 'summary', 'required': True},
            {'name': 'timings', 'path': 'timings.csv', 'kind': 'timing', 'required': True},
        ]

    def _write_task_json(
        self,
        result_dir: Path,
        record: TaskExecutionRecord,
        message: str,
        metadata: dict[str, object],
    ) -> None:
        data = {
            'schema_version': TASK_ARTIFACT_SCHEMA_VERSION,
            'message': message,
            'task': record.to_dict(),
            'metadata': metadata,
        }
        (result_dir / 'task.json').write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _write_task_log(self, result_dir: Path, record: TaskExecutionRecord, message: str) -> None:
        lines = [
            f'task_id={record.task_id}',
            f'task_type={record.task_type}',
            f'input_variant={record.input_variant_id}',
            message,
        ]
        (result_dir / 'task.log').write_text(data='\n'.join(lines) + '\n', encoding='utf-8')

    def _write_task_metrics(self, result_dir: Path, rows: list[dict[str, object]]) -> None:
        normalized_rows = rows or [self.empty_metric_row()]
        with (result_dir / 'metrics.csv').open(mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=TASK_METRICS_FIELDNAMES)
            writer.writeheader()
            for row in normalized_rows:
                writer.writerow({field_name: row.get(field_name, '') for field_name in TASK_METRICS_FIELDNAMES})

    def empty_metric_row(self) -> dict[str, object]:
        return {
            field_name: 1 if field_name == 'epoch' else ''
            for field_name in TASK_METRICS_FIELDNAMES
        }

    def _write_predictions(
        self,
        result_dir: Path,
        record: TaskExecutionRecord,
        adapter_result: TaskAdapterResult,
    ) -> None:
        predictions = {
            'schema_version': TASK_ARTIFACT_SCHEMA_VERSION,
            'task_id': record.task_id,
            'task_type': record.task_type,
            'success': adapter_result.success,
            'records': adapter_result.predictions if adapter_result.success else [],
        }
        (result_dir / 'predictions.json').write_text(
            data=json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _write_status(self, result_dir: Path, status: str) -> None:
        (result_dir / 'status.marker').write_text(data=f'{status}\n', encoding='utf-8')

    def _write_artifacts(self, result_dir: Path, artifacts: list[dict[str, object]]) -> None:
        data = {
            'schema_version': TASK_ARTIFACT_SCHEMA_VERSION,
            'artifacts': artifacts,
        }
        (result_dir / 'artifacts.json').write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _write_summary(
        self,
        result_dir: Path,
        record: TaskExecutionRecord,
        adapter_result: TaskAdapterResult,
    ) -> None:
        lines = [
            '# Task Summary',
            '',
            f'- task_id: {record.task_id}',
            f'- task_type: {record.task_type}',
            f'- status: {record.status}',
            f'- model_id: {record.model_id or ""}',
            f'- adapter: {record.adapter or ""}',
            f'- input_variant: {record.input_variant_id}',
            f'- message: {adapter_result.message}',
            '',
        ]
        (result_dir / 'summary.md').write_text(data='\n'.join(lines), encoding='utf-8')

    def _write_timings(
        self,
        result_dir: Path,
        elapsed_seconds: float,
        rows: list[dict[str, object]] | None = None,
    ) -> None:
        normalized_rows = rows or [{'stage': 'task_total', 'elapsed_seconds': elapsed_seconds}]
        with (result_dir / 'timings.csv').open(mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=TASK_TIMING_FIELDNAMES)
            writer.writeheader()
            for row in normalized_rows:
                writer.writerow(
                    {
                        'stage': str(row.get('stage') or 'task_total'),
                        'elapsed_seconds': self._format_elapsed_seconds(row.get('elapsed_seconds')),
                    },
                )

    def _format_elapsed_seconds(self, value: object) -> str:
        try:
            return f'{float(value):.6f}'
        except (TypeError, ValueError):
            return '0.000000'
