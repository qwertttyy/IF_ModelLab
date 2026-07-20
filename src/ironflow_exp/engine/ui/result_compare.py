from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ironflow_exp.engine.domain import EngineMetricRecord, ExperimentRecord
from ironflow_exp.engine.storage import SQLiteExperimentStorage
from ironflow_exp.engine.time_display import iso_to_kst_display


@dataclass(frozen=True, slots=True)
class ExperimentComparisonRow:
    experiment_id: str
    name: str
    status: str
    runner_type: str
    server_name: str
    best_metric_name: str
    best_metric_value: float | None
    metric_count: int
    last_epoch: int | None
    duration_seconds: float | None
    duration_display: str
    task_total_seconds: float | None
    train_seconds: float | None
    predict_seconds: float | None
    inference_pipeline_seconds: float | None
    started_at_display: str
    finished_at_display: str
    latest_accuracy: float | None
    latest_precision: float | None
    latest_recall: float | None
    latest_macro_f1: float | None
    latest_object_accuracy: float | None
    latest_map50: float | None
    latest_map50_95: float | None
    latest_p95_latency_ms: float | None
    result_path: str
    summary_path: str
    error_type: str

    @property
    def best_metric_display(self) -> str:
        if not self.best_metric_name:
            return ''
        if self.best_metric_value is None:
            return self.best_metric_name

        return f'{self.best_metric_name}={_format_float(self.best_metric_value)}'

    def to_tree_values(self) -> tuple[str, ...]:
        return (
            self.experiment_id,
            self.status,
            self.best_metric_display,
            self.duration_display,
            '' if self.last_epoch is None else str(self.last_epoch),
            _format_optional_float(self.latest_accuracy),
            _format_optional_float(self.latest_macro_f1),
            _format_optional_float(self.latest_map50),
            _format_optional_float(self.latest_map50_95),
            _format_optional_float(self.latest_p95_latency_ms),
            self.name,
        )


def load_experiment_comparison(*, db_path: str | Path) -> list[ExperimentComparisonRow]:
    storage = SQLiteExperimentStorage(db_path=db_path)
    records = storage.list_experiments()

    return [
        comparison_row_from_record(record=record, metrics=storage.list_metrics(record.experiment_id))
        for record in sorted(records, key=_sort_record, reverse=True)
    ]


def comparison_row_from_record(
    *,
    record: ExperimentRecord,
    metrics: list[EngineMetricRecord],
) -> ExperimentComparisonRow:
    latest_metric = _latest_metric(metrics=metrics)
    duration_seconds = _duration_seconds(started_at=record.started_at, finished_at=record.finished_at)
    metadata = record.metadata
    summary_path = metadata.get('summary_path')
    timing_summary = _timing_summary(metadata=metadata)

    return ExperimentComparisonRow(
        experiment_id=record.experiment_id,
        name=record.name,
        status=record.status.value,
        runner_type=record.runner_type,
        server_name=record.server_name or '',
        best_metric_name=record.best_metric_name or '',
        best_metric_value=record.best_metric_value,
        metric_count=len(metrics),
        last_epoch=_last_epoch(metrics=metrics),
        duration_seconds=duration_seconds,
        duration_display=_format_duration(duration_seconds),
        task_total_seconds=_optional_summary_float(summary=timing_summary, key='task_total_seconds'),
        train_seconds=_optional_summary_float(summary=timing_summary, key='train_seconds'),
        predict_seconds=_optional_summary_float(summary=timing_summary, key='predict_seconds'),
        inference_pipeline_seconds=_optional_summary_float(summary=timing_summary, key='inference_pipeline_seconds'),
        started_at_display=iso_to_kst_display(record.started_at) or '',
        finished_at_display=iso_to_kst_display(record.finished_at) or '',
        latest_accuracy=latest_metric.accuracy if latest_metric is not None else None,
        latest_precision=latest_metric.precision if latest_metric is not None else None,
        latest_recall=latest_metric.recall if latest_metric is not None else None,
        latest_macro_f1=latest_metric.macro_f1 if latest_metric is not None else None,
        latest_object_accuracy=latest_metric.object_accuracy if latest_metric is not None else None,
        latest_map50=latest_metric.map50 if latest_metric is not None else None,
        latest_map50_95=latest_metric.map50_95 if latest_metric is not None else None,
        latest_p95_latency_ms=latest_metric.p95_latency_ms if latest_metric is not None else None,
        result_path=record.result_path or '',
        summary_path=str(summary_path) if isinstance(summary_path, str) else '',
        error_type=record.error_type or '',
    )


def _timing_summary(*, metadata: dict[str, object]) -> dict[str, object]:
    value = metadata.get('timing_summary')
    if isinstance(value, dict):
        return value

    return {}


def _optional_summary_float(*, summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_tree_values(metrics: list[EngineMetricRecord]) -> list[tuple[str, ...]]:
    return [
        (
            '' if metric.epoch is None else str(metric.epoch),
            _format_optional_float(metric.train_loss),
            _format_optional_float(metric.val_loss),
            _format_optional_float(metric.accuracy),
            _format_optional_float(metric.precision),
            _format_optional_float(metric.recall),
            _format_optional_float(metric.macro_precision),
            _format_optional_float(metric.macro_recall),
            _format_optional_float(metric.macro_f1),
            _format_optional_float(metric.class_recall),
            _format_optional_float(metric.class_ap50),
            _format_optional_float(metric.object_accuracy),
            _format_optional_float(metric.map50),
            _format_optional_float(metric.map50_95),
            _format_optional_float(metric.num_predictions),
            _format_optional_float(metric.num_gt),
            _format_optional_float(metric.mask_count),
            _format_optional_float(metric.mask_coverage),
            _format_optional_float(metric.embedding_count),
            _format_optional_float(metric.embedding_dim),
            _format_optional_float(metric.retrieval_map),
            _format_optional_float(metric.neighbor_purity),
            _format_optional_float(metric.review_hit_rate),
            _format_optional_float(metric.label_error_rate),
            _format_optional_float(metric.latency_ms_per_image),
            _format_optional_float(metric.p95_latency_ms),
            _format_optional_float(metric.gpu_memory_mb),
            _format_optional_float(metric.lr),
        )
        for metric in metrics
    ]


def _sort_record(record: ExperimentRecord) -> str:
    return record.finished_at or record.started_at or record.created_at or record.experiment_id


def _latest_metric(*, metrics: list[EngineMetricRecord]) -> EngineMetricRecord | None:
    if not metrics:
        return None

    return sorted(metrics, key=lambda metric: (metric.epoch is None, metric.epoch or -1))[-1]


def _last_epoch(*, metrics: list[EngineMetricRecord]) -> int | None:
    epochs = [
        metric.epoch
        for metric in metrics
        if metric.epoch is not None
    ]
    if not epochs:
        return None

    return max(epochs)


def _duration_seconds(*, started_at: str | None, finished_at: str | None) -> float | None:
    started = _parse_iso(value=started_at)
    finished = _parse_iso(value=finished_at)
    if started is None or finished is None:
        return None

    return max((finished - started).total_seconds(), 0.0)


def _parse_iso(*, value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith('Z'):
        normalized = f'{normalized[:-1]}+00:00'
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _format_duration(value: float | None) -> str:
    if value is None:
        return ''
    seconds = int(round(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'

    return f'{minutes:d}:{seconds:02d}'


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ''

    return _format_float(value)


def _format_float(value: float) -> str:
    return f'{value:.4g}'
