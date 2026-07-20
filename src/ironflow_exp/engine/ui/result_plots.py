from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ironflow_exp.engine.domain import EngineMetricRecord
from ironflow_exp.engine.storage import SQLiteExperimentStorage
from ironflow_exp.engine.ui.result_compare import ExperimentComparisonRow, comparison_row_from_record


SUMMARY_FIELDNAMES = [
    'experiment_id',
    'name',
    'status',
    'runner_type',
    'server_name',
    'best_metric_name',
    'best_metric_value',
    'metric_count',
    'last_epoch',
    'duration_seconds',
    'task_total_seconds',
    'train_seconds',
    'predict_seconds',
    'inference_pipeline_seconds',
    'latest_accuracy',
    'latest_precision',
    'latest_recall',
    'latest_macro_f1',
    'latest_object_accuracy',
    'latest_map50',
    'latest_map50_95',
    'latest_p95_latency_ms',
    'result_path',
    'summary_path',
    'error_type',
]

HISTORY_FIELDNAMES = [
    'experiment_id',
    'epoch',
    'train_loss',
    'val_loss',
    'accuracy',
    'precision',
    'recall',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'object_accuracy',
    'map50',
    'map50_95',
    'latency_ms_per_image',
    'p95_latency_ms',
    'gpu_memory_mb',
    'lr',
]


@dataclass(frozen=True, slots=True)
class ComparisonPlotResult:
    output_dir: Path
    summary_csv: Path
    history_csv: Path
    plot_paths: tuple[Path, ...]
    skipped_plots: tuple[str, ...]


def generate_comparison_artifacts(
    *,
    db_path: str | Path,
    output_dir: str | Path = Path('runs') / 'comparisons' / 'latest',
) -> ComparisonPlotResult:
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = SQLiteExperimentStorage(db_path=db_path)
    rows_and_metrics = []
    for record in storage.list_experiments():
        metrics = storage.list_metrics(record.experiment_id)
        rows_and_metrics.append((comparison_row_from_record(record=record, metrics=metrics), metrics))

    rows = [row for row, _metrics in rows_and_metrics]
    summary_csv = output_dir / 'comparison_summary.csv'
    history_csv = output_dir / 'metric_history.csv'
    _write_summary_csv(path=summary_csv, rows=rows)
    _write_history_csv(path=history_csv, rows_and_metrics=rows_and_metrics)

    plot_paths, skipped_plots = _write_plots(output_dir=output_dir, rows=rows, rows_and_metrics=rows_and_metrics)
    return ComparisonPlotResult(
        output_dir=output_dir,
        summary_csv=summary_csv,
        history_csv=history_csv,
        plot_paths=tuple(plot_paths),
        skipped_plots=tuple(skipped_plots),
    )


def _write_summary_csv(*, path: Path, rows: Iterable[ExperimentComparisonRow]) -> None:
    with path.open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'experiment_id': row.experiment_id,
                'name': row.name,
                'status': row.status,
                'runner_type': row.runner_type,
                'server_name': row.server_name,
                'best_metric_name': row.best_metric_name,
                'best_metric_value': _csv_float(row.best_metric_value),
                'metric_count': row.metric_count,
                'last_epoch': '' if row.last_epoch is None else row.last_epoch,
                'duration_seconds': _csv_float(row.duration_seconds),
                'task_total_seconds': _csv_float(row.task_total_seconds),
                'train_seconds': _csv_float(row.train_seconds),
                'predict_seconds': _csv_float(row.predict_seconds),
                'inference_pipeline_seconds': _csv_float(row.inference_pipeline_seconds),
                'latest_accuracy': _csv_float(row.latest_accuracy),
                'latest_precision': _csv_float(row.latest_precision),
                'latest_recall': _csv_float(row.latest_recall),
                'latest_macro_f1': _csv_float(row.latest_macro_f1),
                'latest_object_accuracy': _csv_float(row.latest_object_accuracy),
                'latest_map50': _csv_float(row.latest_map50),
                'latest_map50_95': _csv_float(row.latest_map50_95),
                'latest_p95_latency_ms': _csv_float(row.latest_p95_latency_ms),
                'result_path': row.result_path,
                'summary_path': row.summary_path,
                'error_type': row.error_type,
            })


def _write_history_csv(
    *,
    path: Path,
    rows_and_metrics: Iterable[tuple[ExperimentComparisonRow, list[EngineMetricRecord]]],
) -> None:
    with path.open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        for row, metrics in rows_and_metrics:
            for metric in metrics:
                writer.writerow({
                    'experiment_id': row.experiment_id,
                    'epoch': '' if metric.epoch is None else metric.epoch,
                    'train_loss': _csv_float(metric.train_loss),
                    'val_loss': _csv_float(metric.val_loss),
                    'accuracy': _csv_float(metric.accuracy),
                    'precision': _csv_float(metric.precision),
                    'recall': _csv_float(metric.recall),
                    'macro_precision': _csv_float(metric.macro_precision),
                    'macro_recall': _csv_float(metric.macro_recall),
                    'macro_f1': _csv_float(metric.macro_f1),
                    'object_accuracy': _csv_float(metric.object_accuracy),
                    'map50': _csv_float(metric.map50),
                    'map50_95': _csv_float(metric.map50_95),
                    'latency_ms_per_image': _csv_float(metric.latency_ms_per_image),
                    'p95_latency_ms': _csv_float(metric.p95_latency_ms),
                    'gpu_memory_mb': _csv_float(metric.gpu_memory_mb),
                    'lr': _csv_float(metric.lr),
                })


def _write_plots(
    *,
    output_dir: Path,
    rows: list[ExperimentComparisonRow],
    rows_and_metrics: list[tuple[ExperimentComparisonRow, list[EngineMetricRecord]]],
) -> tuple[list[Path], list[str]]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_paths: list[Path] = []
    skipped: list[str] = []
    scatter_specs = [
        (
            'detection_speed_map50.png',
            'Detection Speed vs mAP',
            lambda row: _first_number(row.inference_pipeline_seconds, row.predict_seconds, row.duration_seconds),
            lambda row: _first_number(row.latest_map50, row.latest_map50_95),
            'Inference seconds',
            'mAP',
        ),
        (
            'classification_speed_f1.png',
            'Classification Speed vs Macro F1',
            lambda row: _first_number(row.inference_pipeline_seconds, row.predict_seconds, row.duration_seconds),
            lambda row: _first_number(row.latest_macro_f1, row.latest_accuracy),
            'Inference seconds',
            'Macro F1 / Accuracy',
        ),
        (
            'end_to_end_speed_score.png',
            'End-to-end Speed vs Score',
            lambda row: _first_number(row.inference_pipeline_seconds, row.task_total_seconds, row.duration_seconds),
            lambda row: _first_number(row.latest_object_accuracy, row.latest_macro_f1, row.best_metric_value),
            'Pipeline seconds',
            'End-to-end score',
        ),
        (
            'training_efficiency.png',
            'Training Time vs Best Metric',
            lambda row: _first_number(row.train_seconds, row.task_total_seconds, row.duration_seconds),
            lambda row: row.best_metric_value,
            'Training seconds',
            'Best metric',
        ),
    ]
    for filename, title, x_getter, y_getter, x_label, y_label in scatter_specs:
        points = [
            (x_getter(row), y_getter(row), row.experiment_id)
            for row in rows
            if x_getter(row) is not None and y_getter(row) is not None
        ]
        if not points:
            skipped.append(filename)
            continue
        path = output_dir / filename
        _scatter_plot(path=path, title=title, x_label=x_label, y_label=y_label, points=points, plt=plt)
        plot_paths.append(path)

    curve_specs = [
        ('epoch_accuracy_curve.png', 'Epoch Accuracy', 'accuracy', 'Accuracy'),
        ('epoch_macro_f1_curve.png', 'Epoch Macro F1', 'macro_f1', 'Macro F1'),
        ('epoch_map50_curve.png', 'Epoch mAP50', 'map50', 'mAP50'),
        ('epoch_loss_curve.png', 'Epoch Loss', 'train_loss', 'Loss'),
    ]
    for filename, title, metric_name, y_label in curve_specs:
        series = _metric_series(rows_and_metrics=rows_and_metrics, metric_name=metric_name)
        if not series:
            skipped.append(filename)
            continue
        path = output_dir / filename
        _line_plot(path=path, title=title, x_label='Epoch', y_label=y_label, series=series, plt=plt)
        plot_paths.append(path)

    return plot_paths, skipped


def _scatter_plot(
    *,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float | None, float | None, str]],
    plt: object,
) -> None:
    figure, axes = plt.subplots(figsize=(8, 5), dpi=140)
    xs = [float(x) for x, _y, _label in points if x is not None]
    ys = [float(y) for _x, y, _label in points if y is not None]
    axes.scatter(xs, ys, s=72, alpha=0.78)
    for x, y, label in points:
        if x is not None and y is not None:
            axes.annotate(label, (float(x), float(y)), textcoords='offset points', xytext=(5, 4), fontsize=8)
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(True, linestyle=':', linewidth=0.7, alpha=0.55)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _line_plot(
    *,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[int, float]]],
    plt: object,
) -> None:
    figure, axes = plt.subplots(figsize=(8, 5), dpi=140)
    for experiment_id, points in series.items():
        sorted_points = sorted(points, key=lambda item: item[0])
        axes.plot(
            [epoch for epoch, _value in sorted_points],
            [value for _epoch, value in sorted_points],
            marker='o',
            linewidth=1.8,
            label=experiment_id,
        )
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(True, linestyle=':', linewidth=0.7, alpha=0.55)
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _metric_series(
    *,
    rows_and_metrics: Iterable[tuple[ExperimentComparisonRow, list[EngineMetricRecord]]],
    metric_name: str,
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for row, metrics in rows_and_metrics:
        points: list[tuple[int, float]] = []
        for metric in metrics:
            if metric.epoch is None:
                continue
            value = getattr(metric, metric_name)
            if value is not None:
                points.append((metric.epoch, float(value)))
        if points:
            series[row.experiment_id] = points

    return series


def _first_number(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return float(value)

    return None


def _csv_float(value: float | None) -> str:
    if value is None:
        return ''

    return f'{float(value):.12g}'
