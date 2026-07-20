from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRIC_FIELDS = (
    'epoch',
    'train_loss',
    'val_loss',
    'accuracy',
    'precision',
    'recall',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'map50',
    'map50_95',
    'num_predictions',
    'num_gt',
    'latency_ms_per_image',
)

ASSET_PATTERNS = (
    '*confusion_matrix*.png',
    '*Confusion*.png',
    '*PR_curve*.png',
    '*F1_curve*.png',
    '*P_curve*.png',
    '*R_curve*.png',
    '*results*.png',
    '*val_batch*labels*.jpg',
    '*val_batch*pred*.jpg',
    '*val_batch*labels*.png',
    '*val_batch*pred*.png',
    '*test*pred*.jpg',
    '*test*pred*.png',
    '*preview*.jpg',
    '*preview*.png',
)


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    metrics: list[dict[str, str]]
    elapsed_seconds: float | None


@dataclass(frozen=True)
class ExperimentInfo:
    experiment_id: str
    result_dir: Path
    kind: str
    status: str
    root_metrics: list[dict[str, str]]
    train_tasks: tuple[TaskInfo, ...]
    predict_tasks: tuple[TaskInfo, ...]
    train_seconds: float | None
    predict_seconds: float | None
    latest: dict[str, str]
    copied_assets: tuple[Path, ...]

    @property
    def primary_score(self) -> float | None:
        if self.kind == 'detection':
            return first_float(self.latest, 'map50_95', 'map50')
        if self.kind == 'classification':
            return first_float(self.latest, 'macro_f1', 'accuracy')
        return first_float(self.latest, 'map50_95', 'map50', 'macro_f1', 'accuracy')

    @property
    def latency_ms_per_image(self) -> float | None:
        value = first_float(self.latest, 'latency_ms_per_image')
        if value is not None:
            return value
        count = first_float(self.latest, 'num_predictions', 'num_gt')
        if count and self.predict_seconds:
            return self.predict_seconds * 1000.0 / count
        return None

    @property
    def fps(self) -> float | None:
        latency = self.latency_ms_per_image
        if latency and latency > 0:
            return 1000.0 / latency
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate standalone visual reports from collected IronFlow experiment result folders.',
    )
    parser.add_argument('--runs-root', type=Path, default=Path('runs') / 'e')
    parser.add_argument('--output-dir', type=Path, default=Path('reports') / 'visual_reports' / 'latest')
    parser.add_argument('--include', default='exp_', help='Only include experiment folders containing this text.')
    parser.add_argument('--force', action='store_true', help='Replace an existing output folder.')
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f'output exists: {args.output_dir} (use --force)')
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(runs_root=args.runs_root, include=args.include, output_dir=args.output_dir)
    if not experiments:
        raise SystemExit(f'no experiments found under: {args.runs_root}')

    write_per_experiment_reports(experiments=experiments, output_dir=args.output_dir)
    write_comparison_reports(experiments=experiments, output_dir=args.output_dir)
    write_index(experiments=experiments, output_dir=args.output_dir)
    print(f'visual report ready: {args.output_dir}')
    print(f'experiments: {len(experiments)}')
    return 0


def discover_experiments(*, runs_root: Path, include: str, output_dir: Path) -> list[ExperimentInfo]:
    experiments: list[ExperimentInfo] = []
    if not runs_root.exists():
        return experiments
    for result_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and include in p.name):
        root_metrics = read_metrics_csv(result_dir / 'metrics.csv')
        tasks = read_tasks(result_dir=result_dir)
        train_tasks = tuple(t for t in tasks if t.task_id.startswith('00_train') or '_train_' in t.task_id)
        predict_tasks = tuple(t for t in tasks if 'predict' in t.task_id or 'inference' in t.task_id)
        latest = root_metrics[-1] if root_metrics else latest_task_metric(tasks)
        kind = infer_kind(result_dir=result_dir, latest=latest)
        exp_out = output_dir / kind / safe_name(result_dir.name)
        assets = copy_known_assets(result_dir=result_dir, experiment_output_dir=exp_out)
        experiments.append(
            ExperimentInfo(
                experiment_id=result_dir.name,
                result_dir=result_dir,
                kind=kind,
                status=read_text(result_dir / 'status.marker').strip() or 'unknown',
                root_metrics=root_metrics,
                train_tasks=train_tasks,
                predict_tasks=predict_tasks,
                train_seconds=sum_optional(t.elapsed_seconds for t in train_tasks),
                predict_seconds=sum_optional(t.elapsed_seconds for t in predict_tasks),
                latest=latest,
                copied_assets=tuple(assets),
            )
        )
    return experiments


def read_tasks(*, result_dir: Path) -> list[TaskInfo]:
    tasks_dir = result_dir / 'tasks'
    if not tasks_dir.exists():
        return []
    tasks: list[TaskInfo] = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        tasks.append(
            TaskInfo(
                task_id=task_dir.name,
                metrics=read_metrics_csv(task_dir / 'metrics.csv'),
                elapsed_seconds=read_task_seconds(task_dir / 'timings.csv'),
            )
        )
    return tasks


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', errors='ignore', newline='') as file:
        return list(csv.DictReader(file))


def read_task_seconds(path: Path) -> float | None:
    rows = read_metrics_csv(path)
    if not rows:
        return None
    last = rows[-1]
    return parse_float(last.get('elapsed_seconds'))


def latest_task_metric(tasks: Iterable[TaskInfo]) -> dict[str, str]:
    for task in reversed(list(tasks)):
        if task.metrics:
            return task.metrics[-1]
    return {}


def copy_known_assets(*, result_dir: Path, experiment_output_dir: Path) -> list[Path]:
    assets_dir = experiment_output_dir / 'assets'
    copied: list[Path] = []
    seen: set[Path] = set()
    for pattern in ASSET_PATTERNS:
        for src in result_dir.rglob(pattern):
            if not src.is_file() or src in seen:
                continue
            if any(part in {'code_package', '.git', '__pycache__'} for part in src.parts):
                continue
            seen.add(src)
            rel = src.relative_to(result_dir)
            dst = assets_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def write_per_experiment_reports(*, experiments: list[ExperimentInfo], output_dir: Path) -> None:
    for exp in experiments:
        exp_out = output_dir / exp.kind / safe_name(exp.experiment_id)
        exp_out.mkdir(parents=True, exist_ok=True)
        write_metric_curves(exp=exp, output_dir=exp_out)
        write_experiment_summary(exp=exp, output_dir=exp_out)


def write_metric_curves(*, exp: ExperimentInfo, output_dir: Path) -> None:
    matplotlib, plt = import_plotting()
    del matplotlib
    train_metrics = []
    for task in exp.train_tasks:
        train_metrics.extend(task.metrics)
    if not train_metrics:
        train_metrics = exp.root_metrics
    if not train_metrics:
        return
    curve_specs = [
        ('loss_curve.png', ('train_loss', 'val_loss'), 'Loss'),
        ('detection_map_curve.png', ('map50', 'map50_95'), 'Detection mAP'),
        ('classification_score_curve.png', ('accuracy', 'macro_f1'), 'Classification Score'),
        ('precision_recall_curve_by_epoch.png', ('precision', 'recall'), 'Precision / Recall'),
    ]
    for filename, fields, title in curve_specs:
        series = []
        for field in fields:
            points = metric_points(rows=train_metrics, field=field)
            if points:
                series.append((field, points))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        for label, points in series:
            xs = [x for x, _y in points]
            ys = [y for _x, y in points]
            ax.plot(xs, ys, marker='o', linewidth=1.8, label=label)
        ax.set_title(f'{exp.experiment_id} - {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def write_experiment_summary(*, exp: ExperimentInfo, output_dir: Path) -> None:
    lines = [
        f'# {exp.experiment_id}',
        '',
        f'- Kind: `{exp.kind}`',
        f'- Status: `{exp.status}`',
        f'- Result dir: `{exp.result_dir}`',
        f'- Train time: `{format_seconds(exp.train_seconds)}`',
        f'- Predict time: `{format_seconds(exp.predict_seconds)}`',
        f'- Latency: `{format_float(exp.latency_ms_per_image)} ms/img`',
        f'- FPS: `{format_float(exp.fps)}`',
        f'- Primary score: `{format_float(exp.primary_score)}`',
        '',
        '## Latest Metrics',
        '',
        '| Metric | Value |',
        '|---|---:|',
    ]
    for field in METRIC_FIELDS:
        value = exp.latest.get(field, '')
        if value not in ('', None):
            lines.append(f'| `{field}` | {value} |')
    lines.extend(['', '## Collected Assets', ''])
    if exp.copied_assets:
        for asset in exp.copied_assets:
            lines.append(f'- `{asset.relative_to(output_dir)}`')
    else:
        lines.append('- No framework-generated visual assets found. Common metric curves were generated where metrics exist.')
    lines.append('')
    (output_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def write_comparison_reports(*, experiments: list[ExperimentInfo], output_dir: Path) -> None:
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(experiments=experiments, path=comparison_dir / 'experiment_summary.csv')
    write_comparison_plots(experiments=experiments, output_dir=comparison_dir)


def write_summary_csv(*, experiments: list[ExperimentInfo], path: Path) -> None:
    fieldnames = [
        'experiment_id',
        'kind',
        'status',
        'primary_score',
        'accuracy',
        'macro_f1',
        'precision',
        'recall',
        'map50',
        'map50_95',
        'latency_ms_per_image',
        'fps',
        'train_seconds',
        'predict_seconds',
        'num_predictions',
        'num_gt',
        'result_dir',
    ]
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            writer.writerow({
                'experiment_id': exp.experiment_id,
                'kind': exp.kind,
                'status': exp.status,
                'primary_score': csv_float(exp.primary_score),
                'accuracy': csv_float(first_float(exp.latest, 'accuracy')),
                'macro_f1': csv_float(first_float(exp.latest, 'macro_f1')),
                'precision': csv_float(first_float(exp.latest, 'precision')),
                'recall': csv_float(first_float(exp.latest, 'recall')),
                'map50': csv_float(first_float(exp.latest, 'map50')),
                'map50_95': csv_float(first_float(exp.latest, 'map50_95')),
                'latency_ms_per_image': csv_float(exp.latency_ms_per_image),
                'fps': csv_float(exp.fps),
                'train_seconds': csv_float(exp.train_seconds),
                'predict_seconds': csv_float(exp.predict_seconds),
                'num_predictions': csv_float(first_float(exp.latest, 'num_predictions')),
                'num_gt': csv_float(first_float(exp.latest, 'num_gt')),
                'result_dir': str(exp.result_dir),
            })


def write_comparison_plots(*, experiments: list[ExperimentInfo], output_dir: Path) -> None:
    matplotlib, plt = import_plotting()
    del matplotlib
    plot_bar(
        plt=plt,
        path=output_dir / 'detection_map50_95.png',
        title='Detection mAP50-95',
        experiments=[e for e in experiments if e.kind == 'detection'],
        value_getter=lambda e: first_float(e.latest, 'map50_95', 'map50'),
        ylabel='mAP',
    )
    plot_bar(
        plt=plt,
        path=output_dir / 'classification_macro_f1.png',
        title='Classification Macro F1',
        experiments=[e for e in experiments if e.kind == 'classification'],
        value_getter=lambda e: first_float(e.latest, 'macro_f1', 'accuracy'),
        ylabel='Score',
    )
    plot_bar(
        plt=plt,
        path=output_dir / 'latency_ms_per_image.png',
        title='Latency by Model',
        experiments=experiments,
        value_getter=lambda e: e.latency_ms_per_image,
        ylabel='ms/img',
        lower_is_better=True,
    )
    plot_bar(
        plt=plt,
        path=output_dir / 'fps_by_model.png',
        title='FPS by Model',
        experiments=experiments,
        value_getter=lambda e: e.fps,
        ylabel='FPS',
    )
    plot_scatter(
        plt=plt,
        path=output_dir / 'score_vs_latency.png',
        title='Score vs Latency',
        experiments=[e for e in experiments if e.primary_score is not None and e.latency_ms_per_image is not None],
        x_getter=lambda e: e.latency_ms_per_image,
        y_getter=lambda e: e.primary_score,
        xlabel='Latency (ms/img)',
        ylabel='Primary Score',
    )


def plot_bar(
    *,
    plt: object,
    path: Path,
    title: str,
    experiments: list[ExperimentInfo],
    value_getter,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    rows = [(exp.experiment_id, value_getter(exp)) for exp in experiments]
    rows = [(label, value) for label, value in rows if value is not None]
    if not rows:
        return
    rows.sort(key=lambda item: item[1], reverse=not lower_is_better)
    labels = [short_label(label) for label, _value in rows]
    values = [float(value) for _label, value in rows]
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.55), 5), dpi=140)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(
    *,
    plt: object,
    path: Path,
    title: str,
    experiments: list[ExperimentInfo],
    x_getter,
    y_getter,
    xlabel: str,
    ylabel: str,
) -> None:
    if not experiments:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for exp in experiments:
        x = x_getter(exp)
        y = y_getter(exp)
        if x is None or y is None:
            continue
        ax.scatter(float(x), float(y), s=70, alpha=0.8)
        ax.annotate(short_label(exp.experiment_id), (float(x), float(y)), textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_index(*, experiments: list[ExperimentInfo], output_dir: Path) -> None:
    lines = [
        '# IronFlow Visual Report',
        '',
        '## Outputs',
        '',
        '- `comparison/experiment_summary.csv`',
        '- `comparison/detection_map50_95.png`',
        '- `comparison/classification_macro_f1.png`',
        '- `comparison/latency_ms_per_image.png`',
        '- `comparison/fps_by_model.png`',
        '- `comparison/score_vs_latency.png`',
        '',
        '## Experiments',
        '',
        '| Kind | Experiment | Status | Score | Latency ms/img | FPS |',
        '|---|---|---|---:|---:|---:|',
    ]
    for exp in sorted(experiments, key=lambda e: (e.kind, -(e.primary_score or -1), e.experiment_id)):
        rel = Path(exp.kind) / safe_name(exp.experiment_id) / 'summary.md'
        lines.append(
            f'| {exp.kind} | [{exp.experiment_id}]({rel.as_posix()}) | {exp.status} | '
            f'{format_float(exp.primary_score)} | {format_float(exp.latency_ms_per_image)} | {format_float(exp.fps)} |'
        )
    lines.append('')
    (output_dir / 'index.md').write_text('\n'.join(lines), encoding='utf-8')


def metric_points(*, rows: list[dict[str, str]], field: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    fallback_epoch = 1
    for row in rows:
        y = parse_float(row.get(field))
        if y is None:
            fallback_epoch += 1
            continue
        x = parse_float(row.get('epoch'))
        if x is None:
            x = float(fallback_epoch)
        points.append((x, y))
        fallback_epoch += 1
    return points


def infer_kind(*, result_dir: Path, latest: dict[str, str]) -> str:
    name = result_dir.name.lower()
    if 'classifier' in name or first_float(latest, 'macro_f1', 'accuracy') is not None and first_float(latest, 'map50', 'map50_95') is None:
        return 'classification'
    if 'detector' in name or first_float(latest, 'map50', 'map50_95') is not None:
        return 'detection'
    return 'other'


def first_float(row: dict[str, str], *fields: str) -> float | None:
    for field in fields:
        value = parse_float(row.get(field))
        if value is not None:
            return value
    return None


def parse_float(value: object) -> float | None:
    if value in (None, ''):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def sum_optional(values: Iterable[float | None]) -> float | None:
    actual = [value for value in values if value is not None]
    if not actual:
        return None
    return sum(actual)


def format_seconds(value: float | None) -> str:
    if value is None:
        return '-'
    seconds = int(round(value))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    rest = seconds % 60
    if hours:
        return f'{hours}h {minutes:02d}m {rest:02d}s'
    if minutes:
        return f'{minutes}m {rest:02d}s'
    return f'{rest}s'


def format_float(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.4f}'


def csv_float(value: float | None) -> str:
    if value is None:
        return ''
    return f'{value:.8g}'


def read_text(path: Path) -> str:
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8', errors='ignore')


def safe_name(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in value)


def short_label(value: str) -> str:
    for prefix in ('exp_aug_', 'exp_candidate_', 'exp_candiate_', 'exp_top10_', 'recommended_'):
        value = value.replace(prefix, '')
    return value[:42]


def import_plotting():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return matplotlib, plt


if __name__ == '__main__':
    raise SystemExit(main())
