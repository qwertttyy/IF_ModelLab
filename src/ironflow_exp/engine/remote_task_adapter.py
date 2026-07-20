import csv
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.core import TASK_METRICS_FIELDNAMES

SUCCESS_EXIT_CODE = 0
TASK_FAILED_EXIT_CODE = 21
UNHANDLED_ERROR_EXIT_CODE = 20


@dataclass(frozen=True, slots=True)
class RemoteTaskAdapterArgs:
    output_dir: Path
    config_path: Path | None = None
    experiment_id: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    return run_remote_task_adapter(args=args)


def run_remote_task_adapter(args: RemoteTaskAdapterArgs) -> int:
    output_dir = args.output_dir.resolve()
    experiment_dir = output_dir.parent
    code_dir = experiment_dir / 'workspace' / 'code'
    config_path = args.config_path or experiment_dir / 'workspace' / 'config.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    write_log(output_dir=output_dir, message='remote task adapter execution started')

    try:
        from ironflow_exp.engine.configs import EngineConfigLoader
        from ironflow_exp.engine.core import TaskOrchestrationService
        from ironflow_exp.engine.tasks import AdapterTaskExecutor, build_builtin_task_adapter_registry

        config = EngineConfigLoader().from_dict(json.loads(config_path.read_text(encoding='utf-8')))
        config = rebase_config_paths(config=config, code_dir=code_dir)
        os.chdir(code_dir)
        plan = TaskOrchestrationService().build_plan(
            config=config,
            experiment_id=args.experiment_id or config.experiment.name,
            result_dir=output_dir,
        )
        executor = AdapterTaskExecutor(registry=build_builtin_task_adapter_registry())
        result = executor.execute_plan(plan=plan)
        executor.write_plan_result(result=result, result_dir=output_dir)
        write_root_outputs(output_dir=output_dir, config=config, result=result)
        write_log(output_dir=output_dir, message=result.message)

        return SUCCESS_EXIT_CODE if result.success else TASK_FAILED_EXIT_CODE
    except Exception as error:
        write_log(output_dir=output_dir, message=f'ERROR {type(error).__name__}: {error}')
        write_failure_outputs(output_dir=output_dir, message=f'{type(error).__name__}: {error}')

        return UNHANDLED_ERROR_EXIT_CODE


def parse_args(argv: list[str]) -> RemoteTaskAdapterArgs:
    values: dict[str, str | None] = {
        'config': None,
        'experiment_id': None,
        'output_dir': None,
    }
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == '--output-dir':
            values['output_dir'] = argv[index + 1]
            index += 2
            continue
        if arg == '--config':
            values['config'] = argv[index + 1]
            index += 2
            continue
        if arg == '--experiment-id':
            values['experiment_id'] = argv[index + 1]
            index += 2
            continue
        if arg.startswith('--'):
            index += 2 if index + 1 < len(argv) and not argv[index + 1].startswith('--') else 1
            continue
        index += 1

    if values['output_dir'] is None:
        raise ValueError('--output-dir is required')

    return RemoteTaskAdapterArgs(
        output_dir=Path(values['output_dir']),
        config_path=Path(values['config']) if values['config'] is not None else None,
        experiment_id=values['experiment_id'],
    )


def rebase_config_paths(config: Any, code_dir: Path) -> Any:
    rebased_variants = []
    for variant in config.data_variants:
        remote_path = variant.params.get('remote_path') if isinstance(variant.params, dict) else None
        if isinstance(remote_path, str) and remote_path.strip():
            rebased_path = remote_path.strip()
        else:
            path = rebase_path(raw_path=variant.path, code_dir=code_dir)
            rebased_path = str(path) if path is not None else None
        rebased_variants.append(replace(variant, path=rebased_path))

    rebased_tasks = []
    for task in config.tasks:
        params = dict(task.params)
        checkpoint = params.get('checkpoint')
        if isinstance(checkpoint, str) and checkpoint.strip():
            checkpoint_path = rebase_path(raw_path=checkpoint, code_dir=code_dir)
            params['checkpoint'] = str(checkpoint_path) if checkpoint_path is not None else checkpoint
        rebased_tasks.append(replace(task, params=params))

    return replace(config, data_variants=rebased_variants, tasks=rebased_tasks)


def rebase_path(raw_path: str | None, code_dir: Path) -> Path | PurePosixPath | None:
    if raw_path is None:
        return None
    normalized = str(raw_path).replace('\\', '/')
    if normalized.startswith('/'):
        return PurePosixPath(normalized)
    for marker, target_root in [
        ('local_remote_simulator/', code_dir / 'local_remote_simulator'),
        ('runs/user_datasets/', code_dir / 'runs' / 'user_datasets'),
    ]:
        if marker in normalized:
            relative = normalized.split(marker, maxsplit=1)[1]
            return target_root / Path(relative)

    path = Path(raw_path)
    if not path.is_absolute():
        return code_dir / path

    return path


def write_root_outputs(output_dir: Path, config: Any, result: object) -> None:
    primary_metric = str(config.analysis.primary_metric)
    representative_row = representative_metric_row_from_task_metrics(result=result, metric_name=primary_metric)
    best_metric_value = metric_value_from_row(row=representative_row, metric_name=primary_metric)
    status = 'finished' if bool(result.success) else 'failed'
    rows = [root_metric_row(row=representative_row, metric_name=primary_metric, metric_value=best_metric_value)]
    write_metrics(output_dir=output_dir, rows=rows, primary_metric=primary_metric, best_metric_value=best_metric_value)
    write_status(output_dir=output_dir, status=status)
    write_experiment_json(
        output_dir=output_dir,
        status=status,
        config=config,
        primary_metric=primary_metric,
        best_metric_value=best_metric_value,
        message=str(result.message),
    )
    write_artifacts_json(output_dir=output_dir)
    write_summary(
        output_dir=output_dir,
        status=status,
        message=str(result.message),
        primary_metric=primary_metric,
        best_metric_value=best_metric_value,
        metric_row=rows[0],
    )
    write_timings(output_dir=output_dir)


def write_failure_outputs(output_dir: Path, message: str) -> None:
    rows = [{
        'epoch': 1,
        'train_loss': '',
        'val_loss': '',
        'accuracy': '',
        'map50': '',
        'map50_95': '',
        'lr': '',
    }]
    write_metrics(output_dir=output_dir, rows=rows, primary_metric='accuracy', best_metric_value=None)
    write_status(output_dir=output_dir, status='failed')
    write_experiment_json(
        output_dir=output_dir,
        status='failed',
        config=None,
        primary_metric='accuracy',
        best_metric_value=None,
        message=message,
    )
    write_artifacts_json(output_dir=output_dir)
    write_summary(
        output_dir=output_dir,
        status='failed',
        message=message,
        primary_metric='accuracy',
        best_metric_value=None,
        metric_row=None,
    )
    write_timings(output_dir=output_dir)


def representative_metric_from_task_metrics(result: object, metric_name: str) -> float | None:
    return metric_value_from_row(
        row=representative_metric_row_from_task_metrics(result=result, metric_name=metric_name),
        metric_name=metric_name,
    )


def representative_metric_row_from_task_metrics(result: object, metric_name: str) -> dict[str, object] | None:
    final_row = final_metric_row_from_task_metrics(result=result, metric_name=metric_name)
    if final_row is not None:
        return final_row

    return best_metric_row_from_task_metrics(result=result, metric_name=metric_name)


def final_metric_from_task_metrics(result: object, metric_name: str) -> float | None:
    return metric_value_from_row(
        row=final_metric_row_from_task_metrics(result=result, metric_name=metric_name),
        metric_name=metric_name,
    )


def final_metric_row_from_task_metrics(result: object, metric_name: str) -> dict[str, object] | None:
    for task_result in reversed(list(getattr(result, 'task_results', []))):
        row = final_metric_row_from_task_result(task_result=task_result, metric_name=metric_name)
        if row is not None:
            return row

    return None


def final_metric_from_task_result(task_result: object, metric_name: str) -> float | None:
    return metric_value_from_row(
        row=final_metric_row_from_task_result(task_result=task_result, metric_name=metric_name),
        metric_name=metric_name,
    )


def final_metric_row_from_task_result(task_result: object, metric_name: str) -> dict[str, object] | None:
    metrics = getattr(task_result, 'metrics', None)
    if isinstance(metrics, list):
        row = final_metric_row_from_rows(rows=metrics, metric_name=metric_name)
        if row is not None:
            return normalize_metric_row_for_task(task_result=task_result, row=row)

    record = getattr(task_result, 'record', None)
    result_dir = getattr(record, 'result_dir', None)
    if result_dir is None:
        return None

    metrics_path = Path(result_dir) / 'metrics.csv'
    if not metrics_path.exists():
        return None

    with metrics_path.open(mode='r', encoding='utf-8', newline='') as file:
        row = final_metric_row_from_rows(rows=list(csv.DictReader(file)), metric_name=metric_name)
    if row is None:
        return None
    return normalize_metric_row_for_task(task_result=task_result, row=row)


def final_metric_from_rows(rows: list[object], metric_name: str) -> float | None:
    return metric_value_from_row(
        row=final_metric_row_from_rows(rows=rows, metric_name=metric_name),
        metric_name=metric_name,
    )


def final_metric_row_from_rows(rows: list[object], metric_name: str) -> dict[str, object] | None:
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        if metric_value_from_row(row=row, metric_name=metric_name) is not None:
            return dict(row)

    return None


def best_metric_from_task_metrics(result: object, metric_name: str) -> float | None:
    return metric_value_from_row(
        row=best_metric_row_from_task_metrics(result=result, metric_name=metric_name),
        metric_name=metric_name,
    )


def best_metric_row_from_task_metrics(result: object, metric_name: str) -> dict[str, object] | None:
    values: list[float] = []
    rows_by_value: dict[float, dict[str, object]] = {}
    for task_result in getattr(result, 'task_results', []):
        metrics_path = Path(task_result.record.result_dir) / 'metrics.csv'
        if not metrics_path.exists():
            continue
        with metrics_path.open(mode='r', encoding='utf-8', newline='') as file:
            for row in csv.DictReader(file):
                value = metric_value_from_row(row=row, metric_name=metric_name)
                if value is not None:
                    values.append(value)
                    rows_by_value[value] = normalize_metric_row_for_task(task_result=task_result, row=dict(row))

    return rows_by_value[max(values)] if values else None


def metric_value_from_row(row: dict[str, object] | None, metric_name: str) -> float | None:
    if row is None:
        return None
    value = row.get(metric_name)
    if value in {None, ''}:
        return None

    return float(value)


def root_metric_row(
    *,
    row: dict[str, object] | None,
    metric_name: str,
    metric_value: float | None,
) -> dict[str, object]:
    root_row = {
        field_name: row.get(field_name, '') if row is not None else ''
        for field_name in TASK_METRICS_FIELDNAMES
    }
    root_row['epoch'] = root_row.get('epoch') or ''
    if metric_value is not None and root_row.get(metric_name) in {None, ''}:
        root_row[metric_name] = metric_value

    return root_row


def normalize_metric_row_for_task(*, task_result: object, row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    if is_inference_task_result(task_result=task_result):
        normalized['epoch'] = ''
    return normalized


def is_inference_task_result(*, task_result: object) -> bool:
    record = getattr(task_result, 'record', None)
    task_id = str(getattr(record, 'task_id', '') or '').lower()
    if task_id.startswith(('predict_', 'infer_', 'eval_', 'test_')):
        return True
    params = getattr(record, 'params', None)
    if isinstance(params, dict):
        execution_mode = str(params.get('execution_mode', '') or '').lower()
        return execution_mode in {'inference', 'inference_smoke', 'predict', 'eval', 'test'}
    return False


def metric_epoch(row: dict[str, object] | None) -> int | None:
    if row is None:
        return None
    value = row.get('epoch')
    if value in {None, ''}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def write_metrics(
    output_dir: Path,
    rows: list[dict[str, object]],
    primary_metric: str,
    best_metric_value: float | None,
) -> None:
    with (output_dir / 'metrics.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=TASK_METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: row.get(field_name, '') for field_name in TASK_METRICS_FIELDNAMES})
    (output_dir / 'metrics.json').write_text(
        data=json.dumps(
            {
                'schema_version': '0.1',
                'primary_metric': primary_metric,
                'higher_is_better': True,
                'best': {
                    'metric_name': primary_metric,
                    'metric_value': best_metric_value,
                    'epoch': metric_epoch(rows[0] if rows else None),
                },
                'records': rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


def write_status(output_dir: Path, status: str) -> None:
    (output_dir / 'status.marker').write_text(data=f'{status}\n', encoding='utf-8')


def write_experiment_json(
    output_dir: Path,
    status: str,
    config: Any | None,
    primary_metric: str,
    best_metric_value: float | None,
    message: str,
) -> None:
    (output_dir / 'experiment.json').write_text(
        data=json.dumps(
            {
                'schema_version': '0.1',
                'status': status,
                'model': config.experiment.name if config is not None else '',
                'dataset': '',
                'epochs': 1,
                'learning_rate': '',
                'best_metric_name': primary_metric,
                'best_metric_value': best_metric_value,
                'message': message,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


def write_artifacts_json(output_dir: Path) -> None:
    (output_dir / 'artifacts.json').write_text(
        data=json.dumps(
            {
                'schema_version': '0.1',
                'artifacts': [
                    {'name': 'train_log', 'path': 'train.log', 'kind': 'log', 'required': True},
                    {'name': 'metrics_json', 'path': 'metrics.json', 'kind': 'metrics', 'required': True},
                    {'name': 'metrics', 'path': 'metrics.csv', 'kind': 'metrics', 'required': True},
                    {'name': 'task_results', 'path': 'task_results.json', 'kind': 'task_results', 'required': True},
                    {'name': 'summary', 'path': 'summary.md', 'kind': 'summary', 'required': True},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


def write_summary(
    output_dir: Path,
    status: str,
    message: str,
    primary_metric: str,
    best_metric_value: float | None,
    metric_row: dict[str, object] | None,
) -> None:
    best_text = 'n/a' if best_metric_value is None else f'{best_metric_value:.6f}'
    metric_lines = latest_metric_lines(row=metric_row)
    (output_dir / 'summary.md').write_text(
        data='\n'.join([
            '# Remote Task Adapter Summary',
            '',
            f'- status: {status}',
            f'- best_{primary_metric}: {best_text}',
            f'- message: {message}',
            '',
            '## Latest Metrics',
            *metric_lines,
            '',
        ]),
        encoding='utf-8',
    )


def latest_metric_lines(row: dict[str, object] | None) -> list[str]:
    if row is None:
        return ['- none']
    lines: list[str] = []
    is_classification_metric = any(
        _has_metric_value(row.get(field_name))
        for field_name in ('macro_precision', 'macro_recall', 'macro_f1', 'class_recall')
    )
    for field_name in [
        'accuracy',
        'precision',
        'recall',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'class_recall',
        'map50',
        'map50_95',
        'num_predictions',
        'num_gt',
        'label_error_rate',
        'latency_ms_per_image',
        'p95_latency_ms',
    ]:
        if is_classification_metric and field_name in {'precision', 'recall'}:
            continue
        value = row.get(field_name)
        if _has_metric_value(value):
            label = 'min_class_recall' if field_name == 'class_recall' else field_name
            lines.append(f'- {label}: {value}')

    return lines or ['- none']


def _has_metric_value(value: object) -> bool:
    return value not in {None, ''}


def write_timings(output_dir: Path) -> None:
    with (output_dir / 'timings.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['stage', 'elapsed_seconds'])
        writer.writeheader()
        writer.writerow({'stage': 'remote_task_adapter', 'elapsed_seconds': '0.000000'})


def write_log(output_dir: Path, message: str) -> None:
    with (output_dir / 'train.log').open(mode='a', encoding='utf-8') as file:
        file.write(message + '\n')
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # The SSH/log monitor can close stdout after result files are written.
        # Treat that as a transport/logging issue, not as a failed experiment.
        return


if __name__ == '__main__':
    raise SystemExit(main())
