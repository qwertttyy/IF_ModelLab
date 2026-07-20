import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ironflow_exp.engine.configs import EngineConfigLoader
from ironflow_exp.engine.domain import ExperimentRecord, ServerRecord
from ironflow_exp.engine.importers import ExportDatasetImporter
from ironflow_exp.engine.runners import ExperimentRunnerResult, LocalExperimentRunner, SshExperimentRunner
from ironflow_exp.engine.scheduler import LocalSchedulerService, SchedulerRunResult
from ironflow_exp.engine.server import (
    RemoteDependencyInstaller,
    ServerChecker,
    ServerCheckResult,
    ServerPreflightRunner,
    ServerProfileLoader,
    ServerProfileValidator,
)
from ironflow_exp.engine.server.metadata_redaction import server_record_public_dict
from ironflow_exp.engine.storage import SQLiteExperimentStorage
from ironflow_exp.engine.time_display import add_kst_display_fields


DEFAULT_DB_PATH = Path('runs') / 'ironflow_experiments.sqlite3'
DEFAULT_SSH_STAGE_TIMEOUT_SECONDS = {
    'bootstrap': 120,
    'upload': 1800,
    'run': None,
    'submit': 120,
    'download': 1800,
    'collect': 1800,
}


def configure_cli_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (OSError, ValueError):
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='ironflow-engine')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Run one local-first engine experiment')
    run_parser.add_argument('--config', required=True, help='Engine config YAML or JSON path')
    run_parser.add_argument('--experiment-id', default=None, help='Optional explicit experiment id')
    run_parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Intentionally reuse an existing experiment id and overwrite the saved experiment record',
    )
    run_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    status_parser = subparsers.add_parser('status', help='Show experiment status')
    status_parser.add_argument('experiment_id')
    status_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    status_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    logs_parser = subparsers.add_parser('logs', help='Show experiment logs')
    logs_parser.add_argument('experiment_id')
    logs_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    logs_parser.add_argument('--tail', type=int, default=None, help='Show only the last N lines')

    collect_parser = subparsers.add_parser('collect', help='Collect experiment metrics and artifacts')
    collect_parser.add_argument('experiment_id')
    collect_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    collect_parser.add_argument('--output-dir', default=None, help='Optional collected output directory')
    collect_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    list_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    compare_parser = subparsers.add_parser('compare', help='Compare experiment best metrics')
    compare_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    compare_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')
    compare_parser.add_argument('--csv', action='store_true', help='Print CSV table')

    dataset_parser = subparsers.add_parser('dataset', help='Import and inspect datasets')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command', required=True)

    dataset_import_parser = dataset_subparsers.add_parser('import-export', help='Import an ImageDataCollector export folder')
    dataset_import_parser.add_argument('--source', required=True, help='Export run folder, or a parent folder containing export_* folders')
    dataset_import_parser.add_argument('--output', required=True, help='Output IronFlow dataset folder')
    dataset_import_parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Overwrite output folder if it already exists',
    )
    dataset_import_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    schedule_parser = subparsers.add_parser('schedule', help='Register a scheduled experiment')
    schedule_parser.add_argument('--config', required=True, help='Engine config YAML or JSON path')
    schedule_parser.add_argument('--at', required=True, help='Scheduled timestamp, ISO format recommended')
    schedule_parser.add_argument('--schedule-id', default=None, help='Optional explicit schedule id')
    schedule_parser.add_argument('--db', default=None, help='SQLite DB path. Defaults to config repository.sqlite_path')
    schedule_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    scheduler_parser = subparsers.add_parser('scheduler', help='Run due scheduled experiments')
    scheduler_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    scheduler_parser.add_argument('--once', action='store_true', help='Run due schedules once and exit')
    scheduler_parser.add_argument('--now', default=None, help='Override current timestamp for testing or manual runs')
    scheduler_parser.add_argument('--limit', type=int, default=None, help='Maximum due schedules to execute')
    scheduler_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    server_parser = subparsers.add_parser('server', help='Manage local and remote server profiles')
    server_subparsers = server_parser.add_subparsers(dest='server_command', required=True)

    server_add_parser = server_subparsers.add_parser('add', help='Register a server profile')
    server_add_parser.add_argument('--config', required=True, help='Server profile YAML or JSON path')
    server_add_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    server_add_parser.add_argument('--check-paths', action='store_true', help='Validate local key_path existence')
    server_add_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    server_check_parser = server_subparsers.add_parser('check', help='Check a registered server profile')
    server_check_parser.add_argument('name')
    server_check_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    server_check_parser.add_argument('--live-ssh', action='store_true', help='Run a live SSH command for SSH-like profiles')
    server_check_parser.add_argument('--gpu-probe', action='store_true', help='Run nvidia-smi on the remote server')
    server_check_parser.add_argument(
        '--remote-task-adapter-deps',
        action='store_true',
        help='Probe remote Python dependencies needed by the remote task-adapter entrypoint',
    )
    server_check_parser.add_argument(
        '--dependency-profile',
        default='all',
        choices=ServerChecker.dependency_profile_names(),
        help='Remote task-adapter dependency profile to probe',
    )
    server_check_parser.add_argument('--timeout', type=int, default=10, help='Remote check timeout in seconds')
    server_check_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    server_preflight_parser = server_subparsers.add_parser('preflight', help='Run live SSH, GPU, and dependency preflights in sequence')
    server_preflight_parser.add_argument('name')
    server_preflight_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    server_preflight_parser.add_argument(
        '--dependency-profile',
        default='all',
        choices=ServerChecker.dependency_profile_names(),
        help='Remote task-adapter dependency profile to probe',
    )
    server_preflight_parser.add_argument('--live-timeout', type=int, default=10, help='Live SSH preflight timeout in seconds')
    server_preflight_parser.add_argument('--gpu-timeout', type=int, default=10, help='GPU probe timeout in seconds')
    server_preflight_parser.add_argument('--deps-timeout', type=int, default=90, help='Dependency probe timeout in seconds')
    server_preflight_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    server_install_parser = server_subparsers.add_parser('install-deps', help='Plan or execute remote Python dependency installation')
    server_install_parser.add_argument('name')
    server_install_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    server_install_parser.add_argument(
        '--dependency-profile',
        default='all',
        choices=ServerChecker.dependency_profile_names(),
        help='Remote dependency profile to install',
    )
    server_install_parser.add_argument(
        '--torch-index-url',
        default=None,
        help='Optional extra pip index URL for torch/torchvision CUDA wheels',
    )
    server_install_parser.add_argument('--no-upgrade', action='store_true', help='Do not pass --upgrade to the package install command')
    server_install_parser.add_argument('--timeout', type=int, default=900, help='Remote install timeout in seconds')
    server_install_parser.add_argument('--execute', action='store_true', help='Actually run the install command over SSH')
    server_install_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    server_list_parser = server_subparsers.add_parser('list', help='List registered server profiles')
    server_list_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
    server_list_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    ssh_parser = subparsers.add_parser('ssh', help='Run staged SSH experiment operations')
    ssh_subparsers = ssh_parser.add_subparsers(dest='ssh_command', required=True)

    ssh_prepare_parser = ssh_subparsers.add_parser('prepare', help='Prepare an SSH experiment plan')
    ssh_prepare_parser.add_argument('--config', required=True, help='Engine config YAML or JSON path')
    ssh_prepare_parser.add_argument('--server', required=True, help='Registered server profile name')
    ssh_prepare_parser.add_argument('--experiment-id', default=None, help='Optional explicit experiment id')
    ssh_prepare_parser.add_argument('--db', default=None, help='SQLite DB path. Defaults to config repository.sqlite_path')
    ssh_prepare_parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Intentionally reuse an existing experiment id and overwrite the saved SSH experiment record',
    )
    ssh_prepare_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    ssh_execute_parser = ssh_subparsers.add_parser('execute', help='Run the full SSH lifecycle in one command')
    ssh_execute_parser.add_argument('--config', required=True, help='Engine config YAML or JSON path')
    ssh_execute_parser.add_argument('--server', required=True, help='Registered server profile name')
    ssh_execute_parser.add_argument('--experiment-id', default=None, help='Optional explicit experiment id')
    ssh_execute_parser.add_argument('--db', default=None, help='SQLite DB path. Defaults to config repository.sqlite_path')
    ssh_execute_parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Intentionally reuse an existing experiment id and overwrite the saved SSH experiment record',
    )
    ssh_execute_parser.add_argument('--timeout', type=int, default=None, help='Override SSH stage timeout in seconds')
    ssh_execute_parser.add_argument('--output-dir', default=None, help='Optional collected output directory')
    ssh_execute_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    for command_name in ['bootstrap', 'upload', 'run', 'submit', 'download', 'collect', 'cancel', 'status', 'logs']:
        ssh_stage_parser = ssh_subparsers.add_parser(command_name, help=f'SSH {command_name} stage')
        ssh_stage_parser.add_argument('experiment_id')
        ssh_stage_parser.add_argument('--server', default=None, help='Registered server profile name')
        ssh_stage_parser.add_argument(
            '--force-server',
            action='store_true',
            help='Allow using a server different from the one saved in the prepared SSH plan',
        )
        ssh_stage_parser.add_argument('--db', default=str(DEFAULT_DB_PATH), help='SQLite DB path')
        if command_name in {'bootstrap', 'upload', 'run', 'submit', 'download', 'collect', 'cancel'}:
            ssh_stage_parser.add_argument('--timeout', type=int, default=None, help='Override SSH stage timeout in seconds')
        if command_name == 'collect':
            ssh_stage_parser.add_argument('--output-dir', default=None, help='Optional collected output directory')
        if command_name == 'logs':
            ssh_stage_parser.add_argument('--tail', type=int, default=None, help='Show only the last N lines')
        if command_name != 'logs':
            ssh_stage_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON summary')

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_cli_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == 'run':
            return run_command(args=args)
        if args.command == 'status':
            return status_command(args=args)
        if args.command == 'logs':
            return logs_command(args=args)
        if args.command == 'collect':
            return collect_command(args=args)
        if args.command == 'list':
            return list_command(args=args)
        if args.command == 'compare':
            return compare_command(args=args)
        if args.command == 'dataset':
            return dataset_command(args=args)
        if args.command == 'schedule':
            return schedule_command(args=args)
        if args.command == 'scheduler':
            return scheduler_command(args=args)
        if args.command == 'server':
            return server_command(args=args)
        if args.command == 'ssh':
            return ssh_command(args=args)
    except Exception as error:
        print(f'error: {type(error).__name__}: {error}', file=sys.stderr)
        return 1

    parser.error(f'unsupported command: {args.command}')

    return 2


def run_command(args: argparse.Namespace) -> int:
    config_path = Path(str(args.config))
    config = EngineConfigLoader().load_file(path=config_path)
    storage = SQLiteExperimentStorage(db_path=config.repository.sqlite_path)
    runner = LocalExperimentRunner(storage=storage)
    experiment_id = args.experiment_id or _make_experiment_id(name=config.experiment.name)
    prepare_result = runner.prepare(
        config=config,
        experiment_id=experiment_id,
        replace_existing=bool(args.replace_existing),
    )
    run_result = runner.run(experiment_id=experiment_id)
    record = storage.get_experiment(experiment_id=experiment_id)
    summary = {
        'success': run_result.is_success,
        'prepare': _runner_result_to_dict(result=prepare_result),
        'run': _runner_result_to_dict(result=run_result),
        'experiment': _record_to_dict(record=record) if record is not None else None,
        'db_path': str(config.repository.sqlite_path),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"status: {run_result.status.value}",
        f"experiment_id: {experiment_id}",
        f"db_path: {config.repository.sqlite_path}",
        f"result_dir: {run_result.metadata.get('result_dir')}",
    ])

    return 0 if run_result.is_success else 1


def status_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    runner = LocalExperimentRunner(storage=storage)
    result = runner.status(experiment_id=str(args.experiment_id))
    record = storage.get_experiment(experiment_id=str(args.experiment_id))
    summary = {
        'success': True,
        'status': _runner_result_to_dict(result=result),
        'experiment': _record_to_dict(record=record) if record is not None else None,
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"status: {result.status.value}",
        f"experiment_id: {result.experiment_id}",
        f"result_dir: {result.metadata.get('result_dir')}",
    ])

    return 0


def logs_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    runner = LocalExperimentRunner(storage=storage)
    print(runner.logs(experiment_id=str(args.experiment_id), tail=args.tail))

    return 0


def collect_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    runner = LocalExperimentRunner(storage=storage)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    result = runner.collect(experiment_id=str(args.experiment_id), output_dir=output_dir)
    record = storage.get_experiment(experiment_id=str(args.experiment_id))
    summary = {
        'success': result.is_success,
        'collect': _runner_result_to_dict(result=result),
        'experiment': _record_to_dict(record=record) if record is not None else None,
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"status: {result.status.value}",
        f"experiment_id: {result.experiment_id}",
        f"result_dir: {result.metadata.get('result_dir')}",
        f"collected_dir: {result.metadata.get('collected_dir')}",
        f"metrics_count: {result.metadata.get('metrics_count')}",
        f"best_metric: {result.metadata.get('best_metric_name')}={result.metadata.get('best_metric_value')}",
    ])

    return 0 if result.is_success else 1


def list_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    records = storage.list_experiments()
    rows = [_record_to_dict(record=record) for record in records]
    summary = {
        'success': True,
        'experiments': rows,
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=_records_table(records=records))

    return 0


def compare_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    records = storage.list_experiments()
    rows = [
        {
            'experiment_id': record.experiment_id,
            'name': record.name,
            'status': record.status.value,
            'best_metric_name': record.best_metric_name,
            'best_metric_value': record.best_metric_value,
            'result_path': record.result_path,
        }
        for record in records
    ]
    summary = {
        'success': True,
        'experiments': rows,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.csv:
        _print_compare_csv(rows=rows)
    else:
        print('\n'.join(_compare_table(rows=rows)))

    return 0


def dataset_command(args: argparse.Namespace) -> int:
    if args.dataset_command == 'import-export':
        return dataset_import_export_command(args=args)

    raise ValueError(f'unsupported dataset command: {args.dataset_command}')


def dataset_import_export_command(args: argparse.Namespace) -> int:
    result = ExportDatasetImporter().import_export(
        source_root=Path(str(args.source)),
        output_root=Path(str(args.output)),
        replace_existing=bool(args.replace_existing),
    )
    summary = {
        'success': True,
        'import': result.to_dict(),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        'status: imported',
        f'source_root: {result.source_root}',
        f'output_root: {result.output_root}',
        f'detection_manifest: {result.detection_manifest_path}',
        f'classification_manifest: {result.classification_manifest_path}',
        f'detection_images: {result.detection_image_count}',
        f'detection_objects: {result.detection_object_count}',
        f'classification_images: {result.classification_image_count}',
        f'warnings: {len(result.warnings)}',
    ])

    return 0


def schedule_command(args: argparse.Namespace) -> int:
    config_path = Path(str(args.config))
    config = EngineConfigLoader().load_file(path=config_path)
    db_path = args.db or config.repository.sqlite_path
    storage = SQLiteExperimentStorage(db_path=db_path)
    service = LocalSchedulerService(storage=storage)
    schedule_id = args.schedule_id or _make_schedule_id(config_name=config.experiment.name)
    record = service.add_schedule(
        config_path=str(config_path),
        scheduled_at=str(args.at),
        schedule_id=schedule_id,
    )
    summary = {
        'success': True,
        'schedule': asdict(record),
        'db_path': str(db_path),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"schedule_id: {record.schedule_id}",
        f"status: {record.status}",
        f"scheduled_at: {record.scheduled_at}",
        f"db_path: {db_path}",
    ])

    return 0


def scheduler_command(args: argparse.Namespace) -> int:
    if not args.once:
        raise ValueError('scheduler currently supports --once only')

    storage = SQLiteExperimentStorage(db_path=args.db)
    now = _parse_cli_datetime(value=args.now) if args.now is not None else None
    service = LocalSchedulerService(storage=storage)
    results = service.run_due_once(now=now, limit=args.limit)
    summary = {
        'success': True,
        'results': [
            _scheduler_result_to_dict(result=result)
            for result in results
        ],
        'db_path': str(args.db),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=_scheduler_table(results=results))

    return 0


def server_command(args: argparse.Namespace) -> int:
    if args.server_command == 'add':
        return server_add_command(args=args)
    if args.server_command == 'check':
        return server_check_command(args=args)
    if args.server_command == 'preflight':
        return server_preflight_command(args=args)
    if args.server_command == 'install-deps':
        return server_install_deps_command(args=args)
    if args.server_command == 'list':
        return server_list_command(args=args)

    raise ValueError(f'unsupported server command: {args.server_command}')


def server_add_command(args: argparse.Namespace) -> int:
    record = ServerProfileLoader().load_file(path=args.config)
    validation = ServerProfileValidator().validate(record=record, check_paths=bool(args.check_paths))
    if not validation.is_valid:
        summary = {
            'success': False,
            'server': _server_record_to_dict(record=record),
            'errors': [
                asdict(issue)
                for issue in validation.errors
            ],
            'warnings': [
                asdict(issue)
                for issue in validation.warnings
            ],
        }
        _print_summary(
            summary=summary,
            as_json=bool(args.json),
            human_lines=[
                'status: failed',
                f'server: {record.name}',
                *[
                    f"error: {issue['field']}: {issue['message']}"
                    for issue in summary['errors']
                ],
            ],
        )

        return 1

    storage = SQLiteExperimentStorage(db_path=args.db)
    storage.save_server(record=record)
    summary = {
        'success': True,
        'server': _server_record_to_dict(record=record),
        'warnings': [
            asdict(issue)
            for issue in validation.warnings
        ],
        'db_path': str(args.db),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        'status: saved',
        f'server: {record.name}',
        f'type: {record.server_type}',
        f'db_path: {args.db}',
    ])

    return 0


def server_check_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    record = storage.get_server(name=str(args.name))
    if record is None:
        raise ValueError(f'unknown server: {args.name}')

    check_result = ServerChecker().check(
        record=record,
        live_ssh=bool(args.live_ssh),
        gpu_probe=bool(args.gpu_probe),
        remote_task_adapter_deps=bool(args.remote_task_adapter_deps),
        dependency_profile=str(args.dependency_profile),
        timeout_seconds=int(args.timeout),
    )
    checked_record = replace(
        record,
        last_checked_at=datetime.now(tz=timezone.utc).isoformat(),
        last_status=check_result.status,
    )
    storage.save_server(record=checked_record)
    summary = {
        'success': check_result.is_success,
        'check': _server_check_to_dict(result=check_result),
        'server': _server_record_to_dict(record=checked_record),
        'db_path': str(args.db),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f'status: {check_result.status}',
        f'server: {check_result.name}',
        f'message: {check_result.message}',
    ])

    return 0 if check_result.is_success else 1


def server_preflight_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    record = storage.get_server(name=str(args.name))
    if record is None:
        raise ValueError(f'unknown server: {args.name}')

    preflight_result = ServerPreflightRunner().run(
        record=record,
        dependency_profile=str(args.dependency_profile),
        live_timeout_seconds=int(args.live_timeout),
        gpu_timeout_seconds=int(args.gpu_timeout),
        dependency_timeout_seconds=int(args.deps_timeout),
    )
    checked_record = replace(
        record,
        last_checked_at=datetime.now(tz=timezone.utc).isoformat(),
        last_status=preflight_result.status,
    )
    storage.save_server(record=checked_record)
    summary = {
        'success': preflight_result.is_success,
        'preflight': preflight_result.to_dict(),
        'server': _server_record_to_dict(record=checked_record),
        'db_path': str(args.db),
    }
    stage_lines = [
        f"{stage.stage}: {stage.result.status}"
        for stage in preflight_result.stages
    ]
    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f'status: {preflight_result.status}',
        f'server: {preflight_result.name}',
        f'message: {preflight_result.message}',
        *stage_lines,
    ])

    return 0 if preflight_result.is_success else 1


def server_install_deps_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    record = storage.get_server(name=str(args.name))
    if record is None:
        raise ValueError(f'unknown server: {args.name}')

    installer = RemoteDependencyInstaller()
    upgrade = not bool(args.no_upgrade)
    if bool(args.execute):
        install_result = installer.install(
            server=record,
            dependency_profile=str(args.dependency_profile),
            torch_index_url=args.torch_index_url,
            upgrade=upgrade,
            timeout_seconds=int(args.timeout),
        )
        checked_record = replace(
            record,
            last_checked_at=datetime.now(tz=timezone.utc).isoformat(),
            last_status=install_result.status,
        )
        storage.save_server(record=checked_record)
        summary = {
            'success': install_result.success,
            'executed': True,
            'install': install_result.to_dict(),
            'server': _server_record_to_dict(record=checked_record),
            'db_path': str(args.db),
        }
        _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
            f'status: {install_result.status}',
            f'server: {record.name}',
            f'message: {install_result.message}',
        ])

        return 0 if install_result.success else 1

    plan = installer.plan(
        dependency_profile=str(args.dependency_profile),
        torch_index_url=args.torch_index_url,
        upgrade=upgrade,
    )
    summary = {
        'success': True,
        'executed': False,
        'install_plan': plan.to_dict(),
        'server': _server_record_to_dict(record=record),
        'db_path': str(args.db),
        'next_action': 'rerun with --execute to install dependencies over SSH',
    }
    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        'status: planned',
        f'server: {record.name}',
        f'dependency_profile: {plan.dependency_profile}',
        f'packages: {", ".join(plan.packages)}',
        f'command: {plan.command}',
        'next_action: rerun with --execute to install dependencies over SSH',
    ])

    return 0


def server_list_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    records = storage.list_servers()
    summary = {
        'success': True,
        'servers': [
            _server_record_to_dict(record=record)
            for record in records
        ],
        'db_path': str(args.db),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=_server_table(records=records))

    return 0


def ssh_command(args: argparse.Namespace) -> int:
    if args.ssh_command == 'prepare':
        return ssh_prepare_command(args=args)
    if args.ssh_command == 'execute':
        return ssh_execute_command(args=args)
    if args.ssh_command == 'bootstrap':
        return ssh_stage_command(args=args, stage='bootstrap')
    if args.ssh_command == 'upload':
        return ssh_stage_command(args=args, stage='upload')
    if args.ssh_command == 'run':
        return ssh_stage_command(args=args, stage='run')
    if args.ssh_command == 'submit':
        return ssh_stage_command(args=args, stage='submit')
    if args.ssh_command == 'download':
        return ssh_stage_command(args=args, stage='download')
    if args.ssh_command == 'collect':
        return ssh_stage_command(args=args, stage='collect')
    if args.ssh_command == 'cancel':
        return ssh_stage_command(args=args, stage='cancel')
    if args.ssh_command == 'status':
        return ssh_stage_command(args=args, stage='status')
    if args.ssh_command == 'logs':
        return ssh_logs_command(args=args)

    raise ValueError(f'unsupported ssh command: {args.ssh_command}')


def ssh_prepare_command(args: argparse.Namespace) -> int:
    config_path = Path(str(args.config))
    config = EngineConfigLoader().load_file(path=config_path)
    db_path = args.db or config.repository.sqlite_path
    storage = SQLiteExperimentStorage(db_path=db_path)
    server = _require_server(storage=storage, explicit_name=str(args.server))
    runner = SshExperimentRunner(storage=storage, server=server)
    experiment_id = args.experiment_id or _make_experiment_id(name=config.experiment.name)
    result = runner.prepare(
        config=config,
        experiment_id=experiment_id,
        replace_existing=bool(args.replace_existing),
    )
    record = storage.get_experiment(experiment_id=experiment_id)
    summary = {
        'success': record is not None and result.status.value != 'failed',
        'prepare': _runner_result_to_dict(result=result),
        'experiment': _record_to_dict(record=record) if record is not None else None,
        'server': _server_record_to_dict(record=server),
        'db_path': str(db_path),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"status: {result.status.value}",
        f"experiment_id: {experiment_id}",
        f"server: {server.name}",
        f"ssh_plan_path: {result.metadata.get('ssh_plan_path')}",
        f"db_path: {db_path}",
    ])

    return 0


def ssh_execute_command(args: argparse.Namespace) -> int:
    config_path = Path(str(args.config))
    config = EngineConfigLoader().load_file(path=config_path)
    db_path = args.db or config.repository.sqlite_path
    storage = SQLiteExperimentStorage(db_path=db_path)
    server = _require_server(storage=storage, explicit_name=str(args.server))
    runner = SshExperimentRunner(storage=storage, server=server)
    experiment_id = args.experiment_id or _make_experiment_id(name=config.experiment.name)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    prepare_result = runner.prepare(
        config=config,
        experiment_id=experiment_id,
        replace_existing=bool(args.replace_existing),
    )
    bootstrap_result = runner.bootstrap(
        experiment_id=experiment_id,
        timeout_seconds=_ssh_stage_timeout(args=args, stage='bootstrap'),
    )
    if _runner_stage_failed(result=bootstrap_result):
        return _print_ssh_execute_summary(
            args=args,
            db_path=db_path,
            storage=storage,
            server=server,
            prepare_result=prepare_result,
            bootstrap_result=bootstrap_result,
            upload_result=None,
            run_result=None,
            collect_result=None,
        )

    upload_result = runner.upload(
        experiment_id=experiment_id,
        timeout_seconds=_ssh_stage_timeout(args=args, stage='upload'),
    )
    if _runner_stage_failed(result=upload_result):
        return _print_ssh_execute_summary(
            args=args,
            db_path=db_path,
            storage=storage,
            server=server,
            prepare_result=prepare_result,
            bootstrap_result=bootstrap_result,
            upload_result=upload_result,
            run_result=None,
            collect_result=None,
        )

    run_result = runner.run(
        experiment_id=experiment_id,
        timeout_seconds=_ssh_stage_timeout(args=args, stage='run'),
    )
    if _runner_stage_failed(result=run_result):
        return _print_ssh_execute_summary(
            args=args,
            db_path=db_path,
            storage=storage,
            server=server,
            prepare_result=prepare_result,
            bootstrap_result=bootstrap_result,
            upload_result=upload_result,
            run_result=run_result,
            collect_result=None,
        )

    collect_result = runner.collect(
        experiment_id=experiment_id,
        output_dir=output_dir,
        download_timeout_seconds=_ssh_stage_timeout(args=args, stage='collect'),
    )

    return _print_ssh_execute_summary(
        args=args,
        db_path=db_path,
        storage=storage,
        server=server,
        prepare_result=prepare_result,
        bootstrap_result=bootstrap_result,
        upload_result=upload_result,
        run_result=run_result,
        collect_result=collect_result,
    )


def ssh_stage_command(args: argparse.Namespace, stage: str) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    server = _server_for_experiment(
        storage=storage,
        experiment_id=str(args.experiment_id),
        explicit_name=args.server,
    )
    runner = SshExperimentRunner(
        storage=storage,
        server=server,
        allow_server_mismatch=bool(getattr(args, 'force_server', False)),
    )
    timeout_seconds = _ssh_stage_timeout(args=args, stage=stage)

    if stage == 'bootstrap':
        result = runner.bootstrap(experiment_id=str(args.experiment_id), timeout_seconds=timeout_seconds)
    elif stage == 'upload':
        result = runner.upload(experiment_id=str(args.experiment_id), timeout_seconds=timeout_seconds)
    elif stage == 'run':
        result = runner.run(experiment_id=str(args.experiment_id), timeout_seconds=timeout_seconds)
    elif stage == 'submit':
        result = runner.submit(experiment_id=str(args.experiment_id), timeout_seconds=timeout_seconds)
    elif stage == 'download':
        result = runner.download(experiment_id=str(args.experiment_id), timeout_seconds=timeout_seconds)
    elif stage == 'collect':
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        result = runner.collect(
            experiment_id=str(args.experiment_id),
            output_dir=output_dir,
            download_timeout_seconds=timeout_seconds,
        )
    elif stage == 'cancel':
        result = runner.cancel(experiment_id=str(args.experiment_id), timeout_seconds=timeout_seconds)
    elif stage == 'status':
        result = runner.status(experiment_id=str(args.experiment_id))
    else:
        raise ValueError(f'unsupported ssh stage: {stage}')

    record = storage.get_experiment(experiment_id=str(args.experiment_id))
    success = result.status == ExperimentStatus.CANCELLED if stage == 'cancel' else result.status.value not in {'failed', 'cancelled'}
    summary = {
        'success': success,
        stage: _runner_result_to_dict(result=result),
        'experiment': _record_to_dict(record=record) if record is not None else None,
        'server': _server_record_to_dict(record=server),
        'db_path': str(args.db),
        'timeout_seconds': timeout_seconds,
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"stage: {stage}",
        f"status: {result.status.value}",
        f"experiment_id: {result.experiment_id}",
        f"server: {server.name}",
        f"message: {result.message}",
        f"result_dir: {result.metadata.get('result_dir')}",
        f"collected_dir: {result.metadata.get('collected_dir')}",
        f"next_action: {result.metadata.get('next_action')}",
    ])

    return 0 if summary['success'] else 1


def ssh_logs_command(args: argparse.Namespace) -> int:
    storage = SQLiteExperimentStorage(db_path=args.db)
    server = _server_for_experiment(
        storage=storage,
        experiment_id=str(args.experiment_id),
        explicit_name=args.server,
    )
    runner = SshExperimentRunner(
        storage=storage,
        server=server,
        allow_server_mismatch=bool(getattr(args, 'force_server', False)),
    )
    print(runner.logs(experiment_id=str(args.experiment_id), tail=args.tail))

    return 0


def _print_ssh_execute_summary(
    args: argparse.Namespace,
    db_path: str | Path,
    storage: SQLiteExperimentStorage,
    server: ServerRecord,
    prepare_result: ExperimentRunnerResult,
    bootstrap_result: ExperimentRunnerResult | None,
    upload_result: ExperimentRunnerResult | None,
    run_result: ExperimentRunnerResult | None,
    collect_result: ExperimentRunnerResult | None,
) -> int:
    experiment_id = prepare_result.experiment_id
    record = storage.get_experiment(experiment_id=experiment_id)
    stages = {
        'prepare': _runner_result_to_dict(result=prepare_result),
        'bootstrap': _optional_runner_result_to_dict(result=bootstrap_result),
        'upload': _optional_runner_result_to_dict(result=upload_result),
        'run': _optional_runner_result_to_dict(result=run_result),
        'collect': _optional_runner_result_to_dict(result=collect_result),
    }
    final_result = collect_result or run_result or upload_result or bootstrap_result or prepare_result
    summary = {
        'success': final_result.status.value == 'collected',
        'stages': stages,
        'experiment': _record_to_dict(record=record) if record is not None else None,
        'server': _server_record_to_dict(record=server),
        'db_path': str(db_path),
    }

    _print_summary(summary=summary, as_json=bool(args.json), human_lines=[
        f"status: {final_result.status.value}",
        f"experiment_id: {experiment_id}",
        f"server: {server.name}",
        f"message: {final_result.message}",
        f"db_path: {db_path}",
    ])

    return 0 if summary['success'] else 1


def _runner_stage_failed(result: ExperimentRunnerResult) -> bool:
    return result.status.value in {'failed', 'cancelled'}


def _ssh_stage_timeout(args: argparse.Namespace, stage: str) -> int | None:
    explicit_timeout = getattr(args, 'timeout', None)
    if explicit_timeout is not None:
        return int(explicit_timeout)

    return DEFAULT_SSH_STAGE_TIMEOUT_SECONDS.get(stage)


def _print_summary(summary: dict[str, Any], as_json: bool, human_lines: list[str]) -> None:
    if as_json:
        # Keep CLI JSON ASCII-only so Windows cp949 consoles do not crash when
        # remote logs contain tqdm/block progress characters.
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print('\n'.join(human_lines))


def _runner_result_to_dict(result: ExperimentRunnerResult) -> dict[str, Any]:
    return {
        'experiment_id': result.experiment_id,
        'status': result.status.value,
        'workspace_dir': str(result.workspace_dir) if result.workspace_dir is not None else None,
        'message': result.message,
        'metadata': result.metadata,
    }


def _optional_runner_result_to_dict(result: ExperimentRunnerResult | None) -> dict[str, Any] | None:
    if result is None:
        return None

    return _runner_result_to_dict(result=result)


def _record_to_dict(record: ExperimentRecord) -> dict[str, Any]:
    data = asdict(record)
    data['status'] = record.status.value

    return add_kst_display_fields(data=data, fields=('created_at', 'started_at', 'finished_at'))


def _server_record_to_dict(record: ServerRecord) -> dict[str, Any]:
    return server_record_public_dict(record=record)


def _server_check_to_dict(result: ServerCheckResult) -> dict[str, Any]:
    return {
        'name': result.name,
        'status': result.status,
        'message': result.message,
        'metadata': result.metadata,
    }


def _require_server(
    storage: SQLiteExperimentStorage,
    explicit_name: str,
) -> ServerRecord:
    record = storage.get_server(name=explicit_name)
    if record is None:
        raise ValueError(f'unknown server: {explicit_name}')

    return record


def _server_for_experiment(
    storage: SQLiteExperimentStorage,
    experiment_id: str,
    explicit_name: str | None,
) -> ServerRecord:
    experiment = storage.get_experiment(experiment_id=experiment_id)
    if experiment is None:
        raise ValueError(f'unknown experiment_id: {experiment_id}')

    server_name = explicit_name or experiment.server_name
    if server_name is None:
        raise ValueError(f'experiment {experiment_id} has no server_name')

    return _require_server(storage=storage, explicit_name=server_name)


def _records_table(records: list[ExperimentRecord]) -> list[str]:
    if not records:
        return ['no experiments']

    lines = ['ID\tSTATUS\tNAME\tBEST_METRIC\tRESULT_PATH']
    for record in records:
        best_metric = _format_metric(record=record)
        lines.append(
            f'{record.experiment_id}\t{record.status.value}\t{record.name}\t{best_metric}\t{record.result_path or ""}',
        )

    return lines


def _compare_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ['no experiments']

    lines = ['ID\tSTATUS\tBEST_METRIC\tRESULT_PATH']
    for row in rows:
        metric_name = row['best_metric_name'] or ''
        metric_value = '' if row['best_metric_value'] is None else str(row['best_metric_value'])
        lines.append(
            f"{row['experiment_id']}\t{row['status']}\t{metric_name}={metric_value}\t{row['result_path'] or ''}",
        )

    return lines


def _server_table(records: list[ServerRecord]) -> list[str]:
    if not records:
        return ['no servers']

    lines = ['NAME\tTYPE\tHOST\tLAST_STATUS\tGPU\tVRAM']
    for record in records:
        lines.append(
            (
                f'{record.name}\t{record.server_type}\t{record.host or ""}\t'
                f'{record.last_status or ""}\t{record.gpu_name or ""}\t{record.total_vram or ""}'
            ),
        )

    return lines


def _print_compare_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=['experiment_id', 'name', 'status', 'best_metric_name', 'best_metric_value', 'result_path'],
    )
    writer.writeheader()
    writer.writerows(rows)


def _format_metric(record: ExperimentRecord) -> str:
    if record.best_metric_name is None:
        return ''
    if record.best_metric_value is None:
        return f'{record.best_metric_name}='

    return f'{record.best_metric_name}={record.best_metric_value}'


def _make_experiment_id(name: str) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name.strip().lower()).strip('_') or 'experiment'

    return f'exp_{timestamp}_{slug[:40]}'


def _make_schedule_id(config_name: str) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    slug = re.sub(r'[^A-Za-z0-9]+', '_', config_name.strip().lower()).strip('_') or 'schedule'

    return f'schedule_{timestamp}_{slug[:40]}'


def _parse_cli_datetime(value: str) -> datetime:
    normalized = value.strip().replace('Z', '+00:00')
    if ' ' in normalized and 'T' not in normalized:
        normalized = normalized.replace(' ', 'T', 1)

    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed


def _scheduler_result_to_dict(result: SchedulerRunResult) -> dict[str, Any]:
    return {
        'schedule_id': result.schedule_id,
        'experiment_id': result.experiment_id,
        'status': result.status,
        'message': result.message,
    }


def _scheduler_table(results: list[SchedulerRunResult]) -> list[str]:
    if not results:
        return ['no due schedules']

    lines = ['SCHEDULE_ID\tEXPERIMENT_ID\tSTATUS\tMESSAGE']
    for result in results:
        lines.append(
            f'{result.schedule_id}\t{result.experiment_id or ""}\t{result.status}\t{result.message}',
        )

    return lines


if __name__ == '__main__':
    raise SystemExit(main())
