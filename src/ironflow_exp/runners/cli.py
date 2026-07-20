import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ironflow_exp.configs import ConfigLoader
from ironflow_exp.runners.base import RunnerResult
from ironflow_exp.runners.local_runner import LocalRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='ironflow-exp')
    subparsers = parser.add_subparsers(dest='command', required=True)
    run_parser = subparsers.add_parser('run-preprocessing', help='Run preprocessing and export skeleton outputs')
    run_parser.add_argument('--config', required=True, help='Experiment config YAML or JSON path')
    run_parser.add_argument('--run-id', default=None, help='Optional explicit run id')
    run_parser.add_argument(
        '--no-check-paths',
        action='store_true',
        help='Skip source_root existence validation',
    )
    run_parser.add_argument(
        '--json',
        action='store_true',
        help='Print machine-readable JSON summary',
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'run-preprocessing':
        return run_preprocessing_command(args=args)

    parser.error(f'unsupported command: {args.command}')

    return 2


def run_preprocessing_command(args: argparse.Namespace) -> int:
    config_path = Path(str(args.config))
    config = ConfigLoader().load_file(path=config_path)
    result = LocalRunner().run_preprocessing(
        config=config,
        run_id=args.run_id,
        check_paths=not args.no_check_paths,
    )
    summary = _summary(result=result)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(_human_summary(summary=summary))

    return 0 if result.is_success else 1


def _summary(result: RunnerResult) -> dict[str, object]:
    service_result = result.service_result

    return {
        'success': result.is_success,
        'job': asdict(result.job),
        'output_dir': result.job.metadata.get('output_dir'),
        'artifact_manifest_path': result.job.metadata.get('artifact_manifest_path'),
        'error_type': result.error_type,
        'error_message': result.error_message,
        'stage_statuses': service_result.pipeline_result.stage_statuses if service_result is not None else {},
        'exported_paths': {
            key: str(path)
            for key, path in service_result.export_result.exported_paths.items()
        } if service_result is not None else {},
    }


def _human_summary(summary: dict[str, object]) -> str:
    job = summary['job']
    if not isinstance(job, dict):
        raise TypeError('summary job must be a mapping')

    lines = [
        f"status: {job['status']}",
        f"run_id: {job['run_id']}",
        f"output_dir: {summary['output_dir']}",
    ]

    if summary['artifact_manifest_path'] is not None:
        lines.append(f"artifact_manifest: {summary['artifact_manifest_path']}")
    if summary['error_message'] is not None:
        lines.append(f"error: {summary['error_type']}: {summary['error_message']}")

    return '\n'.join(lines)


if __name__ == '__main__':
    raise SystemExit(main())
