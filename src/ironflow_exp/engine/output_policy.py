from pathlib import Path

from ironflow_exp.engine.configs import EngineExperimentConfig, EngineOutputConfig

OUTPUT_ARG_ENTRYPOINTS = {'mock_train.py', 'test_model_smoke.py'}
METRICS_ONLY_PATTERNS = {
    'metrics.json',
    'experiment.json',
    'artifacts.json',
    'timings.csv',
    'task_results.json',
}


def effective_collect_patterns(output: EngineOutputConfig) -> list[str]:
    mode = _normalized_collection_mode(output=output)
    patterns: list[str] = []
    for pattern in output.collect_patterns:
        if _is_disabled_by_collection_mode(pattern=pattern, output=output, mode=mode):
            continue
        if _is_disabled_csv(pattern=pattern, output=output):
            continue
        if _is_disabled_summary(pattern=pattern, output=output):
            continue
        if _is_disabled_checkpoint(pattern=pattern, output=output):
            continue
        if _is_disabled_preview(pattern=pattern, output=output):
            continue
        patterns.append(pattern)

    return patterns


def _normalized_collection_mode(output: EngineOutputConfig) -> str:
    mode = output.artifact_collection_mode.strip().lower()
    if mode in {'full', 'light', 'metrics_only'}:
        return mode

    return 'full'


def _is_disabled_by_collection_mode(pattern: str, output: EngineOutputConfig, mode: str) -> bool:
    if mode == 'full':
        return False

    normalized = _normalized_pattern(pattern=pattern)
    if _is_task_directory_pattern(pattern=normalized):
        return True
    if _is_checkpoint_pattern(pattern=pattern, output=output):
        return True
    if _is_preview_pattern(pattern=pattern) and not output.save_previews:
        return True

    if mode == 'metrics_only':
        allowed = {
            output.log_file,
            output.metrics_file,
            output.status_file,
            output.summary_file,
            *METRICS_ONLY_PATTERNS,
        }
        return normalized not in allowed

    return False


def build_mock_output_args(config: EngineExperimentConfig) -> list[str]:
    if Path(config.code.entrypoint).name not in OUTPUT_ARG_ENTRYPOINTS:
        return []

    args = [
        '--log-file',
        config.output.log_file,
        '--metrics-file',
        config.output.metrics_file,
        '--checkpoint-file',
        config.output.checkpoint_file,
        '--status-file',
        config.output.status_file,
        '--summary-file',
        config.output.summary_file,
    ]
    if not config.output.save_csv:
        args.append('--no-save-csv')
    if not config.output.save_summary:
        args.append('--no-save-summary')
    if not config.output.save_checkpoints:
        args.append('--no-save-checkpoints')
    if not config.output.save_previews:
        args.append('--no-save-previews')

    return args


def _is_disabled_csv(pattern: str, output: EngineOutputConfig) -> bool:
    if output.save_csv:
        return False

    return pattern == output.metrics_file or pattern.endswith('.csv')


def _is_disabled_summary(pattern: str, output: EngineOutputConfig) -> bool:
    if output.save_summary:
        return False

    return pattern == output.summary_file or pattern.endswith('.md')


def _is_disabled_checkpoint(pattern: str, output: EngineOutputConfig) -> bool:
    if output.save_checkpoints:
        return False

    return _is_checkpoint_pattern(pattern=pattern, output=output)


def _is_checkpoint_pattern(pattern: str, output: EngineOutputConfig) -> bool:
    checkpoint_suffixes = ('.pt', '.pth', '.ckpt')
    return pattern == output.checkpoint_file or pattern.endswith(checkpoint_suffixes)


def _is_disabled_preview(pattern: str, output: EngineOutputConfig) -> bool:
    if output.save_previews:
        return False

    return _is_preview_pattern(pattern=pattern)


def _is_preview_pattern(pattern: str) -> bool:
    normalized = pattern.replace('\\', '/')
    return (
        normalized.startswith('sample_predictions/')
        or normalized.startswith('previews/')
    )


def _is_task_directory_pattern(pattern: str) -> bool:
    return pattern in {'tasks', 'tasks/'}


def _normalized_pattern(pattern: str) -> str:
    return pattern.replace('\\', '/').rstrip('/')
