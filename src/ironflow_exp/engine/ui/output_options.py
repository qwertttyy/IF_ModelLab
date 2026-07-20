import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from ironflow_exp.engine.configs import EngineConfigLoader, EngineExperimentConfig
from ironflow_exp.engine.tasks.model_task_adapters import REAL_MODEL_TASK_ADAPTER_KEYS


WEIGHT_MODE_PRESET = 'preset'
WEIGHT_MODE_ARCHITECTURE = 'architecture'
WEIGHT_MODE_PRETRAINED_DOWNLOAD = 'pretrained_download'
WEIGHT_MODE_CHECKPOINT = 'checkpoint'
COLLECT_MODE_QUICK = 'quick'
COLLECT_MODE_STANDARD = 'standard'
COLLECT_MODE_WEIGHTS = 'weights'
COLLECT_MODE_FULL_DEBUG = 'full_debug'
COLLECT_MODES = frozenset({
    COLLECT_MODE_QUICK,
    COLLECT_MODE_STANDARD,
    COLLECT_MODE_WEIGHTS,
    COLLECT_MODE_FULL_DEBUG,
})
CHECKPOINT_COLLECT_OFF = 'off'
CHECKPOINT_COLLECT_BEST = 'best'
CHECKPOINT_COLLECT_BEST_LAST = 'best_last'
CHECKPOINT_COLLECT_MODES = frozenset({
    CHECKPOINT_COLLECT_OFF,
    CHECKPOINT_COLLECT_BEST,
    CHECKPOINT_COLLECT_BEST_LAST,
})
WEIGHT_MODES = frozenset({
    WEIGHT_MODE_PRESET,
    WEIGHT_MODE_ARCHITECTURE,
    WEIGHT_MODE_PRETRAINED_DOWNLOAD,
    WEIGHT_MODE_CHECKPOINT,
})
TASK_LIGHTWEIGHT_COLLECT_PATTERNS = [
    'tasks/*/task.json',
    'tasks/*/task.log',
    'tasks/*/metrics.csv',
    'tasks/*/metrics.json',
    'tasks/*/metrics/*.csv',
    'tasks/*/status.marker',
    'tasks/*/summary.md',
    'tasks/*/timings.csv',
    'tasks/*/predictions.json',
    'tasks/*/artifacts.json',
    'tasks/*/predictions/*.json',
    'tasks/*/classification_input_manifest.json',
    'tasks/*/augmentation_manifest.json',
]
TASK_CHECKPOINT_COLLECT_PATTERNS = [
    'tasks/*/checkpoints/best.pt',
    'tasks/*/checkpoints/*/weights/best.pt',
]
TASK_LAST_CHECKPOINT_COLLECT_PATTERNS = [
    'tasks/*/checkpoints/last.pt',
    'tasks/*/checkpoints/*/weights/last.pt',
]
TASK_PREVIEW_COLLECT_PATTERNS = [
    'tasks/*/previews/detection_bbox/*.jpg',
]
DETECTION_BBOX_PREVIEW_ADAPTERS = {
    'ultralytics_yolo',
    'rf_detr_detection',
    'rt_detr_detection',
    'rt_detr_v2_detection',
    'lw_detr_detection',
    'd_fine_detection',
}


@dataclass(frozen=True, slots=True)
class GuiOutputOptions:
    save_csv: bool = True
    save_summary: bool = True
    save_previews: bool = True
    save_checkpoints: bool = True
    collect_mode: str = COLLECT_MODE_WEIGHTS
    checkpoint_mode: str = CHECKPOINT_COLLECT_BEST


@dataclass(frozen=True, slots=True)
class GuiWeightOptions:
    mode: str = WEIGHT_MODE_PRESET
    checkpoint_path: str = ''


def write_effective_config(
    project_dir: Path,
    source_config_path: str,
    experiment_id: str,
    options: GuiOutputOptions,
    prefix: str,
    dataset_dir: str | None = None,
    config_transform: Callable[[EngineExperimentConfig], EngineExperimentConfig] | None = None,
    weight_options: GuiWeightOptions | None = None,
) -> str:
    loader = EngineConfigLoader()
    source_path = _resolve_config_path(project_dir=project_dir, config_path=source_config_path)
    config = loader.load_file(source_path)
    if config_transform is not None:
        config = config_transform(config)
    data = loader.to_dict(config=config)
    effective_weight_options = weight_options or GuiWeightOptions()
    _apply_weight_options(data=data, options=effective_weight_options)
    _apply_checkpoint_package_include(
        data=data,
        project_dir=project_dir,
        options=effective_weight_options,
    )
    output = data.setdefault('output', {})
    output['save_json'] = True
    output['save_csv'] = options.save_csv
    output['save_summary'] = options.save_summary
    output['save_previews'] = options.save_previews
    output['save_checkpoints'] = options.save_checkpoints
    if _should_apply_task_collect_policy(data=data, output=output):
        _apply_collect_mode_options(output=output, options=options)
        _apply_preview_collection_options(data=data, output=output, options=options)
        _apply_checkpoint_collection_options(output=output, options=options)
    if dataset_dir:
        _set_train_arg(data=data, option='--dataset-dir', value=str(Path(dataset_dir).resolve()))
        _widen_preview_collect_pattern(output=output)

    output_dir = project_dir / 'runs' / 'configs' / 'gui_effective'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{_safe_name(prefix)}_{_safe_name(experiment_id)}.yaml'
    output_path.write_text(
        data=yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )

    try:
        return str(output_path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return str(output_path.resolve())


def _resolve_config_path(project_dir: Path, config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path

    return project_dir / path


def _set_train_arg(data: dict, option: str, value: str) -> None:
    train = data.setdefault('train', {})
    args = train.setdefault('args', [])
    if not isinstance(args, list):
        raise ValueError('train.args must be a list to apply GUI overrides')

    for index, arg in enumerate(args):
        if arg == option:
            if index + 1 >= len(args):
                args.append(value)
            else:
                args[index + 1] = value
            return

    args.extend([option, value])


def _widen_preview_collect_pattern(output: dict) -> None:
    patterns = output.get('collect_patterns')
    if not isinstance(patterns, list):
        return

    output['collect_patterns'] = [
        'previews/samples/*' if pattern == 'previews/samples/*.ppm' else pattern
        for pattern in patterns
    ]


def _apply_checkpoint_collection_options(*, output: dict, options: GuiOutputOptions) -> None:
    checkpoint_mode = _checkpoint_collect_mode(options=options)
    if checkpoint_mode == CHECKPOINT_COLLECT_OFF:
        output['save_checkpoints'] = False
        return
    if not options.save_checkpoints or output.get('save_checkpoints') is False:
        return

    previous_mode = str(output.get('artifact_collection_mode', 'full')).strip().lower()
    if _collect_mode(options=options) == COLLECT_MODE_FULL_DEBUG and previous_mode == 'full':
        return

    output['artifact_collection_mode'] = 'full'

    patterns = output.setdefault('collect_patterns', [])
    if not isinstance(patterns, list):
        raise ValueError('output.collect_patterns must be a list to collect checkpoints')
    output['collect_patterns'] = _checkpoint_collect_patterns(patterns=patterns, checkpoint_mode=checkpoint_mode)


def _apply_collect_mode_options(*, output: dict, options: GuiOutputOptions) -> None:
    collect_mode = _collect_mode(options=options)
    if collect_mode == COLLECT_MODE_FULL_DEBUG:
        output['artifact_collection_mode'] = 'full'
        patterns = output.setdefault('collect_patterns', [])
        if isinstance(patterns, list) and 'tasks/' not in {str(pattern).replace('\\', '/') for pattern in patterns}:
            patterns.append('tasks/')
        return

    output['artifact_collection_mode'] = 'full'
    base_patterns = _base_collect_patterns(output=output)
    output['collect_patterns'] = _dedupe_patterns([*base_patterns, *TASK_LIGHTWEIGHT_COLLECT_PATTERNS])
    if collect_mode == COLLECT_MODE_QUICK:
        output['save_checkpoints'] = False


def _should_apply_task_collect_policy(*, data: dict, output: dict) -> bool:
    runtime = data.get('runtime')
    runner = runtime.get('runner') if isinstance(runtime, dict) else None
    return runner == 'ssh' or bool(data.get('tasks'))


def _collect_mode(*, options: GuiOutputOptions) -> str:
    mode = options.collect_mode.strip().lower()
    if mode in COLLECT_MODES:
        return mode

    return COLLECT_MODE_WEIGHTS


def _checkpoint_collect_mode(*, options: GuiOutputOptions) -> str:
    if not options.save_checkpoints:
        return CHECKPOINT_COLLECT_OFF
    mode = options.checkpoint_mode.strip().lower()
    if mode in CHECKPOINT_COLLECT_MODES:
        return mode

    return CHECKPOINT_COLLECT_BEST


def _apply_preview_collection_options(*, data: dict, output: dict, options: GuiOutputOptions) -> None:
    if _collect_mode(options=options) == COLLECT_MODE_QUICK:
        output['save_previews'] = False
        return
    if not options.save_previews:
        return
    if not _has_detection_bbox_preview_task(data=data):
        return

    patterns = output.setdefault('collect_patterns', [])
    if not isinstance(patterns, list):
        raise ValueError('output.collect_patterns must be a list to collect previews')
    existing = {
        pattern.replace('\\', '/') if isinstance(pattern, str) else pattern
        for pattern in patterns
    }
    for pattern in TASK_PREVIEW_COLLECT_PATTERNS:
        if pattern not in existing:
            patterns.append(pattern)
            existing.add(pattern)


def _has_detection_bbox_preview_task(*, data: dict) -> bool:
    tasks = data.get('tasks')
    if not isinstance(tasks, list):
        return False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task.get('task_type') == 'detection' and task.get('adapter') in DETECTION_BBOX_PREVIEW_ADAPTERS:
            return True
    return False


def _is_task_collection_pattern(*, pattern: object) -> bool:
    if not isinstance(pattern, str):
        return False

    normalized = pattern.replace('\\', '/').rstrip('/')
    return normalized == 'tasks'


def _checkpoint_collect_patterns(*, patterns: list[object], checkpoint_mode: str) -> list[object]:
    filtered = [
        pattern
        for pattern in patterns
        if not _is_task_collection_pattern(pattern=pattern)
    ]
    existing = {
        pattern.replace('\\', '/') if isinstance(pattern, str) else pattern
        for pattern in filtered
    }
    checkpoint_patterns = list(TASK_CHECKPOINT_COLLECT_PATTERNS)
    if checkpoint_mode == CHECKPOINT_COLLECT_BEST_LAST:
        checkpoint_patterns.extend(TASK_LAST_CHECKPOINT_COLLECT_PATTERNS)
    for pattern in [*TASK_LIGHTWEIGHT_COLLECT_PATTERNS, *checkpoint_patterns]:
        if pattern not in existing:
            filtered.append(pattern)
            existing.add(pattern)

    return filtered


def _base_collect_patterns(*, output: dict) -> list[object]:
    configured = output.get('collect_patterns')
    if not isinstance(configured, list):
        configured = []
    base_names = [
        'train.log',
        'metrics.csv',
        'metrics.json',
        'status.marker',
        'summary.md',
        'experiment.json',
        'artifacts.json',
        'timings.csv',
        'task_results.json',
    ]
    base_name_set = set(base_names)
    filtered = [
        pattern
        for pattern in configured
        if isinstance(pattern, str) and pattern.replace('\\', '/') in base_name_set
    ]
    return _dedupe_patterns([*filtered, *base_names])


def _dedupe_patterns(patterns: list[object]) -> list[object]:
    deduped: list[object] = []
    seen: set[object] = set()
    for pattern in patterns:
        key = pattern.replace('\\', '/') if isinstance(pattern, str) else pattern
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pattern)
    return deduped


def _apply_weight_options(*, data: dict, options: GuiWeightOptions) -> None:
    if options.mode == WEIGHT_MODE_PRESET:
        return
    if options.mode not in WEIGHT_MODES:
        raise ValueError(f'unsupported weight mode: {options.mode}')
    if options.mode == WEIGHT_MODE_CHECKPOINT and not options.checkpoint_path.strip():
        raise ValueError('checkpoint path is required for checkpoint weight mode')

    tasks = data.get('tasks')
    if not isinstance(tasks, list):
        return

    for task in tasks:
        if not isinstance(task, dict) or not _supports_weight_options(task=task):
            continue
        params = task.setdefault('params', {})
        if not isinstance(params, dict):
            raise ValueError(f'task params must be a mapping object: {task.get("id", "")}')
        _apply_task_weight_options(params=params, options=options)


def _supports_weight_options(*, task: dict) -> bool:
    task_type = task.get('task_type')
    adapter = task.get('adapter')
    if not isinstance(task_type, str) or not isinstance(adapter, str):
        return False

    return (task_type, adapter) in REAL_MODEL_TASK_ADAPTER_KEYS


def _apply_task_weight_options(*, params: dict, options: GuiWeightOptions) -> None:
    if options.mode == WEIGHT_MODE_ARCHITECTURE:
        params['pretrained'] = False
        params['allow_pretrained_download'] = False
        params.pop('checkpoint', None)
        return
    if options.mode == WEIGHT_MODE_PRETRAINED_DOWNLOAD:
        params['pretrained'] = True
        params['allow_pretrained_download'] = True
        params.pop('checkpoint', None)
        return
    if options.mode == WEIGHT_MODE_CHECKPOINT:
        params['pretrained'] = False
        params['allow_pretrained_download'] = False
        params['checkpoint'] = options.checkpoint_path.strip()
        return

    raise ValueError(f'unsupported weight mode: {options.mode}')


def _apply_checkpoint_package_include(*, data: dict, project_dir: Path, options: GuiWeightOptions) -> None:
    if options.mode != WEIGHT_MODE_CHECKPOINT or not options.checkpoint_path.strip():
        return

    package_path = _package_checkpoint_path(
        project_dir=project_dir,
        checkpoint_path=options.checkpoint_path.strip(),
    )
    if package_path is None:
        return

    code = data.setdefault('code', {})
    if not isinstance(code, dict):
        raise ValueError('code must be a mapping object to apply checkpoint package include')
    includes = code.setdefault('package_include', [])
    if not isinstance(includes, list):
        raise ValueError('code.package_include must be a list to apply checkpoint package include')
    if package_path not in includes:
        includes.append(package_path)


def _package_checkpoint_path(*, project_dir: Path, checkpoint_path: str) -> str | None:
    path = Path(checkpoint_path)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(project_dir.resolve())
        except ValueError:
            return None
    else:
        relative = path

    return relative.as_posix()


def _safe_name(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip())
    return safe.strip('._') or 'config'
