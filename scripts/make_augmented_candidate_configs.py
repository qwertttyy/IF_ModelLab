#!/usr/bin/env python
"""Generate stable augmented-dataset experiment configs from existing candidate configs.

This generator does not perform augmentation itself. It creates configs that point
at pre-materialized augmented dataset roots, where train is augmented and val/test
remain original. This is intentionally more stable than wiring dynamic augmentation
artifacts directly into model training wrappers.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = 'tank_armor_prepared_v20260630'
DEFAULT_AUG_ROOT_TEMPLATE = '../../../runs/user_datasets/{dataset_name}_aug/{policy}/{dataset_leaf}'
POLICIES = {'light_v1', 'medium_v1'}
TASK_TYPE_TO_DATASET_LEAF = {
    'detection': 'detector_tank_av/detection',
    'classification': 'classifier_mbt/crops',
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate augmented candidate engine configs.')
    parser.add_argument('--base-config', action='append', required=True, help='Base candidate YAML config. Repeatable.')
    parser.add_argument('--policy', action='append', choices=sorted(POLICIES), required=True, help='Aug policy. Repeatable.')
    parser.add_argument('--output-dir', default='configs/engine/augmented_candidates', help='Output directory under project root or absolute path.')
    parser.add_argument('--dataset-name', default=DEFAULT_DATASET_NAME)
    parser.add_argument('--image-size', type=int, default=None, help='Optional override for train/predict image_size.')
    parser.add_argument('--name-prefix', default='aug', help='Prefix for experiment names and DB names.')
    args = parser.parse_args()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for base_config_arg in args.base_config:
        base_path = resolve_project_path(base_config_arg)
        config = read_yaml(base_path)
        task_type = primary_task_type(config)
        dataset_leaf = TASK_TYPE_TO_DATASET_LEAF[task_type]
        for policy in args.policy:
            augmented = build_augmented_config(
                config=config,
                base_path=base_path,
                task_type=task_type,
                policy=policy,
                dataset_name=args.dataset_name,
                dataset_leaf=dataset_leaf,
                image_size=args.image_size,
                name_prefix=args.name_prefix,
            )
            output_path = output_dir / augmented_filename(base_path=base_path, policy=policy, image_size=args.image_size)
            write_yaml(output_path, augmented)
            written.append(output_path)

    for path in written:
        print(path.relative_to(PROJECT_ROOT).as_posix())


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'config root must be a mapping: {path}')
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def primary_task_type(config: dict[str, Any]) -> str:
    tasks = [task for task in config.get('tasks', []) if isinstance(task, dict)]
    task_types = {str(task.get('task_type')) for task in tasks}
    if 'detection' in task_types and 'classification' not in task_types:
        return 'detection'
    if 'classification' in task_types and 'detection' not in task_types:
        return 'classification'
    raise ValueError(f'expected detector-only or classifier-only config, got task types: {sorted(task_types)}')


def build_augmented_config(
    *,
    config: dict[str, Any],
    base_path: Path,
    task_type: str,
    policy: str,
    dataset_name: str,
    dataset_leaf: str,
    image_size: int | None,
    name_prefix: str,
) -> dict[str, Any]:
    output = copy.deepcopy(config)
    base_name = str(output.get('experiment', {}).get('name') or base_path.stem)
    suffix = f'{policy}' if image_size is None else f'{policy}_img{image_size}'
    output.setdefault('experiment', {})['name'] = safe_id(f'{name_prefix}_{base_name}_{suffix}')
    description = str(output['experiment'].get('description') or '')
    output['experiment']['description'] = (description + f' | Augmented dataset policy={policy}').strip()
    tags = list(output['experiment'].get('tags') or [])
    for tag in ['augmentation_ablation', f'aug_{policy}', 'materialized_augmented_dataset']:
        if tag not in tags:
            tags.append(tag)
    output['experiment']['tags'] = tags

    variants = output.get('data_variants')
    if not isinstance(variants, list) or not variants:
        raise ValueError(f'config has no data_variants: {base_path}')
    variant = variants[0]
    if not isinstance(variant, dict):
        raise ValueError(f'first data_variant must be a mapping: {base_path}')
    variant['path'] = DEFAULT_AUG_ROOT_TEMPLATE.format(
        dataset_name=dataset_name,
        policy=policy,
        dataset_leaf=dataset_leaf,
    )
    variant.setdefault('params', {})
    variant['params']['augmentation_policy'] = policy
    variant['params']['augmentation_materialization'] = 'train_augmented_val_test_original'

    if image_size is not None:
        for task in output.get('tasks', []):
            if isinstance(task, dict) and isinstance(task.get('params'), dict):
                task['params']['image_size'] = image_size

    repo = output.setdefault('repository', {})
    sqlite_path = str(repo.get('sqlite_path') or '../../../runs/augmented_experiments.sqlite3')
    repo['sqlite_path'] = suffixed_path(sqlite_path, f'_{suffix}')
    return output


def augmented_filename(*, base_path: Path, policy: str, image_size: int | None) -> str:
    suffix = policy if image_size is None else f'{policy}_img{image_size}'
    return f'{base_path.stem}__aug_{suffix}.yaml'


def suffixed_path(value: str, suffix: str) -> str:
    path = Path(value)
    name = path.name
    if '.' in name:
        stem, extension = name.rsplit('.', 1)
        name = f'{stem}{suffix}.{extension}'
    else:
        name = f'{name}{suffix}'
    return str(path.with_name(name)).replace('\\', '/')


def safe_id(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', value).strip('_') or 'augmented_experiment'


if __name__ == '__main__':
    main()