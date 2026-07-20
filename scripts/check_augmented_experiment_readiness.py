#!/usr/bin/env python
"""Static readiness checks for augmented candidate configs.

Use this before sending a run to Vast. It intentionally fails fast on split
pollution risks and missing materialized augmented dataset roots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.configs import EngineConfigLoader  # noqa: E402
from ironflow_exp.engine.ui.augmentation_policy_catalog import augmentation_policy_by_id  # noqa: E402

VALID_POLICIES = {'light_v1', 'medium_v1'}
IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Check augmented experiment config readiness.')
    parser.add_argument('configs', nargs='+', help='Generated config YAML files.')
    parser.add_argument('--allow-missing-dataset', action='store_true', help='Warn instead of fail when augmented dataset roots are not materialized yet.')
    args = parser.parse_args()

    total_errors = 0
    total_warnings = 0
    for raw_config in args.configs:
        config_path = resolve_project_path(raw_config)
        errors, warnings = check_config(config_path=config_path, allow_missing_dataset=args.allow_missing_dataset)
        total_errors += len(errors)
        total_warnings += len(warnings)
        print(f'[{"OK" if not errors else "FAIL"}] {config_path.relative_to(PROJECT_ROOT).as_posix()}')
        for warning in warnings:
            print(f'  warning: {warning}')
        for error in errors:
            print(f'  error: {error}')
    print(f'summary: errors={total_errors}, warnings={total_warnings}')
    raise SystemExit(1 if total_errors else 0)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_config(*, config_path: Path, allow_missing_dataset: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            raise ValueError('config root must be a mapping')
        EngineConfigLoader().load_file(config_path)
    except Exception as exc:  # noqa: BLE001 - this is a checker, surface exact exception text.
        return [f'config load failed: {type(exc).__name__}: {exc}'], warnings

    variants = [variant for variant in raw.get('data_variants', []) if isinstance(variant, dict)]
    tasks = [task for task in raw.get('tasks', []) if isinstance(task, dict)]
    if not variants:
        errors.append('no data_variants found')
        return errors, warnings
    if not tasks:
        errors.append('no tasks found')
        return errors, warnings

    primary_variant = variants[0]
    params = primary_variant.get('params') if isinstance(primary_variant.get('params'), dict) else {}
    policy = str(params.get('augmentation_policy') or '').strip()
    if policy not in VALID_POLICIES:
        errors.append(f'first data_variant must record augmentation_policy in {sorted(VALID_POLICIES)}, got {policy!r}')
    else:
        validate_policy_catalog(policy=policy, task_types={str(task.get('task_type')) for task in tasks}, errors=errors)

    variant_path = str(primary_variant.get('path') or '').strip()
    if not variant_path:
        errors.append('first data_variant.path is empty')
        return errors, warnings
    dataset_root = resolve_relative_to_config(config_path=config_path, value=variant_path)
    task_type = primary_task_type(tasks=tasks, errors=errors)
    if not dataset_root.exists():
        message = f'augmented dataset root does not exist yet: {dataset_root}'
        if allow_missing_dataset:
            warnings.append(message)
        else:
            errors.append(message)
        return errors, warnings

    if task_type == 'detection':
        check_detection_dataset(dataset_root=dataset_root, errors=errors, warnings=warnings)
    elif task_type == 'classification':
        check_classification_dataset(dataset_root=dataset_root, errors=errors, warnings=warnings)

    for task in tasks:
        if str(task.get('task_type')) == 'augmentation':
            errors.append('generated augmented candidate configs should point to materialized datasets, not dynamic augmentation tasks')
        if isinstance(task.get('params'), dict) and task['params'].get('prediction_split') not in {None, 'test'}:
            warnings.append(f"task {task.get('id')} prediction_split is not test: {task['params'].get('prediction_split')}")

    return errors, warnings


def validate_policy_catalog(*, policy: str, task_types: set[str], errors: list[str]) -> None:
    policy_id = (
        'albumentations_detection_bbox_' + policy
        if task_types == {'detection'}
        else 'albumentations_classification_crop_' + policy
        if task_types == {'classification'}
        else ''
    )
    if not policy_id:
        errors.append(f'unsupported task type mix for augmentation policy validation: {sorted(task_types)}')
        return
    try:
        augmentation_policy_by_id(policy_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f'augmentation policy catalog lookup failed for {policy_id}: {exc}')


def primary_task_type(*, tasks: list[dict[str, Any]], errors: list[str]) -> str | None:
    task_types = {str(task.get('task_type')) for task in tasks}
    if 'detection' in task_types and 'classification' not in task_types:
        return 'detection'
    if 'classification' in task_types and 'detection' not in task_types:
        return 'classification'
    errors.append(f'expected detector-only or classifier-only tasks, got {sorted(task_types)}')
    return None


def resolve_relative_to_config(*, config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def check_detection_dataset(*, dataset_root: Path, errors: list[str], warnings: list[str]) -> None:
    if not (dataset_root / 'manifest.json').exists():
        errors.append(f'detection manifest.json missing: {dataset_root / "manifest.json"}')
    if not (dataset_root / 'data.yaml').exists():
        errors.append(f'YOLO data.yaml missing: {dataset_root / "data.yaml"}')
    coco_root = dataset_root / 'coco'
    if not (coco_root / 'coco_dataset.json').exists():
        errors.append(f'COCO materialization missing: {coco_root / "coco_dataset.json"}')
    for split in ['train', 'val', 'test']:
        if not (dataset_root / 'images' / split).exists():
            errors.append(f'detection split image folder missing: images/{split}')
        if split in {'val', 'test'} and contains_augmented_names(dataset_root / 'images' / split):
            errors.append(f'detection {split} split appears to contain augmented files')
    for split in ['train', 'val', 'test']:
        annotation = coco_root / 'annotations' / f'instances_{split}.json'
        if not annotation.exists():
            warnings.append(f'COCO annotation missing for split {split}: {annotation}')


def check_classification_dataset(*, dataset_root: Path, errors: list[str], warnings: list[str]) -> None:
    for split in ['train', 'val', 'test']:
        split_root = dataset_root / split
        if not split_root.exists():
            errors.append(f'classification split folder missing: {split}')
            continue
        class_dirs = [path for path in split_root.iterdir() if path.is_dir()]
        if not class_dirs:
            errors.append(f'classification split has no class folders: {split}')
        if split in {'val', 'test'} and contains_augmented_names(split_root):
            errors.append(f'classification {split} split appears to contain augmented files')
    if not any((dataset_root / 'train').rglob('*')):
        warnings.append('classification train split appears empty')


def contains_augmented_names(root: Path) -> bool:
    if not root.exists():
        return False
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            lowered = path.name.lower()
            if '__aug_' in lowered or '_aug_' in lowered or 'augmentation' in lowered:
                return True
    return False


if __name__ == '__main__':
    main()