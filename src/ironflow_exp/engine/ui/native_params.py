from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


NativeParamOverrides = Mapping[str, object] | None


def load_native_param_overrides_file(path: str | Path, *, project_dir: Path | None = None) -> dict[str, object]:
    raw_path = Path(path)
    resolved_path = raw_path if raw_path.is_absolute() else (project_dir or Path.cwd()) / raw_path
    data = yaml.safe_load(resolved_path.read_text(encoding='utf-8'))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f'native params file must be a YAML mapping: {resolved_path}')

    return dict(data)


def merge_native_param_overrides(
    *,
    params: Mapping[str, object],
    task_id: str,
    task_type: str,
    adapter: str | None,
    model_id: str | None,
    overrides: NativeParamOverrides,
) -> dict[str, object]:
    merged = dict(params)
    if not overrides:
        return merged

    for values in (
        _mapping_at(overrides, 'defaults'),
        _scoped_mapping(overrides, 'by_task_type', task_type),
        _scoped_mapping(overrides, 'by_adapter', adapter),
        _scoped_mapping(overrides, 'by_model_id', model_id),
        _scoped_mapping(overrides, 'by_task_id', task_id),
        _scoped_mapping(overrides, 'tasks', task_id),
    ):
        merged.update(values)

    return merged


def _mapping_at(data: Mapping[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'native params section must be a mapping: {key}')

    return dict(value)


def _scoped_mapping(data: Mapping[str, object], section: str, key: str | None) -> dict[str, object]:
    if key is None:
        return {}
    values = _mapping_at(data, section)
    scoped = values.get(key)
    if scoped is None:
        return {}
    if not isinstance(scoped, dict):
        raise ValueError(f'native params value must be a mapping: {section}.{key}')

    return dict(scoped)
