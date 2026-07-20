"""Shared utilities for IronFlow external model wrappers.

The wrappers are intentionally small command-line programs. They receive an
IronFlow request JSON, call a model package or repository, and write artifacts
back in IronFlow schemas.
"""

from __future__ import annotations

import csv
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_request(path: str | Path, *, required_keys: tuple[str, ...]) -> dict[str, Any]:
    request_path = Path(path).expanduser().resolve()
    data = json.loads(request_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'request JSON root must be an object: {request_path}')
    for key in required_keys:
        if key not in data:
            raise ValueError(f'request JSON is missing required key: {key}')

    return data


def result_dir_from_request(request: dict[str, Any]) -> Path:
    result_dir = Path(str(request['result_dir'])).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def request_param(request: dict[str, Any], key: str, default: Any = None) -> Any:
    params = request.get('params')
    if isinstance(params, dict) and key in params:
        return params[key]

    return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_contract_metadata(
    *,
    result_dir: Path,
    request_path: Path,
    request: dict[str, Any],
    model_family: str,
    adapter_key: str,
    native_ready: bool,
) -> None:
    stale_error = result_dir / 'external' / 'native_error.json'
    if stale_error.exists():
        stale_error.unlink()
    payload = {
        'schema_version': '0.1',
        'model_family': model_family,
        'adapter_key': adapter_key,
        'request_json': str(request_path),
        'request_adapter': request.get('adapter'),
        'request_model_id': request.get('model_id'),
        'execution_mode': request.get('execution_mode'),
        'result_dir': str(result_dir),
        'native_ready': native_ready,
    }
    write_json(result_dir / 'external' / 'wrapper_contract.json', payload)


def write_error(
    *,
    result_dir: Path,
    adapter_key: str,
    model_family: str,
    error_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> int:
    payload = {
        'ok': False,
        'adapter_key': adapter_key,
        'model_family': model_family,
        'error_type': error_type,
        'error': message,
        'details': details or {},
    }
    write_json(result_dir / 'external' / 'native_error.json', payload)
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)

    return 2


def optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f'missing Python dependency: {module_name}') from exc


def write_metric_rows(result_dir: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / 'metrics.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def write_runtime_metrics(
    *,
    result_dir: Path,
    elapsed_seconds: float,
    extra: dict[str, Any] | None = None,
) -> None:
    base_fieldnames = ['epoch', 'train_loss', 'val_loss', 'accuracy', 'map50', 'map50_95', 'lr', 'elapsed_seconds']
    row = {
        'epoch': 1,
        'train_loss': '',
        'val_loss': '',
        'accuracy': '',
        'map50': '',
        'map50_95': '',
        'lr': '',
        'elapsed_seconds': f'{elapsed_seconds:.6f}',
        **(extra or {}),
    }
    extra_fieldnames = [key for key in (extra or {}) if key not in base_fieldnames]
    write_metric_rows(
        result_dir,
        [row],
        fieldnames=[*base_fieldnames, *extra_fieldnames],
    )


def run_subprocess(
    command: list[str] | str,
    *,
    cwd: Path | None,
    timeout_seconds: int | None,
    result_dir: Path,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=isinstance(command, str),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    external_dir = result_dir / 'external'
    external_dir.mkdir(parents=True, exist_ok=True)
    (external_dir / 'native_stdout.log').write_text(completed.stdout, encoding='utf-8')
    (external_dir / 'native_stderr.log').write_text(completed.stderr, encoding='utf-8')

    return completed


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    return True


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'manifest root must be an object: {path}')
    images = data.get('images')
    if not isinstance(images, list):
        raise ValueError(f'manifest images must be a list: {path}')

    return data


def manifest_image_records(request: dict[str, Any]) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(str(request.get('input_manifest') or '')).expanduser()
    if not manifest_path.is_absolute():
        dataset_root = Path(str(request.get('dataset_root') or '.')).expanduser()
        manifest_path = (dataset_root / manifest_path).resolve()
    manifest = read_manifest(manifest_path)

    return manifest_path, manifest, [row for row in manifest['images'] if isinstance(row, dict)]


def image_path_for_record(*, manifest_path: Path, record: dict[str, Any]) -> Path:
    raw_path = record.get('path') or record.get('image_path') or record.get('source_path')
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError('manifest image record is missing path/image_path/source_path')
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    return (manifest_path.parent / path).resolve()


def write_detection_predictions(
    *,
    result_dir: Path,
    request: dict[str, Any],
    records: list[dict[str, Any]],
) -> Path:
    payload = {
        'schema_version': '0.1',
        'task': 'detection',
        'model_id': str(request.get('model_id') or request.get('adapter') or 'external_detector'),
        'dataset_id': Path(str(request.get('dataset_root') or 'dataset')).name,
        'success': True,
        'records': records,
    }
    path = result_dir / 'predictions' / 'detection_predictions.json'
    write_json(path, payload)

    return path


def write_segmentation_predictions(
    *,
    result_dir: Path,
    request: dict[str, Any],
    records: list[dict[str, Any]],
) -> Path:
    payload = {
        'schema_version': '0.1',
        'task': 'segmentation',
        'model_id': str(request.get('model_id') or request.get('adapter') or 'external_segmenter'),
        'dataset_id': Path(str(request.get('dataset_root') or 'dataset')).name,
        'success': True,
        'records': records,
    }
    path = result_dir / 'predictions' / 'segmentation_predictions.json'
    write_json(path, payload)

    return path


def write_embedding_predictions(
    *,
    result_dir: Path,
    request: dict[str, Any],
    records: list[dict[str, Any]],
) -> Path:
    payload = {
        'schema_version': '0.1',
        'task': 'embedding',
        'model_id': str(request.get('model_id') or request.get('adapter') or 'external_embedder'),
        'dataset_id': Path(str(request.get('dataset_root') or 'dataset')).name,
        'success': True,
        'records': records,
    }
    path = result_dir / 'predictions' / 'embedding_predictions.json'
    write_json(path, payload)

    return path


def now_seconds() -> float:
    return time.perf_counter()
