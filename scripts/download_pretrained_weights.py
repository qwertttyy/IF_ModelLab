"""Download/cache core pretrained weights for IronFlow experiments.

This script intentionally uses each library's official weight-loading path.
YOLO weights are copied into ``models/checkpoints/pretrained/ultralytics`` when
Ultralytics exposes a local downloaded file. Torchvision weights are cached by
Torch/PyTorch and also exported as state_dict checkpoints for explicit GUI use.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import shutil
from typing import Any


YOLO_MODEL_IDS = ('yolo11n', 'yolo12n', 'yolo26n', 'yolo11s', 'yolo12s', 'yolo26s', 'yolov8n')
YOLO_CLASSIFIER_MODEL_IDS = ('yolo26n-cls',)
ULTRALYTICS_DETR_MODEL_IDS = ('rtdetr-l',)
MIN_YOLO_WEIGHT_BYTES = 1_000_000
TORCHVISION_MODEL_IDS = (
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'efficientnet_b0',
    'efficientnet_b3',
    'efficientnet_v2_s',
    'resnet50',
    'resnext50_32x4d',
)
TIMM_MODEL_IDS = ('convnext_v2_tiny', 'convnext_small', 'swin_tiny')
TIMM_MODEL_NAMES = {
    'convnext_v2_tiny': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'convnext_small': 'convnext_small.fb_in22k_ft_in1k',
    'swin_tiny': 'swin_tiny_patch4_window7_224',
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        default='models/checkpoints/pretrained',
        help='Directory for explicit checkpoint copies relative to the current working directory.',
    )
    parser.add_argument(
        '--include-reference-yolov8',
        action='store_true',
        help='Also download YOLOv8n as a mature reference baseline.',
    )
    parser.add_argument(
        '--include-timm',
        action='store_true',
        help='Also download timm models when timm is installed.',
    )
    parser.add_argument(
        '--include-rtdetr',
        action='store_true',
        help='Also download Ultralytics RT-DETR pretrained weights.',
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only report which expected checkpoint files already exist; do not import model libraries or download.',
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    yolo_ids = YOLO_MODEL_IDS if args.include_reference_yolov8 else tuple(
        model_id for model_id in YOLO_MODEL_IDS if model_id != 'yolov8n'
    )
    if args.check_only:
        results = _check_cached_weights(
            output_root=output_root,
            yolo_ids=yolo_ids,
            include_timm=args.include_timm,
            include_rtdetr=args.include_rtdetr,
        )
        manifest_path = output_root / 'pretrained_weight_cache_check.json'
        manifest_path.write_text(
            json.dumps(
                {'schema_version': '0.1', 'weights': _manifest_results(results, project_root=project_root)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        print(
            json.dumps(
                {'output_root': str(output_root), 'manifest': str(manifest_path), 'weights': results},
                indent=2,
            ),
        )
        return 0 if all(bool(result.get('ok')) for result in results) else 1

    for model_id in yolo_ids:
        results.append(_download_yolo(model_id=model_id, output_root=output_root))
    for model_id in YOLO_CLASSIFIER_MODEL_IDS:
        results.append(_download_yolo(model_id=model_id, output_root=output_root))

    if args.include_rtdetr:
        for model_id in ULTRALYTICS_DETR_MODEL_IDS:
            results.append(_download_ultralytics_rtdetr(model_id=model_id, output_root=output_root))

    for model_id in TORCHVISION_MODEL_IDS:
        results.append(_download_torchvision(model_id=model_id, output_root=output_root))

    if args.include_timm:
        for model_id in TIMM_MODEL_IDS:
            results.append(_download_timm(model_id=model_id, output_root=output_root))

    manifest_path = output_root / 'pretrained_weight_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {'schema_version': '0.1', 'weights': _manifest_results(results, project_root=project_root)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    print(json.dumps({'output_root': str(output_root), 'manifest': str(manifest_path), 'weights': results}, indent=2))

    return 0 if all(bool(result.get('ok')) for result in results) else 1


def _check_cached_weights(
    *,
    output_root: Path,
    yolo_ids: tuple[str, ...],
    include_timm: bool,
    include_rtdetr: bool,
) -> list[dict[str, object]]:
    expected_paths: list[tuple[str, str, Path]] = []
    expected_paths.extend(
        ('ultralytics_yolo', model_id, output_root / 'ultralytics' / f'{model_id}.pt')
        for model_id in yolo_ids
    )
    expected_paths.extend(
        ('ultralytics_yolo_classifier', model_id, output_root / 'ultralytics' / f'{model_id}.pt')
        for model_id in YOLO_CLASSIFIER_MODEL_IDS
    )
    if include_rtdetr:
        expected_paths.extend(
            ('ultralytics_rtdetr', model_id, output_root / 'ultralytics' / f'{model_id}.pt')
            for model_id in ULTRALYTICS_DETR_MODEL_IDS
        )
    expected_paths.extend(
        ('torchvision', model_id, output_root / 'torchvision' / f'{model_id}_imagenet.pt')
        for model_id in TORCHVISION_MODEL_IDS
    )
    if include_timm:
        expected_paths.extend(
            ('timm', model_id, output_root / 'timm' / f'{model_id}_pretrained.pt')
            for model_id in TIMM_MODEL_IDS
        )

    results: list[dict[str, object]] = []
    for family, model_id, path in expected_paths:
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        results.append(
            {
                'ok': exists and size_bytes > 0,
                'family': family,
                'model_id': model_id,
                'checkpoint_path': str(path),
                'exists': exists,
                'size_bytes': size_bytes,
            },
        )

    return results


def _download_yolo(*, model_id: str, output_root: Path) -> dict[str, object]:
    config_dir = output_root / 'ultralytics_config'
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    try:
        from ultralytics import YOLO
    except Exception as error:
        return _failure(model_id=model_id, family='ultralytics_yolo', error=error)

    target_dir = output_root / 'ultralytics'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f'{model_id}.pt'
    try:
        _remove_invalid_yolo_file(target_path)
        original_cwd = Path.cwd()
        os.chdir(target_dir)
        try:
            model = YOLO(f'{model_id}.pt')
            source = _yolo_weight_path(model=model, base_dir=target_dir)
        finally:
            os.chdir(original_cwd)
        copied_path = None
        if source is not None and source.exists():
            copied_path = target_path
            if source.resolve() != copied_path.resolve():
                shutil.copy2(source, copied_path)
        if copied_path is None and target_path.exists():
            copied_path = target_path
        if copied_path is None:
            raise FileNotFoundError(f'YOLO weight file was not created: {target_path}')
        if copied_path.stat().st_size < MIN_YOLO_WEIGHT_BYTES:
            raise RuntimeError(
                f'YOLO weight file is unexpectedly small: {copied_path} '
                f'({copied_path.stat().st_size} bytes)',
            )
        return {
            'ok': True,
            'family': 'ultralytics_yolo',
            'model_id': model_id,
            'checkpoint_path': None if copied_path is None else str(copied_path),
            'library_source_path': None if source is None else str(source),
        }
    except Exception as error:
        return _failure(model_id=model_id, family='ultralytics_yolo', error=error)


def _download_ultralytics_rtdetr(*, model_id: str, output_root: Path) -> dict[str, object]:
    config_dir = output_root / 'ultralytics_config'
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    try:
        from ultralytics import RTDETR
    except Exception as error:
        return _failure(model_id=model_id, family='ultralytics_rtdetr', error=error)

    target_dir = output_root / 'ultralytics'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f'{model_id}.pt'
    try:
        _remove_invalid_yolo_file(target_path)
        original_cwd = Path.cwd()
        os.chdir(target_dir)
        try:
            model = RTDETR(f'{model_id}.pt')
            source = _yolo_weight_path(model=model, base_dir=target_dir)
        finally:
            os.chdir(original_cwd)
        copied_path = None
        if source is not None and source.exists():
            copied_path = target_path
            if source.resolve() != copied_path.resolve():
                shutil.copy2(source, copied_path)
        if copied_path is None and target_path.exists():
            copied_path = target_path
        if copied_path is None:
            raise FileNotFoundError(f'RT-DETR weight file was not created: {target_path}')
        if copied_path.stat().st_size < MIN_YOLO_WEIGHT_BYTES:
            raise RuntimeError(
                f'RT-DETR weight file is unexpectedly small: {copied_path} '
                f'({copied_path.stat().st_size} bytes)',
            )
        return {
            'ok': True,
            'family': 'ultralytics_rtdetr',
            'model_id': model_id,
            'checkpoint_path': str(copied_path),
            'library_source_path': None if source is None else str(source),
        }
    except Exception as error:
        return _failure(model_id=model_id, family='ultralytics_rtdetr', error=error)


def _remove_invalid_yolo_file(path: Path) -> None:
    if path.exists() and path.stat().st_size < MIN_YOLO_WEIGHT_BYTES:
        path.unlink()


def _yolo_weight_path(*, model: object, base_dir: Path) -> Path | None:
    candidate = getattr(model, 'ckpt_path', None)
    if candidate:
        path = Path(str(candidate)).expanduser()
        return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
    model_obj = getattr(model, 'model', None)
    candidate = getattr(model_obj, 'pt_path', None)
    if candidate:
        path = Path(str(candidate)).expanduser()
        return path.resolve() if path.is_absolute() else (base_dir / path).resolve()

    return None


def _download_torchvision(*, model_id: str, output_root: Path) -> dict[str, object]:
    try:
        import torch
        import torchvision.models as models
    except Exception as error:
        return _failure(model_id=model_id, family='torchvision', error=error)

    specs = {
        'mobilenet_v3_small': ('mobilenet_v3_small', 'MobileNet_V3_Small_Weights'),
        'mobilenet_v3_large': ('mobilenet_v3_large', 'MobileNet_V3_Large_Weights'),
        'efficientnet_b0': ('efficientnet_b0', 'EfficientNet_B0_Weights'),
        'efficientnet_b3': ('efficientnet_b3', 'EfficientNet_B3_Weights'),
        'efficientnet_v2_s': ('efficientnet_v2_s', 'EfficientNet_V2_S_Weights'),
        'resnet50': ('resnet50', 'ResNet50_Weights'),
        'resnext50_32x4d': ('resnext50_32x4d', 'ResNeXt50_32X4D_Weights'),
    }
    builder_name, weights_name = specs[model_id]
    target_dir = output_root / 'torchvision'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f'{model_id}_imagenet.pt'
    try:
        weights_enum = getattr(models, weights_name)
        weights = weights_enum.DEFAULT
        builder = getattr(models, builder_name)
        model = builder(weights=weights)
        torch.save(
            {
                'model_id': model_id,
                'source': 'torchvision',
                'weights': str(weights),
                'state_dict': model.state_dict(),
            },
            target_path,
        )
        return {
            'ok': True,
            'family': 'torchvision',
            'model_id': model_id,
            'checkpoint_path': str(target_path),
            'weights': str(weights),
        }
    except Exception as error:
        return _failure(model_id=model_id, family='torchvision', error=error)


def _download_timm(*, model_id: str, output_root: Path) -> dict[str, object]:
    try:
        import timm
        import torch
    except Exception as error:
        return _failure(model_id=model_id, family='timm', error=error)

    target_dir = output_root / 'timm'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f'{model_id}_pretrained.pt'
    model_name = TIMM_MODEL_NAMES[model_id]
    try:
        model = timm.create_model(model_name, pretrained=True)
        torch.save(
            {
                'model_id': model_id,
                'source': 'timm',
                'timm_model_name': model_name,
                'state_dict': model.state_dict(),
            },
            target_path,
        )
        return {
            'ok': True,
            'family': 'timm',
            'model_id': model_id,
            'checkpoint_path': str(target_path),
            'timm_model_name': model_name,
        }
    except Exception as error:
        return _failure(model_id=model_id, family='timm', error=error)


def _failure(*, model_id: str, family: str, error: BaseException) -> dict[str, Any]:
    return {
        'ok': False,
        'family': family,
        'model_id': model_id,
        'error_type': type(error).__name__,
        'error': str(error),
    }


def _manifest_results(results: list[dict[str, object]], *, project_root: Path) -> list[dict[str, object]]:
    return [
        {
            key: _manifest_value(value=value, project_root=project_root)
            for key, value in result.items()
        }
        for result in results
    ]


def _manifest_value(*, value: object, project_root: Path) -> object:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return value


if __name__ == '__main__':
    raise SystemExit(main())
