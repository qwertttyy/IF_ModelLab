from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRS = (
    PROJECT_ROOT / 'configs' / 'engine' / 'top10_balanced',
    PROJECT_ROOT / 'configs' / 'engine' / 'detector_only_balanced',
    PROJECT_ROOT / 'configs' / 'engine' / 'classifier_only_balanced',
)
CACHE_ROOT = PROJECT_ROOT / 'models' / 'checkpoints' / 'pretrained'
CACHE_CHECK = CACHE_ROOT / 'pretrained_weight_cache_check.json'

CACHED_CHECKPOINTS = {
    ('ultralytics_yolo', 'yolo11n'): CACHE_ROOT / 'ultralytics' / 'yolo11n.pt',
    ('ultralytics_yolo', 'yolo11s'): CACHE_ROOT / 'ultralytics' / 'yolo11s.pt',
    ('ultralytics_yolo', 'yolo12n'): CACHE_ROOT / 'ultralytics' / 'yolo12n.pt',
    ('ultralytics_yolo', 'yolo12s'): CACHE_ROOT / 'ultralytics' / 'yolo12s.pt',
    ('ultralytics_yolo', 'yolo26n'): CACHE_ROOT / 'ultralytics' / 'yolo26n.pt',
    ('ultralytics_yolo', 'yolo26s'): CACHE_ROOT / 'ultralytics' / 'yolo26s.pt',
    ('ultralytics_yolo', 'yolov8n'): CACHE_ROOT / 'ultralytics' / 'yolov8n.pt',
    ('ultralytics_yolo_classifier', 'yolo26n-cls'): CACHE_ROOT / 'ultralytics' / 'yolo26n-cls.pt',
    ('rt_detr_detection', 'rt_detr'): CACHE_ROOT / 'ultralytics' / 'rtdetr-l.pt',
    ('rf_detr_detection', 'rf_detr'): CACHE_ROOT / 'rf_detr' / 'rf_detr_base.pth',
    ('d_fine_detection', 'd_fine'): CACHE_ROOT / 'd_fine' / 'dfine_hgnetv2_n_coco.pth',
    ('torchvision_classifier', 'mobilenet_v3_small'): CACHE_ROOT / 'torchvision' / 'mobilenet_v3_small_imagenet.pt',
    ('torchvision_classifier', 'mobilenet_v3_large'): CACHE_ROOT / 'torchvision' / 'mobilenet_v3_large_imagenet.pt',
    ('torchvision_classifier', 'efficientnet_b0'): CACHE_ROOT / 'torchvision' / 'efficientnet_b0_imagenet.pt',
    ('torchvision_classifier', 'efficientnet_b3'): CACHE_ROOT / 'torchvision' / 'efficientnet_b3_imagenet.pt',
    ('torchvision_classifier', 'efficientnet_v2_s'): CACHE_ROOT / 'torchvision' / 'efficientnet_v2_s_imagenet.pt',
    ('torchvision_classifier', 'resnet50'): CACHE_ROOT / 'torchvision' / 'resnet50_imagenet.pt',
    ('torchvision_classifier', 'resnext50_32x4d'): CACHE_ROOT / 'torchvision' / 'resnext50_32x4d_imagenet.pt',
    ('timm_classifier', 'convnext_v2_tiny'): CACHE_ROOT / 'timm' / 'convnext_v2_tiny_pretrained.pt',
    ('timm_classifier', 'convnext_small'): CACHE_ROOT / 'timm' / 'convnext_small_pretrained.pt',
    ('timm_classifier', 'swin_tiny'): CACHE_ROOT / 'timm' / 'swin_tiny_pretrained.pt',
}

FOUNDATION_RUNTIME_REQUIRED = {
    'rt_detr_detection': 'Ultralytics RT-DETR native runtime is required on the GPU server',
    'rt_detr_v2_detection': 'RT-DETRv2 Hugging Face/Transformers native runtime is required on the GPU server',
    'lw_detr_detection': 'LW-DETR Hugging Face/Transformers native runtime is required on the GPU server',
    'rf_detr_detection': 'RF-DETR native runtime is required on the GPU server',
    'd_fine_detection': 'D-FINE repo/native runtime is required on the GPU server',
    'grounding_dino_open_vocab_detection': 'GroundingDINO native wrapper/weights are not a supervised training path yet',
    'sam_promptable_segmentation': 'SAM2 HF checkpoint/cache on GPU server',
    'clip_embedding': 'OpenCLIP pretrained cache on GPU server',
    'dinov3_embedding': 'DINOv3 pretrained is deferred until a permitted weight file is available',
}


def main() -> int:
    configs = [
        config
        for config_dir in CONFIG_DIRS
        for config in sorted(config_dir.glob('*.yaml'))
    ]
    results = [_inspect_config(path) for path in configs]
    cache_results = _cache_results()
    blockers = [
        issue
        for result in results
        for issue in result['issues']
        if issue['severity'] == 'blocker'
    ]
    payload = {
        'schema_version': '0.1',
        'config_dirs': [str(path) for path in CONFIG_DIRS],
        'cache_check_manifest': str(CACHE_CHECK),
        'local_cache': cache_results,
        'configs': results,
        'summary': {
            'config_count': len(results),
            'blocker_count': len(blockers),
            'ready_for_supervised_training_count': len([
                result for result in results if result['supervised_training_ready']
            ]),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0 if not blockers else 1


def _inspect_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    tasks = data.get('tasks') if isinstance(data, dict) else []
    package_include = _package_include(data)
    issues: list[dict[str, str]] = []
    train_task_count = 0
    supervised_train_task_count = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_type = str(task.get('task_type') or '')
        adapter = str(task.get('adapter') or '')
        model_id = str(task.get('model_id') or '')
        params = task.get('params') if isinstance(task.get('params'), dict) else {}
        execution_mode = str(params.get('execution_mode') or '')
        if execution_mode == 'train':
            train_task_count += 1
            if task_type in {'detection', 'classification'}:
                supervised_train_task_count += 1
        cache_path = CACHED_CHECKPOINTS.get((adapter, model_id))
        if cache_path is not None and execution_mode == 'train':
            relative_cache_path = _project_relative(cache_path)
            configured_checkpoint = str(params.get('checkpoint') or '')
            if configured_checkpoint.replace('\\', '/') != relative_cache_path:
                issues.append({
                    'severity': 'blocker',
                    'task_id': str(task.get('id') or ''),
                    'message': f'cached pretrained checkpoint is not configured for {adapter}/{model_id}',
                })
            if relative_cache_path not in package_include:
                issues.append({
                    'severity': 'blocker',
                    'task_id': str(task.get('id') or ''),
                    'message': f'cached pretrained checkpoint is not package-included: {relative_cache_path}',
                })
            if not cache_path.exists() or cache_path.stat().st_size <= 0:
                issues.append({
                    'severity': 'blocker',
                    'task_id': str(task.get('id') or ''),
                    'message': f'cached pretrained checkpoint file is missing: {_project_relative(cache_path)}',
                })
        if adapter in FOUNDATION_RUNTIME_REQUIRED:
            severity = 'blocker' if adapter in {'grounding_dino_open_vocab_detection', 'dinov3_embedding'} else 'external_runtime'
            issues.append({
                'severity': severity,
                'task_id': str(task.get('id') or ''),
                'message': FOUNDATION_RUNTIME_REQUIRED[adapter],
            })

    return {
        'config': _project_relative(path),
        'train_task_count': train_task_count,
        'supervised_train_task_count': supervised_train_task_count,
        'supervised_training_ready': supervised_train_task_count > 0 and not any(
            issue['severity'] == 'blocker' for issue in issues
        ),
        'issues': issues,
    }


def _package_include(data: object) -> set[str]:
    if not isinstance(data, dict):
        return set()
    code = data.get('code')
    if not isinstance(code, dict):
        return set()
    include = code.get('package_include')
    if not isinstance(include, list):
        return set()

    return {
        str(item).replace('\\', '/')
        for item in include
        if isinstance(item, str)
    }


def _cache_results() -> dict[str, Any]:
    manifest_exists = CACHE_CHECK.exists()
    weights = []
    for (family, model_id), checkpoint_path in sorted(CACHED_CHECKPOINTS.items()):
        weights.append({
            'ok': checkpoint_path.exists() and checkpoint_path.stat().st_size > 0,
            'family': family,
            'model_id': model_id,
            'checkpoint_path': _project_relative(checkpoint_path),
            'exists': checkpoint_path.exists(),
            'size_bytes': checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        })

    return {'exists': manifest_exists, 'weights': weights}


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == '__main__':
    raise SystemExit(main())
