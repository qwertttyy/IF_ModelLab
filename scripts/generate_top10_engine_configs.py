from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from ironflow_exp.engine.configs import (
    EngineCodeConfig,
    EngineOutputConfig,
    EngineRepositoryConfig,
    EngineRuntimeConfig,
    EngineTrainConfig,
)
from ironflow_exp.engine.configs.config_loader import EngineConfigLoader
from ironflow_exp.engine.ui.native_params import load_native_param_overrides_file
from ironflow_exp.engine.ui.recommended_experiments import (
    build_recommended_combination_config,
    load_recommended_model_matrix,
)


DEFAULT_COLLECT_PATTERNS = [
    'train.log',
    'metrics.csv',
    'metrics.json',
    'status.marker',
    'summary.md',
    'experiment.json',
    'artifacts.json',
    'timings.csv',
    'task_results.json',
    'tasks/',
]


def main() -> int:
    args = _parse_args()
    project_dir = args.project_dir.resolve()
    loader = EngineConfigLoader()
    base_data = loader.load_dict(project_dir / args.base_config)
    base_config = loader.from_dict(base_data)
    native_params = load_native_param_overrides_file(args.native_params, project_dir=project_dir)
    matrix = load_recommended_model_matrix(project_dir=project_dir)
    output_dir = project_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for combination in sorted(matrix.combinations, key=lambda item: item.priority):
        config = _build_locked_config(
            base_config=base_config,
            combination=combination,
            args=args,
            native_params=native_params,
        )
        filename = f'{combination.priority:02d}_{_safe_filename(combination.combination_id)}.yaml'
        output_path = output_dir / filename
        output_path.write_text(
            yaml.safe_dump(loader.to_dict(config), sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )
        generated.append(output_path)

    manifest_path = output_dir / 'MANIFEST.md'
    manifest_path.write_text(
        _manifest(
            paths=generated,
            project_dir=project_dir,
            dinov3_architecture_smoke=args.dinov3_architecture_smoke,
            use_local_pretrained_cache=args.use_local_pretrained_cache,
        ),
        encoding='utf-8',
    )

    if args.include_standalone:
        generated_standalone: dict[str, list[Path]] = {}
        for combination in sorted(matrix.standalone_experiments, key=lambda item: (item.experiment_type, item.priority)):
            folder = _standalone_output_dir(project_dir=project_dir, args=args, experiment_type=combination.experiment_type)
            folder.mkdir(parents=True, exist_ok=True)
            config = _build_locked_config(
                base_config=base_config,
                combination=combination,
                args=args,
                native_params=native_params,
            )
            filename = f'{combination.priority:02d}_{_safe_filename(combination.combination_id)}.yaml'
            output_path = folder / filename
            output_path.write_text(
                yaml.safe_dump(loader.to_dict(config), sort_keys=False, allow_unicode=True),
                encoding='utf-8',
            )
            generated_standalone.setdefault(combination.experiment_type, []).append(output_path)
            print(_display_path(path=output_path, project_dir=project_dir))

        for experiment_type, paths in sorted(generated_standalone.items()):
            manifest = _standalone_output_dir(project_dir=project_dir, args=args, experiment_type=experiment_type) / 'MANIFEST.md'
            manifest.write_text(
                _manifest(
                    paths=paths,
                    project_dir=project_dir,
                    dinov3_architecture_smoke=args.dinov3_architecture_smoke,
                    use_local_pretrained_cache=args.use_local_pretrained_cache,
                    title=f'{experiment_type.replace("_", " ").title()} Balanced GPU Configs',
                ),
                encoding='utf-8',
            )
            print(_display_path(path=manifest, project_dir=project_dir))
    for path in generated:
        print(_display_path(path=path, project_dir=project_dir))
    print(_display_path(path=manifest_path, project_dir=project_dir))

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate locked Top10 balanced GPU engine configs.')
    parser.add_argument('--project-dir', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--base-config', default='configs/engine/ssh_gpu_mock_smoke_template.yaml')
    parser.add_argument('--native-params', default='configs/engine/native_params/rtx5090_balanced_defaults.yaml')
    parser.add_argument('--output-dir', default='configs/engine/top10_balanced')
    parser.add_argument('--include-standalone', action='store_true')
    parser.add_argument('--detector-output-dir', default='configs/engine/detector_only_balanced')
    parser.add_argument('--classifier-output-dir', default='configs/engine/classifier_only_balanced')
    parser.add_argument(
        '--detection-dataset',
        default='../../../runs/user_datasets/tank14_prepared_v20260629/combined_model_name_detection/detection',
    )
    parser.add_argument(
        '--classification-crops',
        default='../../../runs/user_datasets/tank14_prepared_v20260629/combined_model_name_classification/crops',
    )
    parser.add_argument('--test-detection-dataset', default=None)
    parser.add_argument('--test-classification-crops', default=None)
    parser.add_argument('--remote-python-executable', default='auto')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--allow-pretrained-download', action='store_true', default=True)
    parser.add_argument(
        '--use-local-pretrained-cache',
        action='store_true',
        help='Point supported YOLO/Torchvision/timm tasks at models/checkpoints/pretrained and disable downloads for those tasks.',
    )
    parser.add_argument(
        '--dinov3-architecture-smoke',
        action='store_true',
        help='Set DINOv3 embedding tasks to dino_pretrained=false for wrapper execution smoke only.',
    )

    return parser.parse_args()


def _build_locked_config(
    *,
    base_config: Any,
    combination: Any,
    args: argparse.Namespace,
    native_params: dict[str, Any] | None,
) -> Any:
    config = build_recommended_combination_config(
        base_config=base_config,
        combination=combination,
        detection_dataset_path=args.detection_dataset,
        classification_crop_dataset_path=args.classification_crops,
        test_detection_dataset_path=args.test_detection_dataset,
        test_classification_crop_dataset_path=args.test_classification_crops,
        execution_mode='train',
        external_command_mode='native',
        native_param_overrides=native_params,
    )
    return _apply_team_gpu_defaults(
        config=config,
        max_seconds=args.max_seconds,
        remote_python_executable=args.remote_python_executable,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        allow_pretrained_download=args.allow_pretrained_download,
        dinov3_architecture_smoke=args.dinov3_architecture_smoke,
        use_local_pretrained_cache=args.use_local_pretrained_cache,
    )


def _standalone_output_dir(*, project_dir: Path, args: argparse.Namespace, experiment_type: str) -> Path:
    if experiment_type == 'detector_only':
        return project_dir / args.detector_output_dir
    if experiment_type == 'classifier_only':
        return project_dir / args.classifier_output_dir

    return project_dir / 'configs' / 'engine' / f'{_safe_filename(experiment_type)}_balanced'


def _apply_team_gpu_defaults(
    *,
    config: Any,
    max_seconds: int,
    remote_python_executable: str,
    epochs: int | None,
    batch_size: int | None,
    image_size: int | None,
    allow_pretrained_download: bool,
    dinov3_architecture_smoke: bool,
    use_local_pretrained_cache: bool,
) -> Any:
    runtime = EngineRuntimeConfig(
        runner='ssh',
        workspace='../../../runs/w',
        experiment_root='../../../runs/e',
        python_executable='python',
        remote_python_executable=remote_python_executable,
    )
    code = EngineCodeConfig(
        entrypoint='../../../src/ironflow_exp/engine/remote_task_adapter.py',
        working_dir='../../..',
        args=[],
        package_include=config.code.package_include,
        package_exclude=config.code.package_exclude,
    )
    output = EngineOutputConfig(
        log_file=config.output.log_file,
        metrics_file=config.output.metrics_file,
        checkpoint_file=config.output.checkpoint_file,
        status_file=config.output.status_file,
        summary_file=config.output.summary_file,
        save_json=True,
        save_csv=True,
        save_summary=True,
        save_checkpoints=True,
        save_previews=False,
        artifact_collection_mode='light',
        collect_patterns=DEFAULT_COLLECT_PATTERNS,
    )
    tasks = []
    for task in config.tasks:
        params = dict(task.params)
        if task.task_type in {'detection', 'classification', 'segmentation', 'embedding'}:
            params['allow_pretrained_download'] = allow_pretrained_download
        if use_local_pretrained_cache and not params.get('checkpoint_from_task_id'):
            local_checkpoint = _local_pretrained_checkpoint(task=task)
            if local_checkpoint is not None:
                params['checkpoint'] = local_checkpoint
                params['allow_pretrained_download'] = False
        if dinov3_architecture_smoke and task.adapter == 'dinov3_embedding':
            params['dino_pretrained'] = False
            params['pretrained_smoke_note'] = 'DINOv3 architecture smoke only; set dino_pretrained=true and provide weights for real experiments.'
        if (
            task.task_type == 'detection'
            and task.adapter == 'ultralytics_yolo'
            and params.get('execution_mode') == 'inference_smoke'
        ):
            params['allow_synthetic_smoke_detection'] = True
            params.setdefault('fallback_class_id', 'object')
        if params.get('execution_mode') == 'inference_smoke' and task.task_type in {
            'detection',
            'classification',
            'segmentation',
            'embedding',
        }:
            params.setdefault('prediction_split', 'test')
        if params.get('execution_mode') == 'train':
            if epochs is not None:
                params['epochs'] = epochs
            if batch_size is not None:
                params['batch_size'] = batch_size
            if task.task_type == 'detection' and image_size is not None:
                params['image_size'] = image_size
            if task.task_type == 'classification':
                params.setdefault('image_size', 224)
        elif task.task_type == 'classification':
            for train_only_param in ('learning_rate', 'scheduler', 'min_learning_rate', 'eta_min'):
                params.pop(train_only_param, None)
        tasks.append(replace(task, params=params))

    tags = sorted({
        tag
        for tag in config.experiment.tags
        if tag not in {'gpu-smoke', 'mock'}
    } | {'gpu-balanced'})

    return replace(
        config,
        experiment=replace(config.experiment, tags=tags),
        runtime=runtime,
        code=code,
        train=EngineTrainConfig(args=[], env={}, max_seconds=max_seconds, fail_fast=True),
        output=output,
        repository=EngineRepositoryConfig(
            sqlite_path=f'../../../runs/{_safe_filename(config.experiment.name)}_experiments.sqlite3',
        ),
        tasks=tasks,
    )


def _local_pretrained_checkpoint(*, task: Any) -> str | None:
    model_id = task.model_id
    if task.adapter == 'ultralytics_yolo' and model_id in {
        'yolo11n',
        'yolo11s',
        'yolo12n',
        'yolo12s',
        'yolo26n',
        'yolo26s',
        'yolov8n',
    }:
        return f'models/checkpoints/pretrained/ultralytics/{model_id}.pt'
    if task.adapter == 'ultralytics_yolo_classifier' and model_id in {'yolo26n-cls'}:
        return f'models/checkpoints/pretrained/ultralytics/{model_id}.pt'
    if task.adapter == 'rt_detr_detection' and model_id == 'rt_detr':
        return 'models/checkpoints/pretrained/ultralytics/rtdetr-l.pt'
    if task.adapter == 'rf_detr_detection' and model_id == 'rf_detr':
        return 'models/checkpoints/pretrained/rf_detr/rf_detr_base.pth'
    if task.adapter == 'd_fine_detection' and model_id == 'd_fine':
        return 'models/checkpoints/pretrained/d_fine/dfine_hgnetv2_n_coco.pth'
    if task.adapter == 'torchvision_classifier' and model_id in {
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'efficientnet_b0',
        'efficientnet_b3',
        'efficientnet_v2_s',
        'resnet50',
        'resnext50_32x4d',
    }:
        return f'models/checkpoints/pretrained/torchvision/{model_id}_imagenet.pt'
    if task.adapter == 'timm_classifier' and model_id in {
        'convnext_v2_tiny',
        'convnext_small',
        'swin_tiny',
    }:
        return f'models/checkpoints/pretrained/timm/{model_id}_pretrained.pt'

    return None


def _manifest(
    *,
    paths: list[Path],
    project_dir: Path,
    dinov3_architecture_smoke: bool,
    use_local_pretrained_cache: bool,
    title: str = 'Top10 Balanced GPU Configs',
) -> str:
    lines = [
        f'# {title}',
        '',
        'Generated by `scripts/generate_top10_engine_configs.py` from `configs/experiments/recommended_model_combinations.yaml`.',
        '',
        'Default profile:',
        '',
        '- Runtime: SSH GPU',
        '- Dataset: `imported_20260617` detection dataset and classification crop dataset',
        '- External wrappers: native mode',
        '- Training profile: RTX 5090 32GB balanced defaults from native params',
        '- Detector defaults: YOLO batch 32, transformer-style detectors batch 8, 20 epochs',
        '- Classifier defaults: batch 64, 25 epochs',
        '- Artifact collection: light',
        f'- DINOv3 mode: {"architecture smoke (not pretrained)" if dinov3_architecture_smoke else "pretrained required/default"}',
        f'- Local pretrained cache: {"enabled for cached YOLO/Torchvision/timm tasks" if use_local_pretrained_cache else "not forced"}',
        '',
        'Configs:',
        '',
    ]
    for path in paths:
        lines.append(f'- `{_display_path(path=path, project_dir=project_dir)}`')
    lines.append('')

    return '\n'.join(lines)


def _display_path(*, path: Path, project_dir: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_filename(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value.lower()).strip('_')


if __name__ == '__main__':
    raise SystemExit(main())
