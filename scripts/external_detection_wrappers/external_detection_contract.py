"""Shared command-line contract helpers for external COCO detectors.

These helpers keep model-specific wrappers small. They validate the request
JSON produced by the IronFlow engine and can write lightweight contract
artifacts without importing GPU-only model packages.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_wrapper_common import (
    copy_if_exists,
    now_seconds,
    optional_import,
    request_param,
    result_dir_from_request,
    run_subprocess,
    write_contract_metadata,
    write_detection_predictions,
    write_error,
    write_json,
    write_metric_rows,
    write_runtime_metrics,
)


def run_contract_wrapper(*, model_family: str, adapter_key: str) -> int:
    parser = argparse.ArgumentParser(
        description=f'IronFlow external detection wrapper for {model_family}.',
    )
    parser.add_argument('--request', required=True, help='Path to runtime/external_detection_request.json.')
    parser.add_argument(
        '--dry-run-contract',
        action='store_true',
        help='Validate the request and write placeholder outputs without loading the model package.',
    )
    parser.add_argument(
        '--write-predictions',
        action='store_true',
        help='Write predictions/detection_predictions.json during contract dry-runs.',
    )
    args = parser.parse_args()

    request_path = Path(args.request).expanduser().resolve()
    request = _read_request(path=request_path)
    result_dir = result_dir_from_request(request)
    write_contract_metadata(
        result_dir=result_dir,
        request_path=request_path,
        request=request,
        model_family=model_family,
        adapter_key=adapter_key,
        native_ready=_native_supported(model_family=model_family),
    )

    if args.dry_run_contract:
        _write_metrics(result_dir=result_dir)
        _write_checkpoints(result_dir=result_dir, model_family=model_family)
        if args.write_predictions or request.get('execution_mode') == 'inference_smoke':
            _write_predictions(result_dir=result_dir, request=request)
        print(json.dumps({
            'ok': True,
            'mode': 'dry_run_contract',
            'model_family': model_family,
            'result_dir': str(result_dir),
        }, ensure_ascii=False))
        return 0

    return _run_native_detection(
        model_family=model_family,
        adapter_key=adapter_key,
        request=request,
        result_dir=result_dir,
    )


def _read_request(*, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'request JSON root must be an object: {path}')
    for key in ('adapter', 'model_id', 'execution_mode', 'result_dir', 'annotations', 'classes'):
        if key not in data:
            raise ValueError(f'request JSON is missing required key: {key}')
    if not isinstance(data['annotations'], dict):
        raise ValueError('request annotations must be an object')
    if not isinstance(data['classes'], list):
        raise ValueError('request classes must be a list')

    return data


def _write_contract_metadata(
    *,
    result_dir: Path,
    request_path: Path,
    request: dict[str, Any],
    model_family: str,
    adapter_key: str,
) -> None:
    payload = {
        'schema_version': '0.1',
        'model_family': model_family,
        'adapter_key': adapter_key,
        'request_json': str(request_path),
        'request_adapter': request.get('adapter'),
        'request_model_id': request.get('model_id'),
        'execution_mode': request.get('execution_mode'),
        'result_dir': str(result_dir),
    }
    external_dir = result_dir / 'external'
    external_dir.mkdir(parents=True, exist_ok=True)
    (external_dir / 'wrapper_contract.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _write_metrics(*, result_dir: Path) -> None:
    with (result_dir / 'metrics.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['epoch', 'train_loss', 'val_loss', 'map50', 'map50_95', 'lr'],
        )
        writer.writeheader()
        writer.writerow({
            'epoch': 1,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'map50': 0.0,
            'map50_95': 0.0,
            'lr': 0.0,
        })


def _write_checkpoints(*, result_dir: Path, model_family: str) -> None:
    checkpoint_dir = result_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'best.pt').write_bytes(f'ironflow {model_family} dry-run best\n'.encode('utf-8'))
    (checkpoint_dir / 'last.pt').write_bytes(f'ironflow {model_family} dry-run last\n'.encode('utf-8'))


def _write_predictions(*, result_dir: Path, request: dict[str, Any]) -> None:
    classes = request.get('classes') if isinstance(request.get('classes'), list) else []
    class_id = str(classes[0]) if classes else 'object'
    payload = {
        'schema_version': '0.1',
        'task': 'detection',
        'model_id': str(request.get('model_id') or request.get('adapter') or 'external_detector'),
        'dataset_id': Path(str(request.get('dataset_root') or 'dataset')).name,
        'success': True,
        'records': [
            {
                'image_id': 'dry_run_image_0001',
                'sample_id': 'dry_run_sample_0001',
                'prediction_id': 'dry_run_prediction_0001',
                'class_id': class_id,
                'score': 0.01,
                'bbox_xyxy': [0.0, 0.0, 1.0, 1.0],
                'image_width': 1,
                'image_height': 1,
            },
        ],
    }
    prediction_path = result_dir / 'predictions' / 'detection_predictions.json'
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _native_supported(*, model_family: str) -> bool:
    return model_family in {'rt_detr', 'rt_detr_v2', 'lw_detr', 'rf_detr', 'd_fine'}


def _run_native_detection(
    *,
    model_family: str,
    adapter_key: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    if model_family == 'rt_detr':
        return _run_native_rt_detr(adapter_key=adapter_key, model_family=model_family, request=request, result_dir=result_dir)
    if model_family in {'rt_detr_v2', 'lw_detr'}:
        return _run_native_hf_detr(adapter_key=adapter_key, model_family=model_family, request=request, result_dir=result_dir)
    if model_family == 'rf_detr':
        return _run_native_rf_detr(adapter_key=adapter_key, model_family=model_family, request=request, result_dir=result_dir)
    if model_family == 'd_fine':
        return _run_native_d_fine(adapter_key=adapter_key, model_family=model_family, request=request, result_dir=result_dir)

    return write_error(
        result_dir=result_dir,
        adapter_key=adapter_key,
        model_family=model_family,
        error_type='native_not_supported',
        message='native execution is not implemented for this detection wrapper yet',
    )


def _run_native_rt_detr(
    *,
    adapter_key: str,
    model_family: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    start = now_seconds()
    try:
        ultralytics = optional_import('ultralytics')
        model_ref = str(request_param(request, 'model_reference', request_param(request, 'checkpoint', 'rtdetr-l.pt')))
        model = ultralytics.RTDETR(model_ref)
        execution_mode = str(request.get('execution_mode') or '').strip().lower()
        if execution_mode == 'train':
            data_yaml = _write_ultralytics_data_yaml(result_dir=result_dir, request=request)
            train_kwargs = {
                'data': str(data_yaml),
                'epochs': _optional_int(request_param(request, 'epochs'), 1),
                'imgsz': _optional_int(request_param(request, 'image_size'), 640),
                'batch': _optional_int(request_param(request, 'batch_size'), -1),
                'project': str(result_dir / 'native' / 'ultralytics'),
                'name': 'train',
                'exist_ok': True,
            }
            patience = _optional_int(
                request_param(request, 'patience')
                if request_param(request, 'patience') is not None
                else request_param(request, 'early_stopping_patience'),
            )
            if patience is not None:
                train_kwargs['patience'] = patience
            learning_rate = _optional_float(request_param(request, 'learning_rate'))
            if learning_rate is not None:
                train_kwargs['lr0'] = learning_rate
            device = request_param(request, 'device')
            if device not in {None, ''}:
                train_kwargs['device'] = str(device)
            workers = _optional_int(
                request_param(request, 'workers')
                if request_param(request, 'workers') is not None
                else request_param(request, 'num_workers'),
            )
            if workers is not None:
                train_kwargs['workers'] = workers
            amp = request_param(request, 'amp')
            if amp is not None:
                train_kwargs['amp'] = _optional_bool(amp, True)
            model.train(**train_kwargs)
            weights_dir = result_dir / 'native' / 'ultralytics' / 'train' / 'weights'
            copy_if_exists(weights_dir / 'best.pt', result_dir / 'checkpoints' / 'best.pt')
            copy_if_exists(weights_dir / 'last.pt', result_dir / 'checkpoints' / 'last.pt')
            write_runtime_metrics(result_dir=result_dir, elapsed_seconds=now_seconds() - start)
            print(json.dumps({'ok': True, 'mode': 'native_train', 'model_family': model_family, 'result_dir': str(result_dir)}))
            return 0

        source = _native_detection_source(request=request)
        results = model.predict(
            source=str(source),
            conf=float(request_param(request, 'confidence_threshold', 0.25)),
            imgsz=_optional_int(request_param(request, 'image_size'), 640),
            save=False,
            verbose=False,
        )
        records = _ultralytics_results_to_records(results=results, request=request)
        write_detection_predictions(result_dir=result_dir, request=request, records=records)
        write_runtime_metrics(
            result_dir=result_dir,
            elapsed_seconds=now_seconds() - start,
            extra={'prediction_count': len(records), 'image_count': len(results)},
        )
        print(json.dumps({'ok': True, 'mode': 'native_inference', 'model_family': model_family, 'prediction_count': len(records)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_rt_detr_failed',
            message=str(exc),
        )


def _run_native_hf_detr(
    *,
    adapter_key: str,
    model_family: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    start = now_seconds()
    try:
        torch = optional_import('torch')
        image_module = optional_import('PIL.Image')
        transformers = optional_import('transformers')
        model_ref = _hf_detr_model_ref(model_family=model_family, request=request)
        image_processor, model = _build_hf_detr_model(
            transformers=transformers,
            model_family=model_family,
            model_ref=model_ref,
            request=request,
        )
        device = _torch_device(torch=torch, requested=request_param(request, 'device', 'cpu'))
        model.to(device)
        execution_mode = str(request.get('execution_mode') or '').strip().lower()
        device_is_cuda = str(device).startswith('cuda')
        amp_enabled = _optional_bool(request_param(request, 'amp'), device_is_cuda) and device_is_cuda
        if execution_mode == 'train':
            train_dataset = _HfCocoDetectionDataset(
                request=request,
                split='train',
                image_processor=image_processor,
                image_module=image_module,
            )
            if len(train_dataset) == 0:
                raise RuntimeError('HF DETR train requires at least one COCO train image')
            num_workers = _optional_int(request_param(request, 'num_workers'), 0) or 0
            data_loader_kwargs = {
                'batch_size': _optional_int(request_param(request, 'batch_size'), 1) or 1,
                'shuffle': True,
                'num_workers': num_workers,
                'collate_fn': _hf_detr_collate,
                'pin_memory': device_is_cuda,
            }
            if num_workers > 0:
                data_loader_kwargs['persistent_workers'] = True
            data_loader = torch.utils.data.DataLoader(
                train_dataset,
                **data_loader_kwargs,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=_optional_float(request_param(request, 'learning_rate'), 1e-5) or 1e-5,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            epochs = _optional_int(request_param(request, 'epochs'), 1) or 1
            best_loss: float | None = None
            rows: list[dict[str, Any]] = []
            checkpoint_dir = result_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                for batch in data_loader:
                    optimizer.zero_grad(set_to_none=True)
                    inputs = _hf_detr_batch_to_device(batch=batch, device=device)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(**inputs)
                        loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += float(loss.detach().cpu())
                    batch_count += 1
                mean_loss = epoch_loss / max(batch_count, 1)
                rows.append(
                    {
                        'epoch': epoch,
                        'train_loss': mean_loss,
                        'val_loss': '',
                        'map50': '',
                        'map50_95': '',
                        'lr': optimizer.param_groups[0].get('lr', ''),
                        'elapsed_seconds': f'{now_seconds() - start:.6f}',
                    },
                )
                payload = _hf_detr_checkpoint_payload(
                    model=model,
                    model_family=model_family,
                    model_ref=model_ref,
                    request=request,
                )
                torch.save(payload, checkpoint_dir / 'last.pt')
                if best_loss is None or mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(payload, checkpoint_dir / 'best.pt')
            write_metric_rows(
                result_dir,
                rows,
                fieldnames=['epoch', 'train_loss', 'val_loss', 'map50', 'map50_95', 'lr', 'elapsed_seconds'],
            )
            write_json(
                result_dir / 'external' / 'hf_detr_train.json',
                {
                    'schema_version': '0.1',
                    'model_family': model_family,
                    'hf_model_id': model_ref,
                    'image_count': len(train_dataset),
                    'epochs': epochs,
                    'best_train_loss': best_loss,
                    'elapsed_seconds': now_seconds() - start,
                },
            )
            print(json.dumps({'ok': True, 'mode': 'native_train', 'model_family': model_family, 'result_dir': str(result_dir)}))
            return 0

        checkpoint = request_param(request, 'checkpoint')
        if checkpoint:
            _load_hf_detr_checkpoint(torch=torch, model=model, checkpoint=Path(str(checkpoint)).expanduser())
        source = _native_detection_source(request=request)
        image_paths = _native_detection_image_sources(source=source)
        threshold = _optional_float(request_param(request, 'confidence_threshold'), 0.25) or 0.25
        top_k = _optional_int(request_param(request, 'top_k'), None)
        records: list[dict[str, Any]] = []
        model.eval()
        with torch.no_grad():
            for image_path in image_paths:
                with image_module.open(image_path) as image:
                    image_rgb = image.convert('RGB')
                    width, height = image_rgb.size
                    inputs = image_processor(images=image_rgb, return_tensors='pt')
                inputs = _hf_batch_feature_to_device(inputs, device=device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(**inputs)
                target_sizes = torch.tensor([(height, width)], device=device)
                processed = image_processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=threshold,
                )[0]
                records.extend(
                    _hf_detr_processed_to_records(
                        processed=processed,
                        request=request,
                        image_path=image_path,
                        image_width=width,
                        image_height=height,
                        top_k=top_k,
                    ),
                )
        prediction_path = write_detection_predictions(result_dir=result_dir, request=request, records=records)
        write_runtime_metrics(
            result_dir=result_dir,
            elapsed_seconds=now_seconds() - start,
            extra={'prediction_count': len(records), 'image_count': len(image_paths)},
        )
        write_json(
            result_dir / 'external' / 'hf_detr_prediction_export.json',
            {
                'schema_version': '0.1',
                'model_family': model_family,
                'hf_model_id': model_ref,
                'prediction_path': str(prediction_path),
                'image_count': len(image_paths),
                'record_count': len(records),
                'confidence_threshold': threshold,
                'top_k': top_k,
                'elapsed_seconds': now_seconds() - start,
            },
        )
        print(json.dumps({'ok': True, 'mode': 'native_inference', 'model_family': model_family, 'prediction_count': len(records)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_hf_detr_failed',
            message=str(exc),
        )


def _run_native_rf_detr(
    *,
    adapter_key: str,
    model_family: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    start = now_seconds()
    try:
        rfdetr = optional_import('rfdetr')
        model_class_name = str(
            request_param(
                request,
                'rf_detr_model_class',
                request_param(request, 'rf_detr_class', 'RFDETRMedium'),
            ),
        )
        model_class = getattr(rfdetr, model_class_name)
        model = _build_rf_detr_model(model_class=model_class, request=request)
        execution_mode = str(request.get('execution_mode') or '').strip().lower()
        if execution_mode == 'train':
            if not hasattr(model, 'train'):
                raise RuntimeError(f'{model_class_name} has no train method')
            dataset_dir = _prepare_rf_detr_dataset_dir(request=request, result_dir=result_dir)
            train_kwargs = {
                'dataset_dir': str(dataset_dir),
                'epochs': _optional_int(request_param(request, 'epochs'), 1),
                'output_dir': str(result_dir / 'native' / 'rf_detr'),
                **_rf_detr_weight_kwargs(callable_obj=model.train, request=request),
                **_accepted_explicit_kwargs(
                    callable_obj=model.train,
                    candidates=_rf_detr_train_option_candidates(request=request),
                ),
            }
            model.train(**train_kwargs)
            _copy_first_existing_checkpoint(
                candidates=list((result_dir / 'native' / 'rf_detr').rglob('*.pth')) + list((result_dir / 'native' / 'rf_detr').rglob('*.pt')),
                result_dir=result_dir,
            )
            write_runtime_metrics(result_dir=result_dir, elapsed_seconds=now_seconds() - start)
            print(json.dumps({'ok': True, 'mode': 'native_train', 'model_family': model_family, 'result_dir': str(result_dir)}))
            return 0

        source = _native_detection_source(request=request)
        records: list[dict[str, Any]] = []
        image_sources = _native_detection_image_sources(source=source)
        for image_source in image_sources:
            detections = model.predict(str(image_source), threshold=float(request_param(request, 'confidence_threshold', 0.25)))
            records.extend(_supervision_detections_to_records(detections=detections, request=request, source=image_source))
        write_detection_predictions(result_dir=result_dir, request=request, records=records)
        write_runtime_metrics(
            result_dir=result_dir,
            elapsed_seconds=now_seconds() - start,
            extra={'prediction_count': len(records), 'image_count': len(image_sources)},
        )
        print(json.dumps({'ok': True, 'mode': 'native_inference', 'model_family': model_family, 'prediction_count': len(records)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_rf_detr_failed',
            message=str(exc),
        )


def _run_native_d_fine(
    *,
    adapter_key: str,
    model_family: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    start = now_seconds()
    try:
        raw_repo_dir = request_param(request, 'd_fine_repo_dir')
        if raw_repo_dir in {None, ''}:
            raise RuntimeError('D-FINE native execution requires params.d_fine_repo_dir pointing to the cloned D-FINE repo')
        repo_dir = Path(str(raw_repo_dir)).expanduser()
        if not repo_dir.exists():
            raise RuntimeError('D-FINE native execution requires params.d_fine_repo_dir pointing to the cloned D-FINE repo')
        config_path = str(_prepare_d_fine_config(request=request, repo_dir=repo_dir, result_dir=result_dir))
        nproc = str(request_param(request, 'd_fine_nproc_per_node', 1))
        master_port = str(request_param(request, 'd_fine_master_port', 7777))
        python_executable = str(request_param(request, 'python_executable', sys.executable))
        execution_mode = str(request.get('execution_mode') or '').strip().lower()
        command = [
            python_executable,
            '-m',
            'torch.distributed.run',
            f'--master_port={master_port}',
            f'--nproc_per_node={nproc}',
            'train.py',
            '-c',
            config_path,
        ]
        if execution_mode == 'train':
            command.extend(['--use-amp', '--seed=0'])
            tuning_checkpoint = request_param(request, 'checkpoint')
            if tuning_checkpoint:
                command.extend(['-t', str(tuning_checkpoint)])
        else:
            checkpoint = request_param(request, 'checkpoint')
            if not checkpoint:
                raise RuntimeError('D-FINE inference requires params.checkpoint')
            command.extend(['--test-only', '-r', str(checkpoint)])

        output_dir = result_dir / 'native' / 'd_fine'
        completed = _run_d_fine_subprocess(
            command=command,
            cwd=repo_dir,
            timeout_seconds=_optional_int(request_param(request, 'external_timeout_seconds'), None),
            result_dir=result_dir,
            output_dir=output_dir,
            request=request,
            monitor_early_stopping=execution_mode == 'train',
        )
        if completed.returncode != 0:
            raise RuntimeError(f'D-FINE command failed with exit code {completed.returncode}')
        _copy_d_fine_outputs(repo_dir=repo_dir, result_dir=result_dir, request=request)
        if execution_mode != 'train' and not (result_dir / 'predictions' / 'detection_predictions.json').exists():
            checkpoint = request_param(request, 'checkpoint')
            if checkpoint:
                _write_d_fine_native_predictions(
                    repo_dir=repo_dir,
                    config_path=Path(config_path),
                    checkpoint_path=Path(str(checkpoint)).expanduser(),
                    result_dir=result_dir,
                    request=request,
                )
        write_runtime_metrics(result_dir=result_dir, elapsed_seconds=now_seconds() - start)
        print(json.dumps({'ok': True, 'mode': f'native_{execution_mode or "run"}', 'model_family': model_family, 'result_dir': str(result_dir)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_d_fine_failed',
            message=str(exc),
        )


def _build_rf_detr_model(*, model_class: Any, request: dict[str, Any]) -> Any:
    kwargs = _rf_detr_weight_kwargs(callable_obj=model_class, request=request)
    try:
        return model_class(**kwargs)
    except TypeError:
        if kwargs:
            return model_class()
        raise


def _rf_detr_weight_kwargs(*, callable_obj: Any, request: dict[str, Any]) -> dict[str, Any]:
    checkpoint = (
        request_param(request, 'rf_detr_pretrain_weights')
        or request_param(request, 'rf_detr_checkpoint')
        or request_param(request, 'checkpoint')
    )
    if checkpoint in {None, ''}:
        return {}
    checkpoint_value = str(checkpoint)
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    parameter_names = set(signature.parameters)
    if 'pretrain_weights' in parameter_names:
        return {'pretrain_weights': checkpoint_value}
    for legacy_name in ('pretrained_weights', 'checkpoint', 'weights'):
        if legacy_name in parameter_names:
            return {legacy_name: checkpoint_value}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        # Current RF-DETR constructors accept **kwargs but validate them with a
        # pydantic config. Passing aliases such as checkpoint/weights makes that
        # validation fail, so use the public RF-DETR config key only.
        return {'pretrain_weights': checkpoint_value}
    return {}


def _accepted_kwargs(*, callable_obj: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return candidates

    return {
        key: value
        for key, value in candidates.items()
        if key in signature.parameters
    }


def _accepted_explicit_kwargs(*, callable_obj: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return candidates

    return {
        key: value
        for key, value in candidates.items()
        if key in signature.parameters
    }


def _rf_detr_train_option_candidates(*, request: dict[str, Any]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    batch_size = _optional_int(request_param(request, 'batch_size'))
    if batch_size is not None:
        candidates['batch_size'] = batch_size
    num_workers = _optional_int(request_param(request, 'num_workers'))
    if num_workers is not None:
        candidates['num_workers'] = num_workers
    learning_rate = _optional_float(request_param(request, 'learning_rate'))
    if learning_rate is not None:
        candidates['learning_rate'] = learning_rate
        candidates['lr'] = learning_rate
    patience = _optional_int(
        request_param(request, 'patience')
        if request_param(request, 'patience') is not None
        else request_param(request, 'early_stopping_patience'),
    )
    if patience is not None:
        candidates['patience'] = patience
        candidates['early_stopping_patience'] = patience
    min_delta = _optional_float(request_param(request, 'early_stopping_min_delta'))
    if min_delta is not None:
        candidates['early_stopping_min_delta'] = min_delta

    return candidates


def _run_d_fine_subprocess(
    *,
    command: list[str],
    cwd: Path,
    timeout_seconds: int | None,
    result_dir: Path,
    output_dir: Path,
    request: dict[str, Any],
    monitor_early_stopping: bool,
) -> subprocess.CompletedProcess[str]:
    patience = _optional_int(
        request_param(request, 'early_stopping_patience')
        if request_param(request, 'early_stopping_patience') is not None
        else request_param(request, 'patience'),
    )
    if not monitor_early_stopping or patience is None:
        return run_subprocess(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            result_dir=result_dir,
        )

    min_delta = _optional_float(request_param(request, 'early_stopping_min_delta')) or 0.0
    external_dir = result_dir / 'external'
    external_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = external_dir / 'native_stdout.log'
    stderr_path = external_dir / 'native_stderr.log'
    monitor_path = external_dir / 'd_fine_early_stopping.json'
    log_path = output_dir / 'log.txt'
    start = now_seconds()
    best_metric = float('-inf')
    best_epoch: int | None = None
    epochs_without_improvement = 0
    seen_epochs: set[int] = set()
    stopped_epoch: int | None = None
    timed_out = False

    with stdout_path.open('w', encoding='utf-8') as stdout_file, stderr_path.open('w', encoding='utf-8') as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            if timeout_seconds is not None and now_seconds() - start > timeout_seconds:
                timed_out = True
                _terminate_process_group(process=process)
                break
            for epoch, metric in _d_fine_log_metrics(log_path=log_path):
                if epoch in seen_epochs:
                    continue
                seen_epochs.add(epoch)
                if metric > best_metric + min_delta:
                    best_metric = metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        stopped_epoch = epoch
                        _terminate_process_group(process=process)
                        break
            if stopped_epoch is not None:
                break
            time.sleep(5)
        return_code = process.wait()

    monitor_payload = {
        'enabled': True,
        'patience': patience,
        'min_delta': min_delta,
        'best_metric': None if best_metric == float('-inf') else best_metric,
        'best_epoch': best_epoch,
        'early_stopped': stopped_epoch is not None,
        'stopped_epoch': stopped_epoch,
        'timed_out': timed_out,
        'observed_epochs': sorted(seen_epochs),
    }
    write_json(monitor_path, monitor_payload)
    if stopped_epoch is not None:
        return_code = 0
    if timed_out:
        return_code = 124

    stdout = stdout_path.read_text(encoding='utf-8', errors='replace') if stdout_path.exists() else ''
    stderr = stderr_path.read_text(encoding='utf-8', errors='replace') if stderr_path.exists() else ''
    return subprocess.CompletedProcess(command, return_code, stdout=stdout, stderr=stderr)


def _d_fine_log_metrics(*, log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    metrics: list[tuple[int, float]] = []
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        text = line.strip()
        if not text.startswith('{'):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        epoch = _optional_int(payload.get('epoch'))
        values = payload.get('test_coco_eval_bbox')
        if epoch is None or not isinstance(values, list) or not values:
            continue
        try:
            metric = float(values[0])
        except (TypeError, ValueError):
            continue
        metrics.append((epoch, metric))

    return metrics


def _terminate_process_group(*, process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()


def _prepare_rf_detr_dataset_dir(*, request: dict[str, Any], result_dir: Path) -> Path:
    explicit = request_param(request, 'rf_detr_dataset_dir')
    auto_view = _optional_bool(request_param(request, 'rf_detr_auto_dataset_view', True), True)
    if explicit and not auto_view:
        return Path(str(explicit)).expanduser()

    dataset_root = Path(str(explicit or request.get('dataset_root') or request.get('coco_root') or '')).expanduser()
    if not str(dataset_root):
        raise RuntimeError('RF-DETR train requires params.rf_detr_dataset_dir or request dataset_root/coco_root')
    dataset_root = dataset_root.resolve()

    if _has_rf_detr_layout(dataset_root):
        return dataset_root

    view_dir = Path(str(request_param(request, 'rf_detr_dataset_view_dir', result_dir / 'runtime' / 'rf_detr_dataset_view'))).expanduser().resolve()
    view_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        'train': 'train',
        'val': 'valid',
        'valid': 'valid',
        'test': 'test',
    }
    for source_split, target_split in split_map.items():
        image_dir = dataset_root / 'images' / source_split
        label_dir = dataset_root / 'labels' / source_split
        if not image_dir.exists() or not label_dir.exists():
            continue
        _link_or_copy_dir(source=image_dir, target=view_dir / target_split / 'images')
        _link_or_copy_dir(source=label_dir, target=view_dir / target_split / 'labels')

    if not (view_dir / 'train' / 'images').exists():
        raise RuntimeError(f'RF-DETR dataset bridge could not find YOLO images/train under: {dataset_root}')
    if not (view_dir / 'valid' / 'images').exists():
        raise RuntimeError(f'RF-DETR dataset bridge could not find YOLO images/val or images/valid under: {dataset_root}')

    _write_rf_detr_data_yaml(dataset_dir=view_dir, request=request)
    write_json(
        result_dir / 'external' / 'rf_detr_dataset_bridge.json',
        {
            'source_dataset_root': str(dataset_root),
            'rf_detr_dataset_dir': str(view_dir),
            'layout': 'train/images + train/labels + valid/images + valid/labels',
        },
    )

    return view_dir


class _HfCocoDetectionDataset:
    def __init__(
        self,
        *,
        request: dict[str, Any],
        split: str,
        image_processor: Any,
        image_module: Any,
    ) -> None:
        self.request = request
        self.split = split
        self.image_processor = image_processor
        self.image_module = image_module
        annotations = request.get('annotations') if isinstance(request.get('annotations'), dict) else {}
        annotation_path = Path(str(annotations.get(split) or annotations.get('val') or annotations.get('test') or '')).expanduser()
        if not annotation_path.exists():
            raise RuntimeError(f'HF DETR {split} split requires an existing COCO annotation file: {annotation_path}')
        payload = json.loads(annotation_path.read_text(encoding='utf-8'))
        self.annotation_path = annotation_path
        self.images = [row for row in payload.get('images', []) if isinstance(row, dict)]
        annotation_rows = [row for row in payload.get('annotations', []) if isinstance(row, dict)]
        self.annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotation_rows:
            image_id = int(annotation.get('image_id'))
            self.annotations_by_image_id.setdefault(image_id, []).append(annotation)
        self.category_id_to_label = _hf_coco_category_id_to_label(request=request, coco_payload=payload)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_info = self.images[index]
        image_id = int(image_info.get('id', index))
        image_path = _hf_coco_image_path(
            request=self.request,
            image_info=image_info,
            annotation_path=self.annotation_path,
            split=self.split,
        )
        with self.image_module.open(image_path) as image:
            image_rgb = image.convert('RGB')
        annotations = [
            _hf_coco_annotation(annotation=annotation, category_id_to_label=self.category_id_to_label)
            for annotation in self.annotations_by_image_id.get(image_id, [])
            if _valid_coco_bbox(annotation.get('bbox'))
        ]
        encoding = self.image_processor(
            images=image_rgb,
            annotations={'image_id': image_id, 'annotations': annotations},
            return_tensors='pt',
        )
        item = {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': encoding['labels'][0],
        }
        if 'pixel_mask' in encoding:
            item['pixel_mask'] = encoding['pixel_mask'].squeeze(0)

        return item


def _hf_detr_model_ref(*, model_family: str, request: dict[str, Any]) -> str:
    explicit = (
        request_param(request, 'hf_model_id')
        or request_param(request, 'model_reference')
        or request_param(request, 'checkpoint_model_id')
    )
    if explicit:
        return str(explicit)
    if model_family == 'rt_detr_v2':
        return 'PekingU/rtdetr_v2_r18vd'
    if model_family == 'lw_detr':
        return 'AnnaZhang/lwdetr_small_60e_coco'

    raise RuntimeError(f'unknown HF DETR model family: {model_family}')


def _build_hf_detr_model(
    *,
    transformers: Any,
    model_family: str,
    model_ref: str,
    request: dict[str, Any],
) -> tuple[Any, Any]:
    classes = [str(value) for value in request.get('classes', [])] if isinstance(request.get('classes'), list) else []
    if not classes:
        raise RuntimeError('HF DETR native execution requires request.classes')
    id2label = {index: class_id for index, class_id in enumerate(classes)}
    label2id = {class_id: index for index, class_id in id2label.items()}
    if model_family == 'rt_detr_v2':
        processor_class = getattr(transformers, 'RTDetrImageProcessor')
        model_class = getattr(transformers, 'RTDetrV2ForObjectDetection')
    elif model_family == 'lw_detr':
        processor_class = getattr(transformers, 'AutoImageProcessor')
        model_class = getattr(transformers, 'LwDetrForObjectDetection')
    else:
        raise RuntimeError(f'unsupported HF DETR model family: {model_family}')
    image_processor = processor_class.from_pretrained(model_ref)
    model = model_class.from_pretrained(
        model_ref,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return image_processor, model


def _torch_device(*, torch: Any, requested: Any) -> Any:
    device = str(requested or 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    return torch.device(device)


def _hf_detr_batch_to_device(*, batch: dict[str, Any], device: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {'pixel_values': batch['pixel_values'].to(device)}
    if 'pixel_mask' in batch:
        payload['pixel_mask'] = batch['pixel_mask'].to(device)
    payload['labels'] = [
        {
            key: value.to(device) if hasattr(value, 'to') else value
            for key, value in labels.items()
        }
        for labels in batch['labels']
    ]

    return payload


def _hf_batch_feature_to_device(batch_feature: Any, *, device: Any) -> Any:
    if hasattr(batch_feature, 'to'):
        return batch_feature.to(device)

    return {
        key: value.to(device) if hasattr(value, 'to') else value
        for key, value in batch_feature.items()
    }


def _hf_detr_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = {
        'pixel_values': torch.stack([item['pixel_values'] for item in items]),
        'labels': [item['labels'] for item in items],
    }
    if all('pixel_mask' in item for item in items):
        batch['pixel_mask'] = torch.stack([item['pixel_mask'] for item in items])

    return batch


def _hf_detr_checkpoint_payload(*, model: Any, model_family: str, model_ref: str, request: dict[str, Any]) -> dict[str, Any]:
    classes = [str(value) for value in request.get('classes', [])] if isinstance(request.get('classes'), list) else []

    return {
        'schema_version': '0.1',
        'model_family': model_family,
        'hf_model_id': model_ref,
        'classes': classes,
        'state_dict': model.state_dict(),
    }


def _load_hf_detr_checkpoint(*, torch: Any, model: Any, checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise RuntimeError(f'HF DETR checkpoint not found: {checkpoint}')
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location='cpu')
    state_dict = payload.get('state_dict') if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise RuntimeError(f'HF DETR checkpoint does not contain a state_dict: {checkpoint}')
    model.load_state_dict(state_dict, strict=False)


def _hf_coco_category_id_to_label(*, request: dict[str, Any], coco_payload: dict[str, Any]) -> dict[int, int]:
    classes = [str(value) for value in request.get('classes', [])] if isinstance(request.get('classes'), list) else []
    label_by_name = {class_name: index for index, class_name in enumerate(classes)}
    categories = [row for row in coco_payload.get('categories', []) if isinstance(row, dict)]
    mapping: dict[int, int] = {}
    for category in categories:
        category_id = int(category.get('id'))
        category_name = str(category.get('name') or category_id)
        mapping[category_id] = label_by_name.get(category_name, len(mapping))
    if mapping:
        return mapping

    return {index: index for index in range(len(classes))}


def _hf_coco_annotation(*, annotation: dict[str, Any], category_id_to_label: dict[int, int]) -> dict[str, Any]:
    category_id = int(annotation.get('category_id', 0))
    bbox = [float(value) for value in annotation.get('bbox', [0, 0, 1, 1])]
    area = annotation.get('area')
    if area is None:
        area = max(0.0, bbox[2]) * max(0.0, bbox[3])

    return {
        'bbox': bbox,
        'category_id': category_id_to_label.get(category_id, category_id),
        'area': float(area),
        'iscrowd': int(annotation.get('iscrowd', 0)),
    }


def _valid_coco_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        _, _, width, height = [float(item) for item in value]
    except (TypeError, ValueError):
        return False

    return width > 0 and height > 0


def _hf_coco_image_path(
    *,
    request: dict[str, Any],
    image_info: dict[str, Any],
    annotation_path: Path,
    split: str,
) -> Path:
    file_name = str(image_info.get('file_name') or '')
    if not file_name:
        raise RuntimeError('COCO image record is missing file_name')
    path = Path(file_name).expanduser()
    if path.is_absolute() and path.exists():
        return path
    basename = path.name
    roots = [
        Path(str(request.get('image_root') or '')).expanduser(),
        Path(str(request.get('coco_root') or '')).expanduser(),
        Path(str(request.get('dataset_root') or '')).expanduser(),
        annotation_path.parents[1] if len(annotation_path.parents) > 1 else annotation_path.parent,
    ]
    candidates: list[Path] = []
    for root in roots:
        if not str(root):
            continue
        candidates.extend(
            [
                root / file_name,
                root / 'images' / file_name,
                root / 'images' / split / basename,
                root / split / basename,
                root / basename,
            ],
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(f'COCO image file not found for {file_name}; checked {len(candidates)} locations')


def _hf_detr_processed_to_records(
    *,
    processed: dict[str, Any],
    request: dict[str, Any],
    image_path: Path,
    image_width: int,
    image_height: int,
    top_k: int | None,
) -> list[dict[str, Any]]:
    classes = request.get('classes') if isinstance(request.get('classes'), list) else []
    scores = [float(value) for value in processed.get('scores', []).detach().cpu().tolist()]
    labels = [int(value) for value in processed.get('labels', []).detach().cpu().tolist()]
    boxes = processed.get('boxes', []).detach().cpu().tolist()
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    image_id = image_path.stem
    records: list[dict[str, Any]] = []
    for rank, index in enumerate(order):
        class_index = labels[index]
        records.append(
            {
                'image_id': image_id,
                'sample_id': image_id,
                'prediction_id': f'{image_id}_hf_detr_{rank:04d}',
                'class_id': str(classes[class_index]) if 0 <= class_index < len(classes) else str(class_index),
                'score': scores[index],
                'bbox_xyxy': _clamped_bbox_xyxy(
                    bbox=boxes[index],
                    image_width=image_width,
                    image_height=image_height,
                ),
                'image_width': image_width,
                'image_height': image_height,
            },
        )

    return records


def _has_rf_detr_layout(dataset_root: Path) -> bool:
    return (
        (dataset_root / 'train' / 'images').exists()
        and (dataset_root / 'train' / 'labels').exists()
        and (dataset_root / 'valid' / 'images').exists()
        and (dataset_root / 'valid' / 'labels').exists()
    )


def _write_rf_detr_data_yaml(*, dataset_dir: Path, request: dict[str, Any]) -> None:
    classes = request.get('classes') if isinstance(request.get('classes'), list) else []
    lines = [
        f'path: {dataset_dir.as_posix()}',
        'train: train/images',
        'val: valid/images',
        'test: test/images',
        'names:',
    ]
    lines.extend(f'  {index}: {name}' for index, name in enumerate(classes))
    (dataset_dir / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _link_or_copy_dir(*, source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def _prepare_d_fine_config(*, request: dict[str, Any], repo_dir: Path, result_dir: Path) -> Path:
    base_config = Path(str(request_param(request, 'd_fine_config', 'configs/dfine/dfine_hgnetv2_n_coco.yml'))).expanduser()
    if not base_config.is_absolute():
        base_config = repo_dir / base_config
    auto_config = _optional_bool(request_param(request, 'd_fine_auto_config', True), True)
    if not auto_config:
        return base_config

    runtime_ann_dir = result_dir / 'runtime' / 'd_fine_coco' / 'annotations'
    runtime_ann_dir.mkdir(parents=True, exist_ok=True)
    annotation_paths = request.get('annotations') if isinstance(request.get('annotations'), dict) else {}
    train_ann = _write_zero_based_coco_annotation(
        source=Path(str(annotation_paths.get('train') or '')).expanduser(),
        target=runtime_ann_dir / 'instances_train_zero_based.json',
    )
    val_ann = _write_zero_based_coco_annotation(
        source=Path(str(annotation_paths.get('val') or annotation_paths.get('valid') or annotation_paths.get('test') or '')).expanduser(),
        target=runtime_ann_dir / 'instances_val_zero_based.json',
    )
    dataset_root = Path(str(request.get('dataset_root') or request.get('coco_root') or '.')).expanduser().resolve()
    image_folder = Path(str(request_param(request, 'd_fine_img_folder', dataset_root))).expanduser().resolve()
    output_dir = result_dir / 'native' / 'd_fine'
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = repo_dir / 'configs' / 'dfine' / 'ironflow_runtime'
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_config = generated_dir / f'{_safe_config_stem(result_dir.name)}.yml'
    include_ref = _d_fine_include_reference(base_config=base_config, generated_dir=generated_dir)
    class_count = len(request.get('classes') if isinstance(request.get('classes'), list) else [])
    if class_count <= 0:
        class_count = _coco_category_count(train_ann)

    config_lines = [
        f'__include__: [{include_ref}]',
        '',
        f'output_dir: {output_dir.as_posix()}',
        f'num_classes: {class_count}',
        'remap_mscoco_category: False',
        f'epochs: {_optional_int(request_param(request, "epochs"), 1)}',
        f'print_freq: {_optional_int(request_param(request, "d_fine_print_freq"), 20)}',
        f'checkpoint_freq: {_optional_int(request_param(request, "d_fine_checkpoint_freq"), 1)}',
        'sync_bn: False',
    ]
    learning_rate = _optional_float(request_param(request, 'learning_rate'))
    if learning_rate is not None:
        config_lines.extend([
            '',
            'optimizer:',
            f'  lr: {learning_rate}',
        ])
    config_lines.extend([
        '',
        'train_dataloader:',
        f'  total_batch_size: {_optional_int(request_param(request, "batch_size"), 1)}',
        f'  num_workers: {_optional_int(request_param(request, "num_workers"), 2)}',
        '  dataset:',
        f'    img_folder: {image_folder.as_posix()}',
        f'    ann_file: {train_ann.as_posix()}',
        '  collate_fn:',
        f'    base_size: {_optional_int(request_param(request, "image_size"), 640)}',
        f'    stop_epoch: {_optional_int(request_param(request, "epochs"), 1)}',
        '  drop_last: False',
        '',
        'val_dataloader:',
        f'  total_batch_size: {_optional_int(request_param(request, "batch_size"), 1)}',
        f'  num_workers: {_optional_int(request_param(request, "num_workers"), 2)}',
        '  dataset:',
        f'    img_folder: {image_folder.as_posix()}',
        f'    ann_file: {val_ann.as_posix()}',
    ])
    generated_config.write_text('\n'.join(config_lines) + '\n', encoding='utf-8')
    write_json(
        result_dir / 'external' / 'd_fine_dataset_bridge.json',
        {
            'base_config': str(base_config),
            'generated_config': str(generated_config),
            'dataset_root': str(dataset_root),
            'image_folder': str(image_folder),
            'train_annotation': str(train_ann),
            'val_annotation': str(val_ann),
            'class_count': class_count,
        },
    )

    return generated_config


def _write_zero_based_coco_annotation(*, source: Path, target: Path) -> Path:
    if not str(source) or not source.exists():
        raise RuntimeError(f'D-FINE dataset bridge requires an existing COCO annotation file: {source}')
    payload = json.loads(source.read_text(encoding='utf-8'))
    categories = payload.get('categories')
    if not isinstance(categories, list) or not categories:
        raise RuntimeError(f'D-FINE dataset bridge requires COCO categories: {source}')
    category_ids = [int(category['id']) for category in categories if isinstance(category, dict) and 'id' in category]
    id_map = {old_id: index for index, old_id in enumerate(sorted(category_ids))}
    for category in categories:
        if isinstance(category, dict) and 'id' in category:
            category['id'] = id_map[int(category['id'])]
    annotations = payload.get('annotations', [])
    if isinstance(annotations, list):
        for annotation in annotations:
            if isinstance(annotation, dict) and 'category_id' in annotation:
                annotation['category_id'] = id_map.get(int(annotation['category_id']), int(annotation['category_id']))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    return target


def _d_fine_include_reference(*, base_config: Path, generated_dir: Path) -> str:
    try:
        return Path('..', base_config.resolve().relative_to(generated_dir.parent.resolve()).as_posix()).as_posix()
    except ValueError:
        return base_config.as_posix()


def _safe_config_stem(value: str) -> str:
    return ''.join(character if character.isalnum() or character in {'-', '_'} else '_' for character in value) or 'ironflow'


def _coco_category_count(annotation_path: Path) -> int:
    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    categories = payload.get('categories')
    return len(categories) if isinstance(categories, list) else 0


def _write_ultralytics_data_yaml(*, result_dir: Path, request: dict[str, Any]) -> Path:
    names = request.get('classes') if isinstance(request.get('classes'), list) else []
    dataset_root = Path(str(request.get('dataset_root') or request.get('coco_root') or '.')).expanduser().resolve()
    payload = {
        'path': dataset_root.as_posix(),
        'train': str(request_param(request, 'ultralytics_train_images', 'images/train')),
        'val': str(request_param(request, 'ultralytics_val_images', 'images/val')),
        'test': str(request_param(request, 'ultralytics_test_images', 'images/test')),
        'names': {index: str(name) for index, name in enumerate(names)},
    }
    data_yaml = result_dir / 'runtime' / 'ultralytics_rtdetr_data.yaml'
    lines = [
        f'path: {payload["path"]}',
        f'train: {payload["train"]}',
        f'val: {payload["val"]}',
        f'test: {payload["test"]}',
        'names:',
    ]
    lines.extend(f'  {index}: {name}' for index, name in payload['names'].items())
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(result_dir / 'external' / 'ultralytics_data_yaml.json', payload)

    return data_yaml


def _native_detection_source(*, request: dict[str, Any]) -> Path:
    explicit = request_param(request, 'source') or request_param(request, 'image_source')
    if explicit:
        return Path(str(explicit)).expanduser()
    image_root = request.get('image_root')
    if image_root:
        return Path(str(image_root)).expanduser()
    dataset_root = request.get('dataset_root') or request.get('coco_root')
    if dataset_root:
        return Path(str(dataset_root)).expanduser()

    raise RuntimeError('native detection inference requires params.source, request image_root, or dataset_root')


def _native_detection_image_sources(*, source: Path) -> list[Path]:
    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if source.is_file():
        return [source]
    if not source.exists():
        raise RuntimeError(f'native detection source not found: {source}')
    if not source.is_dir():
        raise RuntimeError(f'native detection source must be a file or directory: {source}')

    return [
        path
        for path in sorted(source.rglob('*'))
        if path.is_file() and path.suffix.lower() in image_suffixes
    ]


def _ultralytics_results_to_records(*, results: Any, request: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    classes = request.get('classes') if isinstance(request.get('classes'), list) else []
    for image_index, result in enumerate(results):
        image_path = Path(str(getattr(result, 'path', f'image_{image_index:04d}')))
        image_id = image_path.stem
        height, width = [int(value) for value in getattr(result, 'orig_shape', (0, 0))]
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        for box_index, bbox in enumerate(xyxy):
            class_index = int(cls[box_index])
            records.append(
                {
                    'image_id': image_id,
                    'sample_id': image_id,
                    'prediction_id': f'{image_id}_det_{box_index:04d}',
                    'class_id': str(classes[class_index]) if 0 <= class_index < len(classes) else str(class_index),
                    'score': float(conf[box_index]),
                    'bbox_xyxy': _clamped_bbox_xyxy(bbox=bbox, image_width=width, image_height=height),
                    'image_width': width,
                    'image_height': height,
                },
            )

    return records


def _supervision_detections_to_records(*, detections: Any, request: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    classes = request.get('classes') if isinstance(request.get('classes'), list) else []
    xyxy = getattr(detections, 'xyxy', [])
    confidence = getattr(detections, 'confidence', None)
    if confidence is None:
        confidence = [0.0 for _ in xyxy]
    class_ids = getattr(detections, 'class_id', None)
    if class_ids is None:
        class_ids = [0 for _ in xyxy]
    image_id = source.stem
    image_width, image_height = _image_dimensions(path=source)
    records: list[dict[str, Any]] = []
    for index, bbox in enumerate(xyxy):
        class_index = int(class_ids[index])
        records.append(
            {
                'image_id': image_id,
                'sample_id': image_id,
                'prediction_id': f'{image_id}_det_{index:04d}',
                'class_id': str(classes[class_index]) if 0 <= class_index < len(classes) else str(class_index),
                'score': float(confidence[index]),
                'bbox_xyxy': _clamped_bbox_xyxy(bbox=bbox, image_width=image_width, image_height=image_height),
                'image_width': image_width,
                'image_height': image_height,
            },
        )

    return records


def _image_dimensions(*, path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _clamped_bbox_xyxy(*, bbox: Any, image_width: int, image_height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if image_width <= 0 or image_height <= 0:
        return [x1, y1, x2, y2]

    max_x = float(image_width)
    max_y = float(image_height)
    x1 = max(0.0, min(max_x, x1))
    x2 = max(0.0, min(max_x, x2))
    y1 = max(0.0, min(max_y, y1))
    y2 = max(0.0, min(max_y, y2))
    if x1 == x2:
        if x1 >= max_x:
            x1 = max(0.0, max_x - 1.0)
        else:
            x2 = min(max_x, x1 + 1.0)
    if y1 == y2:
        if y1 >= max_y:
            y1 = max(0.0, max_y - 1.0)
        else:
            y2 = min(max_y, y1 + 1.0)

    return [x1, y1, x2, y2]


def _copy_first_existing_checkpoint(*, candidates: list[Path], result_dir: Path) -> None:
    for candidate in candidates:
        if copy_if_exists(candidate, result_dir / 'checkpoints' / 'best.pt'):
            copy_if_exists(candidate, result_dir / 'checkpoints' / 'last.pt')
            return


def _copy_d_fine_outputs(*, repo_dir: Path, result_dir: Path, request: dict[str, Any]) -> None:
    best_path = request_param(request, 'd_fine_best_checkpoint')
    last_path = request_param(request, 'd_fine_last_checkpoint')
    copied_best = copy_if_exists(Path(str(best_path)).expanduser() if best_path else None, result_dir / 'checkpoints' / 'best.pt')
    copied_last = copy_if_exists(Path(str(last_path)).expanduser() if last_path else None, result_dir / 'checkpoints' / 'last.pt')
    native_output_dir = result_dir / 'native' / 'd_fine'
    if not copied_best:
        _copy_first_existing_checkpoint(
            candidates=[
                native_output_dir / 'best_stg1.pth',
                native_output_dir / 'best.pth',
                *sorted(native_output_dir.rglob('best*.pth')),
                *sorted(native_output_dir.rglob('checkpoint*.pth')),
            ],
            result_dir=result_dir,
        )
    if not copied_last:
        copy_if_exists(native_output_dir / 'last.pth', result_dir / 'checkpoints' / 'last.pt')
    prediction_path = request_param(request, 'd_fine_prediction_json')
    if prediction_path:
        src = Path(str(prediction_path)).expanduser()
        if src.exists():
            write_json(result_dir / 'predictions' / 'detection_predictions.json', json.loads(src.read_text(encoding='utf-8')))


def _write_d_fine_native_predictions(
    *,
    repo_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    result_dir: Path,
    request: dict[str, Any],
) -> None:
    if not checkpoint_path.exists():
        raise RuntimeError(f'D-FINE prediction export checkpoint not found: {checkpoint_path}')
    sys.path.insert(0, str(repo_dir))
    torch = optional_import('torch')
    transforms_module = optional_import('torchvision.transforms')
    image_module = optional_import('PIL.Image')
    yaml_config_module = optional_import('src.core')
    yaml_config = getattr(yaml_config_module, 'YAMLConfig')

    cfg = yaml_config(str(config_path), resume=str(checkpoint_path))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    payload = _torch_load_d_fine_checkpoint(torch=torch, checkpoint_path=checkpoint_path)
    if isinstance(payload, dict) and 'ema' in payload and isinstance(payload['ema'], dict):
        state = payload['ema'].get('module')
    elif isinstance(payload, dict):
        state = payload.get('model')
    else:
        state = None
    if state is None:
        raise RuntimeError(f'D-FINE checkpoint does not contain model or ema.module: {checkpoint_path}')
    cfg.model.load_state_dict(state)
    device = str(request_param(request, 'device', 'cpu') or 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    class DfineDeployModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: Any, orig_target_sizes: Any) -> Any:
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DfineDeployModel().to(device)
    model.eval()
    image_size = _optional_int(request_param(request, 'image_size'), 640) or 640
    transform = transforms_module.Compose([
        transforms_module.Resize((image_size, image_size)),
        transforms_module.ToTensor(),
    ])
    source = _native_detection_source(request=request)
    image_paths = _native_detection_image_sources(source=source)
    threshold = _optional_float(
        request_param(request, 'd_fine_confidence_threshold', request_param(request, 'confidence_threshold')),
        0.25,
    )
    top_k = _optional_int(request_param(request, 'd_fine_prediction_top_k'), 20)
    records: list[dict[str, Any]] = []
    started = now_seconds()
    with torch.no_grad():
        for image_path in image_paths:
            with image_module.open(image_path) as image:
                image_rgb = image.convert('RGB')
                width, height = image_rgb.size
                image_tensor = transform(image_rgb).unsqueeze(0).to(device)
            orig_size = torch.tensor([[width, height]], device=device)
            labels, boxes, scores = model(image_tensor, orig_size)
            records.extend(
                _d_fine_tensors_to_records(
                    labels=labels[0],
                    boxes=boxes[0],
                    scores=scores[0],
                    request=request,
                    image_path=image_path,
                    image_width=width,
                    image_height=height,
                    threshold=threshold,
                    top_k=top_k,
                ),
            )
    prediction_path = write_detection_predictions(result_dir=result_dir, request=request, records=records)
    write_json(
        result_dir / 'external' / 'd_fine_prediction_export.json',
        {
            'schema_version': '0.1',
            'prediction_path': str(prediction_path),
            'image_count': len(image_paths),
            'record_count': len(records),
            'confidence_threshold': threshold,
            'top_k': top_k,
            'elapsed_seconds': now_seconds() - started,
        },
    )


def _torch_load_d_fine_checkpoint(*, torch: Any, checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def _d_fine_tensors_to_records(
    *,
    labels: Any,
    boxes: Any,
    scores: Any,
    request: dict[str, Any],
    image_path: Path,
    image_width: int,
    image_height: int,
    threshold: float | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    classes = request.get('classes') if isinstance(request.get('classes'), list) else []
    raw_scores = [float(value) for value in scores.detach().cpu().tolist()]
    raw_labels = [int(value) for value in labels.detach().cpu().tolist()]
    raw_boxes = boxes.detach().cpu().tolist()
    order = sorted(range(len(raw_scores)), key=lambda index: raw_scores[index], reverse=True)
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    image_id = image_path.stem
    records: list[dict[str, Any]] = []
    for rank, index in enumerate(order):
        score = raw_scores[index]
        if threshold is not None and score < threshold:
            continue
        class_index = raw_labels[index]
        records.append(
            {
                'image_id': image_id,
                'sample_id': image_id,
                'prediction_id': f'{image_id}_d_fine_{rank:04d}',
                'class_id': str(classes[class_index]) if 0 <= class_index < len(classes) else str(class_index),
                'score': score,
                'bbox_xyxy': _clamped_bbox_xyxy(
                    bbox=raw_boxes[index],
                    image_width=image_width,
                    image_height=image_height,
                ),
                'image_width': image_width,
                'image_height': image_height,
            },
        )

    return records


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == '':
        return default

    return int(value)


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == '':
        return default

    return float(value)


def _optional_bool(value: Any, default: bool) -> bool:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False

    return default
