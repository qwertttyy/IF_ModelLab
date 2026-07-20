"""Shared command-line contract helpers for external segmentation wrappers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_wrapper_common import (
    image_path_for_record,
    manifest_image_records,
    now_seconds,
    optional_import,
    request_param,
    result_dir_from_request,
    write_contract_metadata,
    write_error,
    write_runtime_metrics,
    write_segmentation_predictions,
)


def run_contract_wrapper(*, model_family: str, adapter_key: str) -> int:
    parser = argparse.ArgumentParser(description=f'IronFlow external segmentation wrapper for {model_family}.')
    parser.add_argument('--request', required=True, help='Path to runtime request JSON.')
    parser.add_argument('--dry-run-contract', action='store_true')
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
        native_ready=model_family == 'sam_promptable',
    )

    if args.dry_run_contract:
        _write_metrics(result_dir=result_dir)
        _write_predictions(result_dir=result_dir, request=request)
        print(json.dumps({'ok': True, 'mode': 'dry_run_contract', 'result_dir': str(result_dir)}))
        return 0

    if model_family == 'sam_promptable':
        return _run_native_sam2(
            adapter_key=adapter_key,
            model_family=model_family,
            request=request,
            result_dir=result_dir,
        )

    return write_error(
        result_dir=result_dir,
        adapter_key=adapter_key,
        model_family=model_family,
        error_type='native_not_supported',
        message='native segmentation execution is not implemented for this wrapper yet',
    )


def _read_request(*, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'request JSON root must be an object: {path}')
    for key in ('adapter', 'model_id', 'execution_mode', 'result_dir'):
        if key not in data:
            raise ValueError(f'request JSON is missing required key: {key}')

    return data


def _write_contract_metadata(
    *,
    result_dir: Path,
    request_path: Path,
    request: dict[str, Any],
    model_family: str,
    adapter_key: str,
) -> None:
    external_dir = result_dir / 'external'
    external_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': '0.1',
        'model_family': model_family,
        'adapter_key': adapter_key,
        'request_json': str(request_path),
        'request_model_id': request.get('model_id'),
        'execution_mode': request.get('execution_mode'),
        'result_dir': str(result_dir),
    }
    (external_dir / 'wrapper_contract.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _write_metrics(*, result_dir: Path) -> None:
    with (result_dir / 'metrics.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'val_loss', 'mask_iou', 'lr'])
        writer.writeheader()
        writer.writerow({'epoch': 1, 'train_loss': '', 'val_loss': '', 'mask_iou': 0.0, 'lr': ''})


def _write_predictions(*, result_dir: Path, request: dict[str, Any]) -> None:
    write_segmentation_predictions(
        result_dir=result_dir,
        request=request,
        records=[
            {
                'image_id': 'dry_run_image_0001',
                'sample_id': 'dry_run_sample_0001',
                'prediction_id': 'dry_run_mask_0001',
                'class_id': 'object',
                'score': 0.01,
                'mask_kind': 'polygon',
                'mask_polygon': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                'bbox_xyxy': [0.0, 0.0, 1.0, 1.0],
                'image_width': 1,
                'image_height': 1,
            },
        ],
    )


def _run_native_sam2(
    *,
    adapter_key: str,
    model_family: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    start = now_seconds()
    try:
        torch = optional_import('torch')
        np = optional_import('numpy')
        image_module = optional_import('PIL.Image')
        predictor = _load_sam2_predictor(request=request)
        manifest_path, _manifest, image_records = manifest_image_records(request)
        prompts = _segmentation_prompts(request=request)
        prompt_by_image = _prompts_by_image(prompts=prompts)
        records: list[dict[str, Any]] = []
        for image_index, image_record in enumerate(image_records):
            image_path = image_path_for_record(manifest_path=manifest_path, record=image_record)
            image = image_module.open(image_path).convert('RGB')
            image_id = str(image_record.get('image_id') or image_path.stem)
            sample_id = str(image_record.get('sample_id') or image_id)
            image_width, image_height = image.size
            image_prompts = prompt_by_image.get(image_id) or _fallback_prompts(image_record=image_record)
            if not image_prompts:
                continue
            predictor.set_image(np.asarray(image))
            for prompt_index, prompt in enumerate(image_prompts):
                bbox = [float(value) for value in prompt['bbox_xyxy']]
                with torch.inference_mode():
                    masks, scores, _logits = predictor.predict(
                        box=np.asarray(bbox, dtype=np.float32),
                        multimask_output=False,
                    )
                mask = masks[0] if len(masks) else None
                score = float(scores[0]) if len(scores) else float(prompt.get('score', 0.0))
                polygon = _mask_to_polygon(mask=mask, fallback_bbox=bbox)
                records.append(
                    {
                        'image_id': image_id,
                        'sample_id': sample_id,
                        'prediction_id': f'{image_id}_sam2_{prompt_index:04d}',
                        'class_id': str(prompt.get('class_id') or 'object'),
                        'score': score,
                        'mask_kind': 'polygon',
                        'mask_polygon': polygon,
                        'bbox_xyxy': bbox,
                        'image_width': int(image_width),
                        'image_height': int(image_height),
                    },
                )
        write_segmentation_predictions(
            result_dir=result_dir,
            request=request,
            records=records or [_empty_segmentation_record(request=request)],
        )
        write_runtime_metrics(result_dir=result_dir, elapsed_seconds=now_seconds() - start)
        print(json.dumps({'ok': True, 'mode': 'native_inference', 'model_family': model_family, 'prediction_count': len(records)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_sam2_failed',
            message=str(exc),
        )


def _load_sam2_predictor(*, request: dict[str, Any]) -> Any:
    predictor_module = optional_import('sam2.sam2_image_predictor')
    predictor_class = predictor_module.SAM2ImagePredictor
    hf_model_id = request_param(request, 'sam2_hf_model_id') or request_param(request, 'sam2_model_id')
    if hf_model_id:
        return predictor_class.from_pretrained(str(hf_model_id))

    build_module = optional_import('sam2.build_sam')
    checkpoint = request_param(request, 'checkpoint') or request_param(request, 'sam2_checkpoint')
    model_cfg = request_param(request, 'sam2_model_cfg')
    if not checkpoint or not model_cfg:
        raise RuntimeError('SAM2 native execution requires params.sam2_hf_model_id or params.sam2_checkpoint + params.sam2_model_cfg')

    return predictor_class(build_module.build_sam2(str(model_cfg), str(checkpoint)))


def _segmentation_prompts(*, request: dict[str, Any]) -> list[dict[str, Any]]:
    raw_path = request.get('detection_predictions') or request_param(request, 'detection_predictions_path')
    if not raw_path:
        return []
    path = Path(str(raw_path)).expanduser()
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = payload.get('records') if isinstance(payload, dict) else []

    return [record for record in records if isinstance(record, dict) and isinstance(record.get('bbox_xyxy'), list)]


def _prompts_by_image(*, prompts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prompt in prompts:
        image_id = str(prompt.get('image_id') or '')
        if not image_id:
            continue
        grouped.setdefault(image_id, []).append(prompt)

    return grouped


def _fallback_prompts(*, image_record: dict[str, Any]) -> list[dict[str, Any]]:
    objects = image_record.get('objects')
    if not isinstance(objects, list):
        return []
    prompts: list[dict[str, Any]] = []
    for object_record in objects:
        if not isinstance(object_record, dict) or not isinstance(object_record.get('bbox_xyxy'), list):
            continue
        prompts.append(object_record)

    return prompts


def _mask_to_polygon(*, mask: Any, fallback_bbox: list[float]) -> list[list[float]]:
    if mask is None:
        return _bbox_polygon(bbox=fallback_bbox)
    try:
        np = optional_import('numpy')
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return _bbox_polygon(bbox=fallback_bbox)
        return _bbox_polygon(bbox=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    except Exception:
        return _bbox_polygon(bbox=fallback_bbox)


def _bbox_polygon(*, bbox: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _empty_segmentation_record(*, request: dict[str, Any]) -> dict[str, Any]:
    return {
        'image_id': 'empty_native_image',
        'sample_id': 'empty_native_sample',
        'prediction_id': 'empty_native_mask',
        'class_id': 'object',
        'score': 0.0,
        'mask_kind': 'polygon',
        'mask_polygon': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        'bbox_xyxy': [0.0, 0.0, 1.0, 1.0],
        'image_width': 1,
        'image_height': 1,
    }
