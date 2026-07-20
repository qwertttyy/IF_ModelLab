"""Shared command-line contract helpers for external embedding wrappers."""

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
    write_embedding_predictions,
    write_error,
    write_runtime_metrics,
)


def run_contract_wrapper(*, model_family: str, adapter_key: str) -> int:
    parser = argparse.ArgumentParser(description=f'IronFlow external embedding wrapper for {model_family}.')
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
        native_ready=model_family in {'dinov2', 'dinov3', 'clip', 'siglip'},
    )

    if args.dry_run_contract:
        _write_metrics(result_dir=result_dir)
        _write_embeddings(result_dir=result_dir, request=request)
        print(json.dumps({'ok': True, 'mode': 'dry_run_contract', 'result_dir': str(result_dir)}))
        return 0

    return _run_native_embedding(
        model_family=model_family,
        adapter_key=adapter_key,
        request=request,
        result_dir=result_dir,
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
        writer = csv.DictWriter(file, fieldnames=['epoch', 'retrieval_map', 'neighbor_purity'])
        writer.writeheader()
        writer.writerow({'epoch': 1, 'retrieval_map': 0.0, 'neighbor_purity': 0.0})


def _write_embeddings(*, result_dir: Path, request: dict[str, Any]) -> None:
    write_embedding_predictions(
        result_dir=result_dir,
        request=request,
        records=[
            {
                'image_id': 'dry_run_image_0001',
                'sample_id': 'dry_run_sample_0001',
                'prediction_id': 'dry_run_embedding_prediction_0001',
                'object_id': 'dry_run_object_0001',
                'embedding_id': 'dry_run_embedding_0001',
                'embedding_index': 0,
                'embedding_dim': 4,
                'embedding_path': 'embeddings.npy',
                'source_path': 'dry_run_source.jpg',
            },
        ],
    )
    (result_dir / 'embeddings.npy').write_bytes(b'\x93NUMPY dry-run placeholder\n')
    with (result_dir / 'embeddings_meta.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['embedding_id', 'sample_id', 'image_id', 'object_id', 'embedding_index', 'embedding_dim', 'source_path'])
        writer.writeheader()
        writer.writerow({
            'embedding_id': 'dry_run_embedding_0001',
            'sample_id': 'dry_run_sample_0001',
            'image_id': 'dry_run_image_0001',
            'object_id': 'dry_run_object_0001',
            'embedding_index': 0,
            'embedding_dim': 4,
            'source_path': 'dry_run_source.jpg',
        })


def _run_native_embedding(
    *,
    model_family: str,
    adapter_key: str,
    request: dict[str, Any],
    result_dir: Path,
) -> int:
    if model_family in {'clip', 'siglip'}:
        return _run_native_openclip(adapter_key=adapter_key, model_family=model_family, request=request, result_dir=result_dir)
    if model_family in {'dinov2', 'dinov3'}:
        return _run_native_dino(adapter_key=adapter_key, model_family=model_family, request=request, result_dir=result_dir)

    return write_error(
        result_dir=result_dir,
        adapter_key=adapter_key,
        model_family=model_family,
        error_type='native_not_supported',
        message='native embedding execution is not implemented for this wrapper yet',
    )


def _run_native_openclip(
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
        open_clip = optional_import('open_clip')
        model_name = str(request_param(request, 'openclip_model_name', request_param(request, 'open_clip_model', 'ViT-B-32')))
        pretrained = str(request_param(request, 'openclip_pretrained', request_param(request, 'open_clip_pretrained', 'laion2b_s34b_b79k')))
        device = str(request_param(request, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        model, _unused, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        manifest_path, _manifest, image_records = manifest_image_records(request)
        vectors: list[Any] = []
        records: list[dict[str, Any]] = []
        with torch.no_grad():
            for index, image_record in enumerate(image_records):
                image_path = image_path_for_record(manifest_path=manifest_path, record=image_record)
                image = preprocess(image_module.open(image_path).convert('RGB')).unsqueeze(0).to(device)
                vector = model.encode_image(image)
                vector = vector / vector.norm(dim=-1, keepdim=True)
                vector_np = vector.detach().cpu().numpy()[0].astype('float32')
                vectors.append(vector_np)
                records.append(_embedding_record(index=index, image_record=image_record, image_path=image_path, dim=int(vector_np.shape[0])))
        _write_native_embedding_outputs(result_dir=result_dir, request=request, records=records, vectors=vectors, np=np)
        write_runtime_metrics(result_dir=result_dir, elapsed_seconds=now_seconds() - start)
        print(json.dumps({'ok': True, 'mode': 'native_inference', 'model_family': model_family, 'embedding_count': len(records)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_openclip_failed',
            message=str(exc),
        )


def _run_native_dino(
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
        transforms = optional_import('torchvision.transforms')
        repo_dir = request_param(request, 'dino_repo_dir') or request_param(request, 'dinov3_repo_dir')
        if not repo_dir:
            raise RuntimeError('DINO native execution requires params.dino_repo_dir or params.dinov3_repo_dir')
        hub_model = str(request_param(request, 'dino_hub_model', 'dinov3_vits16' if model_family == 'dinov3' else 'dinov2_vits14'))
        weights = request_param(request, 'dino_weights') or request_param(request, 'dinov3_weights')
        device = str(request_param(request, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        pretrained = _optional_bool(request_param(request, 'dino_pretrained'), True)
        kwargs = {'source': 'local', 'pretrained': pretrained}
        if weights:
            kwargs['weights'] = str(weights)
        model = torch.hub.load(str(Path(str(repo_dir)).expanduser()), hub_model, **kwargs)
        model = model.to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        manifest_path, _manifest, image_records = manifest_image_records(request)
        vectors: list[Any] = []
        records: list[dict[str, Any]] = []
        with torch.no_grad():
            for index, image_record in enumerate(image_records):
                image_path = image_path_for_record(manifest_path=manifest_path, record=image_record)
                image = preprocess(image_module.open(image_path).convert('RGB')).unsqueeze(0).to(device)
                output = model(image)
                vector = _pooled_tensor(output)
                vector_np = vector.detach().cpu().numpy()[0].astype('float32')
                vectors.append(vector_np)
                records.append(_embedding_record(index=index, image_record=image_record, image_path=image_path, dim=int(vector_np.shape[0])))
        _write_native_embedding_outputs(result_dir=result_dir, request=request, records=records, vectors=vectors, np=np)
        write_runtime_metrics(result_dir=result_dir, elapsed_seconds=now_seconds() - start)
        print(json.dumps({'ok': True, 'mode': 'native_inference', 'model_family': model_family, 'embedding_count': len(records)}))
        return 0
    except Exception as exc:
        return write_error(
            result_dir=result_dir,
            adapter_key=adapter_key,
            model_family=model_family,
            error_type='native_dino_failed',
            message=str(exc),
        )


def _pooled_tensor(output: Any) -> Any:
    if isinstance(output, dict):
        for key in ('pooler_output', 'x_norm_clstoken', 'last_hidden_state'):
            value = output.get(key)
            if value is not None:
                output = value
                break
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, 'ndim') and output.ndim == 3:
        return output[:, 0, :]
    if hasattr(output, 'ndim') and output.ndim == 1:
        return output.unsqueeze(0)

    return output


def _embedding_record(*, index: int, image_record: dict[str, Any], image_path: Path, dim: int) -> dict[str, Any]:
    image_id = str(image_record.get('image_id') or image_path.stem)
    sample_id = str(image_record.get('sample_id') or image_id)
    object_id = str(image_record.get('object_id') or image_record.get('source_object_id') or image_id)
    embedding_id = f'{sample_id}_embedding_{index:04d}'
    return {
        'image_id': image_id,
        'sample_id': sample_id,
        'prediction_id': f'{sample_id}_embedding_prediction_{index:04d}',
        'object_id': object_id,
        'embedding_id': embedding_id,
        'embedding_index': index,
        'embedding_dim': dim,
        'embedding_path': 'embeddings.npy',
        'source_path': str(image_path),
    }


def _write_native_embedding_outputs(
    *,
    result_dir: Path,
    request: dict[str, Any],
    records: list[dict[str, Any]],
    vectors: list[Any],
    np: Any,
) -> None:
    if not vectors:
        vectors = [np.zeros((1,), dtype='float32')]
        records = [_embedding_record(index=0, image_record={'image_id': 'empty_native_image'}, image_path=Path('empty_native_image'), dim=1)]
    matrix = np.stack(vectors).astype('float32')
    np.save(result_dir / 'embeddings.npy', matrix)
    with (result_dir / 'embeddings_meta.csv').open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['embedding_id', 'sample_id', 'image_id', 'object_id', 'embedding_index', 'embedding_dim', 'source_path'])
        writer.writeheader()
        for record in records:
            writer.writerow({
                'embedding_id': record['embedding_id'],
                'sample_id': record['sample_id'],
                'image_id': record['image_id'],
                'object_id': record['object_id'],
                'embedding_index': record['embedding_index'],
                'embedding_dim': record['embedding_dim'],
                'source_path': record['source_path'],
            })
    write_embedding_predictions(result_dir=result_dir, request=request, records=records)


def _optional_bool(value: Any, default: bool) -> bool:
    if value in {None, ''}:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'n', 'off'}:
            return False

    return bool(value)
