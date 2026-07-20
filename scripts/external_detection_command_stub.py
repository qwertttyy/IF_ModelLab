"""Contract stub for COCO external-command detection adapters.

The real RF-DETR, D-FINE, or RT-DETR wrapper should accept the same
``--request`` JSON path and write the same output files under ``result_dir``.
This script is intentionally lightweight so the IronFlow external-command
contract can be tested without a GPU or model package.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, help='Path to runtime/external_detection_request.json.')
    parser.add_argument(
        '--write-predictions',
        action='store_true',
        help='Also write predictions/detection_predictions.json for inference-smoke contract checks.',
    )
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    request = _read_json(request_path)
    result_dir = Path(str(request['result_dir'])).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(result_dir=result_dir)
    _write_checkpoints(result_dir=result_dir)
    if args.write_predictions:
        _write_predictions(result_dir=result_dir, request=request)

    print(json.dumps({'ok': True, 'result_dir': str(result_dir)}, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'request JSON must be an object: {path}')

    return data


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


def _write_checkpoints(*, result_dir: Path) -> None:
    checkpoint_dir = result_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'best.pt').write_bytes(b'ironflow external detection contract stub best\n')
    (checkpoint_dir / 'last.pt').write_bytes(b'ironflow external detection contract stub last\n')


def _write_predictions(*, result_dir: Path, request: dict[str, Any]) -> None:
    classes = request.get('classes') if isinstance(request.get('classes'), list) else ['object']
    class_id = str(classes[0]) if classes else 'object'
    payload = {
        'schema_version': '0.1',
        'task': 'detection',
        'model_id': str(request.get('model_id') or request.get('adapter') or 'external_detector'),
        'dataset_id': Path(str(request.get('dataset_root') or 'dataset')).name,
        'success': True,
        'records': [
            {
                'image_id': 'stub_image_0001',
                'sample_id': 'stub_sample_0001',
                'prediction_id': 'stub_prediction_0001',
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


if __name__ == '__main__':
    raise SystemExit(main())
