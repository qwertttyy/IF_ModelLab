"""Build an object-crop classification dataset from model-name detection labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-root',
        default='runs/user_datasets/tank14_prepared_v20260629/combined_model_name_detection/detection',
        help='Detection dataset root, relative to the project root.',
    )
    parser.add_argument(
        '--output-root',
        default='runs/user_datasets/tank14_prepared_v20260629/combined_model_name_classification/crops',
        help='Output classification crop dataset root, relative to the project root.',
    )
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.08,
        help='BBox-relative padding added around each object crop.',
    )
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output folder.')
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    _prepare_output_root(output_root=output_root, overwrite=args.overwrite)

    manifest = _read_json(input_root / 'manifest.json')
    classes = [str(class_name) for class_name in manifest.get('classes') or []]
    if not classes:
        raise ValueError(f'manifest classes are required: {input_root / "manifest.json"}')

    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for image_record in manifest.get('images') or []:
        image_rel = Path(str(image_record['path']))
        image_path = input_root / image_rel
        split = str(image_record.get('split') or 'train')
        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            width, height = rgb_image.size
            for object_index, object_record in enumerate(image_record.get('objects') or []):
                class_name = str(object_record.get('class_id') or image_record.get('label') or '')
                if class_name not in classes:
                    raise ValueError(f'unknown class_id={class_name!r} in {image_path}')
                bbox_yolo = object_record.get('bbox_yolo')
                if not isinstance(bbox_yolo, list) or len(bbox_yolo) != 4:
                    raise ValueError(f'object bbox_yolo must have four values: {image_path}')

                crop_box = _crop_box(
                    bbox_yolo=[float(value) for value in bbox_yolo],
                    image_width=width,
                    image_height=height,
                    padding_ratio=args.padding_ratio,
                )
                object_id = str(object_record.get('object_id') or f'obj_{object_index + 1:04d}')
                stem = _safe_name(f'{Path(image_rel).stem}_{object_id}')
                crop_rel = Path(split) / class_name / f'{stem}.jpg'
                crop_path = output_root / crop_rel
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                rgb_image.crop(crop_box).save(crop_path, quality=95)

                record = {
                    'image_id': stem,
                    'sample_id': stem,
                    'label': class_name,
                    'split': split,
                    'path': crop_rel.as_posix(),
                    'source_image_path': image_rel.as_posix(),
                    'source_object_id': object_id,
                    'bbox_yolo': bbox_yolo,
                    'crop_box_xyxy': list(crop_box),
                    'padding_ratio': args.padding_ratio,
                }
                records.append(record)
                split_counts[split] += 1
                class_counts[f'{split}/{class_name}'] += 1

    output_manifest = {
        'schema_version': '0.1',
        'dataset_id': 'imported_20260617_combined_model_name_classification_crops',
        'dataset_type': 'classification',
        'input_type': 'object_crop',
        'source_dataset': str(input_root),
        'classes': classes,
        'images': records,
    }
    _write_json(output_root / 'manifest.json', output_manifest)
    report = {
        'schema_version': '0.1',
        'dataset_id': output_manifest['dataset_id'],
        'target': str(output_root),
        'source_dataset': str(input_root),
        'crop_count': len(records),
        'classes': classes,
        'split_counts': dict(sorted(split_counts.items())),
        'class_counts': dict(sorted(class_counts.items())),
        'padding_ratio': args.padding_ratio,
    }
    _write_json(output_root.parent / 'crop_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _prepare_output_root(*, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f'output root already exists; pass --overwrite to replace it: {output_root}')
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _crop_box(*, bbox_yolo: list[float], image_width: int, image_height: int, padding_ratio: float) -> tuple[int, int, int, int]:
    cx, cy, box_width, box_height = bbox_yolo
    x1 = (cx - box_width / 2.0) * image_width
    y1 = (cy - box_height / 2.0) * image_height
    x2 = (cx + box_width / 2.0) * image_width
    y2 = (cy + box_height / 2.0) * image_height
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(image_width, int(x2 + pad_x + 0.999999))
    crop_y2 = min(image_height, int(y2 + pad_y + 0.999999))
    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
        raise ValueError(f'invalid crop box: {bbox_yolo}')
    return crop_x1, crop_y1, crop_x2, crop_y2


def _safe_name(value: str) -> str:
    return ''.join(character if character.isalnum() or character in {'_', '-'} else '_' for character in value).strip('_')


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'expected JSON object: {path}')
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
