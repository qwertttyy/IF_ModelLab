"""Build a YOLO detection dataset whose classes are target model names."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.datasets.coco_detection_materializer import CocoDetectionMaterializer


DEFAULT_SOURCES = (
    ('k2', 'K-2'),
    ('k21_ifv', 'K-21_IFV'),
    ('k9_thunder', 'K-9_thunder'),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-root',
        default='runs/user_datasets/tank14_prepared_v20260629',
        help='Root containing imported per-model datasets, relative to the project root.',
    )
    parser.add_argument(
        '--output-root',
        default='runs/user_datasets/tank14_prepared_v20260629/combined_model_name_detection/detection',
        help='Output detection dataset root, relative to the project root.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing generated output folder.',
    )
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    _prepare_output_root(output_root=output_root, overwrite=args.overwrite)

    classes = [target_name for _, target_name in DEFAULT_SOURCES]
    class_index = {
        class_name: index
        for index, class_name in enumerate(classes)
    }
    manifest_images: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    object_counts: Counter[str] = Counter()

    for source_name, target_name in DEFAULT_SOURCES:
        source_root = input_root / source_name / 'detection'
        source_manifest = _read_manifest(source_root / 'manifest.json')
        source_image_count = 0
        source_object_count = 0
        split_counts: Counter[str] = Counter()

        for image_record in source_manifest['images']:
            rel_image_path = Path(str(image_record['path']))
            source_image_path = source_root / rel_image_path
            split = str(image_record.get('split') or 'train')
            dest_image_rel = Path('images') / split / f'{source_name}_{source_image_path.name}'
            dest_image_path = output_root / dest_image_rel
            dest_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image_path, dest_image_path)

            new_record = deepcopy(image_record)
            original_image_id = str(new_record.get('image_id') or source_image_path.stem)
            original_sample_id = str(new_record.get('sample_id') or source_image_path.stem)
            new_record['image_id'] = f'{source_name}_{original_image_id}'
            new_record['sample_id'] = f'{source_name}_{original_sample_id}'
            new_record['path'] = dest_image_rel.as_posix()
            new_record['label'] = target_name
            new_record['split'] = split
            if isinstance(new_record.get('source'), dict):
                new_record['source']['source_dataset'] = source_name
                new_record['source']['model_name_target'] = target_name

            label_lines: list[str] = []
            new_objects: list[dict[str, Any]] = []
            for object_index, object_record in enumerate(image_record.get('objects') or []):
                bbox_yolo = object_record.get('bbox_yolo')
                if not isinstance(bbox_yolo, list) or len(bbox_yolo) != 4:
                    raise ValueError(f'object bbox_yolo must have four values: {source_image_path}')
                new_object = deepcopy(object_record)
                new_object['object_id'] = f'{source_name}_{object_record.get("object_id") or object_index + 1}'
                new_object['class_id'] = target_name
                new_objects.append(new_object)
                label_lines.append(
                    f'{class_index[target_name]} '
                    f'{float(bbox_yolo[0]):.6f} {float(bbox_yolo[1]):.6f} '
                    f'{float(bbox_yolo[2]):.6f} {float(bbox_yolo[3]):.6f}',
                )

            new_record['objects'] = new_objects
            manifest_images.append(new_record)
            object_counts[target_name] += len(new_objects)
            source_image_count += 1
            source_object_count += len(new_objects)
            split_counts[split] += 1

            label_rel = Path('labels') / split / f'{dest_image_path.stem}.txt'
            label_path = output_root / label_rel
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('\n'.join(label_lines) + ('\n' if label_lines else ''), encoding='utf-8')

        source_reports.append(
            {
                'source': source_name,
                'model_name': target_name,
                'path': str(source_root),
                'image_count': source_image_count,
                'object_count': source_object_count,
                'split_counts': dict(sorted(split_counts.items())),
            },
        )

    manifest = {
        'schema_version': '0.1',
        'dataset_id': 'imported_20260617_combined_model_name_detection',
        'dataset_type': 'detection',
        'classes': classes,
        'images': manifest_images,
    }
    _write_json(output_root / 'manifest.json', manifest)
    _write_data_yaml(output_root=output_root, classes=classes)
    coco_result = CocoDetectionMaterializer().materialize(dataset_root=output_root)

    report = {
        'schema_version': '0.1',
        'dataset_id': manifest['dataset_id'],
        'target': str(output_root),
        'sources': source_reports,
        'image_count': len(manifest_images),
        'object_count': sum(object_counts.values()),
        'classes': classes,
        'object_counts': dict(object_counts),
        'coco': coco_result.to_dict(),
    }
    report_path = output_root.parent / 'merge_report.json'
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _prepare_output_root(*, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f'output root already exists; pass --overwrite to replace it: {output_root}')
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('dataset_type') != 'detection':
        raise ValueError(f'expected a detection manifest: {path}')
    if not isinstance(data.get('images'), list):
        raise ValueError(f'detection manifest images must be a list: {path}')
    return data


def _write_data_yaml(*, output_root: Path, classes: list[str]) -> None:
    payload = {
        'path': output_root.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            index: class_name
            for index, class_name in enumerate(classes)
        },
    }
    (output_root / 'data.yaml').write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
