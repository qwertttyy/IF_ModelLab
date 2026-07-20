"""Import a class-folder image dataset into the IronFlow Top10 training layout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.datasets.coco_detection_materializer import CocoDetectionMaterializer


SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm'}
DEFAULT_CLASSES = (
    'altay',
    'challenger_2',
    'k2',
    'leopard_2',
    'leclerc',
    'm1_abrams',
    'merkava_mk4',
    'type_10',
)


def main() -> int:
    return main_with_args(None)


def main_with_args(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='D:/data')
    parser.add_argument('--dataset-root', default='runs/user_datasets/tank14_prepared_v20260629')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260622)
    parser.add_argument(
        '--source-subdir',
        default='exports',
        help='Class-local subdirectory to import. Use an empty value to import each whole class folder.',
    )
    parser.add_argument(
        '--skip-empty-classes',
        action='store_true',
        help='Skip expected classes that have no supported images under the selected source root/subdir.',
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    source_root = Path(args.source_root).resolve()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    detection_root = dataset_root / 'combined_model_name_detection' / 'detection'
    classification_root = dataset_root / 'combined_model_name_classification' / 'crops'

    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f'source root does not exist: {source_root}')
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f'dataset root already exists; pass --overwrite: {dataset_root}')
        resolved_dataset_root = dataset_root.resolve()
        resolved_project_root = PROJECT_ROOT.resolve()
        if not resolved_dataset_root.is_relative_to(resolved_project_root):
            raise RuntimeError(f'refusing to remove outside project: {resolved_dataset_root}')
        shutil.rmtree(_windows_long_path(dataset_root))

    detection_root.mkdir(parents=True, exist_ok=True)
    classification_root.mkdir(parents=True, exist_ok=True)
    metadata_root = dataset_root / 'source_metadata'
    metadata_root.mkdir(parents=True, exist_ok=True)

    candidate_classes = [class_name for class_name in DEFAULT_CLASSES if (source_root / class_name).is_dir()]
    if not candidate_classes:
        raise ValueError(f'no expected class folders found under {source_root}')

    source_subdir = args.source_subdir.strip().replace('\\', '/')
    classification_roots_by_class = {
        class_name: _class_image_roots(source_root=source_root, class_name=class_name, source_subdir=source_subdir)
        for class_name in candidate_classes
    }
    detection_roots_by_class = {
        class_name: _class_detection_root(source_root=source_root, class_name=class_name)
        for class_name in candidate_classes
    }
    classification_images_by_class: dict[str, list[Path]] = {}
    skipped_empty_classes: list[str] = []
    for class_name in candidate_classes:
        paths = _image_paths(classification_roots_by_class[class_name], allow_empty=args.skip_empty_classes)
        if not paths:
            skipped_empty_classes.append(class_name)
            continue
        classification_images_by_class[class_name] = paths
    classes = list(classification_images_by_class)
    if not classes:
        raise ValueError(f'no supported images found under selected source roots: {source_root}')
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    rng = random.Random(args.seed)
    detection_split_counts: Counter[str] = Counter()
    classification_split_counts: Counter[str] = Counter()
    class_split_counts: Counter[str] = Counter()
    detection_records: list[dict[str, Any]] = []
    classification_records: list[dict[str, Any]] = []

    for class_name in classes:
        _copy_source_metadata(source_class_root=source_root / class_name, metadata_root=metadata_root / class_name)
        detection_root_for_class = detection_roots_by_class.get(class_name)
        if detection_root_for_class is not None:
            for record in _iter_class_detection_records(
                source_root=source_root,
                class_name=class_name,
                class_index=class_to_index[class_name],
                detection_source_root=detection_root_for_class,
                output_detection_root=detection_root,
            ):
                detection_records.append(record)
                detection_split_counts[str(record.get('split') or 'train')] += 1
        paths = list(classification_images_by_class[class_name])
        rng.shuffle(paths)
        for index, source_path in enumerate(paths):
            split = _split_for_index(
                index=index,
                total=len(paths),
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            image_id = _safe_stem(f'{class_name}_{index + 1:06d}')
            ext = source_path.suffix.lower() if source_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else '.jpg'
            filename = f'{image_id}{ext}'
            detection_image_rel = Path('images') / split / filename
            detection_image_path = detection_root / detection_image_rel

            classification_image_rel = Path(split) / class_name / filename
            classification_image_path = classification_root / classification_image_rel
            classification_image_path.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(source=source_path, target=classification_image_path)

            width, height = _image_size(source_path)
            object_record = {
                'object_id': f'{image_id}_obj_0001',
                'class_id': class_name,
                'bbox_yolo': [0.5, 0.5, 1.0, 1.0],
                'bbox_xyxy': [0, 0, width, height],
            }
            if detection_root_for_class is None:
                detection_image_path.parent.mkdir(parents=True, exist_ok=True)
                _copy_file(source=source_path, target=detection_image_path)
                detection_records.append({
                    'image_id': image_id,
                    'sample_id': image_id,
                    'path': detection_image_rel.as_posix(),
                    'label': class_name,
                    'split': split,
                    'source': _source_record(source_path=source_path, source_root=source_root),
                    'objects': [object_record],
                    'width': width,
                    'height': height,
                })
                label_rel = Path('labels') / split / f'{image_id}.txt'
                label_path = detection_root / label_rel
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    f'{class_to_index[class_name]} 0.500000 0.500000 1.000000 1.000000\n',
                    encoding='utf-8',
                )
                detection_split_counts[split] += 1
            classification_records.append({
                'image_id': image_id,
                'sample_id': image_id,
                'label': class_name,
                'split': split,
                'path': classification_image_rel.as_posix(),
                'source_image_path': detection_image_rel.as_posix(),
                'source_object_id': object_record['object_id'],
                'bbox_yolo': object_record['bbox_yolo'],
                'crop_box_xyxy': object_record['bbox_xyxy'],
                'input_type': 'object_crop',
            })
            classification_split_counts[split] += 1
            class_split_counts[f'{split}/{class_name}'] += 1

    detection_manifest = {
        'schema_version': '0.1',
        'dataset_id': 'imported_20260617_combined_model_name_detection',
        'dataset_type': 'detection',
        'classes': classes,
        'candidate_classes': candidate_classes,
        'skipped_empty_classes': skipped_empty_classes,
        'images': detection_records,
        'source_root': str(source_root),
        'source_subdir': source_subdir,
        'bbox_policy': 'full_image_bbox_for_class_folder_crops',
    }
    classification_manifest = {
        'schema_version': '0.1',
        'dataset_id': 'imported_20260617_combined_model_name_classification_crops',
        'dataset_type': 'classification',
        'input_type': 'object_crop',
        'source_dataset': str(detection_root),
        'classes': classes,
        'candidate_classes': candidate_classes,
        'skipped_empty_classes': skipped_empty_classes,
        'images': classification_records,
        'source_root': str(source_root),
        'source_subdir': source_subdir,
    }
    _write_json(detection_root / 'manifest.json', detection_manifest)
    _write_json(classification_root / 'manifest.json', classification_manifest)
    _write_detection_yaml(output_root=detection_root, classes=classes)
    coco_result = CocoDetectionMaterializer().materialize(dataset_root=detection_root)

    report = {
        'schema_version': '0.1',
        'source_root': str(source_root),
        'source_subdir': source_subdir,
        'dataset_root': str(dataset_root),
        'classes': classes,
        'candidate_classes': candidate_classes,
        'skipped_empty_classes': skipped_empty_classes,
        'classification_roots_by_class': {
            class_name: [str(path) for path in classification_roots_by_class[class_name]]
            for class_name in candidate_classes
        },
        'detection_roots_by_class': {
            class_name: None if detection_roots_by_class[class_name] is None else str(detection_roots_by_class[class_name])
            for class_name in candidate_classes
        },
        'image_count': len(detection_records),
        'detection_image_count': len(detection_records),
        'classification_image_count': len(classification_records),
        'split_counts': dict(sorted(detection_split_counts.items())),
        'detection_split_counts': dict(sorted(detection_split_counts.items())),
        'classification_split_counts': dict(sorted(classification_split_counts.items())),
        'class_split_counts': dict(sorted(class_split_counts.items())),
        'detection_manifest': str(detection_root / 'manifest.json'),
        'classification_manifest': str(classification_root / 'manifest.json'),
        'bbox_policy': 'full_image_bbox_for_class_folder_crops',
        'coco': coco_result.to_dict(),
    }
    _write_json(dataset_root / 'import_report.json', report)
    _write_json(detection_root.parent / 'merge_report.json', report)
    _write_json(classification_root.parent / 'crop_report.json', {
        **report,
        'crop_count': len(classification_records),
        'target': str(classification_root),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _image_paths(class_roots: tuple[Path, ...], *, allow_empty: bool = False) -> list[Path]:
    paths = [
        path
        for class_root in class_roots
        for path in class_root.rglob('*')
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not paths and not allow_empty:
        roots = ', '.join(str(root) for root in class_roots) or '<none>'
        raise ValueError(f'no supported images found: {roots}')
    return sorted(paths)


def _class_detection_root(*, source_root: Path, class_name: str) -> Path | None:
    detection_root = source_root / class_name / 'detection'
    if (
        detection_root.is_dir()
        and (detection_root / 'images').is_dir()
        and (detection_root / 'labels').is_dir()
    ):
        return detection_root
    return None


def _iter_class_detection_records(
    *,
    source_root: Path,
    class_name: str,
    class_index: int,
    detection_source_root: Path,
    output_detection_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    image_root = detection_source_root / 'images'
    for source_image_path in sorted(
        path
        for path in image_root.rglob('*')
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ):
        rel_under_images = source_image_path.relative_to(image_root)
        split = rel_under_images.parts[0] if len(rel_under_images.parts) > 1 else 'train'
        label_rel = (
            rel_under_images.with_suffix('.txt')
            if len(rel_under_images.parts) > 1
            else Path(split) / f'{source_image_path.stem}.txt'
        )
        label_path = detection_source_root / 'labels' / label_rel
        if not label_path.exists():
            raise FileNotFoundError(f'detection label is missing: {label_path}')
        image_id = _safe_stem(f'{class_name}_{rel_under_images.with_suffix("").as_posix()}')
        dest_image_rel = Path('images') / split / f'{image_id}{source_image_path.suffix.lower()}'
        dest_image_path = output_detection_root / dest_image_rel
        _copy_file(source=source_image_path, target=dest_image_path)

        label_lines: list[str] = []
        objects: list[dict[str, Any]] = []
        for object_index, bbox_yolo in enumerate(_read_yolo_label_boxes(label_path=label_path), start=1):
            label_lines.append(
                f'{class_index} '
                f'{bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} '
                f'{bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}',
            )
            objects.append({
                'object_id': f'{image_id}_obj_{object_index:04d}',
                'class_id': class_name,
                'bbox_yolo': list(bbox_yolo),
            })
        dest_label_path = output_detection_root / 'labels' / split / f'{image_id}.txt'
        dest_label_path.parent.mkdir(parents=True, exist_ok=True)
        dest_label_path.write_text('\n'.join(label_lines) + ('\n' if label_lines else ''), encoding='utf-8')

        width, height = _image_size(dest_image_path)
        for object_record in objects:
            object_record['bbox_xyxy'] = _yolo_to_xyxy(
                bbox_yolo=object_record['bbox_yolo'],
                width=width,
                height=height,
            )
        records.append({
            'image_id': image_id,
            'sample_id': image_id,
            'path': dest_image_rel.as_posix(),
            'label': class_name,
            'split': split,
            'source': _source_record(source_path=source_image_path, source_root=source_root),
            'objects': objects,
            'width': width,
            'height': height,
        })
    return records


def _read_yolo_label_boxes(*, label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding='utf-8').splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            raise ValueError(f'invalid YOLO label line {label_path}:{line_number}: {line}')
        boxes.append(tuple(float(value) for value in parts[1:5]))
    return boxes


def _yolo_to_xyxy(*, bbox_yolo: list[float], width: int, height: int) -> list[float]:
    cx, cy, box_w, box_h = bbox_yolo
    x_min = (cx - box_w / 2.0) * width
    y_min = (cy - box_h / 2.0) * height
    x_max = (cx + box_w / 2.0) * width
    y_max = (cy + box_h / 2.0) * height
    return [x_min, y_min, x_max, y_max]


def _class_image_roots(*, source_root: Path, class_name: str, source_subdir: str) -> tuple[Path, ...]:
    class_root = source_root / class_name
    if not source_subdir:
        return (class_root,)

    # Supported clean source layouts:
    #   D:/data/<class>/exports/...
    #   D:/data/<class>/<class>/exports/...
    #   D:/data/<class>/export_YYYY.../...
    #   D:/data/<class>/tank_model_classification/{train,val,test}/<class>/...
    # Avoid falling back to the whole class folder because it may include raw,
    # review_queue, label_review, duplicate_review_excluded, or excluded_auto data.
    candidates = [
        class_root / source_subdir,
        class_root / class_name / source_subdir,
    ]
    roots = [path for path in candidates if path.is_dir()]
    roots.extend(path for path in sorted(class_root.glob('export_*')) if path.is_dir())
    roots.extend(_prepared_classification_roots(class_root=class_root, class_name=class_name))
    if roots:
        return tuple(dict.fromkeys(roots))

    image_root = class_root / source_subdir
    raise FileNotFoundError(
        f'class source subdir does not exist and no direct export_* or prepared classification folder was found: {image_root}',
    )


def _prepared_classification_roots(*, class_root: Path, class_name: str) -> tuple[Path, ...]:
    classification_root = class_root / 'tank_model_classification'
    roots = []
    for split in ('train', 'val', 'test'):
        split_class_root = classification_root / split / class_name
        if split_class_root.is_dir():
            roots.append(split_class_root)
    return tuple(roots)


def _split_for_index(*, index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio))) if total >= 3 else total
    if index < train_end:
        return 'train'
    if index < val_end:
        return 'val'
    return 'test'


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(_windows_long_path(path)) as image:
        return image.size


def _source_record(*, source_path: Path, source_root: Path) -> dict[str, str]:
    rel = source_path.relative_to(source_root).as_posix()
    return {
        'source_url': '',
        'file_url': '',
        'license': 'unknown',
        'author': 'unknown',
        'attribution': rel,
        'retrieved_at_kst': '',
        'source_path': rel,
    }


def _copy_source_metadata(*, source_class_root: Path, metadata_root: Path) -> None:
    metadata_root.mkdir(parents=True, exist_ok=True)
    for path in source_class_root.iterdir():
        if path.is_file() and path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            _copy_file(source=path, target=metadata_root / path.name)


def _copy_file(*, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_windows_long_path(source), _windows_long_path(target))


def _windows_long_path(path: Path) -> str:
    text = str(path.resolve())
    if os.name != 'nt' or text.startswith('\\\\?\\'):
        return text
    return f'\\\\?\\{text}'


def _safe_stem(value: str) -> str:
    sanitized = ''.join(character if character.isalnum() or character in {'_', '-'} else '_' for character in value)
    return sanitized.strip('_')[:180]


def _write_detection_yaml(*, output_root: Path, classes: list[str]) -> None:
    payload = {
        'path': output_root.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
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
