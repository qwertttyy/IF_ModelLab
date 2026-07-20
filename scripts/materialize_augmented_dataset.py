"""Materialize train-only weak augmentation datasets for IronFlow.

This script keeps the prepared dataset layout intact:

  detector_tank_av/detection
  classifier_mbt/crops
  classifier_av/crops

Only train split receives augmented copies. Validation and test files are
linked/copied unchanged so evaluation remains comparable with the baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ('train', 'val', 'test')
DEFAULT_INPUT_ROOT = 'runs/user_datasets/tank_armor_prepared_v20260630'
DEFAULT_POLICY = 'weak_v1'
REMOTE_PRESTAGED_ROOT = '/workspace/ironflow/prestaged'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-root', default='')
    parser.add_argument('--policy', default=DEFAULT_POLICY, choices=('weak_v1', 'light_v1', 'medium_v1'))
    parser.add_argument('--copies-per-train-image', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260702)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--include',
        nargs='+',
        default=('detection', 'classifier_mbt', 'classifier_av'),
        choices=('detection', 'classifier_mbt', 'classifier_av'),
        help='Dataset branches to augment. All existing branches are copied regardless.',
    )
    args = parser.parse_args()

    input_root = _resolve_project_path(args.input_root)
    output_root = _resolve_project_path(
        args.output_root or f'{args.input_root.rstrip("/").rstrip("\\")}_aug_{args.policy}'
    )
    if args.copies_per_train_image < 1:
        raise ValueError('--copies-per-train-image must be at least 1')
    if not input_root.exists():
        raise FileNotFoundError(f'input dataset root not found: {input_root}')
    if input_root.resolve() == output_root.resolve():
        raise ValueError('output root must be different from input root')

    _prepare_output_root(input_root=input_root, output_root=output_root, overwrite=args.overwrite)
    _clone_tree_with_links(source=input_root, target=output_root)

    report: dict[str, Any] = {
        'schema_version': '0.1',
        'source_dataset_root': str(input_root),
        'output_dataset_root': str(output_root),
        'policy': args.policy,
        'copies_per_train_image': args.copies_per_train_image,
        'seed': args.seed,
        'split_policy': 'augment train only; keep val/test unchanged',
        'branches': {},
    }

    if 'detection' in args.include:
        detection_root = output_root / 'detector_tank_av' / 'detection'
        if detection_root.exists():
            report['branches']['detection'] = _augment_detection_branch(
                detection_root=detection_root,
                policy=args.policy,
                copies_per_train_image=args.copies_per_train_image,
                seed=args.seed,
                dataset_root_name=output_root.name,
            )

    for branch_name, branch_rel in (
        ('classifier_mbt', Path('classifier_mbt') / 'crops'),
        ('classifier_av', Path('classifier_av') / 'crops'),
    ):
        if branch_name not in args.include:
            continue
        classifier_root = output_root / branch_rel
        if classifier_root.exists():
            report['branches'][branch_name] = _augment_classification_branch(
                classifier_root=classifier_root,
                policy=args.policy,
                copies_per_train_image=args.copies_per_train_image,
                seed=args.seed + _stable_int(branch_name),
            )

    report_path = output_root / f'augmentation_report_{args.policy}.json'
    _write_json(report_path, report)
    print(json.dumps(_compact_report(report=report, report_path=report_path), ensure_ascii=False, indent=2))
    return 0


def _augment_detection_branch(
    *,
    detection_root: Path,
    policy: str,
    copies_per_train_image: int,
    seed: int,
    dataset_root_name: str,
) -> dict[str, Any]:
    transform = _detection_transform(policy=policy)
    train_images = _image_files(detection_root / 'images' / 'train')
    original_manifest = _read_json_optional(detection_root / 'manifest.json') or {}
    manifest_records = list(original_manifest.get('images') or [])
    manifest_by_path = {
        str(record.get('path')).replace('\\', '/'): record
        for record in manifest_records
        if isinstance(record, dict) and record.get('path')
    }
    classes = _detection_classes(detection_root=detection_root, manifest=original_manifest)
    counts: Counter[str] = Counter()
    augmented_records: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(train_images, start=1):
        label_path = detection_root / 'labels' / 'train' / f'{image_path.stem}.txt'
        if not label_path.exists():
            counts['skipped_missing_label'] += 1
            continue
        bboxes, class_labels = _read_yolo_label(path=label_path)
        if not bboxes:
            counts['skipped_empty_label'] += 1
            continue
        rel_image = image_path.relative_to(detection_root).as_posix()
        source_record = manifest_by_path.get(rel_image)

        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            image_array = np.array(rgb_image)

        for copy_index in range(1, copies_per_train_image + 1):
            local_seed = seed + image_index * 1000 + copy_index
            random.seed(local_seed)
            np.random.seed(local_seed % (2**32 - 1))
            transformed = transform(image=image_array, bboxes=bboxes, class_labels=class_labels)
            new_bboxes = [
                _clip_yolo_bbox([float(value) for value in bbox])
                for bbox in transformed.get('bboxes', [])
            ]
            new_class_labels = [int(value) for value in transformed.get('class_labels', [])]
            valid_pairs = [
                (class_id, bbox)
                for class_id, bbox in zip(new_class_labels, new_bboxes, strict=False)
                if _valid_yolo_bbox(bbox)
            ]
            if not valid_pairs:
                counts['skipped_no_valid_bbox_after_aug'] += 1
                continue

            aug_stem = f'{image_path.stem}__aug_{policy}_{copy_index:02d}'
            aug_image_path = image_path.with_name(f'{aug_stem}{image_path.suffix.lower()}')
            aug_label_path = label_path.with_name(f'{aug_stem}.txt')
            Image.fromarray(transformed['image']).save(aug_image_path, quality=95)
            _write_yolo_label(path=aug_label_path, pairs=valid_pairs)
            counts['augmented_images'] += 1
            counts['augmented_labels'] += 1

            width, height = _image_size(aug_image_path)
            aug_record = _augmented_detection_record(
                source_record=source_record,
                image_rel=aug_image_path.relative_to(detection_root).as_posix(),
                class_labels=[pair[0] for pair in valid_pairs],
                bboxes=[pair[1] for pair in valid_pairs],
                classes=classes,
                width=width,
                height=height,
                policy=policy,
                seed=local_seed,
            )
            augmented_records.append(aug_record)

    if manifest_records or augmented_records:
        manifest = dict(original_manifest)
        manifest['schema_version'] = str(manifest.get('schema_version') or '0.1')
        manifest['dataset_type'] = 'detection'
        manifest['classes'] = classes
        manifest['images'] = [*manifest_records, *augmented_records]
        manifest['augmentation'] = {
            'policy': policy,
            'scope': 'train_only',
            'augmented_image_count': counts['augmented_images'],
            'note': 'val/test records are copied unchanged from the source dataset',
        }
        _write_json(detection_root / 'manifest.json', manifest)

    _write_detection_data_yaml(detection_root=detection_root, classes=classes, dataset_root_name=dataset_root_name)
    coco_report = _materialize_coco(detection_root=detection_root)

    return {
        'policy': policy,
        'library': 'albumentations',
        'input_train_images': len(train_images),
        'counts': dict(sorted(counts.items())),
        'coco': coco_report,
    }


def _augment_classification_branch(
    *,
    classifier_root: Path,
    policy: str,
    copies_per_train_image: int,
    seed: int,
) -> dict[str, Any]:
    transform = _classification_transform(policy=policy)
    train_root = classifier_root / 'train'
    train_images = _image_files(train_root)
    original_manifest = _read_json_optional(classifier_root / 'manifest.json') or {}
    manifest_records = list(original_manifest.get('images') or [])
    manifest_by_path = {
        str(record.get('path')).replace('\\', '/'): record
        for record in manifest_records
        if isinstance(record, dict) and record.get('path')
    }
    counts: Counter[str] = Counter()
    augmented_records: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(train_images, start=1):
        try:
            class_name = image_path.relative_to(train_root).parts[0]
        except IndexError:
            counts['skipped_no_class'] += 1
            continue
        rel_image = image_path.relative_to(classifier_root).as_posix()
        source_record = manifest_by_path.get(rel_image)

        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            for copy_index in range(1, copies_per_train_image + 1):
                local_seed = seed + image_index * 1000 + copy_index
                _seed_torchvision(local_seed)
                aug_stem = f'{image_path.stem}__aug_{policy}_{copy_index:02d}'
                aug_image_path = image_path.with_name(f'{aug_stem}{image_path.suffix.lower()}')
                transformed = transform(rgb_image)
                transformed.save(aug_image_path, quality=95)
                counts['augmented_images'] += 1
                counts[f'augmented_class/{class_name}'] += 1
                augmented_records.append(
                    _augmented_classification_record(
                        source_record=source_record,
                        image_rel=aug_image_path.relative_to(classifier_root).as_posix(),
                        class_name=class_name,
                        policy=policy,
                        seed=local_seed,
                    )
                )

    if manifest_records or augmented_records:
        classes = _classification_classes(classifier_root=classifier_root, manifest=original_manifest)
        manifest = dict(original_manifest)
        manifest['schema_version'] = str(manifest.get('schema_version') or '0.1')
        manifest['dataset_type'] = 'classification'
        manifest['classes'] = classes
        manifest['images'] = [*manifest_records, *augmented_records]
        manifest['augmentation'] = {
            'policy': policy,
            'scope': 'train_only',
            'augmented_image_count': counts['augmented_images'],
            'note': 'val/test records are copied unchanged from the source dataset',
        }
        _write_json(classifier_root / 'manifest.json', manifest)

    return {
        'policy': policy,
        'library': 'torchvision.transforms',
        'input_train_images': len(train_images),
        'counts': dict(sorted(counts.items())),
    }


def _detection_transform(*, policy: str) -> Any:
    try:
        import albumentations as A
    except ImportError as error:
        raise RuntimeError('albumentations is required for detection augmentation') from error

    if policy in {'weak_v1', 'light_v1'}:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=8, p=0.35),
        ]
    else:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.85),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=10, p=0.45),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-3, 3),
                shear=(-1, 1),
                fit_output=False,
                keep_ratio=True,
                p=0.35,
            ),
        ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=4,
            min_visibility=0.05,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


def _classification_transform(*, policy: str) -> Any:
    try:
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError('torchvision is required for classification augmentation') from error

    if policy in {'weak_v1', 'light_v1'}:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ])

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
        transforms.RandomApply([transforms.RandomRotation(degrees=3)], p=0.25),
    ])


def _clone_tree_with_links(*, source: Path, target: Path) -> None:
    for current_root, dir_names, file_names in os.walk(source):
        current = Path(current_root)
        relative_root = current.relative_to(source)
        target_root = target / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for dir_name in dir_names:
            (target_root / dir_name).mkdir(exist_ok=True)
        for file_name in file_names:
            src = current / file_name
            dst = target_root / file_name
            if dst.exists():
                continue
            _link_or_copy(src, dst)


def _prepare_output_root(*, input_root: Path, output_root: Path, overwrite: bool) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f'output root already exists; pass --overwrite to replace it: {output_root}')
    _safe_rmtree(path=output_root, input_root=input_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _safe_rmtree(*, path: Path, input_root: Path) -> None:
    resolved = path.resolve()
    input_resolved = input_root.resolve()
    if resolved == input_resolved:
        raise RuntimeError(f'refusing to remove input root: {resolved}')
    if resolved.anchor == str(resolved):
        raise RuntimeError(f'refusing to remove drive/root path: {resolved}')
    if '_aug_' not in resolved.name:
        raise RuntimeError(f"refusing to remove output without '_aug_' in folder name: {resolved}")
    shutil.rmtree(resolved)


def _materialize_coco(*, detection_root: Path) -> dict[str, Any]:
    from ironflow_exp.engine.datasets.coco_detection_materializer import CocoDetectionMaterializer

    result = CocoDetectionMaterializer().materialize(dataset_root=detection_root)
    return result.to_dict()


def _write_detection_data_yaml(*, detection_root: Path, classes: list[str], dataset_root_name: str) -> None:
    payload = {
        'path': f'{REMOTE_PRESTAGED_ROOT}/{dataset_root_name}/detector_tank_av/detection',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {index: class_name for index, class_name in enumerate(classes)},
    }
    (detection_root / 'data.yaml').write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )


def _augmented_detection_record(
    *,
    source_record: dict[str, Any] | None,
    image_rel: str,
    class_labels: list[int],
    bboxes: list[list[float]],
    classes: list[str],
    width: int,
    height: int,
    policy: str,
    seed: int,
) -> dict[str, Any]:
    stem = Path(image_rel).stem
    record = deepcopy(source_record) if source_record is not None else {}
    objects = []
    for index, (class_index, bbox) in enumerate(zip(class_labels, bboxes, strict=False), start=1):
        class_name = classes[class_index] if 0 <= class_index < len(classes) else str(class_index)
        objects.append({
            'object_id': f'{stem}_{index:02d}',
            'class_id': class_name,
            'class_index': class_index,
            'bbox_yolo': [round(value, 6) for value in bbox],
            'bbox_xyxy': [round(value, 3) for value in _yolo_to_xyxy(bbox=bbox, width=width, height=height)],
        })
    record.update({
        'image_id': stem,
        'sample_id': stem,
        'path': image_rel,
        'split': 'train',
        'width': width,
        'height': height,
        'objects': objects,
        'augmentation_policy': policy,
        'augmentation_seed': seed,
        'source_type': 'augmentation',
    })
    if objects:
        record['label'] = objects[0]['class_id']
    return record


def _augmented_classification_record(
    *,
    source_record: dict[str, Any] | None,
    image_rel: str,
    class_name: str,
    policy: str,
    seed: int,
) -> dict[str, Any]:
    stem = Path(image_rel).stem
    record = deepcopy(source_record) if source_record is not None else {}
    record.update({
        'image_id': stem,
        'sample_id': stem,
        'path': image_rel,
        'split': 'train',
        'label': class_name,
        'augmentation_policy': policy,
        'augmentation_seed': seed,
        'source_type': 'augmentation',
    })
    return record


def _read_yolo_label(*, path: Path) -> tuple[list[list[float]], list[int]]:
    bboxes: list[list[float]] = []
    class_labels: list[int] = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        class_labels.append(int(float(parts[0])))
        bboxes.append(_clip_yolo_bbox([float(value) for value in parts[1:5]]))
    return bboxes, class_labels


def _write_yolo_label(*, path: Path, pairs: Iterable[tuple[int, list[float]]]) -> None:
    lines = [
        f'{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}'
        for class_id, bbox in pairs
    ]
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def _clip_yolo_bbox(values: list[float]) -> list[float]:
    cx, cy, width, height = values
    return [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, width)),
        max(0.0, min(1.0, height)),
    ]


def _valid_yolo_bbox(values: list[float]) -> bool:
    return 0.0 <= values[0] <= 1.0 and 0.0 <= values[1] <= 1.0 and values[2] > 0.0 and values[3] > 0.0


def _detection_classes(*, detection_root: Path, manifest: dict[str, Any]) -> list[str]:
    classes = manifest.get('classes')
    if isinstance(classes, list) and classes:
        return [str(value) for value in classes]
    data_yaml = detection_root / 'data.yaml'
    if data_yaml.exists():
        payload = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
        names = payload.get('names') if isinstance(payload, dict) else None
        if isinstance(names, list):
            return [str(value) for value in names]
        if isinstance(names, dict):
            return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    return ['tank', 'armored_vehicle']


def _classification_classes(*, classifier_root: Path, manifest: dict[str, Any]) -> list[str]:
    classes = manifest.get('classes')
    if isinstance(classes, list) and classes:
        return [str(value) for value in classes]
    names = set()
    for split in SPLITS:
        split_root = classifier_root / split
        if split_root.exists():
            names.update(child.name for child in split_root.iterdir() if child.is_dir())
    return sorted(names)


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _yolo_to_xyxy(*, bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, box_width, box_height = bbox
    abs_width = box_width * width
    abs_height = box_height * height
    x1 = (cx * width) - (abs_width / 2.0)
    y1 = (cy * height) - (abs_height / 2.0)
    x2 = x1 + abs_width
    y2 = y1 + abs_height
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _seed_torchvision(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'expected JSON object: {path}')
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _stable_int(value: str) -> int:
    import hashlib

    return int(hashlib.sha256(value.encode('utf-8')).hexdigest()[:8], 16)


def _compact_report(*, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        'output_dataset_root': report['output_dataset_root'],
        'policy': report['policy'],
        'copies_per_train_image': report['copies_per_train_image'],
        'branches': {
            name: {
                'library': branch.get('library'),
                'input_train_images': branch.get('input_train_images'),
                'augmented_images': (branch.get('counts') or {}).get('augmented_images', 0),
            }
            for name, branch in report.get('branches', {}).items()
            if isinstance(branch, dict)
        },
        'report_path': str(report_path),
    }


if __name__ == '__main__':
    raise SystemExit(main())
