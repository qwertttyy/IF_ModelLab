from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
from typing import Iterable

from PIL import Image


DATASET_ID = "tank_armor_prepared_v20260630"
SEED = 20260629
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
BROAD_CLASS = {"mbt": ("tank", 0), "av": ("armored_vehicle", 1)}


@dataclass(frozen=True)
class DetectionItem:
    group: str
    model: str
    image: Path
    label: Path
    source_key: str


@dataclass(frozen=True)
class ClassificationItem:
    group: str
    model: str
    image: Path
    source_key: str
    source_folder: str


@dataclass(frozen=True)
class DetectionRecord:
    split: str
    relative_image_path: str
    broad_name: str
    source_model: str
    source_key: str
    width: int
    height: int
    objects: list[dict]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="D:/data")
    parser.add_argument("--dataset-root", default=f"datawork/{DATASET_ID}")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_id = args.dataset_id

    mbt_root = source_root / "mbt"
    av_root = source_root / "av"
    if not mbt_root.is_dir():
        raise FileNotFoundError(mbt_root)
    if not av_root.is_dir():
        raise FileNotFoundError(av_root)

    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_root}")
        _safe_rmtree(dataset_root)

    detector_root = dataset_root / "detector_tank_av" / "detection"
    classifier_roots = {
        "mbt": dataset_root / "classifier_mbt" / "crops",
        "av": dataset_root / "classifier_av" / "crops",
    }
    metadata_root = dataset_root / "source_metadata"
    detector_root.mkdir(parents=True, exist_ok=True)
    for classifier_root in classifier_roots.values():
        classifier_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    detection_items: list[DetectionItem] = []
    classification_items_by_group: dict[str, list[ClassificationItem]] = {"mbt": [], "av": []}
    class_roots: dict[str, dict[str, str]] = {"mbt": {}, "av": {}}
    scan_notes: list[dict[str, str]] = []

    for group, group_root in (("mbt", mbt_root), ("av", av_root)):
        for class_dir in sorted(path for path in group_root.iterdir() if path.is_dir()):
            model = class_dir.name
            class_root = _select_class_root(class_dir)
            if class_root is None:
                scan_notes.append({"group": group, "class": model, "status": "skipped_no_dataset_root"})
                continue
            class_roots[group][model] = str(class_root)
            detection_items.extend(_collect_detection(group, model, class_root))
            classification_items_by_group[group].extend(_collect_classification(group, model, class_root))
            _copy_metadata_files(class_root, metadata_root / group / model)

    all_classification_items = [
        item
        for group_items in classification_items_by_group.values()
        for item in group_items
    ]
    assignments = _make_split_assignments(detection_items, all_classification_items, seed=args.seed)
    det_counts, detection_records = _materialize_detection(detection_items, assignments, detector_root)
    cls_counts = {
        group: _materialize_classification(group_items, assignments, classifier_roots[group])
        for group, group_items in classification_items_by_group.items()
    }

    _write_detection_yaml(detector_root, dataset_id)
    _write_detection_manifest(detector_root, dataset_id, detection_records)
    coco_report = _write_coco_detection(detector_root)
    report = _build_report(
        dataset_id=dataset_id,
        source_root=source_root,
        dataset_root=dataset_root,
        class_roots=class_roots,
        detection_items=detection_items,
        classification_items_by_group=classification_items_by_group,
        detection_counts=det_counts,
        classification_counts=cls_counts,
        coco_report=coco_report,
        scan_notes=scan_notes,
    )
    _write_json(dataset_root / "import_report.json", report)
    _write_json(dataset_root / "split_manifest.json", _split_manifest(assignments))

    print(json.dumps({
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "detector_images": {
            split: det_counts[f"images/{split}"] for split in ("train", "val", "test")
        },
        "detector_labels": {
            split: det_counts[f"labels/{split}"] for split in ("train", "val", "test")
        },
        "classifier_mbt_images": {
            split: sum(
                value
                for key, value in cls_counts["mbt"].items()
                if key.startswith(f"{split}/")
            )
            for split in ("train", "val", "test")
        },
        "classifier_av_images": {
            split: sum(
                value
                for key, value in cls_counts["av"].items()
                if key.startswith(f"{split}/")
            )
            for split in ("train", "val", "test")
        },
        "mbt_classes": sorted(class_roots["mbt"]),
        "av_classes": sorted(class_roots["av"]),
    }, ensure_ascii=False, indent=2))
    return 0


def _select_class_root(class_dir: Path) -> Path | None:
    candidates = [class_dir]
    candidates.extend(path for path in class_dir.glob("export_*") if path.is_dir())
    exports = class_dir / "exports"
    if exports.is_dir():
        candidates.extend(path for path in exports.glob("export_*") if path.is_dir())
    for path in class_dir.rglob("*"):
        if path.is_dir() and ((path / "detection").is_dir() or (path / "tank_model_classification").is_dir()):
            candidates.append(path)

    best: tuple[int, Path] | None = None
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        score = _candidate_score(path)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, path)
    return best[1] if best else None


def _candidate_score(path: Path) -> int:
    det_pairs = 0
    for split in ("train", "val", "test"):
        image_count = _count_files(path / "detection" / "images" / split, IMAGE_EXTS)
        label_count = _count_files(path / "detection" / "labels" / split, {".txt"})
        det_pairs += min(image_count, label_count)
    cls_count = _count_files(path / "tank_model_classification", IMAGE_EXTS)
    return det_pairs * 10 + cls_count


def _collect_detection(group: str, model: str, class_root: Path) -> list[DetectionItem]:
    result: list[DetectionItem] = []
    det_root = class_root / "detection"
    for split in ("train", "val", "test"):
        image_root = det_root / "images" / split
        label_root = det_root / "labels" / split
        if not image_root.is_dir() or not label_root.is_dir():
            continue
        for image in sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            label = label_root / image.relative_to(image_root).with_suffix(".txt")
            if not label.exists():
                continue
            result.append(DetectionItem(
                group=group,
                model=model,
                image=image,
                label=label,
                source_key=_source_key(image),
            ))
    return result


def _collect_classification(group: str, model: str, class_root: Path) -> list[ClassificationItem]:
    result: list[ClassificationItem] = []
    cls_root = class_root / "tank_model_classification"
    for split in ("train", "val", "test"):
        split_root = cls_root / split
        if not split_root.is_dir():
            continue
        for folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for image in sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
                result.append(ClassificationItem(
                    group=group,
                    model=model,
                    image=image,
                    source_key=_source_key(image),
                    source_folder=folder.name,
                ))
    return result


def _source_key(path: Path) -> str:
    stem = path.stem
    if "__bbox_" in stem:
        stem = stem.split("__bbox_", 1)[0]
    return stem


def _make_split_assignments(
    detection_items: Iterable[DetectionItem],
    classification_items: Iterable[ClassificationItem],
    *,
    seed: int,
) -> dict[tuple[str, str, str], str]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in detection_items:
        grouped[(item.group, item.model)].add(item.source_key)
    for item in classification_items:
        grouped[(item.group, item.model)].add(item.source_key)

    assignments: dict[tuple[str, str, str], str] = {}
    for (group, model), keys in sorted(grouped.items()):
        split_keys = _split_keys(sorted(keys), seed + _stable_int(f"{group}/{model}"))
        for split, values in split_keys.items():
            for key in values:
                assignments[(group, model, key)] = split
    return assignments


def _split_keys(keys: list[str], seed: int) -> dict[str, list[str]]:
    values = list(keys)
    random.Random(seed).shuffle(values)
    total = len(values)
    if total == 0:
        return {"train": [], "val": [], "test": []}
    if total < 3:
        return {"train": values, "val": [], "test": []}
    val_count = max(1, round(total * 0.10))
    test_count = max(1, round(total * 0.10))
    train_count = total - val_count - test_count
    if train_count < 1:
        train_count = 1
        overflow = train_count + val_count + test_count - total
        test_count = max(0, test_count - overflow)
    return {
        "train": values[:train_count],
        "val": values[train_count:train_count + val_count],
        "test": values[train_count + val_count:],
    }


def _materialize_detection(
    items: Iterable[DetectionItem],
    assignments: dict[tuple[str, str, str], str],
    detector_root: Path,
) -> tuple[Counter[str], list[DetectionRecord]]:
    counts: Counter[str] = Counter()
    records: list[DetectionRecord] = []
    used: set[Path] = set()
    for item in _dedupe_detection_items(items, assignments):
        split = assignments[(item.group, item.model, item.source_key)]
        broad_name, broad_index = BROAD_CLASS[item.group]
        stem = _safe_name(f"{broad_name}_{item.model}_{item.source_key}")
        image_dst = _unique_path(detector_root / "images" / split / f"{stem}{item.image.suffix.lower()}", used)
        label_dst = detector_root / "labels" / split / f"{image_dst.stem}.txt"
        _link_or_copy(item.image, image_dst)
        width, height = _image_size(image_dst)
        objects = _rewrite_detection_label(item.label, label_dst, broad_index, broad_name, width, height)
        records.append(DetectionRecord(
            split=split,
            relative_image_path=image_dst.relative_to(detector_root).as_posix(),
            broad_name=broad_name,
            source_model=item.model,
            source_key=item.source_key,
            width=width,
            height=height,
            objects=objects,
        ))
        counts[f"images/{split}"] += 1
        counts[f"labels/{split}"] += 1
        counts[f"class/{broad_name}/{split}"] += 1
        counts[f"source_model/{item.model}/{split}"] += 1
    return counts, records


def _dedupe_detection_items(
    items: Iterable[DetectionItem],
    assignments: dict[tuple[str, str, str], str],
) -> list[DetectionItem]:
    split_priority = {"train": 0, "val": 1, "test": 2}
    by_digest: dict[str, list[DetectionItem]] = defaultdict(list)
    for item in items:
        by_digest[_file_sha256(item.image)].append(item)

    selected: list[DetectionItem] = []
    for candidates in by_digest.values():
        selected.append(min(
            candidates,
            key=lambda value: (
                split_priority[assignments[(value.group, value.model, value.source_key)]],
                value.group,
                value.model,
                value.source_key,
                value.image.name,
            ),
        ))

    return sorted(selected, key=lambda value: (value.group, value.model, value.source_key, value.image.name))


def _materialize_classification(
    items: Iterable[ClassificationItem],
    assignments: dict[tuple[str, str, str], str],
    classifier_root: Path,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    used: set[Path] = set()
    for item in sorted(items, key=lambda value: (value.model, value.source_key, value.image.name)):
        split = assignments[(item.group, item.model, item.source_key)]
        stem = _safe_name(item.image.stem)
        image_dst = _unique_path(classifier_root / split / item.model / f"{stem}{item.image.suffix.lower()}", used)
        _link_or_copy(item.image, image_dst)
        counts[f"{split}/{item.model}"] += 1
        if item.source_folder != item.model:
            counts[f"source_folder_alias/{item.model}/{item.source_folder}"] += 1
    return counts


def _rewrite_detection_label(
    source: Path,
    target: Path,
    broad_index: int,
    broad_name: str,
    width: int,
    height: int,
) -> list[dict]:
    lines: list[str] = []
    objects: list[dict] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        bbox = [float(value) for value in parts[1:5]]
        x1, y1, x2, y2 = _yolo_to_xyxy(bbox, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{broad_index} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        objects.append({
            "object_id": f"{target.stem}_{len(objects) + 1:02d}",
            "class_id": broad_name,
            "class_index": broad_index,
            "bbox_yolo": [round(value, 6) for value in bbox],
            "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return objects


def _write_detection_yaml(detector_root: Path, dataset_id: str) -> None:
    text = "\n".join([
        f"path: /workspace/ironflow/prestaged/{dataset_id}/detector_tank_av/detection",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: tank",
        "  1: armored_vehicle",
        "",
    ])
    (detector_root / "data.yaml").write_text(text, encoding="utf-8")


def _write_detection_manifest(detector_root: Path, dataset_id: str, records: list[DetectionRecord]) -> None:
    images = [
        {
            "image_id": f"{record.split}_{index:06d}",
            "sample_id": Path(record.relative_image_path).stem,
            "path": record.relative_image_path,
            "split": record.split,
            "label": record.broad_name,
            "source_model": record.source_model,
            "source_key": record.source_key,
            "width": record.width,
            "height": record.height,
            "objects": record.objects,
        }
        for index, record in enumerate(records, start=1)
    ]
    payload = {
        "schema_version": "0.1",
        "dataset_id": f"{dataset_id}_detector_tank_av",
        "dataset_type": "detection",
        "format": "yolo_detection_with_coco_manifest",
        "classes": ["tank", "armored_vehicle"],
        "images": images,
        "bbox_policy": "source_yolo_normalized_rewritten_to_broad_classes",
    }
    _write_json(detector_root / "manifest.json", payload)


def _write_coco_detection(detector_root: Path) -> dict:
    from importlib import import_module

    module = import_module("ironflow_exp.engine.datasets.coco_detection_materializer")
    result = module.CocoDetectionMaterializer().materialize(dataset_root=detector_root)
    return result.to_dict()


def _build_report(
    *,
    dataset_id: str,
    source_root: Path,
    dataset_root: Path,
    class_roots: dict[str, dict[str, str]],
    detection_items: list[DetectionItem],
    classification_items_by_group: dict[str, list[ClassificationItem]],
    detection_counts: Counter[str],
    classification_counts: dict[str, Counter[str]],
    coco_report: dict,
    scan_notes: list[dict[str, str]],
) -> dict:
    return {
        "schema_version": "0.1",
        "dataset_id": dataset_id,
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "split_policy": "source-key 80/10/10 per model class from merged train/val/test inputs",
        "detector": {
            "classes": ["tank", "armored_vehicle"],
            "source_groups": {
                "tank": sorted(class_roots["mbt"]),
                "armored_vehicle": sorted(class_roots["av"]),
            },
            "counts": dict(sorted(detection_counts.items())),
            "input_pair_count": len(detection_items),
            "coco": coco_report,
        },
        "classifier_mbt": {
            "classes": sorted(class_roots["mbt"]),
            "counts": dict(sorted(classification_counts["mbt"].items())),
            "input_image_count": len(classification_items_by_group["mbt"]),
        },
        "classifier_av": {
            "classes": sorted(class_roots["av"]),
            "counts": dict(sorted(classification_counts["av"].items())),
            "input_image_count": len(classification_items_by_group["av"]),
        },
        "class_roots": class_roots,
        "scan_notes": scan_notes,
    }


def _split_manifest(assignments: dict[tuple[str, str, str], str]) -> dict:
    rows = [
        {"group": group, "class": model, "source_key": source_key, "split": split}
        for (group, model, source_key), split in sorted(assignments.items())
    ]
    counts: Counter[str] = Counter(f"{row['group']}/{row['class']}/{row['split']}" for row in rows)
    return {
        "schema_version": "0.1",
        "split_policy": "source-key 80/10/10 per model class",
        "counts": dict(sorted(counts.items())),
        "assignments": rows,
    }


def _copy_metadata_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("dataset_info.json", "dataset_card.md", "export_manifest.parquet", "export_errors.parquet"):
        src = source / name
        if src.is_file():
            _link_or_copy(src, target / name)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if not resolved.is_relative_to(cwd):
        raise RuntimeError(f"Refusing to remove outside workspace: {resolved}")
    shutil.rmtree(resolved)


def _count_files(path: Path, suffixes: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def _file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _yolo_to_xyxy(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
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


def _unique_path(path: Path, used: set[Path]) -> Path:
    path = Path(str(path)[:240]) if len(str(path)) > 240 else path
    if path not in used and not path.exists():
        used.add(path)
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter:04d}{path.suffix}")
        if candidate not in used and not candidate.exists():
            used.add(candidate)
            return candidate
        counter += 1


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return cleaned[:180] or "item"


def _stable_int(value: str) -> int:
    return int(sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
