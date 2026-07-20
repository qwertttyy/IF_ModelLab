"""Download the shared Drive tank dataset and build the local 14-class dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile
from typing import Any

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ironflow_exp.engine.datasets.coco_detection_materializer import CocoDetectionMaterializer


DRIVE_URL = "https://drive.google.com/drive/folders/YOUR_TEAM_DATASET_FOLDER_ID"
DATASET_ID = "tank14_prepared_v20260629"
CLASSES = (
    "altay",
    "challenger_2",
    "k2",
    "leopard_2",
    "leclerc",
    "m1_abrams",
    "merkava_mk4",
    "type_10",
    "t62",
    "m2020_chonma_2",
    "t72",
    "t90",
    "type_96_ztz96",
    "type_99_ztz99",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-url", default=DRIVE_URL)
    parser.add_argument("--download-root", default=str(PROJECT_ROOT / "runs" / "drive_sources" / "tank14_raw"))
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "runs" / "user_datasets" / DATASET_ID))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-gdown-install", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    download_root = Path(args.download_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if not args.skip_gdown_install:
        _run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    if not args.skip_download:
        download_root.mkdir(parents=True, exist_ok=True)
        _run(["gdown", "--folder", args.drive_url, "-O", str(download_root)])

    _extract_zip_class_archives(download_root)
    class_roots = _find_class_roots(download_root)
    missing = [name for name in CLASSES if name not in class_roots]
    if missing:
        raise FileNotFoundError(f"Missing class folders after Drive download: {missing}")

    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_root}")
        _safe_rmtree(dataset_root)

    detection_root = dataset_root / "combined_model_name_detection" / "detection"
    classification_root = dataset_root / "combined_model_name_classification" / "crops"
    metadata_root = dataset_root / "source_metadata"
    detection_root.mkdir(parents=True, exist_ok=True)
    classification_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    class_to_index = {name: index for index, name in enumerate(CLASSES)}
    detection_records: list[dict[str, Any]] = []
    classification_records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    class_split_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()

    for class_name in CLASSES:
        class_root = class_roots[class_name]
        _copy_source_metadata(class_root, metadata_root / class_name)

        detection_pairs = _detection_pairs(class_root)
        if not detection_pairs:
            raise FileNotFoundError(f"No detection image/label pairs found for {class_name}: {class_root}")
        for split, selected in _split_items(detection_pairs, seed=args.seed + class_to_index[class_name]).items():
            for index, (image_path, label_path) in enumerate(selected, start=1):
                image_id = _safe_stem(f"{class_name}_{split}_{index:06d}")
                dest_image_rel = Path("images") / split / f"{image_id}{image_path.suffix.lower()}"
                dest_label_rel = Path("labels") / split / f"{image_id}.txt"
                dest_image = detection_root / dest_image_rel
                dest_label = detection_root / dest_label_rel
                _copy_file(image_path, dest_image)
                objects = _rewrite_yolo_label(
                    source_label=label_path,
                    target_label=dest_label,
                    class_name=class_name,
                    class_index=class_to_index[class_name],
                    image_path=dest_image,
                    image_id=image_id,
                )
                width, height = _image_size(dest_image)
                detection_records.append({
                    "image_id": image_id,
                    "sample_id": image_id,
                    "path": dest_image_rel.as_posix(),
                    "label": class_name,
                    "split": split,
                    "source": _source_record(image_path, download_root),
                    "objects": objects,
                    "width": width,
                    "height": height,
                })
                split_counts[split] += 1
                class_split_counts[f"detection/{split}/{class_name}"] += 1

        class_images = _classification_images(class_root, class_name)
        if not class_images:
            class_images = [pair[0] for pair in detection_pairs]
        for split, selected in _split_items(class_images, seed=args.seed + 100 + class_to_index[class_name]).items():
            for index, image_path in enumerate(selected, start=1):
                image_id = _safe_stem(f"{class_name}_{split}_{index:06d}")
                dest_rel = Path(split) / class_name / f"{image_id}{image_path.suffix.lower()}"
                dest_image = classification_root / dest_rel
                _copy_file(image_path, dest_image)
                width, height = _image_size(dest_image)
                classification_records.append({
                    "image_id": image_id,
                    "sample_id": image_id,
                    "label": class_name,
                    "split": split,
                    "path": dest_rel.as_posix(),
                    "source": _source_record(image_path, download_root),
                    "bbox_yolo": [0.5, 0.5, 1.0, 1.0],
                    "crop_box_xyxy": [0, 0, width, height],
                    "input_type": "object_crop",
                })
                classification_counts[split] += 1
                class_split_counts[f"classification/{split}/{class_name}"] += 1

    detection_manifest = {
        "schema_version": "0.1",
        "dataset_id": f"{DATASET_ID}_detection",
        "dataset_type": "detection",
        "classes": list(CLASSES),
        "images": detection_records,
        "source_root": str(download_root),
        "bbox_policy": "source_yolo_boxes_resplit_80_10_10",
    }
    classification_manifest = {
        "schema_version": "0.1",
        "dataset_id": f"{DATASET_ID}_classification_crops",
        "dataset_type": "classification",
        "input_type": "object_crop",
        "source_dataset": str(detection_root),
        "classes": list(CLASSES),
        "images": classification_records,
        "source_root": str(download_root),
    }
    _write_json(detection_root / "manifest.json", detection_manifest)
    _write_json(classification_root / "manifest.json", classification_manifest)
    _write_detection_yaml(detection_root)
    coco_result = CocoDetectionMaterializer().materialize(dataset_root=detection_root)

    report = {
        "schema_version": "0.1",
        "dataset_id": DATASET_ID,
        "drive_url": args.drive_url,
        "download_root": str(download_root),
        "dataset_root": str(dataset_root),
        "classes": list(CLASSES),
        "detection_split_counts": dict(sorted(split_counts.items())),
        "classification_split_counts": dict(sorted(classification_counts.items())),
        "class_split_counts": dict(sorted(class_split_counts.items())),
        "coco": coco_result.to_dict(),
    }
    _write_json(dataset_root / "import_report.json", report)
    _write_json(detection_root.parent / "merge_report.json", report)
    _write_json(classification_root.parent / "crop_report.json", {**report, "crop_count": len(classification_records)})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    project = PROJECT_ROOT.resolve()
    if not resolved.is_relative_to(project):
        raise RuntimeError(f"Refusing to remove outside project: {resolved}")
    shutil.rmtree(_windows_long_path(resolved))


def _extract_zip_class_archives(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".zip" and not zipfile.is_zipfile(path):
            continue
        target = path.with_suffix("") if path.suffix.lower() == ".zip" else path.parent / path.stem
        if target.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)


def _find_class_roots(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for class_name in CLASSES:
        candidates = [
            path for path in root.rglob(class_name)
            if path.is_dir() and path.name == class_name
        ]
        candidates.sort(key=_class_root_score, reverse=True)
        if candidates:
            result[class_name] = candidates[0]
    return result


def _class_root_score(path: Path) -> tuple[int, int]:
    score = 0
    if (path / "detection" / "images").is_dir() and (path / "detection" / "labels").is_dir():
        score += 10
    if (path / "tank_model_classification").is_dir():
        score += 5
    if (path / "exports").is_dir():
        score += 3
    score += min(1000, sum(1 for _ in path.rglob("*")))
    return score, -len(path.parts)


def _detection_pairs(class_root: Path) -> list[tuple[Path, Path]]:
    detection_root = class_root / "detection"
    image_root = detection_root / "images"
    label_root = detection_root / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        return []
    pairs = []
    for image in sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
        rel = image.relative_to(image_root)
        label = label_root / rel.with_suffix(".txt")
        if label.exists():
            pairs.append((image, label))
    return pairs


def _classification_images(class_root: Path, class_name: str) -> list[Path]:
    roots: list[Path] = []
    cls_root = class_root / "tank_model_classification"
    for split in ("train", "val", "test"):
        direct = cls_root / split / class_name
        if direct.is_dir():
            roots.append(direct)
    for candidate in (class_root / "exports", class_root / class_name / "exports"):
        if candidate.is_dir():
            roots.append(candidate)
    roots.extend(path for path in sorted(class_root.glob("export_*")) if path.is_dir())
    paths = [
        path for root in dict.fromkeys(roots)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(paths)


def _split_items(items: list[Any], *, seed: int) -> dict[str, list[Any]]:
    values = list(items)
    random.Random(seed).shuffle(values)
    total = len(values)
    if total < 3:
        return {"train": values, "val": [], "test": []}
    val_count = max(1, round(total * 0.10))
    test_count = max(1, round(total * 0.10))
    train_count = total - val_count - test_count
    return {
        "train": values[:train_count],
        "val": values[train_count:train_count + val_count],
        "test": values[train_count + val_count:],
    }


def _rewrite_yolo_label(
    *,
    source_label: Path,
    target_label: Path,
    class_name: str,
    class_index: int,
    image_path: Path,
    image_id: str,
) -> list[dict[str, Any]]:
    width, height = _image_size(image_path)
    lines = []
    objects = []
    for object_index, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line in {source_label}: {line}")
        bbox = [float(value) for value in parts[1:5]]
        lines.append(f"{class_index} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        objects.append({
            "object_id": f"{image_id}_obj_{object_index:04d}",
            "class_id": class_name,
            "bbox_yolo": bbox,
            "bbox_xyxy": _yolo_to_xyxy(bbox, width, height),
        })
    target_label.parent.mkdir(parents=True, exist_ok=True)
    target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return objects


def _yolo_to_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    cx, cy, box_w, box_h = bbox
    return [
        (cx - box_w / 2.0) * width,
        (cy - box_h / 2.0) * height,
        (cx + box_w / 2.0) * width,
        (cy + box_h / 2.0) * height,
    ]


def _write_detection_yaml(root: Path) -> None:
    payload = {
        "path": root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: class_name for index, class_name in enumerate(CLASSES)},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _copy_source_metadata(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for path in source_root.iterdir():
        if path.is_file() and path.suffix.lower() not in IMAGE_EXTS:
            _copy_file(path, target_root / path.name)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_windows_long_path(source), _windows_long_path(target))


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(_windows_long_path(path)) as image:
        return image.size


def _source_record(path: Path, root: Path) -> dict[str, str]:
    rel = path.relative_to(root).as_posix()
    return {"source_path": rel, "source_url": "", "file_url": "", "license": "unknown", "author": "unknown"}


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")[:180]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _windows_long_path(path: Path) -> str:
    text = str(path.resolve())
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    return f"\\\\?\\{text}"


if __name__ == "__main__":
    raise SystemExit(main())
