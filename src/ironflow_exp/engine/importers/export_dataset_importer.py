import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from ironflow_exp.engine.core import UserImageDatasetValidator
from ironflow_exp.engine.datasets import CocoDetectionMaterializer


IMAGE_SUFFIXES = frozenset({'.jpg', '.jpeg', '.png', '.ppm'})
EXPORT_RUN_NAME_PATTERN = re.compile(r'^exports?_\d{8}_\d{6}$')


@dataclass(frozen=True, slots=True)
class ExportDatasetImportResult:
    source_root: Path
    output_root: Path
    detection_manifest_path: Path | None = None
    classification_manifest_path: Path | None = None
    detection_data_yaml_path: Path | None = None
    detection_coco_root: Path | None = None
    detection_coco_annotation_paths: dict[str, Path] = field(default_factory=dict)
    report_path: Path | None = None
    detection_image_count: int = 0
    detection_object_count: int = 0
    classification_image_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            'source_root': str(self.source_root),
            'output_root': str(self.output_root),
            'detection_manifest_path': None if self.detection_manifest_path is None else str(self.detection_manifest_path),
            'classification_manifest_path': None if self.classification_manifest_path is None else str(self.classification_manifest_path),
            'detection_data_yaml_path': None if self.detection_data_yaml_path is None else str(self.detection_data_yaml_path),
            'detection_coco_root': None if self.detection_coco_root is None else str(self.detection_coco_root),
            'detection_coco_annotation_paths': {
                split: str(path)
                for split, path in self.detection_coco_annotation_paths.items()
            },
            'report_path': None if self.report_path is None else str(self.report_path),
            'detection_image_count': self.detection_image_count,
            'detection_object_count': self.detection_object_count,
            'classification_image_count': self.classification_image_count,
            'warnings': self.warnings,
        }


class ExportDatasetImporter:
    def resolve_export_root(self, source_root: str | Path) -> Path:
        return self._resolve_export_root(Path(source_root))

    def import_export(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path,
        replace_existing: bool = False,
    ) -> ExportDatasetImportResult:
        source = self._resolve_export_root(Path(source_root))
        output = Path(output_root).resolve()
        warnings: list[str] = []

        self._validate_export_root(source=source)
        self._prepare_output_root(source=source, output=output, replace_existing=replace_existing)
        dataset_info = self._read_dataset_info(source=source)

        detection_result = self._import_detection(
            source=source,
            output=output,
            dataset_info=dataset_info,
            warnings=warnings,
        )
        classification_result = self._import_classification(
            source=source,
            output=output,
            dataset_info=dataset_info,
            warnings=warnings,
        )

        report = {
            'schema_version': '0.1',
            'source_root': str(source),
            'output_root': str(output),
            'dataset_info': dataset_info,
            'detection': detection_result,
            'classification': classification_result,
            'warnings': warnings,
        }
        report_path = output / 'import_report.json'
        self._write_json(report_path, report)

        return ExportDatasetImportResult(
            source_root=source,
            output_root=output,
            detection_manifest_path=output / 'detection' / 'manifest.json',
            classification_manifest_path=Path(str(classification_result['manifest_path']))
            if classification_result.get('manifest_path') is not None
            else None,
            detection_data_yaml_path=output / 'detection' / 'data.yaml',
            detection_coco_root=output / 'detection' / 'coco',
            detection_coco_annotation_paths={
                split: Path(str(path))
                for split, path in detection_result.get('coco_annotation_paths', {}).items()
            } if isinstance(detection_result.get('coco_annotation_paths'), dict) else {},
            report_path=report_path,
            detection_image_count=int(detection_result['image_count']),
            detection_object_count=int(detection_result['object_count']),
            classification_image_count=int(classification_result['image_count']),
            warnings=warnings,
        )

    def _resolve_export_root(self, source: Path) -> Path:
        candidate = source.resolve()
        if (candidate / 'dataset_info.json').exists():
            return candidate

        if not candidate.exists() or not candidate.is_dir():
            return candidate

        export_children = [
            child
            for child in candidate.iterdir()
            if child.is_dir()
            and EXPORT_RUN_NAME_PATTERN.match(child.name)
            and (child / 'dataset_info.json').exists()
        ]
        if len(export_children) == 1:
            return export_children[0].resolve()
        if len(export_children) > 1:
            return max(export_children, key=lambda path: path.stat().st_mtime).resolve()

        return candidate

    def _validate_export_root(self, *, source: Path) -> None:
        required = [
            source / 'dataset_info.json',
            source / 'detection' / 'data.yaml',
            source / 'detection' / 'images' / 'train',
            source / 'detection' / 'labels' / 'train',
        ]
        missing = [
            path
            for path in required
            if not path.exists()
        ]
        if missing:
            joined = ', '.join(str(path) for path in missing)
            raise ValueError(f'not a supported export dataset folder; missing: {joined}')

    def _prepare_output_root(self, *, source: Path, output: Path, replace_existing: bool) -> None:
        if source == output:
            raise ValueError('output_root must be different from source_root')
        if output.exists() and any(output.iterdir()):
            if not replace_existing:
                raise FileExistsError(f'output_root already exists and is not empty: {output}')
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

    def _read_dataset_info(self, *, source: Path) -> dict[str, Any]:
        path = source / 'dataset_info.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('dataset_info.json root must be an object')

        return data

    def _import_detection(
        self,
        *,
        source: Path,
        output: Path,
        dataset_info: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, object]:
        source_detection = source / 'detection'
        output_detection = output / 'detection'
        self._copy_tree_contents(source=source_detection, output=output_detection)
        class_map = self._detection_class_map(dataset_info=dataset_info, source_detection=source_detection)
        classes = [
            class_map[index]
            for index in sorted(class_map)
        ]
        images: list[dict[str, object]] = []
        object_count = 0

        for image_path in self._iter_split_images(output_detection / 'images'):
            split = image_path.parent.name
            relative_path = image_path.relative_to(output_detection).as_posix()
            with Image.open(image_path) as image:
                width, height = image.size
            label_path = output_detection / 'labels' / split / f'{image_path.stem}.txt'
            objects = self._objects_from_yolo_label(
                label_path=label_path,
                class_map=class_map,
                image_width=width,
                image_height=height,
                item_id=image_path.stem,
                start_index=object_count,
            )
            object_count += len(objects)
            label = str(objects[0]['class_id']) if objects else (classes[0] if classes else 'unknown')
            images.append(
                {
                    'image_id': f'img_{self._safe_id(image_path.stem)}',
                    'sample_id': f'sample_{self._safe_id(image_path.stem)}',
                    'path': relative_path,
                    'width': width,
                    'height': height,
                    'label': label,
                    'split': split,
                    'source': self._source_metadata(source=source, dataset_info=dataset_info, image_name=image_path.name),
                    'objects': objects,
                },
            )

        manifest = {
            'schema_version': '0.1',
            'dataset_id': f"{self._dataset_id(dataset_info=dataset_info)}_detection",
            'dataset_type': 'detection',
            'classes': classes,
            'images': images,
        }
        self._write_json(output_detection / 'manifest.json', manifest)
        self._write_fixed_data_yaml(output_detection=output_detection, class_map=class_map)
        coco_result = CocoDetectionMaterializer().materialize(dataset_root=output_detection)
        validation = UserImageDatasetValidator(max_images=max(100, len(images))).validate_folder(output_detection)
        if not validation.is_valid:
            raise ValueError('generated detection manifest is invalid: ' + '; '.join(validation.error_messages()))
        if not images:
            warnings.append('detection import produced zero images')

        return {
            'manifest_path': str(output_detection / 'manifest.json'),
            'data_yaml_path': str(output_detection / 'data.yaml'),
            'coco_root': str(coco_result.coco_root),
            'coco_annotation_paths': {
                split: str(path)
                for split, path in coco_result.annotation_paths.items()
            },
            'image_count': len(images),
            'object_count': object_count,
            'classes': classes,
        }

    def _import_classification(
        self,
        *,
        source: Path,
        output: Path,
        dataset_info: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, object]:
        source_classification = source / 'tank_model_classification'
        output_classification = output / 'classification'
        if not source_classification.exists():
            warnings.append('classification export folder is missing; detection-only import was produced')
            return {
                'manifest_path': None,
                'image_count': 0,
                'classes': [],
            }
        self._copy_tree_contents(source=source_classification, output=output_classification)
        class_names = self._classification_classes(root=output_classification)
        images: list[dict[str, object]] = []

        for image_path in self._iter_classification_images(output_classification):
            split = image_path.parents[1].name
            label = image_path.parent.name
            relative_path = image_path.relative_to(output_classification).as_posix()
            with Image.open(image_path) as image:
                width, height = image.size
            images.append(
                {
                    'image_id': f'img_{self._safe_id(image_path.stem)}',
                    'sample_id': f'sample_{self._safe_id(image_path.stem)}',
                    'path': relative_path,
                    'width': width,
                    'height': height,
                    'label': label,
                    'split': split,
                    'source': self._source_metadata(source=source, dataset_info=dataset_info, image_name=image_path.name),
                    'objects': [],
                },
            )

        manifest = {
            'schema_version': '0.1',
            'dataset_id': f"{self._dataset_id(dataset_info=dataset_info)}_classification",
            'dataset_type': 'classification',
            'classes': class_names,
            'images': images,
        }
        self._write_json(output_classification / 'manifest.json', manifest)
        validation = UserImageDatasetValidator(max_images=max(100, len(images))).validate_folder(output_classification)
        if not validation.is_valid:
            raise ValueError('generated classification manifest is invalid: ' + '; '.join(validation.error_messages()))
        if not images:
            warnings.append('classification import produced zero images')

        return {
            'manifest_path': str(output_classification / 'manifest.json'),
            'image_count': len(images),
            'classes': class_names,
        }

    def _copy_tree_contents(self, *, source: Path, output: Path) -> None:
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            destination = output / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)

    def _detection_class_map(self, *, dataset_info: dict[str, Any], source_detection: Path) -> dict[int, str]:
        raw_map = dataset_info.get('detection_class_map')
        if isinstance(raw_map, dict) and raw_map:
            return {
                int(class_index): str(class_name)
                for class_name, class_index in raw_map.items()
            }

        data_yaml = yaml.safe_load((source_detection / 'data.yaml').read_text(encoding='utf-8'))
        if not isinstance(data_yaml, dict) or not isinstance(data_yaml.get('names'), dict):
            raise ValueError('detection class map is missing from dataset_info.json and data.yaml')

        return {
            int(class_index): str(class_name)
            for class_index, class_name in data_yaml['names'].items()
        }

    def _classification_classes(self, *, root: Path) -> list[str]:
        classes: set[str] = set()
        for split in ('train', 'val', 'test'):
            split_root = root / split
            if not split_root.exists():
                continue
            classes.update(
                class_dir.name
                for class_dir in split_root.iterdir()
                if class_dir.is_dir()
            )
        if not classes:
            raise ValueError('classification export has no class folders')

        return sorted(classes)

    def _iter_split_images(self, image_root: Path) -> list[Path]:
        if not image_root.exists():
            return []

        return sorted(
            path
            for split_dir in image_root.iterdir()
            if split_dir.is_dir()
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _iter_classification_images(self, root: Path) -> list[Path]:
        return sorted(
            path
            for split in ('train', 'val', 'test')
            if (root / split).exists()
            for class_dir in (root / split).iterdir()
            if class_dir.is_dir()
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _objects_from_yolo_label(
        self,
        *,
        label_path: Path,
        class_map: dict[int, str],
        image_width: int,
        image_height: int,
        item_id: str,
        start_index: int,
    ) -> list[dict[str, object]]:
        if not label_path.exists():
            return []

        objects: list[dict[str, object]] = []
        for line_index, line in enumerate(label_path.read_text(encoding='utf-8').splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f'invalid YOLO label field count at {label_path}:{line_index}')
            class_index = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
            self._validate_yolo_bbox(label_path=label_path, line_index=line_index, class_index=class_index, values=(x_center, y_center, width, height))
            objects.append(
                {
                    'object_id': f'obj_{self._safe_id(item_id)}_{start_index + len(objects) + 1:08d}',
                    'class_id': class_map.get(class_index, str(class_index)),
                    'bbox_xyxy': self._yolo_to_xyxy(
                        x_center=x_center,
                        y_center=y_center,
                        bbox_width=width,
                        bbox_height=height,
                        image_width=image_width,
                        image_height=image_height,
                    ),
                    'bbox_yolo': [x_center, y_center, width, height],
                    'label_line_index': line_index - 1,
                },
            )

        return objects

    def _validate_yolo_bbox(
        self,
        *,
        label_path: Path,
        line_index: int,
        class_index: int,
        values: tuple[float, float, float, float],
    ) -> None:
        if class_index < 0:
            raise ValueError(f'invalid YOLO class id at {label_path}:{line_index}')
        x_center, y_center, width, height = values
        if not 0.0 <= x_center <= 1.0:
            raise ValueError(f'invalid YOLO x_center at {label_path}:{line_index}')
        if not 0.0 <= y_center <= 1.0:
            raise ValueError(f'invalid YOLO y_center at {label_path}:{line_index}')
        if not 0.0 < width <= 1.0:
            raise ValueError(f'invalid YOLO width at {label_path}:{line_index}')
        if not 0.0 < height <= 1.0:
            raise ValueError(f'invalid YOLO height at {label_path}:{line_index}')

    def _yolo_to_xyxy(
        self,
        *,
        x_center: float,
        y_center: float,
        bbox_width: float,
        bbox_height: float,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        center_x = x_center * image_width
        center_y = y_center * image_height
        width = bbox_width * image_width
        height = bbox_height * image_height

        return [
            max(0.0, center_x - width / 2.0),
            max(0.0, center_y - height / 2.0),
            min(float(image_width), center_x + width / 2.0),
            min(float(image_height), center_y + height / 2.0),
        ]

    def _write_fixed_data_yaml(self, *, output_detection: Path, class_map: dict[int, str]) -> None:
        data = {
            'path': output_detection.resolve().as_posix(),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                index: class_map[index]
                for index in sorted(class_map)
            },
            'nc': len(class_map),
        }
        test_dir = output_detection / 'images' / 'test'
        if test_dir.exists():
            data['test'] = 'images/test'

        with (output_detection / 'data.yaml').open(mode='w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)

    def _dataset_id(self, *, dataset_info: dict[str, Any]) -> str:
        version = str(dataset_info.get('dataset_version') or 'export_dataset')
        target = str(dataset_info.get('target_name') or '').strip()
        raw = f'{target}_{version}' if target else version

        return self._safe_id(raw).lower()

    def _source_metadata(self, *, source: Path, dataset_info: dict[str, Any], image_name: str) -> dict[str, object]:
        created_at = str(dataset_info.get('created_at') or '')
        return {
            'source_url': str(source),
            'file_url': image_name,
            'license': 'user_provided',
            'author': 'unknown',
            'attribution': f"Imported from {source.name}",
            'retrieved_at_kst': created_at,
        }

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _safe_id(self, value: str) -> str:
        return re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_') or 'item'
