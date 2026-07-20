import json
import shutil
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

from PIL import Image

from ironflow_exp.engine.core import (
    CHAIN_INPUT_SOURCES,
    INTERMEDIATE_ARTIFACT_SCHEMA_VERSION,
    DetectionToClassificationArtifactValidator,
    PredictionArtifactValidator,
    UserImageDatasetValidator,
)


SUPPORTED_CROP_OUTPUT_FORMATS = frozenset({'jpg', 'jpeg', 'png'})


@dataclass(frozen=True, slots=True)
class DetectionToClassificationCropTransformResult:
    crop_region_manifest_path: Path
    classification_input_manifest_path: Path
    copied_prediction_path: Path
    crop_paths: list[Path]

    @property
    def crop_count(self) -> int:
        return len(self.crop_paths)


class DetectionToClassificationCropTransformer:
    def __init__(
        self,
        *,
        prediction_validator: PredictionArtifactValidator | None = None,
        dataset_validator: UserImageDatasetValidator | None = None,
        intermediate_validator: DetectionToClassificationArtifactValidator | None = None,
    ) -> None:
        self.prediction_validator = prediction_validator or PredictionArtifactValidator()
        self.dataset_validator = dataset_validator or UserImageDatasetValidator()
        self.intermediate_validator = intermediate_validator or DetectionToClassificationArtifactValidator()

    def transform(
        self,
        *,
        detection_prediction_path: str | Path,
        source_dataset_root: str | Path,
        output_dir: str | Path,
        padding_ratio: float = 0.0,
        image_format: str = 'jpg',
        min_score: float = 0.0,
        input_source: str = 'detector_crop',
        use_source_labels: bool = True,
        classification_dataset_id: str | None = None,
    ) -> DetectionToClassificationCropTransformResult:
        if padding_ratio < 0.0:
            raise ValueError('padding_ratio must be greater than or equal to 0')
        if not 0.0 <= min_score <= 1.0:
            raise ValueError('min_score must be between 0.0 and 1.0')
        if input_source not in CHAIN_INPUT_SOURCES:
            raise ValueError(f'input_source is not allowed: {input_source}')
        normalized_image_format = self._normalize_image_format(image_format)

        prediction_path = Path(detection_prediction_path)
        dataset_root = Path(source_dataset_root)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        prediction_payload = self._read_valid_detection_prediction(prediction_path)
        dataset_payload = self._read_valid_dataset(dataset_root)
        copied_prediction_path = self._copy_source_prediction(prediction_path=prediction_path, output_root=output_root)

        source_images = self._index_source_images(dataset_payload)
        classes = self._classes(dataset_payload)
        crop_records: list[dict[str, Any]] = []
        classification_records: list[dict[str, Any]] = []
        crop_paths: list[Path] = []

        for index, prediction in enumerate(prediction_payload['records']):
            if float(prediction.get('score', 0.0)) < min_score:
                continue
            source_image = self._source_image_for_prediction(
                prediction=prediction,
                source_images=source_images,
                index=index,
            )
            crop_record, classification_record, crop_path = self._write_crop(
                prediction=prediction,
                source_image=source_image,
                dataset_root=dataset_root,
                output_root=output_root,
                copied_prediction_path=copied_prediction_path,
                source_model_id=str(prediction_payload['model_id']),
                padding_ratio=padding_ratio,
                image_format=normalized_image_format,
                input_source=input_source,
                use_source_labels=use_source_labels,
                classes=classes,
            )
            crop_records.append(crop_record)
            classification_records.append(classification_record)
            crop_paths.append(crop_path)

        if not crop_records:
            raise ValueError('no detection records were available after applying min_score')

        crop_manifest = {
            'schema_version': INTERMEDIATE_ARTIFACT_SCHEMA_VERSION,
            'artifact_type': 'crop_region_manifest',
            'source_task_id': 'detection',
            'source_model_id': prediction_payload['model_id'],
            'source_prediction_path': copied_prediction_path.relative_to(output_root).as_posix(),
            'source_dataset_id': prediction_payload['dataset_id'],
            'records': crop_records,
        }
        classification_manifest = {
            'schema_version': INTERMEDIATE_ARTIFACT_SCHEMA_VERSION,
            'artifact_type': 'classification_input_manifest',
            'dataset_id': classification_dataset_id or f"{prediction_payload['dataset_id']}_{input_source}",
            'input_source': input_source,
            'classes': classes,
            'source_manifest_path': 'crop_region_manifest.json',
            'images': classification_records,
        }

        crop_manifest_path = output_root / 'crop_region_manifest.json'
        classification_manifest_path = output_root / 'classification_input_manifest.json'
        self._write_json(crop_manifest_path, crop_manifest)
        self._write_json(classification_manifest_path, classification_manifest)
        self._validate_intermediate(crop_manifest_path, expected_artifact_type='crop_region_manifest')
        self._validate_intermediate(classification_manifest_path, expected_artifact_type='classification_input_manifest')

        return DetectionToClassificationCropTransformResult(
            crop_region_manifest_path=crop_manifest_path,
            classification_input_manifest_path=classification_manifest_path,
            copied_prediction_path=copied_prediction_path,
            crop_paths=crop_paths,
        )

    def _read_valid_detection_prediction(self, prediction_path: Path) -> dict[str, Any]:
        validation = self.prediction_validator.validate_file(prediction_path, expected_task='detection')
        if not validation.is_valid:
            raise ValueError('invalid detection prediction artifact: ' + '; '.join(validation.error_messages()))

        return self._read_json(prediction_path)

    def _read_valid_dataset(self, dataset_root: Path) -> dict[str, Any]:
        validator = self._dataset_validator_for_root(dataset_root)
        validation = validator.validate_folder(dataset_root)
        if not validation.is_valid:
            raise ValueError('invalid source image dataset: ' + '; '.join(validation.error_messages()))

        return self._read_json(dataset_root / 'manifest.json')

    def _dataset_validator_for_root(self, dataset_root: Path) -> UserImageDatasetValidator:
        manifest_path = dataset_root / 'manifest.json'
        if not manifest_path.exists():
            return self.dataset_validator

        data = self._read_json(manifest_path)
        images = data.get('images') if isinstance(data, dict) else None
        if not isinstance(images, list) or len(images) <= self.dataset_validator.max_images:
            return self.dataset_validator

        return UserImageDatasetValidator(
            max_images=len(images),
            max_file_size_bytes=self.dataset_validator.max_file_size_bytes,
            supported_extensions=self.dataset_validator.supported_extensions,
        )

    def _copy_source_prediction(self, prediction_path: Path, output_root: Path) -> Path:
        copied_path = output_root / 'inputs' / 'detection_predictions.json'
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        if prediction_path.resolve() != copied_path.resolve():
            shutil.copy2(prediction_path, copied_path)

        return copied_path

    def _index_source_images(self, dataset_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        images: dict[str, dict[str, Any]] = {}
        for image in dataset_payload['images']:
            images[str(image['image_id'])] = image

        return images

    def _classes(self, dataset_payload: dict[str, Any]) -> list[str]:
        return [
            str(class_id)
            for class_id in dataset_payload.get('classes', [])
            if isinstance(class_id, str) and class_id
        ]

    def _normalize_image_format(self, image_format: str) -> str:
        normalized = image_format.lower().lstrip('.')
        if normalized not in SUPPORTED_CROP_OUTPUT_FORMATS:
            raise ValueError(f'image_format is not supported: {image_format}')

        return normalized

    def _source_image_for_prediction(
        self,
        *,
        prediction: dict[str, Any],
        source_images: dict[str, dict[str, Any]],
        index: int,
    ) -> dict[str, Any]:
        image_id = str(prediction.get('image_id', ''))
        if image_id not in source_images:
            raise ValueError(f'detection record {index} references unknown image_id: {image_id}')

        return source_images[image_id]

    def _write_crop(
        self,
        *,
        prediction: dict[str, Any],
        source_image: dict[str, Any],
        dataset_root: Path,
        output_root: Path,
        copied_prediction_path: Path,
        source_model_id: str,
        padding_ratio: float,
        image_format: str,
        input_source: str,
        use_source_labels: bool,
        classes: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        prediction_id = str(prediction['prediction_id'])
        source_image_path = (dataset_root / str(source_image['path'])).resolve()
        crop_id = f'crop_{prediction_id}'
        object_id = str(prediction.get('object_id') or f'obj_{prediction_id}')
        crop_path = output_root / 'crops' / input_source / f'{crop_id}.{image_format}'
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_image_path) as image:
            rgb = image.convert('RGB')
            crop_box = self._crop_box(
                bbox=prediction['bbox_xyxy'],
                image_width=rgb.width,
                image_height=rgb.height,
                padding_ratio=padding_ratio,
                prediction_id=prediction_id,
            )
            crop = rgb.crop(crop_box)
            crop.save(crop_path)
            crop_width, crop_height = crop.size

        relative_crop_path = crop_path.relative_to(output_root).as_posix()
        relative_prediction_path = copied_prediction_path.relative_to(output_root).as_posix()
        label = self._classification_label(source_image=source_image, classes=classes, use_source_labels=use_source_labels)
        sample_id = str(prediction['sample_id'])
        source_sample_id = str(source_image.get('sample_id') or sample_id)
        split = str(source_image.get('split') or 'train')

        crop_record = {
            'crop_id': crop_id,
            'image_id': str(prediction['image_id']),
            'sample_id': sample_id,
            'object_id': object_id,
            'source_prediction_id': prediction_id,
            'source_class_id': str(prediction['class_id']),
            'bbox_xyxy': [float(value) for value in prediction['bbox_xyxy']],
            'crop_path': relative_crop_path,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'input_source': input_source,
            'lineage': {
                'source_image_id': str(source_image['image_id']),
                'source_sample_id': source_sample_id,
                'source_task_id': 'detection',
                'source_model_id': source_model_id,
                'source_prediction_id': prediction_id,
                'source_prediction_path': relative_prediction_path,
            },
        }

        classification_record = {
            'image_id': crop_id,
            'sample_id': f'{source_sample_id}__{prediction_id}',
            'path': relative_crop_path,
            'label': label,
            'object_id': object_id,
            'source_image_id': str(source_image['image_id']),
            'source_sample_id': source_sample_id,
            'source_prediction_id': prediction_id,
            'source_task_id': 'detection',
            'source_model_id': source_model_id,
            'input_source': input_source,
            'split': split,
        }

        return crop_record, classification_record, crop_path

    def _crop_box(
        self,
        *,
        bbox: list[object],
        image_width: int,
        image_height: int,
        padding_ratio: float,
        prediction_id: str,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width <= 0.0 or bbox_height <= 0.0:
            raise ValueError(f'invalid zero-area bbox for prediction_id={prediction_id}')

        padding_x = bbox_width * padding_ratio
        padding_y = bbox_height * padding_ratio
        crop_x1 = max(0, floor(x1 - padding_x))
        crop_y1 = max(0, floor(y1 - padding_y))
        crop_x2 = min(image_width, ceil(x2 + padding_x))
        crop_y2 = min(image_height, ceil(y2 + padding_y))
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            raise ValueError(f'invalid crop box for prediction_id={prediction_id}')

        return crop_x1, crop_y1, crop_x2, crop_y2

    def _classification_label(self, *, source_image: dict[str, Any], classes: list[str], use_source_labels: bool) -> str | None:
        if not use_source_labels:
            return None
        label = source_image.get('label')
        if not isinstance(label, str) or label not in classes:
            return None

        return label

    def _validate_intermediate(self, manifest_path: Path, *, expected_artifact_type: str) -> None:
        validation = self.intermediate_validator.validate_file(
            manifest_path,
            expected_artifact_type=expected_artifact_type,
            check_files=True,
        )
        if not validation.is_valid:
            raise ValueError('invalid intermediate artifact: ' + '; '.join(validation.error_messages()))

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')

        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
