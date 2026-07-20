import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


INTERMEDIATE_ARTIFACT_SCHEMA_VERSION = '0.1'
INTERMEDIATE_ARTIFACT_ALLOWED_TYPES = frozenset({
    'crop_region_manifest',
    'classification_input_manifest',
})
CROP_REGION_ENVELOPE_REQUIRED_FIELDS = frozenset({
    'schema_version',
    'artifact_type',
    'source_task_id',
    'source_model_id',
    'source_prediction_path',
    'source_dataset_id',
    'records',
})
CROP_REGION_RECORD_REQUIRED_FIELDS = frozenset({
    'crop_id',
    'image_id',
    'sample_id',
    'object_id',
    'source_prediction_id',
    'source_class_id',
    'bbox_xyxy',
    'crop_path',
    'crop_width',
    'crop_height',
    'input_source',
    'lineage',
})
CLASSIFICATION_INPUT_ENVELOPE_REQUIRED_FIELDS = frozenset({
    'schema_version',
    'artifact_type',
    'dataset_id',
    'input_source',
    'classes',
    'source_manifest_path',
    'images',
})
CLASSIFICATION_INPUT_RECORD_REQUIRED_FIELDS = frozenset({
    'image_id',
    'sample_id',
    'path',
    'label',
    'object_id',
    'source_image_id',
    'source_sample_id',
    'source_prediction_id',
    'source_task_id',
    'source_model_id',
    'input_source',
})
LINEAGE_REQUIRED_FIELDS = frozenset({
    'source_image_id',
    'source_sample_id',
    'source_task_id',
    'source_model_id',
    'source_prediction_id',
    'source_prediction_path',
})
CHAIN_INPUT_SOURCES = frozenset({'detector_crop', 'gt_crop', 'seg_crop'})


@dataclass(frozen=True, slots=True)
class IntermediateArtifactValidationIssue:
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class IntermediateArtifactValidationResult:
    errors: list[IntermediateArtifactValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def error_messages(self) -> list[str]:
        return [
            f'{issue.path}: {issue.message}'
            for issue in self.errors
        ]


class DetectionToClassificationArtifactValidator:
    def validate_file(
        self,
        path: str | Path,
        *,
        expected_artifact_type: str | None = None,
        check_files: bool = False,
    ) -> IntermediateArtifactValidationResult:
        manifest_path = Path(path)
        if not manifest_path.exists():
            return IntermediateArtifactValidationResult(
                errors=[
                    IntermediateArtifactValidationIssue(
                        path=str(manifest_path),
                        message='intermediate artifact does not exist',
                    ),
                ],
            )

        data, errors = self._read_json(manifest_path)
        if errors:
            return IntermediateArtifactValidationResult(errors=errors)

        return self.validate_payload(
            data,
            expected_artifact_type=expected_artifact_type,
            root_dir=manifest_path.parent,
            check_files=check_files,
            path=manifest_path.name,
        )

    def validate_payload(
        self,
        data: dict[str, Any],
        *,
        expected_artifact_type: str | None = None,
        root_dir: str | Path | None = None,
        check_files: bool = False,
        path: str = 'intermediate_artifact',
    ) -> IntermediateArtifactValidationResult:
        errors: list[IntermediateArtifactValidationIssue] = []
        artifact_type = data.get('artifact_type')

        if data.get('schema_version') != INTERMEDIATE_ARTIFACT_SCHEMA_VERSION:
            errors.append(
                IntermediateArtifactValidationIssue(
                    path=f'{path}.schema_version',
                    message=f'expected schema version {INTERMEDIATE_ARTIFACT_SCHEMA_VERSION!r}',
                ),
            )
        if not isinstance(artifact_type, str) or not artifact_type:
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.artifact_type', message='artifact_type must be a non-empty string'))
        elif artifact_type not in INTERMEDIATE_ARTIFACT_ALLOWED_TYPES:
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.artifact_type', message='artifact_type is not allowed'))
        if expected_artifact_type is not None and artifact_type != expected_artifact_type:
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.artifact_type', message=f'expected artifact_type {expected_artifact_type!r}'))

        if artifact_type == 'crop_region_manifest':
            self._validate_crop_region_manifest(errors, path, data, root_dir, check_files)
        elif artifact_type == 'classification_input_manifest':
            self._validate_classification_input_manifest(errors, path, data, root_dir, check_files)

        return IntermediateArtifactValidationResult(errors=errors)

    def _validate_crop_region_manifest(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        data: dict[str, Any],
        root_dir: str | Path | None,
        check_files: bool,
    ) -> None:
        errors.extend(self._missing_fields(path, data, CROP_REGION_ENVELOPE_REQUIRED_FIELDS))
        self._check_string(errors, f'{path}.source_task_id', data.get('source_task_id'))
        self._check_string(errors, f'{path}.source_model_id', data.get('source_model_id'))
        self._check_string(errors, f'{path}.source_prediction_path', data.get('source_prediction_path'))
        self._check_relative_path(
            errors,
            f'{path}.source_prediction_path',
            data.get('source_prediction_path'),
            root_dir,
            check_files=False,
        )
        self._check_string(errors, f'{path}.source_dataset_id', data.get('source_dataset_id'))

        records = data.get('records')
        if not isinstance(records, list):
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.records', message='records must be a list'))
            return
        if not records:
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.records', message='records must not be empty'))
            return

        crop_ids: set[str] = set()
        for index, record in enumerate(records):
            record_path = f'{path}.records[{index}]'
            if not isinstance(record, dict):
                errors.append(IntermediateArtifactValidationIssue(path=record_path, message='record must be an object'))
                continue
            errors.extend(self._missing_fields(record_path, record, CROP_REGION_RECORD_REQUIRED_FIELDS))
            self._validate_crop_record(errors, record_path, record, crop_ids, root_dir, check_files)

    def _validate_crop_record(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        record_path: str,
        record: dict[str, Any],
        crop_ids: set[str],
        root_dir: str | Path | None,
        check_files: bool,
    ) -> None:
        crop_id = record.get('crop_id')
        self._check_string(errors, f'{record_path}.crop_id', crop_id)
        if isinstance(crop_id, str):
            if crop_id in crop_ids:
                errors.append(IntermediateArtifactValidationIssue(path=f'{record_path}.crop_id', message='crop_id must be unique'))
            crop_ids.add(crop_id)

        for field_name in ('image_id', 'sample_id', 'object_id', 'source_prediction_id', 'source_class_id', 'input_source'):
            self._check_string(errors, f'{record_path}.{field_name}', record.get(field_name))
        if record.get('input_source') not in CHAIN_INPUT_SOURCES:
            errors.append(IntermediateArtifactValidationIssue(path=f'{record_path}.input_source', message='input_source is not allowed'))

        self._check_bbox(errors, f'{record_path}.bbox_xyxy', record.get('bbox_xyxy'))
        self._check_positive_number(errors, f'{record_path}.crop_width', record.get('crop_width'))
        self._check_positive_number(errors, f'{record_path}.crop_height', record.get('crop_height'))
        self._check_relative_path(errors, f'{record_path}.crop_path', record.get('crop_path'), root_dir, check_files)
        self._validate_lineage(errors, f'{record_path}.lineage', record.get('lineage'))

    def _validate_classification_input_manifest(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        data: dict[str, Any],
        root_dir: str | Path | None,
        check_files: bool,
    ) -> None:
        errors.extend(self._missing_fields(path, data, CLASSIFICATION_INPUT_ENVELOPE_REQUIRED_FIELDS))
        self._check_string(errors, f'{path}.dataset_id', data.get('dataset_id'))
        self._check_string(errors, f'{path}.input_source', data.get('input_source'))
        self._check_string(errors, f'{path}.source_manifest_path', data.get('source_manifest_path'))
        self._check_relative_path(
            errors,
            f'{path}.source_manifest_path',
            data.get('source_manifest_path'),
            root_dir,
            check_files=False,
        )
        if data.get('input_source') not in CHAIN_INPUT_SOURCES:
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.input_source', message='input_source is not allowed'))

        classes = data.get('classes')
        if not isinstance(classes, list) or not all(isinstance(item, str) and item for item in classes):
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.classes', message='classes must be a list of non-empty strings'))
            class_ids: set[str] = set()
        else:
            class_ids = set(classes)

        images = data.get('images')
        if not isinstance(images, list):
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.images', message='images must be a list'))
            return
        if not images:
            errors.append(IntermediateArtifactValidationIssue(path=f'{path}.images', message='images must not be empty'))
            return

        image_ids: set[str] = set()
        for index, record in enumerate(images):
            record_path = f'{path}.images[{index}]'
            if not isinstance(record, dict):
                errors.append(IntermediateArtifactValidationIssue(path=record_path, message='image record must be an object'))
                continue
            errors.extend(self._missing_fields(record_path, record, CLASSIFICATION_INPUT_RECORD_REQUIRED_FIELDS))
            self._validate_classification_input_record(errors, record_path, record, image_ids, class_ids, root_dir, check_files)

    def _validate_classification_input_record(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        record_path: str,
        record: dict[str, Any],
        image_ids: set[str],
        class_ids: set[str],
        root_dir: str | Path | None,
        check_files: bool,
    ) -> None:
        image_id = record.get('image_id')
        self._check_string(errors, f'{record_path}.image_id', image_id)
        if isinstance(image_id, str):
            if image_id in image_ids:
                errors.append(IntermediateArtifactValidationIssue(path=f'{record_path}.image_id', message='image_id must be unique'))
            image_ids.add(image_id)

        for field_name in (
            'sample_id',
            'object_id',
            'source_image_id',
            'source_sample_id',
            'source_prediction_id',
            'source_task_id',
            'source_model_id',
            'input_source',
        ):
            self._check_string(errors, f'{record_path}.{field_name}', record.get(field_name))
        if record.get('input_source') not in CHAIN_INPUT_SOURCES:
            errors.append(IntermediateArtifactValidationIssue(path=f'{record_path}.input_source', message='input_source is not allowed'))

        label = record.get('label')
        if label is not None:
            self._check_string(errors, f'{record_path}.label', label)
            if isinstance(label, str) and class_ids and label not in class_ids:
                errors.append(IntermediateArtifactValidationIssue(path=f'{record_path}.label', message='label must exist in classes'))
        self._check_relative_path(errors, f'{record_path}.path', record.get('path'), root_dir, check_files)

    def _validate_lineage(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        lineage: object,
    ) -> None:
        if not isinstance(lineage, dict):
            errors.append(IntermediateArtifactValidationIssue(path=path, message='lineage must be an object'))
            return
        errors.extend(self._missing_fields(path, lineage, LINEAGE_REQUIRED_FIELDS))
        for field_name in LINEAGE_REQUIRED_FIELDS:
            self._check_string(errors, f'{path}.{field_name}', lineage.get(field_name))

    def _check_bbox(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        bbox: object,
    ) -> None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='bbox_xyxy must contain exactly four numbers'))
            return
        if not all(self._is_number(value) for value in bbox):
            errors.append(IntermediateArtifactValidationIssue(path=path, message='bbox_xyxy values must be numeric'))
            return
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if x1 > x2 or y1 > y2:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='bbox coordinates must be ordered as x1 <= x2 and y1 <= y2'))

    def _check_relative_path(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        value: object,
        root_dir: str | Path | None,
        check_files: bool,
    ) -> None:
        if not isinstance(value, str) or not value:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='path must be a non-empty string'))
            return
        relative_path = Path(value)
        if relative_path.is_absolute():
            errors.append(IntermediateArtifactValidationIssue(path=path, message='path must be relative'))
            return
        if '..' in relative_path.parts:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='path must not contain parent traversal'))
            return
        if root_dir is None:
            return
        root_path = Path(root_dir).resolve()
        resolved_path = (root_path / relative_path).resolve()
        try:
            resolved_path.relative_to(root_path)
        except ValueError:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='path must stay inside artifact root'))
            return
        if check_files and not resolved_path.exists():
            errors.append(IntermediateArtifactValidationIssue(path=path, message='referenced file does not exist'))

    def _read_json(self, path: Path) -> tuple[dict[str, Any], list[IntermediateArtifactValidationIssue]]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            return {}, [IntermediateArtifactValidationIssue(path=path.name, message=f'invalid JSON: {exc.msg}')]

        if not isinstance(data, dict):
            return {}, [IntermediateArtifactValidationIssue(path=path.name, message='JSON root must be an object')]

        return data, []

    def _missing_fields(
        self,
        path: str,
        data: dict[str, Any],
        required_fields: frozenset[str],
    ) -> list[IntermediateArtifactValidationIssue]:
        missing = sorted(required_fields.difference(data.keys()))
        return [
            IntermediateArtifactValidationIssue(path=path, message=f'missing required field: {field_name}')
            for field_name in missing
        ]

    def _check_string(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not isinstance(value, str) or not value:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='field must be a non-empty string'))

    def _check_positive_number(
        self,
        errors: list[IntermediateArtifactValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not self._is_number(value) or float(value) <= 0.0:
            errors.append(IntermediateArtifactValidationIssue(path=path, message='field must be a positive number'))

    def _is_number(self, value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
