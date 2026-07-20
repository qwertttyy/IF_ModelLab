import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


USER_IMAGE_DATASET_SCHEMA_VERSION = '0.1'
USER_IMAGE_DATASET_REQUIRED_FIELDS = frozenset({
    'schema_version',
    'dataset_id',
    'dataset_type',
    'classes',
    'images',
})
USER_IMAGE_RECORD_REQUIRED_FIELDS = frozenset({
    'image_id',
    'sample_id',
    'path',
    'label',
    'split',
    'source',
    'objects',
})
USER_IMAGE_SOURCE_REQUIRED_FIELDS = frozenset({
    'source_url',
    'file_url',
    'license',
    'author',
    'attribution',
    'retrieved_at_kst',
})
USER_IMAGE_ALLOWED_DATASET_TYPES = frozenset({
    'classification',
    'detection',
    'mixed',
    'segmentation',
    'unlabeled_smoke',
})
USER_IMAGE_SUPPORTED_EXTENSIONS = frozenset({
    '.jpg',
    '.jpeg',
    '.png',
    '.ppm',
})


@dataclass(frozen=True, slots=True)
class UserImageDatasetValidationIssue:
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class UserImageDatasetValidationResult:
    dataset_root: Path
    manifest_path: Path
    image_paths: list[Path] = field(default_factory=list)
    errors: list[UserImageDatasetValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def error_messages(self) -> list[str]:
        return [
            f'{issue.path}: {issue.message}'
            for issue in self.errors
        ]


class UserImageDatasetValidator:
    def __init__(
        self,
        *,
        max_images: int = 100,
        max_file_size_bytes: int = 20 * 1024 * 1024,
        supported_extensions: frozenset[str] = USER_IMAGE_SUPPORTED_EXTENSIONS,
    ) -> None:
        self.max_images = max_images
        self.max_file_size_bytes = max_file_size_bytes
        self.supported_extensions = supported_extensions

    def validate_folder(self, dataset_root: str | Path) -> UserImageDatasetValidationResult:
        root = Path(dataset_root)
        manifest_path = root / 'manifest.json'
        errors: list[UserImageDatasetValidationIssue] = []
        image_paths: list[Path] = []

        if not root.exists() or not root.is_dir():
            return UserImageDatasetValidationResult(
                dataset_root=root,
                manifest_path=manifest_path,
                errors=[UserImageDatasetValidationIssue(path=str(root), message='dataset root does not exist or is not a directory')],
            )
        if not manifest_path.exists():
            return UserImageDatasetValidationResult(
                dataset_root=root,
                manifest_path=manifest_path,
                errors=[UserImageDatasetValidationIssue(path='manifest.json', message='manifest file does not exist')],
            )

        data, read_errors = self._read_json(manifest_path)
        errors.extend(read_errors)
        if read_errors:
            return UserImageDatasetValidationResult(
                dataset_root=root,
                manifest_path=manifest_path,
                errors=errors,
            )

        errors.extend(self._validate_manifest_shape(data=data))
        if errors:
            return UserImageDatasetValidationResult(
                dataset_root=root,
                manifest_path=manifest_path,
                errors=errors,
            )

        classes = data['classes']
        dataset_type = data['dataset_type']
        images = data['images']
        if len(images) > self.max_images:
            errors.append(UserImageDatasetValidationIssue(path='manifest.images', message=f'image count exceeds limit: {self.max_images}'))

        seen_image_ids: set[str] = set()
        seen_paths: set[str] = set()
        for index, record in enumerate(images):
            record_path = f'manifest.images[{index}]'
            if not isinstance(record, dict):
                errors.append(UserImageDatasetValidationIssue(path=record_path, message='image record must be an object'))
                continue

            errors.extend(self._missing_fields(record_path, record, USER_IMAGE_RECORD_REQUIRED_FIELDS))
            if errors:
                continue

            image_id = record.get('image_id')
            if not isinstance(image_id, str) or not image_id:
                errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.image_id', message='image_id must be a non-empty string'))
            elif image_id in seen_image_ids:
                errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.image_id', message='image_id must be unique'))
            else:
                seen_image_ids.add(image_id)

            relative_path = record.get('path')
            resolved_path = self._resolve_record_path(root=root, relative_path=relative_path, record_path=record_path, errors=errors)
            if resolved_path is not None:
                normalized_relative = str(Path(str(relative_path).replace('\\', '/')))
                if normalized_relative in seen_paths:
                    errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message='image path must be unique'))
                else:
                    seen_paths.add(normalized_relative)
                image_paths.append(resolved_path)
                self._validate_image_file(path=resolved_path, record_path=record_path, errors=errors)

            self._validate_label(record=record, record_path=record_path, dataset_type=dataset_type, classes=classes, errors=errors)
            self._validate_source(record=record, record_path=record_path, errors=errors)
            self._validate_objects(record=record, record_path=record_path, errors=errors)

        if dataset_type != 'unlabeled_smoke' and not classes:
            errors.append(UserImageDatasetValidationIssue(path='manifest.classes', message='classes must not be empty for labeled datasets'))

        return UserImageDatasetValidationResult(
            dataset_root=root,
            manifest_path=manifest_path,
            image_paths=image_paths,
            errors=errors,
        )

    def _validate_manifest_shape(self, data: dict[str, Any]) -> list[UserImageDatasetValidationIssue]:
        errors: list[UserImageDatasetValidationIssue] = []
        errors.extend(self._missing_fields('manifest', data, USER_IMAGE_DATASET_REQUIRED_FIELDS))
        if errors:
            return errors

        if data.get('schema_version') != USER_IMAGE_DATASET_SCHEMA_VERSION:
            errors.append(
                UserImageDatasetValidationIssue(
                    path='manifest.schema_version',
                    message=f'expected schema version {USER_IMAGE_DATASET_SCHEMA_VERSION!r}',
                ),
            )
        self._check_non_empty_string(errors, 'manifest.dataset_id', data.get('dataset_id'))
        dataset_type = data.get('dataset_type')
        if dataset_type not in USER_IMAGE_ALLOWED_DATASET_TYPES:
            errors.append(UserImageDatasetValidationIssue(path='manifest.dataset_type', message='dataset_type is not allowed'))
        classes = data.get('classes')
        if not isinstance(classes, list) or not all(isinstance(item, str) and item for item in classes):
            errors.append(UserImageDatasetValidationIssue(path='manifest.classes', message='classes must be a list of non-empty strings'))
        images = data.get('images')
        if not isinstance(images, list) or not images:
            errors.append(UserImageDatasetValidationIssue(path='manifest.images', message='images must be a non-empty list'))

        return errors

    def _resolve_record_path(
        self,
        root: Path,
        relative_path: object,
        record_path: str,
        errors: list[UserImageDatasetValidationIssue],
    ) -> Path | None:
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message='path must be a non-empty string'))
            return None

        path = Path(relative_path)
        if path.is_absolute():
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message='path must be relative'))
            return None

        root_resolved = root.resolve()
        resolved = (root_resolved / path).resolve()
        if not resolved.is_relative_to(root_resolved):
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message='path must stay inside dataset root'))
            return None

        return resolved

    def _validate_image_file(
        self,
        path: Path,
        record_path: str,
        errors: list[UserImageDatasetValidationIssue],
    ) -> None:
        if not path.exists() or not path.is_file():
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message='image file does not exist'))
            return
        if path.suffix.lower() not in self.supported_extensions:
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message='image extension is not supported'))
        if path.stat().st_size > self.max_file_size_bytes:
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.path', message=f'image file exceeds size limit: {self.max_file_size_bytes}'))

    def _validate_label(
        self,
        record: dict[str, Any],
        record_path: str,
        dataset_type: object,
        classes: object,
        errors: list[UserImageDatasetValidationIssue],
    ) -> None:
        label = record.get('label')
        if dataset_type == 'unlabeled_smoke':
            if label is not None:
                errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.label', message='label must be null for unlabeled_smoke'))
            return

        if not isinstance(label, str) or not label:
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.label', message='label must be a non-empty string'))
            return
        if isinstance(classes, list) and label not in classes:
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.label', message='label must exist in classes'))

    def _validate_source(
        self,
        record: dict[str, Any],
        record_path: str,
        errors: list[UserImageDatasetValidationIssue],
    ) -> None:
        source = record.get('source')
        if not isinstance(source, dict):
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.source', message='source must be an object'))
            return

        errors.extend(self._missing_fields(f'{record_path}.source', source, USER_IMAGE_SOURCE_REQUIRED_FIELDS))
        license_text = source.get('license')
        if not isinstance(license_text, str) or not license_text:
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.source.license', message='license must be a non-empty string'))

    def _validate_objects(
        self,
        record: dict[str, Any],
        record_path: str,
        errors: list[UserImageDatasetValidationIssue],
    ) -> None:
        objects = record.get('objects')
        if not isinstance(objects, list):
            errors.append(UserImageDatasetValidationIssue(path=f'{record_path}.objects', message='objects must be a list'))
            return
        for index, item in enumerate(objects):
            object_path = f'{record_path}.objects[{index}]'
            if not isinstance(item, dict):
                errors.append(UserImageDatasetValidationIssue(path=object_path, message='object must be an object'))
                continue
            self._check_non_empty_string(errors, f'{object_path}.object_id', item.get('object_id'))
            self._check_non_empty_string(errors, f'{object_path}.class_id', item.get('class_id'))
            bbox = item.get('bbox_xyxy')
            if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or not all(self._is_number(value) for value in bbox)):
                errors.append(UserImageDatasetValidationIssue(path=f'{object_path}.bbox_xyxy', message='bbox_xyxy must contain four numbers when provided'))

    def _read_json(self, path: Path) -> tuple[dict[str, Any], list[UserImageDatasetValidationIssue]]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            return {}, [UserImageDatasetValidationIssue(path='manifest.json', message=f'invalid JSON: {exc.msg}')]

        if not isinstance(data, dict):
            return {}, [UserImageDatasetValidationIssue(path='manifest.json', message='JSON root must be an object')]

        return data, []

    def _missing_fields(
        self,
        path: str,
        data: dict[str, Any],
        required_fields: frozenset[str],
    ) -> list[UserImageDatasetValidationIssue]:
        missing = sorted(required_fields.difference(data.keys()))
        return [
            UserImageDatasetValidationIssue(path=path, message=f'missing required field: {field_name}')
            for field_name in missing
        ]

    def _check_non_empty_string(
        self,
        errors: list[UserImageDatasetValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not isinstance(value, str) or not value:
            errors.append(UserImageDatasetValidationIssue(path=path, message='field must be a non-empty string'))

    def _is_number(self, value: object) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)
