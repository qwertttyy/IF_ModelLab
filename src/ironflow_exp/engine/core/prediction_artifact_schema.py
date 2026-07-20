import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PREDICTION_ARTIFACT_SCHEMA_VERSION = '0.1'
PREDICTION_ENVELOPE_REQUIRED_FIELDS = frozenset({
    'schema_version',
    'task',
    'model_id',
    'dataset_id',
    'success',
    'records',
})
CLASSIFICATION_RECORD_REQUIRED_FIELDS = frozenset({
    'image_id',
    'sample_id',
    'object_id',
    'top1_class_id',
    'top1_score',
    'scores',
})
DETECTION_RECORD_REQUIRED_FIELDS = frozenset({
    'image_id',
    'sample_id',
    'prediction_id',
    'class_id',
    'score',
    'bbox_xyxy',
    'image_width',
    'image_height',
})
TRACKING_RECORD_REQUIRED_FIELDS = frozenset({
    'frame_id',
    'image_id',
    'sample_id',
    'track_id',
    'source_prediction_id',
    'class_id',
    'score',
    'bbox_xyxy',
    'image_width',
    'image_height',
})
SEGMENTATION_RECORD_REQUIRED_FIELDS = frozenset({
    'image_id',
    'sample_id',
    'prediction_id',
    'class_id',
    'score',
    'mask_kind',
    'mask_polygon',
    'bbox_xyxy',
    'image_width',
    'image_height',
})
EMBEDDING_RECORD_REQUIRED_FIELDS = frozenset({
    'image_id',
    'sample_id',
    'prediction_id',
    'object_id',
    'embedding_id',
    'embedding_index',
    'embedding_dim',
    'embedding_path',
    'source_path',
})
PREDICTION_ALLOWED_TASKS = frozenset({'classification', 'detection', 'tracking', 'segmentation', 'embedding'})


@dataclass(frozen=True, slots=True)
class PredictionArtifactValidationIssue:
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class PredictionArtifactValidationResult:
    errors: list[PredictionArtifactValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def error_messages(self) -> list[str]:
        return [
            f'{issue.path}: {issue.message}'
            for issue in self.errors
        ]


class PredictionArtifactValidator:
    def validate_file(
        self,
        path: str | Path,
        *,
        expected_task: str | None = None,
    ) -> PredictionArtifactValidationResult:
        prediction_path = Path(path)
        if not prediction_path.exists():
            return PredictionArtifactValidationResult(
                errors=[
                    PredictionArtifactValidationIssue(
                        path=str(prediction_path),
                        message='prediction artifact does not exist',
                    ),
                ],
            )

        data, errors = self._read_json(prediction_path)
        if errors:
            return PredictionArtifactValidationResult(errors=errors)

        return self.validate_payload(data, expected_task=expected_task, path=prediction_path.name)

    def validate_payload(
        self,
        data: dict[str, Any],
        *,
        expected_task: str | None = None,
        path: str = 'prediction',
    ) -> PredictionArtifactValidationResult:
        errors: list[PredictionArtifactValidationIssue] = []

        errors.extend(self._missing_fields(path, data, PREDICTION_ENVELOPE_REQUIRED_FIELDS))
        if errors:
            return PredictionArtifactValidationResult(errors=errors)

        task = data.get('task')
        if not isinstance(task, str) or not task:
            errors.append(PredictionArtifactValidationIssue(path=f'{path}.task', message='field must be a non-empty string'))
        elif task not in PREDICTION_ALLOWED_TASKS:
            errors.append(PredictionArtifactValidationIssue(path=f'{path}.task', message='task is not allowed'))

        if expected_task is not None and task != expected_task:
            errors.append(PredictionArtifactValidationIssue(path=f'{path}.task', message=f'expected task {expected_task!r}'))

        if data.get('schema_version') != PREDICTION_ARTIFACT_SCHEMA_VERSION:
            errors.append(
                PredictionArtifactValidationIssue(
                    path=f'{path}.schema_version',
                    message=f'expected schema version {PREDICTION_ARTIFACT_SCHEMA_VERSION!r}',
                ),
            )

        self._check_string(errors, f'{path}.model_id', data.get('model_id'))
        self._check_string(errors, f'{path}.dataset_id', data.get('dataset_id'))

        success = data.get('success')
        if not isinstance(success, bool):
            errors.append(PredictionArtifactValidationIssue(path=f'{path}.success', message='success must be a boolean'))

        records = data.get('records')
        if not isinstance(records, list):
            errors.append(PredictionArtifactValidationIssue(path=f'{path}.records', message='records must be a list'))
            return PredictionArtifactValidationResult(errors=errors)

        if success is True and not records and task != 'detection':
            errors.append(PredictionArtifactValidationIssue(path=f'{path}.records', message='successful predictions must contain at least one record'))

        if task == 'classification':
            self._validate_classification_records(errors, path, records)
        elif task == 'detection':
            self._validate_detection_records(errors, path, records)
        elif task == 'tracking':
            self._validate_tracking_records(errors, path, records)
        elif task == 'segmentation':
            self._validate_segmentation_records(errors, path, records)
        elif task == 'embedding':
            self._validate_embedding_records(errors, path, records)

        return PredictionArtifactValidationResult(errors=errors)

    def _validate_classification_records(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        records: list[Any],
    ) -> None:
        for index, record in enumerate(records):
            record_path = f'{path}.records[{index}]'
            if not isinstance(record, dict):
                errors.append(PredictionArtifactValidationIssue(path=record_path, message='record must be an object'))
                continue

            errors.extend(self._missing_fields(record_path, record, CLASSIFICATION_RECORD_REQUIRED_FIELDS))
            self._check_string(errors, f'{record_path}.image_id', record.get('image_id'))
            self._check_string(errors, f'{record_path}.sample_id', record.get('sample_id'))
            object_id = record.get('object_id')
            if object_id is not None and not isinstance(object_id, str):
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.object_id', message='object_id must be a string or null'))
            self._check_string(errors, f'{record_path}.top1_class_id', record.get('top1_class_id'))
            self._check_score(errors, f'{record_path}.top1_score', record.get('top1_score'))

            scores = record.get('scores')
            if not isinstance(scores, dict) or not scores:
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.scores', message='scores must be a non-empty object'))
                continue

            for class_id, score in scores.items():
                if not isinstance(class_id, str) or not class_id:
                    errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.scores', message='score keys must be non-empty strings'))
                    continue
                self._check_score(errors, f'{record_path}.scores.{class_id}', score)

            top1_class_id = record.get('top1_class_id')
            if isinstance(top1_class_id, str) and top1_class_id not in scores:
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.top1_class_id', message='top1_class_id must exist in scores'))

            top1_score = record.get('top1_score')
            if (
                isinstance(top1_class_id, str)
                and top1_class_id in scores
                and self._is_number(top1_score)
                and self._is_number(scores[top1_class_id])
                and abs(float(top1_score) - float(scores[top1_class_id])) > 1e-9
            ):
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.top1_score', message='top1_score must match scores[top1_class_id]'))

    def _validate_detection_records(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        records: list[Any],
    ) -> None:
        for index, record in enumerate(records):
            record_path = f'{path}.records[{index}]'
            if not isinstance(record, dict):
                errors.append(PredictionArtifactValidationIssue(path=record_path, message='record must be an object'))
                continue

            errors.extend(self._missing_fields(record_path, record, DETECTION_RECORD_REQUIRED_FIELDS))
            self._check_string(errors, f'{record_path}.image_id', record.get('image_id'))
            self._check_string(errors, f'{record_path}.sample_id', record.get('sample_id'))
            self._check_string(errors, f'{record_path}.prediction_id', record.get('prediction_id'))
            self._check_string(errors, f'{record_path}.class_id', record.get('class_id'))
            self._check_score(errors, f'{record_path}.score', record.get('score'))

            image_width = record.get('image_width')
            image_height = record.get('image_height')
            self._check_positive_number(errors, f'{record_path}.image_width', image_width)
            self._check_positive_number(errors, f'{record_path}.image_height', image_height)
            self._check_bbox(errors, record_path, record.get('bbox_xyxy'), image_width, image_height)

    def _validate_tracking_records(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        records: list[Any],
    ) -> None:
        for index, record in enumerate(records):
            record_path = f'{path}.records[{index}]'
            if not isinstance(record, dict):
                errors.append(PredictionArtifactValidationIssue(path=record_path, message='record must be an object'))
                continue

            errors.extend(self._missing_fields(record_path, record, TRACKING_RECORD_REQUIRED_FIELDS))
            self._check_string(errors, f'{record_path}.frame_id', record.get('frame_id'))
            self._check_string(errors, f'{record_path}.image_id', record.get('image_id'))
            self._check_string(errors, f'{record_path}.sample_id', record.get('sample_id'))
            self._check_string(errors, f'{record_path}.track_id', record.get('track_id'))
            self._check_string(errors, f'{record_path}.source_prediction_id', record.get('source_prediction_id'))
            self._check_string(errors, f'{record_path}.class_id', record.get('class_id'))
            self._check_score(errors, f'{record_path}.score', record.get('score'))

            image_width = record.get('image_width')
            image_height = record.get('image_height')
            self._check_positive_number(errors, f'{record_path}.image_width', image_width)
            self._check_positive_number(errors, f'{record_path}.image_height', image_height)
            self._check_bbox(errors, record_path, record.get('bbox_xyxy'), image_width, image_height)

    def _validate_segmentation_records(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        records: list[Any],
    ) -> None:
        for index, record in enumerate(records):
            record_path = f'{path}.records[{index}]'
            if not isinstance(record, dict):
                errors.append(PredictionArtifactValidationIssue(path=record_path, message='record must be an object'))
                continue

            errors.extend(self._missing_fields(record_path, record, SEGMENTATION_RECORD_REQUIRED_FIELDS))
            self._check_string(errors, f'{record_path}.image_id', record.get('image_id'))
            self._check_string(errors, f'{record_path}.sample_id', record.get('sample_id'))
            self._check_string(errors, f'{record_path}.prediction_id', record.get('prediction_id'))
            self._check_string(errors, f'{record_path}.class_id', record.get('class_id'))
            self._check_score(errors, f'{record_path}.score', record.get('score'))
            self._check_string(errors, f'{record_path}.mask_kind', record.get('mask_kind'))

            image_width = record.get('image_width')
            image_height = record.get('image_height')
            self._check_positive_number(errors, f'{record_path}.image_width', image_width)
            self._check_positive_number(errors, f'{record_path}.image_height', image_height)
            self._check_bbox(errors, record_path, record.get('bbox_xyxy'), image_width, image_height)
            self._check_polygon(errors, record_path, record.get('mask_polygon'), image_width, image_height)

    def _validate_embedding_records(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        records: list[Any],
    ) -> None:
        seen_indices: set[int] = set()
        for index, record in enumerate(records):
            record_path = f'{path}.records[{index}]'
            if not isinstance(record, dict):
                errors.append(PredictionArtifactValidationIssue(path=record_path, message='record must be an object'))
                continue

            errors.extend(self._missing_fields(record_path, record, EMBEDDING_RECORD_REQUIRED_FIELDS))
            self._check_string(errors, f'{record_path}.image_id', record.get('image_id'))
            self._check_string(errors, f'{record_path}.sample_id', record.get('sample_id'))
            self._check_string(errors, f'{record_path}.prediction_id', record.get('prediction_id'))
            object_id = record.get('object_id')
            if object_id is not None and not isinstance(object_id, str):
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.object_id', message='object_id must be a string or null'))
            self._check_string(errors, f'{record_path}.embedding_id', record.get('embedding_id'))
            self._check_string(errors, f'{record_path}.embedding_path', record.get('embedding_path'))
            self._check_string(errors, f'{record_path}.source_path', record.get('source_path'))
            embedding_index = record.get('embedding_index')
            if not self._is_non_negative_int(embedding_index):
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.embedding_index', message='embedding_index must be a non-negative integer'))
            else:
                if int(embedding_index) in seen_indices:
                    errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.embedding_index', message='embedding_index must be unique'))
                seen_indices.add(int(embedding_index))
            if not self._is_positive_int(record.get('embedding_dim')):
                errors.append(PredictionArtifactValidationIssue(path=f'{record_path}.embedding_dim', message='embedding_dim must be a positive integer'))

    def _check_bbox(
        self,
        errors: list[PredictionArtifactValidationIssue],
        record_path: str,
        bbox: object,
        image_width: object,
        image_height: object,
    ) -> None:
        bbox_path = f'{record_path}.bbox_xyxy'
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(PredictionArtifactValidationIssue(path=bbox_path, message='bbox_xyxy must contain exactly four numbers'))
            return

        if not all(self._is_number(value) for value in bbox):
            errors.append(PredictionArtifactValidationIssue(path=bbox_path, message='bbox_xyxy values must be numeric'))
            return

        x1, y1, x2, y2 = [float(value) for value in bbox]
        if x1 > x2 or y1 > y2:
            errors.append(PredictionArtifactValidationIssue(path=bbox_path, message='bbox coordinates must be ordered as x1 <= x2 and y1 <= y2'))

        if not self._is_positive_number(image_width) or not self._is_positive_number(image_height):
            return

        width = float(image_width)
        height = float(image_height)
        if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
            errors.append(PredictionArtifactValidationIssue(path=bbox_path, message='bbox coordinates must be inside image bounds'))

    def _check_polygon(
        self,
        errors: list[PredictionArtifactValidationIssue],
        record_path: str,
        polygon: object,
        image_width: object,
        image_height: object,
    ) -> None:
        polygon_path = f'{record_path}.mask_polygon'
        if not isinstance(polygon, list) or len(polygon) < 3:
            errors.append(PredictionArtifactValidationIssue(path=polygon_path, message='mask_polygon must contain at least three points'))
            return

        points: list[tuple[float, float]] = []
        for point_index, point in enumerate(polygon):
            point_path = f'{polygon_path}[{point_index}]'
            if not isinstance(point, list) or len(point) != 2:
                errors.append(PredictionArtifactValidationIssue(path=point_path, message='polygon point must contain exactly two numbers'))
                continue
            if not all(self._is_number(value) for value in point):
                errors.append(PredictionArtifactValidationIssue(path=point_path, message='polygon point values must be numeric'))
                continue
            points.append((float(point[0]), float(point[1])))

        if not self._is_positive_number(image_width) or not self._is_positive_number(image_height):
            return

        width = float(image_width)
        height = float(image_height)
        for point_index, (x, y) in enumerate(points):
            if x < 0 or x > width or y < 0 or y > height:
                errors.append(
                    PredictionArtifactValidationIssue(
                        path=f'{polygon_path}[{point_index}]',
                        message='polygon point must be inside image bounds',
                    ),
                )

    def _read_json(self, path: Path) -> tuple[dict[str, Any], list[PredictionArtifactValidationIssue]]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            return {}, [PredictionArtifactValidationIssue(path=path.name, message=f'invalid JSON: {exc.msg}')]

        if not isinstance(data, dict):
            return {}, [PredictionArtifactValidationIssue(path=path.name, message='JSON root must be an object')]

        return data, []

    def _missing_fields(
        self,
        path: str,
        data: dict[str, Any],
        required_fields: frozenset[str],
    ) -> list[PredictionArtifactValidationIssue]:
        missing = sorted(required_fields.difference(data.keys()))
        return [
            PredictionArtifactValidationIssue(path=path, message=f'missing required field: {field_name}')
            for field_name in missing
        ]

    def _check_string(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not isinstance(value, str) or not value:
            errors.append(PredictionArtifactValidationIssue(path=path, message='field must be a non-empty string'))

    def _check_score(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not self._is_number(value) or not 0.0 <= float(value) <= 1.0:
            errors.append(PredictionArtifactValidationIssue(path=path, message='score must be a number between 0.0 and 1.0'))

    def _check_positive_number(
        self,
        errors: list[PredictionArtifactValidationIssue],
        path: str,
        value: object,
    ) -> None:
        if not self._is_positive_number(value):
            errors.append(PredictionArtifactValidationIssue(path=path, message='field must be a positive number'))

    def _is_positive_number(self, value: object) -> bool:
        return self._is_number(value) and float(value) > 0.0

    def _is_number(self, value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _is_positive_int(self, value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def _is_non_negative_int(self, value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
