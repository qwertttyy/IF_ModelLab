import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.core import PredictionArtifactValidator, TaskArtifactValidator
from ironflow_exp.engine.domain import EngineMetricRecord
from ironflow_exp.engine.log_utils import read_text_tail
from ironflow_exp.engine.output_policy import effective_collect_patterns


@dataclass(frozen=True, slots=True)
class TaskArtifactCollection:
    task_dir: str
    is_valid: bool
    elapsed_seconds: float | None = None
    timing_rows: list[dict[str, object]] = field(default_factory=list)
    metrics_rows: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            'task_dir': self.task_dir,
            'is_valid': self.is_valid,
            'elapsed_seconds': self.elapsed_seconds,
            'timing_rows': self.timing_rows,
            'metrics_rows': self.metrics_rows,
            'errors': self.errors,
        }


@dataclass(frozen=True, slots=True)
class PredictionArtifactCollection:
    path: str
    is_valid: bool
    task: str | None = None
    record_count: int | None = None
    track_count: int | None = None
    frame_range: str | None = None
    sample_range: str | None = None
    min_class_recall_class: str | None = None
    min_class_recall: float | None = None
    per_class_metrics: list[dict[str, object]] = field(default_factory=list)
    per_class_metrics_csv: str | None = None
    per_class_metrics_json: str | None = None
    per_class_metrics_split: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            'path': self.path,
            'is_valid': self.is_valid,
            'task': self.task,
            'record_count': self.record_count,
            'track_count': self.track_count,
            'frame_range': self.frame_range,
            'sample_range': self.sample_range,
            'min_class_recall_class': self.min_class_recall_class,
            'min_class_recall': self.min_class_recall,
            'per_class_metrics': self.per_class_metrics,
            'per_class_metrics_csv': self.per_class_metrics_csv,
            'per_class_metrics_json': self.per_class_metrics_json,
            'per_class_metrics_split': self.per_class_metrics_split,
            'errors': self.errors,
        }


@dataclass(frozen=True, slots=True)
class AugmentationArtifactCollection:
    path: str
    is_valid: bool
    policy_id: str | None = None
    target_split: str | None = None
    augmentation_count: int | None = None
    record_count: int | None = None
    source_dataset_id: str | None = None
    dataset_id: str | None = None
    label_transform_count: int | None = None
    bbox_transform_count: int | None = None
    bbox_drop_count: int | None = None
    dropped_object_count: int | None = None
    mask_transform_count: int | None = None
    mask_record_transform_count: int | None = None
    mask_object_transform_count: int | None = None
    mask_artifact_count: int | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            'path': self.path,
            'is_valid': self.is_valid,
            'policy_id': self.policy_id,
            'target_split': self.target_split,
            'augmentation_count': self.augmentation_count,
            'record_count': self.record_count,
            'source_dataset_id': self.source_dataset_id,
            'dataset_id': self.dataset_id,
            'label_transform_count': self.label_transform_count,
            'bbox_transform_count': self.bbox_transform_count,
            'bbox_drop_count': self.bbox_drop_count,
            'dropped_object_count': self.dropped_object_count,
            'mask_transform_count': self.mask_transform_count,
            'mask_record_transform_count': self.mask_record_transform_count,
            'mask_object_transform_count': self.mask_object_transform_count,
            'mask_artifact_count': self.mask_artifact_count,
            'errors': self.errors,
        }


@dataclass(frozen=True, slots=True)
class CollectionResult:
    experiment_id: str
    result_dir: Path
    metrics: list[EngineMetricRecord] = field(default_factory=list)
    found_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    task_artifacts: list[TaskArtifactCollection] = field(default_factory=list)
    prediction_artifacts: list[PredictionArtifactCollection] = field(default_factory=list)
    augmentation_artifacts: list[AugmentationArtifactCollection] = field(default_factory=list)
    log_tail: str = ''
    metrics_source: str | None = None


class ResultCollector:
    def __init__(
        self,
        task_validator: TaskArtifactValidator | None = None,
        prediction_validator: PredictionArtifactValidator | None = None,
    ) -> None:
        self.task_validator = task_validator or TaskArtifactValidator()
        self.prediction_validator = prediction_validator or PredictionArtifactValidator()

    def collect(
        self,
        experiment_id: str,
        config: EngineExperimentConfig,
        result_dir: Path,
        created_at: str,
        log_tail_lines: int = 50,
    ) -> CollectionResult:
        found_files, missing_files = self._inspect_collect_patterns(
            result_dir=result_dir,
            collect_patterns=effective_collect_patterns(output=config.output),
        )
        log_tail = self._read_log_tail(
            log_path=result_dir / config.output.log_file,
            line_count=log_tail_lines,
        )
        metrics, metrics_source = self._load_metrics(
            experiment_id=experiment_id,
            metric_name=config.analysis.primary_metric,
            metrics_path=result_dir / config.output.metrics_file,
            created_at=created_at,
        )
        task_artifacts = self._collect_task_artifacts(result_dir=result_dir)
        prediction_artifacts = self._collect_prediction_artifacts(result_dir=result_dir)
        augmentation_artifacts = self._collect_augmentation_artifacts(result_dir=result_dir)

        return CollectionResult(
            experiment_id=experiment_id,
            result_dir=result_dir,
            metrics=metrics,
            found_files=found_files,
            missing_files=missing_files,
            task_artifacts=task_artifacts,
            prediction_artifacts=prediction_artifacts,
            augmentation_artifacts=augmentation_artifacts,
            log_tail=log_tail,
            metrics_source=metrics_source,
        )

    def _inspect_collect_patterns(
        self,
        result_dir: Path,
        collect_patterns: list[str],
    ) -> tuple[list[str], list[str]]:
        found_files: list[str] = []
        missing_files: list[str] = []

        for pattern in collect_patterns:
            if self._is_glob_pattern(pattern=pattern):
                matches = sorted(result_dir.glob(pattern))
                if matches:
                    found_files.extend(
                        self._relative_path(result_dir=result_dir, path=path)
                        for path in matches
                        if path.is_file()
                    )
                # Glob patterns describe optional artifact groups. Exact files below
                # are still reported as missing, but an empty glob should not make a
                # valid quick/standard collection look failed or incomplete.
            else:
                path = result_dir / pattern
                if path.exists():
                    found_files.append(self._relative_path(result_dir=result_dir, path=path))
                else:
                    missing_files.append(pattern)

        return sorted(set(found_files)), sorted(set(missing_files))

    def _load_metrics(
        self,
        experiment_id: str,
        metric_name: str,
        metrics_path: Path,
        created_at: str,
    ) -> tuple[list[EngineMetricRecord], str | None]:
        task_metrics = self._load_representative_task_metric(
            experiment_id=experiment_id,
            result_dir=metrics_path.parent,
            metric_name=metric_name,
            created_at=created_at,
        )
        if task_metrics:
            return task_metrics, self._relative_path(result_dir=metrics_path.parent, path=metrics_path.parent / 'task_results.json')

        metrics_json_path = metrics_path.with_suffix('.json')
        json_metrics = self._load_metrics_json(
            experiment_id=experiment_id,
            metrics_json_path=metrics_json_path,
            created_at=created_at,
        )
        if json_metrics:
            return json_metrics, self._relative_path(result_dir=metrics_path.parent, path=metrics_json_path)

        csv_metrics = self._load_metrics_csv(
            experiment_id=experiment_id,
            metrics_path=metrics_path,
            created_at=created_at,
        )
        if csv_metrics:
            return csv_metrics, self._relative_path(result_dir=metrics_path.parent, path=metrics_path)

        return [], None

    def _load_representative_task_metric(
        self,
        experiment_id: str,
        result_dir: Path,
        metric_name: str,
        created_at: str,
    ) -> list[EngineMetricRecord]:
        task_results_path = result_dir / 'task_results.json'
        if not task_results_path.exists():
            return []

        try:
            data = json.loads(task_results_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, dict):
            return []

        task_results = data.get('task_results')
        if not isinstance(task_results, list):
            return []

        for item in reversed(task_results):
            if not isinstance(item, dict):
                continue
            rows = item.get('metrics')
            if not isinstance(rows, list):
                continue
            for row in reversed(rows):
                if not isinstance(row, dict):
                    continue
                if row.get(metric_name) in {None, ''}:
                    continue
                return [
                    self._metric_record_from_mapping(
                        experiment_id=experiment_id,
                        row=self._normalize_representative_task_metric_row(item=item, row=row),
                        created_at=created_at,
                    ),
                ]

        return []

    def _normalize_representative_task_metric_row(self, *, item: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        record = item.get('record')
        task_id = ''
        execution_mode = ''
        if isinstance(record, dict):
            task_id = str(record.get('task_id') or '').lower()
            params = record.get('params')
            if isinstance(params, dict):
                execution_mode = str(params.get('execution_mode') or '').lower()
        if task_id.startswith(('predict_', 'infer_', 'eval_', 'test_')) or execution_mode in {'inference', 'inference_smoke', 'predict', 'eval', 'test'}:
            normalized['epoch'] = ''
        return normalized

    def _load_metrics_json(
        self,
        experiment_id: str,
        metrics_json_path: Path,
        created_at: str,
    ) -> list[EngineMetricRecord]:
        if not metrics_json_path.exists():
            return []

        try:
            data = json.loads(metrics_json_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(data, dict):
            return []
        records = data.get('records')
        if not isinstance(records, list):
            return []

        return [
            self._metric_record_from_mapping(
                experiment_id=experiment_id,
                row=row,
                created_at=created_at,
            )
            for row in records
            if isinstance(row, dict)
        ]

    def _load_metrics_csv(
        self,
        experiment_id: str,
        metrics_path: Path,
        created_at: str,
    ) -> list[EngineMetricRecord]:
        if not metrics_path.exists():
            return []

        with metrics_path.open(mode='r', encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))

        return [
            self._metric_record_from_mapping(
                experiment_id=experiment_id,
                row=row,
                created_at=created_at,
            )
            for row in rows
        ]

    def _metric_record_from_mapping(
        self,
        experiment_id: str,
        row: dict[str, Any],
        created_at: str,
    ) -> EngineMetricRecord:
        return EngineMetricRecord(
            experiment_id=experiment_id,
            epoch=self._optional_int(row.get('epoch')),
            train_loss=self._optional_float(row.get('train_loss')),
            val_loss=self._optional_float(row.get('val_loss')),
            accuracy=self._optional_float(row.get('accuracy')),
            precision=self._optional_float(row.get('precision')),
            recall=self._optional_float(row.get('recall')),
            macro_precision=self._optional_float(row.get('macro_precision')),
            macro_recall=self._optional_float(row.get('macro_recall')),
            macro_f1=self._optional_float(row.get('macro_f1')),
            class_recall=self._optional_float(row.get('class_recall')),
            class_ap50=self._optional_float(row.get('class_ap50')),
            object_accuracy=self._optional_float(row.get('object_accuracy')),
            map50=self._optional_float(row.get('map50')),
            map50_95=self._optional_float(row.get('map50_95')),
            num_predictions=self._optional_float(row.get('num_predictions')),
            num_gt=self._optional_float(row.get('num_gt')),
            mask_count=self._optional_float(row.get('mask_count')),
            mask_coverage=self._optional_float(row.get('mask_coverage')),
            embedding_count=self._optional_float(row.get('embedding_count')),
            embedding_dim=self._optional_float(row.get('embedding_dim')),
            retrieval_map=self._optional_float(row.get('retrieval_map')),
            neighbor_purity=self._optional_float(row.get('neighbor_purity')),
            review_hit_rate=self._optional_float(row.get('review_hit_rate')),
            label_error_rate=self._optional_float(row.get('label_error_rate')),
            latency_ms_per_image=self._optional_float(row.get('latency_ms_per_image')),
            p95_latency_ms=self._optional_float(row.get('p95_latency_ms')),
            gpu_memory_mb=self._optional_float(row.get('gpu_memory_mb')),
            lr=self._optional_float(row.get('lr')),
            created_at=created_at,
        )

    def _read_log_tail(self, log_path: Path, line_count: int) -> str:
        return read_text_tail(path=log_path, line_count=line_count)

    def _collect_task_artifacts(self, result_dir: Path) -> list[TaskArtifactCollection]:
        task_root = result_dir / 'tasks'
        if not task_root.exists():
            return self._collect_task_artifacts_from_task_results(result_dir=result_dir)

        task_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
        if not task_dirs:
            return self._collect_task_artifacts_from_task_results(result_dir=result_dir)

        task_collections: list[TaskArtifactCollection] = []
        for task_dir in task_dirs:
            validation = self.task_validator.validate_task_dir(task_dir=task_dir)
            timing_rows = self._read_task_timing_rows(task_dir=task_dir)
            metrics_rows = self._read_task_metric_rows(task_dir=task_dir)
            task_collections.append(
                TaskArtifactCollection(
                    task_dir=self._relative_path(result_dir=result_dir, path=task_dir),
                    is_valid=validation.is_valid,
                    elapsed_seconds=self._task_total_seconds(timing_rows=timing_rows),
                    timing_rows=timing_rows,
                    metrics_rows=metrics_rows,
                    errors=validation.error_messages(),
                ),
            )

        return task_collections

    def _collect_task_artifacts_from_task_results(self, *, result_dir: Path) -> list[TaskArtifactCollection]:
        task_results_path = result_dir / 'task_results.json'
        if not task_results_path.exists():
            return []

        try:
            data = json.loads(task_results_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, dict):
            return []

        task_results = data.get('task_results')
        if not isinstance(task_results, list):
            return []

        collections: list[TaskArtifactCollection] = []
        for index, item in enumerate(task_results):
            if not isinstance(item, dict):
                continue
            task_dir = self._task_dir_from_task_result(item=item, index=index)
            timing_rows = self._task_timing_rows_from_task_result(item=item)
            metrics_rows = self._task_metric_rows_from_task_result(item=item)
            collections.append(
                TaskArtifactCollection(
                    task_dir=task_dir,
                    is_valid=self._task_result_is_valid(item=item),
                    elapsed_seconds=self._task_elapsed_seconds(item=item, timing_rows=timing_rows),
                    timing_rows=timing_rows,
                    metrics_rows=metrics_rows,
                    errors=self._task_result_errors(item=item),
                ),
            )

        return collections

    def _task_dir_from_task_result(self, *, item: dict[str, object], index: int) -> str:
        record = item.get('record')
        if isinstance(record, dict):
            result_dir = record.get('result_dir')
            if isinstance(result_dir, str) and result_dir:
                marker = '/results/'
                normalized = result_dir.replace('\\', '/')
                if marker in normalized:
                    return normalized.split(marker, maxsplit=1)[1]
                task_id = record.get('task_id')
                if isinstance(task_id, str) and task_id:
                    return f'tasks/{index:02}_{task_id}'

        return f'tasks/{index:02}'

    def _task_timing_rows_from_task_result(self, *, item: dict[str, object]) -> list[dict[str, object]]:
        rows = item.get('timings')
        if not isinstance(rows, list):
            return []

        timing_rows: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stage = str(row.get('stage') or '').strip()
            if not stage:
                continue
            timing_rows.append(
                {
                    'stage': stage,
                    'elapsed_seconds': self._optional_float(row.get('elapsed_seconds')),
                },
            )

        return timing_rows

    def _task_metric_rows_from_task_result(self, *, item: dict[str, object]) -> list[dict[str, object]]:
        rows = item.get('metrics')
        if not isinstance(rows, list):
            return []

        return [
            dict(row)
            for row in rows
            if isinstance(row, dict)
        ]

    def _task_result_is_valid(self, *, item: dict[str, object]) -> bool:
        validation_errors = item.get('validation_errors')
        if isinstance(validation_errors, list) and validation_errors:
            return False

        return item.get('success') is True and item.get('status') == 'finished'

    def _task_elapsed_seconds(
        self,
        *,
        item: dict[str, object],
        timing_rows: list[dict[str, object]],
    ) -> float | None:
        elapsed_seconds = self._optional_float(item.get('elapsed_seconds'))
        if elapsed_seconds is not None:
            return elapsed_seconds

        return self._task_total_seconds(timing_rows=timing_rows)

    def _task_result_errors(self, *, item: dict[str, object]) -> list[str]:
        errors: list[str] = []
        validation_errors = item.get('validation_errors')
        if isinstance(validation_errors, list):
            errors.extend(str(error) for error in validation_errors if error)

        if item.get('success') is not True:
            message = item.get('message')
            if isinstance(message, str) and message:
                errors.append(message)

        return errors

    def _read_task_timing_rows(self, *, task_dir: Path) -> list[dict[str, object]]:
        timing_path = task_dir / 'timings.csv'
        if not timing_path.exists():
            return []
        with timing_path.open(mode='r', encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))

        timing_rows: list[dict[str, object]] = []
        for row in rows:
            stage = str(row.get('stage') or '').strip()
            if not stage:
                continue
            timing_rows.append(
                {
                    'stage': stage,
                    'elapsed_seconds': self._optional_float(row.get('elapsed_seconds')),
                },
            )

        return timing_rows

    def _read_task_metric_rows(self, *, task_dir: Path) -> list[dict[str, object]]:
        metrics_path = task_dir / 'metrics.csv'
        if not metrics_path.exists():
            return []
        with metrics_path.open(mode='r', encoding='utf-8', newline='') as file:
            return [
                dict(row)
                for row in csv.DictReader(file)
            ]

    def _task_total_seconds(self, *, timing_rows: list[dict[str, object]]) -> float | None:
        for row in timing_rows:
            if row.get('stage') == 'task_total':
                value = row.get('elapsed_seconds')
                return value if isinstance(value, float) else None

        return None

    def _collect_prediction_artifacts(self, result_dir: Path) -> list[PredictionArtifactCollection]:
        prediction_paths = self._prediction_artifact_paths(result_dir=result_dir)
        if not prediction_paths:
            return []

        prediction_collections: list[PredictionArtifactCollection] = []
        for prediction_path in prediction_paths:
            expected_task = self._expected_prediction_task(path=prediction_path)
            validation = self.prediction_validator.validate_file(
                prediction_path,
                expected_task=expected_task,
            )
            summary = self._prediction_summary(result_dir=result_dir, path=prediction_path)
            prediction_collections.append(
                PredictionArtifactCollection(
                    path=self._relative_path(result_dir=result_dir, path=prediction_path),
                    task=expected_task,
                    is_valid=validation.is_valid,
                    record_count=summary.get('record_count'),
                    track_count=summary.get('track_count'),
                    frame_range=summary.get('frame_range'),
                    sample_range=summary.get('sample_range'),
                    min_class_recall_class=summary.get('min_class_recall_class')
                    if isinstance(summary.get('min_class_recall_class'), str)
                    else None,
                    min_class_recall=self._summary_float(summary.get('min_class_recall')),
                    per_class_metrics=summary.get('per_class_metrics')
                    if isinstance(summary.get('per_class_metrics'), list)
                    else [],
                    per_class_metrics_csv=summary.get('per_class_metrics_csv')
                    if isinstance(summary.get('per_class_metrics_csv'), str)
                    else None,
                    per_class_metrics_json=summary.get('per_class_metrics_json')
                    if isinstance(summary.get('per_class_metrics_json'), str)
                    else None,
                    per_class_metrics_split=summary.get('per_class_metrics_split')
                    if isinstance(summary.get('per_class_metrics_split'), str)
                    else None,
                    errors=validation.error_messages(),
                ),
            )

        return prediction_collections

    def _prediction_artifact_paths(self, result_dir: Path) -> list[Path]:
        candidates = [
            *result_dir.glob('predictions/*.json'),
            *result_dir.glob('tasks/*_predictions.json'),
            *result_dir.glob('tasks/**/predictions/*.json'),
        ]
        seen: set[str] = set()
        paths: list[Path] = []
        for path in sorted(candidate for candidate in candidates if candidate.is_file()):
            key = path.resolve().as_posix()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)

        return paths

    def _expected_prediction_task(self, path: Path) -> str | None:
        name = path.name
        if name == 'classification_predictions.json':
            return 'classification'
        if name == 'detection_predictions.json':
            return 'detection'
        if name == 'tracking_predictions.json':
            return 'tracking'
        if name == 'segmentation_predictions.json':
            return 'segmentation'
        if name == 'embedding_predictions.json':
            return 'embedding'

        return None

    def _prediction_summary(self, *, result_dir: Path, path: Path) -> dict[str, object]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}
        records = data.get('records')
        if not isinstance(records, list):
            return {}

        summary: dict[str, int | float | str | None] = {
            'record_count': len(records),
        }
        task = data.get('task')
        if task == 'tracking':
            track_ids = sorted({
                str(record.get('track_id'))
                for record in records
                if isinstance(record, dict) and record.get('track_id') is not None
            })
            frame_ids = sorted({
                str(record.get('frame_id'))
                for record in records
                if isinstance(record, dict) and record.get('frame_id') is not None
            })
            sample_ids = sorted({
                str(record.get('sample_id'))
                for record in records
                if isinstance(record, dict) and record.get('sample_id') is not None
            })
            summary['track_count'] = len(track_ids)
            summary['frame_range'] = self._range_text(frame_ids)
            summary['sample_range'] = self._range_text(sample_ids)
        if task == 'classification':
            summary.update(self._classification_prediction_summary(result_dir=result_dir, path=path, records=records))

        return summary

    def _classification_prediction_summary(
        self,
        *,
        result_dir: Path,
        path: Path,
        records: list[object],
    ) -> dict[str, object]:
        manifest_path = self._classification_manifest_for_predictions(path=path)
        if manifest_path is None:
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return {}
        raw_images = manifest.get('images') if isinstance(manifest, dict) else None
        if not isinstance(raw_images, list) or len(raw_images) != len(records):
            return {}
        rows: list[tuple[str, str, str | None]] = []
        for image, record in zip(raw_images, records, strict=True):
            if not isinstance(image, dict) or not isinstance(record, dict):
                return {}
            label = image.get('label')
            prediction = record.get('top1_class_id')
            if not isinstance(label, str) or not label or not isinstance(prediction, str) or not prediction:
                return {}
            split = image.get('split')
            rows.append((label, prediction, split if isinstance(split, str) and split else None))

        split_name = 'test' if any(split == 'test' for _, _, split in rows) else 'all'
        selected_rows = [
            (label, prediction)
            for label, prediction, split in rows
            if split_name == 'all' or split == split_name
        ]
        if not selected_rows:
            return {}

        manifest_classes = manifest.get('classes')
        if isinstance(manifest_classes, list):
            class_ids = [
                class_id
                for class_id in manifest_classes
                if isinstance(class_id, str) and class_id
            ]
        else:
            class_ids = []
        class_ids = sorted(set(class_ids) | {label for label, _ in selected_rows} | {prediction for _, prediction in selected_rows})
        if not class_ids:
            return {}

        per_class_metrics = self._classification_per_class_metrics(
            class_ids=class_ids,
            rows=selected_rows,
        )
        self._write_per_class_metrics_files(
            result_dir=result_dir,
            prediction_path=path,
            manifest_path=manifest_path,
            split_name=split_name,
            per_class_metrics=per_class_metrics,
        )

        recalls: dict[str, float] = {}
        for class_id in class_ids:
            metric = next((item for item in per_class_metrics if item.get('class_id') == class_id), None)
            if metric is None:
                continue
            support = metric.get('support')
            if support == 0:
                continue
            recall = metric.get('recall')
            if isinstance(recall, (int, float)) and not isinstance(recall, bool):
                recalls[class_id] = float(recall)
        if not recalls:
            return {}
        min_class, min_recall = min(recalls.items(), key=lambda item: (item[1], item[0]))

        return {
            'min_class_recall_class': min_class,
            'min_class_recall': min_recall,
            'per_class_metrics': per_class_metrics,
            'per_class_metrics_csv': 'per_class_metrics.csv',
            'per_class_metrics_json': 'per_class_metrics.json',
            'per_class_metrics_split': split_name,
        }

    def _classification_per_class_metrics(
        self,
        *,
        class_ids: list[str],
        rows: list[tuple[str, str]],
    ) -> list[dict[str, object]]:
        metrics: list[dict[str, object]] = []
        for class_id in class_ids:
            true_positive = sum(1 for label, prediction in rows if label == class_id and prediction == class_id)
            false_positive = sum(1 for label, prediction in rows if label != class_id and prediction == class_id)
            false_negative = sum(1 for label, prediction in rows if label == class_id and prediction != class_id)
            support = true_positive + false_negative
            predicted = true_positive + false_positive
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / support if support else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            metrics.append(
                {
                    'class_id': class_id,
                    'support': support,
                    'predicted': predicted,
                    'true_positive': true_positive,
                    'false_positive': false_positive,
                    'false_negative': false_negative,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'correct': true_positive,
                    'total': support,
                },
            )

        return metrics

    def _write_per_class_metrics_files(
        self,
        *,
        result_dir: Path,
        prediction_path: Path,
        manifest_path: Path,
        split_name: str,
        per_class_metrics: list[dict[str, object]],
    ) -> None:
        csv_path = result_dir / 'per_class_metrics.csv'
        json_path = result_dir / 'per_class_metrics.json'
        fieldnames = [
            'class_id',
            'support',
            'predicted',
            'true_positive',
            'false_positive',
            'false_negative',
            'precision',
            'recall',
            'f1',
            'correct',
            'total',
        ]
        with csv_path.open(mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_class_metrics)

        payload = {
            'schema_version': '0.1',
            'artifact_type': 'per_class_classification_metrics',
            'split': split_name,
            'source_prediction_path': self._relative_path(result_dir=result_dir, path=prediction_path),
            'source_manifest_path': self._relative_path(result_dir=result_dir, path=manifest_path),
            'records': per_class_metrics,
        }
        json_path.write_text(
            data=json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    def _classification_manifest_for_predictions(self, *, path: Path) -> Path | None:
        candidates = [
            path.parent.parent / 'classification_input_manifest.json',
            path.parent / 'classification_input_manifest.json',
            path.parent.parent.parent / 'classification_input_manifest.json',
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _range_text(self, values: list[str]) -> str | None:
        if not values:
            return None
        if len(values) == 1:
            return values[0]

        return f'{values[0]}..{values[-1]}'

    def _collect_augmentation_artifacts(self, result_dir: Path) -> list[AugmentationArtifactCollection]:
        augmentation_paths = self._augmentation_artifact_paths(result_dir=result_dir)
        if not augmentation_paths:
            return []

        collections: list[AugmentationArtifactCollection] = []
        for path in augmentation_paths:
            summary, errors = self._augmentation_summary(path=path)
            collections.append(
                AugmentationArtifactCollection(
                    path=self._relative_path(result_dir=result_dir, path=path),
                    is_valid=not errors,
                    policy_id=summary.get('policy_id') if isinstance(summary.get('policy_id'), str) else None,
                    target_split=summary.get('target_split') if isinstance(summary.get('target_split'), str) else None,
                    augmentation_count=self._summary_int(summary.get('augmentation_count')),
                    record_count=self._summary_int(summary.get('record_count')),
                    source_dataset_id=summary.get('source_dataset_id') if isinstance(summary.get('source_dataset_id'), str) else None,
                    dataset_id=summary.get('dataset_id') if isinstance(summary.get('dataset_id'), str) else None,
                    label_transform_count=self._summary_int(summary.get('label_transform_count')),
                    bbox_transform_count=self._summary_int(summary.get('bbox_transform_count')),
                    bbox_drop_count=self._summary_int(summary.get('bbox_drop_count')),
                    dropped_object_count=self._summary_int(summary.get('dropped_object_count')),
                    mask_transform_count=self._summary_int(summary.get('mask_transform_count')),
                    mask_record_transform_count=self._summary_int(summary.get('mask_record_transform_count')),
                    mask_object_transform_count=self._summary_int(summary.get('mask_object_transform_count')),
                    mask_artifact_count=self._summary_int(summary.get('mask_artifact_count')),
                    errors=errors,
                ),
            )

        return collections

    def _augmentation_artifact_paths(self, result_dir: Path) -> list[Path]:
        candidates = [
            *result_dir.glob('augmentation_manifest.json'),
            *result_dir.glob('tasks/**/augmentation_manifest.json'),
        ]
        seen: set[str] = set()
        paths: list[Path] = []
        for path in sorted(candidate for candidate in candidates if candidate.is_file()):
            key = path.resolve().as_posix()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)

        return paths

    def _augmentation_summary(self, path: Path) -> tuple[dict[str, object], list[str]]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as error:
            return {}, [f'{path.name}: invalid augmentation manifest: {error}']

        if not isinstance(data, dict):
            return {}, [f'{path.name}: JSON root must be an object']

        errors: list[str] = []
        if data.get('artifact_type') != 'augmentation_manifest':
            errors.append(f"{path.name}.artifact_type: expected 'augmentation_manifest'")
        records = data.get('records')
        if not isinstance(records, list):
            errors.append(f'{path.name}.records: records must be a list')
            records = []

        summary: dict[str, object] = {
            'policy_id': data.get('policy_id'),
            'target_split': data.get('target_split'),
            'augmentation_count': data.get('augmentation_count'),
            'record_count': len(records),
            'source_dataset_id': data.get('source_dataset_id'),
            'dataset_id': data.get('dataset_id'),
            'label_transform_count': self._record_field_count(records=records, field_name='label_transform'),
            'bbox_transform_count': self._bbox_transform_count(records=records),
            'bbox_drop_count': self._bbox_drop_count(records=records),
            'dropped_object_count': self._dropped_object_count(records=records),
            'mask_transform_count': self._mask_transform_reference_count(records=records),
            'mask_record_transform_count': self._mask_record_transform_count(records=records),
            'mask_object_transform_count': self._mask_object_transform_count(records=records),
            'mask_artifact_count': self._mask_artifact_count(records=records),
        }
        augmentation_count = data.get('augmentation_count')
        if isinstance(augmentation_count, int) and augmentation_count != len(records):
            errors.append(
                f'{path.name}.augmentation_count: expected {len(records)}, got {augmentation_count}',
            )

        return summary, errors

    def _record_field_count(self, *, records: list[object], field_name: str) -> int:
        return sum(
            1
            for record in records
            if isinstance(record, dict) and isinstance(record.get(field_name), str) and record.get(field_name)
        )

    def _bbox_transform_count(self, *, records: list[object]) -> int:
        count = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            objects = record.get('objects')
            if not isinstance(objects, list):
                continue
            count += sum(
                1
                for item in objects
                if isinstance(item, dict) and isinstance(item.get('bbox_transform'), str) and item.get('bbox_transform')
            )

        return count

    def _bbox_drop_count(self, *, records: list[object]) -> int:
        count = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            value = record.get('bbox_drop_count')
            if isinstance(value, int) and not isinstance(value, bool):
                count += value

        return count

    def _dropped_object_count(self, *, records: list[object]) -> int:
        count = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            dropped_objects = record.get('dropped_objects')
            if isinstance(dropped_objects, list):
                count += len([item for item in dropped_objects if isinstance(item, dict)])

        return count

    def _mask_transform_reference_count(self, *, records: list[object]) -> int:
        return (
            self._mask_record_transform_count(records=records)
            + self._mask_object_transform_count(records=records)
        )

    def _mask_record_transform_count(self, *, records: list[object]) -> int:
        return sum(
            1
            for record in records
            if isinstance(record, dict) and isinstance(record.get('mask_transform'), str) and record.get('mask_transform')
        )

    def _mask_object_transform_count(self, *, records: list[object]) -> int:
        count = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            objects = record.get('objects')
            if not isinstance(objects, list):
                continue
            count += sum(
                1
                for item in objects
                if isinstance(item, dict) and isinstance(item.get('mask_transform'), str) and item.get('mask_transform')
            )

        return count

    def _mask_artifact_count(self, *, records: list[object]) -> int:
        paths: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            mask_path = record.get('mask_path')
            if isinstance(mask_path, str) and mask_path:
                paths.add(mask_path)
            objects = record.get('objects')
            if not isinstance(objects, list):
                continue
            for item in objects:
                if not isinstance(item, dict):
                    continue
                object_mask_path = item.get('mask_path')
                if isinstance(object_mask_path, str) and object_mask_path:
                    paths.add(object_mask_path)

        return len(paths)

    def _summary_int(self, value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value

        return None

    def _summary_float(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)

        return None

    def _relative_path(self, result_dir: Path, path: Path) -> str:
        try:
            return path.relative_to(result_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _is_glob_pattern(self, pattern: str) -> bool:
        return any(token in pattern for token in ['*', '?', '['])

    def _optional_int(self, value: object) -> int | None:
        if value is None or value == '':
            return None

        return int(value)

    def _optional_float(self, value: object) -> float | None:
        if value is None or value == '':
            return None

        return float(value)
