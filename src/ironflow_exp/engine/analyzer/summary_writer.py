from pathlib import Path

from ironflow_exp.engine.analyzer.result_analyzer import ExperimentAnalysis
from ironflow_exp.engine.collector import CollectionResult
from ironflow_exp.engine.domain import ExperimentRecord


class SummaryWriter:
    def write_summary(
        self,
        record: ExperimentRecord,
        analysis: ExperimentAnalysis,
        collection: CollectionResult,
        summary_path: Path,
    ) -> Path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            data=self._summary_text(
                record=record,
                analysis=analysis,
                collection=collection,
            ),
            encoding='utf-8',
        )

        return summary_path

    def _summary_text(
        self,
        record: ExperimentRecord,
        analysis: ExperimentAnalysis,
        collection: CollectionResult,
    ) -> str:
        best_metric_value = 'n/a' if analysis.best_metric_value is None else str(analysis.best_metric_value)
        best_epoch = 'n/a' if analysis.best_epoch is None else str(analysis.best_epoch)
        metrics_source = collection.metrics_source or 'n/a'
        error_type = analysis.error_type or 'none'
        observations = analysis.observations or ['none']
        found_files = collection.found_files or ['none']
        missing_files = collection.missing_files or ['none']
        task_artifact_count = len(collection.task_artifacts)
        invalid_task_artifact_count = len([
            task_artifact
            for task_artifact in collection.task_artifacts
            if not task_artifact.is_valid
        ])
        prediction_artifact_count = len(collection.prediction_artifacts)
        invalid_prediction_artifact_count = len([
            prediction_artifact
            for prediction_artifact in collection.prediction_artifacts
            if not prediction_artifact.is_valid
        ])
        augmentation_artifact_count = len(collection.augmentation_artifacts)
        invalid_augmentation_artifact_count = len([
            augmentation_artifact
            for augmentation_artifact in collection.augmentation_artifacts
            if not augmentation_artifact.is_valid
        ])

        lines = [
            '# Experiment Summary',
            '',
            '## Basic Info',
            f'- experiment_id: {record.experiment_id}',
            f'- name: {record.name}',
            f'- status: {record.status.value}',
            f'- runner_type: {record.runner_type}',
            f'- result_path: {record.result_path or ""}',
            '',
            '## Best Metric',
            f'- metric: {analysis.best_metric_name}',
            f'- value: {best_metric_value}',
            f'- epoch: {best_epoch}',
            f'- source: {metrics_source}',
            '',
            '## Latest Metrics',
            *self._latest_metric_lines(collection=collection),
            '',
            '## Task Metrics by Stage',
            *self._task_metric_lines(collection=collection),
            '',
            '## Collection',
            f'- metrics_count: {len(collection.metrics)}',
            f'- found_files: {", ".join(found_files)}',
            f'- missing_files: {", ".join(missing_files)}',
            f'- task_artifact_dirs: {task_artifact_count}',
            f'- invalid_task_artifact_dirs: {invalid_task_artifact_count}',
            f'- prediction_artifacts: {prediction_artifact_count}',
            f'- invalid_prediction_artifacts: {invalid_prediction_artifact_count}',
            f'- augmentation_artifacts: {augmentation_artifact_count}',
            f'- invalid_augmentation_artifacts: {invalid_augmentation_artifact_count}',
            '',
            '## Prediction Artifacts',
        ]
        if collection.prediction_artifacts:
            lines.extend(
                f'- {self._prediction_artifact_line(artifact)}'
                for artifact in collection.prediction_artifacts
            )
        else:
            lines.append('- none')
        lines.extend([
            '',
            *self._per_class_metric_lines(collection=collection),
        ])
        lines.extend([
            '',
            '## Augmentation Artifacts',
        ])
        if collection.augmentation_artifacts:
            lines.extend(
                f'- {self._augmentation_artifact_line(artifact)}'
                for artifact in collection.augmentation_artifacts
            )
        else:
            lines.append('- none')
        lines.extend([
            '',
            '## Failure',
            f'- error_type: {error_type}',
            '',
            '## Observations',
        ])
        lines.extend(f'- {observation}' for observation in observations)
        lines.append('')

        return '\n'.join(lines)

    def _latest_metric_lines(self, collection: CollectionResult) -> list[str]:
        if not collection.metrics:
            return ['- none']
        metric = collection.metrics[-1]
        lines: list[str] = []
        is_classification_metric = any(
            self._has_metric_value(getattr(metric, attribute, None))
            for attribute in ('macro_precision', 'macro_recall', 'macro_f1', 'class_recall')
        )
        for attribute in [
            'accuracy',
            'precision',
            'recall',
            'macro_precision',
            'macro_recall',
            'macro_f1',
            'class_recall',
            'object_accuracy',
            'map50',
            'map50_95',
            'num_predictions',
            'num_gt',
            'label_error_rate',
            'latency_ms_per_image',
            'p95_latency_ms',
            'gpu_memory_mb',
        ]:
            if is_classification_metric and attribute in {'precision', 'recall'}:
                continue
            value = getattr(metric, attribute, None)
            if self._has_metric_value(value):
                label = 'min_class_recall' if attribute == 'class_recall' else attribute
                detail = self._min_class_recall_detail(collection=collection) if attribute == 'class_recall' else ''
                lines.append(f'- {label}: {value}{detail}')

        return lines or ['- none']

    def _task_metric_lines(self, collection: CollectionResult) -> list[str]:
        lines: list[str] = []
        for task_artifact in collection.task_artifacts:
            metrics_rows = getattr(task_artifact, 'metrics_rows', None)
            if not isinstance(metrics_rows, list) or not metrics_rows:
                continue
            row = self._last_non_empty_metric_row(rows=metrics_rows)
            if row is None:
                continue
            task_dir = getattr(task_artifact, 'task_dir', 'unknown')
            profile = self._task_metric_profile(task_dir=str(task_dir), row=row)
            metric_parts = self._profile_metric_parts(row=row, profile=profile)
            if not metric_parts:
                continue
            lines.append(f'- {task_dir} ({profile}): {", ".join(metric_parts)}')

        return lines or ['- none']

    def _last_non_empty_metric_row(self, *, rows: list[object]) -> dict[str, object] | None:
        for row in reversed(rows):
            if not isinstance(row, dict):
                continue
            if any(self._has_metric_value(value) for key, value in row.items() if key != 'epoch'):
                return row

        return None

    def _task_metric_profile(self, *, task_dir: str, row: dict[str, object]) -> str:
        lowered = task_dir.lower()
        if self._row_has_any(row=row, keys=('map50', 'map50_95')) or 'detect' in lowered or 'yolo' in lowered:
            if 'classify' not in lowered and 'classification' not in lowered:
                return 'detection'
        if self._row_has_any(row=row, keys=('mask_count', 'mask_coverage')) or 'seg' in lowered or 'mask' in lowered:
            return 'segmentation'
        if self._row_has_any(row=row, keys=('retrieval_map', 'neighbor_purity', 'review_hit_rate', 'embedding_count')):
            return 'embedding'
        if self._row_has_any(row=row, keys=('macro_precision', 'macro_recall', 'macro_f1', 'class_recall', 'accuracy')):
            return 'classification'

        return 'generic'

    def _row_has_any(self, *, row: dict[str, object], keys: tuple[str, ...]) -> bool:
        return any(self._has_metric_value(row.get(key)) for key in keys)

    def _profile_metric_parts(self, *, row: dict[str, object], profile: str) -> list[str]:
        keys_by_profile = {
            'detection': (
                'precision',
                'recall',
                'map50',
                'map50_95',
                'num_predictions',
                'num_gt',
                'p95_latency_ms',
                'latency_ms_per_image',
            ),
            'classification': (
                'accuracy',
                'macro_precision',
                'macro_recall',
                'macro_f1',
                'class_recall',
                'num_predictions',
                'num_gt',
                'p95_latency_ms',
                'latency_ms_per_image',
            ),
            'segmentation': (
                'mask_count',
                'mask_coverage',
                'map50',
                'map50_95',
                'p95_latency_ms',
                'latency_ms_per_image',
            ),
            'embedding': (
                'embedding_count',
                'embedding_dim',
                'retrieval_map',
                'neighbor_purity',
                'review_hit_rate',
                'p95_latency_ms',
                'latency_ms_per_image',
            ),
            'generic': (
                'accuracy',
                'precision',
                'recall',
                'macro_f1',
                'map50',
                'map50_95',
                'object_accuracy',
                'p95_latency_ms',
                'latency_ms_per_image',
            ),
        }
        parts: list[str] = []
        epoch = row.get('epoch')
        if self._has_metric_value(epoch):
            parts.append(f'epoch={epoch}')
        for key in keys_by_profile[profile]:
            value = row.get(key)
            if self._has_metric_value(value):
                label = 'min_class_recall' if key == 'class_recall' else key
                parts.append(f'{label}={value}')

        return parts

    def _has_metric_value(self, value: object) -> bool:
        return value not in {None, ''}

    def _min_class_recall_detail(self, collection: CollectionResult) -> str:
        for artifact in collection.prediction_artifacts:
            if getattr(artifact, 'task', None) != 'classification':
                continue
            class_id = getattr(artifact, 'min_class_recall_class', None)
            recall = getattr(artifact, 'min_class_recall', None)
            if class_id and recall is not None:
                return f' (class={class_id})'

        return ''

    def _per_class_metric_lines(self, collection: CollectionResult) -> list[str]:
        for artifact in collection.prediction_artifacts:
            if getattr(artifact, 'task', None) != 'classification':
                continue
            rows = getattr(artifact, 'per_class_metrics', None)
            if not isinstance(rows, list) or not rows:
                continue
            split = getattr(artifact, 'per_class_metrics_split', None) or 'all'
            lines = [
                f'## Per-Class Metrics ({split} split)',
                '',
                '| class | support | predicted | precision | recall | f1 | correct | total |',
                '|---|---:|---:|---:|---:|---:|---:|---:|',
            ]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    '| '
                    f'{row.get("class_id", "")} | '
                    f'{self._format_int(row.get("support"))} | '
                    f'{self._format_int(row.get("predicted"))} | '
                    f'{self._format_float(row.get("precision"))} | '
                    f'{self._format_float(row.get("recall"))} | '
                    f'{self._format_float(row.get("f1"))} | '
                    f'{self._format_int(row.get("correct"))} | '
                    f'{self._format_int(row.get("total"))} |'
                )
            lines.extend([
                '',
                '- files: '
                f'{getattr(artifact, "per_class_metrics_csv", None) or "n/a"}, '
                f'{getattr(artifact, "per_class_metrics_json", None) or "n/a"}',
            ])

            return lines

        return ['## Per-Class Metrics', '', '- none']

    def _format_int(self, value: object) -> str:
        if isinstance(value, bool):
            return ''
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))

        return ''

    def _format_float(self, value: object) -> str:
        if isinstance(value, bool):
            return ''
        if isinstance(value, (int, float)):
            return f'{float(value):.6g}'

        return ''

    def _prediction_artifact_line(self, artifact: object) -> str:
        path = getattr(artifact, 'path', 'unknown')
        task = getattr(artifact, 'task', None) or 'unknown'
        valid = 'valid' if getattr(artifact, 'is_valid', False) else 'invalid'
        parts = [f'{path}', f'task={task}', valid]
        record_count = getattr(artifact, 'record_count', None)
        track_count = getattr(artifact, 'track_count', None)
        frame_range = getattr(artifact, 'frame_range', None)
        sample_range = getattr(artifact, 'sample_range', None)
        if record_count is not None:
            parts.append(f'records={record_count}')
        min_class_recall_class = getattr(artifact, 'min_class_recall_class', None)
        min_class_recall = getattr(artifact, 'min_class_recall', None)
        if min_class_recall_class and min_class_recall is not None:
            parts.append(f'min_class_recall={min_class_recall}')
            parts.append(f'min_class_recall_class={min_class_recall_class}')
        per_class_metrics_csv = getattr(artifact, 'per_class_metrics_csv', None)
        per_class_metrics_json = getattr(artifact, 'per_class_metrics_json', None)
        if per_class_metrics_csv:
            parts.append(f'per_class_metrics_csv={per_class_metrics_csv}')
        if per_class_metrics_json:
            parts.append(f'per_class_metrics_json={per_class_metrics_json}')
        if track_count is not None:
            parts.append(f'tracks={track_count}')
        if frame_range:
            parts.append(f'frames={frame_range}')
        if sample_range:
            parts.append(f'samples={sample_range}')

        return '; '.join(parts)

    def _augmentation_artifact_line(self, artifact: object) -> str:
        path = getattr(artifact, 'path', 'unknown')
        valid = 'valid' if getattr(artifact, 'is_valid', False) else 'invalid'
        parts = [f'{path}', valid]
        for attribute, label in [
            ('policy_id', 'policy'),
            ('target_split', 'target_split'),
            ('augmentation_count', 'augmentations'),
            ('record_count', 'records'),
            ('label_transform_count', 'label_transforms'),
            ('bbox_transform_count', 'bbox_transforms'),
            ('bbox_drop_count', 'bbox_drops'),
            ('dropped_object_count', 'dropped_objects'),
            ('mask_record_transform_count', 'mask_record_transforms'),
            ('mask_object_transform_count', 'mask_object_transforms'),
            ('mask_transform_count', 'mask_transform_refs'),
            ('mask_artifact_count', 'mask_artifacts'),
        ]:
            value = getattr(artifact, attribute, None)
            if value is not None and value != 0:
                parts.append(f'{label}={value}')

        return '; '.join(parts)
