from dataclasses import dataclass, field

from ironflow_exp.engine.collector import CollectionResult
from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.domain import EngineMetricRecord


@dataclass(frozen=True, slots=True)
class ExperimentAnalysis:
    best_metric_name: str
    best_metric_value: float | None = None
    best_epoch: int | None = None
    error_type: str | None = None
    observations: list[str] = field(default_factory=list)


class ResultAnalyzer:
    def analyze(
        self,
        config: EngineExperimentConfig,
        collection: CollectionResult,
    ) -> ExperimentAnalysis:
        best_metric_value, best_epoch = self._best_metric(
            metrics=collection.metrics,
            metric_name=config.analysis.primary_metric,
            higher_is_better=config.analysis.higher_is_better,
        )
        missing_files = list(collection.missing_files)
        if collection.metrics:
            missing_files = [
                missing_file
                for missing_file in missing_files
                if missing_file != config.output.metrics_file
            ]

        error_type = self.classify_failure(
            log_text=collection.log_tail,
            missing_files=missing_files,
            metrics_file=config.output.metrics_file,
        )
        if error_type is not None and self._has_successful_task_outputs(collection=collection):
            error_type = None
        if error_type is None and not collection.metrics:
            error_type = 'metric_missing'

        return ExperimentAnalysis(
            best_metric_name=config.analysis.primary_metric,
            best_metric_value=best_metric_value,
            best_epoch=best_epoch,
            error_type=error_type,
            observations=self._observations(
                collection=collection,
                best_metric_name=config.analysis.primary_metric,
                best_metric_value=best_metric_value,
                best_epoch=best_epoch,
            ),
        )

    def classify_failure(
        self,
        log_text: str,
        missing_files: list[str] | None = None,
        metrics_file: str = 'metrics.csv',
    ) -> str | None:
        normalized_missing = set(missing_files or [])
        if metrics_file in normalized_missing:
            return 'metric_missing'
        if 'CUDA out of memory' in log_text:
            return 'cuda_oom'
        if 'dataset path not found' in log_text:
            return 'dataset_path_error'
        if 'No module named' in log_text:
            return 'dependency_error'

        return None

    def _has_successful_task_outputs(self, collection: CollectionResult) -> bool:
        return bool(collection.metrics) and bool(collection.task_artifacts) and all(
            task_artifact.is_valid
            for task_artifact in collection.task_artifacts
        )

    def _best_metric(
        self,
        metrics: list[EngineMetricRecord],
        metric_name: str,
        higher_is_better: bool,
    ) -> tuple[float | None, int | None]:
        candidates: list[tuple[float, int | None]] = []
        for metric in metrics:
            value = getattr(metric, metric_name, None)
            if value is not None:
                candidates.append((float(value), metric.epoch))

        if not candidates:
            return None, None

        best_value, best_epoch = (
            max(candidates, key=lambda item: item[0])
            if higher_is_better
            else min(candidates, key=lambda item: item[0])
        )

        return best_value, best_epoch

    def _observations(
        self,
        collection: CollectionResult,
        best_metric_name: str,
        best_metric_value: float | None,
        best_epoch: int | None,
    ) -> list[str]:
        observations: list[str] = []
        if collection.missing_files:
            observations.append(f"missing files: {', '.join(collection.missing_files)}")
        invalid_task_artifacts = [
            task_artifact
            for task_artifact in collection.task_artifacts
            if not task_artifact.is_valid
        ]
        if invalid_task_artifacts:
            observations.append(f'invalid task artifact dirs: {len(invalid_task_artifacts)}')
        if collection.task_artifacts:
            observations.append(f'task artifact dirs: {len(collection.task_artifacts)}')
        invalid_prediction_artifacts = [
            prediction_artifact
            for prediction_artifact in collection.prediction_artifacts
            if not prediction_artifact.is_valid
        ]
        if invalid_prediction_artifacts:
            observations.append(f'invalid prediction artifacts: {len(invalid_prediction_artifacts)}')
        if collection.prediction_artifacts:
            observations.append(f'prediction artifacts: {len(collection.prediction_artifacts)}')
            observations.extend(self._prediction_observations(collection=collection))
        invalid_augmentation_artifacts = [
            augmentation_artifact
            for augmentation_artifact in collection.augmentation_artifacts
            if not augmentation_artifact.is_valid
        ]
        if invalid_augmentation_artifacts:
            observations.append(f'invalid augmentation artifacts: {len(invalid_augmentation_artifacts)}')
        if collection.augmentation_artifacts:
            observations.append(f'augmentation artifacts: {len(collection.augmentation_artifacts)}')
            observations.extend(self._augmentation_observations(collection=collection))
        if collection.metrics_source is not None:
            observations.append(f'metrics source: {collection.metrics_source}')
        if best_metric_value is not None:
            observations.append(f'best {best_metric_name}: {best_metric_value} at epoch {best_epoch}')
        if not collection.metrics:
            observations.append('no metric rows collected')

        return observations

    def _augmentation_observations(self, collection: CollectionResult) -> list[str]:
        observations: list[str] = []
        for artifact in collection.augmentation_artifacts:
            parts = [f'augmentation artifact: {artifact.path}']
            if artifact.policy_id:
                parts.append(f'policy={artifact.policy_id}')
            if artifact.target_split:
                parts.append(f'target_split={artifact.target_split}')
            if artifact.augmentation_count is not None:
                parts.append(f'augmentations={artifact.augmentation_count}')
            if artifact.record_count is not None:
                parts.append(f'records={artifact.record_count}')
            if artifact.label_transform_count:
                parts.append(f'label_transforms={artifact.label_transform_count}')
            if artifact.bbox_transform_count:
                parts.append(f'bbox_transforms={artifact.bbox_transform_count}')
            if artifact.bbox_drop_count:
                parts.append(f'bbox_drops={artifact.bbox_drop_count}')
            if artifact.dropped_object_count:
                parts.append(f'dropped_objects={artifact.dropped_object_count}')
            if artifact.mask_record_transform_count:
                parts.append(f'mask_record_transforms={artifact.mask_record_transform_count}')
            if artifact.mask_object_transform_count:
                parts.append(f'mask_object_transforms={artifact.mask_object_transform_count}')
            if artifact.mask_transform_count:
                parts.append(f'mask_transform_refs={artifact.mask_transform_count}')
            if artifact.mask_artifact_count:
                parts.append(f'mask_artifacts={artifact.mask_artifact_count}')
            observations.append('; '.join(parts))

        return observations

    def _prediction_observations(self, collection: CollectionResult) -> list[str]:
        observations: list[str] = []
        for artifact in collection.prediction_artifacts:
            if artifact.task is None:
                continue
            parts = [f'{artifact.task} prediction artifact: {artifact.path}']
            if artifact.record_count is not None:
                parts.append(f'records={artifact.record_count}')
            if artifact.task == 'tracking' and artifact.track_count is not None:
                parts.append(f'tracks={artifact.track_count}')
            if artifact.task == 'tracking' and artifact.frame_range:
                parts.append(f'frames={artifact.frame_range}')
            if artifact.task == 'tracking' and artifact.sample_range:
                parts.append(f'samples={artifact.sample_range}')
            observations.append('; '.join(parts))

        return observations
