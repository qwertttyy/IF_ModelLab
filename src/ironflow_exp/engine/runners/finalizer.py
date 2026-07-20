from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ironflow_exp.engine.analyzer import ResultAnalyzer, SummaryWriter
from ironflow_exp.engine.collector import ResultCollector
from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.domain import ExperimentRecord, ExperimentStatus
from ironflow_exp.engine.runners.base import ExperimentRunnerResult
from ironflow_exp.engine.storage import SQLiteExperimentStorage


class ExperimentResultFinalizer:
    def __init__(
        self,
        storage: SQLiteExperimentStorage,
        result_collector: ResultCollector | None = None,
        result_analyzer: ResultAnalyzer | None = None,
        summary_writer: SummaryWriter | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self.result_collector = result_collector or ResultCollector()
        self.result_analyzer = result_analyzer or ResultAnalyzer()
        self.summary_writer = summary_writer or SummaryWriter()
        self.clock = clock or (lambda: datetime.now(tz=timezone.utc))

    def finalize(
        self,
        record: ExperimentRecord,
        config: EngineExperimentConfig,
        workspace_dir: Path,
        result_dir: Path,
        output_dir: Path | None = None,
    ) -> ExperimentRunnerResult:
        collection = self.result_collector.collect(
            experiment_id=record.experiment_id,
            config=config,
            result_dir=result_dir,
            created_at=self._timestamp(),
        )
        analysis = self.result_analyzer.analyze(config=config, collection=collection)
        summary_path = result_dir / config.output.summary_file if config.output.save_summary else None

        if analysis.error_type == 'metric_missing':
            if summary_path is not None:
                self.summary_writer.write_summary(
                    record=record,
                    analysis=analysis,
                    collection=collection,
                    summary_path=summary_path,
                )
            return self.mark_failed(
                record=self._require_record(experiment_id=record.experiment_id),
                workspace_dir=workspace_dir,
                error_type=analysis.error_type,
                message='metrics file is missing',
            )

        if collection.metrics:
            self.storage.delete_metrics(experiment_id=record.experiment_id)
            self.storage.save_metrics(records=collection.metrics)

        collected_dir = output_dir or result_dir
        latest_record = self._require_record(experiment_id=record.experiment_id)
        timing_summary = self._timing_summary(collection=collection)
        updated_record = replace(
            latest_record,
            status=ExperimentStatus.COLLECTED,
            finished_at=self._timestamp(),
            best_metric_name=analysis.best_metric_name,
            best_metric_value=analysis.best_metric_value,
            error_type=analysis.error_type,
            result_path=str(collected_dir),
            metadata={
                **latest_record.metadata,
                'found_files': collection.found_files,
                'missing_files': collection.missing_files,
                'metrics_source': collection.metrics_source,
                'summary_path': str(summary_path) if summary_path is not None else None,
                'summary_written': summary_path is not None,
                'best_epoch': analysis.best_epoch,
                'observations': analysis.observations,
                'timing_summary': timing_summary,
                'task_artifacts': [task_artifact.to_dict() for task_artifact in collection.task_artifacts],
                'prediction_artifacts': [
                    prediction_artifact.to_dict()
                    for prediction_artifact in collection.prediction_artifacts
                ],
                'augmentation_artifacts': [
                    augmentation_artifact.to_dict()
                    for augmentation_artifact in collection.augmentation_artifacts
                ],
            },
        )
        if summary_path is not None:
            self.summary_writer.write_summary(
                record=updated_record,
                analysis=analysis,
                collection=collection,
                summary_path=summary_path,
            )
        status_marker = result_dir / 'status.marker'
        try:
            status_marker.write_text(ExperimentStatus.COLLECTED.value, encoding='utf-8')
        except OSError:
            pass
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=record.experiment_id,
            status=ExperimentStatus.COLLECTED,
            workspace_dir=workspace_dir,
            message='experiment collected',
            metadata={
                'result_dir': str(result_dir),
                'collected_dir': str(collected_dir),
                'metrics_count': len(collection.metrics),
                'best_metric_name': analysis.best_metric_name,
                'best_metric_value': analysis.best_metric_value,
                'best_epoch': analysis.best_epoch,
                'metrics_source': collection.metrics_source,
                'found_files': collection.found_files,
                'missing_files': collection.missing_files,
                'summary_path': str(summary_path) if summary_path is not None else None,
                'summary_written': summary_path is not None,
                'observations': analysis.observations,
                'timing_summary': timing_summary,
                'task_artifacts': [task_artifact.to_dict() for task_artifact in collection.task_artifacts],
                'prediction_artifacts': [
                    prediction_artifact.to_dict()
                    for prediction_artifact in collection.prediction_artifacts
                ],
                'augmentation_artifacts': [
                    augmentation_artifact.to_dict()
                    for augmentation_artifact in collection.augmentation_artifacts
                ],
            },
        )

    def mark_failed(
        self,
        record: ExperimentRecord,
        workspace_dir: Path,
        error_type: str,
        message: str,
    ) -> ExperimentRunnerResult:
        updated_record = replace(
            self._require_record(experiment_id=record.experiment_id),
            status=ExperimentStatus.FAILED,
            finished_at=self._timestamp(),
            error_type=error_type,
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=record.experiment_id,
            status=ExperimentStatus.FAILED,
            workspace_dir=workspace_dir,
            message=message,
            metadata={'error_type': error_type},
        )

    def _require_record(self, experiment_id: str) -> ExperimentRecord:
        record = self.storage.get_experiment(experiment_id=experiment_id)
        if record is None:
            raise KeyError(f'unknown experiment_id: {experiment_id}')

        return record

    def _timestamp(self) -> str:
        return self.clock().isoformat()

    def _timing_summary(self, *, collection) -> dict[str, object]:
        stage_seconds: dict[str, float] = {}
        task_total_seconds = 0.0
        task_total_count = 0
        for task_artifact in collection.task_artifacts:
            if task_artifact.elapsed_seconds is not None:
                task_total_seconds += task_artifact.elapsed_seconds
                task_total_count += 1
            for row in task_artifact.timing_rows:
                stage = row.get('stage')
                elapsed_seconds = row.get('elapsed_seconds')
                if not isinstance(stage, str) or not stage:
                    continue
                if not isinstance(elapsed_seconds, float):
                    continue
                if stage == 'task_total':
                    continue
                stage_seconds[stage] = stage_seconds.get(stage, 0.0) + elapsed_seconds

        inference_stages = ('predict', 'preprocessing', 'tracking', 'segmentation', 'embedding')
        inference_pipeline_seconds = sum(stage_seconds.get(stage, 0.0) for stage in inference_stages)

        return {
            'task_count': len(collection.task_artifacts),
            'timed_task_count': task_total_count,
            'task_total_seconds': round(task_total_seconds, 6),
            'train_seconds': round(stage_seconds.get('train', 0.0), 6),
            'predict_seconds': round(stage_seconds.get('predict', 0.0), 6),
            'preprocessing_seconds': round(stage_seconds.get('preprocessing', 0.0), 6),
            'tracking_seconds': round(stage_seconds.get('tracking', 0.0), 6),
            'segmentation_seconds': round(stage_seconds.get('segmentation', 0.0), 6),
            'embedding_seconds': round(stage_seconds.get('embedding', 0.0), 6),
            'inference_pipeline_seconds': round(inference_pipeline_seconds, 6),
            'stage_seconds': {
                stage: round(seconds, 6)
                for stage, seconds in sorted(stage_seconds.items())
            },
        }
