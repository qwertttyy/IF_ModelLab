from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ironflow_exp.core.enums import ArtifactRole, JobStatus, TaskType
from ironflow_exp.datasets import DatasetManifest, GtObjectManifestBuilder, ObjectManifest, YoloDataYamlBuilder
from ironflow_exp.domain import ArtifactRecord, DetectionPredictionRecord, ExperimentContext, MetricRecord
from ironflow_exp.models import ModelRegistry, create_default_model_registry
from ironflow_exp.models.base import PredictionRecord
from ironflow_exp.pipelines.manifest_io import PipelineManifestWriter
from ironflow_exp.pipelines.result import PipelineStageResult


if TYPE_CHECKING:
    from ironflow_exp.services.metrics_service import MetricsService


@dataclass(frozen=True, slots=True)
class DetectionPipelineResult:
    predictions: list[DetectionPredictionRecord] = field(default_factory=list)
    metrics: list[MetricRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    stages: list[PipelineStageResult] = field(default_factory=list)

    @property
    def stage_statuses(self) -> dict[str, str]:
        return {
            stage.name: stage.status
            for stage in self.stages
        }


class DetectionPipeline:
    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        yolo_data_yaml_builder: YoloDataYamlBuilder | None = None,
        object_manifest_builder: GtObjectManifestBuilder | None = None,
        metrics_service: 'MetricsService | None' = None,
        manifest_writer: PipelineManifestWriter | None = None,
    ) -> None:
        from ironflow_exp.services.metrics_service import MetricsService

        self.model_registry = model_registry or create_default_model_registry()
        self.yolo_data_yaml_builder = yolo_data_yaml_builder or YoloDataYamlBuilder()
        self.object_manifest_builder = object_manifest_builder or GtObjectManifestBuilder()
        self.metrics_service = metrics_service or MetricsService()
        self.manifest_writer = manifest_writer or PipelineManifestWriter()

    def run(
        self,
        context: ExperimentContext,
        dataset_manifest: DatasetManifest,
        object_manifest: ObjectManifest | None = None,
    ) -> DetectionPipelineResult:
        model_config = context.config.models.detection
        stages: list[PipelineStageResult] = []
        artifacts: list[ArtifactRecord] = []
        predictions: list[DetectionPredictionRecord] = []
        metrics: list[MetricRecord] = []

        if not model_config.enabled:
            stages.append(
                PipelineStageResult(
                    name='detection',
                    status=JobStatus.SKIPPED.value,
                    message='detection model is disabled',
                ),
            )
            return DetectionPipelineResult(stages=stages)

        if model_config.model_id is None:
            raise ValueError('detection model_id is required')

        adapter = self.model_registry.create_adapter(
            task=TaskType.DETECTION.value,
            model_id=model_config.model_id,
            model_config=model_config,
        )

        if model_config.train.enabled:
            yolo_result = self.yolo_data_yaml_builder.build(
                dataset_manifest=dataset_manifest,
                source_root=context.config.dataset.source_root,
                output_root=context.output_dir,
            )
            context.metadata.update(yolo_result.context_metadata)
            artifacts.extend(yolo_result.artifacts)
            stages.append(
                PipelineStageResult(
                    name='detection_data_yaml',
                    status=JobStatus.SUCCESS.value,
                    metadata={
                        'data_yaml_path': str(yolo_result.data_yaml_path),
                        'split_files': {
                            split: str(path)
                            for split, path in yolo_result.split_file_paths.items()
                        },
                    },
                ),
            )
            train_artifacts = adapter.train(context=context)
            artifacts.extend(train_artifacts)
            stages.append(
                PipelineStageResult(
                    name='detection_train',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_artifacts': len(train_artifacts)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='detection_train',
                    status=JobStatus.SKIPPED.value,
                    message='detection train is disabled',
                ),
            )

        if model_config.predict.enabled:
            artifacts.extend(
                self._ensure_dataset_manifest_path(
                    context=context,
                    dataset_manifest=dataset_manifest,
                    start_index=len(artifacts),
                ),
            )
            raw_predictions = adapter.predict(context=context)
            predictions = self._detection_predictions(raw_predictions=raw_predictions)
            stages.append(
                PipelineStageResult(
                    name='detection_predict',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_predictions': len(predictions)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='detection_predict',
                    status=JobStatus.SKIPPED.value,
                    message='detection predict is disabled',
                ),
            )

        if context.config.evaluation.detection.enabled and predictions:
            reference_manifest = object_manifest or self.object_manifest_builder.build(
                dataset_manifest=dataset_manifest,
                source_root=context.config.dataset.source_root,
                schema_version=context.config.schema_version,
            )
            detection_result = self.metrics_service.calculate_detection(
                run_id=context.run_id,
                model_id=model_config.model_id,
                predictions=predictions,
                references=reference_manifest.objects,
                metric_names=context.config.evaluation.detection.metrics,
            )
            predictions = detection_result.matched_predictions
            metrics = detection_result.metrics
            stages.append(
                PipelineStageResult(
                    name='detection_evaluate',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_metrics': len(metrics)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='detection_evaluate',
                    status=JobStatus.SKIPPED.value,
                    message='detection evaluation is disabled or no predictions exist',
                ),
            )

        return DetectionPipelineResult(
            predictions=predictions,
            metrics=metrics,
            artifacts=artifacts,
            stages=stages,
        )

    def _ensure_dataset_manifest_path(
        self,
        context: ExperimentContext,
        dataset_manifest: DatasetManifest,
        start_index: int,
    ) -> list[ArtifactRecord]:
        if context.dataset_manifest_path is not None:
            return []

        manifest_path = Path(context.output_dir) / 'manifests' / 'dataset_manifest.json'
        self.manifest_writer.write_dataset_manifest(dataset_manifest=dataset_manifest, output_path=manifest_path)
        context.dataset_manifest_path = str(manifest_path)

        return [
            ArtifactRecord(
                artifact_id=f'artifact_detection_manifest_{start_index + 1:08d}',
                task=TaskType.DETECTION.value,
                role=ArtifactRole.MANIFEST.value,
                path=manifest_path.relative_to(Path(context.output_dir)).as_posix(),
                format='json',
                model_id=context.config.models.detection.model_id,
                sample_id=None,
                object_id=None,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                metadata={'source': 'detection_pipeline'},
            ),
        ]

    def _detection_predictions(
        self,
        raw_predictions: list[PredictionRecord],
    ) -> list[DetectionPredictionRecord]:
        invalid_predictions = [
            prediction
            for prediction in raw_predictions
            if not isinstance(prediction, DetectionPredictionRecord)
        ]
        if invalid_predictions:
            invalid_types = sorted({type(prediction).__name__ for prediction in invalid_predictions})
            raise TypeError(f'detection adapter returned non-detection predictions: {invalid_types}')

        return list(raw_predictions)
