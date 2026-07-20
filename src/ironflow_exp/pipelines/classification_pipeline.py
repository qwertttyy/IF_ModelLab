from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ironflow_exp.core.enums import ArtifactRole, JobStatus, TaskType
from ironflow_exp.datasets import DatasetManifest, ObjectManifest
from ironflow_exp.domain import ArtifactRecord, ClassificationPredictionRecord, ExperimentContext, MetricRecord
from ironflow_exp.models import ModelRegistry, create_default_model_registry
from ironflow_exp.models.base import PredictionRecord
from ironflow_exp.pipelines.manifest_io import PipelineManifestWriter
from ironflow_exp.pipelines.result import PipelineStageResult


if TYPE_CHECKING:
    from ironflow_exp.services.metrics_service import MetricsService


@dataclass(frozen=True, slots=True)
class ClassificationPipelineResult:
    predictions: list[ClassificationPredictionRecord] = field(default_factory=list)
    metrics: list[MetricRecord] = field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    stages: list[PipelineStageResult] = field(default_factory=list)

    @property
    def stage_statuses(self) -> dict[str, str]:
        return {
            stage.name: stage.status
            for stage in self.stages
        }


class ClassificationPipeline:
    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        metrics_service: 'MetricsService | None' = None,
        manifest_writer: PipelineManifestWriter | None = None,
    ) -> None:
        from ironflow_exp.services.metrics_service import MetricsService

        self.model_registry = model_registry or create_default_model_registry()
        self.metrics_service = metrics_service or MetricsService()
        self.manifest_writer = manifest_writer or PipelineManifestWriter()

    def run(
        self,
        context: ExperimentContext,
        dataset_manifest: DatasetManifest,
        object_manifest: ObjectManifest | None = None,
    ) -> ClassificationPipelineResult:
        model_config = context.config.models.classification
        stages: list[PipelineStageResult] = []
        artifacts: list[ArtifactRecord] = []
        predictions: list[ClassificationPredictionRecord] = []
        metrics: list[MetricRecord] = []
        confusion_matrix: dict[str, dict[str, int]] = {}

        if not model_config.enabled:
            stages.append(
                PipelineStageResult(
                    name='classification',
                    status=JobStatus.SKIPPED.value,
                    message='classification model is disabled',
                ),
            )
            return ClassificationPipelineResult(stages=stages)

        if model_config.model_id is None:
            raise ValueError('classification model_id is required')

        artifacts.extend(
            self._ensure_input_manifest_paths(
                context=context,
                dataset_manifest=dataset_manifest,
                object_manifest=object_manifest,
                start_index=len(artifacts),
            ),
        )
        adapter = self.model_registry.create_adapter(
            task=TaskType.CLASSIFICATION.value,
            model_id=model_config.model_id,
            model_config=model_config,
        )

        if model_config.train.enabled:
            train_artifacts = adapter.train(context=context)
            artifacts.extend(train_artifacts)
            stages.append(
                PipelineStageResult(
                    name='classification_train',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_artifacts': len(train_artifacts)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='classification_train',
                    status=JobStatus.SKIPPED.value,
                    message='classification train is disabled',
                ),
            )

        if model_config.predict.enabled:
            raw_predictions = adapter.predict(context=context)
            predictions = self._classification_predictions(raw_predictions=raw_predictions)
            stages.append(
                PipelineStageResult(
                    name='classification_predict',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_predictions': len(predictions)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='classification_predict',
                    status=JobStatus.SKIPPED.value,
                    message='classification predict is disabled',
                ),
            )

        if context.config.evaluation.classification.enabled and predictions:
            classification_result = self.metrics_service.calculate_classification(
                run_id=context.run_id,
                model_id=model_config.model_id,
                predictions=predictions,
                metric_names=context.config.evaluation.classification.metrics,
            )
            metrics.extend(classification_result.metrics)
            confusion_matrix = classification_result.confusion_matrix
            stages.append(
                PipelineStageResult(
                    name='classification_evaluate',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_metrics': len(classification_result.metrics)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='classification_evaluate',
                    status=JobStatus.SKIPPED.value,
                    message='classification evaluation is disabled or no predictions exist',
                ),
            )

        if context.config.evaluation.latency.enabled and predictions:
            latency_metrics = self.metrics_service.calculate_classification_latency(
                run_id=context.run_id,
                model_id=model_config.model_id,
                predictions=predictions,
            )
            metrics.extend(latency_metrics)
            stages.append(
                PipelineStageResult(
                    name='classification_latency',
                    status=JobStatus.SUCCESS.value,
                    metadata={'num_metrics': len(latency_metrics)},
                ),
            )
        else:
            stages.append(
                PipelineStageResult(
                    name='classification_latency',
                    status=JobStatus.SKIPPED.value,
                    message='classification latency is disabled or no predictions exist',
                ),
            )

        return ClassificationPipelineResult(
            predictions=predictions,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            artifacts=artifacts,
            stages=stages,
        )

    def _ensure_input_manifest_paths(
        self,
        context: ExperimentContext,
        dataset_manifest: DatasetManifest,
        object_manifest: ObjectManifest | None,
        start_index: int,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []

        if context.dataset_manifest_path is None:
            manifest_path = Path(context.output_dir) / 'manifests' / 'dataset_manifest.json'
            self.manifest_writer.write_dataset_manifest(dataset_manifest=dataset_manifest, output_path=manifest_path)
            context.dataset_manifest_path = str(manifest_path)
            artifacts.append(
                self._manifest_artifact(
                    context=context,
                    index=start_index + len(artifacts),
                    path=manifest_path,
                    role=ArtifactRole.MANIFEST.value,
                ),
            )

        if object_manifest is not None and context.object_manifest_path is None:
            manifest_path = Path(context.output_dir) / 'manifests' / 'object_manifest.json'
            self.manifest_writer.write_object_manifest(object_manifest=object_manifest, output_path=manifest_path)
            context.object_manifest_path = str(manifest_path)
            artifacts.append(
                self._manifest_artifact(
                    context=context,
                    index=start_index + len(artifacts),
                    path=manifest_path,
                    role=ArtifactRole.MANIFEST.value,
                ),
            )

        return artifacts

    def _manifest_artifact(
        self,
        context: ExperimentContext,
        index: int,
        path: Path,
        role: str,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=f'artifact_classification_manifest_{index + 1:08d}',
            task=TaskType.CLASSIFICATION.value,
            role=role,
            path=path.relative_to(Path(context.output_dir)).as_posix(),
            format='json',
            model_id=context.config.models.classification.model_id,
            sample_id=None,
            object_id=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            metadata={'source': 'classification_pipeline'},
        )

    def _classification_predictions(
        self,
        raw_predictions: list[PredictionRecord],
    ) -> list[ClassificationPredictionRecord]:
        invalid_predictions = [
            prediction
            for prediction in raw_predictions
            if not isinstance(prediction, ClassificationPredictionRecord)
        ]
        if invalid_predictions:
            invalid_types = sorted({type(prediction).__name__ for prediction in invalid_predictions})
            raise TypeError(f'classification adapter returned non-classification predictions: {invalid_types}')

        return list(raw_predictions)
