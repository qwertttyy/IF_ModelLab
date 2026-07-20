import hashlib
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ironflow_exp.domain import ArtifactRecord, MetricRecord
from ironflow_exp.exporters import CsvExporter, JsonExporter
from ironflow_exp.pipelines import ExperimentPipelineResult, PipelineStageResult


@dataclass(frozen=True, slots=True)
class ExportServiceResult:
    output_root: Path
    exported_paths: dict[str, Path]
    artifacts: list[ArtifactRecord]


class ExportService:
    def __init__(
        self,
        json_exporter: JsonExporter | None = None,
        csv_exporter: CsvExporter | None = None,
    ) -> None:
        self.json_exporter = json_exporter or JsonExporter()
        self.csv_exporter = csv_exporter or CsvExporter()

    def export_pipeline_result(
        self,
        pipeline_result: ExperimentPipelineResult,
        output_root: str | Path,
        schema_version: str,
        run_id: str,
        save_json: bool = True,
        save_csv: bool = True,
        metrics: list[MetricRecord] | None = None,
        predictions_by_task: dict[str, list[object]] | None = None,
    ) -> ExportServiceResult:
        output_root_path = Path(output_root)
        exported_paths: dict[str, Path] = {}
        export_artifacts: list[ArtifactRecord] = []
        metrics = metrics or []
        predictions_by_task = predictions_by_task or {}

        if save_json:
            export_artifacts.extend(
                self._export_json_outputs(
                    pipeline_result=pipeline_result,
                    output_root=output_root_path,
                    schema_version=schema_version,
                    run_id=run_id,
                    metrics=metrics,
                    predictions_by_task=predictions_by_task,
                    exported_paths=exported_paths,
                ),
            )

        if save_csv:
            export_artifacts.extend(
                self._export_csv_outputs(
                    pipeline_result=pipeline_result,
                    output_root=output_root_path,
                    metrics=metrics,
                    exported_paths=exported_paths,
                    start_index=len(export_artifacts),
                ),
            )

        all_artifacts = [*pipeline_result.artifacts, *export_artifacts]
        if save_json:
            artifact_manifest_path = output_root_path / 'manifests' / 'artifact_manifest.json'
            self.json_exporter.export_artifact_manifest(
                schema_version=schema_version,
                run_id=run_id,
                artifacts=all_artifacts,
                output_path=artifact_manifest_path,
            )
            exported_paths['artifact_manifest_json'] = artifact_manifest_path
            all_artifacts.append(
                self._make_export_artifact(
                    index=len(all_artifacts),
                    output_root=output_root_path,
                    path=artifact_manifest_path,
                    task='export',
                    role='manifest',
                    model_id=None,
                ),
            )

        return ExportServiceResult(
            output_root=output_root_path,
            exported_paths=exported_paths,
            artifacts=all_artifacts,
        )

    def _export_json_outputs(
        self,
        pipeline_result: ExperimentPipelineResult,
        output_root: Path,
        schema_version: str,
        run_id: str,
        metrics: list[MetricRecord],
        predictions_by_task: dict[str, list[object]],
        exported_paths: dict[str, Path],
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        dataset_manifest_path = output_root / 'manifests' / 'dataset_manifest.json'
        self.json_exporter.export_dataset_manifest(
            dataset_manifest=pipeline_result.dataset_manifest,
            output_path=dataset_manifest_path,
        )
        exported_paths['dataset_manifest_json'] = dataset_manifest_path
        artifacts.append(
            self._make_export_artifact(
                index=len(artifacts),
                output_root=output_root,
                path=dataset_manifest_path,
                task='preprocessing',
                role='manifest',
                model_id=None,
            ),
        )

        if pipeline_result.object_manifest is not None:
            object_manifest_path = output_root / 'manifests' / 'object_manifest.json'
            self.json_exporter.export_object_manifest(
                object_manifest=pipeline_result.object_manifest,
                output_path=object_manifest_path,
            )
            exported_paths['object_manifest_json'] = object_manifest_path
            artifacts.append(
                self._make_export_artifact(
                    index=len(artifacts),
                    output_root=output_root,
                    path=object_manifest_path,
                    task='preprocessing',
                    role='manifest',
                    model_id=None,
                ),
            )

        if metrics:
            metrics_path = output_root / 'metrics' / 'metrics.json'
            self.json_exporter.export_mapping(
                data={
                    'schema_version': schema_version,
                    'run_id': run_id,
                    'metrics': [asdict(metric) for metric in metrics],
                    'summary': {'num_metrics': len(metrics)},
                },
                output_path=metrics_path,
            )
            exported_paths['metrics_json'] = metrics_path
            artifacts.append(
                self._make_export_artifact(
                    index=len(artifacts),
                    output_root=output_root,
                    path=metrics_path,
                    task='evaluation',
                    role='metric',
                    model_id=None,
                ),
            )

        for task, predictions in sorted(predictions_by_task.items()):
            if not predictions:
                continue
            predictions_path = output_root / 'predictions' / f'{task}_predictions.json'
            self.json_exporter.export_mapping(
                data={
                    'schema_version': schema_version,
                    'run_id': run_id,
                    'task': task,
                    'model_id': self._prediction_model_id(predictions=predictions),
                    'predictions': [self._to_mapping(value=prediction) for prediction in predictions],
                },
                output_path=predictions_path,
            )
            exported_paths[f'{task}_predictions_json'] = predictions_path
            artifacts.append(
                self._make_export_artifact(
                    index=len(artifacts),
                    output_root=output_root,
                    path=predictions_path,
                    task=task,
                    role='prediction',
                    model_id=self._prediction_model_id(predictions=predictions),
                ),
            )

        return artifacts

    def _export_csv_outputs(
        self,
        pipeline_result: ExperimentPipelineResult,
        output_root: Path,
        metrics: list[MetricRecord],
        exported_paths: dict[str, Path],
        start_index: int,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []

        if metrics:
            metrics_path = output_root / 'metrics' / 'metrics.csv'
            self.csv_exporter.export_metric_records(metrics=metrics, output_path=metrics_path)
            exported_paths['metrics_csv'] = metrics_path
            artifacts.append(
                self._make_export_artifact(
                    index=start_index + len(artifacts),
                    output_root=output_root,
                    path=metrics_path,
                    task='evaluation',
                    role='metric',
                    model_id=None,
                ),
            )

        if pipeline_result.stages:
            stages_path = output_root / 'logs' / 'stage_results.csv'
            self.csv_exporter.export_stage_results(stages=pipeline_result.stages, output_path=stages_path)
            exported_paths['stage_results_csv'] = stages_path
            artifacts.append(
                self._make_export_artifact(
                    index=start_index + len(artifacts),
                    output_root=output_root,
                    path=stages_path,
                    task='export',
                    role='report',
                    model_id=None,
                ),
            )

        return artifacts

    def _to_mapping(self, value: object) -> dict[str, Any]:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return value

        raise TypeError('export value must be a dataclass instance or mapping')

    def _prediction_model_id(self, predictions: list[object]) -> str | None:
        for prediction in predictions:
            mapping = self._to_mapping(value=prediction)
            metadata = mapping.get('metadata', {})
            if isinstance(metadata, dict) and metadata.get('source_model_id') is not None:
                return str(metadata['source_model_id'])

        return None

    def _make_export_artifact(
        self,
        index: int,
        output_root: Path,
        path: Path,
        task: str,
        role: str,
        model_id: str | None,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=f'artifact_export_{index + 1:08d}',
            task=task,
            role=role,
            path=self._relative_path(output_root=output_root, path=path),
            format=path.suffix.lstrip('.'),
            model_id=model_id,
            sample_id=None,
            object_id=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            checksum_sha256=self._file_sha256(path=path),
            metadata={'source': 'export_service'},
        )

    def _relative_path(self, output_root: Path, path: Path) -> str:
        try:
            return path.relative_to(output_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _file_sha256(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with path.open(mode='rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                sha256.update(chunk)

        return sha256.hexdigest()
