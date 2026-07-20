from dataclasses import asdict
from pathlib import Path
from typing import Any

# *************************
# External Library
#   - pandas : ver 2.2.3
# *************************
import pandas as pd

from ironflow_exp.domain import ArtifactRecord, MetricRecord
from ironflow_exp.pipelines import PipelineStageResult


class CsvExporter:
    def export_metric_records(
        self,
        metrics: list[MetricRecord],
        output_path: str | Path,
    ) -> Path:
        rows = [
            asdict(metric)
            for metric in metrics
        ]

        return self.export_rows(rows=rows, output_path=output_path)

    def export_artifact_records(
        self,
        artifacts: list[ArtifactRecord],
        output_path: str | Path,
    ) -> Path:
        rows = [
            asdict(artifact)
            for artifact in artifacts
        ]

        return self.export_rows(rows=rows, output_path=output_path)

    def export_stage_results(
        self,
        stages: list[PipelineStageResult],
        output_path: str | Path,
    ) -> Path:
        rows = [
            {
                'stage_name': stage.name,
                'status': stage.status,
                'message': stage.message,
                'metadata': stage.metadata,
            }
            for stage in stages
        ]

        return self.export_rows(rows=rows, output_path=output_path)

    def export_dataframe(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False, encoding='utf-8')

        return path

    def export_rows(
        self,
        rows: list[dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        dataframe = pd.DataFrame(rows)

        return self.export_dataframe(dataframe=dataframe, output_path=output_path)
