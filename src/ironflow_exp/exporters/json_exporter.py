import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ironflow_exp.datasets import DatasetManifest, ObjectManifest
from ironflow_exp.domain import ArtifactRecord


class JsonExporter:
    def export_dataset_manifest(
        self,
        dataset_manifest: DatasetManifest,
        output_path: str | Path,
    ) -> Path:
        data = self._with_summary(value=dataset_manifest, summary=dataset_manifest.summary)

        return self.export_mapping(data=data, output_path=output_path)

    def export_object_manifest(
        self,
        object_manifest: ObjectManifest,
        output_path: str | Path,
    ) -> Path:
        data = self._with_summary(value=object_manifest, summary=object_manifest.summary)

        return self.export_mapping(data=data, output_path=output_path)

    def export_artifact_manifest(
        self,
        schema_version: str,
        run_id: str,
        artifacts: list[ArtifactRecord],
        output_path: str | Path,
    ) -> Path:
        data = {
            'schema_version': schema_version,
            'run_id': run_id,
            'artifacts': [
                asdict(artifact)
                for artifact in artifacts
            ],
            'summary': {
                'num_artifacts': len(artifacts),
            },
        }

        return self.export_mapping(data=data, output_path=output_path)

    def export_mapping(
        self,
        data: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode='w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        return path

    def _with_summary(
        self,
        value: object,
        summary: dict[str, object],
    ) -> dict[str, Any]:
        if not is_dataclass(value):
            raise TypeError('value must be a dataclass instance')

        data = asdict(value)
        data['summary'] = summary

        return data
