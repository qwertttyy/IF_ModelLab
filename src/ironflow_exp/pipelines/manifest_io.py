from dataclasses import asdict
import json
from pathlib import Path

from ironflow_exp.datasets import DatasetManifest, ObjectManifest


class PipelineManifestWriter:
    def write_dataset_manifest(
        self,
        dataset_manifest: DatasetManifest,
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        data = asdict(dataset_manifest)
        data['summary'] = dataset_manifest.summary

        return self._write_mapping(data=data, output_path=path)

    def write_object_manifest(
        self,
        object_manifest: ObjectManifest,
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        data = asdict(object_manifest)
        data['summary'] = object_manifest.summary

        return self._write_mapping(data=data, output_path=path)

    def _write_mapping(
        self,
        data: dict[str, object],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode='w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        return output_path
