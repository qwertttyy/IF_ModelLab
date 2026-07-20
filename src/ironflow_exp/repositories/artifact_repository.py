import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ironflow_exp.domain import ArtifactRecord


class ArtifactRepository:
    def save_many(
        self,
        artifacts: list[ArtifactRecord],
        manifest_path: str | Path,
    ) -> Path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'artifacts': [
                asdict(artifact)
                for artifact in artifacts
            ],
        }

        with path.open(mode='w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        return path

    def load_many(self, manifest_path: str | Path) -> list[ArtifactRecord]:
        path = Path(manifest_path)

        with path.open(mode='r', encoding='utf-8') as file:
            data = json.load(file)

        rows = data.get('artifacts', [])
        if not isinstance(rows, list):
            raise ValueError('artifact manifest must contain artifacts list')

        return [
            self._record_from_dict(data=row)
            for row in rows
        ]

    def find_by_role(
        self,
        artifacts: list[ArtifactRecord],
        role: str,
    ) -> list[ArtifactRecord]:
        return [
            artifact
            for artifact in artifacts
            if artifact.role == role
        ]

    def find_by_task(
        self,
        artifacts: list[ArtifactRecord],
        task: str,
    ) -> list[ArtifactRecord]:
        return [
            artifact
            for artifact in artifacts
            if artifact.task == task
        ]

    def find_by_object_id(
        self,
        artifacts: list[ArtifactRecord],
        object_id: str,
    ) -> list[ArtifactRecord]:
        return [
            artifact
            for artifact in artifacts
            if artifact.object_id == object_id
        ]

    def _record_from_dict(self, data: dict[str, Any]) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=str(data['artifact_id']),
            task=str(data['task']),
            role=str(data['role']),
            path=str(data['path']),
            format=str(data['format']),
            model_id=self._optional_str(value=data.get('model_id')),
            sample_id=self._optional_str(value=data.get('sample_id')),
            object_id=self._optional_str(value=data.get('object_id')),
            created_at=str(data['created_at']),
            checksum_sha256=self._optional_str(value=data.get('checksum_sha256')),
            metadata=dict(data.get('metadata', {})),
        )

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)
