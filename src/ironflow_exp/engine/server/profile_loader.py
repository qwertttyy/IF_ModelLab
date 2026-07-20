import json
from pathlib import Path
from typing import Any

import yaml

from ironflow_exp.engine.domain import ServerRecord


class ServerProfileLoader:
    def load_file(self, path: str | Path) -> ServerRecord:
        profile_path = Path(path)
        data = self.load_dict(path=profile_path)
        record = self.from_dict(data=data)

        return self._resolve_file_relative_paths(record=record, base_dir=profile_path.parent)

    def load_dict(self, path: str | Path) -> dict[str, Any]:
        profile_path = Path(path)
        suffix = profile_path.suffix.lower()

        with profile_path.open(mode='r', encoding='utf-8') as file:
            if suffix == '.json':
                loaded = json.load(file)
            elif suffix in {'.yaml', '.yml'}:
                loaded = yaml.safe_load(file)
            else:
                raise ValueError(f'unsupported server profile extension: {suffix}')

        if not isinstance(loaded, dict):
            raise ValueError('server profile file must contain a mapping object')

        server_data = loaded.get('server', loaded)
        if not isinstance(server_data, dict):
            raise ValueError('server profile must be a mapping object')

        return dict(server_data)

    def from_dict(self, data: dict[str, Any]) -> ServerRecord:
        return ServerRecord(
            name=str(data.get('name', '')),
            server_type=str(data.get('server_type', data.get('type', 'local'))),
            host=self._optional_str(value=data.get('host')),
            port=self._optional_int(value=data.get('port')),
            username=self._optional_str(value=data.get('username')),
            key_path=self._optional_str(value=data.get('key_path')),
            remote_workspace=self._optional_str(value=data.get('remote_workspace')),
            last_checked_at=self._optional_str(value=data.get('last_checked_at')),
            last_status=self._optional_str(value=data.get('last_status')),
            gpu_name=self._optional_str(value=data.get('gpu_name')),
            total_vram=self._optional_str(value=data.get('total_vram')),
        )

    def _resolve_file_relative_paths(
        self,
        record: ServerRecord,
        base_dir: Path,
    ) -> ServerRecord:
        key_path = record.key_path
        if key_path is not None and key_path.strip():
            key_path = self._resolve_path(value=key_path, base_dir=base_dir)

        remote_workspace = record.remote_workspace
        if (
            record.server_type == 'local_ssh_simulator'
            and remote_workspace is not None
            and remote_workspace.strip()
        ):
            remote_workspace = self._resolve_path(value=remote_workspace, base_dir=base_dir)

        if key_path == record.key_path and remote_workspace == record.remote_workspace:
            return record

        return ServerRecord(
            name=record.name,
            server_type=record.server_type,
            host=record.host,
            port=record.port,
            username=record.username,
            key_path=key_path,
            remote_workspace=remote_workspace,
            last_checked_at=record.last_checked_at,
            last_status=record.last_status,
            gpu_name=record.gpu_name,
            total_vram=record.total_vram,
        )

    def _resolve_path(self, value: str, base_dir: Path) -> str:
        expanded = Path(value).expanduser()
        if expanded.is_absolute():
            return str(expanded)

        return str((base_dir / expanded).resolve())

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None

        return int(value)
