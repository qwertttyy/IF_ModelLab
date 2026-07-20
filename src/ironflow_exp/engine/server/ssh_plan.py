import posixpath
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.output_policy import build_mock_output_args, effective_collect_patterns
from ironflow_exp.engine.server.code_package import CodePackageResult


AUTO_REMOTE_PYTHON_VALUES = {'auto', 'detect', '__ironflow_python_auto__'}


def is_auto_remote_python(value: str) -> bool:
    return value.strip().lower() in AUTO_REMOTE_PYTHON_VALUES


def remote_python_selector_prefix() -> str:
    return (
        'PYTHON_BIN="${IRONFLOW_REMOTE_PYTHON:-}"; '
        'if [ -z "$PYTHON_BIN" ] && [ -x /venv/main/bin/python ]; then '
        'PYTHON_BIN=/venv/main/bin/python; '
        'fi; '
        'if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then '
        'PYTHON_BIN=python3; '
        'fi; '
        'if [ -z "$PYTHON_BIN" ] && command -v python >/dev/null 2>&1; then '
        'PYTHON_BIN=python; '
        'fi; '
        'if [ -z "$PYTHON_BIN" ]; then '
        'echo "IronFlow remote Python auto-detect failed" >&2; '
        'exit 127; '
        'fi; '
    )


def remote_command_text_from_args(command_args: list[str]) -> str:
    if not command_args:
        return ''
    if not is_auto_remote_python(command_args[0]):
        return ' '.join(shlex.quote(part) for part in command_args)
    return remote_python_selector_prefix() + '"$PYTHON_BIN" ' + ' '.join(
        shlex.quote(part)
        for part in command_args[1:]
    )


@dataclass(frozen=True, slots=True)
class RemoteTransferItem:
    local_path: str
    remote_path: str
    kind: str = 'file'
    required: bool = True
    exclude_patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SshExecutionPlan:
    experiment_id: str
    server_name: str
    local_workspace_dir: str
    local_result_dir: str
    remote_workspace_dir: str
    remote_code_dir: str
    remote_result_dir: str
    remote_config_path: str
    bootstrap_commands: list[str] = field(default_factory=list)
    upload_items: list[RemoteTransferItem] = field(default_factory=list)
    download_items: list[RemoteTransferItem] = field(default_factory=list)
    remote_command_args: list[str] = field(default_factory=list)

    @property
    def remote_command_text(self) -> str:
        return remote_command_text_from_args(self.remote_command_args)

    def to_dict(self) -> dict[str, object]:
        return {
            'experiment_id': self.experiment_id,
            'server_name': self.server_name,
            'local_workspace_dir': self.local_workspace_dir,
            'local_result_dir': self.local_result_dir,
            'remote_workspace_dir': self.remote_workspace_dir,
            'remote_code_dir': self.remote_code_dir,
            'remote_result_dir': self.remote_result_dir,
            'remote_config_path': self.remote_config_path,
            'bootstrap_commands': self.bootstrap_commands,
            'upload_items': [
                {
                    'local_path': item.local_path,
                    'remote_path': item.remote_path,
                    'kind': item.kind,
                    'required': item.required,
                    'exclude_patterns': list(item.exclude_patterns),
                }
                for item in self.upload_items
            ],
            'download_items': [
                {
                    'local_path': item.local_path,
                    'remote_path': item.remote_path,
                    'kind': item.kind,
                    'required': item.required,
                    'exclude_patterns': list(item.exclude_patterns),
                }
                for item in self.download_items
            ],
            'remote_command_args': self.remote_command_args,
            'remote_command_text': self.remote_command_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> 'SshExecutionPlan':
        return cls(
            experiment_id=str(data['experiment_id']),
            server_name=str(data['server_name']),
            local_workspace_dir=str(data['local_workspace_dir']),
            local_result_dir=str(data['local_result_dir']),
            remote_workspace_dir=str(data['remote_workspace_dir']),
            remote_code_dir=str(data['remote_code_dir']),
            remote_result_dir=str(data['remote_result_dir']),
            remote_config_path=str(data['remote_config_path']),
            bootstrap_commands=[
                str(command)
                for command in data.get('bootstrap_commands', [])
            ],
            upload_items=[
                RemoteTransferItem(
                    local_path=str(item['local_path']),
                    remote_path=str(item['remote_path']),
                    kind=str(item.get('kind', 'file')),
                    required=bool(item.get('required', True)),
                    exclude_patterns=cls._exclude_patterns_from_item(item=item),
                )
                for item in cls._transfer_items_from_data(data=data, key='upload_items')
            ],
            download_items=[
                RemoteTransferItem(
                    local_path=str(item['local_path']),
                    remote_path=str(item['remote_path']),
                    kind=str(item.get('kind', 'file')),
                    required=bool(item.get('required', True)),
                    exclude_patterns=cls._exclude_patterns_from_item(item=item),
                )
                for item in cls._transfer_items_from_data(data=data, key='download_items')
            ],
            remote_command_args=[
                str(part)
                for part in data.get('remote_command_args', [])
            ],
        )

    @staticmethod
    def _transfer_items_from_data(data: dict[str, object], key: str) -> list[dict[str, object]]:
        raw_items = data.get(key, [])
        if not isinstance(raw_items, list):
            return []

        return [
            item
            for item in raw_items
            if isinstance(item, dict)
        ]

    @staticmethod
    def _exclude_patterns_from_item(item: dict[str, object]) -> tuple[str, ...]:
        raw_patterns = item.get('exclude_patterns', [])
        if not isinstance(raw_patterns, list):
            return ()

        return tuple(str(pattern) for pattern in raw_patterns)


class SshExecutionPlanBuilder:
    def build(
        self,
        config: EngineExperimentConfig,
        server: ServerRecord,
        experiment_id: str,
        local_workspace_dir: Path,
        local_result_dir: Path,
        code_package: CodePackageResult | None = None,
    ) -> SshExecutionPlan:
        if server.remote_workspace is None or not server.remote_workspace.strip():
            raise ValueError('remote_workspace is required to build SSH execution plan')

        remote_experiment_dir = self._remote_join(server.remote_workspace, 'experiments', experiment_id)
        remote_workspace_dir = self._remote_join(remote_experiment_dir, 'workspace')
        remote_code_dir = self._remote_join(remote_workspace_dir, 'code')
        remote_result_dir = self._remote_join(remote_experiment_dir, 'results')
        remote_config_path = self._remote_join(remote_workspace_dir, 'config.json')
        remote_entrypoint = self._remote_entrypoint(
            config=config,
            remote_code_dir=remote_code_dir,
            code_package=code_package,
        )
        upload_items = self._upload_items(
            config=config,
            remote_code_dir=remote_code_dir,
            remote_config_path=remote_config_path,
            local_workspace_dir=local_workspace_dir,
            code_package=code_package,
        )

        return SshExecutionPlan(
            experiment_id=experiment_id,
            server_name=server.name,
            local_workspace_dir=str(local_workspace_dir),
            local_result_dir=str(local_result_dir),
            remote_workspace_dir=remote_workspace_dir,
            remote_code_dir=remote_code_dir,
            remote_result_dir=remote_result_dir,
            remote_config_path=remote_config_path,
            bootstrap_commands=[
                f'mkdir -p {shlex.quote(remote_workspace_dir)}',
                f'mkdir -p {shlex.quote(remote_code_dir)}',
                f'mkdir -p {shlex.quote(remote_result_dir)}',
            ],
            upload_items=upload_items,
            download_items=self._download_items(
                config=config,
                remote_result_dir=remote_result_dir,
                local_result_dir=local_result_dir,
            ),
            remote_command_args=[
                config.runtime.remote_python_executable,
                remote_entrypoint,
                '--output-dir',
                remote_result_dir,
                *build_mock_output_args(config=config),
                *config.code.args,
                *config.train.args,
            ],
        )

    def _download_items(
        self,
        config: EngineExperimentConfig,
        remote_result_dir: str,
        local_result_dir: Path,
    ) -> list[RemoteTransferItem]:
        items: list[RemoteTransferItem] = []
        for pattern in effective_collect_patterns(output=config.output):
            kind = self._download_kind(pattern=pattern)
            normalized_pattern = self._normalize_download_pattern(pattern=pattern)
            items.append(
                RemoteTransferItem(
                    local_path=str(self._local_download_dir(
                        local_result_dir=local_result_dir,
                        pattern=normalized_pattern,
                        kind=kind,
                    )),
                    remote_path=self._remote_join(remote_result_dir, normalized_pattern),
                    kind=kind,
                    required=pattern == config.output.metrics_file,
                    exclude_patterns=self._download_exclude_patterns(config=config, kind=kind),
                ),
            )

        return items

    def _local_download_dir(self, local_result_dir: Path, pattern: str, kind: str) -> Path:
        normalized_pattern = PurePosixPath(pattern.replace('\\', '/'))
        if kind == 'directory':
            return local_result_dir.joinpath(*normalized_pattern.parts)

        concrete_parts: list[str] = []
        for part in normalized_pattern.parts:
            if any(character in part for character in ('*', '?', '[')):
                break
            concrete_parts.append(part)

        if concrete_parts == list(normalized_pattern.parts):
            parent = normalized_pattern.parent
            concrete_parts = [] if str(parent) == '.' else list(parent.parts)

        if not concrete_parts:
            return local_result_dir

        return local_result_dir.joinpath(*concrete_parts)

    def _download_kind(self, pattern: str) -> str:
        if pattern.replace('\\', '/').endswith('/'):
            return 'directory'
        if any(char in pattern for char in ('*', '?', '[')):
            return 'glob'

        return 'file'

    def _download_exclude_patterns(self, config: EngineExperimentConfig, kind: str) -> tuple[str, ...]:
        if kind != 'directory' or config.output.save_checkpoints:
            return ()

        return ('*.pt', '*.pth', '*.ckpt')

    def _normalize_download_pattern(self, pattern: str) -> str:
        return pattern.replace('\\', '/').rstrip('/')

    def _upload_items(
        self,
        config: EngineExperimentConfig,
        remote_code_dir: str,
        remote_config_path: str,
        local_workspace_dir: Path,
        code_package: CodePackageResult | None,
    ) -> list[RemoteTransferItem]:
        items = [
            RemoteTransferItem(
                local_path=str(local_workspace_dir / 'config.json'),
                remote_path=remote_config_path,
                kind='file',
            ),
        ]

        if code_package is not None:
            items.append(
                RemoteTransferItem(
                    local_path=code_package.package_dir,
                    remote_path=remote_code_dir,
                    kind='directory',
                ),
            )
            return items

        if config.code.working_dir is not None:
            items.append(
                RemoteTransferItem(
                    local_path=config.code.working_dir,
                    remote_path=remote_code_dir,
                    kind='directory',
                ),
            )
        else:
            items.append(
                RemoteTransferItem(
                    local_path=config.code.entrypoint,
                    remote_path=self._remote_join(remote_code_dir, Path(config.code.entrypoint).name),
                    kind='file',
                ),
            )

        return items

    def _remote_entrypoint(
        self,
        config: EngineExperimentConfig,
        remote_code_dir: str,
        code_package: CodePackageResult | None,
    ) -> str:
        if code_package is not None:
            return self._remote_join(remote_code_dir, *Path(code_package.entrypoint_relative_path).parts)

        entrypoint = Path(config.code.entrypoint)
        if config.code.working_dir is None:
            return self._remote_join(remote_code_dir, entrypoint.name)

        working_dir = Path(config.code.working_dir)
        try:
            relative_entrypoint = entrypoint.relative_to(working_dir)
        except ValueError:
            return self._remote_join(remote_code_dir, entrypoint.name)

        return self._remote_join(remote_code_dir, *relative_entrypoint.parts)

    def _remote_join(self, *parts: str) -> str:
        normalized_parts = [part.replace('\\', '/') for part in parts if part]
        if not normalized_parts:
            return ''

        first = normalized_parts[0]
        joined = posixpath.join(first, *normalized_parts[1:])

        return posixpath.normpath(joined)
