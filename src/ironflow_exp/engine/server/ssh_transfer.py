import json
import os
import posixpath
import shlex
import subprocess
import sys
import tarfile
import tempfile
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.ssh_plan import RemoteTransferItem, SshExecutionPlan
from ironflow_exp.engine.server.ssh_client import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_STRICT_HOST_KEY_CHECKING,
    build_open_ssh_options,
)
from ironflow_exp.engine.server.metadata_redaction import command_args_were_redacted, redact_command_args
from ironflow_exp.engine.server.subprocess_utils import stream_to_text


CHUNKED_SCP_UPLOAD_THRESHOLD_BYTES = int(os.environ.get('IRONFLOW_CHUNKED_SCP_THRESHOLD_BYTES', str(128 * 1024 * 1024)))
CHUNKED_SCP_UPLOAD_CHUNK_BYTES = int(os.environ.get('IRONFLOW_CHUNKED_SCP_CHUNK_BYTES', str(64 * 1024 * 1024)))
CHUNKED_SCP_UPLOAD_MAX_ATTEMPTS = int(os.environ.get('IRONFLOW_CHUNKED_SCP_MAX_ATTEMPTS', '3'))
REMOTE_PREPARE_TIMEOUT_SECONDS = int(os.environ.get('IRONFLOW_REMOTE_PREPARE_TIMEOUT_SECONDS', '120'))
REMOTE_ARCHIVE_OPERATION_TIMEOUT_SECONDS = int(os.environ.get('IRONFLOW_REMOTE_ARCHIVE_OPERATION_TIMEOUT_SECONDS', '900'))
SCP_DOWNLOAD_MAX_ATTEMPTS = int(os.environ.get('IRONFLOW_SCP_DOWNLOAD_MAX_ATTEMPTS', '3'))
SCP_DOWNLOAD_RETRY_DELAY_SECONDS = float(os.environ.get('IRONFLOW_SCP_DOWNLOAD_RETRY_DELAY_SECONDS', '3'))


@dataclass(frozen=True, slots=True)
class SshTransferResult:
    operation: str
    local_path: str
    remote_path: str
    kind: str
    exit_code: int
    required: bool = True
    stdout: str = ''
    stderr: str = ''
    command_args: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        redacted_command_args = redact_command_args(args=self.command_args)
        return {
            'operation': self.operation,
            'local_path': self.local_path,
            'remote_path': self.remote_path,
            'kind': self.kind,
            'exit_code': self.exit_code,
            'required': self.required,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'metadata': self.metadata,
            'command_args': redacted_command_args,
            'command_args_redacted': command_args_were_redacted(
                original=self.command_args,
                redacted=redacted_command_args,
            ),
        }


@dataclass(frozen=True, slots=True)
class _UploadCommandOutcome:
    exit_code: int
    stdout: str = ''
    stderr: str = ''
    command_args: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class BaseSshTransferClient(ABC):
    @abstractmethod
    def upload_item(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        """Upload one planned transfer item to a remote server."""

    def download_item(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        """Download one planned transfer item from a remote server."""
        raise NotImplementedError('download_item is not implemented')


class OpenScpTransferClient(BaseSshTransferClient):
    def __init__(
        self,
        scp_executable: str = 'scp',
        ssh_executable: str = 'ssh',
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        strict_host_key_checking: str = DEFAULT_STRICT_HOST_KEY_CHECKING,
    ) -> None:
        self.scp_executable = scp_executable
        self.ssh_executable = ssh_executable
        self.connect_timeout_seconds = connect_timeout_seconds
        self.strict_host_key_checking = strict_host_key_checking

    def upload_item(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        self._validate_upload_target(server=server, item=item)
        if item.kind == 'directory':
            return self._upload_directory_archive(
                server=server,
                item=item,
                timeout_seconds=timeout_seconds,
            )

        local_file = Path(item.local_path)
        if not local_file.is_file():
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=1,
                required=item.required,
                stderr=f'upload file does not exist: {local_file}',
                command_args=[],
            )
        local_size_bytes = local_file.stat().st_size
        prepare_args = self._build_ssh_args(
            server=server,
            command=f'mkdir -p {shlex.quote(posixpath.dirname(item.remote_path.rstrip("/")) or "/")}',
        )
        try:
            prepare_completed = subprocess.run(
                prepare_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_prepare_timeout_seconds(timeout_seconds=timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh file upload parent prepare timed out',
                command_args=prepare_args,
            )
        if prepare_completed.returncode != 0:
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=prepare_completed.returncode,
                required=item.required,
                stdout=prepare_completed.stdout,
                stderr=prepare_completed.stderr,
                command_args=prepare_args,
            )

        cache_check_args: list[str] = []
        if self._is_shared_asset_file_upload(server=server, item=item):
            cache_check_args = self._build_ssh_args(
                server=server,
                command=self._build_file_cache_check_command(
                    remote_path=item.remote_path,
                    size_bytes=local_size_bytes,
                ),
            )
            try:
                cache_check_completed = subprocess.run(
                    cache_check_args,
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=self._remote_prepare_timeout_seconds(timeout_seconds=timeout_seconds),
                )
            except subprocess.TimeoutExpired as error:
                return self._timeout_result(
                    operation='upload',
                    item=item,
                    stdout=stream_to_text(error.stdout),
                    stderr=stream_to_text(error.stderr) or 'ssh file upload cache check timed out',
                    command_args=prepare_args + [';'] + cache_check_args,
                )
            if cache_check_completed.returncode == 0:
                return SshTransferResult(
                    operation='upload',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=0,
                    required=item.required,
                    stdout='\n'.join(part for part in [
                        prepare_completed.stdout,
                        cache_check_completed.stdout,
                    ] if part),
                    stderr='\n'.join(part for part in [
                        prepare_completed.stderr,
                        cache_check_completed.stderr,
                    ] if part),
                    command_args=prepare_args + [';'] + cache_check_args,
                    metadata={
                        'upload_cache_enabled': True,
                        'upload_cache_hit': True,
                        'local_size_bytes': local_size_bytes,
                    },
                )

        command_args = self._build_upload_args(server=server, item=item)
        try:
            completed = subprocess.run(
                command_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stderr = stream_to_text(error.stderr) or 'scp upload timed out'
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=124,
                required=item.required,
                stdout=stream_to_text(error.stdout),
                stderr=stderr,
                command_args=prepare_args + ([';'] + cache_check_args if cache_check_args else []) + [';'] + command_args,
            )

        return SshTransferResult(
            operation='upload',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=completed.returncode,
            required=item.required,
            stdout='\n'.join(part for part in [prepare_completed.stdout, completed.stdout] if part),
            stderr='\n'.join(part for part in [prepare_completed.stderr, completed.stderr] if part),
            command_args=prepare_args + ([';'] + cache_check_args if cache_check_args else []) + [';'] + command_args,
            metadata={
                'upload_cache_enabled': bool(cache_check_args),
                'upload_cache_hit': False,
                'local_size_bytes': local_size_bytes,
            } if cache_check_args else {},
        )

    def _upload_directory_archive(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        self._validate_upload_target(server=server, item=item)
        source_dir = Path(item.local_path)
        if not source_dir.is_dir():
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=1,
                required=item.required,
                stderr=f'upload directory does not exist: {source_dir}',
                command_args=[],
            )

        remote_archive_path = self._remote_archive_path(remote_path=item.remote_path)
        local_file_count, local_size_bytes = self._directory_stats(source_dir=source_dir)
        timings: dict[str, float] = {}
        cache_metadata = {
            'upload_cache_enabled': True,
            'upload_cache_hit': False,
            'local_file_count': local_file_count,
            'local_size_bytes': local_size_bytes,
            'timings_seconds': timings,
        }
        cache_key_started = time.perf_counter()
        cache_key = self._directory_cache_key(source_dir=source_dir)
        timings['local_hash'] = round(time.perf_counter() - cache_key_started, 3)
        remote_cache_path = self._remote_directory_cache_path(server=server, cache_key=cache_key)
        if remote_cache_path is None:
            return self._upload_directory_archive_without_cache(
                server=server,
                item=item,
                source_dir=source_dir,
                remote_archive_path=remote_archive_path,
                timeout_seconds=timeout_seconds,
            )
        cache_metadata.update({
            'upload_cache_key': cache_key,
            'upload_cache_path': remote_cache_path,
        })

        cache_check_args = self._build_ssh_args(
            server=server,
            command=self._build_cache_check_command(remote_cache_path=remote_cache_path, cache_key=cache_key),
        )
        try:
            cache_check_started = time.perf_counter()
            cache_check_completed = subprocess.run(
                cache_check_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_prepare_timeout_seconds(timeout_seconds=timeout_seconds),
            )
            timings['cache_check'] = round(time.perf_counter() - cache_check_started, 3)
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh upload cache check timed out',
                command_args=cache_check_args,
            )

        if cache_check_completed.returncode == 0:
            materialize_args = self._build_ssh_args(
                server=server,
                command=self._build_materialize_cached_directory_command(
                    server=server,
                    remote_cache_path=remote_cache_path,
                    remote_target_path=item.remote_path,
                    cache_key=cache_key,
                ),
            )
            try:
                materialize_started = time.perf_counter()
                materialize_completed = subprocess.run(
                    materialize_args,
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=self._remote_archive_operation_timeout_seconds(timeout_seconds=timeout_seconds),
                )
                timings['cache_materialize'] = round(time.perf_counter() - materialize_started, 3)
            except subprocess.TimeoutExpired as error:
                return self._timeout_result(
                    operation='upload',
                    item=item,
                    stdout=stream_to_text(error.stdout),
                    stderr=stream_to_text(error.stderr) or 'ssh upload cache materialize timed out',
                    command_args=cache_check_args + [';'] + materialize_args,
                )

            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=materialize_completed.returncode,
                required=item.required,
                stdout='\n'.join(part for part in [
                    cache_check_completed.stdout,
                    materialize_completed.stdout,
                ] if part),
                stderr='\n'.join(part for part in [
                    cache_check_completed.stderr,
                    materialize_completed.stderr,
                ] if part),
                command_args=cache_check_args + [';'] + materialize_args,
                metadata={
                    **cache_metadata,
                    'upload_cache_hit': True,
                },
            )

        remote_archive_path = self._remote_cache_archive_path(remote_cache_path=remote_cache_path)
        remote_parent = posixpath.dirname(remote_archive_path.rstrip('/')) or '/'
        prepare_args = self._build_ssh_args(
            server=server,
            command=f'mkdir -p {shlex.quote(remote_parent)}',
        )

        try:
            prepare_started = time.perf_counter()
            prepare_completed = subprocess.run(
                prepare_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_prepare_timeout_seconds(timeout_seconds=timeout_seconds),
            )
            timings['cache_prepare'] = round(time.perf_counter() - prepare_started, 3)
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh upload cache prepare timed out',
                command_args=cache_check_args + [';'] + prepare_args,
            )
        if prepare_completed.returncode != 0:
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=prepare_completed.returncode,
                required=item.required,
                stdout=prepare_completed.stdout,
                stderr=prepare_completed.stderr,
                command_args=cache_check_args + [';'] + prepare_args,
                metadata={
                    **cache_metadata,
                },
            )

        with tempfile.TemporaryDirectory(prefix='ironflow_upload_') as tmp_dir:
            archive_path = Path(tmp_dir) / 'payload.tar.gz'
            try:
                archive_started = time.perf_counter()
                self._create_directory_archive(
                    source_dir=source_dir,
                    archive_path=archive_path,
                )
                timings['archive_create'] = round(time.perf_counter() - archive_started, 3)
            except (OSError, tarfile.TarError) as error:
                return SshTransferResult(
                    operation='upload',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=1,
                    required=item.required,
                    stderr=str(error),
                    command_args=[],
                    metadata={
                        **cache_metadata,
                    },
                )
            archive_item = RemoteTransferItem(
                local_path=str(archive_path),
                remote_path=remote_archive_path,
                kind='file',
                required=item.required,
            )
            upload_started = time.perf_counter()
            upload_completed = self._upload_archive_file(
                server=server,
                item=archive_item,
                timeout_seconds=timeout_seconds,
            )
            timings['scp_upload'] = round(time.perf_counter() - upload_started, 3)
            if upload_completed.exit_code != 0:
                return SshTransferResult(
                    operation='upload',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=upload_completed.exit_code,
                    required=item.required,
                    stdout=upload_completed.stdout,
                    stderr=upload_completed.stderr,
                    command_args=cache_check_args + [';'] + prepare_args + [';'] + upload_completed.command_args,
                    metadata={
                        **cache_metadata,
                        **upload_completed.metadata,
                    },
                )

        extract_command = self._build_extract_cache_archive_command(
            remote_archive_path=remote_archive_path,
            remote_cache_path=remote_cache_path,
            cache_key=cache_key,
        )
        extract_args = self._build_ssh_args(server=server, command=extract_command)
        try:
            extract_started = time.perf_counter()
            extract_completed = subprocess.run(
                extract_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_archive_operation_timeout_seconds(timeout_seconds=timeout_seconds),
            )
            timings['cache_extract'] = round(time.perf_counter() - extract_started, 3)
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh upload cache extract timed out',
                command_args=cache_check_args + [';'] + prepare_args + [';'] + upload_completed.command_args + [';'] + extract_args,
            )
        if extract_completed.returncode != 0:
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=extract_completed.returncode,
                required=item.required,
                stdout=extract_completed.stdout,
                stderr=extract_completed.stderr,
                command_args=cache_check_args + [';'] + prepare_args + [';'] + upload_completed.command_args + [';'] + extract_args,
                metadata={
                    **cache_metadata,
                    **upload_completed.metadata,
                },
            )

        materialize_args = self._build_ssh_args(
            server=server,
            command=self._build_materialize_cached_directory_command(
                server=server,
                remote_cache_path=remote_cache_path,
                remote_target_path=item.remote_path,
                cache_key=cache_key,
            ),
        )
        try:
            materialize_started = time.perf_counter()
            materialize_completed = subprocess.run(
                materialize_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_archive_operation_timeout_seconds(timeout_seconds=timeout_seconds),
            )
            timings['cache_materialize'] = round(time.perf_counter() - materialize_started, 3)
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh upload cache materialize timed out',
                command_args=cache_check_args + [';'] + prepare_args + [';'] + upload_completed.command_args + [';'] + extract_args + [';'] + materialize_args,
            )

        return SshTransferResult(
            operation='upload',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=materialize_completed.returncode,
            required=item.required,
            stdout='\n'.join(part for part in [
                cache_check_completed.stdout,
                prepare_completed.stdout,
                upload_completed.stdout,
                extract_completed.stdout,
                materialize_completed.stdout,
            ] if part),
            stderr='\n'.join(part for part in [
                cache_check_completed.stderr,
                prepare_completed.stderr,
                upload_completed.stderr,
                extract_completed.stderr,
                materialize_completed.stderr,
            ] if part),
            command_args=cache_check_args + [';'] + prepare_args + [';'] + upload_completed.command_args + [';'] + extract_args + [';'] + materialize_args,
            metadata={
                **cache_metadata,
                **upload_completed.metadata,
            },
        )

    def _upload_directory_archive_without_cache(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        source_dir: Path,
        remote_archive_path: str,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        remote_parent = posixpath.dirname(item.remote_path.rstrip('/')) or '/'
        prepare_args = self._build_ssh_args(
            server=server,
            command=f'mkdir -p {shlex.quote(remote_parent)}',
        )

        try:
            prepare_completed = subprocess.run(
                prepare_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_prepare_timeout_seconds(timeout_seconds=timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh directory prepare timed out',
                command_args=prepare_args,
            )
        if prepare_completed.returncode != 0:
            return SshTransferResult(
                operation='upload',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=prepare_completed.returncode,
                required=item.required,
                stdout=prepare_completed.stdout,
                stderr=prepare_completed.stderr,
                command_args=prepare_args,
            )

        with tempfile.TemporaryDirectory(prefix='ironflow_upload_') as tmp_dir:
            archive_path = Path(tmp_dir) / 'payload.tar.gz'
            try:
                self._create_directory_archive(
                    source_dir=source_dir,
                    archive_path=archive_path,
                )
            except (OSError, tarfile.TarError) as error:
                return SshTransferResult(
                    operation='upload',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=1,
                    required=item.required,
                    stderr=str(error),
                    command_args=[],
                )
            archive_item = RemoteTransferItem(
                local_path=str(archive_path),
                remote_path=remote_archive_path,
                kind='file',
                required=item.required,
            )
            upload_completed = self._upload_archive_file(
                server=server,
                item=archive_item,
                timeout_seconds=timeout_seconds,
            )
            if upload_completed.exit_code != 0:
                return SshTransferResult(
                    operation='upload',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=upload_completed.exit_code,
                    required=item.required,
                    stdout=upload_completed.stdout,
                    stderr=upload_completed.stderr,
                    command_args=upload_completed.command_args,
                    metadata=upload_completed.metadata,
                )

        extract_command = self._build_extract_archive_command(
            server=server,
            remote_archive_path=remote_archive_path,
            remote_target_path=item.remote_path,
        )
        extract_args = self._build_ssh_args(server=server, command=extract_command)
        try:
            extract_completed = subprocess.run(
                extract_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._remote_archive_operation_timeout_seconds(timeout_seconds=timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='upload',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh archive extract timed out',
                command_args=upload_completed.command_args + [';'] + extract_args,
            )

        return SshTransferResult(
            operation='upload',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=extract_completed.returncode,
            required=item.required,
            stdout='\n'.join(part for part in [upload_completed.stdout, extract_completed.stdout] if part),
            stderr='\n'.join(part for part in [upload_completed.stderr, extract_completed.stderr] if part),
            command_args=prepare_args + [';'] + upload_completed.command_args + [';'] + extract_args,
            metadata={
                'upload_cache_enabled': False,
                **upload_completed.metadata,
            },
        )

    def _upload_archive_file(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None,
    ) -> _UploadCommandOutcome:
        archive_path = Path(item.local_path)
        archive_size = archive_path.stat().st_size
        if archive_size < CHUNKED_SCP_UPLOAD_THRESHOLD_BYTES:
            command_args = self._build_upload_args(server=server, item=item)
            return self._run_command_with_retries(
                command_args=command_args,
                timeout_seconds=timeout_seconds,
                retry_label='scp archive upload',
                metadata={
                    'archive_size_bytes': archive_size,
                    'upload_mode': 'single_scp',
                },
            )

        return self._upload_archive_file_in_chunks(
            server=server,
            item=item,
            archive_path=archive_path,
            archive_size=archive_size,
            timeout_seconds=timeout_seconds,
        )

    def _upload_archive_file_in_chunks(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        archive_path: Path,
        archive_size: int,
        timeout_seconds: int | None,
    ) -> _UploadCommandOutcome:
        chunk_size = max(1, CHUNKED_SCP_UPLOAD_CHUNK_BYTES)
        remote_parts_dir = f'{item.remote_path}.parts'
        prepare_args = self._build_ssh_args(
            server=server,
            command=f'rm -rf {shlex.quote(remote_parts_dir)}; mkdir -p {shlex.quote(remote_parts_dir)}',
        )
        prepare_outcome = self._run_command_with_retries(
            command_args=prepare_args,
            timeout_seconds=self._remote_prepare_timeout_seconds(timeout_seconds=timeout_seconds),
            retry_label='ssh chunk prepare',
        )
        command_args = list(prepare_outcome.command_args)
        stdout_parts = [prepare_outcome.stdout] if prepare_outcome.stdout else []
        stderr_parts = [prepare_outcome.stderr] if prepare_outcome.stderr else []
        if prepare_outcome.exit_code != 0:
            return _UploadCommandOutcome(
                exit_code=prepare_outcome.exit_code,
                stdout='\n'.join(stdout_parts),
                stderr='\n'.join(stderr_parts),
                command_args=command_args,
                metadata=prepare_outcome.metadata,
            )

        chunk_count = 0
        with archive_path.open(mode='rb') as source:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                chunk_path = archive_path.parent / f'{archive_path.name}.part-{chunk_count:06d}'
                chunk_path.write_bytes(chunk)
                try:
                    chunk_item = RemoteTransferItem(
                        local_path=str(chunk_path),
                        remote_path=f'{remote_parts_dir}/part-{chunk_count:06d}',
                        kind='file',
                        required=item.required,
                    )
                    upload_args = self._build_upload_args(server=server, item=chunk_item)
                    chunk_outcome = self._run_command_with_retries(
                        command_args=upload_args,
                        timeout_seconds=timeout_seconds,
                        retry_label=f'scp archive chunk {chunk_count}',
                    )
                finally:
                    try:
                        chunk_path.unlink()
                    except FileNotFoundError:
                        pass
                command_args.extend([';'])
                command_args.extend(chunk_outcome.command_args)
                if chunk_outcome.stdout:
                    stdout_parts.append(chunk_outcome.stdout)
                if chunk_outcome.stderr:
                    stderr_parts.append(chunk_outcome.stderr)
                if chunk_outcome.exit_code != 0:
                    return _UploadCommandOutcome(
                        exit_code=chunk_outcome.exit_code,
                        stdout='\n'.join(stdout_parts),
                        stderr='\n'.join(stderr_parts),
                        command_args=command_args,
                        metadata={
                            'archive_size_bytes': archive_size,
                            'upload_mode': 'chunked_scp',
                            'chunk_size_bytes': chunk_size,
                            'chunk_count': chunk_count + 1,
                            'failed_chunk_index': chunk_count,
                        },
                    )
                chunk_count += 1

        assemble_command = (
            f'cat {shlex.quote(remote_parts_dir)}/part-* > {shlex.quote(item.remote_path)}; '
            'status=$?; '
            f'if [ $status -eq 0 ]; then rm -rf {shlex.quote(remote_parts_dir)}; fi; '
            'exit $status'
        )
        assemble_args = self._build_ssh_args(server=server, command=assemble_command)
        assemble_outcome = self._run_command_with_retries(
            command_args=assemble_args,
            timeout_seconds=timeout_seconds,
            retry_label='ssh chunk assemble',
        )
        command_args.extend([';'])
        command_args.extend(assemble_outcome.command_args)
        if assemble_outcome.stdout:
            stdout_parts.append(assemble_outcome.stdout)
        if assemble_outcome.stderr:
            stderr_parts.append(assemble_outcome.stderr)

        return _UploadCommandOutcome(
            exit_code=assemble_outcome.exit_code,
            stdout='\n'.join(stdout_parts),
            stderr='\n'.join(stderr_parts),
            command_args=command_args,
            metadata={
                'archive_size_bytes': archive_size,
                'upload_mode': 'chunked_scp',
                'chunk_size_bytes': chunk_size,
                'chunk_count': chunk_count,
            },
        )

    def _run_command_with_retries(
        self,
        command_args: list[str],
        timeout_seconds: int | None,
        retry_label: str,
        metadata: dict[str, object] | None = None,
    ) -> _UploadCommandOutcome:
        attempts = max(1, CHUNKED_SCP_UPLOAD_MAX_ATTEMPTS)
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        all_command_args: list[str] = []
        for attempt_index in range(attempts):
            if all_command_args:
                all_command_args.extend([';'])
            all_command_args.extend(command_args)
            try:
                completed = subprocess.run(
                    command_args,
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                stdout = stream_to_text(error.stdout)
                stderr = stream_to_text(error.stderr) or f'{retry_label} timed out'
                stdout_parts.append(stdout)
                stderr_parts.append(stderr)
                return _UploadCommandOutcome(
                    exit_code=124,
                    stdout='\n'.join(part for part in stdout_parts if part),
                    stderr='\n'.join(part for part in stderr_parts if part),
                    command_args=all_command_args,
                    metadata={
                        **(metadata or {}),
                        'upload_attempts': attempt_index + 1,
                    },
                )
            if completed.stdout:
                stdout_parts.append(completed.stdout)
            if completed.stderr:
                stderr_parts.append(completed.stderr)
            if completed.returncode == 0:
                return _UploadCommandOutcome(
                    exit_code=0,
                    stdout='\n'.join(stdout_parts),
                    stderr='\n'.join(stderr_parts),
                    command_args=all_command_args,
                    metadata={
                        **(metadata or {}),
                        'upload_attempts': attempt_index + 1,
                    },
                )
            if not self._is_retryable_transfer_failure(stderr=completed.stderr, exit_code=completed.returncode):
                return _UploadCommandOutcome(
                    exit_code=completed.returncode,
                    stdout='\n'.join(stdout_parts),
                    stderr='\n'.join(stderr_parts),
                    command_args=all_command_args,
                    metadata={
                        **(metadata or {}),
                        'upload_attempts': attempt_index + 1,
                    },
                )

        return _UploadCommandOutcome(
            exit_code=completed.returncode,
            stdout='\n'.join(stdout_parts),
            stderr='\n'.join(stderr_parts),
            command_args=all_command_args,
            metadata={
                **(metadata or {}),
                'upload_attempts': attempts,
            },
        )

    def _is_retryable_transfer_failure(self, *, stderr: str, exit_code: int) -> bool:
        if exit_code == 255:
            return True
        normalized = stderr.lower()
        return any(
            text in normalized
            for text in [
                'connection reset',
                'broken pipe',
                'connection timed out',
                'connection closed',
                'lost connection',
            ]
        )

    def _is_retryable_download_failure(self, *, stderr: str, exit_code: int) -> bool:
        if self._is_retryable_transfer_failure(stderr=stderr, exit_code=exit_code):
            return True
        normalized = stderr.lower()
        return any(
            text in normalized
            for text in [
                'no such file or directory',
                'not found',
            ]
        )

    def _remote_prepare_timeout_seconds(self, *, timeout_seconds: int | None) -> int:
        if timeout_seconds is None:
            return REMOTE_PREPARE_TIMEOUT_SECONDS
        return max(1, min(timeout_seconds, REMOTE_PREPARE_TIMEOUT_SECONDS))

    def _remote_archive_operation_timeout_seconds(self, *, timeout_seconds: int | None) -> int:
        if timeout_seconds is None:
            return REMOTE_ARCHIVE_OPERATION_TIMEOUT_SECONDS
        return max(1, min(timeout_seconds, REMOTE_ARCHIVE_OPERATION_TIMEOUT_SECONDS))

    def _build_upload_args(self, server: ServerRecord, item: RemoteTransferItem) -> list[str]:
        self._validate_upload_target(server=server, item=item)

        args = self._build_scp_base_args(server=server)
        if item.kind == 'directory':
            args.append('-r')

        args.extend([
            self._local_source_path(item=item),
            f'{server.username}@{server.host}:{item.remote_path}',
        ])

        return args

    def _validate_upload_target(self, server: ServerRecord, item: RemoteTransferItem) -> None:
        if server.host is None or not server.host.strip():
            raise ValueError('host is required for OpenScpTransferClient')
        if server.username is None or not server.username.strip():
            raise ValueError('username is required for OpenScpTransferClient')
        if item.kind not in {'file', 'directory'}:
            raise ValueError(f'unsupported upload item kind: {item.kind}')

    def _build_scp_base_args(self, server: ServerRecord) -> list[str]:
        args = [self.scp_executable]
        args.extend(
            build_open_ssh_options(
                connect_timeout_seconds=self.connect_timeout_seconds,
                strict_host_key_checking=self.strict_host_key_checking,
            ),
        )
        if server.port is not None:
            args.extend(['-P', str(server.port)])
        if server.key_path is not None and server.key_path.strip():
            args.extend(['-i', server.key_path])

        return args

    def _build_ssh_args(self, server: ServerRecord, command: str) -> list[str]:
        if server.host is None or not server.host.strip():
            raise ValueError('host is required for OpenScpTransferClient')
        if server.username is None or not server.username.strip():
            raise ValueError('username is required for OpenScpTransferClient')

        args = [self.ssh_executable, '-n']
        args.extend(
            build_open_ssh_options(
                connect_timeout_seconds=self.connect_timeout_seconds,
                strict_host_key_checking=self.strict_host_key_checking,
            ),
        )
        if server.port is not None:
            args.extend(['-p', str(server.port)])
        if server.key_path is not None and server.key_path.strip():
            args.extend(['-i', server.key_path])

        args.extend([f'{server.username}@{server.host}', command])
        return args

    def _local_source_path(self, item: RemoteTransferItem) -> str:
        if item.kind == 'directory':
            return f'{item.local_path.rstrip("/\\\\")}/.'

        return item.local_path

    def _remote_archive_path(self, remote_path: str) -> str:
        return f'{remote_path.rstrip("/")}.upload.tar.gz'

    def _remote_cache_archive_path(self, remote_cache_path: str) -> str:
        return f'{remote_cache_path.rstrip("/")}.upload.tar.gz'

    def _remote_download_archive_path(self, remote_path: str) -> str:
        return f'{remote_path.rstrip("/")}.download.tar.gz'

    def _create_directory_archive(self, source_dir: Path, archive_path: Path) -> None:
        if not source_dir.is_dir():
            raise FileNotFoundError(f'upload directory does not exist: {source_dir}')

        source_root = Path(self._filesystem_path(source_dir))
        with tarfile.open(self._filesystem_path(archive_path), 'w:gz') as archive:
            for root, dirnames, filenames in os.walk(source_root):
                for name in [*dirnames, *filenames]:
                    source_path = Path(root) / name
                    arcname = source_path.relative_to(source_root).as_posix()
                    archive.add(
                        str(source_path),
                        arcname=arcname,
                        recursive=False,
                    )

    def _directory_cache_key(self, source_dir: Path) -> str:
        digest = hashlib.sha256()
        source_root = Path(self._filesystem_path(source_dir))
        for root, dirnames, filenames in os.walk(source_root):
            dirnames.sort()
            filenames.sort()
            for name in filenames:
                source_path = Path(root) / name
                relative_path = source_path.relative_to(source_root).as_posix()
                if relative_path == 'code_package_manifest.json':
                    continue
                stat = source_path.stat()
                digest.update(relative_path.encode('utf-8', errors='surrogateescape'))
                digest.update(b'\0')
                digest.update(str(stat.st_size).encode('ascii'))
                digest.update(b'\0')
                digest.update(str(stat.st_mtime_ns).encode('ascii'))
                digest.update(b'\n')

        return digest.hexdigest()

    def _directory_stats(self, source_dir: Path) -> tuple[int, int]:
        file_count = 0
        size_bytes = 0
        source_root = Path(self._filesystem_path(source_dir))
        for root, _, filenames in os.walk(source_root):
            for name in filenames:
                source_path = Path(root) / name
                if not source_path.is_file():
                    continue
                file_count += 1
                size_bytes += source_path.stat().st_size
        return file_count, size_bytes

    def _remote_directory_cache_path(self, server: ServerRecord, cache_key: str) -> str | None:
        if server.remote_workspace is None or not server.remote_workspace.strip():
            return None

        return posixpath.join(
            server.remote_workspace.rstrip('/'),
            '.ironflow_upload_cache',
            'dirs',
            cache_key[:2],
            cache_key,
        )

    def _build_cache_check_command(self, remote_cache_path: str, cache_key: str) -> str:
        quoted_cache = shlex.quote(remote_cache_path)
        quoted_key = shlex.quote(cache_key)
        marker = f'{quoted_cache}/.ironflow_upload_cache_key'
        return (
            f'test -d {quoted_cache} && '
            f'test -f {marker} && '
            f'grep -qx {quoted_key} {marker}'
        )

    def _build_file_cache_check_command(self, remote_path: str, size_bytes: int) -> str:
        quoted_path = shlex.quote(remote_path)
        return (
            f'test -f {quoted_path} && '
            f'test "$(wc -c < {quoted_path})" -eq {int(size_bytes)}'
        )

    def _is_shared_asset_file_upload(self, server: ServerRecord, item: RemoteTransferItem) -> bool:
        if item.kind != 'file':
            return False
        if server.remote_workspace is None or not server.remote_workspace.strip():
            return False
        asset_root = posixpath.normpath(posixpath.join(server.remote_workspace.rstrip('/'), 'assets'))
        remote_path = posixpath.normpath(item.remote_path)

        return remote_path == asset_root or remote_path.startswith(f'{asset_root}/')

    def _build_extract_cache_archive_command(
        self,
        remote_archive_path: str,
        remote_cache_path: str,
        cache_key: str,
    ) -> str:
        quoted_archive = shlex.quote(remote_archive_path)
        quoted_cache = shlex.quote(remote_cache_path)
        quoted_tmp = shlex.quote(f'{remote_cache_path}.tmp')
        quoted_key = shlex.quote(cache_key)
        return (
            f'rm -rf {quoted_tmp}; '
            f'mkdir -p {quoted_tmp}; '
            f'tar -xzf {quoted_archive} -C {quoted_tmp}; '
            'status=$?; '
            'if [ $status -eq 0 ]; then '
            f'printf %s {quoted_key} > {quoted_tmp}/.ironflow_upload_cache_key; '
            f'rm -rf {quoted_cache}; '
            f'mv {quoted_tmp} {quoted_cache}; '
            'status=$?; '
            'fi; '
            f'rm -f {quoted_archive}; '
            f'if [ $status -ne 0 ]; then rm -rf {quoted_tmp}; fi; '
            'exit $status'
        )

    def _build_materialize_cached_directory_command(
        self,
        server: ServerRecord,
        remote_cache_path: str,
        remote_target_path: str,
        cache_key: str,
    ) -> str:
        self._validate_replaceable_remote_directory(server=server, remote_path=remote_target_path)
        quoted_cache = shlex.quote(remote_cache_path)
        quoted_key = shlex.quote(cache_key)
        normalized_target = posixpath.normpath(remote_target_path)
        remote_parent = posixpath.dirname(normalized_target.rstrip('/')) or '/'
        remote_tmp = f'{normalized_target}.tmp'
        quoted_parent = shlex.quote(remote_parent)
        quoted_target = shlex.quote(normalized_target)
        quoted_tmp = shlex.quote(remote_tmp)
        target_marker = f'{quoted_target}/.ironflow_materialized_cache_key'
        return (
            f'if test -d {quoted_target} && '
            f'test -f {target_marker} && '
            f'grep -qx {quoted_key} {target_marker}; then exit 0; fi; '
            f'rm -rf {quoted_tmp}; '
            f'mkdir -p {quoted_tmp}; '
            f'cp -a {quoted_cache}/. {quoted_tmp}/; '
            'status=$?; '
            'if [ $status -eq 0 ]; then '
            f'printf %s {quoted_key} > {quoted_tmp}/.ironflow_materialized_cache_key; '
            f'mkdir -p {quoted_parent}; '
            f'rm -rf {quoted_target}; '
            f'mv {quoted_tmp} {quoted_target}; '
            'status=$?; '
            'fi; '
            f'if [ $status -ne 0 ]; then rm -rf {quoted_tmp}; fi; '
            'exit $status'
        )

    def _build_extract_archive_command(
        self,
        server: ServerRecord,
        remote_archive_path: str,
        remote_target_path: str,
    ) -> str:
        self._validate_replaceable_remote_directory(server=server, remote_path=remote_target_path)
        quoted_archive = shlex.quote(remote_archive_path)
        normalized_target = posixpath.normpath(remote_target_path)
        remote_parent = posixpath.dirname(normalized_target.rstrip('/')) or '/'
        remote_tmp = f'{normalized_target}.tmp'
        quoted_parent = shlex.quote(remote_parent)
        quoted_target = shlex.quote(normalized_target)
        quoted_tmp = shlex.quote(remote_tmp)
        return (
            f'rm -rf {quoted_tmp}; '
            f'mkdir -p {quoted_tmp}; '
            f'tar -xzf {quoted_archive} -C {quoted_tmp}; '
            'status=$?; '
            'if [ $status -eq 0 ]; then '
            f'mkdir -p {quoted_parent}; '
            f'rm -rf {quoted_target}; '
            f'mv {quoted_tmp} {quoted_target}; '
            'status=$?; '
            'fi; '
            f'rm -f {quoted_archive}; '
            f'if [ $status -ne 0 ]; then rm -rf {quoted_tmp}; fi; '
            'exit $status'
        )

    def _validate_replaceable_remote_directory(self, server: ServerRecord, remote_path: str) -> None:
        normalized = posixpath.normpath(remote_path.strip())
        if not normalized or normalized in {'/', '.', '..'}:
            raise ValueError(f'refusing to replace unsafe remote directory: {remote_path!r}')
        if not normalized.startswith('/'):
            raise ValueError(f'remote directory replacement requires an absolute path: {remote_path!r}')
        if server.remote_workspace is not None and server.remote_workspace.strip():
            workspace = posixpath.normpath(server.remote_workspace.strip())
            if normalized == workspace:
                raise ValueError(f'refusing to replace remote workspace root: {remote_path!r}')

    def _timeout_result(
        self,
        operation: str,
        item: RemoteTransferItem,
        stdout: str,
        stderr: str,
        command_args: list[str],
    ) -> SshTransferResult:
        return SshTransferResult(
            operation=operation,
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=124,
            required=item.required,
            stdout=stdout,
            stderr=stderr,
            command_args=command_args,
        )

    def download_item(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        if item.kind == 'directory':
            return self._download_directory_archive(
                server=server,
                item=item,
                timeout_seconds=timeout_seconds,
            )
        if item.kind == 'glob':
            return self._download_glob_archive(
                server=server,
                item=item,
                timeout_seconds=timeout_seconds,
            )

        command_args = self._build_download_args(server=server, item=item)
        attempts = max(1, SCP_DOWNLOAD_MAX_ATTEMPTS if item.required else 1)
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        all_command_args: list[str] = []
        for attempt_index in range(attempts):
            if all_command_args:
                all_command_args.extend([';'])
            all_command_args.extend(command_args)
            try:
                completed = subprocess.run(
                    command_args,
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                stderr = stream_to_text(error.stderr) or 'scp download timed out'
                stdout_parts.append(stream_to_text(error.stdout))
                stderr_parts.append(stderr)
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=124,
                    required=item.required,
                    stdout='\n'.join(part for part in stdout_parts if part),
                    stderr='\n'.join(part for part in stderr_parts if part),
                    command_args=all_command_args,
                    metadata={'download_attempts': attempt_index + 1},
                )
            if completed.stdout:
                stdout_parts.append(completed.stdout)
            if completed.stderr:
                stderr_parts.append(completed.stderr)
            if completed.returncode == 0:
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=0,
                    required=item.required,
                    stdout='\n'.join(stdout_parts),
                    stderr='\n'.join(stderr_parts),
                    command_args=all_command_args,
                    metadata={'download_attempts': attempt_index + 1},
                )
            if (
                attempt_index + 1 >= attempts
                or not self._is_retryable_download_failure(stderr=completed.stderr, exit_code=completed.returncode)
            ):
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=completed.returncode,
                    required=item.required,
                    stdout='\n'.join(stdout_parts),
                    stderr='\n'.join(stderr_parts),
                    command_args=all_command_args,
                    metadata={'download_attempts': attempt_index + 1},
                )
            time.sleep(max(0.0, SCP_DOWNLOAD_RETRY_DELAY_SECONDS))

        return SshTransferResult(
            operation='download',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=completed.returncode,
            required=item.required,
            stdout='\n'.join(stdout_parts),
            stderr='\n'.join(stderr_parts),
            command_args=all_command_args,
            metadata={'download_attempts': attempts},
        )

    def _download_directory_archive(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        self._validate_download_target(server=server, item=item)
        local_target = Path(item.local_path)
        local_target.mkdir(parents=True, exist_ok=True)
        remote_archive_path = self._remote_download_archive_path(remote_path=item.remote_path)
        archive_command = self._build_remote_archive_command(
            remote_source_path=item.remote_path,
            remote_archive_path=remote_archive_path,
            exclude_patterns=item.exclude_patterns,
        )
        archive_args = self._build_ssh_args(server=server, command=archive_command)

        try:
            archive_completed = subprocess.run(
                archive_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='download',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh archive create timed out',
                command_args=archive_args,
            )
        if archive_completed.returncode != 0:
            return SshTransferResult(
                operation='download',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=archive_completed.returncode,
                required=item.required,
                stdout=archive_completed.stdout,
                stderr=archive_completed.stderr,
                command_args=archive_args,
            )

        with tempfile.TemporaryDirectory(prefix='ironflow_download_') as tmp_dir:
            archive_item = RemoteTransferItem(
                local_path=tmp_dir,
                remote_path=remote_archive_path,
                kind='file',
                required=item.required,
            )
            download_args = self._build_download_args(server=server, item=archive_item)
            try:
                download_completed = subprocess.run(
                    download_args,
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                self._cleanup_remote_archive(
                    server=server,
                    remote_archive_path=remote_archive_path,
                    timeout_seconds=timeout_seconds,
                )
                return self._timeout_result(
                    operation='download',
                    item=item,
                    stdout=stream_to_text(error.stdout),
                    stderr=stream_to_text(error.stderr) or 'scp archive download timed out',
                    command_args=download_args,
                )

            cleanup_result = self._cleanup_remote_archive(
                server=server,
                remote_archive_path=remote_archive_path,
                timeout_seconds=timeout_seconds,
            )
            if download_completed.returncode != 0:
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=download_completed.returncode,
                    required=item.required,
                    stdout=download_completed.stdout,
                    stderr=download_completed.stderr,
                    command_args=archive_args + [';'] + download_args + [';'] + cleanup_result.command_args,
                )

            archive_path = Path(tmp_dir) / posixpath.basename(remote_archive_path)
            try:
                self._extract_directory_archive(
                    archive_path=archive_path,
                    target_dir=local_target,
                )
            except (OSError, tarfile.TarError, ValueError) as error:
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=1,
                    required=item.required,
                    stdout=download_completed.stdout,
                    stderr=str(error),
                    command_args=archive_args + [';'] + download_args + [';'] + cleanup_result.command_args,
                )

        exit_code = 0 if cleanup_result.is_success else cleanup_result.exit_code
        return SshTransferResult(
            operation='download',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=exit_code,
            required=item.required,
            stdout='\n'.join(part for part in [
                archive_completed.stdout,
                download_completed.stdout,
                cleanup_result.stdout,
            ] if part),
            stderr='\n'.join(part for part in [
                archive_completed.stderr,
                download_completed.stderr,
                cleanup_result.stderr,
            ] if part),
            command_args=archive_args + [';'] + download_args + [';'] + cleanup_result.command_args,
        )

    def _download_glob_archive(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        self._validate_download_target(server=server, item=item)
        local_target = Path(item.local_path)
        local_target.mkdir(parents=True, exist_ok=True)
        remote_base, remote_pattern = self._split_glob_remote_path(remote_path=item.remote_path)
        remote_archive_path = self._remote_glob_download_archive_path(
            remote_base=remote_base,
            remote_pattern=remote_pattern,
        )
        archive_command = self._build_remote_glob_archive_command(
            remote_base=remote_base,
            remote_pattern=remote_pattern,
            remote_archive_path=remote_archive_path,
        )
        archive_args = self._build_ssh_args(server=server, command=archive_command)

        try:
            archive_completed = subprocess.run(
                archive_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return self._timeout_result(
                operation='download',
                item=item,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh glob archive create timed out',
                command_args=archive_args,
            )
        if archive_completed.returncode != 0:
            return SshTransferResult(
                operation='download',
                local_path=item.local_path,
                remote_path=item.remote_path,
                kind=item.kind,
                exit_code=archive_completed.returncode,
                required=item.required,
                stdout=archive_completed.stdout,
                stderr=archive_completed.stderr,
                command_args=archive_args,
                metadata={
                    'download_bundle_enabled': True,
                    'download_bundle_type': 'glob',
                    'download_bundle_stage': 'archive',
                    'remote_glob_base': remote_base,
                    'remote_glob_pattern': remote_pattern,
                },
            )

        with tempfile.TemporaryDirectory(prefix='ironflow_download_') as tmp_dir:
            archive_item = RemoteTransferItem(
                local_path=tmp_dir,
                remote_path=remote_archive_path,
                kind='file',
                required=item.required,
            )
            download_args = self._build_download_args(server=server, item=archive_item)
            try:
                download_completed = subprocess.run(
                    download_args,
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                self._cleanup_remote_archive(
                    server=server,
                    remote_archive_path=remote_archive_path,
                    timeout_seconds=timeout_seconds,
                )
                return self._timeout_result(
                    operation='download',
                    item=item,
                    stdout=stream_to_text(error.stdout),
                    stderr=stream_to_text(error.stderr) or 'scp glob archive download timed out',
                    command_args=download_args,
                )

            cleanup_result = self._cleanup_remote_archive(
                server=server,
                remote_archive_path=remote_archive_path,
                timeout_seconds=timeout_seconds,
            )
            if download_completed.returncode != 0:
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=download_completed.returncode,
                    required=item.required,
                    stdout=download_completed.stdout,
                    stderr=download_completed.stderr,
                    command_args=archive_args + [';'] + download_args + [';'] + cleanup_result.command_args,
                )

            archive_path = Path(tmp_dir) / posixpath.basename(remote_archive_path)
            try:
                self._extract_directory_archive(
                    archive_path=archive_path,
                    target_dir=local_target,
                )
            except (OSError, tarfile.TarError, ValueError) as error:
                return SshTransferResult(
                    operation='download',
                    local_path=item.local_path,
                    remote_path=item.remote_path,
                    kind=item.kind,
                    exit_code=1,
                    required=item.required,
                    stdout=download_completed.stdout,
                    stderr=str(error),
                    command_args=archive_args + [';'] + download_args + [';'] + cleanup_result.command_args,
                )

        exit_code = 0 if cleanup_result.is_success else cleanup_result.exit_code
        return SshTransferResult(
            operation='download',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=exit_code,
            required=item.required,
            stdout='\n'.join(part for part in [
                archive_completed.stdout,
                download_completed.stdout,
                cleanup_result.stdout,
            ] if part),
            stderr='\n'.join(part for part in [
                archive_completed.stderr,
                download_completed.stderr,
                cleanup_result.stderr,
            ] if part),
            command_args=archive_args + [';'] + download_args + [';'] + cleanup_result.command_args,
            metadata={
                'download_bundle_enabled': True,
                'download_bundle_type': 'glob',
                'remote_glob_base': remote_base,
                'remote_glob_pattern': remote_pattern,
            },
        )

    def _validate_download_target(self, server: ServerRecord, item: RemoteTransferItem) -> None:
        if server.host is None or not server.host.strip():
            raise ValueError('host is required for OpenScpTransferClient')
        if server.username is None or not server.username.strip():
            raise ValueError('username is required for OpenScpTransferClient')
        if item.kind not in {'file', 'directory', 'glob'}:
            raise ValueError(f'unsupported download item kind: {item.kind}')

    def _build_remote_archive_command(
        self,
        remote_source_path: str,
        remote_archive_path: str,
        exclude_patterns: tuple[str, ...],
    ) -> str:
        quoted_archive = shlex.quote(remote_archive_path)
        quoted_source = shlex.quote(remote_source_path.rstrip('/'))
        exclude_args = ' '.join(
            f'--exclude={shlex.quote(pattern)}'
            for pattern in exclude_patterns
        )
        exclude_prefix = f'{exclude_args} ' if exclude_args else ''
        return (
            f'tar -czf {quoted_archive} {exclude_prefix}-C {quoted_source} .; '
            'status=$?; '
            'if [ $status -ne 0 ]; then '
            f'rm -f {quoted_archive}; '
            'exit $status; '
            'fi'
        )

    def _build_remote_glob_archive_command(
        self,
        *,
        remote_base: str,
        remote_pattern: str,
        remote_archive_path: str,
    ) -> str:
        quoted_archive = shlex.quote(remote_archive_path)
        quoted_tmp_archive = shlex.quote(f'{remote_archive_path}.tmp')
        script = (
            'import glob\n'
            'import os\n'
            'import tarfile\n'
            f'base = {json.dumps(remote_base)}\n'
            f'pattern = {json.dumps(remote_pattern)}\n'
            f'archive_path = {json.dumps(remote_archive_path)}\n'
            'base_abs = os.path.abspath(base)\n'
            'tmp_archive = archive_path + ".tmp"\n'
            'matches = sorted(glob.glob(os.path.join(base_abs, pattern), recursive=True))\n'
            'file_count = 0\n'
            'os.makedirs(os.path.dirname(os.path.abspath(archive_path)) or ".", exist_ok=True)\n'
            'with tarfile.open(tmp_archive, "w:gz") as archive:\n'
            '    for path in matches:\n'
            '        if not os.path.isfile(path):\n'
            '            continue\n'
            '        resolved = os.path.abspath(path)\n'
            '        if os.path.commonpath([base_abs, resolved]) != base_abs:\n'
            '            raise SystemExit(f"glob match escapes base: {path}")\n'
            '        archive.add(resolved, arcname=os.path.relpath(resolved, base_abs))\n'
            '        file_count += 1\n'
            'os.replace(tmp_archive, archive_path)\n'
            'print(f"ironflow_glob_archive files={file_count} archive={archive_path}")\n'
        )
        return (
            "python3 - <<'PY'\n"
            f'{script}'
            'PY\n'
            'status=$?; '
            'if [ $status -ne 0 ]; then '
            f'rm -f {quoted_archive} {quoted_tmp_archive}; '
            'exit $status; '
            'fi; '
            'if [ ! -f '
            f'{quoted_archive}'
            ' ]; then '
            f'rm -f {quoted_tmp_archive}; '
            'echo ironflow_glob_archive_missing >&2; '
            'exit 1; '
            'fi'
        )

    def _cleanup_remote_archive(
        self,
        server: ServerRecord,
        remote_archive_path: str,
        timeout_seconds: int | None,
    ) -> SshTransferResult:
        cleanup_args = self._build_ssh_args(
            server=server,
            command=f'rm -f {shlex.quote(remote_archive_path)}',
        )
        try:
            completed = subprocess.run(
                cleanup_args,
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return SshTransferResult(
                operation='download',
                local_path='',
                remote_path=remote_archive_path,
                kind='file',
                exit_code=124,
                stdout=stream_to_text(error.stdout),
                stderr=stream_to_text(error.stderr) or 'ssh archive cleanup timed out',
                command_args=cleanup_args,
            )

        return SshTransferResult(
            operation='download',
            local_path='',
            remote_path=remote_archive_path,
            kind='file',
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command_args=cleanup_args,
        )

    def _extract_directory_archive(self, archive_path: Path, target_dir: Path) -> None:
        target_root = target_dir.resolve()
        with tarfile.open(self._filesystem_path(archive_path), 'r:gz') as archive:
            for member in archive.getmembers():
                member_target = (target_dir / member.name).resolve()
                if member_target != target_root and target_root not in member_target.parents:
                    raise ValueError(f'archive member escapes target directory: {member.name}')
            archive.extractall(self._filesystem_path(target_dir))

    def _split_glob_remote_path(self, remote_path: str) -> tuple[str, str]:
        normalized = remote_path.replace('\\', '/')
        parts = normalized.split('/')
        wildcard_index = next(
            (
                index
                for index, part in enumerate(parts)
                if any(character in part for character in ('*', '?', '['))
            ),
            -1,
        )
        if wildcard_index <= 0:
            return '.', normalized
        remote_base = '/'.join(parts[:wildcard_index])
        remote_pattern = '/'.join(parts[wildcard_index:])

        return posixpath.normpath(remote_base), remote_pattern

    def _remote_glob_download_archive_path(self, *, remote_base: str, remote_pattern: str) -> str:
        digest = hashlib.sha256(remote_pattern.encode('utf-8', errors='surrogateescape')).hexdigest()[:12]
        return f'{remote_base.rstrip("/")}.glob-{digest}.download.tar.gz'

    def _build_download_args(self, server: ServerRecord, item: RemoteTransferItem) -> list[str]:
        self._validate_download_target(server=server, item=item)

        local_path = Path(item.local_path)
        Path(self._filesystem_path(local_path)).mkdir(parents=True, exist_ok=True)
        args = [self.scp_executable]
        args.extend(
            build_open_ssh_options(
                connect_timeout_seconds=self.connect_timeout_seconds,
                strict_host_key_checking=self.strict_host_key_checking,
            ),
        )
        if server.port is not None:
            args.extend(['-P', str(server.port)])
        if server.key_path is not None and server.key_path.strip():
            args.extend(['-i', server.key_path])
        if item.kind == 'directory':
            args.append('-r')

        args.extend([
            self._remote_download_source(server=server, item=item),
            str(local_path),
        ])

        return args

    def _remote_download_source(self, server: ServerRecord, item: RemoteTransferItem) -> str:
        remote_path = item.remote_path
        if item.kind == 'directory':
            remote_path = f'{remote_path.rstrip("/")}/.'

        return f'{server.username}@{server.host}:{remote_path}'

    def _filesystem_path(self, path: Path) -> str:
        text = str(path)
        if os.name != 'nt':
            return text
        if text.startswith('\\\\?\\'):
            return text
        if text.startswith('\\\\'):
            return '\\\\?\\UNC\\' + text[2:]
        if path.is_absolute():
            return '\\\\?\\' + text

        return text


@dataclass(frozen=True, slots=True)
class SshUploadResult:
    server_name: str
    experiment_id: str
    success: bool
    transfer_results: list[SshTransferResult] = field(default_factory=list)
    message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'server_name': self.server_name,
            'experiment_id': self.experiment_id,
            'success': self.success,
            'message': self.message,
            'transfer_results': [
                result.to_dict()
                for result in self.transfer_results
            ],
        }


class SshUploader:
    def upload(
        self,
        server: ServerRecord,
        plan: SshExecutionPlan,
        client: BaseSshTransferClient,
        timeout_seconds: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> SshUploadResult:
        results: list[SshTransferResult] = []
        total = len(plan.upload_items)

        for index, item in enumerate(plan.upload_items, start=1):
            self._emit_progress(
                progress=progress,
                text=self._transfer_item_line(
                    prefix='ironflow_upload_item',
                    index=index,
                    total=total,
                    item=item,
                ),
            )
            result = client.upload_item(
                server=server,
                item=item,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)
            self._emit_progress(
                progress=progress,
                text=self._transfer_result_line(
                    prefix='ironflow_upload_result',
                    index=index,
                    total=total,
                    result=result,
                ),
            )
            if not result.is_success:
                return SshUploadResult(
                    server_name=server.name,
                    experiment_id=plan.experiment_id,
                    success=False,
                    transfer_results=results,
                    message='ssh upload failed',
                )

        return SshUploadResult(
            server_name=server.name,
            experiment_id=plan.experiment_id,
            success=True,
            transfer_results=results,
            message='ssh upload completed',
        )

    def _emit_progress(self, *, progress: Callable[[str], None] | None, text: str) -> None:
        if progress is not None:
            progress(text)
            return
        print(text, file=sys.stderr, flush=True)

    def _transfer_item_line(self, *, prefix: str, index: int, total: int, item: RemoteTransferItem) -> str:
        return (
            f'{prefix} {index}/{total} '
            f'kind={item.kind} required={str(item.required).lower()} '
            f'local={self._compact_transfer_path(item.local_path)} '
            f'remote={self._compact_transfer_path(item.remote_path)}'
        )

    def _transfer_result_line(self, *, prefix: str, index: int, total: int, result: SshTransferResult) -> str:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        cache = metadata.get('upload_cache_hit')
        cache_text = ''
        if isinstance(cache, bool):
            cache_text = f' cache={"hit" if cache else "miss"}'
        bundle = metadata.get('download_bundle_enabled')
        bundle_text = f' bundle={str(bundle).lower()}' if isinstance(bundle, bool) else ''
        size = metadata.get('local_size_bytes') or metadata.get('remote_size_bytes')
        size_text = f' bytes={size}' if isinstance(size, int) and size > 0 else ''
        return (
            f'{prefix} {index}/{total} '
            f'success={str(result.is_success).lower()} kind={result.kind}'
            f'{cache_text}{bundle_text}{size_text}'
        )

    def _compact_transfer_path(self, value: str, limit: int = 90) -> str:
        compact = value.replace('\\', '/')
        if len(compact) <= limit:
            return compact
        return '...' + compact[-(limit - 3):]


@dataclass(frozen=True, slots=True)
class SshDownloadResult:
    server_name: str
    experiment_id: str
    success: bool
    transfer_results: list[SshTransferResult] = field(default_factory=list)
    message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'server_name': self.server_name,
            'experiment_id': self.experiment_id,
            'success': self.success,
            'message': self.message,
            'transfer_results': [
                result.to_dict()
                for result in self.transfer_results
            ],
        }


class SshDownloader:
    def download(
        self,
        server: ServerRecord,
        plan: SshExecutionPlan,
        client: BaseSshTransferClient,
        timeout_seconds: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> SshDownloadResult:
        results: list[SshTransferResult] = []
        total = len(plan.download_items)

        helper = SshUploader()
        for index, item in enumerate(plan.download_items, start=1):
            helper._emit_progress(
                progress=progress,
                text=helper._transfer_item_line(
                    prefix='ironflow_download_item',
                    index=index,
                    total=total,
                    item=item,
                ),
            )
            result = client.download_item(
                server=server,
                item=item,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)
            helper._emit_progress(
                progress=progress,
                text=helper._transfer_result_line(
                    prefix='ironflow_download_result',
                    index=index,
                    total=total,
                    result=result,
                ),
            )

        has_required_failures = any(
            item.required and not result.is_success
            for item, result in zip(plan.download_items, results)
        )
        if has_required_failures:
            return SshDownloadResult(
                server_name=server.name,
                experiment_id=plan.experiment_id,
                success=False,
                transfer_results=results,
                message='ssh download failed',
            )

        has_optional_failures = any(not result.is_success for result in results)

        return SshDownloadResult(
            server_name=server.name,
            experiment_id=plan.experiment_id,
            success=True,
            transfer_results=results,
            message='ssh download completed with optional failures' if has_optional_failures else 'ssh download completed',
        )
