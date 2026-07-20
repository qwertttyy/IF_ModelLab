import glob
import shlex
import shutil
import subprocess
from pathlib import Path

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.ssh_client import BaseSshClient, SshCommandResult
from ironflow_exp.engine.server.ssh_plan import RemoteTransferItem
from ironflow_exp.engine.server.ssh_transfer import BaseSshTransferClient, SshTransferResult
from ironflow_exp.engine.server.subprocess_utils import stream_to_text


class LocalSshSimulatorClient(BaseSshClient):
    supports_compound_commands = False

    def run_command(
        self,
        server: ServerRecord,
        command: str,
        timeout_seconds: int | None = None,
    ) -> SshCommandResult:
        args = shlex.split(command, posix=True)
        if self._is_mkdir_command(args=args):
            return self._run_mkdir(command=command, args=args)

        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stderr = stream_to_text(error.stderr) or 'local ssh simulator command timed out'
            return SshCommandResult(
                command=command,
                exit_code=124,
                stdout=stream_to_text(error.stdout),
                stderr=stderr,
            )

        return SshCommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _is_mkdir_command(self, args: list[str]) -> bool:
        return len(args) >= 3 and args[0] == 'mkdir' and args[1] == '-p'

    def _run_mkdir(self, command: str, args: list[str]) -> SshCommandResult:
        for raw_path in args[2:]:
            Path(raw_path).mkdir(parents=True, exist_ok=True)

        return SshCommandResult(command=command, exit_code=0)


class LocalSshSimulatorTransferClient(BaseSshTransferClient):
    def upload_item(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        try:
            if item.kind == 'file':
                self._copy_file(source=Path(item.local_path), destination=Path(item.remote_path))
            elif item.kind == 'directory':
                self._copy_directory_contents(source=Path(item.local_path), destination=Path(item.remote_path))
            else:
                raise ValueError(f'unsupported local simulator upload item kind: {item.kind}')
        except Exception as error:
            return self._transfer_error(operation='upload', item=item, error=error)

        return SshTransferResult(
            operation='upload',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=0,
            required=item.required,
            command_args=['local-copy-upload', item.local_path, item.remote_path],
        )

    def download_item(
        self,
        server: ServerRecord,
        item: RemoteTransferItem,
        timeout_seconds: int | None = None,
    ) -> SshTransferResult:
        try:
            if item.kind == 'file':
                self._download_file(item=item)
            elif item.kind == 'directory':
                self._copy_directory_contents(source=Path(item.remote_path), destination=Path(item.local_path))
            elif item.kind == 'glob':
                self._download_glob(item=item)
            else:
                raise ValueError(f'unsupported local simulator download item kind: {item.kind}')
        except Exception as error:
            return self._transfer_error(operation='download', item=item, error=error)

        return SshTransferResult(
            operation='download',
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=0,
            required=item.required,
            command_args=['local-copy-download', item.remote_path, item.local_path],
        )

    def _download_file(self, item: RemoteTransferItem) -> None:
        source = Path(item.remote_path)
        destination = Path(item.local_path) / source.name
        self._copy_file(source=source, destination=destination)

    def _download_glob(self, item: RemoteTransferItem) -> None:
        matches = [
            Path(path)
            for path in glob.glob(item.remote_path)
        ]
        if not matches:
            raise FileNotFoundError(item.remote_path)

        for source in matches:
            destination = Path(item.local_path) / source.name
            self._copy_file(source=source, destination=destination)

    def _copy_file(self, source: Path, destination: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(source)

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _copy_directory_contents(self, source: Path, destination: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(source)

        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            child_destination = destination / child.name
            if child.is_dir():
                shutil.copytree(child, child_destination, dirs_exist_ok=True)
            else:
                self._copy_file(source=child, destination=child_destination)

    def _transfer_error(
        self,
        operation: str,
        item: RemoteTransferItem,
        error: Exception,
    ) -> SshTransferResult:
        return SshTransferResult(
            operation=operation,
            local_path=item.local_path,
            remote_path=item.remote_path,
            kind=item.kind,
            exit_code=1,
            required=item.required,
            stderr=f'{type(error).__name__}: {error}',
        )
