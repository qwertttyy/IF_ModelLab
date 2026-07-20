import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.subprocess_utils import stream_to_text


@dataclass(frozen=True, slots=True)
class SshCommandResult:
    command: str
    exit_code: int
    stdout: str = ''
    stderr: str = ''

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0


class BaseSshClient(ABC):
    supports_compound_commands = True

    @abstractmethod
    def run_command(
        self,
        server: ServerRecord,
        command: str,
        timeout_seconds: int | None = None,
    ) -> SshCommandResult:
        """Run one command on a remote server."""


DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_STRICT_HOST_KEY_CHECKING = 'accept-new'
DEFAULT_SERVER_ALIVE_INTERVAL_SECONDS = 15
DEFAULT_SERVER_ALIVE_COUNT_MAX = 2


def build_open_ssh_options(
    connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    strict_host_key_checking: str = DEFAULT_STRICT_HOST_KEY_CHECKING,
    server_alive_interval_seconds: int = DEFAULT_SERVER_ALIVE_INTERVAL_SECONDS,
    server_alive_count_max: int = DEFAULT_SERVER_ALIVE_COUNT_MAX,
) -> list[str]:
    return [
        '-o',
        'BatchMode=yes',
        '-o',
        f'ConnectTimeout={connect_timeout_seconds}',
        '-o',
        f'StrictHostKeyChecking={strict_host_key_checking}',
        '-o',
        f'ServerAliveInterval={server_alive_interval_seconds}',
        '-o',
        f'ServerAliveCountMax={server_alive_count_max}',
    ]


class OpenSshClient(BaseSshClient):
    def __init__(
        self,
        ssh_executable: str = 'ssh',
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        strict_host_key_checking: str = DEFAULT_STRICT_HOST_KEY_CHECKING,
    ) -> None:
        self.ssh_executable = ssh_executable
        self.connect_timeout_seconds = connect_timeout_seconds
        self.strict_host_key_checking = strict_host_key_checking

    def run_command(
        self,
        server: ServerRecord,
        command: str,
        timeout_seconds: int | None = None,
    ) -> SshCommandResult:
        try:
            completed = subprocess.run(
                self._build_ssh_args(server=server, command=command),
                capture_output=True,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stderr = stream_to_text(error.stderr) or 'ssh command timed out'
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

    def _build_ssh_args(self, server: ServerRecord, command: str) -> list[str]:
        if server.host is None or not server.host.strip():
            raise ValueError('host is required for OpenSshClient')
        if server.username is None or not server.username.strip():
            raise ValueError('username is required for OpenSshClient')

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
