import subprocess
from dataclasses import dataclass

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.metadata_redaction import command_result_summary
from ironflow_exp.engine.server.ssh_client import BaseSshClient, SshCommandResult
from ironflow_exp.engine.server.ssh_plan import SshExecutionPlan
from ironflow_exp.engine.server.subprocess_utils import stream_to_text


@dataclass(frozen=True, slots=True)
class SshRemoteRunResult:
    server_name: str
    experiment_id: str
    success: bool
    command_result: SshCommandResult
    message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'server_name': self.server_name,
            'experiment_id': self.experiment_id,
            'success': self.success,
            'message': self.message,
            'command_result': command_result_summary(command_result=self.command_result),
        }


class SshRemoteExecutor:
    def run(
        self,
        server: ServerRecord,
        plan: SshExecutionPlan,
        client: BaseSshClient,
        timeout_seconds: int | None = None,
    ) -> SshRemoteRunResult:
        try:
            command_result = client.run_command(
                server=server,
                command=plan.remote_command_text,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stderr = stream_to_text(error.stderr) or 'ssh remote command timed out'
            command_result = SshCommandResult(
                command=plan.remote_command_text,
                exit_code=124,
                stdout=stream_to_text(error.stdout),
                stderr=stderr,
            )

        return SshRemoteRunResult(
            server_name=server.name,
            experiment_id=plan.experiment_id,
            success=command_result.is_success,
            command_result=command_result,
            message='ssh remote command finished' if command_result.is_success else 'ssh remote command failed',
        )
