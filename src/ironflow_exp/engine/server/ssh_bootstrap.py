from dataclasses import dataclass, field

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.metadata_redaction import command_result_summary
from ironflow_exp.engine.server.ssh_client import BaseSshClient, SshCommandResult
from ironflow_exp.engine.server.ssh_plan import SshExecutionPlan


@dataclass(frozen=True, slots=True)
class SshBootstrapResult:
    server_name: str
    experiment_id: str
    success: bool
    command_results: list[SshCommandResult] = field(default_factory=list)
    message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'server_name': self.server_name,
            'experiment_id': self.experiment_id,
            'success': self.success,
            'message': self.message,
            'command_results': [
                command_result_summary(command_result=result)
                for result in self.command_results
            ],
        }


class SshWorkspaceBootstrapper:
    def bootstrap(
        self,
        server: ServerRecord,
        plan: SshExecutionPlan,
        client: BaseSshClient,
        timeout_seconds: int | None = None,
    ) -> SshBootstrapResult:
        if not plan.bootstrap_commands:
            return SshBootstrapResult(
                server_name=server.name,
                experiment_id=plan.experiment_id,
                success=True,
                command_results=[],
                message='remote workspace bootstrapped',
            )

        if not getattr(client, 'supports_compound_commands', True):
            return self._bootstrap_one_by_one(
                server=server,
                plan=plan,
                client=client,
                timeout_seconds=timeout_seconds,
            )

        command = ' && '.join(f'({bootstrap_command})' for bootstrap_command in plan.bootstrap_commands)
        result = client.run_command(
            server=server,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        if not result.is_success:
            return SshBootstrapResult(
                server_name=server.name,
                experiment_id=plan.experiment_id,
                success=False,
                command_results=[result],
                message='remote workspace bootstrap failed',
            )

        return SshBootstrapResult(
            server_name=server.name,
            experiment_id=plan.experiment_id,
            success=True,
            command_results=[result],
            message='remote workspace bootstrapped',
        )

    def _bootstrap_one_by_one(
        self,
        server: ServerRecord,
        plan: SshExecutionPlan,
        client: BaseSshClient,
        timeout_seconds: int | None,
    ) -> SshBootstrapResult:
        results: list[SshCommandResult] = []

        for command in plan.bootstrap_commands:
            result = client.run_command(
                server=server,
                command=command,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)
            if not result.is_success:
                return SshBootstrapResult(
                    server_name=server.name,
                    experiment_id=plan.experiment_id,
                    success=False,
                    command_results=results,
                    message='remote workspace bootstrap failed',
                )

        return SshBootstrapResult(
            server_name=server.name,
            experiment_id=plan.experiment_id,
            success=True,
            command_results=results,
            message='remote workspace bootstrapped',
        )
