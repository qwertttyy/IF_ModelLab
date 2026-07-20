from dataclasses import dataclass, field

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.checker import (
    DEFAULT_REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILE,
    ServerChecker,
    ServerCheckResult,
)


@dataclass(frozen=True, slots=True)
class ServerPreflightStageResult:
    stage: str
    result: ServerCheckResult

    def to_dict(self) -> dict[str, object]:
        return {
            'stage': self.stage,
            'status': self.result.status,
            'success': self.result.is_success,
            'message': self.result.message,
            'metadata': self.result.metadata,
        }


@dataclass(frozen=True, slots=True)
class ServerPreflightResult:
    name: str
    status: str
    message: str
    stages: list[ServerPreflightStageResult] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == 'ok'

    def to_dict(self) -> dict[str, object]:
        return {
            'name': self.name,
            'status': self.status,
            'success': self.is_success,
            'message': self.message,
            'stages': [
                stage.to_dict()
                for stage in self.stages
            ],
            'metadata': self.metadata,
        }


class ServerPreflightRunner:
    def __init__(self, checker: ServerChecker | None = None) -> None:
        self.checker = checker or ServerChecker()

    def run(
        self,
        *,
        record: ServerRecord,
        dependency_profile: str = DEFAULT_REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILE,
        live_timeout_seconds: int = 10,
        gpu_timeout_seconds: int = 10,
        dependency_timeout_seconds: int = 90,
    ) -> ServerPreflightResult:
        stages: list[ServerPreflightStageResult] = []
        for stage_name, result in [
            (
                'live_ssh',
                self.checker.check(
                    record=record,
                    live_ssh=True,
                    timeout_seconds=live_timeout_seconds,
                ),
            ),
        ]:
            stages.append(ServerPreflightStageResult(stage=stage_name, result=result))
            if not result.is_success:
                return self._failed(record=record, stages=stages, failed_stage=stage_name, dependency_profile=dependency_profile)

        gpu_result = self.checker.check(
            record=record,
            gpu_probe=True,
            timeout_seconds=gpu_timeout_seconds,
        )
        stages.append(ServerPreflightStageResult(stage='gpu_probe', result=gpu_result))
        if not gpu_result.is_success:
            return self._failed(record=record, stages=stages, failed_stage='gpu_probe', dependency_profile=dependency_profile)

        dependency_result = self.checker.check(
            record=record,
            remote_task_adapter_deps=True,
            dependency_profile=dependency_profile,
            timeout_seconds=dependency_timeout_seconds,
        )
        stages.append(ServerPreflightStageResult(stage='remote_task_adapter_deps', result=dependency_result))
        if not dependency_result.is_success:
            return self._failed(
                record=record,
                stages=stages,
                failed_stage='remote_task_adapter_deps',
                dependency_profile=dependency_profile,
            )

        return ServerPreflightResult(
            name=record.name,
            status='ok',
            message='server GPU preflight succeeded',
            stages=stages,
            metadata={
                'server_type': record.server_type,
                'dependency_profile': dependency_profile,
                'stage_count': len(stages),
                'failed_stage': None,
            },
        )

    def _failed(
        self,
        *,
        record: ServerRecord,
        stages: list[ServerPreflightStageResult],
        failed_stage: str,
        dependency_profile: str,
    ) -> ServerPreflightResult:
        return ServerPreflightResult(
            name=record.name,
            status='failed',
            message=f'server GPU preflight failed at {failed_stage}',
            stages=stages,
            metadata={
                'server_type': record.server_type,
                'dependency_profile': dependency_profile,
                'stage_count': len(stages),
                'failed_stage': failed_stage,
            },
        )
