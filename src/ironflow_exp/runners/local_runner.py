import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ironflow_exp.configs import ExperimentConfig
from ironflow_exp.core.enums import JobStatus
from ironflow_exp.domain import JobRecord
from ironflow_exp.runners.base import BaseRunner, RunnerResult
from ironflow_exp.services import ExperimentService, ExperimentServiceResult


class LocalRunner(BaseRunner):
    def __init__(self, experiment_service: ExperimentService | None = None) -> None:
        self.experiment_service = experiment_service or ExperimentService()

    def run_preprocessing(
        self,
        config: ExperimentConfig,
        run_id: str | None = None,
        check_paths: bool = True,
    ) -> RunnerResult:
        context = self.experiment_service.create_context(config=config, run_id=run_id)
        output_dir = Path(context.output_dir)
        logs_dir = output_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        started_at = self._timestamp()
        running_job = self._job_record(
            run_id=context.run_id,
            status=JobStatus.RUNNING.value,
            started_at=started_at,
            ended_at=None,
            output_dir=output_dir,
            failure_path=None,
        )
        self._append_log(output_dir=output_dir, message='local preprocessing run started')

        try:
            service_result = self.experiment_service.run_preprocessing(
                config=config,
                run_id=context.run_id,
                check_paths=check_paths,
            )
        except Exception as error:
            ended_at = self._timestamp()
            failure_path = self._write_failure(
                output_dir=output_dir,
                run_id=context.run_id,
                error=error,
                created_at=ended_at,
            )
            failed_job = self._job_record(
                run_id=context.run_id,
                status=JobStatus.FAILED.value,
                started_at=running_job.started_at,
                ended_at=ended_at,
                output_dir=output_dir,
                failure_path=failure_path,
            )
            self._append_log(output_dir=output_dir, message=f'local preprocessing run failed: {type(error).__name__}: {error}')

            return RunnerResult(
                job=failed_job,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        ended_at = self._timestamp()
        success_job = self._job_record(
            run_id=service_result.context.run_id,
            status=JobStatus.SUCCESS.value,
            started_at=running_job.started_at,
            ended_at=ended_at,
            output_dir=Path(service_result.context.output_dir),
            failure_path=None,
            service_result=service_result,
        )
        self._append_log(output_dir=Path(service_result.context.output_dir), message='local preprocessing run completed')

        return RunnerResult(
            job=success_job,
            service_result=service_result,
        )

    def _job_record(
        self,
        run_id: str,
        status: str,
        started_at: str | None,
        ended_at: str | None,
        output_dir: Path,
        failure_path: Path | None,
        service_result: ExperimentServiceResult | None = None,
    ) -> JobRecord:
        return JobRecord(
            job_id=f'job_{run_id}',
            run_id=run_id,
            status=status,
            runner='local',
            started_at=started_at,
            ended_at=ended_at,
            log_path='logs/job.log',
            failure_path=self._relative_path(output_dir=output_dir, path=failure_path) if failure_path is not None else None,
            metadata={
                'output_dir': str(output_dir),
                'artifact_manifest_path': service_result.context.artifact_manifest_path if service_result is not None else None,
            },
        )

    def _write_failure(
        self,
        output_dir: Path,
        run_id: str,
        error: Exception,
        created_at: str,
    ) -> Path:
        failure_path = output_dir / 'logs' / 'failure.json'
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'schema_version': '0.1',
            'run_id': run_id,
            'failed_stage': 'local_runner.run_preprocessing',
            'task': 'preprocessing',
            'model_id': None,
            'error_type': type(error).__name__,
            'message': str(error),
            'log_path': 'logs/job.log',
            'created_at': created_at,
        }

        extra_data = self._failure_extra_data(error=error)
        if extra_data:
            data['metadata'] = extra_data

        with failure_path.open(mode='w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        return failure_path

    def _failure_extra_data(self, error: Exception) -> dict[str, object]:
        extra_data: dict[str, object] = {}

        validation_result = getattr(error, 'validation_result', None)
        if validation_result is not None:
            extra_data['validation_result'] = asdict(validation_result)

        compatibility_result = getattr(error, 'compatibility_result', None)
        if compatibility_result is not None:
            extra_data['compatibility_result'] = asdict(compatibility_result)

        stages = getattr(error, 'stages', None)
        if stages is not None:
            extra_data['stages'] = [
                asdict(stage)
                for stage in stages
            ]

        return extra_data

    def _append_log(self, output_dir: Path, message: str) -> None:
        log_path = output_dir / 'logs' / 'job.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = f'{self._timestamp()} {message}\n'

        with log_path.open(mode='a', encoding='utf-8') as file:
            file.write(line)

    def _relative_path(self, output_dir: Path, path: Path) -> str:
        try:
            return path.relative_to(output_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _timestamp(self) -> str:
        return datetime.now(tz=timezone.utc).isoformat()
