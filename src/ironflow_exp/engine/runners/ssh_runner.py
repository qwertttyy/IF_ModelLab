import json
import os
import shutil
import shlex
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ironflow_exp.engine.analyzer import ResultAnalyzer, SummaryWriter
from ironflow_exp.engine.collector import ResultCollector
from ironflow_exp.engine.configs import EngineConfigLoader, EngineExperimentConfig
from ironflow_exp.engine.core import TaskOrchestrationService
from ironflow_exp.engine.domain import ExperimentRecord, ExperimentStatus, ServerRecord
from ironflow_exp.engine.log_utils import read_text_tail
from ironflow_exp.engine.runners.base import BaseExperimentRunner, ExperimentRunnerResult
from ironflow_exp.engine.runners.finalizer import ExperimentResultFinalizer
from ironflow_exp.engine.server import (
    BaseSshClient,
    BaseSshTransferClient,
    CodePackageService,
    LocalSshSimulatorClient,
    LocalSshSimulatorTransferClient,
    OpenScpTransferClient,
    OpenSshClient,
    ServerProfileValidator,
    SshCommandResult,
    SshDownloader,
    SshRemoteExecutor,
    SshUploader,
    SshWorkspaceBootstrapper,
)
from ironflow_exp.engine.server.metadata_redaction import command_result_summary
from ironflow_exp.engine.server.ssh_job import (
    SshBackgroundJobService,
    SshJobStatusPollFailure,
    SshJobSupportFiles,
)
from ironflow_exp.engine.server.ssh_plan import SshExecutionPlan, SshExecutionPlanBuilder
from ironflow_exp.engine.server.shared_assets import SharedAssetPlan, SharedAssetPlanner
from ironflow_exp.engine.storage import SQLiteExperimentStorage


class SshExperimentRunner(BaseExperimentRunner):
    def __init__(
        self,
        storage: SQLiteExperimentStorage,
        server: ServerRecord,
        config_loader: EngineConfigLoader | None = None,
        plan_builder: SshExecutionPlanBuilder | None = None,
        ssh_client: BaseSshClient | None = None,
        transfer_client: BaseSshTransferClient | None = None,
        workspace_bootstrapper: SshWorkspaceBootstrapper | None = None,
        uploader: SshUploader | None = None,
        downloader: SshDownloader | None = None,
        remote_executor: SshRemoteExecutor | None = None,
        result_collector: ResultCollector | None = None,
        result_analyzer: ResultAnalyzer | None = None,
        summary_writer: SummaryWriter | None = None,
        result_finalizer: ExperimentResultFinalizer | None = None,
        task_orchestrator: TaskOrchestrationService | None = None,
        code_packager: CodePackageService | None = None,
        background_jobs: SshBackgroundJobService | None = None,
        allow_server_mismatch: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self.server = server
        self.config_loader = config_loader or EngineConfigLoader()
        self.plan_builder = plan_builder or SshExecutionPlanBuilder()
        self.ssh_client = ssh_client or self._default_ssh_client(server=server)
        self.transfer_client = transfer_client or self._default_transfer_client(server=server)
        self.workspace_bootstrapper = workspace_bootstrapper or SshWorkspaceBootstrapper()
        self.uploader = uploader or SshUploader()
        self.downloader = downloader or SshDownloader()
        self.remote_executor = remote_executor or SshRemoteExecutor()
        self.result_collector = result_collector or ResultCollector()
        self.result_analyzer = result_analyzer or ResultAnalyzer()
        self.summary_writer = summary_writer or SummaryWriter()
        self.task_orchestrator = task_orchestrator or TaskOrchestrationService()
        self.code_packager = code_packager or CodePackageService()
        self.background_jobs = background_jobs or SshBackgroundJobService()
        self.shared_asset_planner = SharedAssetPlanner()
        self.allow_server_mismatch = allow_server_mismatch
        self.clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self.result_finalizer = result_finalizer or ExperimentResultFinalizer(
            storage=self.storage,
            result_collector=self.result_collector,
            result_analyzer=self.result_analyzer,
            summary_writer=self.summary_writer,
            clock=self.clock,
        )

    def prepare(
        self,
        config: object,
        experiment_id: str,
        replace_existing: bool = False,
    ) -> ExperimentRunnerResult:
        if not isinstance(config, EngineExperimentConfig):
            raise TypeError('SshExperimentRunner requires EngineExperimentConfig')
        self._require_new_experiment_id(experiment_id=experiment_id, replace_existing=replace_existing)

        validation = ServerProfileValidator().validate(record=self.server, check_paths=False)
        if not validation.is_valid:
            errors = ', '.join(f'{issue.field}: {issue.message}' for issue in validation.errors)
            raise ValueError(f'invalid SSH server profile: {errors}')

        if self.server.server_type == 'local':
            raise ValueError('SshExperimentRunner requires ssh, vast_manual, or local_ssh_simulator server profile')

        workspace_dir = Path(config.runtime.workspace) / experiment_id
        result_dir = Path(config.runtime.experiment_root) / experiment_id
        if replace_existing:
            self.storage.delete_metrics(experiment_id=experiment_id)
            self._reset_run_directory(path=workspace_dir, root=Path(config.runtime.workspace))
            self._reset_run_directory(path=result_dir, root=Path(config.runtime.experiment_root))
        workspace_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = workspace_dir / 'config.json'
        plan_path = workspace_dir / 'ssh_plan.json'
        task_plan_path = workspace_dir / 'task_plan.json'
        shared_asset_plan = self._shared_asset_plan(config=config)
        remote_config = self.shared_asset_planner.apply_to_config(config=config, plan=shared_asset_plan)

        config_snapshot_path.write_text(
            data=json.dumps(self.config_loader.to_dict(config=remote_config), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        code_package = self.code_packager.package(config=remote_config, workspace_dir=workspace_dir)
        plan = self.plan_builder.build(
            config=remote_config,
            server=self.server,
            experiment_id=experiment_id,
            local_workspace_dir=workspace_dir,
            local_result_dir=result_dir,
            code_package=code_package,
        )
        if shared_asset_plan.assets:
            plan = replace(
                plan,
                upload_items=[
                    plan.upload_items[0],
                    *shared_asset_plan.upload_items(),
                    *plan.upload_items[1:],
                ],
            )
        job_support_files = self.background_jobs.write_support_files(
            plan=plan,
            timeout_seconds=remote_config.train.max_seconds,
        )
        plan = replace(
            plan,
            upload_items=[
                *plan.upload_items,
                *job_support_files.upload_items(),
            ],
        )
        plan_path.write_text(
            data=json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        task_plan = self.task_orchestrator.build_plan(
            config=remote_config,
            experiment_id=experiment_id,
            result_dir=result_dir,
        )
        task_plan_path.write_text(
            data=json.dumps(task_plan.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        self.storage.save_experiment(
            record=ExperimentRecord(
                experiment_id=experiment_id,
                name=config.experiment.name,
                status=ExperimentStatus.PENDING,
                runner_type='ssh',
                server_name=self.server.name,
                config_path=str(config_snapshot_path),
                result_path=str(result_dir),
                created_at=self._timestamp(),
                memo=config.experiment.description,
                metadata={
                    'workspace_dir': str(workspace_dir),
                    'ssh_plan_path': str(plan_path),
                    'ssh_plan': plan.to_dict(),
                    'code_package': code_package.to_dict(),
                    'task_plan_path': str(task_plan_path),
                    'task_plan': task_plan.to_dict(),
                    'remote_workspace_dir': plan.remote_workspace_dir,
                    'remote_result_dir': plan.remote_result_dir,
                    'remote_command_text': plan.remote_command_text,
                    'ssh_job_support': job_support_files.to_dict(),
                    'shared_assets': shared_asset_plan.to_dict(),
                    'tags': config.experiment.tags,
                },
            ),
        )

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.PENDING,
            workspace_dir=workspace_dir,
            message='ssh experiment prepared',
            metadata={
                'result_dir': str(result_dir),
                'ssh_plan_path': str(plan_path),
                'code_package': code_package.to_dict(),
                'task_plan_path': str(task_plan_path),
                'task_plan': task_plan.to_dict(),
                'remote_result_dir': plan.remote_result_dir,
                'remote_command_text': plan.remote_command_text,
                'ssh_job_support': job_support_files.to_dict(),
                'shared_assets': shared_asset_plan.to_dict(),
            },
        )

    def _shared_asset_plan(self, *, config: EngineExperimentConfig) -> SharedAssetPlan:
        if self.server.server_type == 'local_ssh_simulator':
            return SharedAssetPlan(enabled=False, package_include=list(config.code.package_include))

        return self.shared_asset_planner.build(config=config, remote_workspace=self.server.remote_workspace)

    def bootstrap(self, experiment_id: str, timeout_seconds: int | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='bootstrap')
        bootstrap_result = self.workspace_bootstrapper.bootstrap(
            server=self.server,
            plan=plan,
            client=self.ssh_client,
            timeout_seconds=timeout_seconds,
        )
        status = ExperimentStatus.PENDING if bootstrap_result.success else ExperimentStatus.FAILED
        timestamp = self._timestamp()
        updated_record = replace(
            record,
            status=status,
            finished_at=None if bootstrap_result.success else timestamp,
            error_type=None if bootstrap_result.success else 'ssh_bootstrap_failed',
            metadata={
                **record.metadata,
                'ssh_bootstrap': bootstrap_result.to_dict(),
                'ssh_bootstrapped_at': timestamp if bootstrap_result.success else None,
            },
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=self._workspace_dir(record=record),
            message=bootstrap_result.message,
            metadata={
                'server_name': self.server.name,
                'remote_workspace_dir': plan.remote_workspace_dir,
                'remote_result_dir': plan.remote_result_dir,
                'ssh_bootstrap': bootstrap_result.to_dict(),
            },
        )

    def upload(self, experiment_id: str, timeout_seconds: int | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        self._require_stage_success(record=record, metadata_key='ssh_bootstrap', stage_name='bootstrap')
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='upload')
        upload_result = self.uploader.upload(
            server=self.server,
            plan=plan,
            client=self.transfer_client,
            timeout_seconds=timeout_seconds,
        )
        status = ExperimentStatus.PENDING if upload_result.success else ExperimentStatus.FAILED
        timestamp = self._timestamp()
        updated_record = replace(
            record,
            status=status,
            finished_at=None if upload_result.success else timestamp,
            error_type=None if upload_result.success else 'ssh_upload_failed',
            metadata={
                **record.metadata,
                'ssh_upload': upload_result.to_dict(),
                'ssh_uploaded_at': timestamp if upload_result.success else None,
            },
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=self._workspace_dir(record=record),
            message=upload_result.message,
            metadata={
                'server_name': self.server.name,
                'remote_workspace_dir': plan.remote_workspace_dir,
                'remote_result_dir': plan.remote_result_dir,
                'ssh_upload': upload_result.to_dict(),
            },
        )

    def run(self, experiment_id: str, timeout_seconds: int | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        self._require_stage_success(record=record, metadata_key='ssh_upload', stage_name='upload')
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='run')
        config = self._load_config(record=record)
        workspace_dir = self._workspace_dir(record=record)
        stdout_path = workspace_dir / 'remote_stdout.log'
        stderr_path = workspace_dir / 'remote_stderr.log'
        started_at = self._timestamp()

        self.storage.update_experiment_status(
            experiment_id=experiment_id,
            status=ExperimentStatus.RUNNING,
            started_at=started_at,
        )
        run_result = self.remote_executor.run(
            server=self.server,
            plan=plan,
            client=self.ssh_client,
            timeout_seconds=timeout_seconds or config.train.max_seconds,
        )
        stdout_path.write_text(data=run_result.command_result.stdout, encoding='utf-8')
        stderr_path.write_text(data=run_result.command_result.stderr, encoding='utf-8')
        status = ExperimentStatus.FINISHED if run_result.success else ExperimentStatus.FAILED
        finished_at = self._timestamp()
        failure_download = None
        failure_downloaded_at = None
        if not run_result.success:
            failure_download = self._download_failed_remote_outputs(plan=plan, result_dir=self._result_dir(record=record))
            failure_downloaded_at = self._timestamp()

        latest_record = self._require_record(experiment_id=experiment_id)
        metadata = {
            **latest_record.metadata,
            'ssh_remote_run': run_result.to_dict(),
            'remote_stdout_path': str(stdout_path),
            'remote_stderr_path': str(stderr_path),
            'ssh_finished_at': finished_at,
        }
        result_metadata = {
            'server_name': self.server.name,
            'remote_result_dir': plan.remote_result_dir,
            'remote_command_text': plan.remote_command_text,
            'remote_stdout_path': str(stdout_path),
            'remote_stderr_path': str(stderr_path),
            'ssh_remote_run': run_result.to_dict(),
        }
        if failure_download is not None:
            metadata.update({
                'ssh_failure_download': failure_download,
                'ssh_failure_downloaded_at': failure_downloaded_at,
            })
            result_metadata.update({
                'ssh_failure_download': failure_download,
                'ssh_failure_downloaded_at': failure_downloaded_at,
            })

        updated_record = replace(
            latest_record,
            status=status,
            finished_at=finished_at,
            error_type=None if run_result.success else 'ssh_remote_command_failed',
            metadata=metadata,
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=workspace_dir,
            message=run_result.message,
            metadata=result_metadata,
        )

    def submit(self, experiment_id: str, timeout_seconds: int | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        self._require_stage_success(record=record, metadata_key='ssh_upload', stage_name='upload')
        config = self._load_config(record=record)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='submit')
        support_files = self._load_job_support_files(record=record)
        started_at = self._timestamp()
        command_result = self.background_jobs.submit(
            server=self.server,
            config=config,
            experiment_id=experiment_id,
            plan=plan,
            support_files=support_files,
            client=self.ssh_client,
            timeout_seconds=timeout_seconds,
        )
        background_metadata = self.background_jobs.build_submit_metadata(
            server=self.server,
            experiment_id=experiment_id,
            plan=plan,
            support_files=support_files,
            command_result=command_result,
            submitted_at=started_at,
        )
        status = ExperimentStatus.RUNNING if command_result.is_success else ExperimentStatus.FAILED
        latest_record = self._require_record(experiment_id=experiment_id)
        updated_record = replace(
            latest_record,
            status=status,
            started_at=started_at if command_result.is_success else latest_record.started_at,
            finished_at=None if command_result.is_success else self._timestamp(),
            error_type=None if command_result.is_success else 'ssh_background_submit_failed',
            metadata={
                **latest_record.metadata,
                'ssh_background_job': background_metadata,
            },
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=self._workspace_dir(record=record),
            message='ssh background job submitted' if command_result.is_success else 'ssh background submit failed',
            metadata={
                'server_name': self.server.name,
                'remote_result_dir': plan.remote_result_dir,
                'remote_command_text': plan.remote_command_text,
                'ssh_background_job': updated_record.metadata['ssh_background_job'],
            },
        )

    def download(self, experiment_id: str, timeout_seconds: int | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        self._require_remote_execution_success(record=record)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='download')
        result_dir = self._result_dir(record=record)
        result_dir.mkdir(parents=True, exist_ok=True)
        download_result = self.downloader.download(
            server=self.server,
            plan=plan,
            client=self.transfer_client,
            timeout_seconds=timeout_seconds,
        )
        status = record.status if download_result.success else ExperimentStatus.FAILED
        timestamp = self._timestamp()
        updated_record = replace(
            record,
            status=status,
            finished_at=timestamp if not download_result.success else record.finished_at,
            error_type=record.error_type if download_result.success else 'ssh_download_failed',
            metadata={
                **record.metadata,
                'ssh_download': download_result.to_dict(),
                'ssh_downloaded_at': timestamp if download_result.success else None,
            },
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=self._workspace_dir(record=record),
            message=download_result.message,
            metadata={
                'server_name': self.server.name,
                'remote_result_dir': plan.remote_result_dir,
                'result_dir': str(result_dir),
                'ssh_download': download_result.to_dict(),
            },
        )

    def status(self, experiment_id: str) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='status')
        record = self._refresh_background_status(record=record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=record.status,
            workspace_dir=self._workspace_dir(record=record),
            message='ssh status loaded from SQLite',
            metadata={
                'server_name': record.server_name,
                'remote_result_dir': record.metadata.get('remote_result_dir'),
                'result_dir': str(self._result_dir(record=record)),
                'workspace_dir': str(self._workspace_dir(record=record)),
                'ssh_background_status': record.metadata.get('ssh_background_status'),
                'ssh_background_last_poll_error': record.metadata.get('ssh_background_last_poll_error'),
                'next_action': self._next_action_for_status(status=record.status),
            },
        )

    def cancel(self, experiment_id: str, timeout_seconds: int | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='cancel')
        config = self._load_config(record=record)
        support_files = self._load_job_support_files(record=record)
        command_result = self.background_jobs.cancel(
            server=self.server,
            config=config,
            experiment_id=experiment_id,
            plan=plan,
            support_files=support_files,
            client=self.ssh_client,
            timeout_seconds=timeout_seconds,
        )
        timestamp = self._timestamp()
        status = ExperimentStatus.CANCELLED if command_result.is_success else ExperimentStatus.FAILED
        latest_record = self._require_record(experiment_id=experiment_id)
        updated_record = replace(
            latest_record,
            status=status,
            finished_at=timestamp,
            error_type=None if command_result.is_success else 'ssh_cancel_failed',
            metadata={
                **latest_record.metadata,
                'ssh_cancel': command_result_summary(command_result=command_result),
                'ssh_cancelled_at': timestamp if command_result.is_success else None,
            },
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=self._workspace_dir(record=record),
            message='ssh remote job cancelled' if command_result.is_success else 'ssh remote cancel failed',
            metadata={
                'server_name': self.server.name,
                'remote_result_dir': plan.remote_result_dir,
                'ssh_cancel': command_result_summary(command_result=command_result),
                'next_action': self._next_action_for_status(status=status),
            },
        )

    def logs(self, experiment_id: str, tail: int | None = None) -> str:
        record = self._require_record(experiment_id=experiment_id)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='logs')
        workspace_dir = self._workspace_dir(record=record)
        log_path = workspace_dir / 'remote_stdout.log'
        if log_path.exists():
            local_log = self._read_local_log_tail(log_path=log_path, tail=tail)
            if local_log:
                return local_log

        background_log = self._background_log_tail(record=record, tail=tail)
        if background_log:
            return background_log

        remote_log = self._remote_log_tail(record=record, tail=tail)
        if remote_log:
            return remote_log

        plan_path = workspace_dir / 'ssh_plan.json'
        if not plan_path.exists():
            return ''

        plan_text = self._read_local_log_tail(log_path=plan_path, tail=tail)
        if not plan_text:
            return ''

        return '\n'.join([
            'ssh plan is prepared; no run logs are available yet.',
            'next: run Bootstrap, Upload, then Submit before checking live logs.',
            '',
            plan_text,
        ])

    def _read_local_log_tail(self, log_path: Path, tail: int | None) -> str:
        return read_text_tail(path=log_path, line_count=tail)

    def _remote_log_tail(self, record: ExperimentRecord, tail: int | None) -> str:
        if record.status != ExperimentStatus.RUNNING and 'ssh_remote_run' not in record.metadata:
            return ''

        try:
            config = self._load_config(record=record)
            plan = self._load_plan(record=record)
            self._require_matching_plan_server(record=record, plan=plan, stage_name='remote_log')
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._save_diagnostic_metadata(
                record=record,
                key='ssh_remote_last_log_error',
                diagnostic=self._diagnostic(
                    error_type='ssh_remote_log_metadata_load_failed',
                    message=f'{type(error).__name__}: {error}',
                    stage='remote_log',
                ),
            )
            return ''

        log_path = f'{plan.remote_result_dir.rstrip("/")}/{config.output.log_file}'
        result = self._remote_file_tail(
            remote_python_executable=config.runtime.remote_python_executable,
            remote_path=log_path,
            tail=50 if tail is None else tail,
        )
        if not result.is_success:
            self._save_diagnostic_metadata(
                record=record,
                key='ssh_remote_last_log_error',
                diagnostic=self._diagnostic(
                    error_type='ssh_remote_log_tail_failed',
                    message='ssh remote log tail command failed',
                    stage='remote_log',
                    remote_path=log_path,
                    command_result=self._command_result_summary(command_result=result),
                ),
            )
            return ''

        return result.stdout.strip()

    def _background_log_tail(self, record: ExperimentRecord, tail: int | None) -> str:
        if 'ssh_background_job' not in record.metadata:
            return ''

        try:
            config = self._load_config(record=record)
            support_files = self._load_job_support_files(record=record)
        except (FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self._save_diagnostic_metadata(
                record=record,
                key='ssh_background_last_log_error',
                diagnostic=self._diagnostic(
                    error_type='ssh_background_log_metadata_load_failed',
                    message=f'{type(error).__name__}: {error}',
                    stage='background_log',
                ),
            )
            return ''

        result = self._remote_file_tail(
            remote_python_executable=config.runtime.remote_python_executable,
            remote_path=support_files.remote_stdout_path,
            tail=50 if tail is None else tail,
        )
        if not result.is_success:
            self._save_diagnostic_metadata(
                record=record,
                key='ssh_background_last_log_error',
                diagnostic=self._diagnostic(
                    error_type='ssh_background_log_tail_failed',
                    message='ssh background log tail command failed',
                    stage='background_log',
                    remote_path=support_files.remote_stdout_path,
                    command_result=self._command_result_summary(command_result=result),
                ),
            )
            return ''

        return result.stdout.strip()

    def _remote_file_tail(self, remote_python_executable: str, remote_path: str, tail: int):
        line_count = max(tail, 0)
        script = (
            'from collections import deque\n'
            'from pathlib import Path\n'
            f'path = Path({json.dumps(remote_path)})\n'
            f'line_count = {line_count}\n'
            'if not path.exists():\n'
            '    raise SystemExit(2)\n'
            'if line_count <= 0:\n'
            '    selected = []\n'
            'else:\n'
            '    with path.open(mode="r", encoding="utf-8", errors="replace") as file:\n'
            '        selected = [line.rstrip("\\r\\n") for line in deque(file, maxlen=line_count)]\n'
            'print("\\n".join(selected))\n'
        )
        command_args = [remote_python_executable, '-c', script]
        command = ' '.join(shlex.quote(part) for part in command_args)

        return self.ssh_client.run_command(server=self.server, command=command, timeout_seconds=10)

    def collect(
        self,
        experiment_id: str,
        output_dir: Path | None = None,
        download_timeout_seconds: int | None = None,
    ) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        plan = self._load_plan(record=record)
        self._require_matching_plan_server(record=record, plan=plan, stage_name='collect')
        record = self._refresh_background_status(record=record)
        if record.status == ExperimentStatus.RUNNING:
            return ExperimentRunnerResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.RUNNING,
                workspace_dir=self._workspace_dir(record=record),
                message='ssh background job is still running',
                metadata={
                    'server_name': self.server.name,
                    'ssh_background_status': record.metadata.get('ssh_background_status'),
                },
            )
        self._require_remote_execution_success(record=record)
        config = self._load_config(record=record)
        workspace_dir = self._workspace_dir(record=record)
        result_dir = self._result_dir(record=record)
        download_result = self.download(experiment_id=experiment_id, timeout_seconds=download_timeout_seconds)
        if not bool(download_result.metadata['ssh_download']['success']):
            return ExperimentRunnerResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.FAILED,
                workspace_dir=workspace_dir,
                message='ssh download failed',
                metadata=download_result.metadata,
            )

        result = self.result_finalizer.finalize(
            record=record,
            config=config,
            workspace_dir=workspace_dir,
            result_dir=result_dir,
            output_dir=output_dir,
        )
        if result.status == ExperimentStatus.COLLECTED:
            return replace(result, message='ssh experiment collected')

        return result

    def _mark_failed(
        self,
        record: ExperimentRecord,
        workspace_dir: Path,
        error_type: str,
        message: str,
    ) -> ExperimentRunnerResult:
        return self.result_finalizer.mark_failed(
            record=record,
            workspace_dir=workspace_dir,
            error_type=error_type,
            message=message,
        )

    def _download_failed_remote_outputs(
        self,
        plan: SshExecutionPlan,
        result_dir: Path,
    ) -> dict[str, object]:
        result_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_result = self.downloader.download(
                server=self.server,
                plan=plan,
                client=self.transfer_client,
            )
        except Exception as error:
            return {
                'server_name': self.server.name,
                'experiment_id': plan.experiment_id,
                'success': False,
                'message': f'ssh failure artifact download raised {type(error).__name__}: {error}',
                'transfer_results': [],
                'best_effort': True,
                'error_type': type(error).__name__,
            }

        return {
            **download_result.to_dict(),
            'best_effort': True,
        }

    def _require_record(self, experiment_id: str) -> ExperimentRecord:
        record = self.storage.get_experiment(experiment_id=experiment_id)
        if record is None:
            raise KeyError(f'unknown experiment_id: {experiment_id}')

        return record

    def _require_new_experiment_id(self, experiment_id: str, replace_existing: bool) -> None:
        if replace_existing:
            return
        if self.storage.get_experiment(experiment_id=experiment_id) is None:
            return

        raise ValueError(
            f'experiment_id already exists: {experiment_id}. '
            'Choose a new experiment id, or pass --replace-existing to intentionally reuse it.',
        )

    def _reset_run_directory(self, path: Path, root: Path) -> None:
        if not path.exists():
            return

        resolved_path = path.resolve()
        resolved_root = root.resolve()
        if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
            raise ValueError(f'refusing to reset run directory outside configured root: {resolved_path}')

        try:
            shutil.rmtree(resolved_path)
        except OSError as error:
            stale_path = self._next_stale_run_directory_path(path=resolved_path)
            try:
                resolved_path.rename(stale_path)
            except OSError as rename_error:
                raise RuntimeError(
                    f'failed to reset run directory: {resolved_path}; '
                    f'initial delete error: {error}; stale rename target: {stale_path}',
                ) from rename_error

    def _next_stale_run_directory_path(self, path: Path) -> Path:
        for index in range(1, 1000):
            candidate = path.with_name(f'{path.name}.stale_{os.getpid()}_{index}')
            if not candidate.exists():
                return candidate
        raise RuntimeError(f'could not allocate stale run directory name for {path}')

    def _workspace_dir(self, record: ExperimentRecord) -> Path:
        workspace_dir = record.metadata.get('workspace_dir')
        if not isinstance(workspace_dir, str):
            raise ValueError(f'experiment {record.experiment_id} has no workspace_dir metadata')

        return Path(workspace_dir)

    def _result_dir(self, record: ExperimentRecord) -> Path:
        if record.result_path is None:
            raise ValueError(f'experiment {record.experiment_id} has no result_path')

        return Path(record.result_path)

    def _require_stage_success(
        self,
        record: ExperimentRecord,
        metadata_key: str,
        stage_name: str,
    ) -> None:
        stage_data = record.metadata.get(metadata_key)
        if not isinstance(stage_data, dict) or stage_data.get('success') is not True:
            raise ValueError(
                f'experiment {record.experiment_id} requires successful ssh {stage_name} before this stage',
            )

    def _require_remote_execution_success(self, record: ExperimentRecord) -> None:
        stage_data = record.metadata.get('ssh_remote_run')
        if isinstance(stage_data, dict) and stage_data.get('success') is True:
            return

        background_data = record.metadata.get('ssh_background_status')
        if (
            isinstance(background_data, dict)
            and background_data.get('status') == 'finished'
            and record.status == ExperimentStatus.FINISHED
        ):
            return

        raise ValueError(
            f'experiment {record.experiment_id} requires successful ssh run or submit before this stage',
        )

    def _next_action_for_status(self, status: ExperimentStatus) -> str:
        if status == ExperimentStatus.PENDING:
            return 'bootstrap/upload/submit'
        if status == ExperimentStatus.RUNNING:
            return 'logs/status'
        if status == ExperimentStatus.FINISHED:
            return 'collect'
        if status == ExperimentStatus.COLLECTED:
            return 'done'
        if status == ExperimentStatus.FAILED:
            return 'logs'

        return ''

    def _load_config(self, record: ExperimentRecord) -> EngineExperimentConfig:
        if record.config_path is None:
            raise ValueError(f'experiment {record.experiment_id} has no config snapshot')

        data = json.loads(Path(record.config_path).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('config snapshot must contain a mapping object')

        return self.config_loader.from_dict(data=data)

    def _load_plan(self, record: ExperimentRecord) -> SshExecutionPlan:
        plan_data = record.metadata.get('ssh_plan')
        if isinstance(plan_data, dict):
            return SshExecutionPlan.from_dict(data=plan_data)

        plan_path = record.metadata.get('ssh_plan_path')
        if not isinstance(plan_path, str):
            raise ValueError(f'experiment {record.experiment_id} has no ssh_plan metadata')

        data = json.loads(Path(plan_path).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('ssh plan file must contain a mapping object')

        return SshExecutionPlan.from_dict(data=data)

    def _require_matching_plan_server(
        self,
        record: ExperimentRecord,
        plan: SshExecutionPlan,
        stage_name: str,
    ) -> None:
        if self.allow_server_mismatch:
            return
        if plan.server_name == self.server.name:
            return

        raise ValueError(
            f'experiment {record.experiment_id} ssh {stage_name} server mismatch: '
            f'prepared with server {plan.server_name!r}, but current server is {self.server.name!r}. '
            'Use --force-server only when intentionally recovering or migrating a prepared run.',
        )

    def _load_job_support_files(self, record: ExperimentRecord) -> SshJobSupportFiles:
        data = record.metadata.get('ssh_job_support')
        if not isinstance(data, dict):
            raise ValueError(f'experiment {record.experiment_id} has no ssh_job_support metadata')

        return SshJobSupportFiles.from_dict(data=data)

    def _refresh_background_status(self, record: ExperimentRecord) -> ExperimentRecord:
        if 'ssh_background_job' not in record.metadata:
            return record
        if record.status not in {ExperimentStatus.RUNNING, ExperimentStatus.PENDING}:
            return record

        try:
            config = self._load_config(record=record)
            support_files = self._load_job_support_files(record=record)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
            return self._save_diagnostic_metadata(
                record=record,
                key='ssh_background_last_poll_error',
                diagnostic=self._diagnostic(
                    error_type='ssh_background_status_metadata_load_failed',
                    message=f'{type(error).__name__}: {error}',
                    stage='background_status',
                ),
            )

        checked_at = self._timestamp()
        status_poll = self.background_jobs.poll_status(
            server=self.server,
            config=config,
            experiment_id=record.experiment_id,
            remote_command_text=str(record.metadata.get('remote_command_text', '')),
            support_files=support_files,
            client=self.ssh_client,
            fallback_status=record.status,
            fallback_error_type=record.error_type,
            fallback_finished_at=record.finished_at or checked_at,
        )
        if isinstance(status_poll, SshJobStatusPollFailure):
            return self._save_diagnostic_metadata(
                record=record,
                key='ssh_background_last_poll_error',
                diagnostic={
                    **status_poll.to_dict(),
                    'stage': 'background_status',
                    'checked_at': checked_at,
                },
            )

        remote_run = status_poll.remote_run_metadata or record.metadata.get('ssh_remote_run')

        metadata = {
            **record.metadata,
            'ssh_background_status': status_poll.status_data,
            'ssh_background_checked_at': checked_at,
        }
        if status_poll.finished_at_source is not None:
            metadata['ssh_background_finished_at_source'] = status_poll.finished_at_source
        if isinstance(remote_run, dict):
            metadata['ssh_remote_run'] = remote_run

        updated_record = replace(
            record,
            status=status_poll.experiment_status,
            finished_at=status_poll.finished_at,
            error_type=status_poll.error_type,
            metadata=metadata,
        )
        self.storage.save_experiment(record=updated_record)

        return updated_record

    def _save_diagnostic_metadata(
        self,
        record: ExperimentRecord,
        key: str,
        diagnostic: dict[str, object],
    ) -> ExperimentRecord:
        latest_record = self.storage.get_experiment(experiment_id=record.experiment_id) or record
        metadata = {
            **latest_record.metadata,
            key: diagnostic,
        }
        updated_record = replace(latest_record, metadata=metadata)
        self.storage.save_experiment(record=updated_record)

        return updated_record

    def _diagnostic(
        self,
        error_type: str,
        message: str,
        stage: str,
        remote_path: str | None = None,
        command_result: dict[str, object] | None = None,
    ) -> dict[str, object]:
        data: dict[str, object] = {
            'error_type': error_type,
            'message': message,
            'stage': stage,
            'checked_at': self._timestamp(),
        }
        if remote_path is not None:
            data['remote_path'] = remote_path
        if command_result is not None:
            data['command_result'] = command_result

        return data

    def _command_result_summary(self, command_result: SshCommandResult) -> dict[str, object]:
        return command_result_summary(command_result=command_result)

    def _timestamp(self) -> str:
        return self.clock().isoformat()

    def _default_ssh_client(self, server: ServerRecord) -> BaseSshClient:
        if server.server_type == 'local_ssh_simulator':
            return LocalSshSimulatorClient()

        return OpenSshClient()

    def _default_transfer_client(self, server: ServerRecord) -> BaseSshTransferClient:
        if server.server_type == 'local_ssh_simulator':
            return LocalSshSimulatorTransferClient()

        return OpenScpTransferClient()
