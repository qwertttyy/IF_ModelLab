import json
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ironflow_exp.engine.analyzer import ResultAnalyzer, SummaryWriter
from ironflow_exp.engine.collector import ResultCollector
from ironflow_exp.engine.configs import EngineConfigLoader, EngineExperimentConfig
from ironflow_exp.engine.core import TaskOrchestrationService
from ironflow_exp.engine.domain import ExperimentRecord, ExperimentStatus
from ironflow_exp.engine.log_utils import read_text_tail
from ironflow_exp.engine.output_policy import build_mock_output_args
from ironflow_exp.engine.runners.base import BaseExperimentRunner, ExperimentRunnerResult
from ironflow_exp.engine.runners.finalizer import ExperimentResultFinalizer
from ironflow_exp.engine.simulator import LocalMockTaskExecutor
from ironflow_exp.engine.storage import SQLiteExperimentStorage
from ironflow_exp.engine.tasks import AdapterTaskExecutor, build_builtin_task_adapter_registry


class LocalExperimentRunner(BaseExperimentRunner):
    def __init__(
        self,
        storage: SQLiteExperimentStorage,
        config_loader: EngineConfigLoader | None = None,
        result_collector: ResultCollector | None = None,
        result_analyzer: ResultAnalyzer | None = None,
        summary_writer: SummaryWriter | None = None,
        result_finalizer: ExperimentResultFinalizer | None = None,
        task_orchestrator: TaskOrchestrationService | None = None,
        task_executor: LocalMockTaskExecutor | None = None,
        adapter_task_executor: AdapterTaskExecutor | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self.config_loader = config_loader or EngineConfigLoader()
        self.result_collector = result_collector or ResultCollector()
        self.result_analyzer = result_analyzer or ResultAnalyzer()
        self.summary_writer = summary_writer or SummaryWriter()
        self.task_orchestrator = task_orchestrator or TaskOrchestrationService()
        self.task_executor = task_executor or LocalMockTaskExecutor()
        self.adapter_task_executor = adapter_task_executor or AdapterTaskExecutor(registry=build_builtin_task_adapter_registry())
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
            raise TypeError('LocalExperimentRunner requires EngineExperimentConfig')
        self._require_new_experiment_id(experiment_id=experiment_id, replace_existing=replace_existing)

        workspace_dir = Path(config.runtime.workspace) / experiment_id
        result_dir = Path(config.runtime.experiment_root) / experiment_id
        if replace_existing:
            self.storage.delete_metrics(experiment_id=experiment_id)
            self._reset_run_directory(path=workspace_dir, root=Path(config.runtime.workspace))
            self._reset_run_directory(path=result_dir, root=Path(config.runtime.experiment_root))
        workspace_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = workspace_dir / 'config.json'
        command_path = workspace_dir / 'command.json'
        task_plan_path = workspace_dir / 'task_plan.json'
        command = self._build_command(config=config, result_dir=result_dir)
        task_plan = self.task_orchestrator.build_plan(
            config=config,
            experiment_id=experiment_id,
            result_dir=result_dir,
        )

        config_snapshot_path.write_text(
            data=json.dumps(self.config_loader.to_dict(config=config), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        command_path.write_text(
            data=json.dumps({'command': command}, ensure_ascii=False, indent=2),
            encoding='utf-8',
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
                runner_type=config.runtime.runner,
                config_path=str(config_snapshot_path),
                result_path=str(result_dir),
                created_at=self._timestamp(),
                memo=config.experiment.description,
                metadata={
                    'workspace_dir': str(workspace_dir),
                    'command_path': str(command_path),
                    'command': command,
                    'task_plan_path': str(task_plan_path),
                    'task_plan': task_plan.to_dict(),
                    'tags': config.experiment.tags,
                },
            ),
        )

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.PENDING,
            workspace_dir=workspace_dir,
            message='experiment prepared',
            metadata={
                'result_dir': str(result_dir),
                'command': command,
                'task_plan_path': str(task_plan_path),
                'task_plan': task_plan.to_dict(),
            },
        )

    def run(self, experiment_id: str) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        config = self._load_config(record=record)
        workspace_dir = self._workspace_dir(record=record)
        result_dir = self._result_dir(record=record)
        command = self._build_command(config=config, result_dir=result_dir)
        runner_stdout_path = workspace_dir / 'runner_stdout.log'
        started_at = self._timestamp()

        self.storage.update_experiment_status(
            experiment_id=experiment_id,
            status=ExperimentStatus.RUNNING,
            started_at=started_at,
        )

        try:
            with runner_stdout_path.open(mode='w', encoding='utf-8') as stdout_file:
                completed = subprocess.run(
                    command,
                    cwd=config.code.working_dir or workspace_dir,
                    env={**os.environ, **config.train.env},
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    timeout=config.train.max_seconds,
                    check=False,
                    text=True,
                )
        except subprocess.TimeoutExpired:
            return self._mark_failed(
                record=record,
                workspace_dir=workspace_dir,
                error_type='timeout',
                message='experiment timed out',
            )

        marker_status = self._read_marker_status(config=config, result_dir=result_dir)
        status = self._status_from_marker(marker_status=marker_status, return_code=completed.returncode)
        error_type = None if status != ExperimentStatus.FAILED else self._error_type_from_logs(config=config, result_dir=result_dir)
        task_execution = None
        task_results_path = None
        if status != ExperimentStatus.FAILED:
            task_plan = self.task_orchestrator.build_plan(
                config=config,
                experiment_id=experiment_id,
                result_dir=result_dir,
            )
            task_executor = self._task_executor_for_config(config=config)
            task_execution = task_executor.execute_plan(plan=task_plan)
            task_results_path = task_executor.write_plan_result(result=task_execution, result_dir=result_dir)
            if not task_execution.success:
                status = ExperimentStatus.FAILED
                error_type = 'task_execution_failed'

        ended_at = self._timestamp()
        updated_record = replace(
            self._require_record(experiment_id=experiment_id),
            status=status,
            finished_at=ended_at,
            error_type=error_type,
            metadata={
                **self._require_record(experiment_id=experiment_id).metadata,
                'return_code': completed.returncode,
                'marker_status': marker_status,
                'runner_stdout_path': str(runner_stdout_path),
                'task_execution': task_execution.to_dict() if task_execution is not None else None,
                'task_results_path': str(task_results_path) if task_results_path is not None else None,
            },
        )
        self.storage.save_experiment(record=updated_record)

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=workspace_dir,
            message='experiment finished' if status == ExperimentStatus.FINISHED else 'experiment failed',
            metadata={
                'result_dir': str(result_dir),
                'return_code': completed.returncode,
                'marker_status': marker_status,
                'error_type': error_type,
                'task_execution': task_execution.to_dict() if task_execution is not None else None,
                'task_results_path': str(task_results_path) if task_results_path is not None else None,
            },
        )

    def status(self, experiment_id: str) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        config = self._load_config(record=record)
        marker_status = self._read_marker_status(config=config, result_dir=self._result_dir(record=record))
        marker_record_status = self._status_from_marker(marker_status=marker_status, return_code=0) if marker_status else record.status
        status = self._preserve_record_terminal_status(
            record_status=record.status,
            marker_status=marker_record_status,
        )

        if status != record.status:
            self.storage.update_experiment_status(
                experiment_id=experiment_id,
                status=status,
                finished_at=self._timestamp() if status.is_terminal else None,
            )

        return ExperimentRunnerResult(
            experiment_id=experiment_id,
            status=status,
            workspace_dir=self._workspace_dir(record=record),
            message='status loaded',
            metadata={
                'marker_status': marker_status,
                'result_dir': str(self._result_dir(record=record)),
                'workspace_dir': str(self._workspace_dir(record=record)),
                'next_action': self._next_action_for_status(status=status),
            },
        )

    def logs(self, experiment_id: str, tail: int | None = None) -> str:
        record = self._require_record(experiment_id=experiment_id)
        config = self._load_config(record=record)
        log_path = self._result_dir(record=record) / config.output.log_file

        if not log_path.exists():
            fallback_path = self._workspace_dir(record=record) / 'runner_stdout.log'
            log_path = fallback_path

        if not log_path.exists():
            return ''

        return read_text_tail(path=log_path, line_count=tail)

    def collect(self, experiment_id: str, output_dir: Path | None = None) -> ExperimentRunnerResult:
        record = self._require_record(experiment_id=experiment_id)
        config = self._load_config(record=record)
        workspace_dir = self._workspace_dir(record=record)
        result_dir = self._result_dir(record=record)

        return self.result_finalizer.finalize(
            record=record,
            config=config,
            workspace_dir=workspace_dir,
            result_dir=result_dir,
            output_dir=output_dir,
        )

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

    def _build_command(self, config: EngineExperimentConfig, result_dir: Path) -> list[str]:
        return [
            config.runtime.python_executable,
            config.code.entrypoint,
            '--output-dir',
            str(result_dir),
            *build_mock_output_args(config=config),
            *config.code.args,
            *config.train.args,
        ]

    def _task_executor_for_config(self, config: EngineExperimentConfig) -> object:
        for task in config.tasks:
            if task.adapter is None:
                continue
            if self.adapter_task_executor.registry.has(task_type=task.task_type, adapter_key=task.adapter):
                return self.adapter_task_executor

        return self.task_executor

    def _load_config(self, record: ExperimentRecord) -> EngineExperimentConfig:
        if record.config_path is None:
            raise ValueError(f'experiment {record.experiment_id} has no config snapshot')

        data = json.loads(Path(record.config_path).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('config snapshot must contain a mapping object')

        return self.config_loader.from_dict(data=data)

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

        shutil.rmtree(resolved_path)

    def _workspace_dir(self, record: ExperimentRecord) -> Path:
        workspace_dir = record.metadata.get('workspace_dir')
        if not isinstance(workspace_dir, str):
            raise ValueError(f'experiment {record.experiment_id} has no workspace_dir metadata')

        return Path(workspace_dir)

    def _result_dir(self, record: ExperimentRecord) -> Path:
        if record.result_path is None:
            raise ValueError(f'experiment {record.experiment_id} has no result_path')

        return Path(record.result_path)

    def _read_marker_status(self, config: EngineExperimentConfig, result_dir: Path) -> str | None:
        marker_path = result_dir / config.output.status_file
        if not marker_path.exists():
            return None

        return marker_path.read_text(encoding='utf-8').strip()

    def _status_from_marker(self, marker_status: str | None, return_code: int) -> ExperimentStatus:
        if marker_status == 'failed':
            return ExperimentStatus.FAILED
        if marker_status in {'finished', 'finished_with_warning'}:
            return ExperimentStatus.FINISHED
        if marker_status == 'collected':
            return ExperimentStatus.COLLECTED
        if return_code != 0:
            return ExperimentStatus.FAILED

        return ExperimentStatus.FINISHED

    def _preserve_record_terminal_status(
        self,
        record_status: ExperimentStatus,
        marker_status: ExperimentStatus,
    ) -> ExperimentStatus:
        if record_status == ExperimentStatus.COLLECTED and marker_status != ExperimentStatus.FAILED:
            return record_status
        if record_status == ExperimentStatus.FAILED and marker_status != ExperimentStatus.COLLECTED:
            return record_status

        return marker_status

    def _error_type_from_logs(self, config: EngineExperimentConfig, result_dir: Path) -> str:
        log_path = result_dir / config.output.log_file
        if not log_path.exists():
            return 'unknown_error'

        log_text = log_path.read_text(encoding='utf-8')
        if 'CUDA out of memory' in log_text:
            return 'cuda_oom'
        if 'dataset path not found' in log_text:
            return 'dataset_path_error'
        if 'No module named' in log_text:
            return 'dependency_error'

        return 'unknown_error'

    def _next_action_for_status(self, status: ExperimentStatus) -> str:
        if status == ExperimentStatus.PENDING:
            return 'run'
        if status == ExperimentStatus.RUNNING:
            return 'logs/status'
        if status == ExperimentStatus.FINISHED:
            return 'collect'
        if status == ExperimentStatus.COLLECTED:
            return 'done'
        if status == ExperimentStatus.FAILED:
            return 'logs'

        return ''

    def _timestamp(self) -> str:
        return self.clock().isoformat()
