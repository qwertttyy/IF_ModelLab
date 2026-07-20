import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_CONFIG_PATH = 'configs/engine/local_mock.yaml'
DEFAULT_DB_PATH = 'runs/ironflow_experiments.sqlite3'
DEFAULT_EXPERIMENT_ID = 'exp_gui_local_mock'
DEFAULT_TEST_MODEL_CONFIG_PATH = 'configs/engine/local_test_model_smoke.yaml'
DEFAULT_TEST_MODEL_DB_PATH = 'runs/test_model_experiments.sqlite3'
DEFAULT_TEST_MODEL_EXPERIMENT_ID = 'exp_gui_local_test_model'
DEFAULT_CPU_RULE_CONFIG_PATH = 'configs/engine/local_cpu_rule_vision_smoke.yaml'
DEFAULT_CPU_RULE_DB_PATH = 'runs/cpu_rule_vision_experiments.sqlite3'
DEFAULT_CPU_RULE_EXPERIMENT_ID = 'exp_gui_local_cpu_rule'
DEFAULT_K2_LEOPARD2_CONFIG_PATH = 'configs/engine/local_k2_leopard2_smoke.yaml'
DEFAULT_K2_LEOPARD2_DB_PATH = 'runs/k2_leopard2_experiments.sqlite3'
DEFAULT_K2_LEOPARD2_EXPERIMENT_ID = 'exp_gui_local_k2_leopard2'
DEFAULT_ADAPTER_CHAIN_CONFIG_PATH = 'configs/engine/local_adapter_chain_smoke.yaml'
DEFAULT_ADAPTER_CHAIN_DB_PATH = 'runs/adapter_chain_experiments.sqlite3'
DEFAULT_ADAPTER_CHAIN_EXPERIMENT_ID = 'exp_gui_local_adapter_chain'
DEFAULT_YOLO_TORCHVISION_CHAIN_CONFIG_PATH = 'configs/engine/local_yolo_torchvision_chain_smoke.yaml'
DEFAULT_YOLO_TORCHVISION_CHAIN_DB_PATH = 'runs/yolo_torchvision_chain_experiments.sqlite3'
DEFAULT_YOLO_TORCHVISION_CHAIN_EXPERIMENT_ID = 'exp_gui_local_yolo_torchvision_chain'
DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_CONFIG_PATH = 'configs/engine/local_yolo_torchvision_bytetrack_chain_smoke.yaml'
DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_DB_PATH = 'runs/yolo_torchvision_bytetrack_chain_experiments.sqlite3'
DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_EXPERIMENT_ID = 'exp_gui_local_yolo_torchvision_bytetrack_chain'
DEFAULT_SEGMENTATION_SMOKE_CONFIG_PATH = 'configs/engine/local_segmentation_smoke.yaml'
DEFAULT_SEGMENTATION_SMOKE_DB_PATH = 'runs/segmentation_smoke_experiments.sqlite3'
DEFAULT_SEGMENTATION_SMOKE_EXPERIMENT_ID = 'exp_gui_local_segmentation_smoke'
DEFAULT_PREPARED_DETECTION_GATE_CONFIG_PATH = 'configs/engine/local_prepared_detection_gate.yaml'
DEFAULT_PREPARED_DETECTION_GATE_DB_PATH = 'runs/prepared_detection_gate_experiments.sqlite3'
DEFAULT_PREPARED_DETECTION_GATE_EXPERIMENT_ID = 'exp_gui_local_prepared_detection_gate'
DEFAULT_PREPARED_CLASSIFICATION_GATE_CONFIG_PATH = 'configs/engine/local_prepared_classification_gate.yaml'
DEFAULT_PREPARED_CLASSIFICATION_GATE_DB_PATH = 'runs/prepared_classification_gate_experiments.sqlite3'
DEFAULT_PREPARED_CLASSIFICATION_GATE_EXPERIMENT_ID = 'exp_gui_local_prepared_classification_gate'
DEFAULT_PREPARED_TRACKING_GATE_CONFIG_PATH = 'configs/engine/local_prepared_tracking_gate.yaml'
DEFAULT_PREPARED_TRACKING_GATE_DB_PATH = 'runs/prepared_tracking_gate_experiments.sqlite3'
DEFAULT_PREPARED_TRACKING_GATE_EXPERIMENT_ID = 'exp_gui_local_prepared_tracking_gate'
DEFAULT_PREPARED_SEGMENTATION_GATE_CONFIG_PATH = 'configs/engine/local_prepared_segmentation_gate.yaml'
DEFAULT_PREPARED_SEGMENTATION_GATE_DB_PATH = 'runs/prepared_segmentation_gate_experiments.sqlite3'
DEFAULT_PREPARED_SEGMENTATION_GATE_EXPERIMENT_ID = 'exp_gui_local_prepared_segmentation_gate'
DEFAULT_PREPARED_EMBEDDING_GATE_CONFIG_PATH = 'configs/engine/local_prepared_embedding_gate.yaml'
DEFAULT_PREPARED_EMBEDDING_GATE_DB_PATH = 'runs/prepared_embedding_gate_experiments.sqlite3'
DEFAULT_PREPARED_EMBEDDING_GATE_EXPERIMENT_ID = 'exp_gui_local_prepared_embedding_gate'
DEFAULT_CROP_AUGMENTATION_CONFIG_PATH = 'configs/engine/local_classification_crop_augmentation_smoke.yaml'
DEFAULT_CROP_AUGMENTATION_DB_PATH = 'runs/classification_crop_augmentation_experiments.sqlite3'
DEFAULT_CROP_AUGMENTATION_EXPERIMENT_ID = 'exp_gui_local_crop_augmentation'
DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_CONFIG_PATH = 'configs/engine/local_albumentations_classification_crop_augmentation_smoke.yaml'
DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_DB_PATH = 'runs/albumentations_classification_crop_augmentation_experiments.sqlite3'
DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_EXPERIMENT_ID = 'exp_gui_local_albumentations_crop_augmentation'
DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_CONFIG_PATH = 'configs/engine/local_albumentations_detection_bbox_augmentation_smoke.yaml'
DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_DB_PATH = 'runs/albumentations_detection_bbox_augmentation_experiments.sqlite3'
DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_EXPERIMENT_ID = 'exp_gui_local_albumentations_bbox_augmentation'
DEFAULT_LOG_TAIL = 80


@dataclass(frozen=True, slots=True)
class EngineCliResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0

    def status_text(self) -> str:
        summary = self.summary()
        if summary is None:
            return 'done' if self.is_success else 'failed'

        return summary.status_text

    def action_targets(self) -> 'EngineCliActionTargets':
        data = self._stdout_json()
        if data is None:
            return EngineCliActionTargets()

        return _action_targets(command=self.command, data=data)

    def display_text(self) -> str:
        command_text = ' '.join(self.command)
        parts = [
            f'$ {command_text}',
            f'exit_code: {self.exit_code}',
        ]
        summary = self.summary()
        stdout_json = self._stdout_json()
        if summary is not None:
            parts.extend(['', '[summary]', *summary.lines()])
        if stdout_json is not None:
            parts.extend(['', '[raw json hidden]', 'raw_json: parsed into summary'])
        elif self.stdout.strip():
            parts.extend(['', self.stdout.rstrip()])
        if self.stderr.strip():
            parts.extend(['', '[stderr]', self.stderr.rstrip()])

        return '\n'.join(parts)

    def summary(self) -> 'EngineCliDisplaySummary | None':
        data = self._stdout_json()
        if data is None:
            return _plain_text_summary(result=self)

        return _json_summary(result=self, data=data)

    def _stdout_json(self) -> dict[str, Any] | None:
        text = self.stdout.strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        return parsed


@dataclass(frozen=True, slots=True)
class EngineCliDisplaySummary:
    title: str
    status: str | None = None
    next_action: str | None = None
    details: tuple[str, ...] = ()

    @property
    def status_text(self) -> str:
        base = self.title
        if self.status:
            base = f'{base}: {self.status}'
        if self.next_action:
            base = f'{base} | {self.next_action}'

        return base

    def lines(self) -> list[str]:
        lines = [self.title]
        if self.status:
            lines.append(f'status: {self.status}')
        lines.extend(self.details)
        if self.next_action:
            lines.append(f'next: {self.next_action}')

        return lines


@dataclass(frozen=True, slots=True)
class EngineCliActionTargets:
    result_dir: str | None = None
    preview_dir: str | None = None
    summary_path: str | None = None
    prediction_paths: tuple[str, ...] = ()


class LocalMockCommandBuilder:
    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    def run(self, config_path: str, experiment_id: str, replace_existing: bool = False) -> list[str]:
        command = [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'run',
            '--config',
            config_path,
            '--experiment-id',
            experiment_id,
        ]
        if replace_existing:
            command.append('--replace-existing')
        command.append('--json')

        return command

    def status(self, experiment_id: str, db_path: str) -> list[str]:
        return [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'status',
            experiment_id,
            '--db',
            db_path,
            '--json',
        ]

    def logs(self, experiment_id: str, db_path: str, tail: int = DEFAULT_LOG_TAIL) -> list[str]:
        return [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'logs',
            experiment_id,
            '--db',
            db_path,
            '--tail',
            str(tail),
        ]

    def collect(self, experiment_id: str, db_path: str) -> list[str]:
        return [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'collect',
            experiment_id,
            '--db',
            db_path,
            '--json',
        ]


class EngineCliRunner:
    def __init__(self, cwd: Path | str | None = None) -> None:
        self.cwd = None if cwd is None else Path(cwd)
        self._lock = threading.Lock()
        self._current_process: subprocess.Popen[str] | None = None

    def run(self, command: list[str], timeout_seconds: int | None = None) -> EngineCliResult:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._lock:
                self._current_process = process
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            if process is not None:
                process.kill()
                stdout, stderr = process.communicate()
            else:
                stdout = _stream_to_text(error.stdout)
                stderr = _stream_to_text(error.stderr)
            stderr = stderr or 'engine command timed out'
            return EngineCliResult(command=command, exit_code=124, stdout=stdout, stderr=stderr)
        finally:
            with self._lock:
                if self._current_process is process:
                    self._current_process = None

        return EngineCliResult(
            command=command,
            exit_code=process.returncode if process is not None else 1,
            stdout=stdout,
            stderr=stderr,
        )

    def run_streaming(
        self,
        command: list[str],
        timeout_seconds: int | None = None,
        on_stdout_line: Callable[[str], None] | None = None,
        on_stderr_line: Callable[[str], None] | None = None,
    ) -> EngineCliResult:
        process: subprocess.Popen[str] | None = None
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        try:
            process = subprocess.Popen(
                command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._current_process = process
            stdout_thread = threading.Thread(
                target=self._read_stream_lines,
                args=(process.stdout, stdout_lines, on_stdout_line),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._read_stream_lines,
                args=(process.stderr, stderr_lines, on_stderr_line),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            exit_code = self._wait_for_process(process=process, timeout_seconds=timeout_seconds)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.wait()
                exit_code = 124
                if stdout_thread is not None:
                    stdout_thread.join(timeout=5)
                if stderr_thread is not None:
                    stderr_thread.join(timeout=5)
            else:
                exit_code = 124
            if not stderr_lines:
                stderr_lines.append('engine command timed out')
            return EngineCliResult(
                command=command,
                exit_code=exit_code,
                stdout=''.join(stdout_lines),
                stderr=''.join(stderr_lines),
            )
        finally:
            with self._lock:
                if self._current_process is process:
                    self._current_process = None

        return EngineCliResult(
            command=command,
            exit_code=exit_code if process is not None else 1,
            stdout=''.join(stdout_lines),
            stderr=''.join(stderr_lines),
        )

    def _read_stream_lines(
        self,
        stream: Any,
        sink: list[str],
        callback: Callable[[str], None] | None,
    ) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ''):
            sink.append(line)
            if callback is not None:
                callback(line)
        stream.close()

    def _wait_for_process(self, *, process: subprocess.Popen[str], timeout_seconds: int | None) -> int:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            return_code = process.poll()
            if return_code is not None:
                return return_code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd=process.args, timeout=timeout_seconds)
            time.sleep(0.05)

    def cancel_current(self) -> bool:
        with self._lock:
            process = self._current_process
        if process is None or process.poll() is not None:
            return False
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return True


def _json_summary(result: EngineCliResult, data: dict[str, Any]) -> EngineCliDisplaySummary:
    context = _command_context(command=result.command)
    success = bool(data.get('success', result.is_success))
    payload = _payload_for_context(context=context, data=data)
    if not isinstance(payload, dict):
        payload = {}
    experiment = data.get('experiment', {})
    if not isinstance(experiment, dict):
        experiment = {}
    server = data.get('server', {})
    if not isinstance(server, dict):
        server = {}

    status = _first_text(
        payload.get('status'),
        experiment.get('status'),
        payload.get('message') if context in {'server_check', 'server_dependency_check'} else None,
    )
    title = _title_for_context(context=context, success=success)
    details = _details_for_payload(payload=payload, experiment=experiment, server=server, data=data)
    next_action = _next_action(context=context, status=status, success=success, data=data)

    return EngineCliDisplaySummary(
        title=title,
        status=status,
        next_action=next_action,
        details=tuple(details),
    )


def _action_targets(command: list[str], data: dict[str, Any]) -> EngineCliActionTargets:
    context = _command_context(command=command)
    payload = _payload_for_context(context=context, data=data)
    if not isinstance(payload, dict):
        payload = {}
    experiment = data.get('experiment', {})
    if not isinstance(experiment, dict):
        experiment = {}

    result_dir = _first_text(
        _metadata_value(payload=payload, key='result_dir'),
        experiment.get('result_path'),
    )
    found_files = _metadata_list(payload=payload, key='found_files') or []
    prediction_paths = tuple(
        _artifact_path(result_dir=result_dir, relative_path=path)
        for path in _prediction_files(files=found_files)
    )
    preview_dir = None
    preview_files = _preview_files(files=found_files)
    preview_like_files = preview_files or _crop_files(files=found_files)
    preview_parent = _common_parent(paths=preview_like_files) if preview_like_files else None
    if preview_parent is not None:
        preview_dir = _artifact_path(result_dir=result_dir, relative_path=preview_parent)

    summary_path = _first_text(_metadata_value(payload=payload, key='summary_path'))
    if not summary_path and 'summary.md' in [str(path) for path in found_files]:
        summary_path = _artifact_path(result_dir=result_dir, relative_path='summary.md')

    return EngineCliActionTargets(
        result_dir=result_dir,
        preview_dir=preview_dir,
        summary_path=summary_path,
        prediction_paths=prediction_paths,
    )


def _plain_text_summary(result: EngineCliResult) -> EngineCliDisplaySummary:
    context = _command_context(command=result.command)
    if context.endswith('logs') or context == 'logs':
        return EngineCliDisplaySummary(
            title='Logs loaded' if result.is_success else 'Logs failed',
            next_action='If the run is finished, click Collect.' if result.is_success else 'Check the experiment id and DB path.',
        )

    return EngineCliDisplaySummary(
        title='Command completed' if result.is_success else 'Command failed',
        next_action=None if result.is_success else 'Read stderr, fix the issue, then retry the same step.',
    )


def _command_context(command: list[str]) -> str:
    try:
        module_index = command.index('ironflow_exp.engine.cli.main')
    except ValueError:
        return 'command'

    parts = command[module_index + 1:]
    if not parts:
        return 'command'
    if parts[0] == 'server' and len(parts) > 1:
        if parts[1] == 'check' and '--remote-task-adapter-deps' in parts:
            return 'server_dependency_check'
        return f'server_{parts[1]}'
    if parts[0] == 'ssh' and len(parts) > 1:
        return f'ssh_{parts[1]}'

    return parts[0]


def _stage_key(context: str, data: dict[str, Any]) -> str | None:
    if context in {'server_check', 'server_dependency_check'}:
        return 'check'
    if context.startswith('ssh_'):
        key = context.removeprefix('ssh_')
        return key if key in data else None

    return context if context in data else None


def _payload_for_context(context: str, data: dict[str, Any]) -> dict[str, Any]:
    if context == 'ssh_execute':
        stages = data.get('stages', {})
        if isinstance(stages, dict):
            collect = stages.get('collect', {})
            if isinstance(collect, dict):
                return collect

    stage_key = _stage_key(context=context, data=data)
    payload = data.get(stage_key, {}) if stage_key else {}
    if isinstance(payload, dict):
        return payload

    return {}


def _title_for_context(context: str, success: bool) -> str:
    names = {
        'run': 'Local run succeeded',
        'status': 'Local status loaded',
        'logs': 'Local logs loaded',
        'collect': 'Local collect succeeded',
        'server_check': 'SSH server check succeeded',
        'server_dependency_check': 'SSH dependency check succeeded',
        'ssh_prepare': 'SSH prepare succeeded',
        'ssh_bootstrap': 'SSH bootstrap succeeded',
        'ssh_upload': 'SSH upload succeeded',
        'ssh_submit': 'SSH submit succeeded',
        'ssh_status': 'SSH status loaded',
        'ssh_logs': 'SSH logs loaded',
        'ssh_collect': 'SSH collect succeeded',
        'ssh_execute': 'SSH execute succeeded',
    }
    title = names.get(context, 'Command succeeded')
    if success:
        return title

    return title.replace('succeeded', 'failed').replace('loaded', 'failed')


def _details_for_payload(
    payload: dict[str, Any],
    experiment: dict[str, Any],
    server: dict[str, Any],
    data: dict[str, Any],
) -> list[str]:
    details: list[str] = []
    experiment_id = _first_text(payload.get('experiment_id'), experiment.get('experiment_id'))
    server_name = _first_text(server.get('name'), payload.get('server_name'))
    message = _first_text(payload.get('message'))
    result_dir = _first_text(
        _metadata_value(payload=payload, key='result_dir'),
        experiment.get('result_path'),
    )
    collected_dir = _first_text(_metadata_value(payload=payload, key='collected_dir'))
    best_name = _first_text(
        _metadata_value(payload=payload, key='best_metric_name'),
        experiment.get('best_metric_name'),
    )
    best_value = _first_text(
        _metadata_value(payload=payload, key='best_metric_value'),
        experiment.get('best_metric_value'),
    )
    metrics_count = _first_text(_metadata_value(payload=payload, key='metrics_count'))
    metrics_source = _first_text(_metadata_value(payload=payload, key='metrics_source'))
    found_files = _metadata_list(payload=payload, key='found_files')
    missing_files = _metadata_list(payload=payload, key='missing_files')
    task_artifacts = _metadata_list(payload=payload, key='task_artifacts')
    prediction_artifacts = _metadata_list(payload=payload, key='prediction_artifacts')
    augmentation_artifacts = _metadata_list(payload=payload, key='augmentation_artifacts')
    observations = _metadata_list(payload=payload, key='observations')
    summary_written = _metadata_value(payload=payload, key='summary_written')
    db_path = _first_text(data.get('db_path'))

    if experiment_id:
        details.append(f'experiment_id: {experiment_id}')
    if server_name:
        details.append(f'server: {server_name}')
    if message:
        details.append(f'message: {message}')
    if result_dir:
        details.append(f'result_dir: {result_dir}')
    if collected_dir and collected_dir != result_dir:
        details.append(f'collected_dir: {collected_dir}')
    if metrics_count:
        details.append(f'metrics_count: {metrics_count}')
    if metrics_source:
        details.append(f'metrics_source: {metrics_source}')
    if best_name and best_value:
        details.append(f'best_metric: {best_name}={best_value}')
    if found_files:
        details.append(f'found_files: {len(found_files)}')
        if 'model_snapshot.json' in found_files:
            details.append('model_snapshot: model_snapshot.json')
        prediction_files = _prediction_files(files=found_files)
        augmentation_files = _augmentation_files(files=found_files)
        preview_files = _preview_files(files=found_files)
        crop_files = _crop_files(files=found_files)
        if prediction_files:
            details.append(_files_summary(label='prediction_files', files=prediction_files))
        if augmentation_files:
            details.append(_files_summary(label='augmentation_files', files=augmentation_files))
        if preview_files:
            details.append(f'preview_files: {len(preview_files)}')
            preview_dir = _common_parent(paths=preview_files)
            if preview_dir:
                details.append(_artifact_dir_summary(label='preview_dir', result_dir=result_dir, relative_dir=preview_dir))
        if crop_files:
            details.append(f'crop_files: {len(crop_files)}')
            crop_dir = _common_parent(paths=crop_files)
            if crop_dir:
                details.append(_artifact_dir_summary(label='crop_dir', result_dir=result_dir, relative_dir=crop_dir))
    if missing_files is not None:
        details.append(_files_summary(label='missing_files', files=missing_files))
    if task_artifacts:
        details.append(_artifact_validation_summary(label='task_artifacts', artifacts=task_artifacts))
    if prediction_artifacts:
        details.append(_artifact_validation_summary(label='prediction_artifacts', artifacts=prediction_artifacts))
        details.extend(_prediction_artifact_details(artifacts=prediction_artifacts))
    if augmentation_artifacts:
        details.append(_artifact_validation_summary(label='augmentation_artifacts', artifacts=augmentation_artifacts))
        details.extend(_augmentation_artifact_details(artifacts=augmentation_artifacts))
    if isinstance(summary_written, bool):
        details.append(f'summary_written: {str(summary_written).lower()}')
    if observations:
        details.append(f'observations: {len(observations)}')
    details.extend(_ssh_transfer_details(payload=payload))
    details.extend(_dependency_probe_details(payload=payload))
    if db_path:
        details.append(f'db_path: {db_path}')

    return details


def _next_action(context: str, status: str | None, success: bool, data: dict[str, Any]) -> str | None:
    normalized_status = (status or '').lower()
    if not success:
        return _failure_next_action(context=context, data=data)

    if context == 'run':
        return 'Click Collect to save metrics and artifacts.'
    if context == 'collect':
        return 'Done. Review best_metric, prediction_files, preview_dir, result_dir, and logs if needed.'
    if context == 'status':
        return _next_for_experiment_status(status=normalized_status)
    if context == 'server_check':
        return 'Click Deps before Execute for WSL real-adapter scenarios, or Prepare for staged diagnosis.'
    if context == 'server_dependency_check':
        return 'Click Execute to run the selected WSL scenario.'
    if context == 'ssh_prepare':
        return 'Click Bootstrap.'
    if context == 'ssh_bootstrap':
        return 'Click Upload.'
    if context == 'ssh_upload':
        return 'Click Submit.'
    if context == 'ssh_submit':
        return 'Click Status, and use Logs while the job is running.'
    if context == 'ssh_status':
        return _next_for_experiment_status(status=normalized_status, ssh=True)
    if context == 'ssh_collect':
        return 'Done. Review best_metric, result_dir, and downloaded artifacts.'
    if context == 'ssh_execute':
        return 'Done. Review best_metric, result_dir, and downloaded artifacts.'
    if context.endswith('logs'):
        return 'If the job is finished, click Collect.'

    return None


def _failure_next_action(context: str, data: dict[str, Any]) -> str:
    error_type = _first_text(
        _nested(data, 'experiment', 'error_type'),
        _nested(data, 'collect', 'metadata', 'error_type'),
        _nested(data, 'status', 'metadata', 'error_type'),
    )
    if error_type == 'metric_missing':
        return 'Open Logs, confirm metrics.json or metrics.csv exists, then rerun Collect.'
    if context == 'server_check':
        return 'Check WSL ssh service, key path, host, port, and username, then run Check again.'
    if context == 'server_dependency_check':
        return 'Install or fix WSL Python dependencies, then run Deps again.'
    if context.startswith('ssh_'):
        return 'Open Logs or Status, fix the failing SSH stage, then retry that stage.'

    return 'Open Logs, fix the reported error, then retry.'


def _next_for_experiment_status(status: str, ssh: bool = False) -> str | None:
    if status == 'pending':
        return 'Continue with Bootstrap.' if ssh else 'Run the experiment.'
    if status == 'running':
        return 'Use Logs while it runs, then click Status again.'
    if status == 'finished':
        return 'Click Collect.'
    if status == 'collected':
        return 'Done. Review result_dir and best_metric.'
    if status in {'failed', 'cancelled'}:
        return 'Open Logs, fix the issue, then rerun the failed step.'

    return None


def _metadata_value(payload: dict[str, Any], key: str) -> object:
    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        return None

    return metadata.get(key)


def _metadata_list(payload: dict[str, Any], key: str) -> list[Any] | None:
    value = _metadata_value(payload=payload, key=key)
    if isinstance(value, list):
        return value

    return None


def _ssh_transfer_details(payload: dict[str, Any]) -> list[str]:
    transfer_results = payload.get('transfer_results')
    if not isinstance(transfer_results, list):
        return []

    details: list[str] = []
    for index, transfer in enumerate(transfer_results, start=1):
        if not isinstance(transfer, dict):
            continue
        kind = _first_text(transfer.get('kind'))
        operation = _first_text(transfer.get('operation'))
        if operation != 'upload' or kind != 'directory':
            continue
        metadata = transfer.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        cache_hit = metadata.get('upload_cache_hit')
        size_mb = _size_mb(metadata.get('local_size_bytes'))
        file_count = _first_text(metadata.get('local_file_count'))
        remote_path = _first_text(transfer.get('remote_path'))
        parts = [f'item={index}']
        if size_mb is not None:
            parts.append(f'size={size_mb:.1f}MB')
        if file_count is not None:
            parts.append(f'files={file_count}')
        if isinstance(cache_hit, bool):
            parts.append(f'cache_hit={str(cache_hit).lower()}')
        timings = metadata.get('timings_seconds')
        if isinstance(timings, dict) and timings:
            timing_parts = [
                f'{key}={value}s'
                for key, value in timings.items()
                if isinstance(key, str) and isinstance(value, int | float)
            ]
            if timing_parts:
                parts.append('timing: ' + ', '.join(timing_parts))
        if remote_path is not None:
            parts.append(f'target={remote_path}')
        details.append('upload_directory: ' + '; '.join(parts))

    return details


def _size_mb(value: object) -> float | None:
    try:
        return float(value) / (1024 * 1024)
    except (TypeError, ValueError):
        return None


def _dependency_probe_details(payload: dict[str, Any]) -> list[str]:
    probe = _metadata_value(payload=payload, key='probe')
    if not isinstance(probe, dict):
        return []
    checks = probe.get('checks', {})
    if not isinstance(checks, dict):
        return []

    details: list[str] = []
    profile = _first_text(probe.get('profile'))
    if profile:
        details.append(f'dependency_profile: {profile}')
    ok_count = sum(
        1
        for value in checks.values()
        if isinstance(value, dict) and bool(value.get('ok'))
    )
    details.append(f'dependency_probe: {ok_count}/{len(checks)} ok')

    version_parts = []
    for name in ['torch', 'torchvision', 'ultralytics', 'albumentations', 'cv2', 'PIL', 'yaml']:
        value = checks.get(name, {})
        if isinstance(value, dict) and value.get('ok'):
            version = _first_text(value.get('version'))
            if version:
                version_parts.append(f'{name}={version}')
    if version_parts:
        details.append(f'dependency_versions: {", ".join(version_parts)}')

    yolo = checks.get('ultralytics_yolo', {})
    if isinstance(yolo, dict) and yolo.get('ok'):
        details.append('ultralytics_yolo: ok')

    failures = []
    for name, value in checks.items():
        if isinstance(value, dict) and not bool(value.get('ok')):
            error = _first_text(value.get('error')) or 'unknown error'
            failures.append(f'{name}: {error}')
    if failures:
        details.append(_files_summary(label='dependency_failures', files=failures))

    return details


def _files_summary(label: str, files: list[Any]) -> str:
    if not files:
        return f'{label}: none'
    file_names = [str(file) for file in files[:3]]
    suffix = '' if len(files) <= 3 else f' (+{len(files) - 3} more)'

    return f"{label}: {', '.join(file_names)}{suffix}"


def _matching_paths(files: list[Any], prefix: str) -> list[str]:
    return [
        str(file)
        for file in files
        if str(file).replace('\\', '/').startswith(prefix)
    ]


def _prediction_files(files: list[Any]) -> list[str]:
    root_predictions = _matching_paths(files=files, prefix='predictions/')
    nested_predictions = _paths_with_segment(files=files, segment='/predictions/')

    return _dedupe_paths([*root_predictions, *nested_predictions])


def _preview_files(files: list[Any]) -> list[str]:
    root_previews = _matching_paths(files=files, prefix='previews/')
    nested_previews = _paths_with_segment(files=files, segment='/previews/')

    return _dedupe_paths([*root_previews, *nested_previews])


def _crop_files(files: list[Any]) -> list[str]:
    return _paths_with_segment(files=files, segment='/crops/')


def _augmentation_files(files: list[Any]) -> list[str]:
    return [
        str(file)
        for file in files
        if str(file).replace('\\', '/').endswith('/augmentation_manifest.json')
        or str(file).replace('\\', '/') == 'augmentation_manifest.json'
    ]


def _paths_with_segment(files: list[Any], segment: str) -> list[str]:
    matched: list[str] = []
    for file in files:
        path = str(file)
        normalized = '/' + path.replace('\\', '/')
        if segment in normalized:
            matched.append(path)

    return matched


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        normalized = path.replace('\\', '/')
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)

    return deduped


def _common_parent(paths: list[str]) -> str | None:
    parents = {
        str(Path(path.replace('\\', '/')).parent).replace('\\', '/')
        for path in paths
    }
    if len(parents) != 1:
        return None

    parent = next(iter(parents))
    return None if parent == '.' else parent


def _artifact_dir_summary(label: str, result_dir: str | None, relative_dir: str) -> str:
    if not result_dir:
        return f'{label}: {relative_dir}'

    return f'{label}: {Path(result_dir) / relative_dir}'


def _artifact_path(result_dir: str | None, relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute() or not result_dir:
        return str(path)

    return str(Path(result_dir) / path)


def _artifact_validation_summary(label: str, artifacts: list[Any]) -> str:
    dict_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    total_count = len(dict_artifacts)
    valid_count = sum(1 for artifact in dict_artifacts if artifact.get('is_valid') is True)
    invalid_count = total_count - valid_count
    if invalid_count == 0:
        return f'{label}: {valid_count}/{total_count} valid'

    invalid_names = [
        _first_text(artifact.get('path'), artifact.get('task_dir'), artifact.get('task'))
        for artifact in dict_artifacts
        if artifact.get('is_valid') is not True
    ]
    invalid_text = ', '.join(name for name in invalid_names if name)

    return f'{label}: {valid_count}/{total_count} valid, invalid={invalid_text}'


def _prediction_artifact_details(artifacts: list[Any]) -> list[str]:
    details: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = _first_text(artifact.get('path')) or 'unknown'
        task = _first_text(artifact.get('task')) or 'unknown'
        record_count = _first_text(artifact.get('record_count'))
        track_count = _first_text(artifact.get('track_count'))
        frame_range = _first_text(artifact.get('frame_range'))
        sample_range = _first_text(artifact.get('sample_range'))
        parts = [f'task={task}']
        if record_count is not None:
            parts.append(f'records={record_count}')
        if track_count is not None:
            parts.append(f'tracks={track_count}')
        if frame_range:
            parts.append(f'frames={frame_range}')
        if sample_range:
            parts.append(f'samples={sample_range}')
        details.append(f'prediction_artifact: {path} [{", ".join(parts)}]')

    return details


def _augmentation_artifact_details(artifacts: list[Any]) -> list[str]:
    details: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = _first_text(artifact.get('path')) or 'unknown'
        parts = []
        for key, label in [
            ('policy_id', 'policy'),
            ('target_split', 'target_split'),
            ('augmentation_count', 'augmentations'),
            ('record_count', 'records'),
            ('label_transform_count', 'label_transforms'),
            ('bbox_transform_count', 'bbox_transforms'),
            ('bbox_drop_count', 'bbox_drops'),
            ('dropped_object_count', 'dropped_objects'),
            ('mask_record_transform_count', 'mask_record_transforms'),
            ('mask_object_transform_count', 'mask_object_transforms'),
            ('mask_transform_count', 'mask_transform_refs'),
            ('mask_artifact_count', 'mask_artifacts'),
        ]:
            value = _first_text(artifact.get(key))
            if value is not None and value != '0':
                parts.append(f'{label}={value}')
        details.append(f'augmentation_artifact: {path} [{", ".join(parts)}]')

    return details


def _nested(data: dict[str, Any], *keys: str) -> object:
    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def _first_text(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    return None


def _stream_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')

    return value
