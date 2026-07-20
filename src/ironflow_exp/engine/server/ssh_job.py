from __future__ import annotations

import json
import posixpath
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.domain import ExperimentStatus, ServerRecord
from ironflow_exp.engine.server.metadata_redaction import command_result_summary, truncate_text
from ironflow_exp.engine.server.ssh_client import BaseSshClient, SshCommandResult
from ironflow_exp.engine.server.ssh_plan import (
    RemoteTransferItem,
    SshExecutionPlan,
    is_auto_remote_python,
    remote_command_text_from_args,
)


JOB_STATUS_SCHEMA_VERSION = '0.1'
JOB_RUNNER_FILENAME = 'ssh_job_runner.py'
REMOTE_COMMAND_FILENAME = 'remote_command.json'
JOB_JSON_FILENAME = 'job.json'
STATUS_JSON_FILENAME = 'status.json'
STATUS_MARKER_FILENAME = 'status.marker'
REMOTE_STDOUT_FILENAME = 'remote_stdout.log'
REMOTE_STDERR_FILENAME = 'remote_stderr.log'
REMOTE_SUBMIT_STDOUT_FILENAME = 'submit_stdout.log'
REMOTE_SUBMIT_STDERR_FILENAME = 'submit_stderr.log'


@dataclass(frozen=True, slots=True)
class SshJobSupportFiles:
    local_job_runner_path: Path
    local_command_json_path: Path
    remote_job_runner_path: str
    remote_command_json_path: str
    remote_job_json_path: str
    remote_status_json_path: str
    remote_status_marker_path: str
    remote_stdout_path: str
    remote_stderr_path: str
    remote_submit_stdout_path: str
    remote_submit_stderr_path: str

    def upload_items(self) -> list[RemoteTransferItem]:
        return [
            RemoteTransferItem(
                local_path=str(self.local_job_runner_path),
                remote_path=self.remote_job_runner_path,
                kind='file',
            ),
            RemoteTransferItem(
                local_path=str(self.local_command_json_path),
                remote_path=self.remote_command_json_path,
                kind='file',
            ),
        ]

    def to_dict(self) -> dict[str, str]:
        return {
            'local_job_runner_path': str(self.local_job_runner_path),
            'local_command_json_path': str(self.local_command_json_path),
            'remote_job_runner_path': self.remote_job_runner_path,
            'remote_command_json_path': self.remote_command_json_path,
            'remote_job_json_path': self.remote_job_json_path,
            'remote_status_json_path': self.remote_status_json_path,
            'remote_status_marker_path': self.remote_status_marker_path,
            'remote_stdout_path': self.remote_stdout_path,
            'remote_stderr_path': self.remote_stderr_path,
            'remote_submit_stdout_path': self.remote_submit_stdout_path,
            'remote_submit_stderr_path': self.remote_submit_stderr_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> 'SshJobSupportFiles':
        return cls(
            local_job_runner_path=Path(str(data['local_job_runner_path'])),
            local_command_json_path=Path(str(data['local_command_json_path'])),
            remote_job_runner_path=str(data['remote_job_runner_path']),
            remote_command_json_path=str(data['remote_command_json_path']),
            remote_job_json_path=str(data['remote_job_json_path']),
            remote_status_json_path=str(data['remote_status_json_path']),
            remote_status_marker_path=str(data['remote_status_marker_path']),
            remote_stdout_path=str(data['remote_stdout_path']),
            remote_stderr_path=str(data['remote_stderr_path']),
            remote_submit_stdout_path=str(data['remote_submit_stdout_path']),
            remote_submit_stderr_path=str(data['remote_submit_stderr_path']),
        )


@dataclass(frozen=True, slots=True)
class SshJobStatusPoll:
    status_data: dict[str, object]
    experiment_status: ExperimentStatus
    error_type: str | None
    finished_at: str | None
    finished_at_source: str | None = None
    remote_run_metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SshJobStatusPollFailure:
    error_type: str
    message: str
    command_result: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            'error_type': self.error_type,
            'message': self.message,
            'command_result': self.command_result,
        }


def write_job_support_files(
    plan: SshExecutionPlan,
    timeout_seconds: int | None = None,
) -> SshJobSupportFiles:
    local_workspace_dir = Path(plan.local_workspace_dir)
    local_job_runner_path = local_workspace_dir / JOB_RUNNER_FILENAME
    local_command_json_path = local_workspace_dir / REMOTE_COMMAND_FILENAME
    local_job_runner_path.write_text(data=job_runner_script(), encoding='utf-8')
    local_command_json_path.write_text(
        data=json.dumps(
            {
                'command_args': plan.remote_command_args,
                'timeout_seconds': timeout_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    return SshJobSupportFiles(
        local_job_runner_path=local_job_runner_path,
        local_command_json_path=local_command_json_path,
        remote_job_runner_path=remote_join(plan.remote_workspace_dir, JOB_RUNNER_FILENAME),
        remote_command_json_path=remote_join(plan.remote_workspace_dir, REMOTE_COMMAND_FILENAME),
        remote_job_json_path=remote_join(plan.remote_workspace_dir, JOB_JSON_FILENAME),
        remote_status_json_path=remote_join(plan.remote_result_dir, STATUS_JSON_FILENAME),
        remote_status_marker_path=remote_join(plan.remote_result_dir, STATUS_MARKER_FILENAME),
        remote_stdout_path=remote_join(plan.remote_workspace_dir, REMOTE_STDOUT_FILENAME),
        remote_stderr_path=remote_join(plan.remote_workspace_dir, REMOTE_STDERR_FILENAME),
        remote_submit_stdout_path=remote_join(plan.remote_workspace_dir, REMOTE_SUBMIT_STDOUT_FILENAME),
        remote_submit_stderr_path=remote_join(plan.remote_workspace_dir, REMOTE_SUBMIT_STDERR_FILENAME),
    )


def remote_join(*parts: str) -> str:
    normalized_parts = [part.replace('\\', '/') for part in parts if part]
    if not normalized_parts:
        return ''

    return posixpath.normpath(posixpath.join(normalized_parts[0], *normalized_parts[1:]))


def build_submit_command(
    remote_python_executable: str,
    experiment_id: str,
    support_files: SshJobSupportFiles,
) -> str:
    runner_args = [
        remote_python_executable,
        support_files.remote_job_runner_path,
        '--experiment-id',
        experiment_id,
        '--command-json',
        support_files.remote_command_json_path,
        '--job-json',
        support_files.remote_job_json_path,
        '--status-json',
        support_files.remote_status_json_path,
        '--status-marker',
        support_files.remote_status_marker_path,
        '--stdout',
        support_files.remote_stdout_path,
        '--stderr',
        support_files.remote_stderr_path,
    ]
    starter_script = (
        'import json, os, subprocess, sys\n'
        f'args = {json.dumps(runner_args)}\n'
        f'stdout_path = {json.dumps(support_files.remote_submit_stdout_path)}\n'
        f'stderr_path = {json.dumps(support_files.remote_submit_stderr_path)}\n'
        'if args and args[0].strip().lower() in {"auto", "detect", "__ironflow_python_auto__"}:\n'
        '    args[0] = sys.executable\n'
        'os.makedirs(os.path.dirname(stdout_path), exist_ok=True)\n'
        'os.makedirs(os.path.dirname(stderr_path), exist_ok=True)\n'
        'stdout = open(stdout_path, "ab")\n'
        'stderr = open(stderr_path, "ab")\n'
        'try:\n'
        '    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True, start_new_session=True)\n'
        'except TypeError:\n'
        '    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True)\n'
        'print(json.dumps({"pid": process.pid, "args": args}))\n'
    )
    command_args = [remote_python_executable, '-c', starter_script]

    return remote_command_text_from_args(command_args)


def build_status_command(
    remote_python_executable: str,
    support_files: SshJobSupportFiles,
) -> str:
    status_script = (
        'import json, os\n'
        'from pathlib import Path\n'
        f'status_path = Path({json.dumps(support_files.remote_status_json_path)})\n'
        f'job_path = Path({json.dumps(support_files.remote_job_json_path)})\n'
        f'marker_path = Path({json.dumps(support_files.remote_status_marker_path)})\n'
        f'data = {{"schema_version": {json.dumps(JOB_STATUS_SCHEMA_VERSION)}, "status": "unknown"}}\n'
        'if status_path.exists():\n'
        '    data = json.loads(status_path.read_text(encoding="utf-8"))\n'
        'elif marker_path.exists():\n'
        f'    data = {{"schema_version": {json.dumps(JOB_STATUS_SCHEMA_VERSION)}, "status": marker_path.read_text(encoding="utf-8", errors="replace").strip()}}\n'
        'if job_path.exists():\n'
        '    data["job"] = json.loads(job_path.read_text(encoding="utf-8"))\n'
        'pid = data.get("pid") or data.get("job", {}).get("pid")\n'
        'if data.get("status") == "running" and isinstance(pid, int):\n'
        '    try:\n'
        '        os.kill(pid, 0)\n'
        '    except OSError:\n'
        '        data["status"] = "stale"\n'
        '        data["exit_code"] = data.get("exit_code") or 1\n'
        '        data["error_type"] = "stale_process"\n'
        '        data["message"] = "background job process is not alive"\n'
        f'data["schema_version"] = data.get("schema_version") or {json.dumps(JOB_STATUS_SCHEMA_VERSION)}\n'
        'print(json.dumps(data))\n'
    )
    command_args = [remote_python_executable, '-c', status_script]

    return remote_command_text_from_args(command_args)


def build_cancel_command(
    remote_python_executable: str,
    experiment_id: str,
    support_files: SshJobSupportFiles,
    remote_result_dir: str,
) -> str:
    cancel_script = (
        'import json, os, signal, subprocess, time\n'
        'from pathlib import Path\n'
        f'experiment_id = {json.dumps(experiment_id)}\n'
        f'remote_result_dir = {json.dumps(remote_result_dir)}\n'
        f'status_path = Path({json.dumps(support_files.remote_status_json_path)})\n'
        f'job_path = Path({json.dumps(support_files.remote_job_json_path)})\n'
        f'marker_path = Path({json.dumps(support_files.remote_status_marker_path)})\n'
        'data = {"schema_version": "0.1", "status": "unknown"}\n'
        'for path in (status_path, job_path):\n'
        '    if path.exists():\n'
        '        try:\n'
        '            loaded = json.loads(path.read_text(encoding="utf-8"))\n'
        '            if isinstance(loaded, dict):\n'
        '                data.update(loaded)\n'
        '        except Exception:\n'
        '            pass\n'
        'pids = []\n'
        'for key in ("command_pid", "pid"):\n'
        '    value = data.get(key) or data.get("job", {}).get(key)\n'
        '    if isinstance(value, int) and value > 1:\n'
        '        pids.append(value)\n'
        'current_pid = os.getpid()\n'
        'try:\n'
        '    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False)\n'
        '    for line in ps.stdout.splitlines():\n'
        '        stripped = line.strip()\n'
        '        if not stripped:\n'
        '            continue\n'
        '        pid_text, _, args = stripped.partition(" ")\n'
        '        try:\n'
        '            pid = int(pid_text)\n'
        '        except ValueError:\n'
        '            continue\n'
        '        if pid == current_pid or pid <= 1:\n'
        '            continue\n'
        '        if remote_result_dir and remote_result_dir in args:\n'
        '            pids.append(pid)\n'
        '        elif experiment_id and experiment_id in args and "ssh_job_runner.py" in args:\n'
        '            pids.append(pid)\n'
        'except Exception:\n'
        '    pass\n'
        'pids = sorted(set(pids), reverse=True)\n'
        'killed = []\n'
        'for pid in pids:\n'
        '    try:\n'
        '        os.killpg(pid, signal.SIGTERM)\n'
        '        killed.append(pid)\n'
        '    except Exception:\n'
        '        try:\n'
        '            os.kill(pid, signal.SIGTERM)\n'
        '            killed.append(pid)\n'
        '        except Exception:\n'
        '            pass\n'
        'time.sleep(0.5)\n'
        'for pid in pids:\n'
        '    try:\n'
        '        os.kill(pid, 0)\n'
        '    except Exception:\n'
        '        continue\n'
        '    try:\n'
        '        os.killpg(pid, signal.SIGKILL)\n'
        '    except Exception:\n'
        '        try:\n'
        '            os.kill(pid, signal.SIGKILL)\n'
        '        except Exception:\n'
        '            pass\n'
        'cancelled = {**data, "schema_version": "0.1", "status": "cancelled", "exit_code": 130, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error_type": "remote_job_cancelled", "message": "remote job cancelled by user", "cancelled_pids": killed}\n'
        'for path in (status_path, job_path):\n'
        '    try:\n'
        '        path.parent.mkdir(parents=True, exist_ok=True)\n'
        '        path.write_text(json.dumps(cancelled, ensure_ascii=False, indent=2), encoding="utf-8")\n'
        '    except Exception:\n'
        '        pass\n'
        'try:\n'
        '    marker_path.parent.mkdir(parents=True, exist_ok=True)\n'
        '    marker_path.write_text("cancelled\\n", encoding="utf-8")\n'
        'except Exception:\n'
        '    pass\n'
        'print(json.dumps({"status": "cancelled", "killed_pids": killed, "pid_count": len(killed)}))\n'
    )
    command_args = [remote_python_executable, '-c', cancel_script]

    return remote_command_text_from_args(command_args)


class SshBackgroundJobService:
    def write_support_files(
        self,
        plan: SshExecutionPlan,
        timeout_seconds: int | None = None,
    ) -> SshJobSupportFiles:
        return write_job_support_files(plan=plan, timeout_seconds=timeout_seconds)

    def submit(
        self,
        server: ServerRecord,
        config: EngineExperimentConfig,
        experiment_id: str,
        plan: SshExecutionPlan,
        support_files: SshJobSupportFiles,
        client: BaseSshClient,
        timeout_seconds: int | None = None,
    ) -> SshCommandResult:
        submit_command = build_submit_command(
            remote_python_executable=config.runtime.remote_python_executable,
            experiment_id=experiment_id,
            support_files=support_files,
        )
        if server.server_type == 'local_ssh_simulator':
            return self._submit_local_ssh_simulator_job(
                config=config,
                experiment_id=experiment_id,
                support_files=support_files,
                submit_command=submit_command,
            )

        return client.run_command(
            server=server,
            command=submit_command,
            timeout_seconds=timeout_seconds,
        )

    def cancel(
        self,
        server: ServerRecord,
        config: EngineExperimentConfig,
        experiment_id: str,
        plan: SshExecutionPlan,
        support_files: SshJobSupportFiles,
        client: BaseSshClient,
        timeout_seconds: int | None = None,
    ) -> SshCommandResult:
        cancel_command = build_cancel_command(
            remote_python_executable=config.runtime.remote_python_executable,
            experiment_id=experiment_id,
            support_files=support_files,
            remote_result_dir=plan.remote_result_dir,
        )
        return client.run_command(
            server=server,
            command=cancel_command,
            timeout_seconds=timeout_seconds,
        )

    def build_submit_metadata(
        self,
        server: ServerRecord,
        experiment_id: str,
        plan: SshExecutionPlan,
        support_files: SshJobSupportFiles,
        command_result: SshCommandResult,
        submitted_at: str | None,
    ) -> dict[str, object]:
        return {
            'success': command_result.is_success,
            'server_name': server.name,
            'experiment_id': experiment_id,
            'submit_command': truncate_text(command_result.command),
            'submit_command_length': len(command_result.command),
            'submit_command_truncated': truncate_text(command_result.command) != command_result.command,
            'submit_result': self.command_result_summary(command_result=command_result),
            'job_support': support_files.to_dict(),
            'remote_result_dir': plan.remote_result_dir,
            'remote_command_text': plan.remote_command_text,
            'submitted_at': submitted_at if command_result.is_success else None,
        }

    def poll_status(
        self,
        server: ServerRecord,
        config: EngineExperimentConfig,
        experiment_id: str,
        remote_command_text: str,
        support_files: SshJobSupportFiles,
        client: BaseSshClient,
        fallback_status: ExperimentStatus,
        fallback_error_type: str | None,
        fallback_finished_at: str | None,
    ) -> SshJobStatusPoll | SshJobStatusPollFailure:
        command = build_status_command(
            remote_python_executable=config.runtime.remote_python_executable,
            support_files=support_files,
        )
        result = client.run_command(server=server, command=command, timeout_seconds=10)
        if not result.is_success:
            return SshJobStatusPollFailure(
                error_type='ssh_background_status_command_failed',
                message='ssh background status command failed',
                command_result=self.command_result_summary(command_result=result),
            )

        try:
            status_data = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            return SshJobStatusPollFailure(
                error_type='ssh_background_status_invalid_json',
                message=f'ssh background status returned invalid JSON: {error}',
                command_result=self.command_result_summary(command_result=result),
            )

        if not isinstance(status_data, dict):
            return SshJobStatusPollFailure(
                error_type='ssh_background_status_invalid_payload',
                message='ssh background status payload is not a JSON object',
                command_result=self.command_result_summary(command_result=result),
            )

        status_data['schema_version'] = status_data.get('schema_version') or JOB_STATUS_SCHEMA_VERSION
        remote_status = str(status_data.get('status', 'unknown'))
        if remote_status == 'finished':
            finished_at, finished_at_source = self.resolve_finished_at(
                status_data=status_data,
                fallback_finished_at=fallback_finished_at,
            )
            status_data['finished_at_source'] = finished_at_source
            return SshJobStatusPoll(
                status_data=status_data,
                experiment_status=ExperimentStatus.FINISHED,
                error_type=None,
                finished_at=finished_at,
                finished_at_source=finished_at_source,
                remote_run_metadata=self.remote_run_metadata(
                    server=server,
                    experiment_id=experiment_id,
                    remote_command_text=remote_command_text,
                    status_data=status_data,
                    success=True,
                ),
            )
        if remote_status in {'failed', 'stale'}:
            finished_at, finished_at_source = self.resolve_finished_at(
                status_data=status_data,
                fallback_finished_at=fallback_finished_at,
            )
            status_data['finished_at_source'] = finished_at_source
            return SshJobStatusPoll(
                status_data=status_data,
                experiment_status=ExperimentStatus.FAILED,
                error_type=str(status_data.get('error_type') or 'ssh_background_job_failed'),
                finished_at=finished_at,
                finished_at_source=finished_at_source,
                remote_run_metadata=self.remote_run_metadata(
                    server=server,
                    experiment_id=experiment_id,
                    remote_command_text=remote_command_text,
                    status_data=status_data,
                    success=False,
                ),
            )
        if remote_status in {'submitted', 'running'}:
            status_data['finished_at_source'] = None
            return SshJobStatusPoll(
                status_data=status_data,
                experiment_status=ExperimentStatus.RUNNING,
                error_type=None,
                finished_at=None,
                finished_at_source=None,
            )
        if remote_status == 'cancelled':
            finished_at, finished_at_source = self.resolve_finished_at(
                status_data=status_data,
                fallback_finished_at=fallback_finished_at,
            )
            status_data['finished_at_source'] = finished_at_source
            return SshJobStatusPoll(
                status_data=status_data,
                experiment_status=ExperimentStatus.CANCELLED,
                error_type='ssh_background_job_cancelled',
                finished_at=finished_at,
                finished_at_source=finished_at_source,
            )

        return SshJobStatusPoll(
            status_data=status_data,
            experiment_status=fallback_status,
            error_type=fallback_error_type,
            finished_at=fallback_finished_at,
            finished_at_source='existing_record' if fallback_finished_at else None,
        )

    def resolve_finished_at(
        self,
        status_data: dict[str, object],
        fallback_finished_at: str | None,
    ) -> tuple[str, str]:
        remote_finished_at = status_data.get('finished_at')
        if isinstance(remote_finished_at, str) and remote_finished_at.strip():
            return remote_finished_at, 'remote_status'
        if fallback_finished_at is not None and fallback_finished_at.strip():
            return fallback_finished_at, 'poll_fallback'

        return '', 'missing'

    def remote_run_metadata(
        self,
        server: ServerRecord,
        experiment_id: str,
        remote_command_text: str,
        status_data: dict[str, object],
        success: bool,
    ) -> dict[str, object]:
        raw_exit_code = status_data.get('exit_code')
        exit_code = int(raw_exit_code) if raw_exit_code is not None else (0 if success else 1)

        return {
            'server_name': server.name,
            'experiment_id': experiment_id,
            'success': success,
            'message': 'ssh background job finished' if success else 'ssh background job failed',
            'command_result': {
                'command': truncate_text(remote_command_text),
                'command_length': len(remote_command_text),
                'command_truncated': truncate_text(remote_command_text) != remote_command_text,
                'exit_code': exit_code,
                'stdout_tail': '',
                'stderr_tail': '',
                'stdout_length': 0,
                'stderr_length': 0,
            },
        }

    def command_result_summary(self, command_result: SshCommandResult) -> dict[str, object]:
        return command_result_summary(command_result=command_result)

    def _submit_local_ssh_simulator_job(
        self,
        config: EngineExperimentConfig,
        experiment_id: str,
        support_files: SshJobSupportFiles,
        submit_command: str,
    ) -> SshCommandResult:
        self._normalize_local_simulator_command_json(
            command_json_path=Path(support_files.remote_command_json_path),
        )
        runner_python = (
            sys.executable
            if config.runtime.remote_python_executable in {'python', 'python.exe'}
            or is_auto_remote_python(config.runtime.remote_python_executable)
            else config.runtime.remote_python_executable
        )
        runner_args = [
            runner_python,
            support_files.remote_job_runner_path,
            '--experiment-id',
            experiment_id,
            '--command-json',
            support_files.remote_command_json_path,
            '--job-json',
            support_files.remote_job_json_path,
            '--status-json',
            support_files.remote_status_json_path,
            '--status-marker',
            support_files.remote_status_marker_path,
            '--stdout',
            support_files.remote_stdout_path,
            '--stderr',
            support_files.remote_stderr_path,
        ]
        Path(support_files.remote_submit_stdout_path).parent.mkdir(parents=True, exist_ok=True)
        stdout_file = Path(support_files.remote_submit_stdout_path).open('ab')
        stderr_file = Path(support_files.remote_submit_stderr_path).open('ab')
        try:
            process = subprocess.Popen(
                runner_args,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                start_new_session=True,
            )
            stdout_file.close()
            stderr_file.close()
            self._wait_for_local_simulator_job_start_or_short_finish(
                process=process,
                support_files=support_files,
            )
        except Exception as error:
            stdout_file.close()
            stderr_file.close()
            return SshCommandResult(
                command=submit_command,
                exit_code=1,
                stderr=f'{type(error).__name__}: {error}',
            )

        return SshCommandResult(
            command=submit_command,
            exit_code=0,
            stdout=json.dumps({'pid': process.pid, 'args': runner_args}) + '\n',
            stderr='',
        )

    def _wait_for_local_simulator_job_start_or_short_finish(
        self,
        process: subprocess.Popen[bytes],
        support_files: SshJobSupportFiles,
    ) -> None:
        job_json_path = Path(support_files.remote_job_json_path)
        status_json_path = Path(support_files.remote_status_json_path)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if status_json_path.exists():
                try:
                    status_data = json.loads(status_json_path.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    status_data = {}
                if status_data.get('status') in {'finished', 'failed'}:
                    return
            if job_json_path.exists() and process.poll() is None:
                return
            if process.poll() is not None:
                return
            time.sleep(0.05)

    def _normalize_local_simulator_command_json(self, command_json_path: Path) -> None:
        if not command_json_path.exists():
            return

        data = json.loads(command_json_path.read_text(encoding='utf-8'))
        command_args = data.get('command_args')
        if not isinstance(command_args, list) or not command_args:
            return
        if str(command_args[0]) not in {'python', 'python.exe'}:
            return

        data['command_args'] = [sys.executable, *[str(part) for part in command_args[1:]]]
        command_json_path.write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )


def job_runner_script() -> str:
    return '''import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_marker(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(status + '\\n', encoding='utf-8')


def parse_timeout(value):
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None

    return timeout


def terminate_process(process: subprocess.Popen) -> None:
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass

    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--command-json', required=True)
    parser.add_argument('--job-json', required=True)
    parser.add_argument('--status-json', required=True)
    parser.add_argument('--status-marker', required=True)
    parser.add_argument('--stdout', required=True)
    parser.add_argument('--stderr', required=True)
    args = parser.parse_args()

    command_data = json.loads(Path(args.command_json).read_text(encoding='utf-8'))
    command_args = [str(part) for part in command_data['command_args']]
    if command_args and command_args[0].strip().lower() in {'auto', 'detect', '__ironflow_python_auto__'}:
        command_args[0] = sys.executable
    timeout_seconds = parse_timeout(command_data.get('timeout_seconds'))
    job_json_path = Path(args.job_json)
    status_json_path = Path(args.status_json)
    status_marker_path = Path(args.status_marker)
    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    started_at = now()
    base_data = {
        'schema_version': '0.1',
        'experiment_id': args.experiment_id,
        'pid': os.getpid(),
        'command_args': command_args,
        'command_text': ' '.join(command_args),
        'started_at': started_at,
        'stdout_path': str(stdout_path),
        'stderr_path': str(stderr_path),
        'status_json_path': str(status_json_path),
        'status_marker_path': str(status_marker_path),
        'timeout_seconds': timeout_seconds,
    }
    running_data = {
        **base_data,
        'status': 'running',
        'exit_code': None,
        'finished_at': None,
        'error_type': None,
        'message': None,
    }
    write_json(job_json_path, running_data)
    write_json(status_json_path, running_data)
    write_marker(status_marker_path, 'running')

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open('ab') as stdout_file, stderr_path.open('ab') as stderr_file:
        try:
            process = subprocess.Popen(
                command_args,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                start_new_session=True,
            )
        except TypeError:
            process = subprocess.Popen(
                command_args,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
            )

        running_data = {**running_data, 'command_pid': process.pid}
        write_json(job_json_path, running_data)
        write_json(status_json_path, running_data)

        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(process=process)
            exit_code = 124

    finished_at = now()
    final_status = 'finished' if exit_code == 0 and not timed_out else 'failed'
    final_data = {
        **base_data,
        'status': final_status,
        'exit_code': exit_code,
        'finished_at': finished_at,
        'error_type': (
            None
            if final_status == 'finished'
            else 'remote_command_timeout' if timed_out else 'remote_command_failed'
        ),
        'message': (
            'remote command finished'
            if final_status == 'finished'
            else 'remote command timed out' if timed_out else 'remote command failed'
        ),
        'timed_out': timed_out,
    }
    write_json(job_json_path, final_data)
    write_json(status_json_path, final_data)
    write_marker(status_marker_path, final_status)

    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
'''
