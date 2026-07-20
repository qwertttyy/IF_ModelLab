import json
import shlex
from dataclasses import dataclass, field

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.profile_validator import ServerProfileValidator
from ironflow_exp.engine.server.ssh_client import BaseSshClient, OpenSshClient


REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES: dict[str, tuple[tuple[str, str], ...]] = {
    'base': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('pandas', 'pandas'),
    ),
    'augmentation': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('albumentations', 'albumentations'),
    ),
    'classification': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
    ),
    'yolo': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
    ),
    'ultralytics_yolo': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
    ),
    'coco_detection': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
    ),
    'mmdetection': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('mmcv', 'mmcv'),
        ('mmdet', 'mmdet'),
    ),
    'open_vocab_detection': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'open_world_detection': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'sahi_detection': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('sahi', 'sahi'),
    ),
    'yolo_segmentation': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
    ),
    'sam_segmentation': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
    ),
    'detection_guided_segmentation': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
    ),
    'grounded_sam_segmentation': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'open_world_segmentation': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'embedding_torch': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
    ),
    'clip_embedding': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'openclip_embedding': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('open_clip_torch', 'open_clip'),
    ),
    'foundation_native': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('rfdetr', 'rfdetr'),
        ('sam2', 'sam2'),
        ('open_clip_torch', 'open_clip'),
        ('transformers', 'transformers'),
    ),
    'native_yolo_torchvision': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
    ),
    'native_yolo_timm': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('ultralytics', 'ultralytics'),
    ),
    'native_yolo_classifier': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
    ),
    'native_rfdetr_timm': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('rfdetr', 'rfdetr'),
    ),
    'native_transformer_torchvision': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'native_d_fine_torchvision': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
        ('scipy', 'scipy'),
        ('tensorboard', 'tensorboard'),
        ('calflops', 'calflops'),
        ('loguru', 'loguru'),
        ('faster_coco_eval', 'faster_coco_eval'),
    ),
    'native_yolo_sam_openclip': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('sam2', 'sam2'),
        ('open_clip_torch', 'open_clip'),
    ),
    'siglip_embedding': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
    ),
    'timm_classifier': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
    ),
    'openclip_classifier': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('open_clip_torch', 'open_clip'),
    ),
    'tracking_motion': (
        ('yaml', 'yaml'),
        ('numpy', 'numpy'),
    ),
    'tracking_reid': (
        ('yaml', 'yaml'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
    ),
    'vision_model': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('ultralytics', 'ultralytics'),
    ),
    'all': (
        ('yaml', 'yaml'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('pandas', 'pandas'),
        ('polars', 'polars'),
        ('albumentations', 'albumentations'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('ultralytics', 'ultralytics'),
    ),
}

REMOTE_TASK_ADAPTER_CORE_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ('yaml', 'yaml'),
    ('PIL', 'PIL'),
    ('pandas', 'pandas'),
    ('polars', 'polars'),
)


def _with_remote_task_adapter_core_dependencies(
    checks: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    merged: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, module_name in (*REMOTE_TASK_ADAPTER_CORE_DEPENDENCIES, *checks):
        if name in seen:
            continue
        seen.add(name)
        merged.append((name, module_name))
    return tuple(merged)


REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES = {
    name: _with_remote_task_adapter_core_dependencies(checks)
    for name, checks in REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES.items()
}
REMOTE_TASK_ADAPTER_YOLO_PROBE_PROFILES = frozenset(
    {
        'all',
        'foundation_native',
        'native_yolo_classifier',
        'native_yolo_sam_openclip',
        'native_yolo_timm',
        'native_yolo_torchvision',
        'ultralytics_yolo',
        'vision_model',
        'yolo',
        'yolo_segmentation',
    },
)
DEFAULT_REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILE = 'all'


def special_dependency_checks_for_profile(dependency_profile: str) -> tuple[str, ...]:
    if dependency_profile in REMOTE_TASK_ADAPTER_YOLO_PROBE_PROFILES:
        return ('ultralytics_yolo',)

    return ()


@dataclass(frozen=True, slots=True)
class ServerCheckResult:
    name: str
    status: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in {'ok', 'configured'}


class ServerChecker:
    def __init__(
        self,
        validator: ServerProfileValidator | None = None,
        ssh_client: BaseSshClient | None = None,
    ) -> None:
        self.validator = validator or ServerProfileValidator()
        self.ssh_client = ssh_client or OpenSshClient()

    def check(
        self,
        record: ServerRecord,
        live_ssh: bool = False,
        gpu_probe: bool = False,
        remote_task_adapter_deps: bool = False,
        dependency_profile: str = DEFAULT_REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILE,
        timeout_seconds: int = 10,
    ) -> ServerCheckResult:
        validation = self.validator.validate(record=record, check_paths=False)
        if not validation.is_valid:
            return ServerCheckResult(
                name=record.name,
                status='failed',
                message='server profile is invalid',
                metadata={
                    'errors': [
                        {'field': issue.field, 'message': issue.message}
                        for issue in validation.errors
                    ],
                },
            )

        requested_remote_checks = self._requested_remote_checks(
            live_ssh=live_ssh,
            gpu_probe=gpu_probe,
            remote_task_adapter_deps=remote_task_adapter_deps,
        )
        if len(requested_remote_checks) > 1:
            return ServerCheckResult(
                name=record.name,
                status='failed',
                message='only one remote preflight can be requested at a time',
                metadata={
                    'server_type': record.server_type,
                    'requested_remote_checks': requested_remote_checks,
                },
            )

        if record.server_type == 'local':
            if requested_remote_checks:
                return self._unsupported_remote_preflight(
                    record=record,
                    requested_remote_checks=requested_remote_checks,
                )
            return ServerCheckResult(
                name=record.name,
                status='ok',
                message='local profile is available',
                metadata={'server_type': record.server_type},
            )
        if record.server_type == 'local_ssh_simulator':
            if requested_remote_checks:
                return self._unsupported_remote_preflight(
                    record=record,
                    requested_remote_checks=requested_remote_checks,
                )
            return ServerCheckResult(
                name=record.name,
                status='configured',
                message='local SSH simulator profile is available for remote runner contract validation',
                metadata={'server_type': record.server_type},
            )

        if remote_task_adapter_deps:
            return self._check_remote_task_adapter_deps(
                record=record,
                dependency_profile=dependency_profile,
                timeout_seconds=timeout_seconds,
            )

        if gpu_probe:
            return self._check_gpu_probe(record=record, timeout_seconds=timeout_seconds)

        if live_ssh:
            return self._check_live_ssh(record=record, timeout_seconds=timeout_seconds)

        return ServerCheckResult(
            name=record.name,
            status='configured',
            message='remote profile is structurally valid; no live SSH/GPU check was requested',
            metadata={'server_type': record.server_type},
        )

    def _requested_remote_checks(
        self,
        live_ssh: bool,
        gpu_probe: bool,
        remote_task_adapter_deps: bool,
    ) -> list[str]:
        checks: list[str] = []
        if live_ssh:
            checks.append('live_ssh')
        if gpu_probe:
            checks.append('gpu_probe')
        if remote_task_adapter_deps:
            checks.append('remote_task_adapter_deps')

        return checks

    def _unsupported_remote_preflight(
        self,
        record: ServerRecord,
        requested_remote_checks: list[str],
    ) -> ServerCheckResult:
        return ServerCheckResult(
            name=record.name,
            status='failed',
            message='remote preflight is not supported for this server profile type',
            metadata={
                'server_type': record.server_type,
                'requested_remote_checks': requested_remote_checks,
            },
        )

    def _check_live_ssh(self, record: ServerRecord, timeout_seconds: int) -> ServerCheckResult:
        command = 'echo ironflow_ssh_ok'
        result = self.ssh_client.run_command(
            server=record,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        metadata = {
            'server_type': record.server_type,
            'command': result.command,
            'exit_code': result.exit_code,
            'stdout_tail': result.stdout[-1000:],
            'stderr_tail': result.stderr[-1000:],
        }
        if result.is_success and 'ironflow_ssh_ok' in result.stdout:
            return ServerCheckResult(
                name=record.name,
                status='ok',
                message='live SSH command succeeded',
                metadata=metadata,
            )

        return ServerCheckResult(
            name=record.name,
            status='failed',
            message='live SSH command failed',
            metadata=metadata,
        )

    def _check_gpu_probe(self, record: ServerRecord, timeout_seconds: int) -> ServerCheckResult:
        command = (
            'nvidia-smi '
            '--query-gpu=name,driver_version,memory.total '
            '--format=csv,noheader,nounits'
        )
        result = self.ssh_client.run_command(
            server=record,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        gpus = self._parse_nvidia_smi_query(stdout=result.stdout)
        metadata = {
            'server_type': record.server_type,
            'command': result.command,
            'exit_code': result.exit_code,
            'stdout_tail': result.stdout[-1000:],
            'stderr_tail': result.stderr[-1000:],
            'gpu_count': len(gpus),
            'gpus': gpus,
        }
        if result.is_success and gpus:
            return ServerCheckResult(
                name=record.name,
                status='ok',
                message='remote GPU probe succeeded',
                metadata=metadata,
            )

        return ServerCheckResult(
            name=record.name,
            status='failed',
            message='remote GPU probe failed',
            metadata=metadata,
        )

    @classmethod
    def dependency_profile_names(cls) -> tuple[str, ...]:
        return tuple(REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES)

    def _check_remote_task_adapter_deps(
        self,
        record: ServerRecord,
        dependency_profile: str,
        timeout_seconds: int,
    ) -> ServerCheckResult:
        command = self._remote_task_adapter_dependency_command(dependency_profile=dependency_profile)
        result = self.ssh_client.run_command(
            server=record,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        probe = self._parse_probe_stdout(stdout=result.stdout)
        metadata = {
            'server_type': record.server_type,
            'command': result.command,
            'exit_code': result.exit_code,
            'stdout_tail': result.stdout[-1000:],
            'stderr_tail': result.stderr[-1000:],
            'probe': probe,
            'dependency_profile': dependency_profile,
        }
        if result.is_success and bool(probe.get('success')):
            return ServerCheckResult(
                name=record.name,
                status='ok',
                message='remote task-adapter dependency probe succeeded',
                metadata=metadata,
            )

        return ServerCheckResult(
            name=record.name,
            status='failed',
            message='remote task-adapter dependency probe failed',
            metadata=metadata,
        )

    def _remote_task_adapter_dependency_command(
        self,
        dependency_profile: str = DEFAULT_REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILE,
    ) -> str:
        checks = REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES.get(dependency_profile)
        if checks is None:
            allowed = ', '.join(self.dependency_profile_names())
            raise ValueError(f'unknown dependency profile: {dependency_profile}; allowed: {allowed}')
        special_checks = special_dependency_checks_for_profile(dependency_profile=dependency_profile)
        script = (
            'import importlib, json, sys\n'
            f'profile = {json.dumps(dependency_profile)}\n'
            f'checks = {json.dumps(checks)}\n'
            f'special_checks = {json.dumps(special_checks)}\n'
            'results = {}\n'
            'success = True\n'
            'for name, module_name in checks:\n'
            '    try:\n'
            '        module = importlib.import_module(module_name)\n'
            '        results[name] = {"ok": True, "version": str(getattr(module, "__version__", ""))}\n'
            '    except Exception as error:\n'
            '        success = False\n'
            '        results[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}\n'
            'if "ultralytics_yolo" in special_checks:\n'
            '  try:\n'
            '    from ultralytics import YOLO\n'
            '    results["ultralytics_yolo"] = {"ok": True, "object": str(YOLO)}\n'
            '  except Exception as error:\n'
            '    success = False\n'
            '    results["ultralytics_yolo"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}\n'
            'payload = {"schema_version": "0.1", "profile": profile, "success": success, "checks": results}\n'
            'print(json.dumps(payload, sort_keys=True))\n'
            'raise SystemExit(0 if success else 2)\n'
        )
        python_selector = (
            'PYTHON_BIN="${IRONFLOW_REMOTE_PYTHON:-}"; '
            'if [ -z "$PYTHON_BIN" ] && [ -x /venv/main/bin/python ]; then '
            'PYTHON_BIN=/venv/main/bin/python; '
            'fi; '
            'if [ -z "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi; '
        )
        command_args = ['-c', script]

        return python_selector + '"$PYTHON_BIN" ' + ' '.join(shlex.quote(part) for part in command_args)

    def _parse_probe_stdout(self, stdout: str) -> dict[str, object]:
        for line in reversed(stdout.splitlines()):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        return {'success': False, 'error': 'probe JSON payload not found'}

    def _parse_nvidia_smi_query(self, stdout: str) -> list[dict[str, object]]:
        gpus: list[dict[str, object]] = []
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 3 or not parts[0]:
                continue
            memory_total_mb: int | None
            try:
                memory_total_mb = int(float(parts[2]))
            except ValueError:
                memory_total_mb = None
            gpus.append(
                {
                    'name': parts[0],
                    'driver_version': parts[1],
                    'memory_total_mb': memory_total_mb,
                },
            )

        return gpus
