import sys
from pathlib import Path

from ironflow_exp.engine.ui.local_mock_controller import DEFAULT_LOG_TAIL
from ironflow_exp.engine.configs.config_loader import EngineConfigLoader


DEFAULT_WSL_CONFIG_PATH = 'runs/configs/wsl_ssh_background.yaml'
DEFAULT_WSL_DB_PATH = 'runs/wsl_ssh_experiments.sqlite3'
DEFAULT_WSL_EXPERIMENT_ID = 'exp_gui_wsl_ssh'
DEFAULT_WSL_TEST_MODEL_CONFIG_PATH = 'configs/engine/wsl_ssh_test_model_smoke.yaml'
DEFAULT_WSL_TEST_MODEL_EXPERIMENT_ID = 'exp_gui_wsl_test_model'
DEFAULT_WSL_YOLO_TORCHVISION_CONFIG_PATH = 'configs/engine/wsl_ssh_yolo_torchvision_remote_adapter_smoke.yaml'
DEFAULT_WSL_YOLO_TORCHVISION_EXPERIMENT_ID = 'exp_gui_wsl_yolo_torchvision_chain'
DEFAULT_WSL_ALBUMENTATIONS_CROP_CONFIG_PATH = 'configs/engine/wsl_ssh_albumentations_classification_crop_augmentation_smoke.yaml'
DEFAULT_WSL_ALBUMENTATIONS_CROP_EXPERIMENT_ID = 'exp_gui_wsl_albumentations_crop_augmentation'
DEFAULT_WSL_ALBUMENTATIONS_BBOX_CONFIG_PATH = 'configs/engine/wsl_ssh_albumentations_detection_bbox_augmentation_smoke.yaml'
DEFAULT_WSL_ALBUMENTATIONS_BBOX_EXPERIMENT_ID = 'exp_gui_wsl_albumentations_bbox_augmentation'
DEFAULT_WSL_SERVER_NAME = 'wsl_ubuntu'
DEFAULT_GPU_YOLO26_CONFIG_PATH = 'configs/engine/gpu_yolo26_detection_imported_20260617_train.yaml'
DEFAULT_GPU_YOLO26_DB_PATH = 'runs/gpu_yolo26_model_name_experiments.sqlite3'
DEFAULT_GPU_YOLO26_EXPERIMENT_ID = 'exp_gpu_yolo26_model_name_imported_20260617_train'
DEFAULT_GPU_SERVER_NAME = 'vast_5090'
DEFAULT_WSL_EXECUTE_TIMEOUT_SECONDS = 600
DEFAULT_REMOTE_DEPENDENCY_INSTALL_TIMEOUT_SECONDS = 900
SSH_TIMEOUT_STAGES = {'bootstrap', 'upload', 'run', 'submit', 'download', 'collect', 'cancel'}


def dependency_profile_for_config_path(config_path: str, *, native_wrappers: bool = False) -> str:
    if native_wrappers:
        profile = native_dependency_profile_for_config_path(config_path)
        return profile or 'foundation_native'
    normalized = config_path.replace('\\', '/').lower()
    if 'albumentations' in normalized:
        return 'augmentation'
    if 'yolo' in normalized or 'torchvision' in normalized:
        return 'vision_model'

    return 'base'


def native_dependency_profile_for_config_path(config_path: str) -> str | None:
    path = _resolve_config_path(config_path=config_path)
    if not path.exists():
        return None
    try:
        config = EngineConfigLoader().load_file(path=path)
    except Exception:
        return None

    adapters = {
        task.adapter
        for task in config.tasks
        if task.enabled and task.adapter
    }
    if 'sam_promptable_segmentation' in adapters or 'clip_embedding' in adapters:
        if 'ultralytics_yolo' in adapters:
            return 'native_yolo_sam_openclip'
        return 'foundation_native'
    if 'rf_detr_detection' in adapters:
        if 'timm_classifier' in adapters:
            return 'native_rfdetr_timm'
        return 'foundation_native'
    if 'd_fine_detection' in adapters:
        return 'native_d_fine_torchvision'
    if {'rt_detr_detection', 'rt_detr_v2_detection', 'lw_detr_detection'} & adapters:
        return 'native_transformer_torchvision'
    if 'ultralytics_yolo_classifier' in adapters:
        return 'native_yolo_classifier'
    if 'ultralytics_yolo' in adapters and 'timm_classifier' in adapters:
        return 'native_yolo_timm'
    if 'ultralytics_yolo' in adapters and 'torchvision_classifier' in adapters:
        return 'native_yolo_torchvision'
    if 'ultralytics_yolo' in adapters:
        return 'ultralytics_yolo'
    if 'timm_classifier' in adapters:
        return 'timm_classifier'
    if 'torchvision_classifier' in adapters:
        return 'classification'

    return None


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.exists() or path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[4]
    project_relative_path = project_root / path
    if project_relative_path.exists():
        return project_relative_path

    return path


class SshStageCommandBuilder:
    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    def server_check(self, server_name: str, db_path: str, timeout_seconds: int = 10) -> list[str]:
        return [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'server',
            'check',
            server_name,
            '--db',
            db_path,
            '--live-ssh',
            '--timeout',
            str(timeout_seconds),
            '--json',
        ]

    def remote_task_adapter_dependency_check(
        self,
        server_name: str,
        db_path: str,
        timeout_seconds: int = 60,
        dependency_profile: str = 'all',
    ) -> list[str]:
        return [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'server',
            'check',
            server_name,
            '--db',
            db_path,
            '--remote-task-adapter-deps',
            '--dependency-profile',
            dependency_profile,
            '--timeout',
            str(timeout_seconds),
            '--json',
        ]

    def install_deps(
        self,
        server_name: str,
        db_path: str,
        dependency_profile: str = 'all',
        timeout_seconds: int = DEFAULT_REMOTE_DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
        upgrade: bool = False,
    ) -> list[str]:
        command = [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'server',
            'install-deps',
            server_name,
            '--db',
            db_path,
            '--dependency-profile',
            dependency_profile,
            '--timeout',
            str(timeout_seconds),
            '--execute',
            '--json',
        ]
        if not upgrade:
            command.append('--no-upgrade')

        return command

    def prepare(
        self,
        config_path: str,
        server_name: str,
        experiment_id: str,
        db_path: str,
        replace_existing: bool = False,
    ) -> list[str]:
        command = [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'ssh',
            'prepare',
            '--config',
            config_path,
            '--server',
            server_name,
            '--experiment-id',
            experiment_id,
            '--db',
            db_path,
        ]
        if replace_existing:
            command.append('--replace-existing')
        command.append('--json')

        return command

    def execute(
        self,
        config_path: str,
        server_name: str,
        experiment_id: str,
        db_path: str,
        replace_existing: bool = False,
        timeout_seconds: int = DEFAULT_WSL_EXECUTE_TIMEOUT_SECONDS,
    ) -> list[str]:
        command = [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'ssh',
            'execute',
            '--config',
            config_path,
            '--server',
            server_name,
            '--experiment-id',
            experiment_id,
            '--db',
            db_path,
            '--timeout',
            str(timeout_seconds),
        ]
        if replace_existing:
            command.append('--replace-existing')
        command.append('--json')

        return command

    def stage(
        self,
        stage_name: str,
        experiment_id: str,
        db_path: str,
        tail: int = DEFAULT_LOG_TAIL,
        timeout_seconds: int | None = None,
    ) -> list[str]:
        command = [
            self.python_executable,
            '-m',
            'ironflow_exp.engine.cli.main',
            'ssh',
            stage_name,
            experiment_id,
            '--db',
            db_path,
        ]
        if stage_name == 'logs':
            command.extend(['--tail', str(tail)])
        else:
            if stage_name in SSH_TIMEOUT_STAGES and timeout_seconds is not None:
                command.extend(['--timeout', str(timeout_seconds)])
            command.append('--json')

        return command
