import json
import os
import posixpath
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import yaml

from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.core import UserImageDatasetValidator
from ironflow_exp.engine.importers import ExportDatasetImportResult, ExportDatasetImporter
from ironflow_exp.engine.ui.augmentation_policy_catalog import augmentation_policy_by_id
from ironflow_exp.engine.ui.candidate_experiments import (
    build_candidate_experiment_config,
    candidate_compact_flow,
    candidate_detail_rows,
    candidate_option_map,
)
from ironflow_exp.engine.ui.local_mock_controller import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_LOG_TAIL,
    EngineCliActionTargets,
    EngineCliResult,
    EngineCliRunner,
    LocalMockCommandBuilder,
)
from ironflow_exp.engine.ui.model_selection import (
    DetectCropClassifyCandidateSelection,
    available_detect_crop_classify_choices,
    build_detect_crop_classify_config_from_selection,
    choice_map,
    default_detect_crop_classify_selection,
    describe_detect_crop_classify_selection,
    label_for_augmentation_policy_id,
    label_for_candidate_id,
)
from ironflow_exp.engine.ui.model_adapter_catalog import candidate_by_id, planned_real_candidate_brief_summary
from ironflow_exp.engine.ui.native_params import load_native_param_overrides_file, merge_native_param_overrides
from ironflow_exp.engine.ui.output_options import (
    CHECKPOINT_COLLECT_BEST,
    CHECKPOINT_COLLECT_BEST_LAST,
    COLLECT_MODE_FULL_DEBUG,
    COLLECT_MODE_QUICK,
    COLLECT_MODE_STANDARD,
    COLLECT_MODE_WEIGHTS,
    WEIGHT_MODE_ARCHITECTURE,
    WEIGHT_MODE_CHECKPOINT,
    WEIGHT_MODE_PRESET,
    WEIGHT_MODE_PRETRAINED_DOWNLOAD,
    GuiOutputOptions,
    GuiWeightOptions,
    write_effective_config,
)
from ironflow_exp.engine.ui.readiness_matrix import gui_readiness_matrix_text
from ironflow_exp.engine.ui.recommended_experiments import (
    DEFAULT_MODEL_NAME_CLASSIFICATION_CROPS,
    DEFAULT_MODEL_NAME_DETECTION_DATASET,
    RecommendedModelMatrix,
    build_recommended_combination_config,
    load_recommended_model_matrix,
    recommended_combination_label,
    recommended_combination_summary,
)
from ironflow_exp.engine.ui.result_compare import (
    ExperimentComparisonRow,
    load_experiment_comparison,
    metric_tree_values,
)
from ironflow_exp.engine.ui.result_plots import generate_comparison_artifacts
from ironflow_exp.engine.ui.scenario_catalog import (
    FLOW_DETECT_CROP_CLASSIFY,
    FLOW_DETECT_CROP_CLASSIFY_AUGMENTED,
    GuiScenario,
    LOCAL_SCENARIOS,
    WSL_SCENARIOS,
    describe_scenario_flow,
)
from ironflow_exp.engine.ui.ssh_stage_controller import (
    DEFAULT_GPU_SERVER_NAME,
    DEFAULT_WSL_YOLO_TORCHVISION_CONFIG_PATH,
    DEFAULT_WSL_CONFIG_PATH,
    DEFAULT_WSL_DB_PATH,
    DEFAULT_WSL_EXPERIMENT_ID,
    DEFAULT_WSL_SERVER_NAME,
    SshStageCommandBuilder,
    dependency_profile_for_config_path,
)
from ironflow_exp.engine.server import ServerProfileLoader, ServerProfileValidator
from ironflow_exp.engine.storage import SQLiteExperimentStorage

SSH_STAGE_TIMEOUT_SECONDS = 3600
SSH_QUICK_STAGE_TIMEOUT_SECONDS = 120
SSH_DATASET_PREPARE_TIMEOUT_SECONDS = 14400
CLI_STAGE_TIMEOUT_SECONDS = {
    'ssh bootstrap': SSH_QUICK_STAGE_TIMEOUT_SECONDS,
    'ssh upload': SSH_STAGE_TIMEOUT_SECONDS,
    'ssh run': SSH_STAGE_TIMEOUT_SECONDS,
    'ssh execute': SSH_STAGE_TIMEOUT_SECONDS,
    'ssh submit': SSH_STAGE_TIMEOUT_SECONDS,
    'ssh collect': SSH_STAGE_TIMEOUT_SECONDS,
}

GUI_COMMAND_TIMEOUT_SECONDS = {
    'run': 180,
    'status': 60,
    'logs': 60,
    'collect': 120,
    'ssh server check': 30,
    'ssh dependency check': 90,
    'ssh install deps': SSH_STAGE_TIMEOUT_SECONDS,
    'ssh prepare data': SSH_DATASET_PREPARE_TIMEOUT_SECONDS,
    'server add': 30,
    # Prepare can build a local code/data package. Top10 real-data runs may
    # stage several GB before any SSH upload starts, so this cannot be tiny.
    'ssh prepare': SSH_STAGE_TIMEOUT_SECONDS,
    'ssh status': 60,
    'ssh logs': 60,
    **{name: timeout for name, timeout in CLI_STAGE_TIMEOUT_SECONDS.items()},
    'ssh bootstrap': SSH_QUICK_STAGE_TIMEOUT_SECONDS + 30,
}
WEIGHT_MODE_LABELS = {
    'Preset config': WEIGHT_MODE_PRESET,
    'Architecture only': WEIGHT_MODE_ARCHITECTURE,
    'Pretrained download': WEIGHT_MODE_PRETRAINED_DOWNLOAD,
    'Explicit checkpoint': WEIGHT_MODE_CHECKPOINT,
}
WEIGHT_MODE_BY_VALUE = {
    value: label
    for label, value in WEIGHT_MODE_LABELS.items()
}
REMOTE_RUNTIME_NATIVE = 'native_python'
REMOTE_RUNTIME_DOCKER = 'docker_image'
DEFAULT_GPU_DOCKER_IMAGE = 'ironflow-gpu:20260625'
REMOTE_RUNTIME_LABELS = {
    'Native Python': REMOTE_RUNTIME_NATIVE,
    'Docker image': REMOTE_RUNTIME_DOCKER,
}
REMOTE_RUNTIME_BY_VALUE = {
    value: label
    for label, value in REMOTE_RUNTIME_LABELS.items()
}
COLLECT_MODE_LABELS = {
    'Quick': COLLECT_MODE_QUICK,
    'Standard': COLLECT_MODE_STANDARD,
    'Weights': COLLECT_MODE_WEIGHTS,
    'Full Debug': COLLECT_MODE_FULL_DEBUG,
}
COLLECT_MODE_BY_VALUE = {
    value: label
    for label, value in COLLECT_MODE_LABELS.items()
}
CHECKPOINT_COLLECT_LABELS = {
    'Best only': CHECKPOINT_COLLECT_BEST,
    'Best + Last': CHECKPOINT_COLLECT_BEST_LAST,
}
CHECKPOINT_COLLECT_BY_VALUE = {
    value: label
    for label, value in CHECKPOINT_COLLECT_LABELS.items()
}
DEFERRED_RECOMMENDED_COMBINATION_REASONS = {
    'open_vocab_grounding_dino_sam2_dinov3_audit': (
        'Legacy audit path is deferred: GroundingDINO supervised training and '
        'DINOv3 pretrained access are not ready. Use Preview/Check only.'
    ),
}
DEFERRED_RECOMMENDED_RUN_STAGES = frozenset({'prepare', 'execute', 'bootstrap', 'upload', 'submit'})
RESULTS_EMPTY_GUIDE = (
    'No results loaded. After Collect, choose Use WSL DB and Refresh. '
    'For local runs, choose Use Local DB and Refresh.'
)
DEFAULT_NATIVE_PARAMS_PATH = 'configs/engine/native_params/rtx5090_balanced_defaults.yaml'
TOP5_AUG_NATIVE_PARAMS_PATH = 'configs/engine/native_params/rtx5090_top5_aug_defaults.yaml'
NATIVE_PARAMS_DIR = 'configs/engine/native_params'
RUN_QUEUE_STATE_PATH = Path('runs/gui_queue/current_queue.json')
MAX_RUN_QUEUE_ITEMS = 12
QUEUE_RUN_STAGE_ORDER = (
    'ssh install deps',
    'ssh prepare',
    'ssh bootstrap',
    'ssh upload',
    'ssh run',
    'ssh collect',
)
QUEUE_RUN_STAGE_INDEX = {
    stage_name: index
    for index, stage_name in enumerate(QUEUE_RUN_STAGE_ORDER, start=1)
}
SSH_RUN_LOG_POLL_INTERVAL_SECONDS = 10.0
SSH_RUN_LOG_HEARTBEAT_SECONDS = 30.0
SSH_RUN_LOG_POLL_TIMEOUT_SECONDS = 45
SSH_RUN_LOG_POLL_TAIL_LINES = 80
DATASET_SOURCE_LOCAL = 'Local upload/cache'
DATASET_SOURCE_REMOTE_PRESTAGED = 'Remote pre-staged path'
DATASET_SOURCE_LABELS = (DATASET_SOURCE_REMOTE_PRESTAGED, DATASET_SOURCE_LOCAL)
DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT = '/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1'
WEAK_AUG_REMOTE_PRESTAGED_DATASET_ROOT = '/workspace/ironflow/prestaged/tank_armor_prepared_v20260630_aug_weak_v1'
AV_CROP_AUG_REMOTE_PRESTAGED_DATASET_ROOT = '/workspace/ironflow/prestaged/tank_armor_prepared_v20260705_av_crop_aug_weak_v1'
LEGACY_REMOTE_PRESTAGED_DATASET_ROOTS = {
    '/workspace/ironflow/prestaged/tank_armor_prepared_v20260630',
    '/workspace/ironflow/prestaged/tank_armor_prepared_v20260629',
    '/workspace/ironflow/prestaged/tank14_prepared_v20260629',
    '/workspace/ironflow/prestaged/tank9_prepared_v20260629',
}
MODEL_NAME_DETECTION_DATASET_SUFFIX = 'detector_tank_av/detection'
MODEL_NAME_CLASSIFICATION_DATASET_SUFFIX = 'classifier_all/images'
MODEL_NAME_AV_CLASSIFICATION_CROP_SUFFIX = 'classifier_av/crops'
CLASSIFIER_CROP_DATASET_SUFFIXES = (
    'classifier_mbt/crops',
    MODEL_NAME_AV_CLASSIFICATION_CROP_SUFFIX,
)
CLASSIFIER_CROP_TOP5_DATASET_MARKERS = (
    'tank_armor_prepared_v20260630_aug_weak_v1/classifier_mbt/crops',
    'tank_armor_prepared_v20260705_av_crop_aug_weak_v1/classifier_av/crops',
)
CLASSIFIER_CROP_TOP5_MODEL_PRESETS: dict[str, dict[str, object]] = {
    'efficientnet_b3': {
        'epochs': 50,
        'batch_size': 32,
        'image_size': 224,
        'learning_rate': 0.0005,
        'scheduler': 'cosine',
        'min_learning_rate': 0.000001,
        'patience': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0,
    },
    'efficientnet_b0': {
        'epochs': 50,
        'batch_size': 32,
        'image_size': 224,
        'learning_rate': 0.0007,
        'scheduler': 'cosine',
        'min_learning_rate': 0.00001,
        'patience': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0,
    },
    'mobilenet_v3_large': {
        'epochs': 50,
        'batch_size': 32,
        'image_size': 224,
        'learning_rate': 0.0008,
        'scheduler': 'cosine',
        'min_learning_rate': 0.00001,
        'patience': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0,
    },
    'efficientnet_v2_s': {
        'epochs': 50,
        'batch_size': 32,
        'image_size': 224,
        'learning_rate': 0.0003,
        'scheduler': 'cosine',
        'min_learning_rate': 0.000001,
        'patience': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0,
    },
    'resnet50': {
        'epochs': 50,
        'batch_size': 32,
        'image_size': 224,
        'learning_rate': 0.0005,
        'scheduler': 'cosine',
        'min_learning_rate': 0.00001,
        'patience': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0,
    },
}


def _normalize_remote_prestaged_dataset_root(root: object) -> str:
    value = str(root or '').replace('\\', '/').rstrip('/')
    if not value:
        return DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT
    if value in LEGACY_REMOTE_PRESTAGED_DATASET_ROOTS:
        return DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT
    if value.endswith('/tank_armor_prepared_v20260630') or value.endswith('/tank_armor_prepared_v20260629'):
        return DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT
    return value


@dataclass(frozen=True, slots=True)
class WorkerMessage:
    kind: str
    text: str
    success: bool | None = None
    status_text: str | None = None
    action_targets: EngineCliActionTargets | None = None
    progress_value: float | None = None
    progress_label: str | None = None
    queue_item_id: str | None = None
    queue_status: str | None = None
    queue_detail: str | None = None


@dataclass(frozen=True, slots=True)
class CommandSequenceStep:
    name: str
    command: list[str]
    timeout_seconds: int | None
    queue_item_id: str | None = None
    final_queue_status: str | None = None
    queue_step_index: int | None = None
    queue_step_total: int | None = None
    queue_item_index: int | None = None
    queue_item_total: int | None = None


@dataclass(frozen=True, slots=True)
class RunQueueItem:
    queue_id: str
    order: int
    source_type: str
    source_id: str | None
    display_name: str
    stage: str
    config_path: str
    db_path: str
    native_params_path: str
    experiment_id: str
    server_profile: str
    native_wrappers: bool
    replace_existing: bool
    save_csv: bool
    save_summary: bool
    save_previews: bool
    save_checkpoints: bool
    collect_mode: str
    checkpoint_collect_mode: str
    weight_mode: str
    checkpoint_path: str
    epochs: str
    det_batch: str
    cls_batch: str
    det_img: str
    cls_img: str
    learning_rate: str
    early_stopping_patience: str
    early_stopping_min_delta: str
    timeout: str
    tail: str
    dataset_source_mode: str = DATASET_SOURCE_REMOTE_PRESTAGED
    remote_dataset_root: str = DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT
    remote_runtime_mode: str = REMOTE_RUNTIME_NATIVE
    container_image: str = ''
    active_recommended_combination_id: str | None = None
    active_gpu_candidate_id: str | None = None
    status: str = 'pending'
    progress_detail: str = ''
    result_dir: str = ''
    error_message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> 'RunQueueItem':
        defaults = cls(
            queue_id='',
            order=1,
            source_type='custom',
            source_id=None,
            display_name='Custom experiment',
            stage='custom',
            config_path='',
            db_path='',
            native_params_path=DEFAULT_NATIVE_PARAMS_PATH,
            experiment_id='',
            server_profile=DEFAULT_GPU_SERVER_NAME,
            native_wrappers=True,
            replace_existing=False,
            save_csv=True,
            save_summary=True,
            save_previews=True,
            save_checkpoints=True,
            collect_mode=COLLECT_MODE_BY_VALUE[COLLECT_MODE_WEIGHTS],
            checkpoint_collect_mode=CHECKPOINT_COLLECT_BY_VALUE[CHECKPOINT_COLLECT_BEST],
            weight_mode=WEIGHT_MODE_BY_VALUE[WEIGHT_MODE_PRESET],
            checkpoint_path='',
            epochs='',
            det_batch='',
            cls_batch='',
            det_img='',
            cls_img='',
            learning_rate='',
            early_stopping_patience='',
            early_stopping_min_delta='',
            timeout='',
            tail=str(DEFAULT_LOG_TAIL),
            dataset_source_mode=DATASET_SOURCE_REMOTE_PRESTAGED,
            remote_dataset_root=DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT,
            remote_runtime_mode=REMOTE_RUNTIME_NATIVE,
            container_image='',
        )
        values = defaults.to_dict()
        values.update(data)
        values['queue_id'] = str(values.get('queue_id') or '')
        values['order'] = int(values.get('order') or 1)
        if values.get('weight_mode') not in WEIGHT_MODE_LABELS:
            values['weight_mode'] = WEIGHT_MODE_BY_VALUE[WEIGHT_MODE_PRESET]
        if values.get('remote_runtime_mode') not in REMOTE_RUNTIME_LABELS.values():
            values['remote_runtime_mode'] = REMOTE_RUNTIME_NATIVE
        values['remote_dataset_root'] = _normalize_remote_prestaged_dataset_root(
            values.get('remote_dataset_root')
        )
        return cls(**values)  # type: ignore[arg-type]


SCENARIO_DISPLAY_LABELS = {
    'Local mock': 'Local mock',
    'Local test model': 'Tiny rule classifier',
    'Local CPU rule model': 'CPU rule classifier',
    'Local K2 vs Leopard 2': 'K2 vs Leopard 2 smoke',
    'Local detect-crop-classify': 'Manifest detector + crop + classifier',
    'Local YOLO crop Torchvision': 'YOLO11n + MobileNetV3',
    'Local YOLO crop Torchvision ByteTrack': 'YOLO11n + MobileNetV3 + ByteTrack',
    'Local segmentation smoke': 'Segmentation smoke',
    'Local prepared detection gate': 'Prepared detector check',
    'Local prepared classification gate': 'Prepared classifier check',
    'Local prepared tracking gate': 'Prepared tracker check',
    'Local prepared segmentation gate': 'Prepared segmentation check',
    'Local prepared embedding gate': 'Prepared embedding check',
    'Local crop augmentation': 'Crop augmentation smoke',
    'Local Albumentations crop augmentation': 'Albumentations crop augmentation',
    'Local Albumentations bbox augmentation': 'Albumentations bbox augmentation',
    'GPU YOLO26 model-name train': 'YOLO26n model-name train',
    'WSL mock': 'WSL mock',
    'WSL test model': 'WSL tiny rule classifier',
    'WSL YOLO crop Torchvision': 'WSL YOLO11n + MobileNetV3',
    'WSL Albumentations crop augmentation': 'WSL Albumentations crop augmentation',
    'WSL Albumentations bbox augmentation': 'WSL Albumentations bbox augmentation',
}


def scenario_choice_label(name: str, scenario: GuiScenario) -> str:
    return SCENARIO_DISPLAY_LABELS.get(name, name)


def scenario_choice_map(scenarios: dict[str, GuiScenario]) -> dict[str, str]:
    return {
        scenario_choice_label(name, scenario): name
        for name, scenario in scenarios.items()
    }


def scenario_compact_flow(scenario: GuiScenario) -> str:
    if not scenario.task_flow:
        return 'Config preset'

    labels = []
    for step in scenario.task_flow:
        labels.append(_task_stage_label(step.task_type, step.adapter))

    return ' -> '.join(labels)


def scenario_detail_rows(scenario: GuiScenario) -> tuple[tuple[str, str, str, str], ...]:
    if not scenario.task_flow:
        return (('Preset', 'Configured by YAML', '-', scenario.config_path),)

    return tuple(
        (
            _task_stage_label(step.task_type, step.adapter),
            step.model_id or step.adapter or 'configured',
            step.input_kind,
            step.adapter or '-',
        )
        for step in scenario.task_flow
    )


def local_override_scope_text(scenario: GuiScenario) -> str:
    if scenario.flow_id in {FLOW_DETECT_CROP_CLASSIFY, FLOW_DETECT_CROP_CLASSIFY_AUGMENTED}:
        return 'Local smoke only'

    return 'Locked local smoke scenario'


def recommended_locked_config_path(combination: object) -> str:
    priority = int(getattr(combination, 'priority'))
    combination_id = str(getattr(combination, 'combination_id'))
    experiment_type = str(getattr(combination, 'experiment_type', 'end_to_end'))

    if experiment_type == 'detector_only':
        return f'configs/engine/detector_only_balanced/{priority:02d}_{_safe_filename(combination_id)}.yaml'
    if experiment_type == 'classifier_only':
        return f'configs/engine/classifier_only_balanced/{priority:02d}_{_safe_filename(combination_id)}.yaml'

    return f'configs/engine/top10_balanced/{priority:02d}_{_safe_filename(combination_id)}.yaml'


def recommended_experiment_prefix(combination: object) -> str:
    experiment_type = str(getattr(combination, 'experiment_type', 'end_to_end'))
    if experiment_type == 'detector_only':
        return 'Detector'
    if experiment_type == 'classifier_only':
        return 'Classifier'
    if experiment_type in {'review_path', 'embedding_review'}:
        return 'Review'

    return 'Top10'


def _safe_filename(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value.lower()).strip('_')


def parse_vast_ssh_command(command: str) -> dict[str, str] | None:
    try:
        tokens = shlex.split(command.strip())
    except ValueError:
        return None
    if not tokens:
        return None

    ssh_index = next((index for index, token in enumerate(tokens) if token.lower() == 'ssh'), 0)
    tokens = tokens[ssh_index:]
    port: str | None = None
    endpoint: str | None = None
    skip_next = False
    options_with_values = {'-i', '-l', '-o', '-L', '-R', '-J', '-F'}

    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in {'-p', '-P'} and index + 1 < len(tokens):
            port = tokens[index + 1]
            skip_next = True
            continue
        if token.startswith('-p') and len(token) > 2:
            port = token[2:]
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if '@' in token and not token.startswith('-'):
            endpoint = token

    if not port or not endpoint or '@' not in endpoint:
        return None
    if not port.isdigit():
        return None
    username, host = endpoint.rsplit('@', 1)
    host = host.strip('[]')
    if not username or not host:
        return None

    return {
        'host': host,
        'port': port,
        'username': username,
    }


def _task_stage_label(task_type: str, adapter: str | None = None) -> str:
    if task_type == 'preprocessing' and adapter == 'detection_to_classification_crop':
        return 'Crop'
    if task_type == 'augmentation':
        return 'Augment'

    return task_type.replace('_', ' ').title()

def _default_project_dir() -> Path:
    module_project_dir = Path(__file__).resolve().parents[4]
    cwd = Path.cwd().resolve()
    if (cwd / 'src' / 'ironflow_exp').exists():
        return cwd
    return module_project_dir


def _resolve_project_dir(project_dir: Path | None) -> Path:
    if project_dir is not None:
        resolved = project_dir.resolve()
        if (resolved / 'src' / 'ironflow_exp').exists():
            return resolved
    return _default_project_dir()

class EngineTkApp:
    def __init__(self, root: tk.Tk, project_dir: Path | None = None) -> None:
        self.root = root
        self.project_dir = _resolve_project_dir(project_dir)
        self.builder = LocalMockCommandBuilder()
        self.ssh_builder = SshStageCommandBuilder()
        self.runner = EngineCliRunner(cwd=self.project_dir)
        self.messages: queue.Queue[WorkerMessage] = queue.Queue()
        self.running = False
        self.stop_requested = False
        self.command_buttons: list[ttk.Button] = []
        self.stop_buttons: list[ttk.Button] = []
        self.result_buttons: list[ttk.Button] = []
        self.local_model_selector_widgets: list[ttk.Combobox] = []
        self.result_targets = EngineCliActionTargets()
        self.local_detect_crop_classify_choices = available_detect_crop_classify_choices()
        self.local_detector_choice_map = choice_map(self.local_detect_crop_classify_choices.detectors)
        self.local_crop_adapter_choice_map = choice_map(self.local_detect_crop_classify_choices.crop_adapters)
        self.local_classifier_choice_map = choice_map(self.local_detect_crop_classify_choices.classifiers)
        self.local_augmentation_policy_choice_map = choice_map(self.local_detect_crop_classify_choices.augmentation_policies)
        self.local_scenario_label_map = scenario_choice_map(LOCAL_SCENARIOS)
        self.ssh_scenario_label_map = scenario_choice_map(WSL_SCENARIOS)
        self.recommended_matrix = self._load_recommended_matrix()
        self.recommended_label_map = {
            recommended_combination_label(combination): combination.combination_id
            for combination in self.recommended_matrix.all_experiments
        }
        first_recommended_label = next(iter(self.recommended_label_map), '')
        self.active_recommended_combination_id: str | None = None
        self.gpu_candidate_label_map = candidate_option_map()
        first_gpu_candidate_label = next(iter(self.gpu_candidate_label_map), '')
        self.active_gpu_candidate_id: str | None = None
        self.run_queue_items: list[RunQueueItem] = []
        self.active_queue_item_id: str | None = None
        self.current_running_queue_item_id: str | None = None
        self._loading_queue_item = False
        self._last_queue_state_save_at = 0.0

        self.local_scenario_var = tk.StringVar(value=scenario_choice_label('Local mock', LOCAL_SCENARIOS['Local mock']))
        self.local_detector_var = tk.StringVar()
        self.local_crop_adapter_var = tk.StringVar()
        self.local_classifier_var = tk.StringVar()
        self.local_augmentation_policy_var = tk.StringVar()
        self.config_var = tk.StringVar(value=DEFAULT_CONFIG_PATH)
        self.db_var = tk.StringVar(value=DEFAULT_DB_PATH)
        self.experiment_var = tk.StringVar(value=DEFAULT_EXPERIMENT_ID)
        self.replace_existing_var = tk.BooleanVar(value=False)
        self.save_csv_var = tk.BooleanVar(value=True)
        self.save_summary_var = tk.BooleanVar(value=True)
        self.save_previews_var = tk.BooleanVar(value=True)
        self.save_checkpoints_var = tk.BooleanVar(value=True)
        self.local_weight_mode_var = tk.StringVar(value=WEIGHT_MODE_BY_VALUE[WEIGHT_MODE_PRESET])
        self.local_checkpoint_var = tk.StringVar(value='')
        self.image_folder_var = tk.StringVar(value='')
        self.local_flow_summary_var = tk.StringVar(
            value=scenario_compact_flow(LOCAL_SCENARIOS['Local mock']),
        )
        self.local_override_scope_var = tk.StringVar(
            value=local_override_scope_text(LOCAL_SCENARIOS['Local mock']),
        )
        self.local_planned_models_var = tk.StringVar(value=planned_real_candidate_brief_summary())
        self.tail_var = tk.StringVar(value=str(DEFAULT_LOG_TAIL))
        self.ssh_scenario_var = tk.StringVar(value=scenario_choice_label('WSL mock', WSL_SCENARIOS['WSL mock']))
        self.ssh_config_var = tk.StringVar(value=DEFAULT_WSL_CONFIG_PATH)
        self.ssh_db_var = tk.StringVar(value=DEFAULT_WSL_DB_PATH)
        self.ssh_experiment_var = tk.StringVar(value=DEFAULT_WSL_EXPERIMENT_ID)
        self.ssh_server_var = tk.StringVar(value=DEFAULT_GPU_SERVER_NAME)
        self.ssh_flow_summary_var = tk.StringVar(
            value=scenario_compact_flow(WSL_SCENARIOS['WSL mock']),
        )
        self.ssh_run_summary_var = tk.StringVar(value='Applied Experiment: WSL mock')
        self.run_queue_summary_var = tk.StringVar(
            value=f'Scheduled Runs (0/{MAX_RUN_QUEUE_ITEMS}) - select experiments, then add them to the queue',
        )
        self.ssh_replace_existing_var = tk.BooleanVar(value=False)
        self.ssh_save_csv_var = tk.BooleanVar(value=True)
        self.ssh_save_summary_var = tk.BooleanVar(value=True)
        self.ssh_save_previews_var = tk.BooleanVar(value=True)
        self.ssh_save_checkpoints_var = tk.BooleanVar(value=True)
        self.ssh_collect_mode_var = tk.StringVar(value=COLLECT_MODE_BY_VALUE[COLLECT_MODE_WEIGHTS])
        self.ssh_checkpoint_collect_var = tk.StringVar(value=CHECKPOINT_COLLECT_BY_VALUE[CHECKPOINT_COLLECT_BEST])
        self.ssh_weight_mode_var = tk.StringVar(value=WEIGHT_MODE_BY_VALUE[WEIGHT_MODE_PRESET])
        self.ssh_checkpoint_var = tk.StringVar(value='')
        self.ssh_tail_var = tk.StringVar(value=str(DEFAULT_LOG_TAIL))
        self.ssh_native_wrappers_var = tk.BooleanVar(value=True)
        self.ssh_native_params_var = tk.StringVar(value=DEFAULT_NATIVE_PARAMS_PATH)
        self.ssh_collect_mode_var.trace_add('write', lambda *_: self._refresh_collect_option_state())
        self.ssh_save_checkpoints_var.trace_add('write', lambda *_: self._refresh_collect_option_state())
        self.ssh_epochs_var = tk.StringVar(value='')
        self.ssh_batch_size_var = tk.StringVar(value='')
        self.ssh_classification_batch_size_var = tk.StringVar(value='')
        self.ssh_detection_image_size_var = tk.StringVar(value='')
        self.ssh_classification_image_size_var = tk.StringVar(value='')
        self.ssh_learning_rate_var = tk.StringVar(value='')
        self.ssh_early_stopping_patience_var = tk.StringVar(value='')
        self.ssh_early_stopping_min_delta_var = tk.StringVar(value='')
        self.ssh_external_timeout_var = tk.StringVar(value='')
        self.ssh_dataset_source_var = tk.StringVar(value=DATASET_SOURCE_REMOTE_PRESTAGED)
        self.ssh_remote_dataset_root_var = tk.StringVar(value=DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT)
        self.vast_profile_name_var = tk.StringVar(value=DEFAULT_GPU_SERVER_NAME)
        self.vast_ssh_command_var = tk.StringVar(value='')
        self.vast_host_var = tk.StringVar(value='')
        self.vast_port_var = tk.StringVar(value='')
        self.vast_username_var = tk.StringVar(value='root')
        self.vast_key_path_var = tk.StringVar(value=str(Path.home() / '.ssh' / 'id_ed25519'))
        self.vast_remote_workspace_var = tk.StringVar(value='/workspace/ironflow')
        self.remote_runtime_mode_var = tk.StringVar(value=REMOTE_RUNTIME_BY_VALUE[REMOTE_RUNTIME_NATIVE])
        self.container_image_var = tk.StringVar(value=DEFAULT_GPU_DOCKER_IMAGE)
        self.gpu_candidate_var = tk.StringVar(value=first_gpu_candidate_label)
        self.recommended_combination_var = tk.StringVar(value=first_recommended_label)
        self.recommended_summary_var = tk.StringVar(value=self._recommended_summary_text())
        self.readiness_matrix_var = tk.StringVar(value=gui_readiness_matrix_text())
        self.compare_db_var = tk.StringVar(value=DEFAULT_DB_PATH)
        self.compare_summary_var = tk.StringVar(value=RESULTS_EMPTY_GUIDE)
        self.progress_var = tk.StringVar(value='idle')
        self.progress_value_var = tk.DoubleVar(value=0.0)
        self.compare_rows: list[ExperimentComparisonRow] = []
        self.comparison_plot_dir: Path | None = None
        self.status_var = tk.StringVar(value='idle')

        self._build()
        self._load_run_queue_state()
        self.root.after(100, self._drain_messages)

    def _build(self) -> None:
        self.root.title('IronFlow Engine')
        self.root.geometry('1120x760')
        self.root.minsize(960, 680)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        notebook = ttk.Notebook(outer, height=500)
        notebook.grid(row=0, column=0, sticky='ew')
        self.notebook = notebook

        recommendation_fields = ttk.Frame(notebook, padding=6)
        notebook.add(recommendation_fields, text='Experiments')
        recommendation_fields.columnconfigure(1, weight=1)
        recommendation_fields.rowconfigure(2, weight=1)
        ttk.Label(recommendation_fields, text='Select Experiment').grid(row=0, column=0, sticky='w', padx=(6, 4), pady=4)
        recommendation_selector = ttk.Combobox(
            recommendation_fields,
            textvariable=self.recommended_combination_var,
            values=list(self.recommended_label_map),
            state='readonly',
        )
        recommendation_selector.grid(row=0, column=1, columnspan=3, sticky='ew', padx=4, pady=4)
        recommendation_selector.bind('<<ComboboxSelected>>', lambda _event: self._refresh_recommended_summary())
        self._flow_summary_row(recommendation_fields, row=1, variable=self.recommended_summary_var)
        self.recommended_tree = self._recommended_tree(recommendation_fields, row=2)
        self.recommended_tree.bind('<<TreeviewSelect>>', lambda _event: self._select_recommended_from_tree())
        self._populate_recommended_tree()
        self._button_row(recommendation_fields, row=3, buttons=[
            ('Add Selected to Queue', self.apply_recommended_to_wsl),
            ('Clear', self.clear_recommended_template),
        ])

        ssh_fields = self._scrollable_notebook_tab(notebook, text='Run')
        self.run_tab = ssh_fields
        ssh_fields.columnconfigure(1, weight=1)
        ssh_fields.rowconfigure(0, weight=1)
        self._run_queue_panel(ssh_fields, row=0)
        self._button_row(ssh_fields, row=2, buttons=[
            ('Run Selected', self.ssh_run_selected),
            ('Run All', self.ssh_run_all),
            ('Stop', self.stop_running_experiment),
            ('Preview', self.ssh_preview),
            ('Clear Log', self.clear_output),
        ])
        ttk.Label(ssh_fields, textvariable=self.ssh_run_summary_var, wraplength=900).grid(
            row=3,
            column=0,
            columnspan=5,
            sticky='ew',
            padx=6,
            pady=(4, 0),
        )
        self._flow_summary_row(ssh_fields, row=4, variable=self.ssh_flow_summary_var)
        self.ssh_scenario_detail_tree = self._scenario_detail_tree(ssh_fields, row=5)
        self._populate_scenario_detail_tree(
            tree=self.ssh_scenario_detail_tree,
            rows=scenario_detail_rows(WSL_SCENARIOS['WSL mock']),
        )
        self._path_row(ssh_fields, row=6, label='Config', variable=self.ssh_config_var, filetypes=[('YAML', '*.yaml *.yml'), ('All', '*.*')])
        self._path_row(ssh_fields, row=7, label='DB', variable=self.ssh_db_var, filetypes=[('SQLite', '*.sqlite *.sqlite3'), ('All', '*.*')])
        ttk.Label(ssh_fields, text='Server').grid(row=8, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Entry(ssh_fields, textvariable=self.ssh_server_var).grid(row=8, column=1, sticky='ew', padx=4, pady=4)
        ttk.Checkbutton(ssh_fields, text='Native wrappers', variable=self.ssh_native_wrappers_var).grid(
            row=8,
            column=2,
            sticky='w',
            padx=(10, 4),
            pady=4,
        )
        self._vast_command_row(ssh_fields, row=9)
        self._path_row(
            ssh_fields,
            row=10,
            label='Native Params',
            variable=self.ssh_native_params_var,
            filetypes=[('YAML', '*.yaml *.yml'), ('All', '*.*')],
            initialdir=NATIVE_PARAMS_DIR,
        )
        self._dataset_source_row(ssh_fields, row=11)
        self._run_options_row(ssh_fields, row=12)
        ttk.Label(ssh_fields, text='Experiment').grid(row=13, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Entry(ssh_fields, textvariable=self.ssh_experiment_var).grid(row=13, column=1, sticky='ew', padx=4, pady=4)
        ttk.Checkbutton(ssh_fields, text='Replace existing', variable=self.ssh_replace_existing_var).grid(row=13, column=2, sticky='w', padx=(10, 4), pady=4)
        ttk.Label(ssh_fields, text='Tail').grid(row=13, column=3, sticky='w', padx=(10, 4), pady=4)
        ttk.Spinbox(ssh_fields, from_=1, to=5000, textvariable=self.ssh_tail_var, width=8).grid(row=13, column=4, sticky='w', padx=4, pady=4)
        self._output_options_row(
            ssh_fields,
            row=14,
            save_csv=self.ssh_save_csv_var,
            save_summary=self.ssh_save_summary_var,
            save_previews=self.ssh_save_previews_var,
            save_checkpoints=self.ssh_save_checkpoints_var,
        )
        self._collect_options_row(ssh_fields, row=15)
        self._weight_options_row(
            ssh_fields,
            row=16,
            mode_variable=self.ssh_weight_mode_var,
            checkpoint_variable=self.ssh_checkpoint_var,
        )
        self._vast_profile_rows(ssh_fields, start_row=17)

        results_fields = ttk.Frame(notebook, padding=6)
        notebook.add(results_fields, text='Results')
        results_fields.columnconfigure(1, weight=1)
        results_fields.rowconfigure(2, weight=1)
        results_fields.rowconfigure(4, weight=1)
        self._path_row(
            results_fields,
            row=0,
            label='DB',
            variable=self.compare_db_var,
            filetypes=[('SQLite', '*.sqlite *.sqlite3'), ('All', '*.*')],
        )
        self._button_row(results_fields, row=1, buttons=[
            ('Refresh', self.refresh_result_comparison),
            ('Use Local DB', self.use_local_db_for_comparison),
            ('Use WSL DB', self.use_wsl_db_for_comparison),
            ('Generate Plots', self.generate_result_plots),
            ('Open Plots', self.open_result_plots),
            ('Open Result', self.open_selected_compare_result),
            ('Open Summary', self.open_selected_compare_summary),
        ])
        self.compare_tree = self._comparison_tree(results_fields, row=2)
        self.compare_tree.bind('<<TreeviewSelect>>', lambda _event: self._refresh_selected_compare_detail())
        ttk.Label(results_fields, textvariable=self.compare_summary_var, wraplength=900).grid(
            row=3,
            column=0,
            columnspan=4,
            sticky='ew',
            padx=6,
            pady=(8, 4),
        )
        self.compare_metric_tree = self._metric_detail_tree(results_fields, row=4)

        advanced_fields = ttk.Frame(notebook, padding=0)
        notebook.add(advanced_fields, text='Advanced')
        advanced_notebook = ttk.Notebook(advanced_fields)
        advanced_notebook.pack(fill=tk.BOTH, expand=True)

        candidate_fields = ttk.Frame(advanced_notebook, padding=6)
        advanced_notebook.add(candidate_fields, text='Candidates')
        candidate_fields.columnconfigure(1, weight=1)
        self._gpu_candidate_row(candidate_fields, row=0)

        stage_fields = ttk.Frame(advanced_notebook, padding=6)
        advanced_notebook.add(stage_fields, text='SSH Stages')
        stage_fields.columnconfigure(1, weight=1)
        self._button_row(stage_fields, row=0, buttons=[
            ('Check', self.ssh_check),
            ('Deps', self.ssh_dependency_check),
            ('Install Deps', self.ssh_install_deps),
            ('Prepare Data', self.ssh_prepare_dataset),
            ('Prepare', self.ssh_prepare),
            ('Execute', self.ssh_execute),
            ('Cancel', lambda: self.ssh_stage('cancel')),
            ('Bootstrap', lambda: self.ssh_stage('bootstrap')),
            ('Upload', lambda: self.ssh_stage('upload')),
            ('Submit', lambda: self.ssh_stage('submit')),
            ('Status', lambda: self.ssh_stage('status')),
            ('Logs', lambda: self.ssh_stage('logs')),
            ('Collect', lambda: self.ssh_stage('collect')),
        ])
        ttk.Label(
            stage_fields,
            text='Use these only when one-click Run fails or a specific SSH stage needs inspection.',
            wraplength=900,
        ).grid(row=1, column=0, columnspan=5, sticky='ew', padx=6, pady=(8, 4))

        fields = ttk.Frame(advanced_notebook, padding=6)
        advanced_notebook.add(fields, text='Local Smoke')
        fields.columnconfigure(1, weight=1)

        self._scenario_row(
            fields,
            row=0,
            variable=self.local_scenario_var,
            labels=list(self.local_scenario_label_map),
            callback=lambda _event: self._apply_local_scenario(),
        )
        self._flow_summary_row(fields, row=1, variable=self.local_flow_summary_var)
        self.local_scenario_detail_tree = self._scenario_detail_tree(fields, row=2)
        self._populate_scenario_detail_tree(
            tree=self.local_scenario_detail_tree,
            rows=scenario_detail_rows(LOCAL_SCENARIOS['Local mock']),
        )
        self._detect_crop_classify_selector_rows(fields, start_row=3)
        self._planned_models_row(fields, row=8, variable=self.local_planned_models_var)
        self._path_row(fields, row=9, label='Config', variable=self.config_var, filetypes=[('YAML', '*.yaml *.yml'), ('All', '*.*')])
        self._path_row(fields, row=10, label='DB', variable=self.db_var, filetypes=[('SQLite', '*.sqlite *.sqlite3'), ('All', '*.*')])
        ttk.Label(fields, text='Experiment').grid(row=11, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Entry(fields, textvariable=self.experiment_var).grid(row=11, column=1, sticky='ew', padx=4, pady=4)
        ttk.Checkbutton(fields, text='Replace existing', variable=self.replace_existing_var).grid(row=11, column=2, sticky='w', padx=(10, 4), pady=4)
        ttk.Label(fields, text='Tail').grid(row=11, column=3, sticky='w', padx=(10, 4), pady=4)
        ttk.Spinbox(fields, from_=1, to=5000, textvariable=self.tail_var, width=8).grid(row=11, column=4, sticky='w', padx=4, pady=4)

        self._output_options_row(
            fields,
            row=12,
            save_csv=self.save_csv_var,
            save_summary=self.save_summary_var,
            save_previews=self.save_previews_var,
            save_checkpoints=self.save_checkpoints_var,
        )
        self._weight_options_row(
            fields,
            row=13,
            mode_variable=self.local_weight_mode_var,
            checkpoint_variable=self.local_checkpoint_var,
        )
        self._folder_row(fields, row=14, label='Image Folder', variable=self.image_folder_var)
        self._button_row(fields, row=15, buttons=[
            ('Run', self.run_local_mock),
            ('Status', self.status),
            ('Logs', self.logs),
            ('Collect', self.collect),
            ('Clear', self.clear_output),
        ])
        self._refresh_local_model_selector_state()

        readiness_fields = ttk.Frame(advanced_notebook, padding=6)
        advanced_notebook.add(readiness_fields, text='Readiness')
        readiness_fields.columnconfigure(0, weight=1)
        ttk.Label(
            readiness_fields,
            textvariable=self.readiness_matrix_var,
            justify='left',
            wraplength=900,
        ).grid(row=0, column=0, sticky='ew', padx=6, pady=6)

        self._result_actions_row(outer, row=1)

        output_frame = ttk.Frame(outer)
        output_frame.grid(row=2, column=0, sticky='nsew', pady=(8, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output = tk.Text(output_frame, wrap='word', height=20, undo=False)
        self.output.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(output_frame, orient='vertical', command=self.output.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.output.configure(yscrollcommand=scrollbar.set)

        progress_frame = ttk.Frame(outer)
        progress_frame.grid(row=3, column=0, sticky='ew', pady=(6, 0))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Label(progress_frame, textvariable=self.progress_var, anchor='w').grid(row=0, column=0, sticky='ew')
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            maximum=100,
            variable=self.progress_value_var,
            length=220,
        )
        self.progress_bar.grid(row=1, column=0, sticky='ew', pady=(3, 0))

        status = ttk.Label(outer, textvariable=self.status_var, anchor='w')
        status.grid(row=4, column=0, sticky='ew', pady=(4, 0))

    def _scrollable_notebook_tab(self, notebook: ttk.Notebook, *, text: str) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=0)
        notebook.add(tab, text=text)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        content = ttk.Frame(canvas, padding=6)
        content_window = canvas.create_window((0, 0), window=content, anchor='nw')

        def update_scroll_region(_event: object | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox('all'))

        def update_content_width(event: object) -> None:
            width = getattr(event, 'width', 0)
            if width:
                canvas.itemconfigure(content_window, width=width)

        def on_mousewheel(event: object) -> None:
            delta = getattr(event, 'delta', 0)
            if delta:
                canvas.yview_scroll(int(-1 * (delta / 120)), 'units')

        content.bind('<Configure>', update_scroll_region)
        canvas.bind('<Configure>', update_content_width)
        canvas.bind('<Enter>', lambda _event: canvas.bind_all('<MouseWheel>', on_mousewheel))
        canvas.bind('<Leave>', lambda _event: canvas.unbind_all('<MouseWheel>'))
        setattr(content, 'notebook_tab', tab)

        return content

    def _scenario_row(
        self,
        parent: ttk.Frame,
        row: int,
        variable: tk.StringVar,
        labels: list[str],
        callback: object,
        label: str = 'Scenario / Model',
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        scenario = ttk.Combobox(parent, textvariable=variable, values=labels, state='readonly', width=92)
        scenario.grid(row=row, column=1, columnspan=3, sticky='ew', padx=4, pady=4)
        scenario.bind('<<ComboboxSelected>>', callback)

    def _gpu_candidate_row(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text='Candidate').grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        selector = ttk.Combobox(
            parent,
            textvariable=self.gpu_candidate_var,
            values=list(self.gpu_candidate_label_map),
            state='readonly',
            width=92,
        )
        selector.grid(row=row, column=1, columnspan=3, sticky='ew', padx=4, pady=4)
        ttk.Button(parent, text='Apply', command=self.apply_candidate_to_wsl).grid(
            row=row,
            column=4,
            sticky='ew',
            padx=4,
            pady=4,
        )

    def _flow_summary_row(self, parent: ttk.Frame, row: int, variable: tk.StringVar) -> None:
        ttk.Label(parent, text='Flow').grid(row=row, column=0, sticky='nw', padx=(6, 4), pady=4)
        ttk.Label(parent, textvariable=variable, wraplength=760).grid(
            row=row,
            column=1,
            columnspan=4,
            sticky='ew',
            padx=4,
            pady=4,
        )

    def _scenario_detail_tree(self, parent: ttk.Frame, row: int) -> ttk.Treeview:
        ttk.Label(parent, text='Models').grid(row=row, column=0, sticky='nw', padx=(6, 4), pady=4)
        tree = ttk.Treeview(
            parent,
            columns=('stage', 'model', 'input', 'adapter'),
            show='headings',
            height=3,
        )
        tree.heading('stage', text='Stage')
        tree.heading('model', text='Model')
        tree.heading('input', text='Input')
        tree.heading('adapter', text='Adapter')
        tree.column('stage', width=110, stretch=False)
        tree.column('model', width=220, stretch=True)
        tree.column('input', width=130, stretch=False)
        tree.column('adapter', width=260, stretch=True)
        tree.grid(row=row, column=1, columnspan=4, sticky='ew', padx=4, pady=4)

        return tree

    def _recommended_tree(self, parent: ttk.Frame, row: int) -> ttk.Treeview:
        ttk.Label(parent, text='Experiments').grid(row=row, column=0, sticky='nw', padx=(6, 4), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=4, sticky='nsew', padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ('priority', 'type', 'state', 'experiment', 'objective')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=8, selectmode='extended')
        headings = {
            'priority': '#',
            'type': 'Type',
            'state': 'State',
            'experiment': 'Experiment',
            'objective': 'Objective',
        }
        widths = {
            'priority': 45,
            'type': 95,
            'state': 120,
            'experiment': 300,
            'objective': 360,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], stretch=column in {'experiment', 'objective'})
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=scrollbar.set)

        return tree

    def _run_queue_panel(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, textvariable=self.run_queue_summary_var).grid(
            row=row,
            column=0,
            columnspan=5,
            sticky='ew',
            padx=6,
            pady=(4, 0),
        )
        frame = ttk.Frame(parent)
        frame.grid(row=row + 1, column=0, columnspan=5, sticky='nsew', padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ('order', 'status', 'experiment', 'model', 'current', 'options')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=3, selectmode='browse')
        headings = {
            'order': '#',
            'status': 'Status',
            'experiment': 'Experiment',
            'model': 'Model',
            'current': 'Current',
            'options': 'Options',
        }
        widths = {
            'order': 45,
            'status': 85,
            'experiment': 260,
            'model': 300,
            'current': 300,
            'options': 260,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], stretch=column in {'experiment', 'model', 'options'})
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=scrollbar.set)
        tree.bind('<<TreeviewSelect>>', lambda _event: self._select_run_queue_from_tree())
        self.run_queue_tree = tree

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        ttk.Button(buttons, text='Remove Selected', command=self.remove_selected_queue_item).grid(row=0, column=0, sticky='w', padx=(0, 6))
        ttk.Button(buttons, text='Clear Queue', command=self.clear_run_queue).grid(row=0, column=1, sticky='w', padx=(0, 6))

    def _comparison_tree(self, parent: ttk.Frame, row: int) -> ttk.Treeview:
        ttk.Label(parent, text='Experiments').grid(row=row, column=0, sticky='nw', padx=(6, 4), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=4, sticky='nsew', padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ('id', 'status', 'best', 'time', 'epoch', 'acc', 'macro_f1', 'map50', 'map50_95', 'p95_ms', 'name')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=8)
        headings = {
            'id': 'Experiment',
            'status': 'Status',
            'best': 'Best',
            'time': 'Time',
            'epoch': 'Epoch',
            'acc': 'Acc',
            'macro_f1': 'Macro F1',
            'map50': 'mAP50',
            'map50_95': 'mAP50-95',
            'p95_ms': 'P95 ms',
            'name': 'Name',
        }
        widths = {
            'id': 230,
            'status': 90,
            'best': 120,
            'time': 80,
            'epoch': 60,
            'acc': 70,
            'macro_f1': 80,
            'map50': 70,
            'map50_95': 80,
            'p95_ms': 80,
            'name': 260,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], stretch=column in {'id', 'name'})
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=scrollbar.set)

        return tree

    def _metric_detail_tree(self, parent: ttk.Frame, row: int) -> ttk.Treeview:
        ttk.Label(parent, text='Metrics').grid(row=row, column=0, sticky='nw', padx=(6, 4), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=4, sticky='nsew', padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = (
            'epoch',
            'train_loss',
            'val_loss',
            'accuracy',
            'precision',
            'recall',
            'macro_precision',
            'macro_recall',
            'macro_f1',
            'class_recall',
            'class_ap50',
            'object_accuracy',
            'map50',
            'map50_95',
            'num_predictions',
            'num_gt',
            'mask_count',
            'mask_coverage',
            'embedding_count',
            'embedding_dim',
            'retrieval_map',
            'neighbor_purity',
            'review_hit_rate',
            'label_error_rate',
            'latency_ms_per_image',
            'p95_latency_ms',
            'gpu_memory_mb',
            'lr',
        )
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=6)
        for column, width in [
            ('epoch', 70),
            ('train_loss', 110),
            ('val_loss', 110),
            ('accuracy', 100),
            ('precision', 100),
            ('recall', 100),
            ('macro_precision', 130),
            ('macro_recall', 120),
            ('macro_f1', 100),
            ('class_recall', 110),
            ('class_ap50', 100),
            ('object_accuracy', 130),
            ('map50', 100),
            ('map50_95', 110),
            ('num_predictions', 130),
            ('num_gt', 90),
            ('mask_count', 110),
            ('mask_coverage', 120),
            ('embedding_count', 130),
            ('embedding_dim', 120),
            ('retrieval_map', 110),
            ('neighbor_purity', 130),
            ('review_hit_rate', 120),
            ('label_error_rate', 120),
            ('latency_ms_per_image', 150),
            ('p95_latency_ms', 130),
            ('gpu_memory_mb', 130),
            ('lr', 100),
        ]:
            heading = 'min_class_recall' if column == 'class_recall' else column
            tree.heading(column, text=heading)
            tree.column(column, width=width, stretch=True)
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=scrollbar.set)

        return tree

    def _populate_scenario_detail_tree(
        self,
        *,
        tree: ttk.Treeview,
        rows: tuple[tuple[str, str, str, str], ...],
    ) -> None:
        for item_id in tree.get_children():
            tree.delete(item_id)
        for row in rows:
            tree.insert('', tk.END, values=row)

    def _detect_crop_classify_selector_rows(self, parent: ttk.Frame, start_row: int) -> None:
        ttk.Label(parent, text='Overrides').grid(row=start_row, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Label(parent, textvariable=self.local_override_scope_var).grid(
            row=start_row,
            column=1,
            columnspan=4,
            sticky='ew',
            padx=4,
            pady=4,
        )
        rows = [
            ('Detector', self.local_detector_var, self.local_detector_choice_map),
            ('Crop', self.local_crop_adapter_var, self.local_crop_adapter_choice_map),
            ('Classifier', self.local_classifier_var, self.local_classifier_choice_map),
            ('Augment', self.local_augmentation_policy_var, self.local_augmentation_policy_choice_map),
        ]
        for offset, (label, variable, choices) in enumerate(rows):
            row = start_row + offset + 1
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
            selector = ttk.Combobox(
                parent,
                textvariable=variable,
                values=list(choices),
                state='disabled',
            )
            selector.grid(row=row, column=1, columnspan=3, sticky='ew', padx=4, pady=4)
            selector.bind('<<ComboboxSelected>>', lambda _event: self._update_local_flow_summary_from_selection())
            self.local_model_selector_widgets.append(selector)

    def _planned_models_row(self, parent: ttk.Frame, row: int, variable: tk.StringVar) -> None:
        ttk.Label(parent, text='Planned').grid(row=row, column=0, sticky='nw', padx=(6, 4), pady=4)
        ttk.Label(parent, textvariable=variable, wraplength=760).grid(
            row=row,
            column=1,
            columnspan=4,
            sticky='ew',
            padx=4,
            pady=4,
        )

    def _output_options_row(
        self,
        parent: ttk.Frame,
        row: int,
        save_csv: tk.BooleanVar,
        save_summary: tk.BooleanVar,
        save_previews: tk.BooleanVar,
        save_checkpoints: tk.BooleanVar,
    ) -> None:
        ttk.Label(parent, text='Output').grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=4, sticky='w', padx=4, pady=4)
        ttk.Checkbutton(frame, text='CSV', variable=save_csv).grid(row=0, column=0, sticky='w', padx=(0, 10))
        ttk.Checkbutton(frame, text='Summary', variable=save_summary).grid(row=0, column=1, sticky='w', padx=(0, 10))
        ttk.Checkbutton(frame, text='Previews', variable=save_previews).grid(row=0, column=2, sticky='w', padx=(0, 10))
        ttk.Checkbutton(frame, text='Weights (.pt)', variable=save_checkpoints).grid(row=0, column=3, sticky='w')

    def _weight_options_row(
        self,
        parent: ttk.Frame,
        row: int,
        mode_variable: tk.StringVar,
        checkpoint_variable: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text='Weights').grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        selector = ttk.Combobox(
            parent,
            textvariable=mode_variable,
            values=list(WEIGHT_MODE_LABELS),
            state='readonly',
            width=20,
        )
        selector.grid(row=row, column=1, sticky='w', padx=4, pady=4)
        ttk.Entry(parent, textvariable=checkpoint_variable).grid(row=row, column=2, sticky='ew', padx=4, pady=4)
        ttk.Button(
            parent,
            text='Browse',
            command=lambda: self._browse_file(
                variable=checkpoint_variable,
                filetypes=[
                    ('Model weights', '*.pt *.pth *.safetensors *.bin'),
                    ('All', '*.*'),
                ],
            ),
        ).grid(row=row, column=3, sticky='e', padx=4, pady=4)

    def _collect_options_row(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text='Collect').grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=4, sticky='w', padx=4, pady=4)
        ttk.Label(frame, text='Mode').grid(row=0, column=0, sticky='w', padx=(0, 4))
        selector = ttk.Combobox(
            frame,
            textvariable=self.ssh_collect_mode_var,
            values=list(COLLECT_MODE_LABELS),
            state='readonly',
            width=12,
        )
        selector.grid(row=0, column=1, sticky='w', padx=(0, 12))
        selector.bind('<<ComboboxSelected>>', lambda _event: self._refresh_collect_option_state())
        self.ssh_checkpoint_collect_label = ttk.Label(frame, text='Weight files')
        self.ssh_checkpoint_collect_label.grid(row=0, column=2, sticky='w', padx=(0, 4))
        self.ssh_checkpoint_collect_selector = ttk.Combobox(
            frame,
            textvariable=self.ssh_checkpoint_collect_var,
            values=list(CHECKPOINT_COLLECT_LABELS),
            state='readonly',
            width=12,
        )
        self.ssh_checkpoint_collect_selector.grid(row=0, column=3, sticky='w', padx=(0, 12))
        self._refresh_collect_option_state()

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        filetypes: list[tuple[str, str]],
        initialdir: str | Path | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=2, sticky='ew', padx=4, pady=4)
        ttk.Button(
            parent,
            text='Browse',
            command=lambda: self._browse_file(variable=variable, filetypes=filetypes, initialdir=initialdir),
        ).grid(row=row, column=3, sticky='e', padx=4, pady=4)

    def _run_options_row(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=5, sticky='ew', padx=4, pady=2)
        ttk.Label(frame, text='Run Options').grid(row=0, column=0, sticky='w', padx=(2, 8), pady=2)
        fields = [
            ('Epochs', self.ssh_epochs_var, 6),
            ('Det Batch', self.ssh_batch_size_var, 8),
            ('Cls Batch', self.ssh_classification_batch_size_var, 8),
            ('Det Img', self.ssh_detection_image_size_var, 7),
            ('Cls Img', self.ssh_classification_image_size_var, 7),
            ('Timeout', self.ssh_external_timeout_var, 8),
        ]
        for index, (label, variable, width) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(row=0, column=index * 2 - 1, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(frame, textvariable=variable, width=width).grid(row=0, column=index * 2, sticky='w', padx=(0, 8), pady=2)
        ttk.Label(frame, text='Training').grid(row=1, column=0, sticky='w', padx=(2, 8), pady=2)
        training_fields = [
            ('LR', self.ssh_learning_rate_var, 10),
            ('Patience', self.ssh_early_stopping_patience_var, 8),
            ('Min Delta', self.ssh_early_stopping_min_delta_var, 10),
        ]
        for index, (label, variable, width) in enumerate(training_fields, start=1):
            ttk.Label(frame, text=label).grid(row=1, column=index * 2 - 1, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(frame, textvariable=variable, width=width).grid(row=1, column=index * 2, sticky='w', padx=(0, 8), pady=2)

    def _dataset_source_row(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=5, sticky='ew', padx=4, pady=2)
        frame.columnconfigure(3, weight=1)
        ttk.Label(frame, text='Dataset Source').grid(row=0, column=0, sticky='w', padx=(2, 8), pady=2)
        ttk.Combobox(
            frame,
            textvariable=self.ssh_dataset_source_var,
            values=list(DATASET_SOURCE_LABELS),
            state='readonly',
            width=22,
        ).grid(row=0, column=1, sticky='w', padx=(0, 8), pady=2)
        ttk.Label(frame, text='Remote Root').grid(row=0, column=2, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.ssh_remote_dataset_root_var).grid(
            row=0,
            column=3,
            sticky='ew',
            padx=(0, 8),
            pady=2,
        )
        ttk.Button(
            frame,
            text='Original',
            command=self._use_original_remote_dataset_root,
        ).grid(row=0, column=4, sticky='e', padx=(0, 4), pady=2)
        ttk.Button(
            frame,
            text='Weak Aug',
            command=self._use_weak_aug_remote_dataset_root,
        ).grid(row=0, column=5, sticky='e', padx=(0, 4), pady=2)

    def _use_original_remote_dataset_root(self) -> None:
        self.ssh_dataset_source_var.set(DATASET_SOURCE_REMOTE_PRESTAGED)
        self.ssh_remote_dataset_root_var.set(DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT)
        self.status_var.set('dataset root preset: original prepared dataset')

    def _use_weak_aug_remote_dataset_root(self) -> None:
        self.ssh_dataset_source_var.set(DATASET_SOURCE_REMOTE_PRESTAGED)
        self.ssh_remote_dataset_root_var.set(WEAK_AUG_REMOTE_PRESTAGED_DATASET_ROOT)
        self.status_var.set('dataset root preset: weak augmentation dataset')

    def _folder_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=2, sticky='ew', padx=4, pady=4)
        ttk.Button(
            parent,
            text='Browse',
            command=lambda: self._browse_folder(variable=variable),
        ).grid(row=row, column=3, sticky='e', padx=4, pady=4)

    def _vast_command_row(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text='Vast SSH').grid(row=row, column=0, sticky='w', padx=(6, 4), pady=4)
        ttk.Entry(parent, textvariable=self.vast_ssh_command_var).grid(
            row=row,
            column=1,
            columnspan=2,
            sticky='ew',
            padx=4,
            pady=4,
        )
        ttk.Button(parent, text='Use Command', command=self.apply_vast_ssh_command).grid(
            row=row,
            column=3,
            sticky='ew',
            padx=4,
            pady=4,
        )
        ttk.Button(parent, text='Copy Public Key', command=self.copy_local_ssh_public_key).grid(
            row=row,
            column=4,
            sticky='ew',
            padx=4,
            pady=4,
        )

    def _vast_profile_rows(self, parent: ttk.Frame, start_row: int) -> None:
        ttk.Separator(parent).grid(row=start_row, column=0, columnspan=5, sticky='ew', padx=6, pady=(10, 4))
        frame = ttk.Frame(parent)
        frame.grid(row=start_row + 1, column=0, columnspan=5, sticky='ew', padx=4, pady=4)
        frame.columnconfigure(3, weight=1)
        frame.columnconfigure(7, weight=1)

        ttk.Label(frame, text='Vast Details').grid(row=0, column=0, sticky='w', padx=(0, 4), pady=2)
        ttk.Label(frame, text='Name').grid(row=1, column=0, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.vast_profile_name_var, width=16).grid(row=1, column=1, sticky='ew', padx=(0, 8), pady=2)
        ttk.Label(frame, text='Host').grid(row=1, column=2, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.vast_host_var, width=18).grid(row=1, column=3, sticky='ew', padx=(0, 8), pady=2)
        ttk.Label(frame, text='Port').grid(row=1, column=4, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.vast_port_var, width=8).grid(row=1, column=5, sticky='w', padx=(0, 8), pady=2)
        ttk.Label(frame, text='User').grid(row=1, column=6, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.vast_username_var, width=10).grid(row=1, column=7, sticky='ew', padx=(0, 8), pady=2)

        ttk.Label(frame, text='Key').grid(row=2, column=0, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.vast_key_path_var).grid(row=2, column=1, columnspan=3, sticky='ew', padx=(0, 8), pady=2)
        ttk.Button(
            frame,
            text='Browse',
            command=lambda: self._browse_file(self.vast_key_path_var, [('SSH key', '*'), ('All', '*.*')]),
        ).grid(row=2, column=4, sticky='ew', padx=(0, 8), pady=2)
        ttk.Button(frame, text='Copy Public Key', command=self.copy_local_ssh_public_key).grid(row=2, column=5, sticky='ew', padx=(0, 8), pady=2)
        ttk.Label(frame, text='Workspace').grid(row=2, column=6, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.vast_remote_workspace_var).grid(row=2, column=7, sticky='ew', padx=(0, 8), pady=2)
        ttk.Button(frame, text='Register Vast', command=self.register_vast_profile).grid(row=2, column=8, sticky='ew', pady=2)

        ttk.Label(frame, text='Runtime').grid(row=3, column=0, sticky='w', padx=(0, 4), pady=2)
        ttk.Combobox(
            frame,
            textvariable=self.remote_runtime_mode_var,
            values=list(REMOTE_RUNTIME_LABELS),
            state='readonly',
            width=16,
        ).grid(row=3, column=1, sticky='ew', padx=(0, 8), pady=2)
        ttk.Label(frame, text='Docker Image').grid(row=3, column=2, sticky='w', padx=(0, 4), pady=2)
        ttk.Entry(frame, textvariable=self.container_image_var).grid(
            row=3,
            column=3,
            columnspan=5,
            sticky='ew',
            padx=(0, 8),
            pady=2,
        )

    def _button_row(self, parent: ttk.Frame, row: int, buttons: list[tuple[str, object]]) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=4, sticky='ew', padx=4, pady=(8, 6))
        for index, (label, callback) in enumerate(buttons):
            button = ttk.Button(frame, text=label, command=callback)
            button.grid(row=index // 5, column=index % 5, sticky='w', padx=(0, 6), pady=2)
            if label == 'Stop':
                button.configure(state=tk.DISABLED)
                self.stop_buttons.append(button)
            elif label != 'Clear':
                self.command_buttons.append(button)

    def _result_actions_row(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='ew', pady=(8, 0))
        buttons = [
            ('Open Result', self.open_result_dir),
            ('Open Previews', self.open_preview_dir),
            ('Open Summary', self.open_summary),
            ('Copy Predictions', self.copy_prediction_paths),
        ]
        for index, (label, callback) in enumerate(buttons):
            button = ttk.Button(frame, text=label, command=callback)
            button.grid(row=0, column=index, sticky='w', padx=(0, 6))
            self.result_buttons.append(button)
        self._refresh_result_buttons()

    def _browse_file(
        self,
        variable: tk.StringVar,
        filetypes: list[tuple[str, str]],
        initialdir: str | Path | None = None,
    ) -> None:
        selected = filedialog.askopenfilename(
            initialdir=self._browse_initial_dir(variable=variable, fallback=initialdir),
            filetypes=filetypes,
        )
        if selected:
            try:
                variable.set(str(Path(selected).resolve().relative_to(self.project_dir.resolve())))
            except ValueError:
                variable.set(selected)

    def _browse_initial_dir(self, *, variable: tk.StringVar, fallback: str | Path | None = None) -> Path:
        value = variable.get().strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = self.project_dir / path
            candidate = path if path.is_dir() else path.parent
            if candidate.exists():
                return candidate

        if fallback is not None:
            fallback_path = Path(fallback)
            if not fallback_path.is_absolute():
                fallback_path = self.project_dir / fallback_path
            if fallback_path.exists():
                return fallback_path

        return self.project_dir

    def _browse_folder(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=self.project_dir)
        if selected:
            try:
                variable.set(str(Path(selected).resolve().relative_to(self.project_dir.resolve())))
            except ValueError:
                variable.set(selected)

    def run_local_mock(self) -> None:
        dataset_dir = self._validated_image_folder()
        if dataset_dir == '':
            return
        config_transform = self._local_config_transform(dataset_dir=dataset_dir)
        if config_transform == '':
            return
        config_path = self._effective_config_path(
            source_config_path=self.config_var.get().strip(),
            experiment_id=self.experiment_var.get().strip(),
            options=self._local_output_options(),
            weight_options=self._local_weight_options(),
            prefix='local',
            dataset_dir=dataset_dir,
            config_transform=config_transform,
        )
        if config_path is None:
            return
        command = self.builder.run(
            config_path=config_path,
            experiment_id=self.experiment_var.get().strip(),
            replace_existing=self.replace_existing_var.get(),
        )
        self._start_command(name='run', command=command)

    def status(self) -> None:
        command = self.builder.status(
            experiment_id=self.experiment_var.get().strip(),
            db_path=self.db_var.get().strip(),
        )
        self._start_command(name='status', command=command)

    def logs(self) -> None:
        command = self.builder.logs(
            experiment_id=self.experiment_var.get().strip(),
            db_path=self.db_var.get().strip(),
            tail=self._tail_value(),
        )
        self._start_command(name='logs', command=command)

    def collect(self) -> None:
        command = self.builder.collect(
            experiment_id=self.experiment_var.get().strip(),
            db_path=self.db_var.get().strip(),
        )
        self._start_command(name='collect', command=command)

    def ssh_check(self) -> None:
        command = self.ssh_builder.server_check(
            server_name=self.ssh_server_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
        )
        self._start_command(name='ssh server check', command=command, experiment_id=self.ssh_server_var.get().strip())

    def ssh_dependency_check(self) -> None:
        command = self.ssh_builder.remote_task_adapter_dependency_check(
            server_name=self.ssh_server_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
            dependency_profile=dependency_profile_for_config_path(
                self.ssh_config_var.get().strip(),
                native_wrappers=self.ssh_native_wrappers_var.get(),
            ),
        )
        self._start_command(
            name='ssh dependency check',
            command=command,
            experiment_id=self.ssh_server_var.get().strip(),
        )

    def ssh_install_deps(self) -> None:
        dependency_profile = dependency_profile_for_config_path(
            self.ssh_config_var.get().strip(),
            native_wrappers=self.ssh_native_wrappers_var.get(),
        )
        command = self._dependency_step_command(
            server_name=self.ssh_server_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
            dependency_profile=dependency_profile,
        )
        self._start_command(
            name='ssh install deps',
            command=command,
            experiment_id=self.ssh_server_var.get().strip(),
        )

    def _ensure_vast_connection_fields(self) -> bool:
        if (not self.vast_host_var.get().strip() or not self.vast_port_var.get().strip()) and self.vast_ssh_command_var.get().strip():
            self.apply_vast_ssh_command()
        missing = []
        if not self.vast_host_var.get().strip():
            missing.append('Host')
        if not self.vast_port_var.get().strip():
            missing.append('Port')
        if not self.vast_username_var.get().strip():
            missing.append('User')
        key_text = self.vast_key_path_var.get().strip()
        key_path = Path(key_text).expanduser()
        if not key_text:
            missing.append('Key')
        elif not key_path.exists():
            self._append(f'SSH key not found: {key_path}\n')
            self.status_var.set('ssh key missing')
            return False
        if missing:
            self._append('Vast connection fields required: ' + ', '.join(missing) + '\n')
            self.status_var.set('vast connection incomplete')
            return False
        return True

    def ssh_prepare_dataset(self) -> None:
        if not self._ensure_vast_connection_fields():
            return
        timeout_seconds = GUI_COMMAND_TIMEOUT_SECONDS['ssh prepare data']
        command = [
            sys.executable,
            'scripts/prepare_vast_dataset_remote.py',
            '--host',
            self.vast_host_var.get().strip(),
            '--port',
            self.vast_port_var.get().strip(),
            '--user',
            self.vast_username_var.get().strip(),
            '--key',
            self.vast_key_path_var.get().strip(),
            '--workspace',
            self.vast_remote_workspace_var.get().strip() or '/workspace/ironflow',
            '--project-root',
            str(self.project_dir),
            '--timeout',
            str(timeout_seconds),
        ]
        self._start_command(
            name='ssh prepare data',
            command=command,
            experiment_id=self.ssh_server_var.get().strip(),
            timeout_seconds=timeout_seconds,
            streaming=True,
        )
    def _dependency_step_command(self, *, server_name: str, db_path: str, dependency_profile: str) -> list[str]:
        if self._uses_docker_runtime():
            return self.ssh_builder.remote_task_adapter_dependency_check(
                server_name=server_name,
                db_path=db_path,
                dependency_profile=dependency_profile,
                timeout_seconds=90,
            )
        return self.ssh_builder.install_deps(
            server_name=server_name,
            db_path=db_path,
            dependency_profile=dependency_profile,
        )

    def ssh_preview(self) -> None:
        config_path = self._effective_config_path(
            source_config_path=self.ssh_config_var.get().strip(),
            experiment_id=self.ssh_experiment_var.get().strip(),
            options=self._ssh_output_options(),
            weight_options=self._ssh_weight_options(),
            config_transform=self._ssh_config_transform(),
            prefix='wsl',
        )
        if config_path is None:
            return

        self._append(self._ssh_preview_text(config_path=config_path))
        self.status_var.set('ssh preview ready')

    def ssh_prepare(self) -> None:
        if self._guard_deferred_recommended_run('prepare'):
            return
        if self._guard_existing_ssh_experiment_id(action='prepare'):
            return
        config_path = self._effective_config_path(
            source_config_path=self.ssh_config_var.get().strip(),
            experiment_id=self.ssh_experiment_var.get().strip(),
            options=self._ssh_output_options(),
            weight_options=self._ssh_weight_options(),
            config_transform=self._ssh_config_transform(),
            prefix='wsl',
        )
        if config_path is None:
            return
        command = self.ssh_builder.prepare(
            config_path=config_path,
            server_name=self.ssh_server_var.get().strip(),
            experiment_id=self.ssh_experiment_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
            replace_existing=self.ssh_replace_existing_var.get(),
        )
        self._start_command(name='ssh prepare', command=command, experiment_id=self.ssh_experiment_var.get().strip())

    def ssh_execute(self) -> None:
        if self._guard_deferred_recommended_run('execute'):
            return
        if self._guard_existing_ssh_experiment_id(action='execute'):
            return
        config_path = self._effective_config_path(
            source_config_path=self.ssh_config_var.get().strip(),
            experiment_id=self.ssh_experiment_var.get().strip(),
            options=self._ssh_output_options(),
            weight_options=self._ssh_weight_options(),
            config_transform=self._ssh_config_transform(),
            prefix='wsl',
        )
        if config_path is None:
            return
        execute_timeout_seconds = self._ssh_execute_stage_timeout_seconds()
        command = self.ssh_builder.execute(
            config_path=config_path,
            server_name=self.ssh_server_var.get().strip(),
            experiment_id=self.ssh_experiment_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
            replace_existing=self.ssh_replace_existing_var.get(),
            timeout_seconds=execute_timeout_seconds,
        )
        self._start_command(
            name='ssh execute',
            command=command,
            experiment_id=self.ssh_experiment_var.get().strip(),
            timeout_seconds=self._ssh_execute_gui_timeout_seconds(stage_timeout_seconds=execute_timeout_seconds),
        )

    def ssh_run_selected(self) -> None:
        item = self._selected_or_current_queue_item()
        if item is None:
            return
        self._run_ssh_queue_items(items=[item], name='ssh run selected')

    def ssh_run_all(self) -> None:
        self._sync_active_queue_item_from_vars()
        items = [
            item
            for item in getattr(self, 'run_queue_items', [])
            if item.status not in {'running', 'collected'}
        ]
        if not items:
            item = self._selected_or_current_queue_item()
            items = [item] if item is not None else []
        if not items:
            self.status_var.set('select an experiment first')
            return
        self._run_ssh_queue_items(items=items, name='ssh run all')

    def _selected_or_current_queue_item(self) -> RunQueueItem | None:
        self._sync_active_queue_item_from_vars()
        queue_id = getattr(self, 'active_queue_item_id', None)
        if queue_id:
            try:
                return self._queue_item_by_id(queue_id)
            except KeyError:
                return None
        return self._queue_item_from_current_vars()

    def _run_ssh_queue_items(self, *, items: list[RunQueueItem], name: str) -> None:
        if not self._ensure_vast_profile_for_run():
            return
        resolved_server_profile = self._string_var_value('ssh_server_var', DEFAULT_GPU_SERVER_NAME)
        items = [
            replace(item, server_profile=item.server_profile or resolved_server_profile)
            for item in items
        ]
        if self._guard_duplicate_queue_experiment_ids(items=items):
            return
        if not self._register_vast_profile_for_queue_items(items=items):
            return
        items = [replace(item, status='pending', progress_detail='', error_message='') for item in items]
        self.run_queue_items = [
            next((updated for updated in items if updated.queue_id == item.queue_id), item)
            for item in getattr(self, 'run_queue_items', [])
        ]
        self._refresh_run_queue_tree()
        self._save_run_queue_state()

        steps: list[CommandSequenceStep] = []
        installed_dependency_profiles: set[tuple[str, str]] = set()
        for item_index, item in enumerate(items, start=1):
            self._apply_queue_item_to_vars(item)
            if self._guard_deferred_recommended_run('execute'):
                return
            if self._guard_existing_ssh_experiment_id(action=name):
                return
            config_path = self._effective_config_path(
                source_config_path=self.ssh_config_var.get().strip(),
                experiment_id=self.ssh_experiment_var.get().strip(),
                options=self._ssh_output_options(),
                weight_options=self._ssh_weight_options(),
                config_transform=self._ssh_config_transform(),
                prefix='wsl',
            )
            if config_path is None:
                return

            dependency_profile = dependency_profile_for_config_path(
                config_path,
                native_wrappers=self.ssh_native_wrappers_var.get(),
            )
            item_steps: list[CommandSequenceStep] = []
            dependency_key = (self.ssh_server_var.get().strip(), dependency_profile)
            if dependency_key not in installed_dependency_profiles:
                installed_dependency_profiles.add(dependency_key)
                item_steps.append(
                    CommandSequenceStep(
                        name='ssh install deps',
                        command=self._dependency_step_command(
                            server_name=self.ssh_server_var.get().strip(),
                            db_path=self.ssh_db_var.get().strip(),
                            dependency_profile=dependency_profile,
                        ),
                        timeout_seconds=GUI_COMMAND_TIMEOUT_SECONDS['ssh install deps'],
                        queue_item_id=item.queue_id,
                    ),
                )

            run_timeout_seconds = self._ssh_run_timeout_seconds()
            item_steps.extend([
                CommandSequenceStep(
                    name='ssh prepare',
                    command=self.ssh_builder.prepare(
                        config_path=config_path,
                        server_name=self.ssh_server_var.get().strip(),
                        experiment_id=self.ssh_experiment_var.get().strip(),
                        db_path=self.ssh_db_var.get().strip(),
                        replace_existing=self.ssh_replace_existing_var.get(),
                    ),
                    timeout_seconds=GUI_COMMAND_TIMEOUT_SECONDS['ssh prepare'],
                    queue_item_id=item.queue_id,
                ),
                CommandSequenceStep(
                    name='ssh bootstrap',
                    command=self.ssh_builder.stage(
                        stage_name='bootstrap',
                        experiment_id=self.ssh_experiment_var.get().strip(),
                        db_path=self.ssh_db_var.get().strip(),
                        timeout_seconds=CLI_STAGE_TIMEOUT_SECONDS['ssh bootstrap'],
                    ),
                    timeout_seconds=GUI_COMMAND_TIMEOUT_SECONDS['ssh bootstrap'],
                    queue_item_id=item.queue_id,
                ),
                CommandSequenceStep(
                    name='ssh upload',
                    command=self.ssh_builder.stage(
                        stage_name='upload',
                        experiment_id=self.ssh_experiment_var.get().strip(),
                        db_path=self.ssh_db_var.get().strip(),
                        timeout_seconds=CLI_STAGE_TIMEOUT_SECONDS['ssh upload'],
                    ),
                    timeout_seconds=GUI_COMMAND_TIMEOUT_SECONDS['ssh upload'],
                    queue_item_id=item.queue_id,
                ),
                CommandSequenceStep(
                    name='ssh run',
                    command=self.ssh_builder.stage(
                        stage_name='run',
                        experiment_id=self.ssh_experiment_var.get().strip(),
                        db_path=self.ssh_db_var.get().strip(),
                        timeout_seconds=run_timeout_seconds,
                    ),
                    timeout_seconds=None if run_timeout_seconds is None else run_timeout_seconds + 60,
                    queue_item_id=item.queue_id,
                ),
                CommandSequenceStep(
                    name='ssh collect',
                    command=self.ssh_builder.stage(
                        stage_name='collect',
                        experiment_id=self.ssh_experiment_var.get().strip(),
                        db_path=self.ssh_db_var.get().strip(),
                        timeout_seconds=CLI_STAGE_TIMEOUT_SECONDS['ssh collect'],
                    ),
                    timeout_seconds=GUI_COMMAND_TIMEOUT_SECONDS['ssh collect'],
                    queue_item_id=item.queue_id,
                    final_queue_status='collected',
                ),
            ])
            item_step_total = len(QUEUE_RUN_STAGE_ORDER)
            steps.extend(
                replace(
                    step,
                    queue_step_index=QUEUE_RUN_STAGE_INDEX.get(step.name, step_index),
                    queue_step_total=item_step_total,
                    queue_item_index=item_index,
                    queue_item_total=len(items),
                )
                for step_index, step in enumerate(item_steps, start=1)
            )

        if not steps:
            self.status_var.set('no runnable queue items')
            return
        self._start_command_sequence(
            name=name,
            steps=steps,
            experiment_id=items[0].experiment_id if items else None,
        )

    def _guard_existing_ssh_experiment_id(self, *, action: str) -> bool:
        if self.ssh_replace_existing_var.get():
            return False

        experiment_id = self.ssh_experiment_var.get().strip()
        db_path = self.ssh_db_var.get().strip()
        if not experiment_id or not db_path:
            return False

        try:
            record = SQLiteExperimentStorage(db_path=db_path).get_experiment(experiment_id=experiment_id)
        except Exception as error:
            self._append(f'\n[SSH {action}]\nfailed to inspect experiment DB: {type(error).__name__}: {error}\n')
            self.status_var.set('ssh experiment db check failed')
            return True

        if record is None:
            return False

        self._append(
            f'\n[SSH {action}]\n'
            f'experiment id already exists: {experiment_id}\n'
            f'current status: {record.status.value}\n'
            'Turn on "Replace existing" to reuse it, or enter a new Experiment id before running.\n'
        )
        self.status_var.set('ssh experiment id exists')
        return True

    def _guard_duplicate_queue_experiment_ids(self, *, items: list[RunQueueItem]) -> bool:
        seen: dict[tuple[str, str], RunQueueItem] = {}
        duplicates: list[tuple[str, str]] = []
        for item in items:
            key = (item.db_path.strip(), item.experiment_id.strip())
            if not all(key):
                continue
            if key in seen:
                duplicates.append(key)
                continue
            seen[key] = item
        if not duplicates:
            return False

        lines = [
            '\n[Queue]\nRun was not started because duplicate experiment ids were found.',
            'Use a unique Experiment value for each queue item. Replace existing only reuses an existing DB record; it does not make duplicate queued runs safe.',
            '',
            *[
                f'- db={db_path}; experiment_id={experiment_id}'
                for db_path, experiment_id in duplicates
            ],
        ]
        self._append('\n'.join(lines) + '\n')
        self.status_var.set('duplicate queue experiment id')
        return True

    def apply_vast_ssh_command(self) -> None:
        parsed = parse_vast_ssh_command(self.vast_ssh_command_var.get())
        if parsed is None:
            self.status_var.set('vast ssh command invalid')
            self._append(
                '\n[Vast SSH Command]\n'
                'Paste a Vast SSH command like: ssh -p 40022 root@203.0.113.10 -L 8080:localhost:8080\n'
            )
            return

        self.vast_host_var.set(parsed['host'])
        self.vast_port_var.set(parsed['port'])
        self.vast_username_var.set(parsed['username'])
        self.status_var.set('vast ssh command applied')
        self._append(
            '\n[Vast SSH Command]\n'
            f"Applied host={parsed['host']} port={parsed['port']} user={parsed['username']}\n"
        )

    def copy_local_ssh_public_key(self) -> None:
        public_key_path = self._vast_public_key_path()
        if public_key_path is None:
            return
        if not public_key_path.exists():
            if not self._ensure_vast_public_key(public_key_path=public_key_path):
                return

        public_key = public_key_path.read_text(encoding='utf-8-sig').strip()
        if not public_key:
            self.status_var.set('ssh public key empty')
            self._append(f'\n[Vast SSH Key]\nPublic key is empty: {public_key_path}\n')
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(public_key)
        self.status_var.set('copied ssh public key')
        self._append(f'\n[Vast SSH Key]\nCopied public key from {public_key_path}\n')

    def _ensure_vast_public_key(self, *, public_key_path: Path) -> bool:
        private_key_path = self._vast_private_key_path(public_key_path=public_key_path)
        if private_key_path.exists():
            return self._derive_vast_public_key(
                private_key_path=private_key_path,
                public_key_path=public_key_path,
            )
        return self._generate_vast_ssh_key_pair(
            private_key_path=private_key_path,
            public_key_path=public_key_path,
        )

    def _vast_private_key_path(self, *, public_key_path: Path) -> Path:
        key_path = Path(self.vast_key_path_var.get().strip()).expanduser()
        if key_path.suffix == '.pub':
            return Path(str(key_path)[:-4])
        return key_path

    def _derive_vast_public_key(self, *, private_key_path: Path, public_key_path: Path) -> bool:
        try:
            result = subprocess.run(
                ['ssh-keygen', '-y', '-f', str(private_key_path)],
                capture_output=True,
                check=False,
                text=True,
            )
        except FileNotFoundError:
            self.status_var.set('ssh-keygen missing')
            self._append('\n[Vast SSH Key]\nssh-keygen was not found. Install OpenSSH, then retry.\n')
            return False
        if result.returncode != 0:
            self.status_var.set('ssh public key create failed')
            stderr = result.stderr.strip() or result.stdout.strip() or 'unknown ssh-keygen error'
            self._append(
                '\n[Vast SSH Key]\n'
                f'Could not derive public key from existing private key: {private_key_path}\n'
                f'{stderr}\n'
            )
            return False

        public_key = result.stdout.strip()
        if not public_key:
            self.status_var.set('ssh public key create failed')
            self._append(f'\n[Vast SSH Key]\nssh-keygen returned an empty public key for {private_key_path}\n')
            return False
        public_key_path.parent.mkdir(parents=True, exist_ok=True)
        public_key_path.write_text(f'{public_key}\n', encoding='utf-8')
        self._append(f'\n[Vast SSH Key]\nCreated public key from existing private key: {public_key_path}\n')
        return True

    def _generate_vast_ssh_key_pair(self, *, private_key_path: Path, public_key_path: Path) -> bool:
        private_key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    'ssh-keygen',
                    '-t',
                    'ed25519',
                    '-f',
                    str(private_key_path),
                    '-N',
                    '',
                    '-C',
                    'ironflow-vast',
                ],
                capture_output=True,
                check=False,
                text=True,
            )
        except FileNotFoundError:
            self.status_var.set('ssh-keygen missing')
            self._append('\n[Vast SSH Key]\nssh-keygen was not found. Install OpenSSH, then retry.\n')
            return False
        if result.returncode != 0:
            self.status_var.set('ssh key generation failed')
            stderr = result.stderr.strip() or result.stdout.strip() or 'unknown ssh-keygen error'
            self._append(
                '\n[Vast SSH Key]\n'
                f'Could not create SSH key pair at {private_key_path}\n'
                f'{stderr}\n'
            )
            return False
        if not public_key_path.exists():
            self.status_var.set('ssh public key missing')
            self._append(
                '\n[Vast SSH Key]\n'
                f'ssh-keygen completed, but public key was not found: {public_key_path}\n'
            )
            return False
        self._append(f'\n[Vast SSH Key]\nCreated new local SSH key pair: {private_key_path}\n')
        return True

    def _vast_public_key_path(self) -> Path | None:
        key_path_text = self.vast_key_path_var.get().strip()
        if not key_path_text:
            self.status_var.set('ssh key path required')
            self._append('\n[Vast SSH Key]\nSet the local SSH key path first.\n')
            return None

        key_path = Path(key_path_text).expanduser()
        if key_path.suffix == '.pub':
            return key_path
        return Path(f'{key_path}.pub')

    def register_vast_profile(self) -> None:
        self._save_vast_profile_to_db()

    def _ensure_vast_profile_for_run(self) -> bool:
        if self.vast_ssh_command_var.get().strip():
            parsed = parse_vast_ssh_command(self.vast_ssh_command_var.get())
            if parsed is None:
                self.status_var.set('vast ssh command invalid')
                self._append(
                    '\n[Vast SSH Command]\n'
                    'Run was not started. Paste a Vast SSH command like: '
                    'ssh -p 40022 root@203.0.113.10 -L 8080:localhost:8080\n'
                )
                return False
            self.vast_host_var.set(parsed['host'])
            self.vast_port_var.set(parsed['port'])
            self.vast_username_var.set(parsed['username'])

        if not self._has_vast_profile_input():
            return True

        return self._save_vast_profile_to_db()

    def _has_vast_profile_input(self) -> bool:
        return any(
            value.get().strip()
            for value in (
                self.vast_ssh_command_var,
                self.vast_host_var,
                self.vast_port_var,
            )
        )

    def _save_vast_profile_to_db(self) -> bool:
        profile_path = self._write_vast_profile_file()
        if profile_path is None:
            return False
        try:
            record = ServerProfileLoader().load_file(path=profile_path)
            validation = ServerProfileValidator().validate(record=record, check_paths=True)
            if not validation.is_valid:
                self.status_var.set('vast profile invalid')
                lines = [
                    '\n[Vast Profile]\nProfile was not registered.',
                    *[
                        f'error: {issue.field}: {issue.message}'
                        for issue in validation.errors
                    ],
                ]
                self._append('\n'.join(lines) + '\n')
                return False
            SQLiteExperimentStorage(db_path=self.ssh_db_var.get().strip()).save_server(record=record)
        except Exception as error:
            self.status_var.set('vast profile failed')
            self._append(f'\n[Vast Profile]\nProfile registration failed: {type(error).__name__}: {error}\n')
            return False

        self.ssh_server_var.set(self.vast_profile_name_var.get().strip())
        self.status_var.set('vast profile registered')
        self._append(
            '\n[Vast Profile]\n'
            f'Registered {self.vast_profile_name_var.get().strip()} into {self.ssh_db_var.get().strip()}\n'
        )
        return True

    def _register_vast_profile_for_queue_items(self, *, items: list[RunQueueItem]) -> bool:
        profile_name = self._string_var_value('ssh_server_var') or self._string_var_value('vast_profile_name_var')
        if not profile_name:
            return True
        profile_path = self._vast_profile_path(profile_name)
        if not profile_path.exists():
            return True
        try:
            record = ServerProfileLoader().load_file(path=profile_path)
            validation = ServerProfileValidator().validate(record=record, check_paths=False)
            if not validation.is_valid:
                self.status_var.set('vast profile invalid')
                lines = [
                    '\n[Vast Profile]\nQueue run was not started because the saved profile is invalid.',
                    *[
                        f'error: {issue.field}: {issue.message}'
                        for issue in validation.errors
                    ],
                ]
                self._append('\n'.join(lines) + '\n')
                return False
            db_paths = sorted({
                item.db_path.strip()
                for item in items
                if item.db_path.strip() and item.server_profile.strip() == record.name
            })
            for db_path in db_paths:
                SQLiteExperimentStorage(db_path=db_path).save_server(record=record)
        except Exception as error:
            self.status_var.set('vast profile failed')
            self._append(f'\n[Vast Profile]\nQueue profile registration failed: {type(error).__name__}: {error}\n')
            return False

        if db_paths:
            self._append(
                '\n[Vast Profile]\n'
                f'Registered {record.name} into {len(db_paths)} queue DB(s).\n'
            )
        return True

    def _write_vast_profile_file(self) -> Path | None:
        name = self.vast_profile_name_var.get().strip()
        host = self.vast_host_var.get().strip()
        port_text = self.vast_port_var.get().strip()
        username = self.vast_username_var.get().strip()
        key_path = self.vast_key_path_var.get().strip()
        remote_workspace = self.vast_remote_workspace_var.get().strip()
        if not name or not host or not port_text or not username or not remote_workspace:
            self.status_var.set('vast profile fields required')
            self._append('\n[Vast Profile]\nName, host, port, user, and workspace are required.\n')
            return None
        try:
            port = int(port_text)
        except ValueError:
            self.status_var.set('vast port invalid')
            self._append('\n[Vast Profile]\nPort must be a number.\n')
            return None

        profile_path = self._vast_profile_path(name)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'server': {
                'name': name,
                'server_type': 'vast_manual',
                'host': host,
                'port': port,
                'username': username,
                'key_path': key_path,
                'remote_workspace': remote_workspace,
                'runtime': {
                    'container_mode': self._remote_runtime_mode(),
                    'container_image': self._container_image(),
                },
            },
        }
        profile_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
        self._append(f'\n[Vast Profile]\nWrote {profile_path}\n')
        return profile_path

    def _vast_profile_path(self, name: str) -> Path:
        return getattr(self, 'project_dir', Path.cwd()) / 'runs' / 'server_profiles' / f'{_safe_filename(name)}.local.yaml'

    def ssh_stage(self, stage_name: str) -> None:
        if stage_name in DEFERRED_RECOMMENDED_RUN_STAGES and self._guard_deferred_recommended_run(stage_name):
            return
        command_name = f'ssh {stage_name}'
        stage_timeout_seconds = self._ssh_stage_timeout_seconds(stage_name=stage_name, command_name=command_name)
        command = self.ssh_builder.stage(
            stage_name=stage_name,
            experiment_id=self.ssh_experiment_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
            tail=self._tail_value(variable=self.ssh_tail_var),
            timeout_seconds=stage_timeout_seconds,
        )
        self._start_command(
            name=command_name,
            command=command,
            experiment_id=self.ssh_experiment_var.get().strip(),
            timeout_seconds=None if stage_timeout_seconds is None else stage_timeout_seconds + 60,
        )

    def stop_running_experiment(self) -> None:
        running_item: RunQueueItem | None = None
        running_queue_item_id = getattr(self, 'current_running_queue_item_id', None)
        if running_queue_item_id:
            try:
                running_item = self._queue_item_by_id(running_queue_item_id)
            except KeyError:
                running_item = None
        experiment_id = running_item.experiment_id if running_item is not None else self.ssh_experiment_var.get().strip()
        db_path = running_item.db_path if running_item is not None else self.ssh_db_var.get().strip()
        if not experiment_id:
            self._append('\n[Stop]\nexperiment id is required\n')
            return
        self.stop_requested = True
        local_cancelled = self.runner.cancel_current()
        self._append(
            '\n[Stop]\n'
            f'local_cli_cancelled: {local_cancelled}\n'
            'requesting remote cancel on the selected server...\n'
        )
        self.status_var.set('stopping remote job')
        self.progress_var.set('Stopping: cancelling remote job on server')
        self._set_progress_value(0)
        for button in self.stop_buttons:
            button.configure(state=tk.DISABLED)
        command = self.ssh_builder.stage(
            stage_name='cancel',
            experiment_id=experiment_id,
            db_path=db_path,
            timeout_seconds=30,
        )
        thread = threading.Thread(target=self._cancel_remote_worker, args=(command, running_queue_item_id), daemon=True)
        thread.start()

    def _guard_deferred_recommended_run(self, action: str) -> bool:
        message = self._deferred_recommended_run_message()
        if message is None:
            return False

        self._append(f'\n[Blocked]\n{action} was not started.\n{message}\n')
        self.status_var.set('deferred audit guarded')
        return True

    def _deferred_recommended_run_message(self) -> str | None:
        active_combination_id = getattr(self, 'active_recommended_combination_id', None)
        config_path = self.ssh_config_var.get().strip().replace('\\', '/')
        experiment_id = self.ssh_experiment_var.get().strip()
        db_path = self.ssh_db_var.get().strip()

        for combination_id, reason in DEFERRED_RECOMMENDED_COMBINATION_REASONS.items():
            safe_id = _safe_filename(combination_id)
            if (
                active_combination_id == combination_id
                or safe_id in config_path
                or safe_id in experiment_id
                or safe_id in db_path
            ):
                return reason

        return None

    def _ssh_preview_text(self, *, config_path: str) -> str:
        server_name = self.ssh_server_var.get().strip()
        db_path = self.ssh_db_var.get().strip()
        experiment_id = self.ssh_experiment_var.get().strip()
        dependency_profile = dependency_profile_for_config_path(
            self.ssh_config_var.get().strip(),
            native_wrappers=self.ssh_native_wrappers_var.get(),
        )
        check_command = self.ssh_builder.server_check(
            server_name=server_name,
            db_path=db_path,
        )
        deps_command = self.ssh_builder.remote_task_adapter_dependency_check(
            server_name=server_name,
            db_path=db_path,
            dependency_profile=dependency_profile,
        )
        prepare_command = self.ssh_builder.prepare(
            config_path=config_path,
            server_name=server_name,
            experiment_id=experiment_id,
            db_path=db_path,
            replace_existing=self.ssh_replace_existing_var.get(),
        )
        execute_command = self.ssh_builder.execute(
            config_path=config_path,
            server_name=server_name,
            experiment_id=experiment_id,
            db_path=db_path,
            replace_existing=self.ssh_replace_existing_var.get(),
        )
        staged_commands = [
            (
                stage.title(),
                self.ssh_builder.stage(
                    stage_name=stage,
                    experiment_id=experiment_id,
                    db_path=db_path,
                    tail=self._tail_value(variable=self.ssh_tail_var),
                    timeout_seconds=CLI_STAGE_TIMEOUT_SECONDS.get(f'ssh {stage}'),
                ),
            )
            for stage in ('bootstrap', 'upload', 'submit', 'status', 'logs', 'collect')
        ]
        lines = [
            '',
            '[Run Preview]',
            'This preview does not open SSH, upload files, or start paid GPU work.',
            f'Server: {server_name}',
            f'Experiment: {experiment_id}',
            f'DB: {db_path}',
            f'Source config: {self.ssh_config_var.get().strip()}',
            f'Effective config: {config_path}',
            f'Remote runtime: {self._remote_runtime_mode()}',
            f'Docker image: {self._container_image() if self._uses_docker_runtime() else "-"}',
            f'Dependency profile: {dependency_profile}',
            f'Native wrappers: {"on" if self.ssh_native_wrappers_var.get() else "off"}',
            f'Native params: {self.ssh_native_params_var.get().strip() or "-"}',
            f'Run options: {self._run_options_summary()}',
            f'Replace existing: {"on" if self.ssh_replace_existing_var.get() else "off"}',
            '',
            'Recommended staged flow:',
            '1. Check',
            '2. Deps',
            '3. Bootstrap',
            '4. Upload',
            '5. Submit',
            '6. Status / Logs',
            '7. Collect',
            '',
            'One-shot alternative:',
            f'Execute: {self._display_command(execute_command)}',
            '',
            'Staged commands:',
            f'Check: {self._display_command(check_command)}',
            f'Deps: {self._display_command(deps_command)}',
            f'Prepare: {self._display_command(prepare_command)}',
        ]
        deferred_message = self._deferred_recommended_run_message()
        if deferred_message is not None:
            flow_index = lines.index('Recommended staged flow:')
            lines[flow_index:flow_index] = ['', f'Guard: {deferred_message}']
        lines.extend(
            f'{name}: {self._display_command(command)}'
            for name, command in staged_commands
        )
        lines.extend([
            '',
            'Guide:',
            '- Use Check before Bootstrap when the server profile changed.',
            '- Use Deps after Bootstrap or when native package state is uncertain.',
            '- Use staged Bootstrap/Upload/Submit for paid GPU runs; Execute is convenient for WSL or short smoke runs.',
            '- Keep Replace existing off for a new experiment id.',
            '- Collect only after Status shows the remote job is finished.',
            '- After Collect, open Results, choose Use WSL DB, then Refresh.',
            '',
        ])

        return '\n'.join(lines)

    def _run_options_summary(self) -> str:
        labels = [
            ('epochs', 'ssh_epochs_var'),
            ('det_batch', 'ssh_batch_size_var'),
            ('cls_batch', 'ssh_classification_batch_size_var'),
            ('det_img', 'ssh_detection_image_size_var'),
            ('cls_img', 'ssh_classification_image_size_var'),
            ('lr', 'ssh_learning_rate_var'),
            ('patience', 'ssh_early_stopping_patience_var'),
            ('min_delta', 'ssh_early_stopping_min_delta_var'),
            ('timeout', 'ssh_external_timeout_var'),
        ]
        parts = []
        for label, variable_name in labels:
            variable = getattr(self, variable_name, None)
            value = variable.get().strip() if variable is not None else ''
            if value:
                parts.append(f'{label}={value}')
        if self._uses_docker_runtime():
            parts.append(f'docker={self._container_image()}')

        return ', '.join(parts) if parts else 'default/YAML'

    def _display_command(self, command: list[str]) -> str:
        return subprocess.list2cmdline(command)

    def clear_output(self) -> None:
        self.output.delete('1.0', tk.END)

    def _run_queue_state_path(self) -> Path:
        return getattr(self, 'project_dir', Path.cwd()) / RUN_QUEUE_STATE_PATH

    def _load_run_queue_state(self) -> None:
        path = self._run_queue_state_path()
        if not path.exists():
            self._refresh_run_queue_tree()
            return
        try:
            payload = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, json.JSONDecodeError):
            self._refresh_run_queue_tree()
            return
        raw_items = payload.get('items') if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            self._refresh_run_queue_tree()
            return
        items = [
            RunQueueItem.from_dict(item)
            for item in raw_items[:MAX_RUN_QUEUE_ITEMS]
            if isinstance(item, dict)
        ]
        items = [
            replace(
                item,
                status='interrupted',
                error_message='GUI was closed while this item was marked running. Check remote Status/Logs before rerun.',
            )
            if item.status == 'running'
            else item
            for item in items
        ]
        self.run_queue_items = self._renumber_queue_items(items)
        active_id = payload.get('active_queue_item_id') if isinstance(payload, dict) else None
        self.active_queue_item_id = str(active_id) if isinstance(active_id, str) else None
        if self.active_queue_item_id not in {item.queue_id for item in self.run_queue_items}:
            self.active_queue_item_id = self.run_queue_items[0].queue_id if self.run_queue_items else None
        self._refresh_run_queue_tree()
        if self.active_queue_item_id:
            self._apply_queue_item_to_vars(self._queue_item_by_id(self.active_queue_item_id))
        if any(item.status == 'interrupted' for item in self.run_queue_items):
            self.status_var.set('queue has interrupted item')

    def _save_run_queue_state(self, *, throttle: bool = False) -> None:
        if not hasattr(self, 'run_queue_items'):
            return
        if not hasattr(self, 'project_dir'):
            return
        now = time.monotonic()
        if throttle and now - getattr(self, '_last_queue_state_save_at', 0.0) < 2.0:
            return
        path = self._run_queue_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'active_queue_item_id': getattr(self, 'active_queue_item_id', None),
            'items': [item.to_dict() for item in self.run_queue_items],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        self._last_queue_state_save_at = now

    def _renumber_queue_items(self, items: list[RunQueueItem]) -> list[RunQueueItem]:
        return [
            replace(item, order=index)
            for index, item in enumerate(items, start=1)
        ]

    def _queue_item_by_id(self, queue_id: str) -> RunQueueItem:
        for item in self.run_queue_items:
            if item.queue_id == queue_id:
                return item
        raise KeyError(f'unknown queue item: {queue_id}')

    def _set_queue_item(self, updated: RunQueueItem) -> None:
        self.run_queue_items = [
            updated if item.queue_id == updated.queue_id else item
            for item in self.run_queue_items
        ]

    def _sync_active_queue_item_from_vars(self, *, refresh: bool = True, save: bool = True) -> None:
        if getattr(self, '_loading_queue_item', False):
            return
        queue_id = getattr(self, 'active_queue_item_id', None)
        if not queue_id:
            return
        try:
            current = self._queue_item_by_id(queue_id)
        except KeyError:
            return
        self._set_queue_item(self._queue_item_from_current_vars(base=current))
        if refresh:
            self._refresh_run_queue_tree()
        if save:
            self._save_run_queue_state()

    def _queue_item_from_current_vars(self, *, base: RunQueueItem | None = None) -> RunQueueItem:
        if base is None:
            queue_id = f'custom_{len(getattr(self, "run_queue_items", [])) + 1:02d}'
            order = len(getattr(self, 'run_queue_items', [])) + 1
            source_type = 'custom'
            source_id = None
            display_name = self._string_var_value('ssh_experiment_var') or 'Custom experiment'
            stage = 'custom'
        else:
            queue_id = base.queue_id
            order = base.order
            source_type = base.source_type
            source_id = base.source_id
            display_name = base.display_name
            stage = base.stage

        return RunQueueItem(
            queue_id=queue_id,
            order=order,
            source_type=source_type,
            source_id=source_id,
            display_name=display_name,
            stage=stage,
            config_path=self._string_var_value('ssh_config_var'),
            db_path=self._string_var_value('ssh_db_var'),
            native_params_path=self._string_var_value('ssh_native_params_var', DEFAULT_NATIVE_PARAMS_PATH),
            experiment_id=self._string_var_value('ssh_experiment_var'),
            server_profile=self._string_var_value('ssh_server_var', DEFAULT_GPU_SERVER_NAME),
            native_wrappers=self._bool_var_value('ssh_native_wrappers_var', True),
            replace_existing=self._bool_var_value('ssh_replace_existing_var', False),
            save_csv=self._bool_var_value('ssh_save_csv_var', True),
            save_summary=self._bool_var_value('ssh_save_summary_var', True),
            save_previews=self._bool_var_value('ssh_save_previews_var', True),
            save_checkpoints=self._bool_var_value('ssh_save_checkpoints_var', True),
            collect_mode=self._string_var_value('ssh_collect_mode_var', COLLECT_MODE_BY_VALUE[COLLECT_MODE_WEIGHTS]),
            checkpoint_collect_mode=self._string_var_value(
                'ssh_checkpoint_collect_var',
                CHECKPOINT_COLLECT_BY_VALUE[CHECKPOINT_COLLECT_BEST],
            ),
            weight_mode=self._string_var_value('ssh_weight_mode_var', WEIGHT_MODE_BY_VALUE[WEIGHT_MODE_PRESET]),
            checkpoint_path=self._string_var_value('ssh_checkpoint_var'),
            epochs=self._string_var_value('ssh_epochs_var'),
            det_batch=self._string_var_value('ssh_batch_size_var'),
            cls_batch=self._string_var_value('ssh_classification_batch_size_var'),
            det_img=self._string_var_value('ssh_detection_image_size_var'),
            cls_img=self._string_var_value('ssh_classification_image_size_var'),
            learning_rate=self._string_var_value('ssh_learning_rate_var'),
            early_stopping_patience=self._string_var_value('ssh_early_stopping_patience_var'),
            early_stopping_min_delta=self._string_var_value('ssh_early_stopping_min_delta_var'),
            timeout=self._string_var_value('ssh_external_timeout_var'),
            tail=self._string_var_value('ssh_tail_var', str(DEFAULT_LOG_TAIL)),
            dataset_source_mode=self._dataset_source_mode(),
            remote_dataset_root=self._remote_dataset_root(),
            remote_runtime_mode=self._remote_runtime_mode(),
            container_image=self._container_image(),
            active_recommended_combination_id=getattr(self, 'active_recommended_combination_id', None),
            active_gpu_candidate_id=getattr(self, 'active_gpu_candidate_id', None),
            status=base.status if base is not None else 'pending',
            result_dir=base.result_dir if base is not None else '',
            error_message=base.error_message if base is not None else '',
        )

    def _string_var_value(self, variable_name: str, default: str = '') -> str:
        variable = getattr(self, variable_name, None)
        if variable is None:
            return default
        return str(variable.get()).strip()

    def _bool_var_value(self, variable_name: str, default: bool = False) -> bool:
        variable = getattr(self, variable_name, None)
        if variable is None:
            return default
        return bool(variable.get())

    def _dataset_source_mode(self) -> str:
        mode = self._string_var_value('ssh_dataset_source_var', DATASET_SOURCE_REMOTE_PRESTAGED)
        return mode if mode in DATASET_SOURCE_LABELS else DATASET_SOURCE_REMOTE_PRESTAGED

    def _remote_dataset_root(self) -> str:
        root = self._string_var_value('ssh_remote_dataset_root_var', DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT)
        return _normalize_remote_prestaged_dataset_root(root)

    def _remote_runtime_mode(self) -> str:
        label = self._string_var_value('remote_runtime_mode_var', REMOTE_RUNTIME_BY_VALUE[REMOTE_RUNTIME_NATIVE])
        return REMOTE_RUNTIME_LABELS.get(label, REMOTE_RUNTIME_NATIVE)

    def _container_image(self) -> str:
        return self._string_var_value('container_image_var', DEFAULT_GPU_DOCKER_IMAGE)

    def _uses_docker_runtime(self) -> bool:
        return self._remote_runtime_mode() == REMOTE_RUNTIME_DOCKER

    def _apply_queue_item_to_vars(self, item: RunQueueItem) -> None:
        self._loading_queue_item = True
        try:
            self.active_queue_item_id = item.queue_id
            self.active_recommended_combination_id = item.active_recommended_combination_id
            self.active_gpu_candidate_id = item.active_gpu_candidate_id
            self._set_var_if_present('ssh_config_var', item.config_path)
            self._set_var_if_present('ssh_db_var', item.db_path)
            self._set_var_if_present('ssh_native_params_var', item.native_params_path)
            self._set_var_if_present('ssh_experiment_var', item.experiment_id)
            self._set_var_if_present('ssh_server_var', item.server_profile)
            self._set_var_if_present('ssh_native_wrappers_var', item.native_wrappers)
            self._set_var_if_present('ssh_replace_existing_var', item.replace_existing)
            self._set_var_if_present('ssh_save_csv_var', item.save_csv)
            self._set_var_if_present('ssh_save_summary_var', item.save_summary)
            self._set_var_if_present('ssh_save_previews_var', item.save_previews)
            self._set_var_if_present('ssh_save_checkpoints_var', item.save_checkpoints)
            self._set_var_if_present('ssh_collect_mode_var', item.collect_mode)
            self._set_var_if_present('ssh_checkpoint_collect_var', item.checkpoint_collect_mode)
            self._set_var_if_present('ssh_weight_mode_var', item.weight_mode)
            self._set_var_if_present('ssh_checkpoint_var', item.checkpoint_path)
            self._set_var_if_present('ssh_epochs_var', item.epochs)
            self._set_var_if_present('ssh_batch_size_var', item.det_batch)
            self._set_var_if_present('ssh_classification_batch_size_var', item.cls_batch)
            self._set_var_if_present('ssh_detection_image_size_var', item.det_img)
            self._set_var_if_present('ssh_classification_image_size_var', item.cls_img)
            self._set_var_if_present('ssh_learning_rate_var', item.learning_rate)
            self._set_var_if_present('ssh_early_stopping_patience_var', item.early_stopping_patience)
            self._set_var_if_present('ssh_early_stopping_min_delta_var', item.early_stopping_min_delta)
            self._set_var_if_present('ssh_external_timeout_var', item.timeout)
            self._set_var_if_present('ssh_tail_var', item.tail or str(DEFAULT_LOG_TAIL))
            self._set_var_if_present('ssh_dataset_source_var', item.dataset_source_mode or DATASET_SOURCE_REMOTE_PRESTAGED)
            self._set_var_if_present(
                'ssh_remote_dataset_root_var',
                item.remote_dataset_root or DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT,
            )
            self._set_var_if_present(
                'remote_runtime_mode_var',
                REMOTE_RUNTIME_BY_VALUE.get(item.remote_runtime_mode, REMOTE_RUNTIME_BY_VALUE[REMOTE_RUNTIME_NATIVE]),
            )
            self._set_var_if_present('container_image_var', item.container_image or DEFAULT_GPU_DOCKER_IMAGE)
            self._refresh_run_detail_for_item(item=item)
        finally:
            self._loading_queue_item = False

    def _set_var_if_present(self, variable_name: str, value: object) -> None:
        variable = getattr(self, variable_name, None)
        if variable is not None:
            variable.set(value)

    def _refresh_run_detail_for_item(self, *, item: RunQueueItem) -> None:
        if item.source_type == 'recommended' and item.source_id:
            combination = self.recommended_matrix.by_id(item.source_id)
            compact_flow = self._recommended_compact_flow(combination=combination)
            if item.source_id in DEFERRED_RECOMMENDED_COMBINATION_REASONS:
                compact_flow = f'DEFERRED: {compact_flow}'
            self._set_var_if_present('ssh_flow_summary_var', compact_flow)
            prefix = recommended_experiment_prefix(combination)
            self._set_var_if_present(
                'ssh_run_summary_var',
                f'Applied Experiment: {prefix} #{combination.priority} - {combination.combination_id} ({combination.stage})'
            )
            detail_tree = getattr(self, 'ssh_scenario_detail_tree', None)
            if detail_tree is not None:
                self._populate_scenario_detail_tree(
                    tree=detail_tree,
                    rows=self._recommended_detail_rows(combination=combination),
                )
            return

        if item.source_type == 'candidate' and item.source_id:
            candidate = candidate_by_id(item.source_id)
            self._set_var_if_present('ssh_flow_summary_var', candidate_compact_flow(candidate=candidate))
            self._set_var_if_present('ssh_run_summary_var', f'Applied Experiment: Candidate - {candidate.model_id or candidate.adapter}')
            detail_tree = getattr(self, 'ssh_scenario_detail_tree', None)
            if detail_tree is not None:
                self._populate_scenario_detail_tree(
                    tree=detail_tree,
                    rows=candidate_detail_rows(candidate=candidate),
                )
            return

        self._set_var_if_present('ssh_run_summary_var', f'Applied Experiment: {item.display_name}')

    def _refresh_run_queue_tree(self) -> None:
        item_count = len(getattr(self, 'run_queue_items', []))
        active_item = None
        if getattr(self, 'active_queue_item_id', None):
            active_item = next(
                (
                    item
                    for item in getattr(self, 'run_queue_items', [])
                    if item.queue_id == self.active_queue_item_id
                ),
                None,
            )
        summary = f'Scheduled Runs ({item_count}/{MAX_RUN_QUEUE_ITEMS})'
        if active_item is not None:
            summary = f'{summary} - selected #{active_item.order}: {active_item.experiment_id}'
        elif item_count == 0:
            summary = f'{summary} - select experiments, then add them to the queue'
        self._set_var_if_present('run_queue_summary_var', summary)
        tree = getattr(self, 'run_queue_tree', None)
        if tree is None:
            return
        self._clear_tree(tree)
        for item in getattr(self, 'run_queue_items', []):
            tree.insert('', tk.END, iid=item.queue_id, values=self._queue_tree_values(item=item))
        children = tree.get_children('')
        if self.active_queue_item_id and self.active_queue_item_id in children:
            if tuple(tree.selection()) != (self.active_queue_item_id,):
                self._programmatic_run_queue_selection = self.active_queue_item_id
                tree.selection_set(self.active_queue_item_id)
            tree.focus(self.active_queue_item_id)

    def _queue_tree_values(self, *, item: RunQueueItem) -> tuple[str, str, str, str, str, str]:
        return (
            f'{item.order:02d}',
            item.status,
            item.experiment_id,
            item.display_name,
            item.progress_detail or '-',
            self._queue_options_summary(item=item),
        )

    def _queue_options_summary(self, *, item: RunQueueItem) -> str:
        parts = []
        for label, value in [
            ('ep', item.epochs),
            ('detB', item.det_batch),
            ('clsB', item.cls_batch),
            ('detImg', item.det_img),
            ('clsImg', item.cls_img),
            ('lr', item.learning_rate),
            ('patience', item.early_stopping_patience),
            ('minDelta', item.early_stopping_min_delta),
            ('timeout', item.timeout),
        ]:
            if value:
                parts.append(f'{label}={value}')
        if item.dataset_source_mode == DATASET_SOURCE_REMOTE_PRESTAGED:
            parts.append('data=remote')
        if item.remote_runtime_mode == REMOTE_RUNTIME_DOCKER:
            image = item.container_image or DEFAULT_GPU_DOCKER_IMAGE
            parts.append(f'docker={image}')
        return ', '.join(parts) if parts else 'defaults'

    def _select_run_queue_from_tree(self) -> None:
        tree = getattr(self, 'run_queue_tree', None)
        if tree is None:
            return
        selection = tree.selection()
        if not selection:
            return
        queue_id = str(selection[0])
        programmatic_queue_id = getattr(self, '_programmatic_run_queue_selection', None)
        if programmatic_queue_id:
            self._programmatic_run_queue_selection = None
            if queue_id == programmatic_queue_id:
                return
        if queue_id == self.active_queue_item_id:
            return
        self._sync_active_queue_item_from_vars(refresh=False, save=False)
        item = self._queue_item_by_id(queue_id)
        self._apply_queue_item_to_vars(item)
        self._refresh_run_queue_tree()
        self._save_run_queue_state()

    def _replace_run_queue(self, items: list[RunQueueItem]) -> None:
        self.run_queue_items = self._renumber_queue_items(items[:MAX_RUN_QUEUE_ITEMS])
        self.active_queue_item_id = self.run_queue_items[0].queue_id if self.run_queue_items else None
        self._refresh_run_queue_tree()
        if self.active_queue_item_id:
            self._apply_queue_item_to_vars(self._queue_item_by_id(self.active_queue_item_id))
        self._save_run_queue_state()

    def remove_selected_queue_item(self) -> None:
        if self.running:
            return
        tree = getattr(self, 'run_queue_tree', None)
        selection = tree.selection() if tree is not None else ()
        queue_id = str(selection[0]) if selection else self.active_queue_item_id
        if not queue_id:
            return
        self.run_queue_items = self._renumber_queue_items([
            item for item in self.run_queue_items if item.queue_id != queue_id
        ])
        self.active_queue_item_id = self.run_queue_items[0].queue_id if self.run_queue_items else None
        self._refresh_run_queue_tree()
        if self.active_queue_item_id:
            self._apply_queue_item_to_vars(self._queue_item_by_id(self.active_queue_item_id))
        self._save_run_queue_state()
        self.status_var.set('queue item removed')

    def clear_run_queue(self) -> None:
        if self.running:
            return
        self.run_queue_items = []
        self.active_queue_item_id = None
        self._refresh_run_queue_tree()
        self._save_run_queue_state()
        self.status_var.set('queue cleared')

    def _update_queue_item_status(
        self,
        *,
        queue_item_id: str,
        status: str,
        error_message: str = '',
        progress_detail: str | None = None,
    ) -> None:
        if not hasattr(self, 'run_queue_items'):
            return
        if status == 'running':
            self.current_running_queue_item_id = queue_item_id
        elif self.current_running_queue_item_id == queue_item_id:
            self.current_running_queue_item_id = None
        updated_items = []
        status_changed = False
        for item in self.run_queue_items:
            if item.queue_id == queue_item_id:
                status_changed = status_changed or item.status != status
                updated_items.append(
                    replace(
                        item,
                        status=status,
                        error_message=error_message,
                        progress_detail=item.progress_detail if progress_detail is None else progress_detail,
                    ),
                )
            else:
                updated_items.append(item)
        self.run_queue_items = updated_items
        self._refresh_run_queue_tree()
        self._save_run_queue_state(throttle=not status_changed)

    def apply_recommended_to_wsl(self) -> None:
        combination_ids = self._selected_recommended_combination_ids()
        if not combination_ids:
            return
        existing_items = list(getattr(self, 'run_queue_items', []))
        if len(existing_items) + len(combination_ids) > MAX_RUN_QUEUE_ITEMS:
            self.status_var.set(f'queue can hold up to {MAX_RUN_QUEUE_ITEMS} experiments')
            return

        items = list(existing_items)
        added_items: list[RunQueueItem] = []
        for index, combination_id in enumerate(combination_ids, start=len(existing_items) + 1):
            base_item = self._queue_item_for_recommended_combination(
                combination=self.recommended_matrix.by_id(combination_id),
                order=index,
            )
            unique_item = self._queue_item_with_unique_identity(base_item, existing_items=items)
            items.append(unique_item)
            added_items.append(unique_item)

        self._replace_run_queue(items)
        newly_added = added_items[0]
        self.active_queue_item_id = newly_added.queue_id
        self._refresh_run_queue_tree()
        self._apply_queue_item_to_vars(newly_added)
        self._save_run_queue_state()
        combination = self.recommended_matrix.by_id(combination_ids[0])
        locked_config_path = recommended_locked_config_path(combination)
        locked_config_exists = (self.project_dir / locked_config_path).exists()
        config_mode = 'locked config' if locked_config_exists else 'generated template'
        deferred_ids = [
            item.source_id
            for item in items
            if item.source_id in DEFERRED_RECOMMENDED_COMBINATION_REASONS
        ]
        if deferred_ids:
            self._append(
                '\n[Deferred Audit]\n'
                + '\n'.join(DEFERRED_RECOMMENDED_COMBINATION_REASONS[str(combination_id)] for combination_id in deferred_ids)
                + '\n',
            )

        suffix = f'{len(added_items)} added, {len(items)} queued'
        self.status_var.set(f'added to queue: {suffix} ({config_mode})')
        self._show_run_tab()

    def apply_candidate_to_wsl(self) -> None:
        candidate_id = self._selected_gpu_candidate_id()
        if candidate_id is None:
            return
        candidate = candidate_by_id(candidate_id)
        self.active_recommended_combination_id = None
        self.active_gpu_candidate_id = candidate_id
        self.ssh_scenario_var.set(scenario_choice_label('WSL YOLO crop Torchvision', WSL_SCENARIOS['WSL YOLO crop Torchvision']))
        self.ssh_config_var.set(DEFAULT_WSL_YOLO_TORCHVISION_CONFIG_PATH)
        self.ssh_experiment_var.set(f'exp_candidate_{candidate.task_type}_{candidate.model_id or candidate.adapter}')
        self.ssh_db_var.set(f'runs/candidate_{candidate.task_type}_{candidate.model_id or candidate.adapter}_experiments.sqlite3')
        if 'gpu' in candidate.runtime_targets:
            self.ssh_server_var.set(DEFAULT_GPU_SERVER_NAME)
        elif 'wsl' in candidate.runtime_targets:
            self.ssh_server_var.set(DEFAULT_WSL_SERVER_NAME)
        self.ssh_flow_summary_var.set(candidate_compact_flow(candidate=candidate))
        self.ssh_run_summary_var.set(f'Applied Experiment: Candidate - {candidate.model_id or candidate.adapter}')
        self._populate_scenario_detail_tree(
            tree=self.ssh_scenario_detail_tree,
            rows=candidate_detail_rows(candidate=candidate),
        )
        self._replace_run_queue([self._queue_item_from_current_vars(base=RunQueueItem(
            queue_id=f'candidate_{_safe_filename(candidate_id)}',
            order=1,
            source_type='candidate',
            source_id=candidate_id,
            display_name=f'Candidate - {candidate.model_id or candidate.adapter}',
            stage=','.join(candidate.runtime_targets),
            config_path=self.ssh_config_var.get().strip(),
            db_path=self.ssh_db_var.get().strip(),
            native_params_path=self.ssh_native_params_var.get().strip(),
            experiment_id=self.ssh_experiment_var.get().strip(),
            server_profile=self.ssh_server_var.get().strip(),
            native_wrappers=self.ssh_native_wrappers_var.get(),
            replace_existing=self.ssh_replace_existing_var.get(),
            save_csv=self.ssh_save_csv_var.get(),
            save_summary=self.ssh_save_summary_var.get(),
            save_previews=self.ssh_save_previews_var.get(),
            save_checkpoints=self.ssh_save_checkpoints_var.get(),
            collect_mode=self._string_var_value('ssh_collect_mode_var', COLLECT_MODE_BY_VALUE[COLLECT_MODE_WEIGHTS]),
            checkpoint_collect_mode=self._string_var_value(
                'ssh_checkpoint_collect_var',
                CHECKPOINT_COLLECT_BY_VALUE[CHECKPOINT_COLLECT_BEST],
            ),
            weight_mode=self.ssh_weight_mode_var.get(),
            checkpoint_path=self.ssh_checkpoint_var.get().strip(),
            epochs=self.ssh_epochs_var.get().strip(),
            det_batch=self.ssh_batch_size_var.get().strip(),
            cls_batch=self.ssh_classification_batch_size_var.get().strip(),
            det_img=self.ssh_detection_image_size_var.get().strip(),
            cls_img=self.ssh_classification_image_size_var.get().strip(),
            learning_rate=self._string_var_value('ssh_learning_rate_var'),
            early_stopping_patience=self._string_var_value('ssh_early_stopping_patience_var'),
            early_stopping_min_delta=self._string_var_value('ssh_early_stopping_min_delta_var'),
            timeout=self.ssh_external_timeout_var.get().strip(),
            tail=self.ssh_tail_var.get().strip(),
            active_gpu_candidate_id=candidate_id,
        ))])
        self.status_var.set(f'candidate applied: {candidate.model_id or candidate.adapter}')
        self._show_run_tab()

    def clear_recommended_template(self) -> None:
        self.active_recommended_combination_id = None
        self.active_gpu_candidate_id = None
        self.status_var.set('experiment selection cleared')

    def _show_run_tab(self) -> None:
        notebook = getattr(self, 'notebook', None)
        run_tab = getattr(self, 'run_tab', None)
        if notebook is not None and run_tab is not None:
            notebook.select(getattr(run_tab, 'notebook_tab', run_tab))

    def open_result_dir(self) -> None:
        self._open_path(self.result_targets.result_dir)

    def open_preview_dir(self) -> None:
        self._open_path(self.result_targets.preview_dir)

    def open_summary(self) -> None:
        self._open_path(self.result_targets.summary_path)

    def copy_prediction_paths(self) -> None:
        if not self.result_targets.prediction_paths:
            return
        text = '\n'.join(self.result_targets.prediction_paths)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set(f'copied prediction paths: {len(self.result_targets.prediction_paths)}')

    def refresh_result_comparison(self) -> None:
        db_path = self._comparison_db_path()
        if db_path is None:
            return
        if not db_path.exists():
            self._clear_tree(self.compare_tree)
            self._clear_tree(self.compare_metric_tree)
            self.compare_rows = []
            self.compare_summary_var.set(f'DB not found: {self.compare_db_var.get().strip()}')
            self.status_var.set('results db missing')
            return

        try:
            self.compare_rows = load_experiment_comparison(db_path=db_path)
        except Exception as error:
            self.compare_rows = []
            self._clear_tree(self.compare_tree)
            self._clear_tree(self.compare_metric_tree)
            self.compare_summary_var.set(f'failed to load results: {type(error).__name__}: {error}')
            self.status_var.set('results failed')
            return

        self._clear_tree(self.compare_tree)
        self._clear_tree(self.compare_metric_tree)
        for row in self.compare_rows:
            self.compare_tree.insert('', tk.END, iid=row.experiment_id, values=row.to_tree_values())
        self.compare_summary_var.set(self._comparison_overview_text(rows=self.compare_rows))
        self.status_var.set(f'results loaded: {len(self.compare_rows)}')

    def use_local_db_for_comparison(self) -> None:
        self.compare_db_var.set(self.db_var.get().strip())
        self.refresh_result_comparison()

    def use_wsl_db_for_comparison(self) -> None:
        self.compare_db_var.set(self.ssh_db_var.get().strip())
        self.refresh_result_comparison()

    def open_selected_compare_result(self) -> None:
        row = self._selected_compare_row()
        if row is None:
            self.status_var.set('select an experiment first')
            return

        self._open_path(row.result_path)

    def open_selected_compare_summary(self) -> None:
        row = self._selected_compare_row()
        if row is None:
            self.status_var.set('select an experiment first')
            return
        summary_path = row.summary_path
        if not summary_path and row.result_path:
            summary_path = str(Path(row.result_path) / 'summary.md')

        self._open_path(summary_path)

    def generate_result_plots(self) -> None:
        db_path = self._comparison_db_path()
        if db_path is None:
            return
        if not db_path.exists():
            self.compare_summary_var.set(f'DB not found: {self.compare_db_var.get().strip()}')
            self.status_var.set('results db missing')
            return

        output_dir = self.project_dir / 'runs' / 'comparisons' / 'latest'
        try:
            result = generate_comparison_artifacts(db_path=db_path, output_dir=output_dir)
        except Exception as error:
            self.compare_summary_var.set(f'failed to generate plots: {type(error).__name__}: {error}')
            self.status_var.set('plot generation failed')
            return

        self.comparison_plot_dir = result.output_dir
        skipped = f'; skipped={len(result.skipped_plots)}' if result.skipped_plots else ''
        self.compare_summary_var.set(
            f'plots generated: {len(result.plot_paths)} PNG; '
            f'summary={result.summary_csv}; history={result.history_csv}{skipped}',
        )
        self.status_var.set(f'plots generated: {len(result.plot_paths)}')

    def open_result_plots(self) -> None:
        path = self.comparison_plot_dir or (self.project_dir / 'runs' / 'comparisons' / 'latest')
        self._open_path(str(path))

    def _refresh_selected_compare_detail(self) -> None:
        row = self._selected_compare_row()
        self._clear_tree(self.compare_metric_tree)
        if row is None:
            return
        db_path = self._comparison_db_path()
        if db_path is None or not db_path.exists():
            return
        metrics = SQLiteExperimentStorage(db_path=db_path).list_metrics(row.experiment_id)
        for values in metric_tree_values(metrics):
            self.compare_metric_tree.insert('', tk.END, values=values)
        detail = [
            f'{row.experiment_id}',
            f'status={row.status}',
            f'best={row.best_metric_display or "n/a"}',
            f'time={row.duration_display or "n/a"}',
            f'epochs={row.last_epoch if row.last_epoch is not None else "n/a"}',
            f'metrics={row.metric_count}',
            f'acc={row.latest_accuracy:.4g}' if row.latest_accuracy is not None else 'acc=n/a',
            f'f1={row.latest_macro_f1:.4g}' if row.latest_macro_f1 is not None else 'f1=n/a',
            f'map50={row.latest_map50:.4g}' if row.latest_map50 is not None else 'map50=n/a',
        ]
        if row.error_type:
            detail.append(f'error={row.error_type}')
        self.compare_summary_var.set(' | '.join(detail))

    def _selected_compare_row(self) -> ExperimentComparisonRow | None:
        selection = self.compare_tree.selection()
        if not selection:
            return None
        experiment_id = str(selection[0])
        for row in self.compare_rows:
            if row.experiment_id == experiment_id:
                return row

        return None

    def _comparison_db_path(self) -> Path | None:
        value = self.compare_db_var.get().strip()
        if not value:
            self.compare_summary_var.set('DB path is required.')
            self.status_var.set('results db required')
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project_dir / path

        return path

    def _comparison_overview_text(self, *, rows: list[ExperimentComparisonRow]) -> str:
        if not rows:
            return f'No experiments in this DB. {RESULTS_EMPTY_GUIDE}'
        finished = len([row for row in rows if row.status in {'finished', 'collected'}])
        best_rows = [row for row in rows if row.best_metric_value is not None]
        if not best_rows:
            return f'{len(rows)} experiments loaded; {finished} finished/collected; no best metric yet.'
        best = max(best_rows, key=lambda row: row.best_metric_value or float('-inf'))

        return (
            f'{len(rows)} experiments loaded; {finished} finished/collected; '
            f'best {best.best_metric_display} from {best.experiment_id}'
        )

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item_id in tree.get_children():
            tree.delete(item_id)

    def _tail_value(self, variable: tk.StringVar | None = None) -> int:
        variable = variable or self.tail_var
        try:
            return max(1, int(variable.get()))
        except ValueError:
            variable.set(str(DEFAULT_LOG_TAIL))
            return DEFAULT_LOG_TAIL

    def _ssh_run_timeout_seconds(self) -> int | None:
        return self._positive_int_option('ssh_external_timeout_var', 'Timeout')

    def _ssh_execute_stage_timeout_seconds(self) -> int:
        return self._ssh_run_timeout_seconds() or SSH_STAGE_TIMEOUT_SECONDS

    def _ssh_execute_gui_timeout_seconds(self, *, stage_timeout_seconds: int) -> int:
        return (
            GUI_COMMAND_TIMEOUT_SECONDS['ssh prepare']
            + CLI_STAGE_TIMEOUT_SECONDS['ssh bootstrap']
            + CLI_STAGE_TIMEOUT_SECONDS['ssh upload']
            + stage_timeout_seconds
            + CLI_STAGE_TIMEOUT_SECONDS['ssh collect']
            + 60
        )

    def _ssh_stage_timeout_seconds(self, *, stage_name: str, command_name: str) -> int | None:
        if stage_name == 'run':
            return self._ssh_run_timeout_seconds() or CLI_STAGE_TIMEOUT_SECONDS.get(command_name)
        return CLI_STAGE_TIMEOUT_SECONDS.get(command_name)

    def _start_command(
        self,
        name: str,
        command: list[str],
        experiment_id: str | None = None,
        timeout_seconds: int | None = None,
        streaming: bool = False,
    ) -> None:
        if self.running:
            return
        if not (experiment_id or self.experiment_var.get().strip()):
            self._append('experiment id is required\n')
            return

        self.running = True
        self.stop_requested = False
        self._set_buttons_enabled(False)
        self.status_var.set(f'running: {name}')
        self._start_progress(name=name)
        if timeout_seconds is None:
            timeout_seconds = GUI_COMMAND_TIMEOUT_SECONDS.get(name)
        timeout_text = f' timeout={timeout_seconds}s' if timeout_seconds is not None else ''
        self._append(f'\n[{name}{timeout_text}]\n')
        thread = threading.Thread(target=self._run_worker, args=(command, timeout_seconds, streaming), daemon=True)
        thread.start()

    def _start_command_sequence(
        self,
        *,
        name: str,
        steps: list[CommandSequenceStep],
        experiment_id: str | None = None,
    ) -> None:
        if self.running:
            return
        if not steps:
            self._append('no commands to run\n')
            return
        if not (experiment_id or self.experiment_var.get().strip()):
            self._append('experiment id is required\n')
            return

        self.running = True
        self.stop_requested = False
        self._set_buttons_enabled(False)
        self.status_var.set(f'running: {name}')
        self._start_progress(name=name)
        step_names = ' -> '.join(step.name.replace('ssh ', '') for step in steps)
        self._append(f'\n[{name}]\nsteps: {step_names}\n')
        thread = threading.Thread(target=self._run_sequence_worker, args=(steps,), daemon=True)
        thread.start()

    def _apply_local_scenario(self) -> None:
        scenario = self._selected_local_scenario()
        self.config_var.set(scenario.config_path)
        self.db_var.set(scenario.db_path)
        self.experiment_var.set(scenario.experiment_id)
        self.local_flow_summary_var.set(scenario_compact_flow(scenario))
        self._populate_scenario_detail_tree(
            tree=self.local_scenario_detail_tree,
            rows=scenario_detail_rows(scenario),
        )
        self._apply_output_options(
            options=scenario.output_options,
            save_csv=self.save_csv_var,
            save_summary=self.save_summary_var,
            save_previews=self.save_previews_var,
            save_checkpoints=self.save_checkpoints_var,
        )
        self._refresh_local_model_selector_state()

    def _apply_wsl_scenario(self) -> None:
        self.active_recommended_combination_id = None
        self.active_gpu_candidate_id = None
        scenario = self._selected_wsl_scenario()
        self.ssh_config_var.set(scenario.config_path)
        self.ssh_db_var.set(scenario.db_path)
        self.ssh_experiment_var.set(scenario.experiment_id)
        if scenario.server_name is not None:
            self.ssh_server_var.set(scenario.server_name)
        self.ssh_flow_summary_var.set(scenario_compact_flow(scenario))
        self.ssh_run_summary_var.set(f'Applied Experiment: {self._selected_wsl_scenario_key()}')
        self._populate_scenario_detail_tree(
            tree=self.ssh_scenario_detail_tree,
            rows=scenario_detail_rows(scenario),
        )
        self._apply_output_options(
            options=scenario.output_options,
            save_csv=self.ssh_save_csv_var,
            save_summary=self.ssh_save_summary_var,
            save_previews=self.ssh_save_previews_var,
            save_checkpoints=self.ssh_save_checkpoints_var,
            collect_mode=self.ssh_collect_mode_var,
            checkpoint_mode=self.ssh_checkpoint_collect_var,
        )
        self._refresh_collect_option_state()

    def _local_output_options(self) -> GuiOutputOptions:
        return GuiOutputOptions(
            save_csv=self.save_csv_var.get(),
            save_summary=self.save_summary_var.get(),
            save_previews=self.save_previews_var.get(),
            save_checkpoints=self.save_checkpoints_var.get(),
        )

    def _ssh_output_options(self) -> GuiOutputOptions:
        return GuiOutputOptions(
            save_csv=self.ssh_save_csv_var.get(),
            save_summary=self.ssh_save_summary_var.get(),
            save_previews=self.ssh_save_previews_var.get(),
            save_checkpoints=self.ssh_save_checkpoints_var.get(),
            collect_mode=self._collect_mode_value(variable=self.ssh_collect_mode_var),
            checkpoint_mode=self._checkpoint_collect_mode_value(variable=self.ssh_checkpoint_collect_var),
        )

    def _local_weight_options(self) -> GuiWeightOptions:
        return GuiWeightOptions(
            mode=self._weight_mode_value(variable=self.local_weight_mode_var),
            checkpoint_path=self.local_checkpoint_var.get(),
        )

    def _ssh_weight_options(self) -> GuiWeightOptions:
        return GuiWeightOptions(
            mode=self._weight_mode_value(variable=self.ssh_weight_mode_var),
            checkpoint_path=self.ssh_checkpoint_var.get(),
        )

    def _ssh_config_transform(self) -> Callable[[EngineExperimentConfig], EngineExperimentConfig] | None:
        external_command_mode = 'native' if self.ssh_native_wrappers_var.get() else 'dry_run_contract'
        native_params_path = self.ssh_native_params_var.get().strip()
        native_param_overrides = self._native_param_overrides(path=native_params_path)
        detection_dataset_path, classification_crop_dataset_path = self._ssh_dataset_paths()
        if self.active_recommended_combination_id is not None:
            combination = self.recommended_matrix.by_id(self.active_recommended_combination_id)

            def transform(config: EngineExperimentConfig) -> EngineExperimentConfig:
                transformed = build_recommended_combination_config(
                    base_config=config,
                    combination=combination,
                    detection_dataset_path=detection_dataset_path,
                    classification_crop_dataset_path=classification_crop_dataset_path,
                    external_command_mode=external_command_mode,
                    native_param_overrides=native_param_overrides,
                )
                transformed = self._apply_ssh_dataset_source_to_config(config=transformed)
                transformed = self._apply_gui_run_options_to_config(config=transformed)
                return self._apply_remote_runtime_to_config(config=transformed)

            return transform

        if self.active_gpu_candidate_id is not None:
            candidate = candidate_by_id(self.active_gpu_candidate_id)

            def transform(config: EngineExperimentConfig) -> EngineExperimentConfig:
                transformed = build_candidate_experiment_config(
                    base_config=config,
                    candidate=candidate,
                    detection_dataset_path=detection_dataset_path,
                    classification_crop_dataset_path=classification_crop_dataset_path,
                    external_command_mode=external_command_mode,
                    native_param_overrides=native_param_overrides,
                )
                transformed = self._apply_ssh_dataset_source_to_config(config=transformed)
                transformed = self._apply_gui_run_options_to_config(config=transformed)
                return self._apply_remote_runtime_to_config(config=transformed)

            return transform

        if (
            not native_param_overrides
            and self._dataset_source_mode() == DATASET_SOURCE_LOCAL
            and self._remote_runtime_mode() == REMOTE_RUNTIME_NATIVE
        ):
            return None

        def transform(config: EngineExperimentConfig) -> EngineExperimentConfig:
            transformed = config
            if native_param_overrides:
                transformed = self._apply_native_param_overrides_to_config(
                    config=transformed,
                    native_param_overrides=native_param_overrides,
                )
            transformed = self._apply_ssh_dataset_source_to_config(config=transformed)
            transformed = self._apply_gui_run_options_to_config(config=transformed)
            return self._apply_remote_runtime_to_config(config=transformed)

        return transform

    def _apply_remote_runtime_to_config(self, *, config: EngineExperimentConfig) -> EngineExperimentConfig:
        return replace(
            config,
            runtime=replace(
                config.runtime,
                container_mode=self._remote_runtime_mode(),
                container_image=self._container_image() if self._uses_docker_runtime() else '',
            ),
        )

    def _ssh_dataset_paths(self) -> tuple[str, str]:
        if self._dataset_source_mode() != DATASET_SOURCE_REMOTE_PRESTAGED:
            return (
                DEFAULT_MODEL_NAME_DETECTION_DATASET,
                DEFAULT_MODEL_NAME_CLASSIFICATION_CROPS,
            )
        remote_root = self._normalized_remote_dataset_root()
        if self._uses_av_crop_aug_dataset():
            remote_root = AV_CROP_AUG_REMOTE_PRESTAGED_DATASET_ROOT
            return (
                posixpath.join(remote_root, MODEL_NAME_DETECTION_DATASET_SUFFIX),
                posixpath.join(remote_root, MODEL_NAME_AV_CLASSIFICATION_CROP_SUFFIX),
            )
        return (
            posixpath.join(remote_root, MODEL_NAME_DETECTION_DATASET_SUFFIX),
            posixpath.join(remote_root, MODEL_NAME_CLASSIFICATION_DATASET_SUFFIX),
        )

    def _uses_av_crop_aug_dataset(self) -> bool:
        markers = (
            self.active_recommended_combination_id,
            self.active_gpu_candidate_id,
            self._string_var_value('ssh_config_var'),
            self._string_var_value('ssh_experiment_var'),
            self._string_var_value('ssh_db_var'),
            self._string_var_value('ssh_remote_dataset_root_var'),
        )
        return any(
            any(token in str(marker or '') for token in {
                'aug_av_classifier_',
                'classifier_av_crop_aug_top5',
                'tank_armor_prepared_v20260705_av_crop_aug_weak_v1',
            })
            for marker in markers
        )

    def _normalized_remote_dataset_root(self) -> str:
        root = self._remote_dataset_root().replace('\\', '/').rstrip('/')
        return root or DEFAULT_REMOTE_PRESTAGED_DATASET_ROOT

    def _apply_ssh_dataset_source_to_config(self, *, config: EngineExperimentConfig) -> EngineExperimentConfig:
        if self._dataset_source_mode() != DATASET_SOURCE_REMOTE_PRESTAGED:
            return config

        detection_path, classification_path = self._ssh_dataset_paths()
        data_variants = [
            replace(
                variant,
                path=self._remote_path_for_variant(
                    variant_id=variant.id,
                    path=variant.path,
                    detection_path=detection_path,
                    classification_path=classification_path,
                ),
            )
            for variant in config.data_variants
        ]
        code = replace(
            config.code,
            package_include=[
                pattern
                for pattern in config.code.package_include
                if not self._is_local_model_name_dataset_package_include(pattern)
            ],
        )
        return replace(config, code=code, data_variants=data_variants)

    def _remote_path_for_variant(
        self,
        *,
        variant_id: str,
        path: str | None,
        detection_path: str,
        classification_path: str,
    ) -> str | None:
        normalized_path = (path or '').replace('\\', '/')
        explicit_split_classification_suffixes = (
            'classifier_mbt/images',
            'classifier_av/images',
            'classifier_mbt/crops',
            'classifier_av/crops',
        )
        if any(suffix in normalized_path for suffix in explicit_split_classification_suffixes):
            return path
        if variant_id in {'model_name_detection', 'model_name_detection_test'}:
            return detection_path
        if variant_id in {'model_name_classification_crops', 'model_name_classification_crops_test', 'model_name_classification_images', 'model_name_classification_images_test'}:
            return classification_path
        if MODEL_NAME_DETECTION_DATASET_SUFFIX in normalized_path:
            return detection_path
        if MODEL_NAME_CLASSIFICATION_DATASET_SUFFIX in normalized_path:
            return classification_path
        return path

    def _is_local_model_name_dataset_package_include(self, pattern: str) -> bool:
        normalized = pattern.replace('\\', '/').lstrip('./')
        while normalized.startswith('../'):
            normalized = normalized[3:]
        return (
            MODEL_NAME_DETECTION_DATASET_SUFFIX in normalized
            or MODEL_NAME_CLASSIFICATION_DATASET_SUFFIX in normalized
        )
    def _apply_native_param_overrides_to_config(
        self,
        *,
        config: EngineExperimentConfig,
        native_param_overrides: dict[str, object],
    ) -> EngineExperimentConfig:
        return replace(
            config,
            tasks=[
                replace(
                    task,
                    params=merge_native_param_overrides(
                        params=task.params,
                        task_id=task.id,
                        task_type=task.task_type,
                        adapter=task.adapter,
                        model_id=task.model_id,
                        overrides=native_param_overrides,
                    ),
                )
                for task in config.tasks
            ],
        )

    def _apply_gui_run_options_to_config(self, *, config: EngineExperimentConfig) -> EngineExperimentConfig:
        timeout = self._positive_int_option('ssh_external_timeout_var', 'Timeout')
        train = config.train if timeout is None else replace(config.train, max_seconds=timeout)
        transformed = replace(
            config,
            train=train,
            tasks=[
                replace(task, params={**task.params, **self._gui_run_option_params_for_task(task_type=task.task_type)})
                for task in config.tasks
            ],
        )
        if (
            not self._is_refinement_experiment_config(config=transformed)
            and (self._uses_av_crop_aug_dataset() or self._is_classifier_crop_top5_config(config=transformed))
        ):
            transformed = self._apply_classifier_crop_top5_presets(config=transformed)
            self._validate_classifier_crop_top5_config(config=transformed)
        return transformed

    @staticmethod
    def _is_refinement_experiment_config(config: EngineExperimentConfig) -> bool:
        markers = [
            str(getattr(config.experiment, 'name', '') or ''),
            str(getattr(config.experiment, 'description', '') or ''),
        ]
        markers.extend(str(tag or '') for tag in getattr(config.experiment, 'tags', []) or [])
        marker_text = ' '.join(markers).lower()
        return any(token in marker_text for token in {'refinement', 'stage_2_', 'stage 2-'})

    def _is_classifier_crop_top5_config(self, *, config: EngineExperimentConfig) -> bool:
        variant_paths = {
            variant.id: str(variant.path or '').replace('\\', '/')
            for variant in config.data_variants
        }
        for task in config.tasks:
            if task.task_type != 'classification':
                continue
            model_id = str(task.model_id or '').strip()
            if model_id not in CLASSIFIER_CROP_TOP5_MODEL_PRESETS:
                continue
            variant_path = variant_paths.get(task.input_variant, '')
            if self._is_classifier_crop_top5_path(variant_path):
                return True
        markers = (
            self.active_recommended_combination_id,
            self.active_gpu_candidate_id,
            self._string_var_value('ssh_config_var'),
            self._string_var_value('ssh_experiment_var'),
            self._string_var_value('ssh_db_var'),
        )
        return any('aug_av_classifier_' in str(marker or '') for marker in markers)

    @staticmethod
    def _is_classifier_crop_top5_path(path: str) -> bool:
        normalized_path = path.replace('\\', '/').strip().lower()
        return any(marker in normalized_path for marker in CLASSIFIER_CROP_TOP5_DATASET_MARKERS)

    def _apply_classifier_crop_top5_presets(self, *, config: EngineExperimentConfig) -> EngineExperimentConfig:
        variant_paths = {
            variant.id: str(variant.path or '').replace('\\', '/')
            for variant in config.data_variants
        }
        tasks = []
        for task in config.tasks:
            preset = self._classifier_crop_top5_preset_for_task(
                task=task,
                variant_path=variant_paths.get(task.input_variant, ''),
            )
            if preset is None:
                tasks.append(task)
                continue
            params = {**task.params, **preset}
            tasks.append(replace(task, params=params))
        return replace(config, tasks=tasks)

    def _classifier_crop_top5_preset_for_task(
        self,
        *,
        task: object,
        variant_path: str,
    ) -> dict[str, object] | None:
        if getattr(task, 'task_type', None) != 'classification':
            return None
        model_id = str(getattr(task, 'model_id', '') or '').strip()
        preset = CLASSIFIER_CROP_TOP5_MODEL_PRESETS.get(model_id)
        if preset is None:
            return None
        if self._is_classifier_crop_top5_path(variant_path) or self._uses_av_crop_aug_dataset():
            return dict(preset)
        return None

    def _validate_classifier_crop_top5_config(self, *, config: EngineExperimentConfig) -> None:
        expected_path = posixpath.join(
            AV_CROP_AUG_REMOTE_PRESTAGED_DATASET_ROOT,
            MODEL_NAME_AV_CLASSIFICATION_CROP_SUFFIX,
        )
        variant_paths = {
            variant.id: str(variant.path or '').replace('\\', '/')
            for variant in config.data_variants
        }
        enforce_av_path = self._uses_av_crop_aug_dataset() or any(
            'tank_armor_prepared_v20260705_av_crop_aug_weak_v1/classifier_av/crops' in path
            for path in variant_paths.values()
        )
        for variant in config.data_variants:
            if enforce_av_path and variant.id in {
                'model_name_classification_crops',
                'model_name_classification_crops_test',
                'model_name_classification_images',
                'model_name_classification_images_test',
            } and str(variant.path).replace('\\', '/') != expected_path:
                raise ValueError(
                    'AV crop top5 config validation failed: '
                    f'{variant.id} path must be {expected_path}, got {variant.path}'
                )
        for task in config.tasks:
            if task.task_type != 'classification':
                continue
            variant_path = variant_paths.get(task.input_variant, '')
            if not self._is_classifier_crop_top5_path(variant_path):
                continue
            params = task.params
            model_id = str(task.model_id or '').strip()
            expected = CLASSIFIER_CROP_TOP5_MODEL_PRESETS.get(model_id)
            if expected is None:
                raise ValueError(
                    'classifier crop top5 config validation failed for '
                    f'{task.id}: unsupported top5 classifier model_id={model_id!r}'
                )
            mismatches = [
                f'{key}={params.get(key)!r} expected {value!r}'
                for key, value in expected.items()
                if params.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    'classifier crop top5 config validation failed for '
                    f'{task.id}: {", ".join(mismatches)}'
                )

    def _gui_run_option_params_for_task(self, *, task_type: str) -> dict[str, object]:
        params: dict[str, object] = {}
        if task_type in {'detection', 'classification'}:
            epochs = self._positive_int_option('ssh_epochs_var', 'Epochs')
            if epochs is not None:
                params['epochs'] = epochs
            learning_rate = self._positive_float_option('ssh_learning_rate_var', 'LR')
            if learning_rate is not None:
                params['learning_rate'] = learning_rate
            patience = self._positive_int_option('ssh_early_stopping_patience_var', 'Patience')
            if patience is not None:
                params['patience'] = patience
                params['early_stopping_patience'] = patience
            min_delta = self._non_negative_float_option('ssh_early_stopping_min_delta_var', 'Min Delta')
            if min_delta is not None:
                params['early_stopping_min_delta'] = min_delta
        if task_type == 'detection':
            batch_size = self._positive_int_option('ssh_batch_size_var', 'Det Batch')
            if batch_size is not None:
                params['batch_size'] = batch_size
            image_size = self._positive_int_option('ssh_detection_image_size_var', 'Det Img')
            if image_size is not None:
                params['image_size'] = image_size
        if task_type == 'classification':
            batch_size = self._positive_int_option('ssh_classification_batch_size_var', 'Cls Batch')
            if batch_size is not None:
                params['batch_size'] = batch_size
            image_size = self._positive_int_option('ssh_classification_image_size_var', 'Cls Img')
            if image_size is not None:
                params['image_size'] = image_size

        timeout = self._positive_int_option('ssh_external_timeout_var', 'Timeout')
        if timeout is not None:
            params['external_timeout_seconds'] = timeout

        return params

    def _native_param_overrides(self, *, path: str) -> dict[str, object]:
        overrides = load_native_param_overrides_file(path, project_dir=self.project_dir) if path else {}
        gui_overrides = self._gui_native_param_overrides()
        if not gui_overrides:
            return overrides

        merged = dict(overrides)
        defaults = dict(merged.get('defaults') or {})
        defaults.update(gui_overrides.get('defaults') or {})
        if defaults:
            merged['defaults'] = defaults

        by_task_type = dict(merged.get('by_task_type') or {})
        for task_type, values in (gui_overrides.get('by_task_type') or {}).items():
            scoped = dict(by_task_type.get(task_type) or {})
            scoped.update(values)
            by_task_type[task_type] = scoped
        if by_task_type:
            merged['by_task_type'] = by_task_type

        return merged

    def _gui_native_param_overrides(self) -> dict[str, object]:
        values = {
            'epochs': self._positive_int_option('ssh_epochs_var', 'Epochs'),
            'detection_batch_size': self._positive_int_option('ssh_batch_size_var', 'Det Batch'),
            'classification_batch_size': self._positive_int_option('ssh_classification_batch_size_var', 'Cls Batch'),
            'detection_image_size': self._positive_int_option('ssh_detection_image_size_var', 'Det Img'),
            'classification_image_size': self._positive_int_option('ssh_classification_image_size_var', 'Cls Img'),
            'learning_rate': self._positive_float_option('ssh_learning_rate_var', 'LR'),
            'early_stopping_patience': self._positive_int_option('ssh_early_stopping_patience_var', 'Patience'),
            'early_stopping_min_delta': self._non_negative_float_option('ssh_early_stopping_min_delta_var', 'Min Delta'),
            'external_timeout_seconds': self._positive_int_option('ssh_external_timeout_var', 'Timeout'),
        }
        defaults: dict[str, object] = {}
        by_task_type: dict[str, dict[str, object]] = {}
        if values['external_timeout_seconds'] is not None:
            defaults['external_timeout_seconds'] = values['external_timeout_seconds']

        detection: dict[str, object] = {}
        classification: dict[str, object] = {}
        if values['epochs'] is not None:
            detection['epochs'] = values['epochs']
            classification['epochs'] = values['epochs']
        if values['learning_rate'] is not None:
            detection['learning_rate'] = values['learning_rate']
            classification['learning_rate'] = values['learning_rate']
        if values['early_stopping_patience'] is not None:
            detection['patience'] = values['early_stopping_patience']
            detection['early_stopping_patience'] = values['early_stopping_patience']
            classification['patience'] = values['early_stopping_patience']
            classification['early_stopping_patience'] = values['early_stopping_patience']
        if values['early_stopping_min_delta'] is not None:
            detection['early_stopping_min_delta'] = values['early_stopping_min_delta']
            classification['early_stopping_min_delta'] = values['early_stopping_min_delta']
        if values['detection_batch_size'] is not None:
            detection['batch_size'] = values['detection_batch_size']
        if values['classification_batch_size'] is not None:
            classification['batch_size'] = values['classification_batch_size']
        if values['detection_image_size'] is not None:
            detection['image_size'] = values['detection_image_size']
        if values['classification_image_size'] is not None:
            classification['image_size'] = values['classification_image_size']

        if detection:
            by_task_type['detection'] = detection
        if classification:
            by_task_type['classification'] = classification

        overrides: dict[str, object] = {}
        if defaults:
            overrides['defaults'] = defaults
        if by_task_type:
            overrides['by_task_type'] = by_task_type

        return overrides

    def _positive_int_option(self, variable_name: str, label: str) -> int | None:
        variable = getattr(self, variable_name, None)
        if variable is None:
            return None
        text = variable.get().strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(f'{label} must be a positive integer') from error
        if value <= 0:
            raise ValueError(f'{label} must be a positive integer')

        return value

    def _positive_float_option(self, variable_name: str, label: str) -> float | None:
        variable = getattr(self, variable_name, None)
        if variable is None:
            return None
        text = variable.get().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(f'{label} must be a positive number') from error
        if value <= 0:
            raise ValueError(f'{label} must be a positive number')

        return value

    def _non_negative_float_option(self, variable_name: str, label: str) -> float | None:
        variable = getattr(self, variable_name, None)
        if variable is None:
            return None
        text = variable.get().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(f'{label} must be a non-negative number') from error
        if value < 0:
            raise ValueError(f'{label} must be a non-negative number')

        return value

    def _weight_mode_value(self, *, variable: tk.StringVar) -> str:
        return WEIGHT_MODE_LABELS.get(variable.get(), WEIGHT_MODE_PRESET)

    def _collect_mode_value(self, *, variable: tk.StringVar) -> str:
        return COLLECT_MODE_LABELS.get(variable.get(), COLLECT_MODE_WEIGHTS)

    def _checkpoint_collect_mode_value(self, *, variable: tk.StringVar) -> str:
        return CHECKPOINT_COLLECT_LABELS.get(variable.get(), CHECKPOINT_COLLECT_BEST)

    def _collect_weight_files_enabled(self) -> bool:
        collect_mode_var = getattr(self, 'ssh_collect_mode_var', None)
        save_checkpoints_var = getattr(self, 'ssh_save_checkpoints_var', None)
        if collect_mode_var is None or save_checkpoints_var is None:
            return False
        return (
            self._collect_mode_value(variable=collect_mode_var) != COLLECT_MODE_QUICK
            and bool(save_checkpoints_var.get())
        )

    def _refresh_collect_option_state(self) -> None:
        selector = getattr(self, 'ssh_checkpoint_collect_selector', None)
        label = getattr(self, 'ssh_checkpoint_collect_label', None)
        if selector is None:
            return
        state = 'readonly' if self._collect_weight_files_enabled() else 'disabled'
        selector.configure(state=state)
        if label is not None:
            label.configure(state='normal' if state == 'readonly' else 'disabled')

    def _apply_output_options(
        self,
        options: GuiOutputOptions,
        save_csv: tk.BooleanVar,
        save_summary: tk.BooleanVar,
        save_previews: tk.BooleanVar,
        save_checkpoints: tk.BooleanVar,
        collect_mode: tk.StringVar | None = None,
        checkpoint_mode: tk.StringVar | None = None,
    ) -> None:
        save_csv.set(options.save_csv)
        save_summary.set(options.save_summary)
        save_previews.set(options.save_previews)
        save_checkpoints.set(options.save_checkpoints)
        if collect_mode is not None:
            collect_mode.set(COLLECT_MODE_BY_VALUE.get(options.collect_mode, COLLECT_MODE_BY_VALUE[COLLECT_MODE_WEIGHTS]))
        if checkpoint_mode is not None:
            checkpoint_mode.set(
                CHECKPOINT_COLLECT_BY_VALUE.get(
                    options.checkpoint_mode,
                    CHECKPOINT_COLLECT_BY_VALUE[CHECKPOINT_COLLECT_BEST],
                ),
            )

    def _refresh_local_model_selector_state(self) -> None:
        scenario = self._selected_local_scenario()
        enabled = scenario.flow_id in {FLOW_DETECT_CROP_CLASSIFY, FLOW_DETECT_CROP_CLASSIFY_AUGMENTED}
        self.local_override_scope_var.set(local_override_scope_text(scenario))
        for selector in self.local_model_selector_widgets:
            selector.configure(state='readonly' if enabled else 'disabled')

        if not enabled:
            self.local_detector_var.set('')
            self.local_crop_adapter_var.set('')
            self.local_classifier_var.set('')
            self.local_augmentation_policy_var.set('')
            self.local_flow_summary_var.set(scenario_compact_flow(scenario))
            self._populate_scenario_detail_tree(
                tree=self.local_scenario_detail_tree,
                rows=scenario_detail_rows(scenario),
            )
            return

        selection = self._scenario_detect_crop_classify_selection(scenario)
        self.local_detector_var.set(label_for_candidate_id(selection.detector_candidate_id))
        self.local_crop_adapter_var.set(label_for_candidate_id(selection.crop_adapter_candidate_id))
        self.local_classifier_var.set(label_for_candidate_id(selection.classifier_candidate_id))
        self.local_augmentation_policy_var.set(label_for_augmentation_policy_id(selection.augmentation_policy_id))
        self.local_flow_summary_var.set(self._detect_crop_classify_compact_flow(selection=selection))
        self._populate_scenario_detail_tree(
            tree=self.local_scenario_detail_tree,
            rows=self._detect_crop_classify_detail_rows(selection=selection),
        )

    def _scenario_detect_crop_classify_selection(
        self,
        scenario: GuiScenario,
    ) -> DetectCropClassifyCandidateSelection:
        candidate_ids = [
            step.candidate_id
            for step in scenario.task_flow
            if step.candidate_id is not None and step.task_type != 'augmentation'
        ]
        augmentation_policy_ids = [
            step.candidate_id
            for step in scenario.task_flow
            if step.candidate_id is not None and step.task_type == 'augmentation'
        ]
        if len(candidate_ids) >= 3:
            return DetectCropClassifyCandidateSelection(
                detector_candidate_id=candidate_ids[0],
                crop_adapter_candidate_id=candidate_ids[1],
                classifier_candidate_id=candidate_ids[2],
                augmentation_policy_id=augmentation_policy_ids[0] if augmentation_policy_ids else 'none',
            )

        return default_detect_crop_classify_selection(choices=self.local_detect_crop_classify_choices)

    def _update_local_flow_summary_from_selection(self) -> None:
        if self._selected_local_scenario().flow_id not in {FLOW_DETECT_CROP_CLASSIFY, FLOW_DETECT_CROP_CLASSIFY_AUGMENTED}:
            return
        try:
            selection = self._selected_detect_crop_classify_selection()
        except ValueError as error:
            self.local_flow_summary_var.set(str(error))
            return

        self.local_flow_summary_var.set(self._detect_crop_classify_compact_flow(selection=selection))
        self._populate_scenario_detail_tree(
            tree=self.local_scenario_detail_tree,
            rows=self._detect_crop_classify_detail_rows(selection=selection),
        )

    def _selected_detect_crop_classify_selection(self) -> DetectCropClassifyCandidateSelection:
        return DetectCropClassifyCandidateSelection(
            detector_candidate_id=self._selected_candidate_id(
                variable=self.local_detector_var,
                choices=self.local_detector_choice_map,
                label='Detector',
            ),
            crop_adapter_candidate_id=self._selected_candidate_id(
                variable=self.local_crop_adapter_var,
                choices=self.local_crop_adapter_choice_map,
                label='Adapter',
            ),
            classifier_candidate_id=self._selected_candidate_id(
                variable=self.local_classifier_var,
                choices=self.local_classifier_choice_map,
                label='Classifier',
            ),
            augmentation_policy_id=self._selected_candidate_id(
                variable=self.local_augmentation_policy_var,
                choices=self.local_augmentation_policy_choice_map,
                label='Augmentation',
            ),
        )

    def _detect_crop_classify_compact_flow(self, *, selection: DetectCropClassifyCandidateSelection) -> str:
        return describe_detect_crop_classify_selection(selection)

    def _detect_crop_classify_detail_rows(
        self,
        *,
        selection: DetectCropClassifyCandidateSelection,
    ) -> tuple[tuple[str, str, str, str], ...]:
        detector = candidate_by_id(selection.detector_candidate_id)
        crop_adapter = candidate_by_id(selection.crop_adapter_candidate_id)
        classifier = candidate_by_id(selection.classifier_candidate_id)
        policy = augmentation_policy_by_id(selection.augmentation_policy_id)
        rows: list[tuple[str, str, str, str]] = []
        detector_input = 'augmentation' if policy.mode == 'detection_bbox' else 'original'
        crop_input = 'augmentation' if policy.mode == 'detection_bbox' else 'original'
        classifier_input = 'detector_crop_augmented' if policy.mode == 'classification_crop' else 'detector_crop'
        if policy.mode == 'detection_bbox':
            rows.append(('Augment', policy.label, 'original', policy.adapter or '-'))
        rows.extend([
            ('Detection', detector.model_id or detector.adapter, detector_input, detector.adapter),
            ('Crop', crop_adapter.model_id or crop_adapter.adapter, crop_input, crop_adapter.adapter),
        ])
        if policy.mode == 'classification_crop':
            rows.append(('Augment', policy.label, 'detector_crop', policy.adapter or '-'))
        rows.append(('Classification', classifier.model_id or classifier.adapter, classifier_input, classifier.adapter))

        return tuple(rows)

    def _selected_candidate_id(
        self,
        *,
        variable: tk.StringVar,
        choices: dict[str, str],
        label: str,
    ) -> str:
        selected = variable.get()
        if selected in choices:
            return choices[selected]

        raise ValueError(f'{label} selection is required')

    def _selected_local_scenario_key(self) -> str:
        label_map = getattr(self, 'local_scenario_label_map', {})
        return label_map.get(self.local_scenario_var.get(), self.local_scenario_var.get())

    def _selected_wsl_scenario_key(self) -> str:
        label_map = getattr(self, 'ssh_scenario_label_map', {})
        return label_map.get(self.ssh_scenario_var.get(), self.ssh_scenario_var.get())

    def _selected_local_scenario(self) -> GuiScenario:
        return LOCAL_SCENARIOS[self._selected_local_scenario_key()]

    def _selected_wsl_scenario(self) -> GuiScenario:
        return WSL_SCENARIOS[self._selected_wsl_scenario_key()]

    def _load_recommended_matrix(self) -> RecommendedModelMatrix:
        try:
            return load_recommended_model_matrix(project_dir=self.project_dir)
        except Exception:
            return RecommendedModelMatrix(schema_version='', combinations=(), standalone_experiments=())

    def _selected_recommended_combination_id(self) -> str | None:
        label = self.recommended_combination_var.get()
        combination_id = self.recommended_label_map.get(label)
        if combination_id is None:
            self.status_var.set('recommendation selection is required')
            return None

        return combination_id

    def _selected_recommended_combination_ids(self) -> list[str]:
        tree = getattr(self, 'recommended_tree', None)
        selection = list(tree.selection()) if tree is not None else []
        if not selection:
            combination_id = self._selected_recommended_combination_id()
            selection = [combination_id] if combination_id else []
        valid_ids = [
            str(combination_id)
            for combination_id in selection
            if str(combination_id) in {combination.combination_id for combination in self.recommended_matrix.all_experiments}
        ]
        if not valid_ids:
            self.status_var.set('select at least 1 experiment')
            return []
        if len(valid_ids) > MAX_RUN_QUEUE_ITEMS:
            self.status_var.set(f'select up to {MAX_RUN_QUEUE_ITEMS} experiments')
            return []
        return valid_ids

    def _queue_item_with_unique_identity(
        self,
        item: RunQueueItem,
        *,
        existing_items: list[RunQueueItem],
    ) -> RunQueueItem:
        existing_queue_ids = {existing.queue_id for existing in existing_items}
        existing_experiment_keys = {
            (existing.db_path.strip(), existing.experiment_id.strip())
            for existing in existing_items
            if existing.db_path.strip() and existing.experiment_id.strip()
        }
        current_key = (item.db_path.strip(), item.experiment_id.strip())
        if item.queue_id not in existing_queue_ids and current_key not in existing_experiment_keys:
            return item

        for suffix_index in range(2, 100):
            suffix = f'_run{suffix_index:02d}'
            queue_id = f'{item.queue_id}{suffix}'
            db_path = self._path_with_suffix(item.db_path, suffix)
            experiment_id = f'{item.experiment_id}{suffix}'
            key = (db_path.strip(), experiment_id.strip())
            if queue_id in existing_queue_ids or key in existing_experiment_keys:
                continue
            return replace(
                item,
                queue_id=queue_id,
                display_name=f'{item.display_name} run{suffix_index:02d}',
                db_path=db_path,
                experiment_id=experiment_id,
            )
        raise ValueError('could not allocate a unique queue item id')

    @staticmethod
    def _path_with_suffix(value: str, suffix: str) -> str:
        head, separator, filename = value.rpartition('/')
        name = filename if separator else value
        stem, dot, extension = name.rpartition('.')
        suffixed_name = f'{stem}{suffix}.{extension}' if dot else f'{name}{suffix}'
        return f'{head}/{suffixed_name}' if separator else suffixed_name

    def _queue_item_for_recommended_combination(self, *, combination: object, order: int) -> RunQueueItem:
        combination_id = str(getattr(combination, 'combination_id'))
        priority = int(getattr(combination, 'priority'))
        experiment_type = str(getattr(combination, 'experiment_type', 'end_to_end'))
        prefix = recommended_experiment_prefix(combination)
        queue_prefix = _safe_filename(experiment_type)
        locked_config_path = recommended_locked_config_path(combination)
        locked_config_exists = (self.project_dir / locked_config_path).exists()
        native_params_path = self.ssh_native_params_var.get().strip() or DEFAULT_NATIVE_PARAMS_PATH
        if self._is_classifier_crop_top5_recommended_item(
            combination_id=combination_id,
            stage=str(getattr(combination, 'stage')),
            config_path=locked_config_path,
        ):
            native_params_path = TOP5_AUG_NATIVE_PARAMS_PATH
        return RunQueueItem(
            queue_id=f'{queue_prefix}_{priority:02d}_{_safe_filename(combination_id)}',
            order=order,
            source_type='recommended',
            source_id=combination_id,
            display_name=f'{prefix} #{priority} - {combination_id}',
            stage=str(getattr(combination, 'stage')),
            config_path=locked_config_path if locked_config_exists else DEFAULT_WSL_YOLO_TORCHVISION_CONFIG_PATH,
            db_path=f'runs/{queue_prefix}_{priority:02d}_{combination_id}_experiments.sqlite3',
            native_params_path=native_params_path,
            experiment_id=f'exp_top10_{priority:02d}_{combination_id}',
            server_profile=self.ssh_server_var.get().strip() or DEFAULT_GPU_SERVER_NAME,
            native_wrappers=True,
            replace_existing=self.ssh_replace_existing_var.get(),
            save_csv=self.ssh_save_csv_var.get(),
            save_summary=self.ssh_save_summary_var.get(),
            save_previews=self.ssh_save_previews_var.get(),
            save_checkpoints=self.ssh_save_checkpoints_var.get(),
            collect_mode=self._string_var_value('ssh_collect_mode_var', COLLECT_MODE_BY_VALUE[COLLECT_MODE_WEIGHTS]),
            checkpoint_collect_mode=self._string_var_value(
                'ssh_checkpoint_collect_var',
                CHECKPOINT_COLLECT_BY_VALUE[CHECKPOINT_COLLECT_BEST],
            ),
            weight_mode=self.ssh_weight_mode_var.get(),
            checkpoint_path=self.ssh_checkpoint_var.get().strip(),
            epochs=self.ssh_epochs_var.get().strip(),
            det_batch=self.ssh_batch_size_var.get().strip(),
            cls_batch=self.ssh_classification_batch_size_var.get().strip(),
            det_img=self.ssh_detection_image_size_var.get().strip(),
            cls_img=self.ssh_classification_image_size_var.get().strip(),
            learning_rate=self._string_var_value('ssh_learning_rate_var'),
            early_stopping_patience=self._string_var_value('ssh_early_stopping_patience_var'),
            early_stopping_min_delta=self._string_var_value('ssh_early_stopping_min_delta_var'),
            timeout=self.ssh_external_timeout_var.get().strip(),
            tail=self.ssh_tail_var.get().strip(),
            dataset_source_mode=self._dataset_source_mode(),
            remote_dataset_root=self._remote_dataset_root(),
            remote_runtime_mode=self._remote_runtime_mode(),
            container_image=self._container_image(),
            active_recommended_combination_id=combination_id,
        )

    @staticmethod
    def _is_classifier_crop_top5_recommended_item(
        *,
        combination_id: str,
        stage: str,
        config_path: str,
    ) -> bool:
        markers = (combination_id, stage, config_path)
        return any(
            any(token in marker for token in {
                'aug_av_classifier_',
                'classifier_av_crop_aug_top5',
            })
            for marker in markers
        )

    def _selected_gpu_candidate_id(self) -> str | None:
        label = self.gpu_candidate_var.get()
        candidate_id = self.gpu_candidate_label_map.get(label)
        if candidate_id is None:
            self.status_var.set('candidate selection is required')
            return None

        return candidate_id

    def _recommended_summary_text(self) -> str:
        combination_id = self.recommended_label_map.get(self.recommended_combination_var.get())
        if combination_id is None:
            return 'No recommendation matrix loaded.'

        return recommended_combination_summary(self.recommended_matrix.by_id(combination_id))

    def _refresh_recommended_summary(self) -> None:
        self.recommended_summary_var.set(self._recommended_summary_text())
        combination_id = self.recommended_label_map.get(self.recommended_combination_var.get())
        if combination_id is not None and hasattr(self, 'recommended_tree'):
            self.recommended_tree.selection_set(combination_id)
            self.recommended_tree.focus(combination_id)
            self.recommended_tree.see(combination_id)

    def _populate_recommended_tree(self) -> None:
        self._clear_tree(self.recommended_tree)
        for combination in self.recommended_matrix.all_experiments:
            self.recommended_tree.insert(
                '',
                tk.END,
                iid=combination.combination_id,
                values=self._recommended_tree_values(combination=combination),
            )

        combination_id = self.recommended_label_map.get(self.recommended_combination_var.get())
        if combination_id is not None:
            self.recommended_tree.selection_set(combination_id)
            self.recommended_tree.focus(combination_id)

    def _recommended_tree_values(self, *, combination: object) -> tuple[str, str, str, str, str]:
        combination_id = str(getattr(combination, 'combination_id'))
        state = 'Deferred' if combination_id in DEFERRED_RECOMMENDED_COMBINATION_REASONS else str(getattr(combination, 'stage'))

        return (
            f'{int(getattr(combination, "priority")):02d}',
            recommended_experiment_prefix(combination),
            state,
            combination_id,
            str(getattr(combination, 'objective')),
        )

    def _select_recommended_from_tree(self) -> None:
        selection = self.recommended_tree.selection()
        if not selection:
            return

        combination_id = selection[0]
        for label, mapped_combination_id in self.recommended_label_map.items():
            if mapped_combination_id == combination_id:
                self.recommended_combination_var.set(label)
                self.recommended_summary_var.set(self._recommended_summary_text())
                return

    def _recommended_compact_flow(self, *, combination: object) -> str:
        pipeline = getattr(combination, 'pipeline')
        return ' -> '.join(
            _task_stage_label(step.task_type, step.task_adapter)
            for step in pipeline
        )

    def _recommended_detail_rows(self, *, combination: object) -> tuple[tuple[str, str, str, str], ...]:
        pipeline = getattr(combination, 'pipeline')
        return tuple(
            (
                _task_stage_label(step.task_type, step.task_adapter),
                step.model_id or step.task_adapter,
                step.input_kind,
                step.task_adapter,
            )
            for step in pipeline
        )

    def _local_config_transform(
        self,
        dataset_dir: str | None,
    ) -> Callable[[EngineExperimentConfig], EngineExperimentConfig] | None | str:
        scenario = self._selected_local_scenario()
        if scenario.flow_id not in {FLOW_DETECT_CROP_CLASSIFY, FLOW_DETECT_CROP_CLASSIFY_AUGMENTED}:
            return None

        try:
            selection = self._selected_detect_crop_classify_selection()
        except ValueError as error:
            self._append(f'failed to build model selection: {error}\n')
            self.status_var.set('failed')
            return ''

        def transform(config: EngineExperimentConfig) -> EngineExperimentConfig:
            return build_detect_crop_classify_config_from_selection(
                base_config=config,
                selection=selection,
                source_variant_path_override=dataset_dir,
            )

        return transform

    def _effective_config_path(
        self,
        source_config_path: str,
        experiment_id: str,
        options: GuiOutputOptions,
        prefix: str,
        dataset_dir: str | None = None,
        config_transform: Callable[[EngineExperimentConfig], EngineExperimentConfig] | None = None,
        weight_options: GuiWeightOptions | None = None,
    ) -> str | None:
        try:
            return write_effective_config(
                project_dir=self.project_dir,
                source_config_path=source_config_path,
                experiment_id=experiment_id,
                options=options,
                weight_options=weight_options,
                prefix=prefix,
                dataset_dir=dataset_dir,
                config_transform=config_transform,
            )
        except Exception as error:
            self._append(f'failed to create effective config: {type(error).__name__}: {error}\n')
            self.status_var.set('failed')
            return None

    def _validated_image_folder(self) -> str | None:
        value = self.image_folder_var.get().strip()
        if not value:
            return None

        path = Path(value)
        if not path.is_absolute():
            path = self.project_dir / path
        path = path.resolve()
        result = UserImageDatasetValidator().validate_folder(path)
        if result.is_valid:
            return str(path)

        imported_root = self._import_export_image_folder(path=path)
        if imported_root is not None:
            return imported_root

        self._append('image folder validation failed:\n')
        for error in result.errors:
            self._append(f'- {error}\n')
        self.status_var.set('failed')
        return ''

    def _import_export_image_folder(self, *, path: Path) -> str | None:
        importer = ExportDatasetImporter()
        try:
            source = importer.resolve_export_root(path)
            if not (source / 'dataset_info.json').exists():
                return None

            result = importer.import_export(
                source_root=source,
                output_root=self.project_dir / 'runs' / 'user_datasets' / source.name,
                replace_existing=True,
            )
            dataset_root = self._preferred_imported_dataset_root(result=result)
            validation = UserImageDatasetValidator().validate_folder(dataset_root)
            if not validation.is_valid:
                self._append(f'imported image folder validation failed: {dataset_root}\n')
                for error in validation.errors:
                    self._append(f'- {error}\n')
                self.status_var.set('failed')
                return ''

            display_path = self._display_path(path=dataset_root)
            self.image_folder_var.set(display_path)
            self._append(
                'image export imported: '
                f'{source} -> {dataset_root} '
                f'(detection_images={result.detection_image_count}, '
                f'detection_objects={result.detection_object_count}, '
                f'classification_images={result.classification_image_count})\n',
            )
            return str(dataset_root)
        except Exception as error:
            self._append(f'image export import failed: {type(error).__name__}: {error}\n')
            self.status_var.set('failed')
            return ''

    def _preferred_imported_dataset_root(self, *, result: ExportDatasetImportResult) -> Path:
        scenario = self._selected_local_scenario()
        task_types = {
            step.task_type
            for step in scenario.task_flow
        }
        prefers_detection = bool(task_types & {'detection', 'preprocessing', 'tracking', 'segmentation', 'augmentation'})
        if prefers_detection and result.detection_manifest_path is not None:
            return result.detection_manifest_path.parent
        if result.classification_manifest_path is not None:
            return result.classification_manifest_path.parent
        if result.detection_manifest_path is not None:
            return result.detection_manifest_path.parent

        raise ValueError('import did not produce a usable dataset manifest')

    def _display_path(self, *, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_dir.resolve()))
        except ValueError:
            return str(path.resolve())

    def _run_streaming_command(self, *, command: list[str], timeout_seconds: int | None) -> EngineCliResult:
        run_streaming = getattr(self.runner, 'run_streaming', None)
        if not callable(run_streaming):
            return self.runner.run(command=command, timeout_seconds=timeout_seconds)

        def emit(stream_name: str, line: str) -> None:
            stripped = line.rstrip()
            if not stripped:
                return
            self.messages.put(WorkerMessage(kind='stream', text=f'[{stream_name}] {stripped}'))

        return run_streaming(
            command=command,
            timeout_seconds=timeout_seconds,
            on_stdout_line=lambda line: emit('stdout', line),
            on_stderr_line=lambda line: emit('stderr', line),
        )

    def _run_worker(self, command: list[str], timeout_seconds: int | None, streaming: bool = False) -> None:
        if streaming:
            result = self._run_streaming_command(command=command, timeout_seconds=timeout_seconds)
        else:
            result = self.runner.run(command=command, timeout_seconds=timeout_seconds)
        if not result.is_success and getattr(self, 'stop_requested', False):
            self.messages.put(
                WorkerMessage(
                    kind='result',
                    text='Command stopped by user.',
                    success=False,
                    status_text='stopped',
                    action_targets=result.action_targets(),
                    progress_value=0,
                    progress_label='Stopped: local command cancelled',
                ),
            )
            return
        self.messages.put(
            WorkerMessage(
                kind='result',
                text=result.display_text(),
                success=result.is_success,
                status_text=result.status_text(),
                action_targets=result.action_targets(),
            ),
        )

    def _cancel_remote_worker(self, command: list[str], queue_item_id: str | None = None) -> None:
        cancel_runner = EngineCliRunner(cwd=self.project_dir)
        result = cancel_runner.run(command=command, timeout_seconds=45)
        self.messages.put(
            WorkerMessage(
                kind='result',
                text='[Remote Stop]\n' + result.display_text(),
                success=result.is_success,
                status_text='remote cancel done' if result.is_success else 'remote cancel failed',
                action_targets=result.action_targets(),
                progress_value=0 if not result.is_success else 100,
                progress_label='Stopped: remote cancel completed' if result.is_success else 'Stop failed: remote cancel failed',
                queue_item_id=queue_item_id,
                queue_status='cancelled' if result.is_success and queue_item_id else None,
            ),
        )

    def _run_sequence_worker(self, steps: list[CommandSequenceStep]) -> None:
        action_targets = EngineCliActionTargets()
        for index, step in enumerate(steps, start=1):
            timeout_text = f' timeout={step.timeout_seconds}s' if step.timeout_seconds is not None else ''
            progress_label = self._sequence_step_progress_label(
                index=index,
                total=len(steps),
                step_name=step.name,
                step=step,
            )
            queue_detail = self._queue_detail_for_step_start(
                index=step.queue_step_index or index,
                total=step.queue_step_total or len(steps),
                step_name=step.name,
                item_index=step.queue_item_index,
                item_total=step.queue_item_total,
            )
            self.messages.put(
                WorkerMessage(
                    kind='progress',
                    text=self._sequence_step_start_text(
                        progress_label=progress_label,
                        global_index=index,
                        global_total=len(steps),
                        step=step,
                        timeout_text=timeout_text,
                    ),
                    progress_value=((index - 1) / len(steps)) * 100,
                    progress_label=progress_label,
                    queue_item_id=step.queue_item_id,
                    queue_status='running' if step.queue_item_id else None,
                    queue_detail=queue_detail,
                ),
            )
            result = self._run_sequence_step_command(
                step=step,
                progress_label=progress_label,
            )
            result_queue_detail = self._queue_detail_for_step_result(step_name=step.name, result=result)
            if not result.is_success and getattr(self, 'stop_requested', False):
                self.messages.put(
                    WorkerMessage(
                        kind='result',
                        text=f'{step.name} stopped by user.',
                        success=False,
                        status_text=f'{step.name} stopped',
                        action_targets=result.action_targets(),
                        queue_item_id=step.queue_item_id,
                        queue_status='cancelled' if step.queue_item_id else None,
                        progress_value=0,
                        progress_label='Stopped: local command cancelled',
                        queue_detail=f'{self._queue_stage_label(step.name)} stopped by user',
                    ),
                )
                return
            if result.is_success:
                action_targets = result.action_targets()
            self.messages.put(
                WorkerMessage(
                    kind='progress',
                    text=result.display_text() + '\n',
                    progress_value=(index / len(steps)) * 100 if result.is_success else None,
                    progress_label=f'Completed: {progress_label}' if result.is_success else progress_label,
                    queue_item_id=step.queue_item_id,
                    queue_status=step.final_queue_status if result.is_success else None,
                    queue_detail=result_queue_detail,
                ),
            )
            if not result.is_success:
                self.messages.put(
                    WorkerMessage(
                        kind='result',
                        text=f'{step.name} failed; fix the issue above, then press Run again.',
                        success=False,
                        status_text=f'{step.name} failed',
                        action_targets=result.action_targets(),
                        queue_item_id=step.queue_item_id,
                        queue_status='failed' if step.queue_item_id else None,
                        queue_detail=result_queue_detail,
                    ),
                )
                return

        self.messages.put(
            WorkerMessage(
                kind='result',
                text='Run sequence completed.',
                success=True,
                status_text='ssh run all: done',
                action_targets=action_targets,
                progress_value=100,
            ),
        )

    def _run_sequence_step_command(self, *, step: CommandSequenceStep, progress_label: str) -> EngineCliResult:
        log_poller = self._start_ssh_run_log_poller(step=step, progress_label=progress_label)
        run_streaming = getattr(self.runner, 'run_streaming', None)
        try:
            if not callable(run_streaming):
                return self.runner.run(command=step.command, timeout_seconds=step.timeout_seconds)

            last_emit_at = {'value': 0.0}

            def emit(stream_name: str, line: str) -> None:
                stripped = line.strip()
                if not stripped or stripped.startswith('{'):
                    return
                now = time.monotonic()
                if now - last_emit_at['value'] < 0.5 and not self._is_important_stream_line(stripped):
                    return
                last_emit_at['value'] = now
                queue_detail = self._queue_detail_for_stream_line(
                    step_name=step.name,
                    stream_name=stream_name,
                    line=stripped,
                )
                self.messages.put(
                    WorkerMessage(
                        kind='stream',
                        text=f'[{step.name} {stream_name}] {stripped}',
                        progress_label=progress_label,
                        queue_item_id=step.queue_item_id,
                        queue_detail=queue_detail,
                    ),
                )

            return run_streaming(
                command=step.command,
                timeout_seconds=step.timeout_seconds,
                on_stdout_line=lambda line: emit('stdout', line),
                on_stderr_line=lambda line: emit('stderr', line),
            )
        finally:
            self._stop_ssh_run_log_poller(log_poller)

    def _start_ssh_run_log_poller(
        self,
        *,
        step: CommandSequenceStep,
        progress_label: str,
    ) -> tuple[threading.Event, threading.Thread] | None:
        command = self._ssh_log_poll_command_for_step(step=step)
        if command is None or getattr(self, 'project_dir', None) is None:
            return None
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._ssh_run_log_poller_worker,
            args=(stop_event, command, step, progress_label),
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    def _stop_ssh_run_log_poller(self, poller: tuple[threading.Event, threading.Thread] | None) -> None:
        if poller is None:
            return
        stop_event, thread = poller
        stop_event.set()
        thread.join(timeout=1.0)

    def _ssh_log_poll_command_for_step(self, *, step: CommandSequenceStep) -> list[str] | None:
        if step.name != 'ssh run':
            return None
        try:
            ssh_index = step.command.index('ssh')
        except ValueError:
            return None
        if len(step.command) <= ssh_index + 2 or step.command[ssh_index + 1] != 'run':
            return None
        experiment_id = step.command[ssh_index + 2]
        if not experiment_id:
            return None
        try:
            db_path = step.command[step.command.index('--db') + 1]
        except (ValueError, IndexError):
            return None
        return [
            *step.command[:ssh_index],
            'ssh',
            'logs',
            experiment_id,
            '--db',
            db_path,
            '--tail',
            str(SSH_RUN_LOG_POLL_TAIL_LINES),
        ]

    def _ssh_run_log_poller_worker(
        self,
        stop_event: threading.Event,
        command: list[str],
        step: CommandSequenceStep,
        progress_label: str,
    ) -> None:
        poll_runner = EngineCliRunner(cwd=self.project_dir)
        last_tail_signature = ''
        last_heartbeat_at = 0.0
        self._emit_ssh_run_monitor_message(
            step=step,
            progress_label=progress_label,
            text='remote train/task log monitor started',
            queue_detail='Run: remote log monitor active',
        )
        while not stop_event.wait(SSH_RUN_LOG_POLL_INTERVAL_SECONDS):
            result = poll_runner.run(command=command, timeout_seconds=SSH_RUN_LOG_POLL_TIMEOUT_SECONDS)
            now = time.monotonic()
            if result.is_success:
                lines = self._remote_log_tail_lines(result.stdout)
                if lines:
                    signature = '\n'.join(lines[-12:])
                    if signature != last_tail_signature:
                        last_tail_signature = signature
                        display_lines = lines[-4:]
                        epoch_detail = self._epoch_progress_from_lines(lines=display_lines)
                        progress_label_text = (
                            f'Training: {epoch_detail}'
                            if epoch_detail is not None
                            else f'Training: {self._compact_stream_line(display_lines[-1])}'
                        )
                        queue_detail = (
                            f'Run: {epoch_detail}'
                            if epoch_detail is not None
                            else f'Run: remote log -> {self._compact_stream_line(display_lines[-1], limit=80)}'
                        )
                        self._emit_ssh_run_monitor_message(
                            step=step,
                            progress_label=progress_label_text,
                            text='\n'.join(f'[ssh run remote-log] {line}' for line in display_lines),
                            queue_detail=queue_detail,
                        )
                        last_heartbeat_at = now
                    continue

            if now - last_heartbeat_at >= SSH_RUN_LOG_HEARTBEAT_SECONDS:
                self._emit_ssh_run_monitor_message(
                    step=step,
                    progress_label=progress_label,
                    text='[ssh run monitor] waiting for remote train.log/task.log update...',
                    queue_detail='Run: waiting for remote log update',
                )
                last_heartbeat_at = now

    def _emit_ssh_run_monitor_message(
        self,
        *,
        step: CommandSequenceStep,
        progress_label: str,
        text: str,
        queue_detail: str,
    ) -> None:
        self.messages.put(
            WorkerMessage(
                kind='stream',
                text=text,
                progress_label=progress_label,
                queue_item_id=step.queue_item_id,
                queue_detail=queue_detail,
            ),
        )

    def _remote_log_tail_lines(self, text: str) -> list[str]:
        ignored_prefixes = (
            'ssh plan is prepared',
            'next: run bootstrap',
            'next: open logs',
            '[raw json hidden]',
            'raw_json:',
        )
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or self._is_remote_log_noise_line(line):
                continue
            lowered = line.lower()
            if any(lowered.startswith(prefix) for prefix in ignored_prefixes):
                continue
            lines.append(line)
        return lines

    def _is_remote_log_noise_line(self, line: str) -> bool:
        if line in {'{', '}', '[', ']', '},', '],'}:
            return True
        stripped = line.strip()
        if stripped and all(character in '{}[],: ' for character in stripped):
            return True
        lowered = stripped.lower()
        if lowered.startswith(('python_bin=', 'command -v python', 'auto-detect failed')):
            return True
        if 'remote_task_adapter.py' in lowered and '--output-dir' in lowered:
            return True
        return False

    def _drain_messages(self) -> None:
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break

            if message.progress_value is not None:
                self._set_progress_value(message.progress_value)
            if message.progress_label is not None:
                self.progress_var.set(message.progress_label)
            if message.queue_item_id and (message.queue_status or message.queue_detail is not None):
                current_item = None
                if not message.queue_status:
                    try:
                        current_item = self._queue_item_by_id(message.queue_item_id)
                    except KeyError:
                        current_item = None
                self._update_queue_item_status(
                    queue_item_id=message.queue_item_id,
                    status=message.queue_status or (current_item.status if current_item is not None else 'running'),
                    error_message='' if message.queue_status != 'failed' else message.status_text or message.text,
                    progress_detail=message.queue_detail,
                )
            self._append(message.text + '\n')
            if message.kind == 'result':
                self.running = False
                self._set_buttons_enabled(True)
                self._stop_progress(success=bool(message.success))
                if message.action_targets is not None:
                    self.result_targets = message.action_targets
                    self._refresh_result_buttons()
                self.status_var.set(message.status_text or ('done' if message.success else 'failed'))

        self.root.after(100, self._drain_messages)

    def _start_progress(self, *, name: str) -> None:
        self.progress_var.set(self._progress_label_for_command(name=name))
        self._set_progress_value(0)

    def _stop_progress(self, *, success: bool) -> None:
        self._set_progress_value(100 if success else 0)
        self.progress_var.set('done' if success else 'failed')

    def _set_progress_value(self, value: float) -> None:
        bounded = max(0.0, min(100.0, float(value)))
        progress_value_var = getattr(self, 'progress_value_var', None)
        if progress_value_var is not None:
            progress_value_var.set(bounded)
            return
        progress_bar = getattr(self, 'progress_bar', None)
        if progress_bar is not None:
            progress_bar.configure(value=bounded)

    def _progress_label_for_command(self, *, name: str) -> str:
        if name == 'ssh run all':
            return 'SSH: install deps -> prepare -> bootstrap -> upload -> run -> collect'
        if name == 'ssh run selected':
            return 'SSH: selected item install deps -> prepare -> bootstrap -> upload -> run -> collect'
        if name == 'ssh execute':
            return 'SSH: prepare -> bootstrap -> upload -> run -> collect'
        if name == 'ssh install deps':
            return 'SSH: install remote dependencies'
        if name == 'ssh prepare data':
            return 'SSH: prepare remote dataset'
        if name == 'ssh prepare':
            return 'SSH: prepare local code/data package'
        if name == 'ssh bootstrap':
            return 'SSH: bootstrap remote workspace'
        if name == 'ssh upload':
            return 'SSH: upload code, data, and weights'
        if name == 'ssh run':
            return 'SSH: run model on remote GPU'
        if name == 'ssh collect':
            return 'SSH: collect results and weights'
        if name.startswith('ssh '):
            return name.replace('ssh ', 'SSH: ', 1)

        return f'Running: {name}'

    def _queue_detail_for_step_start(
        self,
        *,
        index: int,
        total: int,
        step_name: str,
        item_index: int | None = None,
        item_total: int | None = None,
    ) -> str:
        label = self._queue_stage_label(step_name)
        details = {
            'ssh install deps': 'start -> dependency check',
            'ssh prepare': 'config -> package',
            'ssh bootstrap': 'workspace -> env check',
            'ssh upload': 'cache check -> transfer',
            'ssh run': 'launch -> train/evaluate',
            'ssh collect': 'scan results -> download',
        }
        detail = details.get(step_name, self._progress_label_for_command(name=step_name))
        item_prefix = ''
        if item_index is not None and item_total is not None and item_total > 1:
            item_prefix = f'Queue {item_index}/{item_total} - '
        return f'{item_prefix}{label}: {detail} ({index}/{total})'

    def _queue_detail_for_step_result(self, *, step_name: str, result: EngineCliResult) -> str:
        if not result.is_success:
            return f'{self._queue_stage_label(step_name)} failed: check output log'
        if step_name == 'ssh upload':
            return self._upload_queue_detail(result=result)
        if step_name == 'ssh collect':
            return self._collect_queue_detail(result=result)

        return f'{self._queue_stage_label(step_name)} done'

    def _queue_detail_for_stream_line(self, *, step_name: str, stream_name: str, line: str) -> str:
        label = self._queue_stage_label(step_name)
        lowered = line.lower()
        if lowered.startswith('ironflow_upload_item'):
            return f'{label}: upload item -> transfer'
        if lowered.startswith('ironflow_upload_result'):
            if 'cache=hit' in lowered:
                return f'{label}: transfer -> cache hit'
            if 'cache=miss' in lowered:
                return f'{label}: transfer -> uploaded'
            return f'{label}: transfer -> item done'
        if lowered.startswith('ironflow_download_item'):
            return f'{label}: collect item -> download'
        if lowered.startswith('ironflow_download_result'):
            if 'bundle=true' in lowered:
                return f'{label}: archive bundle -> downloaded'
            return f'{label}: download -> item done'
        if 'already_satisfied' in lowered or 'requirement already satisfied' in lowered:
            return f'{label}: dependency check -> already satisfied'
        if 'collecting ' in lowered or 'installing collected packages' in lowered:
            return f'{label}: dependency check -> pip install'
        if 'downloading' in lowered or 'retrieving folder contents' in lowered:
            return f'{label}: remote lookup -> download'
        if 'processing file' in lowered:
            return f'{label}: list files -> process parts'
        if 'download_bundle' in lowered or 'glob archive' in lowered:
            return f'{label}: archive bundle -> download'
        if 'scp' in lowered or 'upload' in lowered:
            return f'{label}: package -> transfer'
        if 'tar ' in lowered or 'extract' in lowered:
            return f'{label}: transfer -> extract'
        if 'epoch' in lowered:
            epoch_detail = self._epoch_progress_from_lines(lines=[line])
            if epoch_detail is not None:
                return f'{label}: {epoch_detail}'
            return f'{label}: train -> epoch update'
        if stream_name == 'stderr':
            return f'{label}: running -> stderr update'
        return f'{label}: running -> output update'

    def _epoch_progress_from_lines(self, *, lines: list[str]) -> str | None:
        for line in reversed(lines):
            progress = self._epoch_progress_from_line(line=line)
            if progress is not None:
                return progress

        return None

    def _epoch_progress_from_line(self, *, line: str) -> str | None:
        text = ' '.join(line.strip().split())
        if not text:
            return None

        bracket_match = re.search(
            r'Epoch:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\](?:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\])?',
            text,
            flags=re.IGNORECASE,
        )
        if bracket_match:
            current_epoch = int(bracket_match.group(1))
            total_epoch = int(bracket_match.group(2))
            parts = [f'epoch {current_epoch}/{total_epoch}']
            if bracket_match.group(3) and bracket_match.group(4):
                current_iter = int(bracket_match.group(3))
                total_iter = int(bracket_match.group(4))
                parts.append(f'iter {current_iter}/{total_iter}')
            return ', '.join(parts)

        slash_match = re.search(
            r'\b(?:epoch|epochs)\D{0,12}(\d+)\s*/\s*(\d+)\b',
            text,
            flags=re.IGNORECASE,
        )
        if slash_match:
            return f'epoch {int(slash_match.group(1))}/{int(slash_match.group(2))}'

        key_value_match = re.search(
            r'\bepoch(?:\s*[=:]\s*|\s+)(\d+)\b',
            text,
            flags=re.IGNORECASE,
        )
        if key_value_match:
            return f'epoch {int(key_value_match.group(1))}'

        return None

    def _is_important_stream_line(self, line: str) -> bool:
        lowered = line.lower()
        keywords = (
            'already_satisfied',
            'requirement already satisfied',
            'collecting ',
            'installing collected packages',
            'downloading',
            'retrieving folder contents',
            'processing file',
            'download_bundle',
            'glob archive',
            'scp',
            'upload',
            'extract',
            'epoch',
            'error',
            'failed',
            'warning',
            'ironflow_upload_item',
            'ironflow_upload_result',
            'ironflow_download_item',
            'ironflow_download_result',
        )
        return any(keyword in lowered for keyword in keywords)

    def _compact_stream_line(self, line: str, limit: int = 120) -> str:
        compact = ' '.join(line.strip().split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + '...'

    def _queue_stage_label(self, step_name: str) -> str:
        if step_name.startswith('ssh '):
            return step_name.removeprefix('ssh ').title()
        return step_name.title()

    def _upload_queue_detail(self, *, result: EngineCliResult) -> str:
        payload = self._payload_from_result(result=result, stage='upload')
        upload = self._metadata_dict(payload=payload, key='ssh_upload')
        transfer_results = upload.get('transfer_results')
        if not isinstance(transfer_results, list):
            transfer_results = payload.get('transfer_results')
        if not isinstance(transfer_results, list):
            return 'Upload done: transfer summary unavailable'

        ok_count = 0
        cache_hits = 0
        cache_misses = 0
        total_files = 0
        total_bytes = 0
        chunked_items = 0
        for transfer in transfer_results:
            if not isinstance(transfer, dict):
                continue
            ok_count += 1 if bool(transfer.get('success')) else 0
            metadata = transfer.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            cache_hit = metadata.get('upload_cache_hit')
            if cache_hit is True:
                cache_hits += 1
            elif cache_hit is False:
                cache_misses += 1
            total_files += self._int_value(metadata.get('local_file_count'))
            total_bytes += self._int_value(metadata.get('local_size_bytes'))
            if metadata.get('upload_mode') == 'chunked_scp':
                chunked_items += 1

        parts = [f'{ok_count}/{len(transfer_results)} transfers']
        if cache_hits or cache_misses:
            parts.append(f'cache hit {cache_hits}, miss {cache_misses}')
        if total_bytes:
            parts.append(self._format_bytes(total_bytes))
        if total_files:
            parts.append(f'{total_files} files')
        if chunked_items:
            parts.append(f'chunked {chunked_items}')
        return 'Upload done: ' + '; '.join(parts)

    def _collect_queue_detail(self, *, result: EngineCliResult) -> str:
        data = self._stdout_json_from_result(result=result)
        payload = self._payload_from_result(result=result, stage='collect')
        metadata = payload.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        download = self._metadata_dict(payload=payload, key='ssh_download')
        if not download:
            download = self._experiment_metadata_dict(data=data, key='ssh_download')
        transfer_results = download.get('transfer_results')
        if not isinstance(transfer_results, list):
            transfer_results = []
        individual_total = 0
        individual_ok_count = 0
        bundle_count = 0
        for transfer in transfer_results:
            if not isinstance(transfer, dict):
                continue
            transfer_metadata = transfer.get('metadata', {})
            if isinstance(transfer_metadata, dict) and transfer_metadata.get('download_bundle_enabled') is True:
                bundle_count += 1
                continue
            individual_total += 1
            if bool(transfer.get('success')):
                individual_ok_count += 1
        found_files = metadata.get('found_files')
        missing_files = metadata.get('missing_files')
        result_dir = metadata.get('result_dir') or payload.get('workspace_dir')

        parts = []
        if bundle_count:
            parts.append(f'archive bundles {bundle_count}')
        if individual_total:
            parts.append(f'downloaded {individual_ok_count}/{individual_total} items')
        if isinstance(found_files, list):
            parts.append(f'found {len(found_files)} files')
        if isinstance(missing_files, list):
            parts.append(f'missing {len(missing_files)}')
        if isinstance(result_dir, str) and result_dir.strip():
            parts.append(f'local {Path(result_dir).name}')
        return 'Collect done: ' + ('; '.join(parts) if parts else 'summary unavailable')

    def _payload_from_result(self, *, result: EngineCliResult, stage: str) -> dict[str, Any]:
        data = self._stdout_json_from_result(result=result)
        payload = data.get(stage)
        if isinstance(payload, dict):
            return payload
        stages = data.get('stages')
        if isinstance(stages, dict):
            stage_payload = stages.get(stage)
            if isinstance(stage_payload, dict):
                return stage_payload
        return {}

    def _stdout_json_from_result(self, *, result: EngineCliResult) -> dict[str, Any]:
        text = result.stdout.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _metadata_dict(self, *, payload: dict[str, Any], key: str) -> dict[str, Any]:
        metadata = payload.get('metadata', {})
        if not isinstance(metadata, dict):
            return {}
        value = metadata.get(key)
        return value if isinstance(value, dict) else {}

    def _experiment_metadata_dict(self, *, data: dict[str, Any], key: str) -> dict[str, Any]:
        experiment = data.get('experiment')
        if not isinstance(experiment, dict):
            return {}
        metadata = experiment.get('metadata')
        if not isinstance(metadata, dict):
            return {}
        value = metadata.get(key)
        return value if isinstance(value, dict) else {}

    def _int_value(self, value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _format_bytes(self, size_bytes: int) -> str:
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        value = float(size_bytes)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f'{value:.1f}{unit}' if unit != 'B' else f'{int(value)}B'
            value /= 1024

        return f'{size_bytes}B'

    def _sequence_step_progress_label(
        self,
        *,
        index: int,
        total: int,
        step_name: str,
        step: CommandSequenceStep | None = None,
    ) -> str:
        if step_name == 'ssh install deps':
            detail = 'install remote dependencies'
        elif step_name == 'ssh execute':
            detail = 'prepare -> bootstrap -> upload -> run -> collect'
        else:
            detail = self._progress_label_for_command(name=step_name)

        if step is not None and step.queue_step_index is not None and step.queue_step_total is not None:
            prefix = ''
            if step.queue_item_index is not None and step.queue_item_total is not None and step.queue_item_total > 1:
                prefix = f'Queue {step.queue_item_index}/{step.queue_item_total} - '
            return f'{prefix}Step {step.queue_step_index}/{step.queue_step_total}: {step_name} ({detail})'

        return f'Step {index}/{total}: {step_name} ({detail})'

    def _sequence_step_start_text(
        self,
        *,
        progress_label: str,
        global_index: int,
        global_total: int,
        step: CommandSequenceStep,
        timeout_text: str,
    ) -> str:
        if step.queue_step_index is None or step.queue_step_total is None:
            return f'\nCurrent stage: {progress_label}\n[{global_index}/{global_total} {step.name}{timeout_text}]\n'
        item_text = ''
        if step.queue_item_index is not None and step.queue_item_total is not None:
            item_text = f'queue {step.queue_item_index}/{step.queue_item_total}, '
        return (
            f'\nCurrent stage: {progress_label}\n'
            f'[{item_text}step {step.queue_step_index}/{step.queue_step_total}; '
            f'overall {global_index}/{global_total} {step.name}{timeout_text}]\n'
        )

    def _append(self, text: str) -> None:
        output = getattr(self, 'output', None)
        if output is None:
            return
        output.insert(tk.END, text)
        output.see(tk.END)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.command_buttons:
            button.configure(state=state)
        stop_state = tk.DISABLED if enabled else tk.NORMAL
        for button in self.stop_buttons:
            button.configure(state=stop_state)

    def _refresh_result_buttons(self) -> None:
        targets = [
            self.result_targets.result_dir,
            self.result_targets.preview_dir,
            self.result_targets.summary_path,
            self.result_targets.prediction_paths,
        ]
        for button, target in zip(self.result_buttons, targets, strict=True):
            enabled = bool(target)
            if isinstance(target, str):
                enabled = Path(target).exists()
            button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _open_path(self, path: str | None) -> None:
        if not path:
            return
        target = Path(path)
        if not target.is_absolute():
            target = self.project_dir / target
        if not target.exists():
            self.status_var.set(f'path not found: {path}')
            return
        if os.name == 'nt':
            os.startfile(str(target))  # type: ignore[attr-defined]
            return
        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
        subprocess.Popen([opener, str(target)])


def main() -> None:
    root = tk.Tk()
    EngineTkApp(root=root, project_dir=_default_project_dir())
    root.mainloop()


if __name__ == '__main__':
    main()
