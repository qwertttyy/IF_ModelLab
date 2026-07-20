from dataclasses import dataclass, field

from ironflow_exp.engine.ui.local_mock_controller import (
    DEFAULT_ADAPTER_CHAIN_CONFIG_PATH,
    DEFAULT_ADAPTER_CHAIN_DB_PATH,
    DEFAULT_ADAPTER_CHAIN_EXPERIMENT_ID,
    DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_CONFIG_PATH,
    DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_DB_PATH,
    DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_EXPERIMENT_ID,
    DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_CONFIG_PATH,
    DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_DB_PATH,
    DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_EXPERIMENT_ID,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CROP_AUGMENTATION_CONFIG_PATH,
    DEFAULT_CROP_AUGMENTATION_DB_PATH,
    DEFAULT_CROP_AUGMENTATION_EXPERIMENT_ID,
    DEFAULT_CPU_RULE_CONFIG_PATH,
    DEFAULT_CPU_RULE_DB_PATH,
    DEFAULT_CPU_RULE_EXPERIMENT_ID,
    DEFAULT_DB_PATH,
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_K2_LEOPARD2_CONFIG_PATH,
    DEFAULT_K2_LEOPARD2_DB_PATH,
    DEFAULT_K2_LEOPARD2_EXPERIMENT_ID,
    DEFAULT_PREPARED_CLASSIFICATION_GATE_CONFIG_PATH,
    DEFAULT_PREPARED_CLASSIFICATION_GATE_DB_PATH,
    DEFAULT_PREPARED_CLASSIFICATION_GATE_EXPERIMENT_ID,
    DEFAULT_PREPARED_DETECTION_GATE_CONFIG_PATH,
    DEFAULT_PREPARED_DETECTION_GATE_DB_PATH,
    DEFAULT_PREPARED_DETECTION_GATE_EXPERIMENT_ID,
    DEFAULT_PREPARED_EMBEDDING_GATE_CONFIG_PATH,
    DEFAULT_PREPARED_EMBEDDING_GATE_DB_PATH,
    DEFAULT_PREPARED_EMBEDDING_GATE_EXPERIMENT_ID,
    DEFAULT_PREPARED_SEGMENTATION_GATE_CONFIG_PATH,
    DEFAULT_PREPARED_SEGMENTATION_GATE_DB_PATH,
    DEFAULT_PREPARED_SEGMENTATION_GATE_EXPERIMENT_ID,
    DEFAULT_PREPARED_TRACKING_GATE_CONFIG_PATH,
    DEFAULT_PREPARED_TRACKING_GATE_DB_PATH,
    DEFAULT_PREPARED_TRACKING_GATE_EXPERIMENT_ID,
    DEFAULT_SEGMENTATION_SMOKE_CONFIG_PATH,
    DEFAULT_SEGMENTATION_SMOKE_DB_PATH,
    DEFAULT_SEGMENTATION_SMOKE_EXPERIMENT_ID,
    DEFAULT_TEST_MODEL_CONFIG_PATH,
    DEFAULT_TEST_MODEL_DB_PATH,
    DEFAULT_TEST_MODEL_EXPERIMENT_ID,
    DEFAULT_YOLO_TORCHVISION_CHAIN_CONFIG_PATH,
    DEFAULT_YOLO_TORCHVISION_CHAIN_DB_PATH,
    DEFAULT_YOLO_TORCHVISION_CHAIN_EXPERIMENT_ID,
    DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_CONFIG_PATH,
    DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_DB_PATH,
    DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_EXPERIMENT_ID,
)
from ironflow_exp.engine.ui.model_adapter_catalog import (
    candidate_by_id,
)
from ironflow_exp.engine.ui.output_options import GuiOutputOptions
from ironflow_exp.engine.ui.ssh_stage_controller import (
    DEFAULT_GPU_SERVER_NAME,
    DEFAULT_GPU_YOLO26_CONFIG_PATH,
    DEFAULT_GPU_YOLO26_DB_PATH,
    DEFAULT_GPU_YOLO26_EXPERIMENT_ID,
    DEFAULT_WSL_ALBUMENTATIONS_BBOX_CONFIG_PATH,
    DEFAULT_WSL_ALBUMENTATIONS_BBOX_EXPERIMENT_ID,
    DEFAULT_WSL_ALBUMENTATIONS_CROP_CONFIG_PATH,
    DEFAULT_WSL_ALBUMENTATIONS_CROP_EXPERIMENT_ID,
    DEFAULT_WSL_CONFIG_PATH,
    DEFAULT_WSL_DB_PATH,
    DEFAULT_WSL_EXPERIMENT_ID,
    DEFAULT_WSL_TEST_MODEL_CONFIG_PATH,
    DEFAULT_WSL_TEST_MODEL_EXPERIMENT_ID,
    DEFAULT_WSL_YOLO_TORCHVISION_CONFIG_PATH,
    DEFAULT_WSL_YOLO_TORCHVISION_EXPERIMENT_ID,
)


SCENARIO_PURPOSE_SMOKE_PRESET = 'smoke_preset'
SCENARIO_PURPOSE_PREPARED_GATE = 'prepared_gate'
SCENARIO_PURPOSE_REAL_TRAIN = 'real_train'
FLOW_LOCAL_MOCK = 'local_mock'
FLOW_SINGLE_IMAGE_MODEL = 'single_image_model'
FLOW_DETECTION_TRAIN = 'detection_train'
FLOW_DETECT_CROP_CLASSIFY = 'detect_crop_classify'
FLOW_DETECT_CROP_CLASSIFY_AUGMENTED = 'detect_crop_classify_augmented'
FLOW_DETECT_CROP_CLASSIFY_TRACK = 'detect_crop_classify_track'
FLOW_SEGMENTATION = 'segmentation'
FLOW_PREPARED_GATE = 'prepared_gate'


@dataclass(frozen=True, slots=True)
class GuiTaskFlowStep:
    task_type: str
    input_kind: str
    adapter: str | None = None
    model_id: str | None = None
    candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class GuiScenario:
    label: str
    config_path: str
    db_path: str
    experiment_id: str
    server_name: str | None = None
    output_options: GuiOutputOptions = GuiOutputOptions()
    purpose: str = SCENARIO_PURPOSE_SMOKE_PRESET
    flow_id: str = FLOW_LOCAL_MOCK
    task_flow: tuple[GuiTaskFlowStep, ...] = field(default_factory=tuple)


def describe_scenario_flow(scenario: GuiScenario) -> str:
    if not scenario.task_flow:
        return f'{scenario.flow_id}: config preset'

    steps = []
    for step in scenario.task_flow:
        runner = step.model_id or step.adapter or 'configured'
        steps.append(f'{step.task_type}/{step.input_kind}: {runner}')

    return ' -> '.join(steps)


def _step_from_candidate(candidate_id: str, input_kind: str) -> GuiTaskFlowStep:
    candidate = candidate_by_id(candidate_id)

    return GuiTaskFlowStep(
        task_type=candidate.task_type,
        input_kind=input_kind,
        adapter=candidate.adapter,
        model_id=candidate.model_id,
        candidate_id=candidate.candidate_id,
    )


LOCAL_SCENARIOS = {
    'Local mock': GuiScenario(
        label='Local mock',
        config_path=DEFAULT_CONFIG_PATH,
        db_path=DEFAULT_DB_PATH,
        experiment_id=DEFAULT_EXPERIMENT_ID,
        flow_id=FLOW_LOCAL_MOCK,
    ),
    'Local test model': GuiScenario(
        label='Local test model',
        config_path=DEFAULT_TEST_MODEL_CONFIG_PATH,
        db_path=DEFAULT_TEST_MODEL_DB_PATH,
        experiment_id=DEFAULT_TEST_MODEL_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_SINGLE_IMAGE_MODEL,
        task_flow=(
            _step_from_candidate(candidate_id='tiny_rule_vision_v1', input_kind='original'),
        ),
    ),
    'Local CPU rule model': GuiScenario(
        label='Local CPU rule model',
        config_path=DEFAULT_CPU_RULE_CONFIG_PATH,
        db_path=DEFAULT_CPU_RULE_DB_PATH,
        experiment_id=DEFAULT_CPU_RULE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_SINGLE_IMAGE_MODEL,
        task_flow=(
            _step_from_candidate(candidate_id='cpu_rule_vision_v1', input_kind='original'),
        ),
    ),
    'Local K2 vs Leopard 2': GuiScenario(
        label='Local K2 vs Leopard 2',
        config_path=DEFAULT_K2_LEOPARD2_CONFIG_PATH,
        db_path=DEFAULT_K2_LEOPARD2_DB_PATH,
        experiment_id=DEFAULT_K2_LEOPARD2_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_SINGLE_IMAGE_MODEL,
        task_flow=(
            _step_from_candidate(candidate_id='k2_leopard2_smoke_rule_v1', input_kind='original'),
        ),
    ),
    'Local detect-crop-classify': GuiScenario(
        label='Local detect-crop-classify',
        config_path=DEFAULT_ADAPTER_CHAIN_CONFIG_PATH,
        db_path=DEFAULT_ADAPTER_CHAIN_DB_PATH,
        experiment_id=DEFAULT_ADAPTER_CHAIN_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY,
        task_flow=(
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            _step_from_candidate(candidate_id='manifest_classification_smoke', input_kind='detector_crop'),
        ),
    ),
    'Local YOLO crop Torchvision': GuiScenario(
        label='Local YOLO crop Torchvision',
        config_path=DEFAULT_YOLO_TORCHVISION_CHAIN_CONFIG_PATH,
        db_path=DEFAULT_YOLO_TORCHVISION_CHAIN_DB_PATH,
        experiment_id=DEFAULT_YOLO_TORCHVISION_CHAIN_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY,
        task_flow=(
            _step_from_candidate(candidate_id='ultralytics_yolo_detector_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            _step_from_candidate(candidate_id='torchvision_mobilenet_classifier_smoke', input_kind='detector_crop'),
        ),
    ),
    'Local YOLO crop Torchvision ByteTrack': GuiScenario(
        label='Local YOLO crop Torchvision ByteTrack',
        config_path=DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_CONFIG_PATH,
        db_path=DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_DB_PATH,
        experiment_id=DEFAULT_YOLO_TORCHVISION_BYTETRACK_CHAIN_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY_TRACK,
        task_flow=(
            _step_from_candidate(candidate_id='ultralytics_yolo_detector_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            _step_from_candidate(candidate_id='torchvision_mobilenet_classifier_smoke', input_kind='detector_crop'),
            _step_from_candidate(candidate_id='bytetrack_tracker_smoke', input_kind='detection_sequence'),
        ),
    ),
    'Local segmentation smoke': GuiScenario(
        label='Local segmentation smoke',
        config_path=DEFAULT_SEGMENTATION_SMOKE_CONFIG_PATH,
        db_path=DEFAULT_SEGMENTATION_SMOKE_DB_PATH,
        experiment_id=DEFAULT_SEGMENTATION_SMOKE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_SEGMENTATION,
        task_flow=(
            _step_from_candidate(candidate_id='manifest_segmentation_smoke', input_kind='original'),
        ),
    ),
    'Local prepared detection gate': GuiScenario(
        label='Local prepared detection gate',
        config_path=DEFAULT_PREPARED_DETECTION_GATE_CONFIG_PATH,
        db_path=DEFAULT_PREPARED_DETECTION_GATE_DB_PATH,
        experiment_id=DEFAULT_PREPARED_DETECTION_GATE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        purpose=SCENARIO_PURPOSE_PREPARED_GATE,
        flow_id=FLOW_PREPARED_GATE,
        task_flow=(
            _step_from_candidate(candidate_id='yolo_world_detector_planned', input_kind='original'),
        ),
    ),
    'Local prepared classification gate': GuiScenario(
        label='Local prepared classification gate',
        config_path=DEFAULT_PREPARED_CLASSIFICATION_GATE_CONFIG_PATH,
        db_path=DEFAULT_PREPARED_CLASSIFICATION_GATE_DB_PATH,
        experiment_id=DEFAULT_PREPARED_CLASSIFICATION_GATE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        purpose=SCENARIO_PURPOSE_PREPARED_GATE,
        flow_id=FLOW_PREPARED_GATE,
        task_flow=(
            _step_from_candidate(candidate_id='catalog_classification_coca_vit', input_kind='original'),
        ),
    ),
    'Local prepared tracking gate': GuiScenario(
        label='Local prepared tracking gate',
        config_path=DEFAULT_PREPARED_TRACKING_GATE_CONFIG_PATH,
        db_path=DEFAULT_PREPARED_TRACKING_GATE_DB_PATH,
        experiment_id=DEFAULT_PREPARED_TRACKING_GATE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        purpose=SCENARIO_PURPOSE_PREPARED_GATE,
        flow_id=FLOW_PREPARED_GATE,
        task_flow=(
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='botsort_tracker_planned', input_kind='detection_sequence'),
        ),
    ),
    'Local prepared segmentation gate': GuiScenario(
        label='Local prepared segmentation gate',
        config_path=DEFAULT_PREPARED_SEGMENTATION_GATE_CONFIG_PATH,
        db_path=DEFAULT_PREPARED_SEGMENTATION_GATE_DB_PATH,
        experiment_id=DEFAULT_PREPARED_SEGMENTATION_GATE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        purpose=SCENARIO_PURPOSE_PREPARED_GATE,
        flow_id=FLOW_PREPARED_GATE,
        task_flow=(
            _step_from_candidate(candidate_id='yolo_segmentation_planned', input_kind='original'),
        ),
    ),
    'Local prepared embedding gate': GuiScenario(
        label='Local prepared embedding gate',
        config_path=DEFAULT_PREPARED_EMBEDDING_GATE_CONFIG_PATH,
        db_path=DEFAULT_PREPARED_EMBEDDING_GATE_DB_PATH,
        experiment_id=DEFAULT_PREPARED_EMBEDDING_GATE_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        purpose=SCENARIO_PURPOSE_PREPARED_GATE,
        flow_id=FLOW_PREPARED_GATE,
        task_flow=(
            _step_from_candidate(candidate_id='dinov2_embedding_planned', input_kind='original'),
        ),
    ),
    'Local crop augmentation': GuiScenario(
        label='Local crop augmentation',
        config_path=DEFAULT_CROP_AUGMENTATION_CONFIG_PATH,
        db_path=DEFAULT_CROP_AUGMENTATION_DB_PATH,
        experiment_id=DEFAULT_CROP_AUGMENTATION_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY_AUGMENTED,
        task_flow=(
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            GuiTaskFlowStep(
                task_type='augmentation',
                input_kind='detector_crop',
                adapter='classification_crop_augmentation_smoke',
                candidate_id='classification_crop_flip_smoke',
            ),
            _step_from_candidate(candidate_id='manifest_classification_smoke', input_kind='detector_crop'),
        ),
    ),
    'Local Albumentations crop augmentation': GuiScenario(
        label='Local Albumentations crop augmentation',
        config_path=DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_CONFIG_PATH,
        db_path=DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_DB_PATH,
        experiment_id=DEFAULT_ALBUMENTATIONS_CROP_AUGMENTATION_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY_AUGMENTED,
        task_flow=(
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            GuiTaskFlowStep(
                task_type='augmentation',
                input_kind='detector_crop',
                adapter='albumentations_classification_crop',
                candidate_id='albumentations_classification_crop',
            ),
            _step_from_candidate(candidate_id='manifest_classification_smoke', input_kind='detector_crop'),
        ),
    ),
    'Local Albumentations bbox augmentation': GuiScenario(
        label='Local Albumentations bbox augmentation',
        config_path=DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_CONFIG_PATH,
        db_path=DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_DB_PATH,
        experiment_id=DEFAULT_ALBUMENTATIONS_BBOX_AUGMENTATION_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY_AUGMENTED,
        task_flow=(
            GuiTaskFlowStep(
                task_type='augmentation',
                input_kind='original',
                adapter='albumentations_detection_bbox',
                candidate_id='albumentations_detection_bbox',
            ),
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='augmentation'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='augmentation'),
            _step_from_candidate(candidate_id='manifest_classification_smoke', input_kind='detector_crop'),
        ),
    ),
}


WSL_SCENARIOS = {
    'GPU YOLO26 model-name train': GuiScenario(
        label='GPU YOLO26 model-name train',
        config_path=DEFAULT_GPU_YOLO26_CONFIG_PATH,
        db_path=DEFAULT_GPU_YOLO26_DB_PATH,
        experiment_id=DEFAULT_GPU_YOLO26_EXPERIMENT_ID,
        server_name=DEFAULT_GPU_SERVER_NAME,
        purpose=SCENARIO_PURPOSE_REAL_TRAIN,
        flow_id=FLOW_DETECTION_TRAIN,
        task_flow=(
            GuiTaskFlowStep(
                task_type='detection',
                input_kind='original',
                adapter='ultralytics_yolo',
                model_id='yolo26n',
                candidate_id='ultralytics_yolo_detector_planned',
            ),
        ),
    ),
    'WSL mock': GuiScenario(
        label='WSL mock',
        config_path=DEFAULT_WSL_CONFIG_PATH,
        db_path=DEFAULT_WSL_DB_PATH,
        experiment_id=DEFAULT_WSL_EXPERIMENT_ID,
        flow_id=FLOW_LOCAL_MOCK,
    ),
    'WSL test model': GuiScenario(
        label='WSL test model',
        config_path=DEFAULT_WSL_TEST_MODEL_CONFIG_PATH,
        db_path=DEFAULT_WSL_DB_PATH,
        experiment_id=DEFAULT_WSL_TEST_MODEL_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_SINGLE_IMAGE_MODEL,
        task_flow=(
            _step_from_candidate(candidate_id='tiny_rule_vision_v1', input_kind='original'),
        ),
    ),
    'WSL YOLO crop Torchvision': GuiScenario(
        label='WSL YOLO crop Torchvision',
        config_path=DEFAULT_WSL_YOLO_TORCHVISION_CONFIG_PATH,
        db_path=DEFAULT_WSL_DB_PATH,
        experiment_id=DEFAULT_WSL_YOLO_TORCHVISION_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY,
        task_flow=(
            _step_from_candidate(candidate_id='ultralytics_yolo_detector_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            _step_from_candidate(candidate_id='torchvision_mobilenet_classifier_smoke', input_kind='detector_crop'),
        ),
    ),
    'WSL Albumentations crop augmentation': GuiScenario(
        label='WSL Albumentations crop augmentation',
        config_path=DEFAULT_WSL_ALBUMENTATIONS_CROP_CONFIG_PATH,
        db_path=DEFAULT_WSL_DB_PATH,
        experiment_id=DEFAULT_WSL_ALBUMENTATIONS_CROP_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY_AUGMENTED,
        task_flow=(
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='original'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='original'),
            GuiTaskFlowStep(
                task_type='augmentation',
                input_kind='detector_crop',
                adapter='albumentations_classification_crop',
                candidate_id='albumentations_classification_crop',
            ),
            _step_from_candidate(candidate_id='manifest_classification_smoke', input_kind='detector_crop'),
        ),
    ),
    'WSL Albumentations bbox augmentation': GuiScenario(
        label='WSL Albumentations bbox augmentation',
        config_path=DEFAULT_WSL_ALBUMENTATIONS_BBOX_CONFIG_PATH,
        db_path=DEFAULT_WSL_DB_PATH,
        experiment_id=DEFAULT_WSL_ALBUMENTATIONS_BBOX_EXPERIMENT_ID,
        output_options=GuiOutputOptions(save_checkpoints=False),
        flow_id=FLOW_DETECT_CROP_CLASSIFY_AUGMENTED,
        task_flow=(
            GuiTaskFlowStep(
                task_type='augmentation',
                input_kind='original',
                adapter='albumentations_detection_bbox',
                candidate_id='albumentations_detection_bbox',
            ),
            _step_from_candidate(candidate_id='manifest_detection_smoke', input_kind='augmentation'),
            _step_from_candidate(candidate_id='detection_to_classification_crop', input_kind='augmentation'),
            _step_from_candidate(candidate_id='manifest_classification_smoke', input_kind='detector_crop'),
        ),
    ),
}
