from dataclasses import dataclass, field

from ironflow_exp.models import create_default_model_registry
from ironflow_exp.models.spec import ModelSpec


AVAILABILITY_AVAILABLE_SMOKE = 'available_smoke'
AVAILABILITY_PLANNED_REAL = 'planned_real'
READINESS_VALIDATED = 'validated'
READINESS_ADAPTER_READY = 'adapter_ready'
READINESS_PREPARED = 'prepared'
READINESS_NEEDS_ADAPTER = 'needs_adapter'
READINESS_INVALID_CHAIN = 'invalid_chain'
RUNTIME_LOCAL = 'local'
RUNTIME_WSL = 'wsl'
RUNTIME_GPU = 'gpu'

READINESS_LABELS = {
    READINESS_VALIDATED: 'Ready',
    READINESS_ADAPTER_READY: 'Adapter-ready',
    READINESS_PREPARED: 'Prepared',
    READINESS_NEEDS_ADAPTER: 'Needs adapter',
    READINESS_INVALID_CHAIN: 'Invalid chain',
}
_CATALOG_TASK_ADAPTER_ALIASES = {
    'bytetrack_tracker': 'bytetrack',
    'ultralytics_yolo_detection': 'ultralytics_yolo',
}
_CATALOG_INPUT_KIND_BY_TASK = {
    'classification': ('original', 'crop', 'detector_crop'),
    'detection': ('original',),
    'embedding': ('original', 'crop', 'detector_crop'),
    'segmentation': ('original',),
    'tracking': ('detection_sequence',),
}


@dataclass(frozen=True, slots=True)
class ModelAdapterCandidate:
    candidate_id: str
    label: str
    task_type: str
    adapter: str
    model_id: str | None
    supported_input_kinds: tuple[str, ...]
    availability: str
    runtime_targets: tuple[str, ...]
    readiness_status: str = READINESS_VALIDATED
    task_params: dict[str, object] = field(default_factory=dict)
    notes: str = ''

    @property
    def is_available(self) -> bool:
        return self.availability == AVAILABILITY_AVAILABLE_SMOKE

    @property
    def readiness_label(self) -> str:
        return readiness_label_for_status(self.readiness_status)

    @property
    def is_runnable(self) -> bool:
        return self.is_available and self.readiness_status == READINESS_VALIDATED


MODEL_ADAPTER_CANDIDATES: tuple[ModelAdapterCandidate, ...] = (
    ModelAdapterCandidate(
        candidate_id='manifest_detection_smoke',
        label='Manifest detection smoke',
        task_type='detection',
        adapter='manifest_detection_smoke',
        model_id='manifest_detection_smoke',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL,),
        notes='Contract-only detector used by the local detect-crop-classify smoke.',
    ),
    ModelAdapterCandidate(
        candidate_id='detection_to_classification_crop',
        label='Detection bbox crop adapter',
        task_type='preprocessing',
        adapter='detection_to_classification_crop',
        model_id=None,
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL),
        notes='Transforms detection predictions into detector_crop classification inputs.',
    ),
    ModelAdapterCandidate(
        candidate_id='manifest_classification_smoke',
        label='Manifest classification smoke',
        task_type='classification',
        adapter='manifest_classification_smoke',
        model_id='manifest_classification_smoke',
        supported_input_kinds=('detector_crop',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL,),
        notes='Contract-only classifier that consumes classification_input_manifest.',
    ),
    ModelAdapterCandidate(
        candidate_id='tiny_rule_vision_v1',
        label='Tiny rule vision smoke',
        task_type='classification',
        adapter='test_model_smoke',
        model_id='tiny_rule_vision_v1',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL),
        notes='File-backed smoke model for image artifact and prediction contracts.',
    ),
    ModelAdapterCandidate(
        candidate_id='cpu_rule_vision_v1',
        label='CPU rule vision smoke',
        task_type='classification',
        adapter='test_model_smoke',
        model_id='cpu_rule_vision_v1',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL,),
        notes='Dependency-light CPU-only rule model preset.',
    ),
    ModelAdapterCandidate(
        candidate_id='k2_leopard2_smoke_rule_v1',
        label='K2 vs Leopard 2 smoke rule',
        task_type='classification',
        adapter='test_model_smoke',
        model_id='k2_leopard2_smoke_rule_v1',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL,),
        notes='Label-aware K2/Leopard2 smoke classifier; not a production visual classifier.',
    ),
    ModelAdapterCandidate(
        candidate_id='ultralytics_yolo_detector_smoke',
        label='YOLO detector smoke',
        task_type='detection',
        adapter='ultralytics_yolo',
        model_id='yolo11n',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL),
        task_params={
            'execution_mode': 'inference_smoke',
            'pretrained': False,
            'image_size': 32,
            'allow_synthetic_smoke_detection': True,
            'fallback_class_id': 'tank',
        },
        notes='Verified local and WSL CPU inference-smoke detector; contract smoke only, not model-quality evidence.',
    ),
    ModelAdapterCandidate(
        candidate_id='torchvision_mobilenet_classifier_smoke',
        label='MobileNetV3 classifier smoke',
        task_type='classification',
        adapter='torchvision_classifier',
        model_id='mobilenet_v3_small',
        supported_input_kinds=('original', 'detector_crop'),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL),
        task_params={
            'execution_mode': 'inference_smoke',
            'pretrained': False,
            'image_size': 224,
            'top_k': 2,
        },
        notes='Verified local and WSL CPU inference-smoke classifier; pretrained downloads stay disabled.',
    ),
    ModelAdapterCandidate(
        candidate_id='bytetrack_tracker_smoke',
        label='ByteTrack tracker smoke',
        task_type='tracking',
        adapter='bytetrack',
        model_id='bytetrack',
        supported_input_kinds=('detection_sequence',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL,),
        task_params={
            'iou_threshold': 0.3,
        },
        notes='Verified local tracking-by-detection contract smoke; consumes detection_predictions and emits tracking_predictions.',
    ),
    ModelAdapterCandidate(
        candidate_id='manifest_segmentation_smoke',
        label='Manifest segmentation smoke',
        task_type='segmentation',
        adapter='manifest_segmentation_smoke',
        model_id=None,
        supported_input_kinds=('original',),
        availability=AVAILABILITY_AVAILABLE_SMOKE,
        runtime_targets=(RUNTIME_LOCAL,),
        task_params={
            'default_class_id': 'tank',
            'score': 0.92,
            'mask_kind': 'polygon',
        },
        notes='Verified local segmentation contract smoke; emits polygon-based segmentation_predictions from manifest objects.',
    ),
    ModelAdapterCandidate(
        candidate_id='ultralytics_yolo_detector_planned',
        label='YOLO-family detector',
        task_type='detection',
        adapter='ultralytics_yolo',
        model_id='yolo11n_or_yolo26n',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_ADAPTER_READY,
        notes='Planned WSL/GPU detector candidate from model_md; local CPU smoke is exposed separately.',
    ),
    ModelAdapterCandidate(
        candidate_id='d_fine_detector_planned',
        label='D-FINE detector',
        task_type='detection',
        adapter='d_fine_detection',
        model_id='d_fine',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared COCO detection skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='rf_detr_detector_planned',
        label='RF-DETR detector',
        task_type='detection',
        adapter='rf_detr_detection',
        model_id='rf_detr',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared COCO detection skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='rt_detr_detector_planned',
        label='RT-DETR detector',
        task_type='detection',
        adapter='rt_detr_detection',
        model_id='rt_detr',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared COCO detection skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='rtm_det_detector_planned',
        label='RTMDet detector',
        task_type='detection',
        adapter='mmdet_detection',
        model_id='rtm_det',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared MMDetection/COCO skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='yolo_world_detector_planned',
        label='YOLO-World detector',
        task_type='detection',
        adapter='yolo_world_open_vocab_detection',
        model_id='yolo_world',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared open-vocabulary detection skeleton with prompt-set contract.',
    ),
    ModelAdapterCandidate(
        candidate_id='grounding_dino_detector_planned',
        label='GroundingDINO detector',
        task_type='detection',
        adapter='grounding_dino_open_vocab_detection',
        model_id='grounding_dino_1_5',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared open-vocabulary detection skeleton with prompt-set contract.',
    ),
    ModelAdapterCandidate(
        candidate_id='torchvision_mobilenet_classifier_planned',
        label='MobileNetV3/EfficientNet classifier',
        task_type='classification',
        adapter='torchvision_classifier',
        model_id='mobilenet_v3_small_or_efficientnet_b0',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_ADAPTER_READY,
        notes='Planned WSL/GPU classifier candidate; local CPU smoke is exposed separately.',
    ),
    ModelAdapterCandidate(
        candidate_id='timm_convnext_classifier_planned',
        label='timm ConvNeXt classifier',
        task_type='classification',
        adapter='timm_classifier',
        model_id='convnext_v2_tiny',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_ADAPTER_READY,
        notes='Prepared timm classifier train path; GPU/runtime validation remains gated.',
    ),
    ModelAdapterCandidate(
        candidate_id='bytetrack_tracker_planned',
        label='ByteTrack tracker',
        task_type='tracking',
        adapter='bytetrack',
        model_id='bytetrack',
        supported_input_kinds=('detection_sequence',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_ADAPTER_READY,
        notes='Planned WSL/GPU tracking-by-detection candidate; local contract smoke is exposed separately.',
    ),
    ModelAdapterCandidate(
        candidate_id='botsort_tracker_planned',
        label='BoT-SORT tracker',
        task_type='tracking',
        adapter='botsort_tracker',
        model_id='bot_sort',
        supported_input_kinds=('detection_sequence',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared tracking skeleton; optional appearance embedding contract is recorded.',
    ),
    ModelAdapterCandidate(
        candidate_id='ocsort_tracker_planned',
        label='OC-SORT tracker',
        task_type='tracking',
        adapter='ocsort_tracker',
        model_id='oc_sort',
        supported_input_kinds=('detection_sequence',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared motion tracking skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='yolo_segmentation_planned',
        label='YOLO-family segmentation',
        task_type='segmentation',
        adapter='ultralytics_yolo_segmentation',
        model_id='yolo11n_seg_or_yolo26n_seg',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared segmentation candidate from model_md; runnable local contract smoke is exposed separately.',
    ),
    ModelAdapterCandidate(
        candidate_id='sam_promptable_segmentation_planned',
        label='SAM-family segmentation',
        task_type='segmentation',
        adapter='sam_promptable_segmentation',
        model_id='sam2_or_sam3_or_mobile_sam',
        supported_input_kinds=('original',),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared promptable segmentation skeleton; prompt source and runtime implementation remain gated.',
    ),
    ModelAdapterCandidate(
        candidate_id='dinov2_embedding_planned',
        label='DINOv2 embedding',
        task_type='embedding',
        adapter='dinov2_embedding',
        model_id='dinov2_vits14',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared embedding skeleton with embeddings.npy sidecar contract.',
    ),
    ModelAdapterCandidate(
        candidate_id='dinov3_embedding_planned',
        label='DINOv3 embedding',
        task_type='embedding',
        adapter='dinov3_embedding',
        model_id='dinov3_vits16',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared embedding skeleton with embeddings.npy sidecar contract.',
    ),
    ModelAdapterCandidate(
        candidate_id='clip_embedding_planned',
        label='CLIP embedding',
        task_type='embedding',
        adapter='clip_embedding',
        model_id='clip_vit_b_32',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared vision-language embedding skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='openclip_embedding_planned',
        label='OpenCLIP embedding',
        task_type='embedding',
        adapter='clip_embedding',
        model_id='openclip_vit_b_32',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_LOCAL, RUNTIME_WSL, RUNTIME_GPU),
        readiness_status=READINESS_PREPARED,
        notes='Prepared OpenCLIP embedding skeleton; runtime execution is not implemented yet.',
    ),
    ModelAdapterCandidate(
        candidate_id='siglip_embedding_planned',
        label='SigLIP embedding',
        task_type='embedding',
        adapter='siglip_embedding',
        model_id='siglip2_base_patch16',
        supported_input_kinds=('original', 'crop', 'detector_crop'),
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=(RUNTIME_GPU,),
        readiness_status=READINESS_PREPARED,
        notes='Prepared SigLIP embedding skeleton; runtime execution is not implemented yet.',
    ),
)


def all_candidates() -> tuple[ModelAdapterCandidate, ...]:
    return _all_candidates()


def available_candidates() -> tuple[ModelAdapterCandidate, ...]:
    return tuple(candidate for candidate in _all_candidates() if candidate.is_available)


def runnable_candidates() -> tuple[ModelAdapterCandidate, ...]:
    return tuple(candidate for candidate in _all_candidates() if candidate.is_runnable)


def planned_real_candidates() -> tuple[ModelAdapterCandidate, ...]:
    return tuple(
        candidate
        for candidate in _all_candidates()
        if candidate.availability == AVAILABILITY_PLANNED_REAL
    )


def planned_real_candidate_summary() -> str:
    return '\n'.join(
        _candidate_summary_line(candidate)
        for candidate in planned_real_candidates()
    )


def planned_real_candidate_brief_summary() -> str:
    planned_candidates = planned_real_candidates()
    adapter_ready_count = sum(
        1
        for candidate in planned_candidates
        if candidate.readiness_status == READINESS_ADAPTER_READY
    )
    prepared_count = sum(
        1
        for candidate in planned_candidates
        if candidate.readiness_status == READINESS_PREPARED
    )

    return (
        f'{len(planned_candidates)} planned candidates: '
        f'{adapter_ready_count} adapter-ready, {prepared_count} prepared. '
        'Open the Readiness tab for the full model matrix.'
    )


def candidates_for(
    *,
    task_type: str | None = None,
    input_kind: str | None = None,
    availability: str | None = None,
    runtime_target: str | None = None,
) -> tuple[ModelAdapterCandidate, ...]:
    candidates = _all_candidates()
    if task_type is not None:
        candidates = tuple(candidate for candidate in candidates if candidate.task_type == task_type)
    if input_kind is not None:
        candidates = tuple(candidate for candidate in candidates if input_kind in candidate.supported_input_kinds)
    if availability is not None:
        candidates = tuple(candidate for candidate in candidates if candidate.availability == availability)
    if runtime_target is not None:
        candidates = tuple(candidate for candidate in candidates if runtime_target in candidate.runtime_targets)

    return candidates


def readiness_label_for_status(status: str) -> str:
    return READINESS_LABELS.get(status, READINESS_LABELS[READINESS_INVALID_CHAIN])


def candidate_by_id(candidate_id: str) -> ModelAdapterCandidate:
    for candidate in _all_candidates():
        if candidate.candidate_id == candidate_id:
            return candidate

    raise KeyError(f'unknown model/adapter candidate: {candidate_id}')


def _candidate_summary_line(candidate: ModelAdapterCandidate) -> str:
    model_or_adapter = candidate.model_id or candidate.adapter
    input_kinds = ', '.join(candidate.supported_input_kinds)
    runtime_targets = ', '.join(candidate.runtime_targets)

    return (
        f'{candidate.task_type}/{input_kinds}: {candidate.label} '
        f'[{candidate.availability}; readiness={candidate.readiness_label}; adapter={candidate.adapter}; '
        f'model={model_or_adapter}; targets={runtime_targets}]'
    )


def _all_candidates() -> tuple[ModelAdapterCandidate, ...]:
    return MODEL_ADAPTER_CANDIDATES + _catalog_model_candidates()


def _catalog_model_candidates() -> tuple[ModelAdapterCandidate, ...]:
    manual_concrete_keys = {
        (candidate.task_type, candidate.model_id)
        for candidate in MODEL_ADAPTER_CANDIDATES
        if candidate.model_id is not None and '_or_' not in candidate.model_id
    }
    generated: list[ModelAdapterCandidate] = []
    for spec in create_default_model_registry().list_specs():
        if (spec.task, spec.model_id) in manual_concrete_keys:
            continue
        input_kinds = _CATALOG_INPUT_KIND_BY_TASK.get(spec.task)
        if input_kinds is None:
            continue
        generated.append(_candidate_from_model_spec(spec=spec, input_kinds=input_kinds))

    return tuple(generated)


def _candidate_from_model_spec(
    *,
    spec: ModelSpec,
    input_kinds: tuple[str, ...],
) -> ModelAdapterCandidate:
    return ModelAdapterCandidate(
        candidate_id=f'catalog_{_safe_candidate_id(spec.task)}_{_safe_candidate_id(spec.model_id)}',
        label=spec.display_name,
        task_type=spec.task,
        adapter=_CATALOG_TASK_ADAPTER_ALIASES.get(spec.adapter_key, spec.adapter_key),
        model_id=spec.model_id,
        supported_input_kinds=input_kinds,
        availability=AVAILABILITY_PLANNED_REAL,
        runtime_targets=spec.runtime_targets,
        readiness_status=spec.readiness_status,
        notes=spec.notes or f'Catalog model candidate from {spec.family}.',
    )


def _safe_candidate_id(value: str) -> str:
    return ''.join(
        character if character.isalnum() else '_'
        for character in value.lower()
    ).strip('_')
