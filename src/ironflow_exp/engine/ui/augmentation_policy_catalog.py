from dataclasses import dataclass, field


AUGMENTATION_POLICY_NONE = 'none'
AUGMENTATION_POLICY_CLASSIFICATION_CROP_FLIP = 'classification_crop_flip_smoke'
AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP = 'albumentations_classification_crop'
AUGMENTATION_POLICY_DETECTION_BBOX_FLIP = 'detection_bbox_flip_smoke'
AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX = 'albumentations_detection_bbox'
AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP_LIGHT_V1 = 'albumentations_classification_crop_light_v1'
AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP_MEDIUM_V1 = 'albumentations_classification_crop_medium_v1'
AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX_LIGHT_V1 = 'albumentations_detection_bbox_light_v1'
AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX_MEDIUM_V1 = 'albumentations_detection_bbox_medium_v1'
TARGET_SPLIT_TRAIN = 'train'
AUGMENTATION_SEED = 20260616


@dataclass(frozen=True, slots=True)
class AugmentationPolicyCandidate:
    policy_id: str
    label: str
    mode: str
    adapter: str | None
    input_kind: str | None
    output_kind: str | None
    task_params: dict[str, object] = field(default_factory=dict)
    notes: str = ''

    @property
    def is_noop(self) -> bool:
        return self.policy_id == AUGMENTATION_POLICY_NONE


def _common_classification_params(*, policy_id: str, recipe: str, transforms: list[dict[str, object]]) -> dict[str, object]:
    return {
        'target_split': TARGET_SPLIT_TRAIN,
        'policy_id': policy_id,
        'augmentation_recipe': recipe,
        'label_transform': 'preserve_class_label',
        'seed': AUGMENTATION_SEED,
        'transforms': transforms,
        'preserve_original_sample_id': True,
        'preserve_split': True,
        'record_augmentation_id': True,
        'record_augmentation_recipe': True,
        'record_label_transform': True,
    }


def _common_detection_params(*, policy_id: str, recipe: str, transforms: list[dict[str, object]]) -> dict[str, object]:
    return {
        'target_split': TARGET_SPLIT_TRAIN,
        'policy_id': policy_id,
        'augmentation_recipe': recipe,
        'label_transform': 'albumentations_bbox_xyxy',
        'bbox_format': 'pascal_voc',
        'bbox_policy': {
            'clip': True,
            'filter_invalid_bboxes': True,
            'min_area': 4.0,
            'min_visibility': 0.05,
            'min_width': 2.0,
            'min_height': 2.0,
        },
        'seed': AUGMENTATION_SEED,
        'transforms': transforms,
        'preserve_original_sample_id': True,
        'preserve_split': True,
        'record_augmentation_id': True,
        'record_augmentation_recipe': True,
        'record_label_transform': True,
    }


LIGHT_TRANSFORMS: list[dict[str, object]] = [
    {
        'name': 'HorizontalFlip',
        'p': 0.5,
    },
    {
        'name': 'RandomBrightnessContrast',
        'brightness_limit': 0.1,
        'contrast_limit': 0.1,
        'p': 0.7,
    },
    {
        'name': 'Blur',
        'blur_limit': 3,
        'p': 0.25,
    },
]

MEDIUM_TRANSFORMS: list[dict[str, object]] = [
    *LIGHT_TRANSFORMS,
    {
        'name': 'Affine',
        'scale': [0.9, 1.1],
        'translate_percent': [-0.05, 0.05],
        'rotate': [-5.0, 5.0],
        'shear': [-2.0, 2.0],
        'fit_output': False,
        'keep_ratio': True,
        'rotate_method': 'largest_box',
        'p': 0.5,
    },
]


AUGMENTATION_POLICY_CANDIDATES: tuple[AugmentationPolicyCandidate, ...] = (
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_NONE,
        label='No augmentation',
        mode='none',
        adapter=None,
        input_kind=None,
        output_kind=None,
        notes='Leaves the selected model chain unchanged.',
    ),
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_CLASSIFICATION_CROP_FLIP,
        label='Classification crop horizontal flip smoke',
        mode='classification_crop',
        adapter='classification_crop_augmentation_smoke',
        input_kind='detector_crop',
        output_kind='detector_crop',
        task_params={
            'target_split': TARGET_SPLIT_TRAIN,
            'policy_id': 'classification_crop_flip_smoke_v1',
            'augmentation_recipe': 'horizontal_flip_smoke_v1',
            'label_transform': 'preserve_class_label',
            'preserve_original_sample_id': True,
            'preserve_split': True,
            'record_augmentation_id': True,
            'record_augmentation_recipe': True,
            'record_label_transform': True,
        },
        notes='Smoke policy for the built-in crop augmentation adapter.',
    ),
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_DETECTION_BBOX_FLIP,
        label='Detection bbox horizontal flip smoke',
        mode='detection_bbox',
        adapter='detection_bbox_augmentation_smoke',
        input_kind='original',
        output_kind='augmentation',
        task_params={
            'target_split': TARGET_SPLIT_TRAIN,
            'policy_id': 'detection_bbox_flip_smoke_v1',
            'augmentation_recipe': 'horizontal_flip_bbox_xyxy_v1',
            'label_transform': 'horizontal_flip_bbox_xyxy',
            'preserve_original_sample_id': True,
            'preserve_split': True,
            'record_augmentation_id': True,
            'record_augmentation_recipe': True,
            'record_label_transform': True,
        },
        notes='Smoke policy for the built-in bbox augmentation adapter.',
    ),
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP_LIGHT_V1,
        label='Albumentations classification crop light v1',
        mode='classification_crop',
        adapter='albumentations_classification_crop',
        input_kind='detector_crop',
        output_kind='detector_crop',
        task_params=_common_classification_params(
            policy_id='albumentations_classification_crop_light_v1',
            recipe='horizontal_flip_brightness_contrast_blur_light_v1',
            transforms=LIGHT_TRANSFORMS,
        ),
        notes='Train-only crop augmentation for capture-quality variation with low label-risk.',
    ),
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP_MEDIUM_V1,
        label='Albumentations classification crop medium v1',
        mode='classification_crop',
        adapter='albumentations_classification_crop',
        input_kind='detector_crop',
        output_kind='detector_crop',
        task_params=_common_classification_params(
            policy_id='albumentations_classification_crop_medium_v1',
            recipe='light_plus_affine_medium_v1',
            transforms=MEDIUM_TRANSFORMS,
        ),
        notes='Train-only crop augmentation with mild viewpoint/framing variation.',
    ),
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX_LIGHT_V1,
        label='Albumentations detection bbox light v1',
        mode='detection_bbox',
        adapter='albumentations_detection_bbox',
        input_kind='original',
        output_kind='augmentation',
        task_params=_common_detection_params(
            policy_id='albumentations_detection_bbox_light_v1',
            recipe='horizontal_flip_brightness_contrast_blur_light_v1',
            transforms=LIGHT_TRANSFORMS,
        ),
        notes='Train-only bbox augmentation for capture-quality variation with bbox-safe transforms.',
    ),
    AugmentationPolicyCandidate(
        policy_id=AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX_MEDIUM_V1,
        label='Albumentations detection bbox medium v1',
        mode='detection_bbox',
        adapter='albumentations_detection_bbox',
        input_kind='original',
        output_kind='augmentation',
        task_params=_common_detection_params(
            policy_id='albumentations_detection_bbox_medium_v1',
            recipe='light_plus_affine_medium_v1',
            transforms=MEDIUM_TRANSFORMS,
        ),
        notes='Train-only bbox augmentation with mild viewpoint/framing variation and bbox clipping/filtering.',
    ),
)

LEGACY_POLICY_ALIASES = {
    AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP: AUGMENTATION_POLICY_ALBUMENTATIONS_CLASSIFICATION_CROP_LIGHT_V1,
    AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX: AUGMENTATION_POLICY_ALBUMENTATIONS_DETECTION_BBOX_LIGHT_V1,
}


def all_augmentation_policies() -> tuple[AugmentationPolicyCandidate, ...]:
    return AUGMENTATION_POLICY_CANDIDATES


def augmentation_policy_by_id(policy_id: str) -> AugmentationPolicyCandidate:
    canonical_policy_id = LEGACY_POLICY_ALIASES.get(policy_id, policy_id)
    for policy in AUGMENTATION_POLICY_CANDIDATES:
        if policy.policy_id == canonical_policy_id:
            return policy

    raise KeyError(f'unknown augmentation policy: {policy_id}')