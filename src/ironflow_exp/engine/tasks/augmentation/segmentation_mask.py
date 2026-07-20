import importlib
import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image

from ironflow_exp.engine.core import UserImageDatasetValidator
from ironflow_exp.engine.tasks.base import TaskAdapterResult, TaskExecutionContext
from ironflow_exp.engine.tasks.augmentation.albumentations_schema import AlbumentationsTransformSchema
from ironflow_exp.engine.tasks.augmentation.image_manifest import ImageManifestAugmentationTaskAdapter


class AlbumentationsSegmentationMaskTaskAdapter(ImageManifestAugmentationTaskAdapter):
    adapter_key = 'albumentations_segmentation_mask'
    adapter_label = 'albumentations segmentation mask augmentation'
    default_policy_id = 'albumentations_segmentation_mask_v1'
    default_recipe = 'albumentations_mask_light_v1'
    default_label_transform = 'albumentations_image_mask'
    output_manifest_artifact_name = 'segmentation_input_manifest'
    output_manifest_metadata_key = 'segmentation_input_manifest'

    def __init__(self, dataset_validator: UserImageDatasetValidator | None = None) -> None:
        super().__init__(dataset_validator=dataset_validator)
        self.schema = AlbumentationsTransformSchema()

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        try:
            self._validate_contract(context=context)
        except (TypeError, ValueError) as error:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} contract validation failed: {error}',
                failure_type='augmentation_contract_invalid',
                metadata={'error': str(error)},
            )

        if importlib.util.find_spec('albumentations') is None:
            return self._failed_result(
                context=context,
                message='albumentations_segmentation_mask requires optional dependency: albumentations',
                failure_type='augmentation_dependency_missing',
                metadata={
                    'package': 'albumentations',
                    'policy_id': str(context.record.params.get('policy_id', self.default_policy_id)),
                    'augmentation_recipe': str(context.record.params.get('augmentation_recipe', self.default_recipe)),
                },
            )

        return super().run(context)

    def _result_artifacts(self, *, policy_id: str) -> list[dict[str, object]]:
        artifacts = super()._result_artifacts(policy_id=policy_id)
        artifacts.append(
            {
                'name': 'augmented_masks',
                'path': f'augmented_masks/{policy_id}',
                'kind': 'mask_dir',
                'required': True,
            },
        )

        return artifacts

    def _validate_contract(self, *, context: TaskExecutionContext) -> None:
        if not context.record.input_variant_path:
            raise ValueError('input_variant_path is required')
        target_split = str(context.record.params.get('target_split', 'train'))
        if target_split != 'train':
            raise ValueError('target_split must be train')
        self._validate_source_masks(context=context, target_split=target_split)
        self.schema.base_seed(context=context)
        for spec in self.schema.transform_specs(context=context):
            self.schema.validated_transform_kwargs(spec=spec)

    def _validate_source_masks(self, *, context: TaskExecutionContext, target_split: str) -> None:
        dataset_root = Path(context.record.input_variant_path or '')
        source_manifest = self._read_json(dataset_root / 'manifest.json')
        for image_record in self._images(manifest=source_manifest):
            split = str(image_record.get('split') or source_manifest.get('split') or '')
            if split != target_split:
                continue
            self._safe_mask_path(source_root=dataset_root, image_record=image_record)

    def _copy_source_image(
        self,
        *,
        context: TaskExecutionContext,
        image_record: dict[str, Any],
        source_path: Path,
        split: str,
    ) -> dict[str, Any]:
        copied_record = super()._copy_source_image(
            context=context,
            image_record=image_record,
            source_path=source_path,
            split=split,
        )
        source_root = Path(context.record.input_variant_path or '')
        mask_path = self._safe_mask_path(source_root=source_root, image_record=image_record)
        mask_destination = context.result_dir / 'inputs' / str(image_record['mask_path'])
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(mask_path) as mask:
            mask.convert('L').save(mask_destination)
        copied_mask_path = mask_destination.relative_to(context.result_dir).as_posix()
        copied_record['mask_path'] = copied_mask_path
        copied_record['objects'] = self._copied_mask_objects(
            image_record=image_record,
            mask_path=copied_mask_path,
        )

        return copied_record

    def _write_augmented_image(
        self,
        *,
        context: TaskExecutionContext,
        image_record: dict[str, Any],
        source_path: Path,
        split: str,
        policy_id: str,
        recipe: str,
        label_transform: str,
        index: int,
    ) -> dict[str, Any]:
        albumentations = importlib.import_module('albumentations')
        seed = self.schema.seed_for_record(context=context, index=index)
        pipeline = self._pipeline(albumentations=albumentations, context=context, seed=seed)
        sample_id = str(image_record['sample_id'])
        image_id = str(image_record['image_id'])
        augmentation_id = f'{policy_id}_{index + 1:04d}'
        destination = (
            context.result_dir
            / 'augmented_images'
            / policy_id
            / f'{self._safe_id(image_id)}__aug_{index + 1:04d}.jpg'
        )
        mask_destination = (
            context.result_dir
            / 'augmented_masks'
            / policy_id
            / f'{self._safe_id(image_id)}__aug_{index + 1:04d}_mask.png'
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        mask_path = self._safe_mask_path(source_root=Path(context.record.input_variant_path or ''), image_record=image_record)
        with Image.open(source_path) as image, Image.open(mask_path) as mask:
            rgb = image.convert('RGB')
            width, height = rgb.size
            mask_l = mask.convert('L')
            if mask_l.size != rgb.size:
                raise ValueError('mask size must match source image size')
            transformed = pipeline(
                image=self.schema.numpy_array(rgb),
                mask=self.schema.numpy_array(mask_l),
            )
            Image.fromarray(transformed['image']).save(destination)
            Image.fromarray(transformed['mask']).convert('L').save(mask_destination)

        augmented_mask_path = mask_destination.relative_to(context.result_dir).as_posix()
        augmented_record = dict(image_record)
        augmented_record.update(
            {
                'image_id': f'{image_id}__aug_{index + 1:04d}',
                'sample_id': f'{sample_id}__aug_{index + 1:04d}',
                'path': destination.relative_to(context.result_dir).as_posix(),
                'mask_path': augmented_mask_path,
                'width': width,
                'height': height,
                'split': split,
                'parent_sample_id': sample_id,
                'original_sample_id': sample_id,
                'augmentation_id': augmentation_id,
                'augmentation_recipe': recipe,
                'label_transform': label_transform,
                'mask_transform': 'albumentations',
                'source_path': str(image_record['path']),
                'source_mask_path': str(image_record['mask_path']),
                'objects': self._mask_objects(
                    image_record=image_record,
                    index=index,
                    mask_path=augmented_mask_path,
                ),
            },
        )
        if seed is not None:
            augmented_record['augmentation_seed'] = seed

        return augmented_record

    def _pipeline(self, *, albumentations: Any, context: TaskExecutionContext, seed: int | None = None) -> Any:
        transforms = []
        for spec in self.schema.transform_specs(context=context):
            name = str(spec['name'])
            kwargs = self.schema.validated_transform_kwargs(spec=spec)
            transforms.append(getattr(albumentations, name)(**kwargs))

        return albumentations.Compose(transforms, seed=seed)

    def _safe_mask_path(self, *, source_root: Path, image_record: dict[str, Any]) -> Path:
        raw_mask_path = image_record.get('mask_path')
        if not isinstance(raw_mask_path, str) or not raw_mask_path:
            raise ValueError('segmentation mask augmentation requires image_record.mask_path')
        path = Path(raw_mask_path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('mask_path must be a safe relative path')
        resolved = (source_root / path).resolve()
        try:
            resolved.relative_to(source_root.resolve())
        except ValueError as exc:
            raise ValueError('mask_path escapes source manifest root') from exc
        if not resolved.exists() or not resolved.is_file():
            raise ValueError('mask_path does not exist')

        return resolved

    def _copied_mask_objects(self, *, image_record: dict[str, Any], mask_path: str) -> list[dict[str, Any]]:
        copied: list[dict[str, Any]] = []
        for object_record in self._objects(image_record=image_record):
            output = dict(object_record)
            source_mask_path = output.get('mask_path') or image_record.get('mask_path')
            if isinstance(source_mask_path, str) and source_mask_path and source_mask_path != mask_path:
                output['source_mask_path'] = source_mask_path
            output['mask_path'] = mask_path
            copied.append(output)

        return copied

    def _mask_objects(self, *, image_record: dict[str, Any], index: int, mask_path: str) -> list[dict[str, Any]]:
        transformed: list[dict[str, Any]] = []
        for object_index, object_record in enumerate(self._objects(image_record=image_record)):
            output = dict(object_record)
            object_id = str(output.get('object_id') or f'object_{object_index:04d}')
            output['object_id'] = f'{object_id}__aug_{index + 1:04d}'
            source_mask_path = output.get('mask_path') or image_record.get('mask_path')
            if isinstance(source_mask_path, str) and source_mask_path:
                output['source_mask_path'] = source_mask_path
            output['mask_path'] = mask_path
            output['mask_transform'] = 'albumentations'
            transformed.append(output)

        return transformed

    def _failed_result(
        self,
        *,
        context: TaskExecutionContext,
        message: str,
        failure_type: str,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        return TaskAdapterResult(
            success=False,
            status='failed',
            message=message,
            metrics=[
                {
                    'epoch': 1,
                    'train_loss': '',
                    'val_loss': '',
                    'accuracy': '',
                    'map50': '',
                    'map50_95': '',
                    'lr': '',
                },
            ],
            metadata={
                'adapter': self.adapter_key,
                'failure_type': failure_type,
                **metadata,
            },
        )
