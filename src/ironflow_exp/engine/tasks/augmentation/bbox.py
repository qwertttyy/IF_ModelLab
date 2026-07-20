import importlib
import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ironflow_exp.engine.core import UserImageDatasetValidator
from ironflow_exp.engine.tasks.base import TaskAdapterResult, TaskExecutionContext
from ironflow_exp.engine.tasks.augmentation.albumentations_schema import AlbumentationsTransformSchema
from ironflow_exp.engine.tasks.augmentation.image_manifest import ImageManifestAugmentationTaskAdapter


class DetectionBBoxAugmentationSmokeTaskAdapter(ImageManifestAugmentationTaskAdapter):
    adapter_key = 'detection_bbox_augmentation_smoke'
    adapter_label = 'detection bbox augmentation smoke'
    default_policy_id = 'detection_bbox_flip_smoke_v1'
    default_recipe = 'horizontal_flip_bbox_xyxy_v1'
    default_label_transform = 'horizontal_flip_bbox_xyxy'

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
        sample_id = str(image_record['sample_id'])
        image_id = str(image_record['image_id'])
        augmentation_id = f'{policy_id}_{index + 1:04d}'
        destination = (
            context.result_dir
            / 'augmented_images'
            / policy_id
            / f'{self._safe_id(image_id)}__aug_{index + 1:04d}.jpg'
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rgb = image.convert('RGB')
            width, height = rgb.size
            ImageOps.mirror(rgb).save(destination)

        augmented_record = dict(image_record)
        augmented_record.update(
            {
                'image_id': f'{image_id}__aug_{index + 1:04d}',
                'sample_id': f'{sample_id}__aug_{index + 1:04d}',
                'path': destination.relative_to(context.result_dir).as_posix(),
                'width': width,
                'height': height,
                'split': split,
                'parent_sample_id': sample_id,
                'original_sample_id': sample_id,
                'augmentation_id': augmentation_id,
                'augmentation_recipe': recipe,
                'label_transform': label_transform,
                'source_path': str(image_record['path']),
                'objects': self._flipped_objects(image_record=image_record, image_width=width, index=index),
            },
        )

        return augmented_record

    def _flipped_objects(
        self,
        *,
        image_record: dict[str, Any],
        image_width: int,
        index: int,
    ) -> list[dict[str, Any]]:
        transformed: list[dict[str, Any]] = []
        for object_index, object_record in enumerate(self._objects(image_record=image_record)):
            output = dict(object_record)
            object_id = str(output.get('object_id') or f'object_{object_index:04d}')
            output['object_id'] = f'{object_id}__aug_{index + 1:04d}'
            bbox = object_record.get('bbox_xyxy')
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                output['bbox_xyxy'] = [float(image_width) - x2, y1, float(image_width) - x1, y2]
                output['bbox_transform'] = 'horizontal_flip_xyxy'
            transformed.append(output)

        return transformed


class AlbumentationsDetectionBBoxTaskAdapter(DetectionBBoxAugmentationSmokeTaskAdapter):
    adapter_key = 'albumentations_detection_bbox'
    adapter_label = 'albumentations detection bbox augmentation'
    default_policy_id = 'albumentations_detection_bbox_v1'
    default_recipe = 'albumentations_horizontal_flip_bbox_xyxy_v1'
    default_label_transform = 'albumentations_bbox_xyxy'

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
                message='albumentations_detection_bbox requires optional dependency: albumentations',
                failure_type='augmentation_dependency_missing',
                metadata={
                    'package': 'albumentations',
                    'policy_id': str(context.record.params.get('policy_id', self.default_policy_id)),
                    'augmentation_recipe': str(context.record.params.get('augmentation_recipe', self.default_recipe)),
                },
            )

        return super().run(context)

    def _validate_contract(self, *, context: TaskExecutionContext) -> None:
        if not context.record.input_variant_path:
            raise ValueError('input_variant_path is required')
        target_split = str(context.record.params.get('target_split', 'train'))
        if target_split != 'train':
            raise ValueError('target_split must be train')
        bbox_format = str(context.record.params.get('bbox_format', 'pascal_voc'))
        if bbox_format not in {'pascal_voc', 'xyxy'}:
            raise ValueError('bbox_format must be pascal_voc or xyxy')
        self.schema.base_seed(context=context)
        self.schema.bbox_policy(context=context)
        for spec in self.schema.transform_specs(context=context):
            self.schema.validated_transform_kwargs(spec=spec)

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
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rgb = image.convert('RGB')
            width, height = rgb.size
            objects = self._objects(image_record=image_record)
            bbox_records = [
                (object_index, [float(value) for value in object_record['bbox_xyxy']])
                for object_index, object_record in enumerate(objects)
                if isinstance(object_record.get('bbox_xyxy'), list) and len(object_record['bbox_xyxy']) == 4
            ]
            transformed = pipeline(
                image=self.schema.numpy_array(rgb),
                bboxes=[bbox for _, bbox in bbox_records],
                bbox_labels=[object_index for object_index, _ in bbox_records],
            )
            Image.fromarray(transformed['image']).save(destination)
        transformed_labels = [
            int(label)
            for label in transformed.get('bbox_labels', [])
        ]
        dropped_objects = self._dropped_objects(
            objects=objects,
            source_object_indices=[object_index for object_index, _ in bbox_records],
            transformed_labels=transformed_labels,
            index=index,
        )

        augmented_record = dict(image_record)
        augmented_record.update(
            {
                'image_id': f'{image_id}__aug_{index + 1:04d}',
                'sample_id': f'{sample_id}__aug_{index + 1:04d}',
                'path': destination.relative_to(context.result_dir).as_posix(),
                'width': width,
                'height': height,
                'split': split,
                'parent_sample_id': sample_id,
                'original_sample_id': sample_id,
                'augmentation_id': augmentation_id,
                'augmentation_recipe': recipe,
                'label_transform': label_transform,
                'source_path': str(image_record['path']),
                'objects': self._transformed_objects(
                    image_record=image_record,
                    transformed_bboxes=transformed['bboxes'],
                    transformed_labels=transformed_labels,
                    index=index,
                ),
                'bbox_policy': self.schema.bbox_policy(context=context),
                'bbox_input_count': len(bbox_records),
                'bbox_output_count': len(transformed['bboxes']),
                'bbox_drop_count': len(dropped_objects),
                'dropped_objects': dropped_objects,
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

        return albumentations.Compose(
            transforms,
            bbox_params=albumentations.BboxParams(
                format='pascal_voc',
                label_fields=['bbox_labels'],
                **self.schema.bbox_policy(context=context),
            ),
            seed=seed,
        )

    def _transformed_objects(
        self,
        *,
        image_record: dict[str, Any],
        transformed_bboxes: list[object],
        transformed_labels: list[int],
        index: int,
    ) -> list[dict[str, Any]]:
        transformed: list[dict[str, Any]] = []
        source_objects = self._objects(image_record=image_record)
        for object_index, object_record in enumerate(self._objects(image_record=image_record)):
            if object_index not in transformed_labels:
                continue
            transformed_index = transformed_labels.index(object_index)
            output = dict(source_objects[object_index])
            object_id = str(output.get('object_id') or f'object_{object_index:04d}')
            output['object_id'] = f'{object_id}__aug_{index + 1:04d}'
            output['bbox_xyxy'] = self._normalized_bbox(transformed_bboxes[transformed_index])
            output['bbox_transform'] = 'albumentations'
            transformed.append(output)

        return transformed

    def _dropped_objects(
        self,
        *,
        objects: list[dict[str, Any]],
        source_object_indices: list[int],
        transformed_labels: list[int],
        index: int,
    ) -> list[dict[str, object]]:
        transformed_label_set = set(transformed_labels)
        dropped: list[dict[str, object]] = []
        for object_index in source_object_indices:
            if object_index in transformed_label_set:
                continue
            object_record = objects[object_index]
            object_id = str(object_record.get('object_id') or f'object_{object_index:04d}')
            dropped.append(
                {
                    'object_id': f'{object_id}__aug_{index + 1:04d}',
                    'source_object_id': object_id,
                    'source_object_index': object_index,
                    'reason': 'bbox_policy_filtered',
                },
            )

        return dropped

    def _normalized_bbox(self, bbox: object) -> list[float]:
        return [
            round(float(value), 6)
            for value in bbox  # type: ignore[operator]
        ]

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
