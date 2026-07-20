import importlib
import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image

from ironflow_exp.engine.core import DetectionToClassificationArtifactValidator
from ironflow_exp.engine.tasks.base import TaskAdapterResult, TaskExecutionContext
from ironflow_exp.engine.tasks.augmentation.albumentations_schema import AlbumentationsTransformSchema
from ironflow_exp.engine.tasks.augmentation.crop import ClassificationCropAugmentationSmokeTaskAdapter


class AlbumentationsClassificationCropTaskAdapter(ClassificationCropAugmentationSmokeTaskAdapter):
    adapter_key = 'albumentations_classification_crop'
    adapter_label = 'albumentations classification crop augmentation'
    default_policy_id = 'albumentations_classification_crop_v1'
    default_recipe = 'albumentations_crop_light_v1'
    default_label_transform = 'preserve_class_label'

    def __init__(self, artifact_validator: DetectionToClassificationArtifactValidator | None = None) -> None:
        super().__init__(artifact_validator=artifact_validator)
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
                message='albumentations_classification_crop requires optional dependency: albumentations',
                failure_type='augmentation_dependency_missing',
                metadata={
                    'package': 'albumentations',
                    'policy_id': str(context.record.params.get('policy_id', self.default_policy_id)),
                    'augmentation_recipe': str(context.record.params.get('augmentation_recipe', self.default_recipe)),
                },
            )

        return super().run(context)

    def _validate_contract(self, *, context: TaskExecutionContext) -> None:
        target_split = str(context.record.params.get('target_split', 'train'))
        if target_split != 'train':
            raise ValueError('target_split must be train')
        self.schema.base_seed(context=context)
        for spec in self.schema.transform_specs(context=context):
            self.schema.validated_transform_kwargs(spec=spec)

    def _write_flipped_crop(
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
        source_sample_id = str(image_record['sample_id'])
        source_image_id = str(image_record['image_id'])
        augmentation_id = f'{policy_id}_{index + 1:04d}'
        destination = (
            context.result_dir
            / 'augmented_crops'
            / policy_id
            / f'{self._safe_id(source_image_id)}__aug_{index + 1:04d}.jpg'
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            transformed = pipeline(image=self.schema.numpy_array(image.convert('RGB')))
            Image.fromarray(transformed['image']).save(destination)

        augmented_record = dict(image_record)
        augmented_record.update(
            {
                'image_id': f'{source_image_id}__aug_{index + 1:04d}',
                'sample_id': f'{source_sample_id}__aug_{index + 1:04d}',
                'path': destination.relative_to(context.result_dir).as_posix(),
                'split': split,
                'parent_sample_id': source_sample_id,
                'original_sample_id': self._original_sample_id(image_record=image_record),
                'augmentation_id': augmentation_id,
                'augmentation_recipe': recipe,
                'label_transform': label_transform,
                'augmentation_engine': 'albumentations',
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
