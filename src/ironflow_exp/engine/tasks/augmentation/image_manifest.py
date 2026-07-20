import json
from pathlib import Path
from typing import Any

from PIL import Image

from ironflow_exp.datasets import AugmentationPolicyValidator
from ironflow_exp.domain import SampleRecord
from ironflow_exp.engine.core import UserImageDatasetValidator
from ironflow_exp.engine.tasks.base import BaseTaskAdapter, TaskAdapterResult, TaskExecutionContext


class ImageManifestAugmentationTaskAdapter(BaseTaskAdapter):
    output_manifest_artifact_name = 'detection_input_manifest'
    output_manifest_metadata_key = 'detection_input_manifest'
    output_manifest_artifact_kind = 'dataset_manifest'

    def __init__(self, dataset_validator: UserImageDatasetValidator | None = None) -> None:
        self.dataset_validator = dataset_validator or UserImageDatasetValidator()

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        dataset_root = self._dataset_root(context=context)
        source_manifest = self._read_json(dataset_root / 'manifest.json')
        target_split = str(context.record.params.get('target_split', 'train'))
        policy_id = str(context.record.params.get('policy_id', self.default_policy_id))
        recipe = str(context.record.params.get('augmentation_recipe', self.default_recipe))
        label_transform = str(context.record.params.get('label_transform', self.default_label_transform))

        source_records, augmented_records = self._write_augmented_dataset(
            context=context,
            dataset_root=dataset_root,
            source_manifest=source_manifest,
            target_split=target_split,
            policy_id=policy_id,
            recipe=recipe,
            label_transform=label_transform,
        )
        validation = AugmentationPolicyValidator().validate_samples(
            samples=self._validation_samples(
                source_records=[
                    record
                    for record in source_records
                    if str(record.get('split') or source_manifest.get('split') or '') == target_split
                ],
                augmented_records=augmented_records,
                target_split=target_split,
            ),
            target_split=target_split,
        )
        if not validation.is_valid:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message=f'{self.adapter_label} validation failed',
                artifacts=[],
                metadata={
                    'adapter': self.adapter_key,
                    'policy_id': policy_id,
                    'dataset_root': str(dataset_root),
                    'errors': validation.error_messages(),
                    'warnings': validation.warning_messages(),
                },
            )

        output_manifest = {
            'schema_version': str(source_manifest.get('schema_version') or '0.1'),
            'dataset_id': f"{source_manifest['dataset_id']}_augmented_{policy_id}",
            'dataset_type': str(source_manifest.get('dataset_type') or 'detection'),
            'classes': source_manifest.get('classes', []),
            'images': [*source_records, *augmented_records],
        }
        augmentation_manifest = {
            'schema_version': '0.1',
            'artifact_type': 'augmentation_manifest',
            'dataset_id': output_manifest['dataset_id'],
            'source_dataset_id': str(source_manifest['dataset_id']),
            'target_split': target_split,
            'policy_id': policy_id,
            'augmentation_count': len(augmented_records),
            'records': augmented_records,
        }
        output_manifest_path = context.result_dir / 'manifest.json'
        augmentation_manifest_path = context.result_dir / 'augmentation_manifest.json'
        self._write_json(output_manifest_path, output_manifest)
        self._write_json(augmentation_manifest_path, augmentation_manifest)

        dataset_validation = self.dataset_validator.validate_folder(context.result_dir)
        if not dataset_validation.is_valid:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message=f'{self.adapter_label} wrote invalid input manifest',
                artifacts=[],
                metadata={
                    'adapter': self.adapter_key,
                    'errors': dataset_validation.error_messages(),
                },
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'{self.adapter_label} completed: augmented_images={len(augmented_records)}',
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
            artifacts=self._result_artifacts(policy_id=policy_id),
            metadata={
                'adapter': self.adapter_key,
                'dataset_root': str(dataset_root),
                'policy_id': policy_id,
                'target_split': target_split,
                'source_image_count': len(source_records),
                'augmentation_count': len(augmented_records),
                self.output_manifest_metadata_key: str(output_manifest_path),
                'warnings': validation.warning_messages(),
            },
        )

    def _result_artifacts(self, *, policy_id: str) -> list[dict[str, object]]:
        return [
            {
                'name': 'augmentation_manifest',
                'path': 'augmentation_manifest.json',
                'kind': 'intermediate_manifest',
                'required': True,
            },
            {
                'name': self.output_manifest_artifact_name,
                'path': 'manifest.json',
                'kind': self.output_manifest_artifact_kind,
                'required': True,
            },
            {
                'name': 'augmented_images',
                'path': f'augmented_images/{policy_id}',
                'kind': 'image_dir',
                'required': True,
            },
        ]

    def _write_augmented_dataset(
        self,
        *,
        context: TaskExecutionContext,
        dataset_root: Path,
        source_manifest: dict[str, Any],
        target_split: str,
        policy_id: str,
        recipe: str,
        label_transform: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source_records: list[dict[str, Any]] = []
        augmented_records: list[dict[str, Any]] = []
        for index, image_record in enumerate(self._images(manifest=source_manifest)):
            split = str(image_record.get('split') or source_manifest.get('split') or '')
            source_path = self._safe_source_path(source_root=dataset_root, image_record=image_record)
            copied_record = self._copy_source_image(
                context=context,
                image_record=image_record,
                source_path=source_path,
                split=split,
            )
            source_records.append(copied_record)
            if split != target_split:
                continue
            augmented_records.append(
                self._write_augmented_image(
                    context=context,
                    image_record=image_record,
                    source_path=source_path,
                    split=split,
                    policy_id=policy_id,
                    recipe=recipe,
                    label_transform=label_transform,
                    index=index,
                ),
            )

        return source_records, augmented_records

    def _copy_source_image(
        self,
        *,
        context: TaskExecutionContext,
        image_record: dict[str, Any],
        source_path: Path,
        split: str,
    ) -> dict[str, Any]:
        destination = context.result_dir / 'inputs' / str(image_record['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            image.convert('RGB').save(destination)

        copied_record = dict(image_record)
        copied_record['path'] = destination.relative_to(context.result_dir).as_posix()
        copied_record['split'] = split
        copied_record['objects'] = self._objects(image_record=image_record)

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
        raise NotImplementedError

    def _validation_samples(
        self,
        *,
        source_records: list[dict[str, Any]],
        augmented_records: list[dict[str, Any]],
        target_split: str,
    ) -> list[SampleRecord]:
        samples = [
            SampleRecord(
                sample_id=str(record['sample_id']),
                image_id=str(record['image_id']),
                split=str(record.get('split') or target_split),
                image_path=str(record['path']),
                label_path=None,
                source_id=str(record['sample_id']),
                source_type='original',
                split_group_id=str(record['sample_id']),
                parent_sample_id=None,
                metadata={},
            )
            for record in source_records
        ]
        samples.extend(
            SampleRecord(
                sample_id=str(record['sample_id']),
                image_id=str(record['image_id']),
                split=str(record.get('split') or target_split),
                image_path=str(record['path']),
                label_path=None,
                source_id=str(record['original_sample_id']),
                source_type='augmentation',
                split_group_id=str(record['original_sample_id']),
                parent_sample_id=str(record['parent_sample_id']),
                metadata={
                    'original_sample_id': str(record['original_sample_id']),
                    'augmentation_id': str(record['augmentation_id']),
                    'augmentation_recipe': str(record['augmentation_recipe']),
                    'label_transform': str(record['label_transform']),
                },
            )
            for record in augmented_records
        )

        return samples

    def _dataset_root(self, *, context: TaskExecutionContext) -> Path:
        if context.record.input_variant_path:
            return Path(context.record.input_variant_path)

        raise ValueError(f'{self.adapter_key} requires input_variant_path')

    def _objects(self, *, image_record: dict[str, Any]) -> list[dict[str, Any]]:
        objects = image_record.get('objects')
        if not isinstance(objects, list):
            return []

        return [
            dict(item)
            for item in objects
            if isinstance(item, dict)
        ]

    def _safe_source_path(self, *, source_root: Path, image_record: dict[str, Any]) -> Path:
        raw_path = Path(str(image_record['path']))
        if raw_path.is_absolute() or '..' in raw_path.parts:
            raise ValueError('image path must be a safe relative path')
        resolved = (source_root / raw_path).resolve()
        try:
            resolved.relative_to(source_root.resolve())
        except ValueError as exc:
            raise ValueError('image path escapes source manifest root') from exc

        return resolved

    def _images(self, *, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        images = manifest.get('images')
        if not isinstance(images, list):
            raise ValueError('manifest.images must be a list')

        return [
            image
            for image in images
            if isinstance(image, dict)
        ]

    def _safe_id(self, value: str) -> str:
        safe_value = ''.join(
            character if character.isalnum() or character in {'_', '-'} else '_'
            for character in value
        )

        return safe_value.strip('_') or 'image'

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')

        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
