import json
from pathlib import Path
from typing import Any

from ironflow_exp.datasets import AugmentationPolicyValidator
from ironflow_exp.domain import SampleRecord
from ironflow_exp.engine.tasks.base import BaseTaskAdapter, TaskAdapterResult, TaskExecutionContext


class AugmentationPolicySmokeTaskAdapter(BaseTaskAdapter):
    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        dataset_root = self._dataset_root(context=context)
        manifest = self._read_json(dataset_root / 'manifest.json')
        target_split = str(context.record.params.get('target_split', 'train'))
        policy_id = str(context.record.params.get('policy_id', 'noop_smoke_v1'))
        recipe = str(context.record.params.get('augmentation_recipe', policy_id))
        label_transform = str(context.record.params.get('label_transform', 'identity'))
        augmented_records = self._augmented_records(
            manifest=manifest,
            target_split=target_split,
            policy_id=policy_id,
            recipe=recipe,
            label_transform=label_transform,
        )
        validation_samples = self._validation_samples(
            manifest=manifest,
            augmented_records=augmented_records,
            target_split=target_split,
        )
        validation = AugmentationPolicyValidator().validate_samples(
            samples=validation_samples,
            target_split=target_split,
        )
        if not validation.is_valid:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message='augmentation policy smoke validation failed',
                artifacts=[],
                metadata={
                    'adapter': 'augmentation_policy_smoke',
                    'policy_id': policy_id,
                    'errors': validation.error_messages(),
                    'warnings': validation.warning_messages(),
                },
            )

        output_path = context.result_dir / 'augmentation_manifest.json'
        payload = {
            'schema_version': '0.1',
            'artifact_type': 'augmentation_manifest',
            'dataset_id': f"{manifest['dataset_id']}_augmented_{policy_id}",
            'source_dataset_id': str(manifest['dataset_id']),
            'target_split': target_split,
            'policy_id': policy_id,
            'augmentation_count': len(augmented_records),
            'records': augmented_records,
        }
        self._write_json(output_path, payload)

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'augmentation policy smoke completed: augmented_samples={len(augmented_records)}',
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
            artifacts=[
                {
                    'name': 'augmentation_manifest',
                    'path': 'augmentation_manifest.json',
                    'kind': 'intermediate_manifest',
                    'required': True,
                },
            ],
            metadata={
                'adapter': 'augmentation_policy_smoke',
                'dataset_root': str(dataset_root),
                'policy_id': policy_id,
                'target_split': target_split,
                'augmentation_count': len(augmented_records),
                'warnings': validation.warning_messages(),
            },
        )

    def _dataset_root(self, *, context: TaskExecutionContext) -> Path:
        if context.record.input_variant_path:
            return Path(context.record.input_variant_path)

        raise ValueError('augmentation_policy_smoke requires input_variant_path')

    def _augmented_records(
        self,
        *,
        manifest: dict[str, Any],
        target_split: str,
        policy_id: str,
        recipe: str,
        label_transform: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, image_record in enumerate(self._images(manifest=manifest)):
            split = str(image_record.get('split') or manifest.get('split') or '')
            if split != target_split:
                continue
            sample_id = str(image_record['sample_id'])
            augmentation_id = f'{policy_id}_{index + 1:04d}'
            records.append(
                {
                    'sample_id': f'{sample_id}__aug_{index + 1:04d}',
                    'image_id': f"{image_record['image_id']}__aug_{index + 1:04d}",
                    'path': str(image_record['path']),
                    'split': target_split,
                    'parent_sample_id': sample_id,
                    'original_sample_id': sample_id,
                    'augmentation_id': augmentation_id,
                    'augmentation_recipe': recipe,
                    'label_transform': label_transform,
                    'label': image_record.get('label'),
                    'source_path': str(image_record['path']),
                },
            )

        return records

    def _validation_samples(
        self,
        *,
        manifest: dict[str, Any],
        augmented_records: list[dict[str, Any]],
        target_split: str,
    ) -> list[SampleRecord]:
        samples = [
            self._sample_from_image(image_record=image_record, default_split=str(manifest.get('split') or ''))
            for image_record in self._images(manifest=manifest)
            if str(image_record.get('split') or manifest.get('split') or '') == target_split
        ]
        samples.extend(
            SampleRecord(
                sample_id=str(record['sample_id']),
                image_id=str(record['image_id']),
                split=str(record['split']),
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

    def _sample_from_image(self, *, image_record: dict[str, Any], default_split: str) -> SampleRecord:
        sample_id = str(image_record['sample_id'])
        return SampleRecord(
            sample_id=sample_id,
            image_id=str(image_record['image_id']),
            split=str(image_record.get('split') or default_split),
            image_path=str(image_record['path']),
            label_path=None,
            source_id=sample_id,
            source_type='original',
            split_group_id=sample_id,
            metadata={},
        )

    def _images(self, *, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        images = manifest.get('images')
        if not isinstance(images, list):
            raise ValueError('manifest.images must be a list')

        return [
            image
            for image in images
            if isinstance(image, dict)
        ]

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
