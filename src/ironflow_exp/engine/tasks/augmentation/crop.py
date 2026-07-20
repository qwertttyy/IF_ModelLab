import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ironflow_exp.datasets import AugmentationPolicyValidator
from ironflow_exp.domain import SampleRecord
from ironflow_exp.engine.core import (
    DetectionToClassificationArtifactValidator,
    INTERMEDIATE_ARTIFACT_SCHEMA_VERSION,
)
from ironflow_exp.engine.datasets.dataset_input_resolver import DatasetInputResolver
from ironflow_exp.engine.tasks.base import BaseTaskAdapter, TaskAdapterResult, TaskExecutionContext


class ClassificationCropAugmentationSmokeTaskAdapter(BaseTaskAdapter):
    def __init__(self, artifact_validator: DetectionToClassificationArtifactValidator | None = None) -> None:
        self.artifact_validator = artifact_validator or DetectionToClassificationArtifactValidator()

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        source_manifest_path, source_manifest, source_root = self._source_manifest(context=context)
        target_split = str(context.record.params.get('target_split', 'train'))
        policy_id = str(context.record.params.get('policy_id', 'classification_crop_flip_smoke_v1'))
        recipe = str(context.record.params.get('augmentation_recipe', 'horizontal_flip_smoke_v1'))
        label_transform = str(context.record.params.get('label_transform', 'preserve_class_label'))

        original_records, augmented_records = self._write_augmented_crops(
            context=context,
            source_manifest=source_manifest,
            source_root=source_root,
            target_split=target_split,
            policy_id=policy_id,
            recipe=recipe,
            label_transform=label_transform,
        )
        validation = AugmentationPolicyValidator().validate_samples(
            samples=self._validation_samples(
                source_records=original_records,
                augmented_records=augmented_records,
                target_split=target_split,
            ),
            target_split=target_split,
        )
        if not validation.is_valid:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message='classification crop augmentation smoke validation failed',
                artifacts=[],
                metadata={
                    'adapter': 'classification_crop_augmentation_smoke',
                    'policy_id': policy_id,
                    'source_manifest': str(source_manifest_path),
                    'errors': validation.error_messages(),
                    'warnings': validation.warning_messages(),
                },
            )

        output_manifest = {
            'schema_version': INTERMEDIATE_ARTIFACT_SCHEMA_VERSION,
            'artifact_type': 'classification_input_manifest',
            'dataset_id': f"{source_manifest['dataset_id']}_augmented_{policy_id}",
            'input_source': str(source_manifest['input_source']),
            'classes': source_manifest.get('classes', []),
            'source_manifest_path': 'augmentation_manifest.json',
            'images': [*original_records, *augmented_records],
        }
        augmentation_manifest = {
            'schema_version': '0.1',
            'artifact_type': 'augmentation_manifest',
            'dataset_id': output_manifest['dataset_id'],
            'source_dataset_id': str(source_manifest['dataset_id']),
            'source_manifest_path': source_manifest_path.as_posix(),
            'target_split': target_split,
            'policy_id': policy_id,
            'augmentation_count': len(augmented_records),
            'records': augmented_records,
        }
        augmentation_manifest_path = context.result_dir / 'augmentation_manifest.json'
        classification_manifest_path = context.result_dir / 'classification_input_manifest.json'
        self._write_json(augmentation_manifest_path, augmentation_manifest)
        self._write_json(classification_manifest_path, output_manifest)
        intermediate_validation = self.artifact_validator.validate_file(
            classification_manifest_path,
            expected_artifact_type='classification_input_manifest',
            check_files=True,
        )
        if not intermediate_validation.is_valid:
            return TaskAdapterResult(
                success=False,
                status='failed',
                message='classification crop augmentation smoke wrote invalid classification manifest',
                artifacts=[],
                metadata={
                    'adapter': 'classification_crop_augmentation_smoke',
                    'errors': intermediate_validation.error_messages(),
                },
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'classification crop augmentation smoke completed: augmented_crops={len(augmented_records)}',
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
                {
                    'name': 'classification_input_manifest',
                    'path': 'classification_input_manifest.json',
                    'kind': 'intermediate_manifest',
                    'required': True,
                },
                {
                    'name': 'augmented_images',
                    'path': f'augmented_crops/{policy_id}',
                    'kind': 'image_dir',
                    'required': True,
                },
            ],
            metadata={
                'adapter': 'classification_crop_augmentation_smoke',
                'source_manifest': str(source_manifest_path),
                'policy_id': policy_id,
                'target_split': target_split,
                'source_crop_count': len(original_records),
                'augmentation_count': len(augmented_records),
                'classification_input_manifest': str(classification_manifest_path),
                'warnings': validation.warning_messages(),
            },
        )

    def _source_manifest(self, *, context: TaskExecutionContext) -> tuple[Path, dict[str, Any], Path]:
        try:
            source_manifest_path = context.first_dependency_artifact_path('classification_input_manifest')
        except KeyError:
            if context.record.input_variant_path is None:
                raise ValueError(
                    'classification crop augmentation requires a classification_input_manifest dependency '
                    'or input_variant_path pointing to an imagefolder dataset',
                )
            return self._source_manifest_from_imagefolder(context=context)

        return source_manifest_path, self._read_json(source_manifest_path), source_manifest_path.parent

    def _source_manifest_from_imagefolder(self, *, context: TaskExecutionContext) -> tuple[Path, dict[str, Any], Path]:
        resolution = DatasetInputResolver().resolve_classification(
            root=context.record.input_variant_path,
            input_kind=context.record.input_variant_kind,
        )
        source_root = Path(resolution.dataset_root)
        classes = [str(class_name) for class_name in resolution.classes if str(class_name)]
        extensions = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
        images: list[dict[str, object]] = []
        for split, split_root in sorted(resolution.split_roots.items()):
            for class_dir in sorted(Path(split_root).iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name
                for image_path in sorted(class_dir.rglob('*')):
                    if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                        continue
                    try:
                        relative_path = image_path.relative_to(source_root)
                    except ValueError:
                        relative_path = Path(split) / label / image_path.name
                    sample_id = f'{split}_{label}_{image_path.stem}'
                    images.append(
                        {
                            'image_id': sample_id,
                            'sample_id': sample_id,
                            'label': label,
                            'path': relative_path.as_posix(),
                            'split': str(split),
                        },
                    )
        if not images:
            raise ValueError(f'classification imagefolder has no images: {source_root}')

        manifest = {
            'schema_version': INTERMEDIATE_ARTIFACT_SCHEMA_VERSION,
            'artifact_type': 'classification_input_manifest',
            'dataset_id': source_root.name,
            'input_source': 'imagefolder',
            'classes': classes,
            'images': images,
        }
        return source_root / 'classification_input_manifest.generated.json', manifest, source_root

    def _write_augmented_crops(
        self,
        *,
        context: TaskExecutionContext,
        source_manifest: dict[str, Any],
        source_root: Path,
        target_split: str,
        policy_id: str,
        recipe: str,
        label_transform: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        original_records: list[dict[str, Any]] = []
        augmented_records: list[dict[str, Any]] = []
        for index, image_record in enumerate(self._images(manifest=source_manifest)):
            split = str(image_record.get('split') or source_manifest.get('split') or target_split)
            if split != target_split:
                continue
            source_path = self._safe_source_path(source_root=source_root, image_record=image_record)
            original_record, copied_source_path = self._copy_source_crop(
                context=context,
                image_record=image_record,
                source_path=source_path,
                split=split,
            )
            augmented_record = self._write_flipped_crop(
                context=context,
                image_record=image_record,
                source_path=source_path,
                split=split,
                policy_id=policy_id,
                recipe=recipe,
                label_transform=label_transform,
                index=index,
            )
            augmented_record['augmentation_source_path'] = copied_source_path.as_posix()
            original_records.append(original_record)
            augmented_records.append(augmented_record)

        return original_records, augmented_records

    def _copy_source_crop(
        self,
        *,
        context: TaskExecutionContext,
        image_record: dict[str, Any],
        source_path: Path,
        split: str,
    ) -> tuple[dict[str, Any], Path]:
        destination = context.result_dir / 'inputs' / str(image_record['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            image.convert('RGB').save(destination)

        copied_record = dict(image_record)
        copied_record['path'] = destination.relative_to(context.result_dir).as_posix()
        copied_record['split'] = split

        return copied_record, Path(copied_record['path'])

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
            ImageOps.mirror(image.convert('RGB')).save(destination)

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
            },
        )

        return augmented_record

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
                source_id=self._original_sample_id(image_record=record),
                source_type='crop',
                split_group_id=self._original_sample_id(image_record=record),
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

    def _original_sample_id(self, *, image_record: dict[str, Any]) -> str:
        original_sample_id = image_record.get('original_sample_id')
        if isinstance(original_sample_id, str) and original_sample_id:
            return original_sample_id

        return str(image_record['sample_id'])

    def _safe_source_path(self, *, source_root: Path, image_record: dict[str, Any]) -> Path:
        raw_path = Path(str(image_record['path']))
        if raw_path.is_absolute() or '..' in raw_path.parts:
            raise ValueError('classification crop image path must be a safe relative path')
        resolved = (source_root / raw_path).resolve()
        try:
            resolved.relative_to(source_root.resolve())
        except ValueError as exc:
            raise ValueError('classification crop image path escapes source manifest root') from exc

        return resolved

    def _images(self, *, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        images = manifest.get('images')
        if not isinstance(images, list):
            raise ValueError('classification_input_manifest.images must be a list')

        return [
            image
            for image in images
            if isinstance(image, dict)
        ]

    def _safe_id(self, value: str) -> str:
        return ''.join(character if character.isalnum() or character in {'_', '-'} else '_' for character in value).strip('_') or 'crop'

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
