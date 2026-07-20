from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import ceil, floor
from pathlib import Path

# *************************
# External Library
#   - Pillow : ver 12.1.1
# *************************
from PIL import Image

from ironflow_exp.datasets import DatasetManifest, ObjectManifest
from ironflow_exp.domain import ArtifactRecord, ObjectRecord, SampleRecord


@dataclass(frozen=True, slots=True)
class CropGenerationResult:
    object_manifest: ObjectManifest
    artifacts: list[ArtifactRecord]


@dataclass(frozen=True, slots=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


class GtCropGenerator:
    def generate(
        self,
        dataset_manifest: DatasetManifest,
        object_manifest: ObjectManifest,
        source_root: str | Path,
        output_root: str | Path,
        padding_ratio: float = 0.08,
        image_format: str = 'jpg',
    ) -> CropGenerationResult:
        source_root_path = Path(source_root)
        output_root_path = Path(output_root)
        sample_by_id = {sample.sample_id: sample for sample in dataset_manifest.samples}
        updated_objects: list[ObjectRecord] = []
        artifacts: list[ArtifactRecord] = []

        for object_record in object_manifest.objects:
            sample = self._get_sample(sample_by_id=sample_by_id, object_record=object_record)
            updated_object, artifact = self._create_crop(
                object_record=object_record,
                sample=sample,
                source_root=source_root_path,
                output_root=output_root_path,
                padding_ratio=padding_ratio,
                image_format=image_format,
                artifact_index=len(artifacts),
            )
            updated_objects.append(updated_object)
            artifacts.append(artifact)

        return CropGenerationResult(
            object_manifest=ObjectManifest(
                schema_version=object_manifest.schema_version,
                run_id=object_manifest.run_id,
                dataset_id=object_manifest.dataset_id,
                source=object_manifest.source,
                objects=updated_objects,
            ),
            artifacts=artifacts,
        )

    def _get_sample(
        self,
        sample_by_id: dict[str, SampleRecord],
        object_record: ObjectRecord,
    ) -> SampleRecord:
        if object_record.sample_id not in sample_by_id:
            raise ValueError(f'sample not found for object: object_id={object_record.object_id}')

        return sample_by_id[object_record.sample_id]

    def _create_crop(
        self,
        object_record: ObjectRecord,
        sample: SampleRecord,
        source_root: Path,
        output_root: Path,
        padding_ratio: float,
        image_format: str,
        artifact_index: int,
    ) -> tuple[ObjectRecord, ArtifactRecord]:
        image_path = source_root / sample.image_path
        if not image_path.exists():
            raise FileNotFoundError(f'image file not found: {image_path}')

        with Image.open(image_path) as image:
            image_rgb = image.convert('RGB')
            bbox_xyxy = self._resolve_object_bbox_xyxy(
                object_record=object_record,
                image_width=image_rgb.width,
                image_height=image_rgb.height,
            )
            crop_box = self._make_crop_box(
                bbox_xyxy=bbox_xyxy,
                image_width=image_rgb.width,
                image_height=image_rgb.height,
                padding_ratio=padding_ratio,
            )
            crop = image_rgb.crop(tuple(crop_box.as_list))
            crop_path = self._crop_path(
                output_root=output_root,
                object_record=object_record,
                image_format=image_format,
            )
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)

        relative_crop_path = crop_path.relative_to(output_root).as_posix()
        updated_object = replace(
            object_record,
            bbox_xyxy=bbox_xyxy,
            crop_path=relative_crop_path,
            crop_params={
                'padding_ratio': padding_ratio,
                'clip_to_image': True,
                'resize_mode': 'none',
                'output_size': None,
                'crop_box_xyxy': crop_box.as_list,
            },
        )

        return updated_object, self._make_artifact(
            object_record=updated_object,
            crop_path=relative_crop_path,
            artifact_index=artifact_index,
            image_format=image_format,
        )

    def _resolve_object_bbox_xyxy(
        self,
        object_record: ObjectRecord,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        bbox_xyxy = object_record.bbox_xyxy

        if bbox_xyxy is None:
            bbox_xyxy = self._yolo_to_xyxy(
                object_record=object_record,
                image_width=image_width,
                image_height=image_height,
            )

        x1, y1, x2, y2 = bbox_xyxy
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width <= 0 or bbox_height <= 0:
            raise ValueError(f'invalid bbox for object: object_id={object_record.object_id}')

        return bbox_xyxy

    def _make_crop_box(
        self,
        bbox_xyxy: list[float],
        image_width: int,
        image_height: int,
        padding_ratio: float,
    ) -> CropBox:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        padding_x = bbox_width * padding_ratio
        padding_y = bbox_height * padding_ratio

        return CropBox(
            x1=max(0, floor(x1 - padding_x)),
            y1=max(0, floor(y1 - padding_y)),
            x2=min(image_width, ceil(x2 + padding_x)),
            y2=min(image_height, ceil(y2 + padding_y)),
        )

    def _yolo_to_xyxy(
        self,
        object_record: ObjectRecord,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        if object_record.bbox_yolo is None:
            raise ValueError(f'bbox is missing for object: object_id={object_record.object_id}')

        x_center, y_center, bbox_width, bbox_height = object_record.bbox_yolo
        absolute_x = x_center * image_width
        absolute_y = y_center * image_height
        absolute_width = bbox_width * image_width
        absolute_height = bbox_height * image_height

        return [
            absolute_x - absolute_width / 2,
            absolute_y - absolute_height / 2,
            absolute_x + absolute_width / 2,
            absolute_y + absolute_height / 2,
        ]

    def _crop_path(
        self,
        output_root: Path,
        object_record: ObjectRecord,
        image_format: str,
    ) -> Path:
        class_name = object_record.class_name or f'class_{object_record.class_id}'

        return (
            output_root
            / 'artifacts'
            / 'crops'
            / object_record.split
            / class_name
            / f'{object_record.object_id}.{image_format}'
        )

    def _make_artifact(
        self,
        object_record: ObjectRecord,
        crop_path: str,
        artifact_index: int,
        image_format: str,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=f'artifact_crop_{artifact_index + 1:08d}',
            task='preprocessing',
            role='crop',
            path=crop_path,
            format=image_format,
            model_id=None,
            sample_id=object_record.sample_id,
            object_id=object_record.object_id,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            metadata={
                'source': object_record.source,
            },
        )
