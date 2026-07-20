import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from ironflow_exp.engine.tasks.augmentation_adapters import (
    AlbumentationsClassificationCropTaskAdapter,
    AlbumentationsDetectionBBoxTaskAdapter,
    AlbumentationsSegmentationMaskTaskAdapter,
    AugmentationPolicySmokeTaskAdapter,
    ClassificationCropAugmentationSmokeTaskAdapter,
    DetectionBBoxAugmentationSmokeTaskAdapter,
)
from ironflow_exp.engine.tasks.base import BaseTaskAdapter, TaskAdapterResult, TaskExecutionContext
from ironflow_exp.engine.tasks.model_task_adapters import (
    AsahiSlicingDetectionTaskAdapter,
    ClassifierPenultimateEmbeddingTaskAdapter,
    ClipEmbeddingTaskAdapter,
    CocaClassifierTaskAdapter,
    CocaEmbeddingTaskAdapter,
    DFineDetectionTaskAdapter,
    DetectionGuidedSegmentationTaskAdapter,
    DinoXOpenWorldDetectionTaskAdapter,
    DinoXSegmentationTaskAdapter,
    Dinov2EmbeddingTaskAdapter,
    Dinov3EmbeddingTaskAdapter,
    FastSamSegmentationTaskAdapter,
    GroundedSamSegmentationTaskAdapter,
    GroundingDinoOpenVocabDetectionTaskAdapter,
    MMDetectionTaskAdapter,
    OwlV2OpenVocabDetectionTaskAdapter,
    RfDetrDetectionTaskAdapter,
    RtDetrDetectionTaskAdapter,
    RtDetrV2DetectionTaskAdapter,
    LwDetrDetectionTaskAdapter,
    SahiSlicingDetectionTaskAdapter,
    SamPromptableSegmentationTaskAdapter,
    SiglipEmbeddingTaskAdapter,
    TimmClassifierTaskAdapter,
    TorchvisionClassifierTaskAdapter,
    UltralyticsYoloClassifierTaskAdapter,
    UltralyticsYoloSegmentationTaskAdapter,
    UltralyticsYoloTaskAdapter,
    VitClassifierTaskAdapter,
    YoloEOpenVocabDetectionTaskAdapter,
    YoloESegmentationTaskAdapter,
    YoloWorldOpenVocabDetectionTaskAdapter,
    Yolov10DetectionTaskAdapter,
)
from ironflow_exp.engine.tasks.registry import TaskAdapterRegistry
from ironflow_exp.engine.tasks.tracking_adapters import (
    BotSortTrackingTaskAdapter,
    BoostTrackTrackingTaskAdapter,
    ByteTrackTaskAdapter,
    DeepSortTrackingTaskAdapter,
    OCSortTrackingTaskAdapter,
    PdSortTrackingTaskAdapter,
    SortTrackingTaskAdapter,
    StrongSortTrackingTaskAdapter,
    TdlpTrackingTaskAdapter,
    TrackTrackTrackingTaskAdapter,
)
from ironflow_exp.engine.transforms import DetectionToClassificationCropTransformer


class ManifestDetectionSmokeTaskAdapter(BaseTaskAdapter):
    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        dataset_root = self._dataset_root(context)
        manifest = self._read_json(dataset_root / 'manifest.json')
        prediction_path = context.result_dir / 'predictions' / 'detection_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._prediction_records(context=context, dataset_root=dataset_root, manifest=manifest)
        payload = {
            'schema_version': '0.1',
            'task': 'detection',
            'model_id': context.record.model_id or context.record.adapter or 'manifest_detection_smoke',
            'dataset_id': str(manifest['dataset_id']),
            'success': True,
            'records': records,
        }
        self._write_json(prediction_path, payload)

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'manifest detection smoke completed: predictions={len(records)}',
            metrics=[
                {
                    'epoch': 1,
                    'train_loss': '',
                    'val_loss': '',
                    'accuracy': '',
                    'map50': 1.0,
                    'map50_95': 1.0,
                    'lr': '',
                },
            ],
            predictions=records,
            artifacts=[
                {
                    'name': 'detection_predictions',
                    'path': 'predictions/detection_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
            ],
            metadata={
                'adapter': 'manifest_detection_smoke',
                'dataset_root': str(dataset_root),
                'prediction_count': len(records),
            },
        )

    def _prediction_records(
        self,
        *,
        context: TaskExecutionContext,
        dataset_root: Path,
        manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        default_class_id = str(context.record.params.get('default_class_id', 'object'))
        default_score = float(context.record.params.get('score', 0.95))

        for image_index, image_record in enumerate(manifest['images']):
            image_path = dataset_root / str(image_record['path'])
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            objects = image_record.get('objects')
            usable_objects = objects if isinstance(objects, list) and objects else [None]
            for object_index, object_record in enumerate(usable_objects):
                bbox = self._bbox_for_record(
                    object_record=object_record,
                    image_width=image_width,
                    image_height=image_height,
                )
                object_id = self._object_id(
                    image_record=image_record,
                    object_record=object_record,
                    image_index=image_index,
                    object_index=object_index,
                )
                class_id = self._class_id(object_record=object_record, default_class_id=default_class_id)
                records.append(
                    {
                        'image_id': str(image_record['image_id']),
                        'sample_id': str(image_record['sample_id']),
                        'prediction_id': f'pred_{self._safe_id(object_id)}',
                        'class_id': class_id,
                        'score': default_score,
                        'bbox_xyxy': bbox,
                        'image_width': image_width,
                        'image_height': image_height,
                    },
                )

        return records

    def _bbox_for_record(self, *, object_record: object, image_width: int, image_height: int) -> list[float]:
        if isinstance(object_record, dict):
            bbox = object_record.get('bbox_xyxy')
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(value) for value in bbox]

        x1 = image_width * 0.25
        y1 = image_height * 0.25
        x2 = image_width * 0.75
        y2 = image_height * 0.75

        return [x1, y1, x2, y2]

    def _object_id(
        self,
        *,
        image_record: dict[str, Any],
        object_record: object,
        image_index: int,
        object_index: int,
    ) -> str:
        if isinstance(object_record, dict) and isinstance(object_record.get('object_id'), str):
            return str(object_record['object_id'])

        image_id = str(image_record.get('image_id') or f'image_{image_index:04d}')
        return f'{image_id}_object_{object_index:04d}'

    def _class_id(self, *, object_record: object, default_class_id: str) -> str:
        if isinstance(object_record, dict) and isinstance(object_record.get('class_id'), str):
            return str(object_record['class_id'])

        return default_class_id

    def _dataset_root(self, context: TaskExecutionContext) -> Path:
        if context.record.input_variant_path:
            return Path(context.record.input_variant_path)
        try:
            return context.first_dependency_artifact_path('detection_input_manifest').parent
        except KeyError:
            pass

        raise ValueError('manifest_detection_smoke requires input_variant_path')

    def _safe_id(self, value: str) -> str:
        return re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_') or 'object'

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')

        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )


class ManifestSegmentationSmokeTaskAdapter(ManifestDetectionSmokeTaskAdapter):
    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        dataset_root = self._dataset_root(context)
        manifest = self._read_json(dataset_root / 'manifest.json')
        prediction_path = context.result_dir / 'predictions' / 'segmentation_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._segmentation_records(context=context, dataset_root=dataset_root, manifest=manifest)
        payload = {
            'schema_version': '0.1',
            'task': 'segmentation',
            'model_id': context.record.model_id or context.record.adapter or 'manifest_segmentation_smoke',
            'dataset_id': str(manifest['dataset_id']),
            'success': True,
            'records': records,
        }
        self._write_json(prediction_path, payload)

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'manifest segmentation smoke completed: predictions={len(records)}',
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
            predictions=records,
            artifacts=[
                {
                    'name': 'segmentation_predictions',
                    'path': 'predictions/segmentation_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
            ],
            metadata={
                'adapter': 'manifest_segmentation_smoke',
                'dataset_root': str(dataset_root),
                'prediction_count': len(records),
            },
        )

    def _segmentation_records(
        self,
        *,
        context: TaskExecutionContext,
        dataset_root: Path,
        manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        default_class_id = str(context.record.params.get('default_class_id', 'object'))
        default_score = float(context.record.params.get('score', 0.92))
        mask_kind = str(context.record.params.get('mask_kind', 'polygon'))

        for image_index, image_record in enumerate(manifest['images']):
            image_path = dataset_root / str(image_record['path'])
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            objects = image_record.get('objects')
            usable_objects = objects if isinstance(objects, list) and objects else [None]
            for object_index, object_record in enumerate(usable_objects):
                bbox = self._bbox_for_record(
                    object_record=object_record,
                    image_width=image_width,
                    image_height=image_height,
                )
                object_id = self._object_id(
                    image_record=image_record,
                    object_record=object_record,
                    image_index=image_index,
                    object_index=object_index,
                )
                class_id = self._class_id(object_record=object_record, default_class_id=default_class_id)
                records.append(
                    {
                        'image_id': str(image_record['image_id']),
                        'sample_id': str(image_record['sample_id']),
                        'prediction_id': f'seg_{self._safe_id(object_id)}',
                        'source_object_id': object_id,
                        'class_id': class_id,
                        'score': default_score,
                        'mask_kind': mask_kind,
                        'mask_polygon': self._bbox_polygon(bbox),
                        'bbox_xyxy': bbox,
                        'image_width': image_width,
                        'image_height': image_height,
                    },
                )

        return records

    def _bbox_polygon(self, bbox: list[float]) -> list[list[float]]:
        x1, y1, x2, y2 = bbox

        return [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]


class DetectionToClassificationCropTaskAdapter(BaseTaskAdapter):
    def __init__(self, transformer: DetectionToClassificationCropTransformer | None = None) -> None:
        self.transformer = transformer or DetectionToClassificationCropTransformer()

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        source_task_id = self._source_task_id(context)
        source_dataset_root = self._source_dataset_root(context)
        detection_prediction_path = context.dependency_artifact_path(source_task_id, 'detection_predictions')
        padding_ratio = float(context.record.params.get('padding_ratio', 0.0))
        min_score = float(context.record.params.get('min_score', 0.0))
        image_format = str(context.record.params.get('image_format', 'jpg'))
        input_source = str(context.record.params.get('input_source', 'detector_crop'))
        use_source_labels = bool(context.record.params.get('use_source_labels', True))
        classification_dataset_id = context.record.params.get('classification_dataset_id')

        result = self.transformer.transform(
            detection_prediction_path=detection_prediction_path,
            source_dataset_root=source_dataset_root,
            output_dir=context.result_dir,
            padding_ratio=padding_ratio,
            image_format=image_format,
            min_score=min_score,
            input_source=input_source,
            use_source_labels=use_source_labels,
            classification_dataset_id=classification_dataset_id if isinstance(classification_dataset_id, str) else None,
        )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'detection-to-classification crop transform completed: crops={result.crop_count}',
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
                    'name': 'detection_predictions',
                    'path': 'inputs/detection_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
                {
                    'name': 'crop_region_manifest',
                    'path': 'crop_region_manifest.json',
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
                    'name': 'crop_images',
                    'path': f'crops/{input_source}',
                    'kind': 'image_dir',
                    'required': True,
                },
            ],
            metadata={
                'adapter': 'detection_to_classification_crop',
                'source_task_id': source_task_id,
                'source_dataset_root': str(Path(source_dataset_root)),
                'crop_count': result.crop_count,
                'classification_input_manifest': str(result.classification_input_manifest_path),
            },
        )

    def _source_task_id(self, context: TaskExecutionContext) -> str:
        configured = context.record.params.get('source_task_id')
        if isinstance(configured, str) and configured:
            return configured
        if len(context.record.depends_on) == 1:
            return context.record.depends_on[0]

        raise ValueError('source_task_id must be set when the transform task has zero or multiple dependencies')

    def _source_dataset_root(self, context: TaskExecutionContext) -> Path:
        configured = context.record.params.get('source_dataset_root')
        if isinstance(configured, str) and configured:
            return Path(configured)
        if context.record.input_variant_path:
            return Path(context.record.input_variant_path)
        try:
            return context.first_dependency_artifact_path('detection_input_manifest').parent
        except KeyError:
            pass

        raise ValueError('source_dataset_root or input_variant_path is required for crop transform')


class ManifestClassificationSmokeTaskAdapter(BaseTaskAdapter):
    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        manifest_path = context.first_dependency_artifact_path('classification_input_manifest')
        manifest = self._read_json(manifest_path)
        prediction_path = context.result_dir / 'predictions' / 'classification_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._prediction_records(context=context, manifest=manifest)
        payload = {
            'schema_version': '0.1',
            'task': 'classification',
            'model_id': context.record.model_id or context.record.adapter or 'manifest_classification_smoke',
            'dataset_id': str(manifest['dataset_id']),
            'success': True,
            'records': records,
        }
        self._write_json(prediction_path, payload)
        accuracy = self._accuracy(records=records, manifest=manifest)

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'manifest classification smoke completed: predictions={len(records)}',
            metrics=[
                {
                    'epoch': 1,
                    'train_loss': '',
                    'val_loss': '',
                    'accuracy': accuracy,
                    'map50': '',
                    'map50_95': '',
                    'lr': '',
                },
            ],
            predictions=records,
            artifacts=[
                {
                    'name': 'classification_predictions',
                    'path': 'predictions/classification_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
            ],
            metadata={
                'adapter': 'manifest_classification_smoke',
                'classification_input_manifest': str(manifest_path),
                'prediction_count': len(records),
            },
        )

    def _prediction_records(self, *, context: TaskExecutionContext, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        classes = [
            str(class_id)
            for class_id in manifest.get('classes', [])
            if isinstance(class_id, str) and class_id
        ]
        fallback_class = str(context.record.params.get('fallback_class_id') or (classes[0] if classes else 'unknown'))
        records: list[dict[str, Any]] = []
        for image in manifest['images']:
            label = image.get('label')
            class_id = label if isinstance(label, str) and label in classes else fallback_class
            records.append(
                {
                    'image_id': str(image['image_id']),
                    'sample_id': str(image['sample_id']),
                    'object_id': image.get('object_id') if isinstance(image.get('object_id'), str) else None,
                    'top1_class_id': class_id,
                    'top1_score': 1.0,
                    'scores': {
                        candidate: 1.0 if candidate == class_id else 0.0
                        for candidate in (classes or [class_id])
                    },
                },
            )

        return records

    def _accuracy(self, *, records: list[dict[str, Any]], manifest: dict[str, Any]) -> float | str:
        labels = [
            image.get('label')
            for image in manifest['images']
        ]
        if not labels or not all(isinstance(label, str) and label for label in labels):
            return ''
        correct = 0
        for record, label in zip(records, labels, strict=True):
            if record['top1_class_id'] == label:
                correct += 1

        return correct / len(labels)

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')

        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )


def build_builtin_task_adapter_registry() -> TaskAdapterRegistry:
    registry = TaskAdapterRegistry()
    registry.register(task_type='detection', adapter_key='manifest_detection_smoke', factory=ManifestDetectionSmokeTaskAdapter)
    registry.register(task_type='segmentation', adapter_key='manifest_segmentation_smoke', factory=ManifestSegmentationSmokeTaskAdapter)
    registry.register(task_type='detection', adapter_key='ultralytics_yolo', factory=UltralyticsYoloTaskAdapter)
    registry.register(task_type='detection', adapter_key='d_fine_detection', factory=DFineDetectionTaskAdapter)
    registry.register(task_type='detection', adapter_key='rf_detr_detection', factory=RfDetrDetectionTaskAdapter)
    registry.register(task_type='detection', adapter_key='rt_detr_detection', factory=RtDetrDetectionTaskAdapter)
    registry.register(task_type='detection', adapter_key='rt_detr_v2_detection', factory=RtDetrV2DetectionTaskAdapter)
    registry.register(task_type='detection', adapter_key='lw_detr_detection', factory=LwDetrDetectionTaskAdapter)
    registry.register(task_type='detection', adapter_key='mmdet_detection', factory=MMDetectionTaskAdapter)
    registry.register(
        task_type='detection',
        adapter_key='yolo_world_open_vocab_detection',
        factory=YoloWorldOpenVocabDetectionTaskAdapter,
    )
    registry.register(
        task_type='detection',
        adapter_key='grounding_dino_open_vocab_detection',
        factory=GroundingDinoOpenVocabDetectionTaskAdapter,
    )
    registry.register(task_type='detection', adapter_key='yolov10_detection', factory=Yolov10DetectionTaskAdapter)
    registry.register(
        task_type='detection',
        adapter_key='owlv2_open_vocab_detection',
        factory=OwlV2OpenVocabDetectionTaskAdapter,
    )
    registry.register(
        task_type='detection',
        adapter_key='dino_x_open_world_detection',
        factory=DinoXOpenWorldDetectionTaskAdapter,
    )
    registry.register(task_type='detection', adapter_key='sahi_slicing_detection', factory=SahiSlicingDetectionTaskAdapter)
    registry.register(task_type='detection', adapter_key='asahi_slicing_detection', factory=AsahiSlicingDetectionTaskAdapter)
    registry.register(
        task_type='detection',
        adapter_key='yoloe_open_vocab_detection',
        factory=YoloEOpenVocabDetectionTaskAdapter,
    )
    registry.register(
        task_type='segmentation',
        adapter_key='ultralytics_yolo_segmentation',
        factory=UltralyticsYoloSegmentationTaskAdapter,
    )
    registry.register(
        task_type='segmentation',
        adapter_key='sam_promptable_segmentation',
        factory=SamPromptableSegmentationTaskAdapter,
    )
    registry.register(task_type='segmentation', adapter_key='fast_sam_segmentation', factory=FastSamSegmentationTaskAdapter)
    registry.register(
        task_type='segmentation',
        adapter_key='detection_guided_segmentation',
        factory=DetectionGuidedSegmentationTaskAdapter,
    )
    registry.register(
        task_type='segmentation',
        adapter_key='grounded_sam_segmentation',
        factory=GroundedSamSegmentationTaskAdapter,
    )
    registry.register(task_type='segmentation', adapter_key='yoloe_segmentation', factory=YoloESegmentationTaskAdapter)
    registry.register(task_type='segmentation', adapter_key='dino_x_segmentation', factory=DinoXSegmentationTaskAdapter)
    registry.register(task_type='embedding', adapter_key='dinov2_embedding', factory=Dinov2EmbeddingTaskAdapter)
    registry.register(task_type='embedding', adapter_key='dinov3_embedding', factory=Dinov3EmbeddingTaskAdapter)
    registry.register(task_type='embedding', adapter_key='clip_embedding', factory=ClipEmbeddingTaskAdapter)
    registry.register(task_type='embedding', adapter_key='siglip_embedding', factory=SiglipEmbeddingTaskAdapter)
    registry.register(
        task_type='embedding',
        adapter_key='classifier_penultimate_embedding',
        factory=ClassifierPenultimateEmbeddingTaskAdapter,
    )
    registry.register(task_type='embedding', adapter_key='coca_embedding', factory=CocaEmbeddingTaskAdapter)
    registry.register(
        task_type='preprocessing',
        adapter_key='detection_to_classification_crop',
        factory=DetectionToClassificationCropTaskAdapter,
    )
    registry.register(
        task_type='classification',
        adapter_key='manifest_classification_smoke',
        factory=ManifestClassificationSmokeTaskAdapter,
    )
    registry.register(
        task_type='classification',
        adapter_key='torchvision_classifier',
        factory=TorchvisionClassifierTaskAdapter,
    )
    registry.register(
        task_type='classification',
        adapter_key='timm_classifier',
        factory=TimmClassifierTaskAdapter,
    )
    registry.register(
        task_type='classification',
        adapter_key='ultralytics_yolo_classifier',
        factory=UltralyticsYoloClassifierTaskAdapter,
    )
    registry.register(task_type='classification', adapter_key='vit_classifier', factory=VitClassifierTaskAdapter)
    registry.register(task_type='classification', adapter_key='coca_classifier', factory=CocaClassifierTaskAdapter)
    registry.register(task_type='tracking', adapter_key='bytetrack', factory=ByteTrackTaskAdapter)
    registry.register(task_type='tracking', adapter_key='sort_tracker', factory=SortTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='botsort_tracker', factory=BotSortTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='ocsort_tracker', factory=OCSortTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='deepsort_tracker', factory=DeepSortTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='strongsort_tracker', factory=StrongSortTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='boosttrack_tracker', factory=BoostTrackTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='tracktrack_tracker', factory=TrackTrackTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='pdsort_tracker', factory=PdSortTrackingTaskAdapter)
    registry.register(task_type='tracking', adapter_key='tdlp_tracker', factory=TdlpTrackingTaskAdapter)
    registry.register(task_type='augmentation', adapter_key='augmentation_policy_smoke', factory=AugmentationPolicySmokeTaskAdapter)
    registry.register(
        task_type='augmentation',
        adapter_key='classification_crop_augmentation_smoke',
        factory=ClassificationCropAugmentationSmokeTaskAdapter,
    )
    registry.register(
        task_type='augmentation',
        adapter_key='detection_bbox_augmentation_smoke',
        factory=DetectionBBoxAugmentationSmokeTaskAdapter,
    )
    registry.register(
        task_type='augmentation',
        adapter_key='albumentations_detection_bbox',
        factory=AlbumentationsDetectionBBoxTaskAdapter,
    )
    registry.register(
        task_type='augmentation',
        adapter_key='albumentations_classification_crop',
        factory=AlbumentationsClassificationCropTaskAdapter,
    )
    registry.register(
        task_type='augmentation',
        adapter_key='albumentations_segmentation_mask',
        factory=AlbumentationsSegmentationMaskTaskAdapter,
    )

    return registry
