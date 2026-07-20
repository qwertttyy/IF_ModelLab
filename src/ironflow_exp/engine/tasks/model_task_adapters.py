from abc import ABC, abstractmethod
import csv
import contextlib
import io
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw
import yaml

from ironflow_exp.configs import ModelConfig, PredictConfig, TrainConfig
from ironflow_exp.domain import DetectionPredictionRecord, ObjectRecord
from ironflow_exp.engine.core import PredictionArtifactValidator, TASK_METRICS_FIELDNAMES
from ironflow_exp.engine.datasets import DatasetInputResolver
from ironflow_exp.engine.tasks.base import BaseTaskAdapter, TaskAdapterResult, TaskExecutionContext
from ironflow_exp.models import (
    MissingTimmDependencyError,
    MissingTorchvisionDependencyError,
    MissingUltralyticsDependencyError,
    TimmClassificationAdapter,
    TorchvisionClassificationAdapter,
    UltralyticsYoloClassificationAdapter,
    UltralyticsYoloDetectionAdapter,
)
from ironflow_exp.services.metrics_service import MetricsService


REAL_MODEL_TASK_ADAPTER_KEYS = frozenset({
    ('detection', 'd_fine_detection'),
    ('detection', 'grounding_dino_open_vocab_detection'),
    ('detection', 'mmdet_detection'),
    ('detection', 'rf_detr_detection'),
    ('detection', 'rt_detr_detection'),
    ('detection', 'rt_detr_v2_detection'),
    ('detection', 'lw_detr_detection'),
    ('detection', 'ultralytics_yolo'),
    ('detection', 'yolo_world_open_vocab_detection'),
    ('classification', 'timm_classifier'),
    ('classification', 'torchvision_classifier'),
    ('classification', 'ultralytics_yolo_classifier'),
    ('embedding', 'clip_embedding'),
    ('embedding', 'dinov2_embedding'),
    ('embedding', 'dinov3_embedding'),
    ('embedding', 'siglip_embedding'),
    ('segmentation', 'sam_promptable_segmentation'),
    ('segmentation', 'ultralytics_yolo_segmentation'),
})
_ORIGINAL_PIL_IMAGE_OPEN = Image.open
DEFAULT_DETECTION_PREVIEW_SAMPLE_COUNT = 80
MODEL_WEIGHT_POLICY_VERSION = '0.1'
RECOMMENDED_MODEL_WEIGHT_ROOTS = (
    'models/checkpoints',
    'runs/model_assets',
    'runs/checkpoints',
    'local_remote_simulator/models',
)


def _windows_extended_path(path: Path) -> Path:
    if os.name != 'nt':
        return path
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith('\\\\?\\'):
        return Path(text)
    if text.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + text.lstrip('\\'))
    return Path('\\\\?\\' + text)


DETECTION_SKELETON_CONTRACTS: dict[str, dict[str, object]] = {
    'd_fine_detection': {
        'default_model_id': 'd_fine',
        'family': 'transformer_detection',
        'input_contract': 'coco_detection',
        'dependency_profile': 'coco_detection',
        'required_artifacts': ('manifest.json', 'coco/annotations/instances_train.json'),
    },
    'rf_detr_detection': {
        'default_model_id': 'rf_detr',
        'family': 'transformer_detection',
        'input_contract': 'coco_detection',
        'dependency_profile': 'coco_detection',
        'required_artifacts': ('manifest.json', 'coco/annotations/instances_train.json'),
    },
    'rt_detr_detection': {
        'default_model_id': 'rt_detr',
        'family': 'transformer_detection',
        'input_contract': 'coco_detection',
        'dependency_profile': 'coco_detection',
        'required_artifacts': ('manifest.json', 'coco/annotations/instances_train.json'),
    },
    'rt_detr_v2_detection': {
        'default_model_id': 'rt_detr_v2',
        'family': 'transformer_detection',
        'input_contract': 'coco_detection',
        'dependency_profile': 'coco_detection',
        'required_artifacts': ('manifest.json', 'coco/annotations/instances_train.json'),
    },
    'lw_detr_detection': {
        'default_model_id': 'lw_detr',
        'family': 'transformer_detection',
        'input_contract': 'coco_detection',
        'dependency_profile': 'coco_detection',
        'required_artifacts': ('manifest.json', 'coco/annotations/instances_train.json'),
    },
    'mmdet_detection': {
        'default_model_id': 'rtm_det',
        'family': 'mmdetection',
        'input_contract': 'coco_detection',
        'dependency_profile': 'mmdetection',
        'required_artifacts': ('manifest.json', 'coco/annotations/instances_train.json'),
    },
    'yolo_world_open_vocab_detection': {
        'default_model_id': 'yolo_world',
        'family': 'open_vocab_detection',
        'input_contract': 'image_manifest_with_prompt_set',
        'dependency_profile': 'open_vocab_detection',
        'required_artifacts': ('manifest.json', 'prompt_set.json', 'checkpoint'),
    },
    'grounding_dino_open_vocab_detection': {
        'default_model_id': 'grounding_dino_1_5',
        'family': 'open_vocab_detection',
        'input_contract': 'image_manifest_with_prompt_set',
        'dependency_profile': 'open_vocab_detection',
        'required_artifacts': ('manifest.json', 'prompt_set.json', 'checkpoint'),
    },
    'yolov10_detection': {
        'default_model_id': 'yolov10n',
        'family': 'ultralytics_yolo_detection',
        'input_contract': 'yolo_or_coco_detection',
        'dependency_profile': 'ultralytics_yolo',
        'required_artifacts': ('manifest.json', 'data.yaml or COCO annotations', 'checkpoint.pt'),
    },
    'owlv2_open_vocab_detection': {
        'default_model_id': 'owlv2',
        'family': 'open_vocab_detection',
        'input_contract': 'image_manifest_with_prompt_set',
        'dependency_profile': 'open_vocab_detection',
        'required_artifacts': ('manifest.json', 'prompt_set.json', 'checkpoint'),
    },
    'dino_x_open_world_detection': {
        'default_model_id': 'dino_x',
        'family': 'open_world_detection',
        'input_contract': 'image_manifest_with_prompt_set',
        'dependency_profile': 'open_world_detection',
        'required_artifacts': ('manifest.json', 'prompt_set.json optional', 'checkpoint'),
    },
    'sahi_slicing_detection': {
        'default_model_id': 'sahi_slicing',
        'family': 'sliced_inference_detection',
        'input_contract': 'image_manifest_with_base_detector',
        'dependency_profile': 'sahi_detection',
        'required_artifacts': ('manifest.json', 'base_detector checkpoint', 'slicing_config.yaml'),
    },
    'asahi_slicing_detection': {
        'default_model_id': 'asahi_slicing',
        'family': 'adaptive_sliced_inference_detection',
        'input_contract': 'image_manifest_with_base_detector',
        'dependency_profile': 'sahi_detection',
        'required_artifacts': ('manifest.json', 'base_detector checkpoint', 'slicing_config.yaml'),
    },
    'yoloe_open_vocab_detection': {
        'default_model_id': 'yoloe',
        'family': 'open_vocab_detection',
        'input_contract': 'image_manifest_with_prompt_set',
        'dependency_profile': 'open_vocab_detection',
        'required_artifacts': ('manifest.json', 'prompt_set.json', 'checkpoint.pt'),
    },
}

SEGMENTATION_SKELETON_CONTRACTS: dict[str, dict[str, object]] = {
    'ultralytics_yolo_segmentation': {
        'default_model_id': 'yolo11n_seg',
        'supported_model_ids': ('yolo11n_seg', 'yolov8n_seg', 'yolo26n_seg', 'yolo26n_sem'),
        'family': 'ultralytics_yolo_segmentation',
        'input_contract': 'image_manifest_with_mask_labels',
        'dependency_profile': 'yolo_segmentation',
        'required_artifacts': ('manifest.json', 'mask_path or polygon segmentation labels', 'checkpoint.pt'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'mask_labels_required',
    },
    'sam_promptable_segmentation': {
        'default_model_id': 'sam2',
        'supported_model_ids': (
            'sam',
            'sam2',
            'sam2_1',
            'sam3',
            'mobile_sam',
            'mobile_sam_v2',
            'hq_sam',
            'efficient_sam',
            'efficientvit_sam',
        ),
        'family': 'sam_promptable_segmentation',
        'input_contract': 'image_manifest_with_prompt_source',
        'dependency_profile': 'sam_segmentation',
        'required_artifacts': ('manifest.json', 'prompt_set.json or detection_predictions.json', 'checkpoint'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'prompt_source_required',
    },
    'fast_sam_segmentation': {
        'default_model_id': 'fast_sam',
        'supported_model_ids': ('fast_sam',),
        'family': 'fast_promptable_segmentation',
        'input_contract': 'image_manifest_with_prompt_source',
        'dependency_profile': 'sam_segmentation',
        'required_artifacts': ('manifest.json', 'prompt_set.json or detection_predictions.json', 'checkpoint'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'prompt_source_required',
    },
    'detection_guided_segmentation': {
        'default_model_id': 'rf_detr_seg',
        'supported_model_ids': ('rf_detr_seg', 'd_fine_seg'),
        'family': 'detection_guided_segmentation',
        'input_contract': 'image_manifest_with_detection_or_mask_labels',
        'dependency_profile': 'detection_guided_segmentation',
        'required_artifacts': ('manifest.json', 'detection_predictions.json or mask labels', 'checkpoint'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'detection_or_mask_source_required',
    },
    'grounded_sam_segmentation': {
        'default_model_id': 'grounded_sam',
        'supported_model_ids': ('grounded_sam', 'grounded_sam2'),
        'family': 'grounded_promptable_segmentation',
        'input_contract': 'image_manifest_with_prompt_set_or_detection_predictions',
        'dependency_profile': 'grounded_sam_segmentation',
        'required_artifacts': ('manifest.json', 'prompt_set.json or detection_predictions.json', 'checkpoint'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'prompt_or_detection_source_required',
    },
    'yoloe_segmentation': {
        'default_model_id': 'yoloe_seg',
        'supported_model_ids': ('yoloe_seg', 'yoloe_26_seg'),
        'family': 'open_vocab_yolo_segmentation',
        'input_contract': 'image_manifest_with_mask_labels_or_prompt_set',
        'dependency_profile': 'yolo_segmentation',
        'required_artifacts': ('manifest.json', 'mask labels or prompt_set.json', 'checkpoint.pt'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'mask_or_prompt_source_required',
    },
    'dino_x_segmentation': {
        'default_model_id': 'dino_x_seg',
        'supported_model_ids': ('dino_x_seg',),
        'family': 'open_world_segmentation',
        'input_contract': 'image_manifest_with_prompt_set_or_detection_predictions',
        'dependency_profile': 'open_world_segmentation',
        'required_artifacts': ('manifest.json', 'prompt_set.json or detection_predictions.json', 'checkpoint'),
        'output_contract': 'segmentation_predictions.json',
        'missing_data_state': 'prompt_or_detection_source_required',
    },
}

EMBEDDING_SKELETON_CONTRACTS: dict[str, dict[str, object]] = {
    'dinov2_embedding': {
        'default_model_id': 'dinov2_vits14',
        'supported_model_ids': ('dinov2_vits14',),
        'family': 'self_supervised_vision_transformer',
        'input_contract': 'image_manifest_or_classification_input_manifest',
        'dependency_profile': 'embedding_torch',
        'required_artifacts': ('manifest.json or classification_input_manifest.json', 'checkpoint'),
        'output_artifacts': ('embedding_predictions.json', 'embeddings.npy', 'embeddings_meta.csv'),
    },
    'dinov3_embedding': {
        'default_model_id': 'dinov3_vits16',
        'supported_model_ids': ('dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'),
        'family': 'self_supervised_vision_transformer',
        'input_contract': 'image_manifest_or_classification_input_manifest',
        'dependency_profile': 'embedding_torch',
        'required_artifacts': ('manifest.json or classification_input_manifest.json', 'checkpoint'),
        'output_artifacts': ('embedding_predictions.json', 'embeddings.npy', 'embeddings_meta.csv'),
    },
    'classifier_penultimate_embedding': {
        'default_model_id': 'classifier_penultimate_embedding',
        'supported_model_ids': ('classifier_penultimate_embedding', 'resnet50_embedding', 'efficientnet_b0_embedding'),
        'family': 'classifier_feature_embedding',
        'input_contract': 'image_manifest_or_classification_input_manifest',
        'dependency_profile': 'embedding_torch',
        'required_artifacts': ('manifest.json or classification_input_manifest.json', 'classifier checkpoint optional'),
        'output_artifacts': ('embedding_predictions.json', 'embeddings.npy', 'embeddings_meta.csv'),
    },
    'coca_embedding': {
        'default_model_id': 'coca_vit_embedding',
        'supported_model_ids': ('coca_vit_embedding',),
        'family': 'vision_language_embedding',
        'input_contract': 'image_manifest_or_classification_input_manifest_with_optional_text_prompt',
        'dependency_profile': 'openclip_embedding',
        'required_artifacts': ('manifest.json or classification_input_manifest.json', 'prompt_set.json optional', 'checkpoint'),
        'output_artifacts': ('embedding_predictions.json', 'embeddings.npy', 'embeddings_meta.csv'),
    },
    'clip_embedding': {
        'default_model_id': 'clip_vit_b_32',
        'supported_model_ids': ('clip_vit_b_32', 'openclip_vit_b_32'),
        'family': 'vision_language_embedding',
        'input_contract': 'image_manifest_or_classification_input_manifest_with_optional_text_prompt',
        'dependency_profile': 'clip_embedding',
        'dependency_profile_by_model_id': {
            'clip_vit_b_32': 'clip_embedding',
            'openclip_vit_b_32': 'openclip_embedding',
        },
        'required_artifacts': ('manifest.json or classification_input_manifest.json', 'prompt_set.json optional', 'checkpoint'),
        'output_artifacts': ('embedding_predictions.json', 'embeddings.npy', 'embeddings_meta.csv'),
    },
    'siglip_embedding': {
        'default_model_id': 'siglip2_base_patch16',
        'supported_model_ids': ('siglip2_base_patch16',),
        'family': 'vision_language_embedding',
        'input_contract': 'image_manifest_or_classification_input_manifest_with_optional_text_prompt',
        'dependency_profile': 'siglip_embedding',
        'required_artifacts': ('manifest.json or classification_input_manifest.json', 'prompt_set.json optional', 'checkpoint'),
        'output_artifacts': ('embedding_predictions.json', 'embeddings.npy', 'embeddings_meta.csv'),
    },
}

CLASSIFICATION_SKELETON_CONTRACTS: dict[str, dict[str, object]] = {
    'vit_classifier': {
        'default_model_id': 'vit_tiny',
        'supported_model_ids': ('vit_tiny', 'vit_base', 'deit_tiny', 'swin_tiny'),
        'family': 'vision_transformer_classification',
        'input_contract': 'classification_folder_or_manifest',
        'dependency_profile': 'timm_classifier',
        'required_artifacts': ('classification dataset root', 'checkpoint optional'),
        'output_contract': 'classification_predictions.json',
    },
    'coca_classifier': {
        'default_model_id': 'coca_vit',
        'supported_model_ids': ('coca_vit',),
        'family': 'vision_language_classification',
        'input_contract': 'classification_folder_or_manifest_with_optional_text_prompt',
        'dependency_profile': 'openclip_classifier',
        'required_artifacts': ('classification dataset root', 'prompt_set.json optional', 'checkpoint optional'),
        'output_contract': 'classification_predictions.json',
    },
}


class RealModelTaskAdapter(BaseTaskAdapter, ABC):
    adapter_key: str
    default_model_id: str

    def run(self, context: TaskExecutionContext) -> TaskAdapterResult:
        try:
            metadata = self._contract_metadata(context=context)
        except (ValueError, TypeError) as error:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} contract validation failed: {error}',
                failure_type='real_adapter_contract_invalid',
                metadata={'error': str(error)},
            )

        if self._execution_mode(context=context) == 'inference_smoke':
            return self._run_inference_smoke(context=context, metadata=metadata)
        if self._execution_mode(context=context) == 'train':
            return self._run_train(context=context, metadata=metadata)

        return self._failed_result(
            context=context,
            message=(
                f'{self.adapter_key} is registered as a real model task adapter, '
                'but engine task execution is not implemented yet'
            ),
            failure_type='real_adapter_not_ready',
            metadata=metadata,
        )

    @abstractmethod
    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        """Return validated adapter metadata without loading optional ML dependencies."""

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        return self._failed_result(
            context=context,
            message=f'{self.adapter_key} inference smoke is not implemented yet',
            failure_type='real_adapter_not_ready',
            metadata={**metadata, 'execution_mode': 'inference_smoke'},
        )

    def _run_train(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        return self._failed_result(
            context=context,
            message=f'{self.adapter_key} train mode is not implemented yet',
            failure_type='real_adapter_not_ready',
            metadata={**metadata, 'execution_mode': 'train'},
        )

    def _execution_mode(self, context: TaskExecutionContext) -> str:
        return str(context.record.params.get('execution_mode', '')).strip().lower()

    def _model_config(self, context: TaskExecutionContext) -> ModelConfig:
        params = context.record.params
        checkpoint = self._resolved_checkpoint(context=context)

        return ModelConfig(
            enabled=True,
            model_id=context.record.model_id or self.default_model_id,
            adapter=self.adapter_key,
            checkpoint=checkpoint,
            pretrained=bool(params.get('pretrained', True)),
            train=TrainConfig(
                enabled=bool(params.get('train_enabled', False)),
                epochs=self._optional_int(params.get('epochs')),
                image_size=self._optional_int(params.get('image_size')),
                batch_size=self._optional_int(params.get('batch_size')),
                learning_rate=self._optional_float(params.get('learning_rate')),
            ),
            predict=PredictConfig(
                enabled=bool(params.get('predict_enabled', True)),
                confidence_threshold=self._optional_float(params.get('confidence_threshold')),
                iou_threshold=self._optional_float(params.get('iou_threshold')),
                top_k=self._optional_int(params.get('top_k')),
            ),
        )

    def _resolved_checkpoint(self, *, context: TaskExecutionContext) -> str | None:
        explicit_checkpoint = self._optional_str(context.record.params.get('checkpoint'))
        if explicit_checkpoint is not None:
            return explicit_checkpoint

        source_task_id = self._optional_str(context.record.params.get('checkpoint_from_task_id'))
        if source_task_id is None:
            return None

        artifact_name = self._optional_str(context.record.params.get('checkpoint_artifact_name')) or 'best_checkpoint'
        return str(context.dependency_artifact_path(source_task_id, artifact_name))

    def _weight_policy_metadata(self, *, context: TaskExecutionContext, model_config: ModelConfig) -> dict[str, object]:
        checkpoint = model_config.checkpoint
        allow_pretrained_download = bool(context.record.params.get('allow_pretrained_download', False))
        if checkpoint is not None:
            weight_source = 'explicit_checkpoint'
            checkpoint_exists: bool | None = Path(checkpoint).expanduser().exists()
        elif model_config.pretrained and allow_pretrained_download:
            weight_source = 'pretrained_download_allowed'
            checkpoint_exists = None
        elif model_config.pretrained:
            weight_source = 'pretrained_download_blocked'
            checkpoint_exists = None
        else:
            weight_source = 'architecture_only'
            checkpoint_exists = None

        return {
            'weight_policy_version': MODEL_WEIGHT_POLICY_VERSION,
            'weight_source': weight_source,
            'checkpoint_path': checkpoint,
            'checkpoint_exists': checkpoint_exists,
            'allow_pretrained_download': allow_pretrained_download,
            'recommended_weight_roots': list(RECOMMENDED_MODEL_WEIGHT_ROOTS),
        }

    def _weight_policy_metadata_from_params(self, *, context: TaskExecutionContext) -> dict[str, object]:
        return self._weight_policy_metadata(
            context=context,
            model_config=self._model_config(context=context),
        )

    def _prepared_checkpoint_path(self, *, model_config: ModelConfig) -> Path | None:
        if model_config.checkpoint is None:
            return None
        path = Path(model_config.checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'checkpoint file not found: {path}')

        return path

    def _checkpoint_safe_build_config(self, *, model_config: ModelConfig) -> ModelConfig:
        if model_config.checkpoint is None or not model_config.pretrained:
            return model_config

        return ModelConfig(
            enabled=model_config.enabled,
            model_id=model_config.model_id,
            adapter=model_config.adapter,
            checkpoint=model_config.checkpoint,
            pretrained=False,
            output_dim=model_config.output_dim,
            train=model_config.train,
            predict=model_config.predict,
        )

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
            metrics=[self._empty_metric_row()],
            metadata={
                'adapter': self.adapter_key,
                'model_id': context.record.model_id or self.default_model_id,
                'failure_type': failure_type,
                **metadata,
            },
        )

    def _empty_metric_row(self) -> dict[str, object]:
        return {
            field_name: 1 if field_name == 'epoch' else ''
            for field_name in TASK_METRICS_FIELDNAMES
        }

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()

        return text or None

    def _optional_int(self, value: Any) -> int | None:
        if value is None or value == '':
            return None

        return int(value)

    def _optional_float(self, value: Any) -> float | None:
        if value is None or value == '':
            return None

        return float(value)

    def _prediction_split(self, *, context: TaskExecutionContext) -> str | None:
        value = context.record.params.get('prediction_split')
        if value is None:
            value = context.record.params.get('eval_split')
        if value is None and str(context.record.params.get('execution_mode') or '').lower() == 'inference_smoke':
            value = 'test'
        text = self._optional_str(value)
        if text is None:
            return None
        normalized = text.lower()
        if normalized in {'all', '*'}:
            return None

        return normalized

    def _filtered_prediction_images(
        self,
        *,
        images: list[dict[str, object]],
        prediction_split: str | None,
        manifest_name: str,
    ) -> list[dict[str, object]]:
        if prediction_split is None:
            return images

        filtered = [
            image
            for image in images
            if str(image.get('split') or '').lower() == prediction_split
        ]
        if not filtered:
            raise ValueError(f'{manifest_name} has no images for prediction_split={prediction_split}')

        return filtered

    def _external_command_template(self, *, context: TaskExecutionContext, command_param: str) -> str | None:
        value = context.record.params.get(command_param) or context.record.params.get('external_command')
        if value is None:
            return None
        text = str(value).strip()

        return text or None

    def _format_external_request_command(
        self,
        *,
        template: str,
        context: TaskExecutionContext,
        execution_mode: str,
        request_path: Path,
        dataset_root: Path | None = None,
        input_manifest: Path | None = None,
        detection_predictions: Path | None = None,
        classification_input_manifest: Path | None = None,
    ) -> str:
        request_params = self._external_request_params(context=context)
        values = {
            'adapter_key': self.adapter_key,
            'model_id': context.record.model_id or self.default_model_id,
            'execution_mode': execution_mode,
            'result_dir': str(context.result_dir),
            'request_json': str(request_path),
            'dataset_root': str(dataset_root or ''),
            'input_manifest': str(input_manifest or ''),
            'manifest_json': str(input_manifest or ''),
            'detection_predictions': str(detection_predictions or ''),
            'classification_input_manifest': str(classification_input_manifest or ''),
            'checkpoint': str(request_params.get('checkpoint') or ''),
            'device': str(context.record.params.get('device') or ''),
            'epochs': str(context.record.params.get('epochs') or ''),
            'batch_size': str(context.record.params.get('batch_size') or ''),
            'image_size': str(context.record.params.get('image_size') or ''),
            'learning_rate': str(context.record.params.get('learning_rate') or ''),
        }

        return template.format(**values)

    def _external_request_params(self, *, context: TaskExecutionContext) -> dict[str, object]:
        params = dict(context.record.params)
        checkpoint = self._model_config(context=context).checkpoint
        if checkpoint is not None:
            params['checkpoint'] = checkpoint

        return params

    def _run_external_command(self, *, context: TaskExecutionContext, command: str) -> subprocess.CompletedProcess[str]:
        timeout = self._optional_int(context.record.params.get('external_timeout_seconds'))
        working_dir = self._external_working_dir(context=context)
        use_shell = bool(context.record.params.get('external_command_shell', os.name == 'nt'))
        command_args: str | list[str] = command if use_shell else shlex.split(command)
        process = subprocess.run(
            command_args,
            cwd=working_dir,
            shell=use_shell,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log_dir = context.result_dir / 'external'
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / 'stdout.log').write_text(process.stdout, encoding='utf-8')
        (log_dir / 'stderr.log').write_text(process.stderr, encoding='utf-8')

        return process

    def _external_working_dir(self, *, context: TaskExecutionContext) -> Path | None:
        raw_working_dir = context.record.params.get('external_working_dir')
        if raw_working_dir is None:
            return None
        working_dir = Path(str(raw_working_dir)).expanduser().resolve()
        if not working_dir.exists():
            raise FileNotFoundError(f'external_working_dir not found: {working_dir}')
        if not working_dir.is_dir():
            raise ValueError(f'external_working_dir must be a directory: {working_dir}')

        return working_dir

    def _external_command_metadata(
        self,
        *,
        context: TaskExecutionContext,
        command: str,
        completed: subprocess.CompletedProcess[str],
        request_path: Path,
    ) -> dict[str, object]:
        return {
            'external_command': command,
            'external_returncode': completed.returncode,
            'external_stdout_log': str(context.result_dir / 'external' / 'stdout.log'),
            'external_stderr_log': str(context.result_dir / 'external' / 'stderr.log'),
            'external_request_json': str(request_path),
            'external_stdout_tail': completed.stdout[-2000:],
            'external_stderr_tail': completed.stderr[-2000:],
        }

    def _external_metric_rows(self, *, context: TaskExecutionContext) -> list[dict[str, object]]:
        raw_path = context.record.params.get('external_metrics_file') or 'metrics.csv'
        metrics_path = self._result_relative_path(context=context, value=str(raw_path))
        if not metrics_path.exists():
            return [self._empty_metric_row()]
        with metrics_path.open(mode='r', encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return [self._empty_metric_row()]

        return [
            {
                'epoch': self._metric_value(row=row, keys=('epoch',), fallback=index + 1),
                'train_loss': self._metric_value(row=row, keys=('train_loss', 'loss')),
                'val_loss': self._metric_value(row=row, keys=('val_loss',)),
                'accuracy': self._metric_value(row=row, keys=('accuracy', 'top1')),
                'precision': self._metric_value(row=row, keys=('precision', 'metrics/precision(B)', 'metrics/precision')),
                'recall': self._metric_value(row=row, keys=('recall', 'metrics/recall(B)', 'metrics/recall')),
                'macro_precision': self._metric_value(row=row, keys=('macro_precision', 'macro_avg_precision')),
                'macro_recall': self._metric_value(row=row, keys=('macro_recall', 'macro_avg_recall')),
                'macro_f1': self._metric_value(row=row, keys=('macro_f1', 'f1', 'macro_avg_f1')),
                'class_recall': self._metric_value(row=row, keys=('class_recall', 'min_class_recall')),
                'class_ap50': self._metric_value(row=row, keys=('class_ap50', 'mean_class_ap50')),
                'object_accuracy': self._metric_value(row=row, keys=('object_accuracy', 'object_level_accuracy')),
                'map50': self._metric_value(row=row, keys=('map50',)),
                'map50_95': self._metric_value(row=row, keys=('map50_95',)),
                'num_predictions': self._metric_value(row=row, keys=('num_predictions', 'prediction_count')),
                'num_gt': self._metric_value(row=row, keys=('num_gt', 'ground_truth_count')),
                'mask_count': self._metric_value(row=row, keys=('mask_count',)),
                'mask_coverage': self._metric_value(row=row, keys=('mask_coverage',)),
                'embedding_count': self._metric_value(row=row, keys=('embedding_count',)),
                'embedding_dim': self._metric_value(row=row, keys=('embedding_dim',)),
                'retrieval_map': self._metric_value(row=row, keys=('retrieval_map',)),
                'neighbor_purity': self._metric_value(row=row, keys=('neighbor_purity',)),
                'review_hit_rate': self._metric_value(row=row, keys=('review_hit_rate',)),
                'label_error_rate': self._metric_value(row=row, keys=('label_error_rate',)),
                'latency_ms_per_image': self._metric_value(row=row, keys=('latency_ms_per_image', 'latency_ms', 'ms_per_image')),
                'p95_latency_ms': self._metric_value(row=row, keys=('p95_latency_ms', 'p95_latency')),
                'gpu_memory_mb': self._metric_value(row=row, keys=('gpu_memory_mb', 'gpu_mem_mb', 'gpu_memory')),
                'lr': self._metric_value(row=row, keys=('lr',)),
            }
            for index, row in enumerate(rows)
        ]

    def _metric_value(self, *, row: dict[str, str], keys: tuple[str, ...], fallback: object = '') -> object:
        normalized = {
            key.strip(): value.strip()
            for key, value in row.items()
            if key is not None and value is not None
        }
        for key in keys:
            value = normalized.get(key)
            if value not in {None, ''}:
                return value

        return fallback

    def _result_relative_path(self, *, context: TaskExecutionContext, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path

        return (context.result_dir / path).resolve()

    def _relative_to_result_dir(self, *, context: TaskExecutionContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.result_dir.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


class UltralyticsYoloTaskAdapter(RealModelTaskAdapter):
    adapter_key = 'ultralytics_yolo'
    default_model_id = 'yolo11n'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        model_config = self._model_config(context=context)
        adapter = UltralyticsYoloDetectionAdapter(model_config=model_config)

        return {
            'model_reference': adapter._model_reference(),
            'pretrained': model_config.pretrained,
            'checkpoint': model_config.checkpoint,
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_ready',
            **self._weight_policy_metadata(context=context, model_config=model_config),
        }

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        model_config = self._model_config(context=context)
        if (
            model_config.checkpoint is None
            and model_config.pretrained
            and not bool(context.record.params.get('allow_pretrained_download', False))
        ):
            return self._failed_result(
                context=context,
                message=(
                    'ultralytics_yolo inference smoke refuses implicit pretrained weight downloads; '
                    'set pretrained=false or provide an explicit prepared checkpoint path'
                ),
                failure_type='real_adapter_contract_invalid',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        try:
            self._prepared_checkpoint_path(model_config=model_config)
            yolo_class = self._load_yolo_runtime(context=context)
            manifest = self._load_detection_manifest(context=context)
            prediction_split = self._prediction_split(context=context)
            manifest = _DetectionManifest(
                dataset_id=manifest.dataset_id,
                images=self._filtered_prediction_images(
                    images=manifest.images,
                    prediction_split=prediction_split,
                    manifest_name='detection manifest',
                ),
                root_dir=manifest.root_dir,
            )
            records, fallback_count = self._predict_records(
                context=context,
                model_config=model_config,
                manifest=manifest,
                yolo_class=yolo_class,
            )
            metric_row = self._inference_metric_row(
                context=context,
                manifest=manifest,
                records=records,
                prediction_split=prediction_split,
                model_id=model_config.model_id or self.default_model_id,
            )
        except (OSError, ValueError, RuntimeError, MissingUltralyticsDependencyError) as error:
            return self._failed_result(
                context=context,
                message=f'ultralytics_yolo inference smoke failed: {error}',
                failure_type='real_adapter_execution_failed'
                if not isinstance(error, MissingUltralyticsDependencyError)
                else 'real_adapter_dependency_missing',
                metadata={**metadata, 'execution_mode': 'inference_smoke', 'error': str(error)},
            )

        prediction_path = context.result_dir / 'predictions' / 'detection_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'task': 'detection',
            'model_id': model_config.model_id or self.default_model_id,
            'dataset_id': manifest.dataset_id,
            'success': True,
            'records': records,
        }
        prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        preview_artifacts = self._write_detection_bbox_previews(
            context=context,
            manifest=manifest,
            records=records,
        )
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='detection')
        if not validation.is_valid:
            return self._failed_result(
                context=context,
                message='ultralytics_yolo prediction artifact validation failed: ' + '; '.join(validation.error_messages()),
                failure_type='real_adapter_output_invalid',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'ultralytics_yolo inference smoke completed: predictions={len(records)}',
            metrics=[metric_row],
            predictions=records,
            artifacts=[
                {
                    'name': 'detection_predictions',
                    'path': 'predictions/detection_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
                *preview_artifacts,
            ],
            metadata={
                **metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'inference_smoke',
                'dataset_id': manifest.dataset_id,
                'input_root': str(manifest.root_dir),
                'prediction_split': prediction_split or 'all',
                'prediction_count': len(records),
                'detection_bbox_preview_count': self._preview_artifact_count(artifacts=preview_artifacts),
                'synthetic_smoke_fallback_count': fallback_count,
                'checkpoint_loaded': model_config.checkpoint is not None,
            },
        )

    def _inference_metric_row(
        self,
        *,
        context: TaskExecutionContext,
        manifest: '_DetectionManifest',
        records: list[dict[str, object]],
        prediction_split: str | None,
        model_id: str,
    ) -> dict[str, object]:
        row = self._empty_metric_row()
        row['num_predictions'] = len(records)
        split = prediction_split or 'test'
        references, class_name_to_id, class_id_to_name = self._coco_references_for_split(
            dataset_root=manifest.root_dir,
            split=split,
            classes=self._manifest_classes(manifest=manifest),
            run_id=context.record.experiment_id,
        )
        if not references:
            return row

        predictions = self._prediction_records_for_metrics(
            records=records,
            split=split,
            class_name_to_id=class_name_to_id,
            class_id_to_name=class_id_to_name,
        )
        metrics = MetricsService().calculate_detection(
            run_id=context.record.experiment_id,
            model_id=model_id,
            predictions=predictions,
            references=references,
        ).metrics
        class_ap50_values: list[float] = []
        for metric in metrics:
            if metric.split != split:
                continue
            if metric.scope == 'overall' and metric.metric_name in row:
                row[metric.metric_name] = metric.metric_value if metric.metric_value is not None else ''
            elif metric.metric_name == 'class_ap50' and metric.metric_value is not None:
                class_ap50_values.append(float(metric.metric_value))
        if class_ap50_values:
            row['class_ap50'] = sum(class_ap50_values) / len(class_ap50_values)
        if row.get('num_gt') in {None, ''}:
            row['num_gt'] = float(len(references))

        return row

    def _coco_references_for_split(
        self,
        *,
        dataset_root: Path,
        split: str,
        classes: list[str],
        run_id: str,
    ) -> tuple[list[ObjectRecord], dict[str, int], dict[int, str]]:
        annotation_path = dataset_root / 'coco' / 'annotations' / f'instances_{split}.json'
        if not annotation_path.exists():
            return [], {}, {}
        try:
            coco = json.loads(annotation_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return [], {}, {}
        if not isinstance(coco, dict):
            return [], {}, {}

        class_id_to_name: dict[int, str] = {}
        categories = coco.get('categories')
        if isinstance(categories, list):
            for category in categories:
                if not isinstance(category, dict):
                    continue
                category_id = self._coerce_int(category.get('id'))
                name = category.get('name')
                if category_id is not None and isinstance(name, str) and name:
                    class_id_to_name[category_id] = name
        known_class_names = set(class_id_to_name.values())
        for index, class_name in enumerate(classes):
            if class_name and class_name not in known_class_names:
                class_id_to_name.setdefault(index, class_name)
                known_class_names.add(class_name)
        class_name_to_id = {name: class_id for class_id, name in class_id_to_name.items()}

        images_by_id: dict[object, str] = {}
        images = coco.get('images')
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    continue
                image_key = image.get('id')
                file_name = image.get('file_name')
                if image_key is None or not isinstance(file_name, str):
                    continue
                images_by_id[image_key] = Path(file_name).stem

        references: list[ObjectRecord] = []
        raw_annotations = coco.get('annotations')
        if not isinstance(raw_annotations, list):
            return references, class_name_to_id, class_id_to_name
        for index, annotation in enumerate(raw_annotations):
            if not isinstance(annotation, dict):
                continue
            image_id_value = annotation.get('image_id')
            sample_id = images_by_id.get(image_id_value) or str(image_id_value)
            class_id = self._coerce_int(annotation.get('category_id'))
            references.append(
                ObjectRecord(
                    object_id=str(annotation.get('id') or f'{run_id}_{split}_gt_{index:06d}'),
                    sample_id=sample_id,
                    image_id=sample_id,
                    split=split,
                    class_id=class_id,
                    class_name=class_id_to_name.get(class_id) if class_id is not None else None,
                    bbox_xyxy=self._coco_bbox_xyxy(annotation.get('bbox')),
                    bbox_yolo=None,
                    crop_path=None,
                    mask_path=None,
                    source='coco_detection_annotation',
                    annotation_id=str(annotation.get('id') or index),
                    is_gt=True,
                    metadata={'source_image_id': image_id_value},
                ),
            )

        return references, class_name_to_id, class_id_to_name

    def _manifest_classes(self, *, manifest: '_DetectionManifest') -> list[str]:
        manifest_path = manifest.root_dir / 'manifest.json'
        if not manifest_path.exists():
            return []
        try:
            data = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return []
        raw_classes = data.get('classes') if isinstance(data, dict) else None
        if not isinstance(raw_classes, list):
            return []

        return [str(class_id) for class_id in raw_classes if isinstance(class_id, str) and class_id]

    def _prediction_records_for_metrics(
        self,
        *,
        records: list[dict[str, object]],
        split: str,
        class_name_to_id: dict[str, int],
        class_id_to_name: dict[int, str],
    ) -> list[DetectionPredictionRecord]:
        predictions: list[DetectionPredictionRecord] = []
        for index, record in enumerate(records):
            class_id, class_name = self._prediction_class_for_metrics(
                record=record,
                class_name_to_id=class_name_to_id,
                class_id_to_name=class_id_to_name,
            )
            image_id = self._string_field(record=record, keys=('image_id', 'sample_id'), fallback=f'image_{index:06d}')
            sample_id = self._string_field(record=record, keys=('sample_id', 'image_id'), fallback=image_id)
            prediction_id = self._string_field(
                record=record,
                keys=('prediction_id',),
                fallback=f'{sample_id}_yolo_{index:06d}',
            )
            confidence = record.get('score')
            if confidence in {None, ''}:
                confidence = record.get('confidence')
            predictions.append(
                DetectionPredictionRecord(
                    prediction_id=prediction_id,
                    sample_id=sample_id,
                    image_id=image_id,
                    object_id=prediction_id,
                    split=split,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=self._optional_float(confidence),
                    bbox_xyxy=self._bbox_from_record(record=record),
                    bbox_yolo=None,
                    metadata=dict(record),
                ),
            )

        return predictions

    def _prediction_class_for_metrics(
        self,
        *,
        record: dict[str, object],
        class_name_to_id: dict[str, int],
        class_id_to_name: dict[int, str],
    ) -> tuple[int | None, str | None]:
        raw_class_id = record.get('class_id')
        raw_class_name = record.get('class_name') or record.get('label')
        class_id = self._coerce_int(raw_class_id)
        class_name = str(raw_class_name) if isinstance(raw_class_name, str) and raw_class_name else None
        if class_id is None and isinstance(raw_class_id, str) and raw_class_id:
            class_name = raw_class_id
            class_id = class_name_to_id.get(raw_class_id)
        if class_id is not None and class_name is None:
            class_name = class_id_to_name.get(class_id)

        return class_id, class_name

    def _coco_bbox_xyxy(self, value: object) -> list[float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            x, y, width, height = [float(item) for item in value]
        except (TypeError, ValueError):
            return None

        return [x, y, x + width, y + height]

    def _string_field(self, *, record: dict[str, object], keys: tuple[str, ...], fallback: str) -> str:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, int):
                return str(value)

        return fallback

    def _coerce_int(self, value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _run_train(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        model_config = self._model_config(context=context)
        if (
            model_config.checkpoint is None
            and model_config.pretrained
            and not bool(context.record.params.get('allow_pretrained_download', False))
        ):
            return self._failed_result(
                context=context,
                message=(
                    'ultralytics_yolo train mode refuses implicit pretrained weight downloads; '
                    'set pretrained=false, provide checkpoint, or set allow_pretrained_download=true'
                ),
                failure_type='real_adapter_contract_invalid',
                metadata={**metadata, 'execution_mode': 'train'},
        )

        try:
            self._prepared_checkpoint_path(model_config=model_config)
            resolution = DatasetInputResolver().resolve_yolo_detection(root=self._required_input_root(context=context))
            data_yaml = self._prepare_train_data_yaml(context=context, source_data_yaml=resolution.yolo_data_yaml_path)
            yolo_class = self._load_yolo_runtime(context=context)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = yolo_class(UltralyticsYoloDetectionAdapter(model_config=model_config)._model_reference())
                train_result = model.train(**self._train_kwargs(context=context, model_config=model_config, data_yaml=data_yaml))
        except (OSError, ValueError, RuntimeError, MissingUltralyticsDependencyError) as error:
            return self._failed_result(
                context=context,
                message=f'ultralytics_yolo train failed: {error}',
                failure_type='real_adapter_execution_failed'
                if not isinstance(error, MissingUltralyticsDependencyError)
                else 'real_adapter_dependency_missing',
                metadata={**metadata, 'execution_mode': 'train', 'error': str(error)},
            )

        save_dir = self._train_save_dir(train_result=train_result)
        best_checkpoint = self._checkpoint_path(save_dir=save_dir, name='best.pt')
        last_checkpoint = self._checkpoint_path(save_dir=save_dir, name='last.pt')
        metrics = self._train_metric_rows(save_dir=save_dir)
        artifacts = self._train_artifacts(
            context=context,
            save_dir=save_dir,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
        )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'ultralytics_yolo train completed: model_id={model_config.model_id or self.default_model_id}',
            metrics=metrics,
            artifacts=artifacts,
            metadata={
                **metadata,
                **resolution.metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'train',
                'runtime_detection_data_yaml': str(data_yaml),
                'train_save_dir': None if save_dir is None else str(save_dir),
                'best_checkpoint': None if best_checkpoint is None else str(best_checkpoint),
                'last_checkpoint': None if last_checkpoint is None else str(last_checkpoint),
                'class_count': len(resolution.classes),
                'checkpoint_loaded': model_config.checkpoint is not None,
            },
        )

    def _load_yolo_runtime(self, context: TaskExecutionContext) -> type[Any]:
        config_dir = context.result_dir / 'runtime' / 'ultralytics_config'
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from ultralytics import YOLO
        except ImportError as error:
            raise MissingUltralyticsDependencyError(
                f'ultralytics is required for ultralytics_yolo inference smoke: {error}',
            ) from error
        finally:
            Image.open = _ORIGINAL_PIL_IMAGE_OPEN

        return YOLO

    def _required_input_root(self, context: TaskExecutionContext) -> str:
        if context.record.input_variant_path is None:
            raise ValueError(f'{self.adapter_key} requires input_variant_path')

        return context.record.input_variant_path

    def _train_kwargs(
        self,
        *,
        context: TaskExecutionContext,
        model_config: ModelConfig,
        data_yaml: Path | None,
    ) -> dict[str, object]:
        if data_yaml is None:
            raise ValueError('YOLO train requires a resolved data.yaml path')
        params = context.record.params
        kwargs: dict[str, object] = {
            'data': str(data_yaml),
            'project': str(context.result_dir / 'checkpoints'),
            'name': str(model_config.model_id or self.default_model_id),
            'exist_ok': True,
            'verbose': bool(params.get('verbose', False)),
        }
        for param_name, yolo_name in [
            ('epochs', 'epochs'),
            ('image_size', 'imgsz'),
            ('batch_size', 'batch'),
            ('workers', 'workers'),
            ('patience', 'patience'),
            ('seed', 'seed'),
        ]:
            value = self._optional_int(params.get(param_name))
            if value is not None:
                kwargs[yolo_name] = value
        learning_rate = self._optional_float(params.get('learning_rate'))
        if learning_rate is not None:
            kwargs['lr0'] = learning_rate
        device = params.get('device')
        if device is not None and str(device).strip():
            kwargs['device'] = str(device)
        for param_name in [
            'hsv_h',
            'hsv_s',
            'hsv_v',
            'degrees',
            'translate',
            'scale',
            'shear',
            'perspective',
            'flipud',
            'fliplr',
            'mosaic',
            'mixup',
            'copy_paste',
            'erasing',
            'crop_fraction',
        ]:
            value = self._optional_float(params.get(param_name))
            if value is not None:
                kwargs[param_name] = value
        close_mosaic = self._optional_int(params.get('close_mosaic'))
        if close_mosaic is not None:
            kwargs['close_mosaic'] = close_mosaic

        return kwargs

    def _prepare_train_data_yaml(
        self,
        *,
        context: TaskExecutionContext,
        source_data_yaml: Path | None,
    ) -> Path:
        if source_data_yaml is None:
            raise ValueError('YOLO train requires a resolved data.yaml path')
        if not source_data_yaml.exists():
            raise FileNotFoundError(f'YOLO data yaml not found: {source_data_yaml}')

        loaded = yaml.safe_load(source_data_yaml.read_text(encoding='utf-8'))
        if not isinstance(loaded, dict):
            raise ValueError(f'YOLO data yaml must be a mapping: {source_data_yaml}')

        data = dict(loaded)
        data['path'] = source_data_yaml.parent.resolve().as_posix()

        runtime_data_yaml = context.result_dir / 'runtime' / 'yolo_data' / 'data.yaml'
        runtime_data_yaml.parent.mkdir(parents=True, exist_ok=True)
        with runtime_data_yaml.open(mode='w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)

        return runtime_data_yaml

    def _train_save_dir(self, *, train_result: object) -> Path | None:
        save_dir = getattr(train_result, 'save_dir', None)
        if save_dir is None:
            return None

        return Path(save_dir)

    def _checkpoint_path(self, *, save_dir: Path | None, name: str) -> Path | None:
        if save_dir is None:
            return None
        path = save_dir / 'weights' / name

        return path if path.exists() else None

    def _train_metric_rows(self, *, save_dir: Path | None) -> list[dict[str, object]]:
        if save_dir is None:
            return [self._empty_metric_row()]
        results_csv = save_dir / 'results.csv'
        if not results_csv.exists():
            return [self._empty_metric_row()]
        with results_csv.open(mode='r', encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return [self._empty_metric_row()]

        return [
            self._metric_row_from_yolo(row=row, fallback_epoch=index + 1)
            for index, row in enumerate(rows)
        ]

    def _metric_row_from_yolo(self, *, row: dict[str, str], fallback_epoch: int) -> dict[str, object]:
        return {
            'epoch': self._metric_value(row=row, keys=('epoch',), fallback=fallback_epoch),
            'train_loss': self._metric_value(row=row, keys=('train/box_loss', 'train_loss')),
            'val_loss': self._metric_value(row=row, keys=('val/box_loss', 'val_loss')),
            'accuracy': '',
            'precision': self._metric_value(row=row, keys=('metrics/precision(B)', 'metrics/precision', 'precision')),
            'recall': self._metric_value(row=row, keys=('metrics/recall(B)', 'metrics/recall', 'recall')),
            'map50': self._metric_value(row=row, keys=('metrics/mAP50(B)', 'metrics/mAP50', 'map50')),
            'map50_95': self._metric_value(row=row, keys=('metrics/mAP50-95(B)', 'metrics/mAP50-95', 'map50_95')),
            'lr': self._metric_value(row=row, keys=('lr/pg0', 'lr')),
        }

    def _metric_value(self, *, row: dict[str, str], keys: tuple[str, ...], fallback: object = '') -> object:
        normalized = {
            key.strip(): value.strip()
            for key, value in row.items()
            if key is not None and value is not None
        }
        for key in keys:
            value = normalized.get(key)
            if value not in {None, ''}:
                return value

        return fallback

    def _train_artifacts(
        self,
        *,
        context: TaskExecutionContext,
        save_dir: Path | None,
        best_checkpoint: Path | None,
        last_checkpoint: Path | None,
    ) -> list[dict[str, object]]:
        artifacts: list[dict[str, object]] = []
        if best_checkpoint is not None:
            artifacts.append(
                {
                    'name': 'best_checkpoint',
                    'path': self._relative_to_result_dir(context=context, path=best_checkpoint),
                    'kind': 'checkpoint',
                    'required': True,
                },
            )
        if last_checkpoint is not None:
            artifacts.append(
                {
                    'name': 'last_checkpoint',
                    'path': self._relative_to_result_dir(context=context, path=last_checkpoint),
                    'kind': 'checkpoint',
                    'required': False,
                },
            )
        if save_dir is not None and (save_dir / 'results.csv').exists():
            artifacts.append(
                {
                    'name': 'yolo_results_csv',
                    'path': self._relative_to_result_dir(context=context, path=save_dir / 'results.csv'),
                    'kind': 'metrics',
                    'required': False,
                },
            )

        return artifacts

    def _relative_to_result_dir(self, *, context: TaskExecutionContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.result_dir.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def _load_detection_manifest(self, context: TaskExecutionContext) -> '_DetectionManifest':
        if context.record.input_variant_path is None:
            raise ValueError('ultralytics_yolo inference smoke requires input_variant_path')

        resolution = DatasetInputResolver().resolve_manifest(
            root=context.record.input_variant_path,
            dataset_type='detection',
        )
        root_dir = resolution.dataset_root
        path = resolution.manifest_path or (root_dir / 'manifest.json')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')

        dataset_id = data.get('dataset_id')
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f'{path.name}.dataset_id must be a non-empty string')
        raw_images = data.get('images')
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f'{path.name}.images must be a non-empty list')
        images = [
            image
            for image in raw_images
            if isinstance(image, dict)
        ]
        if len(images) != len(raw_images):
            raise ValueError(f'{path.name}.images entries must be objects')

        return _DetectionManifest(dataset_id=dataset_id, images=images, root_dir=root_dir)

    def _predict_records(
        self,
        *,
        context: TaskExecutionContext,
        model_config: ModelConfig,
        manifest: '_DetectionManifest',
        yolo_class: type[Any],
    ) -> tuple[list[dict[str, object]], int]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = yolo_class(UltralyticsYoloDetectionAdapter(model_config=model_config)._model_reference())
        Image.open = _ORIGINAL_PIL_IMAGE_OPEN
        image_size = model_config.train.image_size or 32
        records: list[dict[str, object]] = []
        fallback_count = 0

        for image_index, image_record in enumerate(manifest.images):
            source_image_path = self._image_path(root_dir=manifest.root_dir, image_record=image_record)
            converted_image_path, image_width, image_height = self._prepare_yolo_input(
                context=context,
                source_image_path=source_image_path,
                image_index=image_index,
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                results = model.predict(
                    source=str(converted_image_path),
                    device='cpu',
                    imgsz=image_size,
                    verbose=False,
                )
            Image.open = _ORIGINAL_PIL_IMAGE_OPEN
            yolo_records = self._records_from_yolo_results(
                results=results,
                image_record=image_record,
                image_width=image_width,
                image_height=image_height,
                start_index=len(records),
            )
            if not yolo_records:
                if not bool(context.record.params.get('allow_synthetic_smoke_detection', False)):
                    raise ValueError('ultralytics_yolo produced no detections in inference smoke')
                yolo_records = [
                    self._synthetic_smoke_record(
                        image_record=image_record,
                        image_width=image_width,
                        image_height=image_height,
                        prediction_index=len(records),
                        fallback_class_id=str(context.record.params.get('fallback_class_id', 'object')),
                    ),
                ]
                fallback_count += 1
            records.extend(yolo_records)

        return records, fallback_count

    def _write_detection_bbox_previews(
        self,
        *,
        context: TaskExecutionContext,
        manifest: '_DetectionManifest',
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        records_by_image_id: dict[str, list[dict[str, object]]] = {}
        for record in records:
            image_id = record.get('image_id')
            if isinstance(image_id, str) and image_id:
                records_by_image_id.setdefault(image_id, []).append(record)

        image_records = [
            image_record
            for image_record in manifest.images
            if isinstance(image_record.get('image_id'), str)
            and image_record.get('image_id') in records_by_image_id
        ]
        if not image_records:
            return []

        sample_count = self._detection_preview_sample_count(context=context)
        if sample_count <= 0:
            return []
        seed = self._optional_int(context.record.params.get('preview_seed')) or 20260624
        rng = random.Random(seed)
        selected = list(image_records)
        rng.shuffle(selected)
        selected = selected[: min(sample_count, len(selected))]

        preview_dir = context.result_dir / 'previews' / 'detection_bbox'
        preview_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for image_record in selected:
            image_id = str(image_record['image_id'])
            try:
                image_path = self._image_path(root_dir=manifest.root_dir, image_record=image_record)
                self._write_one_detection_bbox_preview(
                    source_image_path=image_path,
                    records=records_by_image_id.get(image_id, []),
                    target_path=preview_dir / f'{self._safe_preview_stem(image_id)}.jpg',
                )
            except (OSError, ValueError):
                continue
            written += 1

        if written == 0:
            return []

        return [
            {
                'name': 'detection_bbox_previews',
                'path': 'previews/detection_bbox',
                'kind': 'preview',
                'required': False,
                'count': written,
            },
        ]

    def _write_one_detection_bbox_preview(
        self,
        *,
        source_image_path: Path,
        records: list[dict[str, object]],
        target_path: Path,
    ) -> None:
        with Image.open(source_image_path) as image:
            canvas = image.convert('RGB')
        draw = ImageDraw.Draw(canvas)
        line_width = max(2, min(canvas.size) // 160)
        for record in records:
            bbox = self._bbox_from_record(record=record)
            if bbox is None:
                continue
            label = self._preview_label(record=record)
            draw.rectangle(bbox, outline=(255, 48, 48), width=line_width)
            if label:
                text_bbox = draw.textbbox((bbox[0], bbox[1]), label)
                padding = 3
                background = [
                    text_bbox[0] - padding,
                    text_bbox[1] - padding,
                    text_bbox[2] + padding,
                    text_bbox[3] + padding,
                ]
                draw.rectangle(background, fill=(255, 48, 48))
                draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255))
        canvas.save(target_path, quality=90)

    def _bbox_from_record(self, *, record: dict[str, object]) -> list[float] | None:
        raw_bbox = record.get('bbox_xyxy')
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            return None
        try:
            return [float(value) for value in raw_bbox]
        except (TypeError, ValueError):
            return None

    def _preview_label(self, *, record: dict[str, object]) -> str:
        class_id = record.get('class_id')
        score = record.get('score')
        class_text = str(class_id) if class_id not in {None, ''} else 'object'
        try:
            return f'{class_text} {float(score):.2f}'
        except (TypeError, ValueError):
            return class_text

    def _detection_preview_sample_count(self, *, context: TaskExecutionContext) -> int:
        raw_value = (
            context.record.params.get('preview_sample_count')
            or context.record.params.get('detection_preview_sample_count')
        )
        if raw_value is None:
            return DEFAULT_DETECTION_PREVIEW_SAMPLE_COUNT
        return max(0, self._optional_int(raw_value) or 0)

    def _preview_artifact_count(self, *, artifacts: list[dict[str, object]]) -> int:
        for artifact in artifacts:
            count = artifact.get('count')
            if isinstance(count, int):
                return count
        return 0

    def _safe_preview_stem(self, value: str) -> str:
        safe = ''.join(character if character.isalnum() or character in {'-', '_', '.'} else '_' for character in value)
        return safe.strip('._') or 'image'

    def _prepare_yolo_input(
        self,
        *,
        context: TaskExecutionContext,
        source_image_path: Path,
        image_index: int,
    ) -> tuple[Path, int, int]:
        input_dir = context.result_dir / 'runtime' / 'yolo_inputs'
        input_dir.mkdir(parents=True, exist_ok=True)
        converted_path = input_dir / f'yolo_input_{image_index:04d}.jpg'
        with Image.open(source_image_path) as image:
            rgb = image.convert('RGB')
            rgb.save(converted_path)
            width, height = rgb.size

        return converted_path, width, height

    def _records_from_yolo_results(
        self,
        *,
        results: object,
        image_record: dict[str, object],
        image_width: int,
        image_height: int,
        start_index: int,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            names = getattr(result, 'names', {})
            class_ids = self._sequence(getattr(boxes, 'cls', []))
            confidences = self._sequence(getattr(boxes, 'conf', []))
            xyxy_rows = self._rows(getattr(boxes, 'xyxy', []))
            for box_index, class_id_value in enumerate(class_ids):
                class_id = int(class_id_value)
                bbox = self._clamped_bbox(
                    bbox=xyxy_rows[box_index] if box_index < len(xyxy_rows) else [0.0, 0.0, image_width, image_height],
                    image_width=image_width,
                    image_height=image_height,
                )
                score = confidences[box_index] if box_index < len(confidences) else 0.0
                records.append(
                    {
                        'image_id': self._required_str(image_record, 'image_id', f'image_{start_index + box_index:04d}'),
                        'sample_id': self._required_str(image_record, 'sample_id', f'sample_{start_index + box_index:04d}'),
                        'prediction_id': f'pred_yolo_{start_index + len(records) + 1:08d}',
                        'class_id': self._class_id(names=names, class_id=class_id),
                        'score': float(score),
                        'bbox_xyxy': bbox,
                        'image_width': image_width,
                        'image_height': image_height,
                    },
                )

        return records

    def _synthetic_smoke_record(
        self,
        *,
        image_record: dict[str, object],
        image_width: int,
        image_height: int,
        prediction_index: int,
        fallback_class_id: str,
    ) -> dict[str, object]:
        bbox = self._object_bbox(image_record=image_record)
        if bbox is None:
            bbox = [
                image_width * 0.25,
                image_height * 0.25,
                image_width * 0.75,
                image_height * 0.75,
            ]

        return {
            'image_id': self._required_str(image_record, 'image_id', f'image_{prediction_index:04d}'),
            'sample_id': self._required_str(image_record, 'sample_id', f'sample_{prediction_index:04d}'),
            'prediction_id': f'pred_yolo_smoke_{prediction_index + 1:08d}',
            'class_id': self._object_class_id(image_record=image_record) or fallback_class_id,
            'score': 0.01,
            'bbox_xyxy': self._clamped_bbox(bbox=bbox, image_width=image_width, image_height=image_height),
            'image_width': image_width,
            'image_height': image_height,
        }

    def _image_path(self, *, root_dir: Path, image_record: dict[str, object]) -> Path:
        raw_path = image_record.get('path')
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError('image record path must be a non-empty string')
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            image_path = relative_path.resolve()
            if not image_path.exists():
                raise FileNotFoundError(f'image file not found: {image_path}')
            return image_path
        if '..' in relative_path.parts:
            raise ValueError(f'image path must be a safe relative path: {raw_path}')
        image_path = (root_dir / relative_path).resolve()
        try:
            image_path.relative_to(root_dir.resolve())
        except ValueError as error:
            raise ValueError(f'image path escapes manifest root: {raw_path}') from error
        if not image_path.exists():
            raise FileNotFoundError(f'image file not found: {image_path}')

        return image_path

    def _sequence(self, value: Any) -> list[float]:
        converted = self._to_python(value=value)
        if converted is None:
            return []
        if isinstance(converted, list):
            return [float(item) for item in converted]

        return [float(converted)]

    def _rows(self, value: Any) -> list[list[float]]:
        converted = self._to_python(value=value)
        if converted is None or not isinstance(converted, list) or not converted:
            return []
        if all(isinstance(item, int | float) for item in converted):
            return [[float(item) for item in converted]]

        return [
            [float(item) for item in row]
            for row in converted
            if isinstance(row, list)
        ]

    def _to_python(self, value: Any) -> Any:
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'tolist'):
            return value.tolist()

        return value

    def _class_id(self, *, names: object, class_id: int) -> str:
        if isinstance(names, dict):
            value = names.get(class_id) or names.get(str(class_id))
            return str(value) if value is not None else str(class_id)
        if isinstance(names, list) and class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    def _object_bbox(self, *, image_record: dict[str, object]) -> list[float] | None:
        objects = image_record.get('objects')
        if not isinstance(objects, list) or not objects:
            return None
        first_object = objects[0]
        if not isinstance(first_object, dict):
            return None
        bbox = first_object.get('bbox_xyxy')
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None

        return [float(value) for value in bbox]

    def _object_class_id(self, *, image_record: dict[str, object]) -> str | None:
        objects = image_record.get('objects')
        if not isinstance(objects, list) or not objects:
            return None
        first_object = objects[0]
        if not isinstance(first_object, dict):
            return None
        class_id = first_object.get('class_id')
        if not isinstance(class_id, str) or not class_id:
            return None

        return class_id

    def _clamped_bbox(self, *, bbox: list[float], image_width: int, image_height: int) -> list[float]:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))
        if x1 == x2:
            x2 = min(float(image_width), x1 + 1.0)
        if y1 == y2:
            y2 = min(float(image_height), y1 + 1.0)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        return [x1, y1, x2, y2]

    def _required_str(self, record: dict[str, object], key: str, fallback: str) -> str:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value

        return fallback


class PlannedDetectionTaskAdapter(RealModelTaskAdapter):
    adapter_key = 'planned_detection'
    default_model_id = 'planned_detection'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        contract = DETECTION_SKELETON_CONTRACTS.get(self.adapter_key)
        if contract is None:
            raise ValueError(f'unsupported planned detection adapter: {self.adapter_key}')
        model_id = context.record.model_id or str(contract['default_model_id'])
        expected_model_id = str(contract['default_model_id'])
        if model_id != expected_model_id:
            raise ValueError(f'{self.adapter_key} expects model_id={expected_model_id}, got {model_id}')

        metadata: dict[str, object] = {
            'family': str(contract['family']),
            'input_contract': str(contract['input_contract']),
            'dependency_profile': str(contract['dependency_profile']),
            'required_artifacts': list(contract['required_artifacts']),
            'output_contract': 'detection_predictions.json',
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'external_command_required'
            if contract['input_contract'] == 'coco_detection'
            else 'adapter_skeleton_not_implemented',
            **self._weight_policy_metadata_from_params(context=context),
        }
        if context.record.input_variant_path is not None and contract['input_contract'] == 'coco_detection':
            resolution = DatasetInputResolver().resolve_coco_detection(root=context.record.input_variant_path)
            metadata.update(resolution.metadata)
            metadata['class_count'] = len(resolution.classes)
            metadata['classes'] = list(resolution.classes)

        return metadata

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        return self._run_external_coco_detection(
            context=context,
            metadata=metadata,
            execution_mode='inference_smoke',
            command_param='external_inference_command',
        )

    def _run_train(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        return self._run_external_coco_detection(
            context=context,
            metadata=metadata,
            execution_mode='train',
            command_param='external_train_command',
        )

    def _run_external_coco_detection(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
        execution_mode: str,
        command_param: str,
    ) -> TaskAdapterResult:
        if metadata.get('input_contract') != 'coco_detection':
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} {execution_mode} is not implemented for {metadata.get("input_contract")}',
                failure_type='real_adapter_not_ready',
                metadata={**metadata, 'execution_mode': execution_mode},
            )
        command_template = self._external_command_template(context=context, command_param=command_param)
        if command_template is None:
            return self._failed_result(
                context=context,
                message=(
                    f'{self.adapter_key} {execution_mode} requires params.{command_param} '
                    'or params.external_command to run the model-specific trainer/inference script'
                ),
                failure_type='real_adapter_external_command_missing',
                metadata={
                    **metadata,
                    'execution_mode': execution_mode,
                    'external_command_param': command_param,
                    'expected_outputs': self._expected_external_outputs(execution_mode=execution_mode),
                },
            )

        try:
            resolution = DatasetInputResolver().resolve_coco_detection(root=self._required_input_root(context=context))
            request_path = self._write_external_request(
                context=context,
                metadata={**metadata, **resolution.metadata},
                execution_mode=execution_mode,
                resolution=resolution,
            )
            command = self._format_external_command(
                template=command_template,
                context=context,
                resolution=resolution,
                execution_mode=execution_mode,
                request_path=request_path,
            )
            completed = self._run_external_command(context=context, command=command)
            external_metadata = self._external_command_metadata(
                context=context,
                command=command,
                completed=completed,
                request_path=request_path,
            )
            if completed.returncode != 0:
                return self._failed_result(
                    context=context,
                    message=f'{self.adapter_key} {execution_mode} external command failed with exit code {completed.returncode}',
                    failure_type='real_adapter_external_command_failed',
                    metadata={
                        **metadata,
                        **resolution.metadata,
                        **external_metadata,
                        'execution_mode': execution_mode,
                    },
                )

            predictions, prediction_artifact = self._external_prediction_records(
                context=context,
                execution_mode=execution_mode,
            )
            preview_artifacts = self._write_external_detection_bbox_previews(
                context=context,
                request_path=request_path,
                records=predictions,
            )
            metrics = self._external_metric_rows(context=context)
            metrics = self._external_detection_metric_rows(
                context=context,
                request_path=request_path,
                existing_rows=metrics,
                prediction_records=predictions,
                execution_mode=execution_mode,
            )
            artifacts = self._external_artifacts(
                context=context,
                request_path=request_path,
                prediction_artifact=prediction_artifact,
            )
            artifacts.extend(preview_artifacts)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} {execution_mode} external execution failed: {error}',
                failure_type='real_adapter_execution_failed',
                metadata={**metadata, 'execution_mode': execution_mode, 'error': str(error)},
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'{self.adapter_key} {execution_mode} external command completed',
            metrics=metrics,
            predictions=predictions,
            artifacts=artifacts,
            metadata={
                **metadata,
                **resolution.metadata,
                **external_metadata,
                'adapter': self.adapter_key,
                'execution_mode': execution_mode,
                'prediction_count': len(predictions),
                'detection_bbox_preview_count': self._preview_artifact_count(artifacts=preview_artifacts),
                'class_count': len(resolution.classes),
            },
        )

    def _external_command_template(self, *, context: TaskExecutionContext, command_param: str) -> str | None:
        value = context.record.params.get(command_param) or context.record.params.get('external_command')
        if value is None:
            return None
        text = str(value).strip()

        return text or None

    def _format_external_command(
        self,
        *,
        template: str,
        context: TaskExecutionContext,
        resolution: object,
        execution_mode: str,
        request_path: Path,
    ) -> str:
        annotation_paths = getattr(resolution, 'coco_annotation_paths')
        image_root = self._coco_image_root_for_mode(
            context=context,
            resolution=resolution,
            execution_mode=execution_mode,
        )
        request_params = self._external_request_params(context=context)
        values = {
            'adapter_key': self.adapter_key,
            'model_id': context.record.model_id or self.default_model_id,
            'execution_mode': execution_mode,
            'result_dir': str(context.result_dir),
            'request_json': str(request_path),
            'dataset_root': str(getattr(resolution, 'dataset_root')),
            'coco_root': str(getattr(resolution, 'coco_root')),
            'image_root': str(image_root),
            'coco_train_json': str(annotation_paths.get('train', '')),
            'coco_val_json': str(annotation_paths.get('val', '')),
            'coco_test_json': str(annotation_paths.get('test', '')),
            'checkpoint': str(request_params.get('checkpoint') or ''),
            'device': str(context.record.params.get('device') or ''),
            'epochs': str(context.record.params.get('epochs') or ''),
            'batch_size': str(context.record.params.get('batch_size') or ''),
            'image_size': str(context.record.params.get('image_size') or ''),
            'learning_rate': str(context.record.params.get('learning_rate') or ''),
        }

        return template.format(**values)

    def _run_external_command(self, *, context: TaskExecutionContext, command: str) -> subprocess.CompletedProcess[str]:
        timeout = self._optional_int(context.record.params.get('external_timeout_seconds'))
        working_dir = self._external_working_dir(context=context)
        use_shell = bool(context.record.params.get('external_command_shell', os.name == 'nt'))
        command_args: str | list[str] = command if use_shell else shlex.split(command)
        process = subprocess.run(
            command_args,
            cwd=working_dir,
            shell=use_shell,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log_dir = context.result_dir / 'external'
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / 'stdout.log').write_text(process.stdout, encoding='utf-8')
        (log_dir / 'stderr.log').write_text(process.stderr, encoding='utf-8')

        return process

    def _external_working_dir(self, *, context: TaskExecutionContext) -> Path | None:
        raw_working_dir = context.record.params.get('external_working_dir')
        if raw_working_dir is None:
            return None
        working_dir = Path(str(raw_working_dir)).expanduser().resolve()
        if not working_dir.exists():
            raise FileNotFoundError(f'external_working_dir not found: {working_dir}')
        if not working_dir.is_dir():
            raise ValueError(f'external_working_dir must be a directory: {working_dir}')

        return working_dir

    def _write_external_request(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
        execution_mode: str,
        resolution: object,
    ) -> Path:
        request_path = context.result_dir / 'runtime' / 'external_detection_request.json'
        request_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_paths = getattr(resolution, 'coco_annotation_paths')
        image_root = self._coco_image_root_for_mode(
            context=context,
            resolution=resolution,
            execution_mode=execution_mode,
        )
        prediction_split = self._prediction_split(context=context) if execution_mode == 'inference_smoke' else None
        payload = {
            'schema_version': '0.1',
            'adapter': self.adapter_key,
            'model_id': context.record.model_id or self.default_model_id,
            'execution_mode': execution_mode,
            'result_dir': str(context.result_dir),
            'dataset_root': str(getattr(resolution, 'dataset_root')),
            'coco_root': str(getattr(resolution, 'coco_root')),
            'image_root': str(image_root),
            'prediction_split': prediction_split or 'all',
            'annotations': {
                split: str(path)
                for split, path in annotation_paths.items()
            },
            'classes': list(getattr(resolution, 'classes')),
            'params': self._external_request_params(context=context),
            'metadata': metadata,
        }
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

        return request_path

    def _coco_image_root_for_mode(
        self,
        *,
        context: TaskExecutionContext,
        resolution: object,
        execution_mode: str,
    ) -> Path:
        image_root = Path(getattr(resolution, 'coco_image_root'))
        if execution_mode != 'inference_smoke':
            return image_root
        prediction_split = self._prediction_split(context=context)
        if prediction_split is None:
            return image_root
        split_root = image_root / 'images' / prediction_split
        if split_root.exists():
            return split_root
        split_root = image_root / prediction_split
        if split_root.exists():
            return split_root

        raise FileNotFoundError(f'COCO inference image root not found for prediction_split={prediction_split}: {image_root}')

    def _external_command_metadata(
        self,
        *,
        context: TaskExecutionContext,
        command: str,
        completed: subprocess.CompletedProcess[str],
        request_path: Path,
    ) -> dict[str, object]:
        return {
            'external_command': command,
            'external_returncode': completed.returncode,
            'external_stdout_log': str(context.result_dir / 'external' / 'stdout.log'),
            'external_stderr_log': str(context.result_dir / 'external' / 'stderr.log'),
            'external_request_json': str(request_path),
            'external_stdout_tail': completed.stdout[-2000:],
            'external_stderr_tail': completed.stderr[-2000:],
        }

    def _write_external_detection_bbox_previews(
        self,
        *,
        context: TaskExecutionContext,
        request_path: Path,
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        image_lookup = self._external_preview_image_lookup(request_path=request_path)
        if not image_lookup:
            return []

        records_by_image_id: dict[str, list[dict[str, object]]] = {}
        for record in records:
            image_id = record.get('image_id')
            if isinstance(image_id, str) and image_id in image_lookup:
                records_by_image_id.setdefault(image_id, []).append(record)
        if not records_by_image_id:
            return []

        sample_count = self._detection_preview_sample_count(context=context)
        if sample_count <= 0:
            return []
        seed = self._optional_int(context.record.params.get('preview_seed')) or 20260624
        rng = random.Random(seed)
        selected = list(records_by_image_id)
        rng.shuffle(selected)
        selected = selected[: min(sample_count, len(selected))]

        preview_dir = context.result_dir / 'previews' / 'detection_bbox'
        preview_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for image_id in selected:
            try:
                self._write_one_detection_bbox_preview(
                    source_image_path=image_lookup[image_id],
                    records=records_by_image_id[image_id],
                    target_path=preview_dir / f'{self._safe_preview_stem(image_id)}.jpg',
                )
            except (OSError, ValueError):
                continue
            written += 1

        if written == 0:
            return []

        return [
            {
                'name': 'detection_bbox_previews',
                'path': 'previews/detection_bbox',
                'kind': 'preview',
                'required': False,
                'count': written,
            },
        ]

    def _external_preview_image_lookup(self, *, request_path: Path) -> dict[str, Path]:
        request = json.loads(request_path.read_text(encoding='utf-8'))
        if not isinstance(request, dict):
            return {}
        raw_image_root = request.get('image_root')
        if not isinstance(raw_image_root, str) or not raw_image_root:
            return {}
        image_root = Path(raw_image_root).expanduser()
        if not image_root.exists():
            return {}
        if image_root.is_file():
            return {image_root.stem: image_root}

        image_paths: dict[str, Path] = {}
        image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for image_path in image_root.rglob('*'):
            if image_path.is_file() and image_path.suffix.lower() in image_suffixes:
                image_paths.setdefault(image_path.stem, image_path)

        return image_paths

    def _write_one_detection_bbox_preview(
        self,
        *,
        source_image_path: Path,
        records: list[dict[str, object]],
        target_path: Path,
    ) -> None:
        with Image.open(source_image_path) as image:
            canvas = image.convert('RGB')
        draw = ImageDraw.Draw(canvas)
        line_width = max(2, min(canvas.size) // 160)
        for record in records:
            bbox = self._bbox_from_record(record=record)
            if bbox is None:
                continue
            label = self._preview_label(record=record)
            draw.rectangle(bbox, outline=(255, 48, 48), width=line_width)
            if label:
                text_bbox = draw.textbbox((bbox[0], bbox[1]), label)
                padding = 3
                background = [
                    text_bbox[0] - padding,
                    text_bbox[1] - padding,
                    text_bbox[2] + padding,
                    text_bbox[3] + padding,
                ]
                draw.rectangle(background, fill=(255, 48, 48))
                draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255))
        canvas.save(target_path, quality=90)

    def _bbox_from_record(self, *, record: dict[str, object]) -> list[float] | None:
        raw_bbox = record.get('bbox_xyxy')
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            return None
        try:
            return [float(value) for value in raw_bbox]
        except (TypeError, ValueError):
            return None

    def _preview_label(self, *, record: dict[str, object]) -> str:
        class_id = record.get('class_id')
        score = record.get('score')
        class_text = str(class_id) if class_id not in {None, ''} else 'object'
        try:
            return f'{class_text} {float(score):.2f}'
        except (TypeError, ValueError):
            return class_text

    def _detection_preview_sample_count(self, *, context: TaskExecutionContext) -> int:
        raw_value = (
            context.record.params.get('preview_sample_count')
            or context.record.params.get('detection_preview_sample_count')
        )
        if raw_value is None:
            return DEFAULT_DETECTION_PREVIEW_SAMPLE_COUNT
        return max(0, self._optional_int(raw_value) or 0)

    def _preview_artifact_count(self, *, artifacts: list[dict[str, object]]) -> int:
        for artifact in artifacts:
            count = artifact.get('count')
            if isinstance(count, int):
                return count
        return 0

    def _safe_preview_stem(self, value: str) -> str:
        safe = ''.join(character if character.isalnum() or character in {'-', '_', '.'} else '_' for character in value)
        return safe.strip('._') or 'image'

    def _external_prediction_records(
        self,
        *,
        context: TaskExecutionContext,
        execution_mode: str,
    ) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        prediction_path = self._external_prediction_path(context=context)
        if not prediction_path.exists():
            if execution_mode == 'inference_smoke':
                raise FileNotFoundError(
                    f'external inference must write detection predictions: {prediction_path}',
                )
            return [], None
        payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f'external prediction artifact root must be an object: {prediction_path}')
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='detection')
        if not validation.is_valid:
            raise ValueError('external detection prediction artifact validation failed: ' + '; '.join(validation.error_messages()))
        records = payload.get('records', [])
        if not isinstance(records, list):
            raise ValueError('external detection prediction artifact records must be a list')

        return records, {
            'name': 'detection_predictions',
            'path': self._relative_to_result_dir(context=context, path=prediction_path),
            'kind': 'prediction',
            'required': execution_mode == 'inference_smoke',
        }

    def _external_prediction_path(self, *, context: TaskExecutionContext) -> Path:
        raw_path = context.record.params.get('external_prediction_file') or 'predictions/detection_predictions.json'

        return self._result_relative_path(context=context, value=str(raw_path))

    def _external_metric_rows(self, *, context: TaskExecutionContext) -> list[dict[str, object]]:
        raw_path = context.record.params.get('external_metrics_file') or 'metrics.csv'
        metrics_path = self._result_relative_path(context=context, value=str(raw_path))
        if not metrics_path.exists():
            return [self._empty_metric_row()]
        with metrics_path.open(mode='r', encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return [self._empty_metric_row()]

        return [
            {
                'epoch': self._metric_value(row=row, keys=('epoch',), fallback=index + 1),
                'train_loss': self._metric_value(row=row, keys=('train_loss', 'train/box_loss', 'loss')),
                'val_loss': self._metric_value(row=row, keys=('val_loss', 'val/box_loss')),
                'accuracy': self._metric_value(row=row, keys=('accuracy',)),
                'precision': self._metric_value(row=row, keys=('precision', 'metrics/precision(B)', 'metrics/precision')),
                'recall': self._metric_value(row=row, keys=('recall', 'metrics/recall(B)', 'metrics/recall')),
                'macro_precision': self._metric_value(row=row, keys=('macro_precision', 'macro_avg_precision')),
                'macro_recall': self._metric_value(row=row, keys=('macro_recall', 'macro_avg_recall')),
                'macro_f1': self._metric_value(row=row, keys=('macro_f1', 'f1', 'macro_avg_f1')),
                'class_recall': self._metric_value(row=row, keys=('class_recall', 'min_class_recall')),
                'class_ap50': self._metric_value(row=row, keys=('class_ap50', 'mean_class_ap50')),
                'object_accuracy': self._metric_value(row=row, keys=('object_accuracy', 'object_level_accuracy')),
                'map50': self._metric_value(row=row, keys=('map50', 'metrics/mAP50(B)', 'metrics/mAP50')),
                'map50_95': self._metric_value(row=row, keys=('map50_95', 'metrics/mAP50-95(B)', 'metrics/mAP50-95')),
                'num_predictions': self._metric_value(row=row, keys=('num_predictions', 'prediction_count')),
                'num_gt': self._metric_value(row=row, keys=('num_gt', 'ground_truth_count')),
                'mask_count': self._metric_value(row=row, keys=('mask_count',)),
                'mask_coverage': self._metric_value(row=row, keys=('mask_coverage',)),
                'embedding_count': self._metric_value(row=row, keys=('embedding_count',)),
                'embedding_dim': self._metric_value(row=row, keys=('embedding_dim',)),
                'retrieval_map': self._metric_value(row=row, keys=('retrieval_map',)),
                'neighbor_purity': self._metric_value(row=row, keys=('neighbor_purity',)),
                'review_hit_rate': self._metric_value(row=row, keys=('review_hit_rate',)),
                'label_error_rate': self._metric_value(row=row, keys=('label_error_rate',)),
                'latency_ms_per_image': self._metric_value(row=row, keys=('latency_ms_per_image', 'latency_ms', 'ms_per_image')),
                'p95_latency_ms': self._metric_value(row=row, keys=('p95_latency_ms', 'p95_latency')),
                'gpu_memory_mb': self._metric_value(row=row, keys=('gpu_memory_mb', 'gpu_mem_mb', 'gpu_memory')),
                'lr': self._metric_value(row=row, keys=('lr', 'lr/pg0')),
            }
            for index, row in enumerate(rows)
        ]

    def _external_detection_metric_rows(
        self,
        *,
        context: TaskExecutionContext,
        request_path: Path,
        existing_rows: list[dict[str, object]],
        prediction_records: list[dict[str, object]],
        execution_mode: str,
    ) -> list[dict[str, object]]:
        if execution_mode != 'inference_smoke':
            return existing_rows
        fallback_row = self._external_detection_metric_fallback_row(
            context=context,
            request_path=request_path,
            prediction_records=prediction_records,
        )
        if fallback_row is None:
            return existing_rows
        if not existing_rows:
            return [fallback_row]
        merged_rows = [dict(row) for row in existing_rows]
        target_row = merged_rows[-1]
        for key, value in fallback_row.items():
            if target_row.get(key) in {None, ''}:
                target_row[key] = value

        return merged_rows

    def _external_detection_metric_fallback_row(
        self,
        *,
        context: TaskExecutionContext,
        request_path: Path,
        prediction_records: list[dict[str, object]],
    ) -> dict[str, object] | None:
        request = self._read_external_detection_request(request_path=request_path)
        if request is None:
            return None
        references, class_name_to_id, class_id_to_name = self._external_detection_references_from_request(
            request=request,
            run_id=context.record.experiment_id,
        )
        if not references:
            return None
        split = str(request.get('prediction_split') or context.record.params.get('prediction_split') or 'test')
        predictions = self._external_detection_predictions_for_metrics(
            records=prediction_records,
            split=split,
            class_name_to_id=class_name_to_id,
            class_id_to_name=class_id_to_name,
        )
        metrics = MetricsService().calculate_detection(
            run_id=context.record.experiment_id,
            model_id=context.record.model_id or self.default_model_id,
            predictions=predictions,
            references=references,
        ).metrics
        row = self._empty_metric_row()
        row['epoch'] = 1
        class_ap50_values: list[float] = []
        for metric in metrics:
            if metric.split != split:
                continue
            if metric.scope == 'overall' and metric.metric_name in row:
                row[metric.metric_name] = metric.metric_value if metric.metric_value is not None else ''
            elif metric.metric_name == 'class_ap50' and metric.metric_value is not None:
                class_ap50_values.append(float(metric.metric_value))
        if class_ap50_values:
            row['class_ap50'] = sum(class_ap50_values) / len(class_ap50_values)
        if row.get('num_predictions') in {None, ''}:
            row['num_predictions'] = float(len(predictions))
        if row.get('num_gt') in {None, ''}:
            row['num_gt'] = float(len(references))

        return row

    def _read_external_detection_request(self, *, request_path: Path) -> dict[str, object] | None:
        try:
            request = json.loads(request_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(request, dict):
            return None

        return request

    def _external_detection_references_from_request(
        self,
        *,
        request: dict[str, object],
        run_id: str,
    ) -> tuple[list[ObjectRecord], dict[str, int], dict[int, str]]:
        annotations = request.get('annotations')
        if not isinstance(annotations, dict):
            return [], {}, {}
        split = str(request.get('prediction_split') or 'test')
        raw_annotation_path = annotations.get(split) or annotations.get('test') or annotations.get('val')
        if not isinstance(raw_annotation_path, str) or not raw_annotation_path:
            return [], {}, {}
        annotation_path = Path(raw_annotation_path)
        if not annotation_path.exists():
            return [], {}, {}
        try:
            coco = json.loads(annotation_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return [], {}, {}
        if not isinstance(coco, dict):
            return [], {}, {}
        categories = coco.get('categories')
        class_id_to_name: dict[int, str] = {}
        if isinstance(categories, list):
            for category in categories:
                if not isinstance(category, dict):
                    continue
                category_id = self._coerce_int(category.get('id'))
                name = category.get('name')
                if category_id is not None and isinstance(name, str) and name:
                    class_id_to_name[category_id] = name
        classes = request.get('classes')
        if isinstance(classes, list):
            known_class_names = set(class_id_to_name.values())
            for index, class_name in enumerate(classes):
                if (
                    isinstance(class_name, str)
                    and class_name
                    and class_name not in known_class_names
                ):
                    class_id_to_name.setdefault(index, class_name)
                    known_class_names.add(class_name)
        class_name_to_id = {name: class_id for class_id, name in class_id_to_name.items()}
        images_by_id: dict[object, str] = {}
        images = coco.get('images')
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    continue
                image_key = image.get('id')
                file_name = image.get('file_name')
                if image_key is None or not isinstance(file_name, str):
                    continue
                images_by_id[image_key] = Path(file_name).stem
        references: list[ObjectRecord] = []
        raw_annotations = coco.get('annotations')
        if not isinstance(raw_annotations, list):
            return references, class_name_to_id, class_id_to_name
        for index, annotation in enumerate(raw_annotations):
            if not isinstance(annotation, dict):
                continue
            image_id_value = annotation.get('image_id')
            sample_id = images_by_id.get(image_id_value) or str(image_id_value)
            class_id = self._coerce_int(annotation.get('category_id'))
            class_name = class_id_to_name.get(class_id) if class_id is not None else None
            bbox_xyxy = self._coco_bbox_xyxy(annotation.get('bbox'))
            references.append(
                ObjectRecord(
                    object_id=str(annotation.get('id') or f'{run_id}_{split}_gt_{index:06d}'),
                    sample_id=sample_id,
                    image_id=sample_id,
                    split=split,
                    class_id=class_id,
                    class_name=class_name,
                    bbox_xyxy=bbox_xyxy,
                    bbox_yolo=None,
                    crop_path=None,
                    mask_path=None,
                    source='coco_detection_annotation',
                    annotation_id=str(annotation.get('id') or index),
                    is_gt=True,
                    metadata={'source_image_id': image_id_value},
                ),
            )

        return references, class_name_to_id, class_id_to_name

    def _external_detection_predictions_for_metrics(
        self,
        *,
        records: list[dict[str, object]],
        split: str,
        class_name_to_id: dict[str, int],
        class_id_to_name: dict[int, str],
    ) -> list[DetectionPredictionRecord]:
        predictions: list[DetectionPredictionRecord] = []
        for index, record in enumerate(records):
            class_id, class_name = self._external_detection_prediction_class(
                record=record,
                class_name_to_id=class_name_to_id,
                class_id_to_name=class_id_to_name,
            )
            image_id = self._string_field(record=record, keys=('image_id', 'sample_id'), fallback=f'image_{index:06d}')
            sample_id = self._string_field(record=record, keys=('sample_id', 'image_id'), fallback=image_id)
            prediction_id = self._string_field(
                record=record,
                keys=('prediction_id',),
                fallback=f'{sample_id}_external_{index:06d}',
            )
            bbox_xyxy = self._bbox_from_record(record=record)
            confidence = record.get('score')
            if confidence in {None, ''}:
                confidence = record.get('confidence')
            predictions.append(
                DetectionPredictionRecord(
                    prediction_id=prediction_id,
                    sample_id=sample_id,
                    image_id=image_id,
                    object_id=prediction_id,
                    split=split,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=self._optional_float(confidence),
                    bbox_xyxy=bbox_xyxy,
                    bbox_yolo=None,
                    metadata=dict(record),
                ),
            )

        return predictions

    def _external_detection_prediction_class(
        self,
        *,
        record: dict[str, object],
        class_name_to_id: dict[str, int],
        class_id_to_name: dict[int, str],
    ) -> tuple[int | None, str | None]:
        raw_class_id = record.get('class_id')
        raw_class_name = record.get('class_name') or record.get('label')
        class_id = self._coerce_int(raw_class_id)
        class_name = str(raw_class_name) if isinstance(raw_class_name, str) and raw_class_name else None
        if class_id is None and isinstance(raw_class_id, str) and raw_class_id:
            class_name = raw_class_id
            class_id = class_name_to_id.get(raw_class_id)
        if class_id is not None and class_name is None:
            class_name = class_id_to_name.get(class_id)

        return class_id, class_name

    def _coco_bbox_xyxy(self, value: object) -> list[float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            x, y, width, height = [float(item) for item in value]
        except (TypeError, ValueError):
            return None

        return [x, y, x + width, y + height]

    def _string_field(self, *, record: dict[str, object], keys: tuple[str, ...], fallback: str) -> str:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, int):
                return str(value)

        return fallback

    def _coerce_int(self, value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _metric_value(self, *, row: dict[str, str], keys: tuple[str, ...], fallback: object = '') -> object:
        normalized = {
            key.strip(): value.strip()
            for key, value in row.items()
            if key is not None and value is not None
        }
        for key in keys:
            value = normalized.get(key)
            if value not in {None, ''}:
                return value

        return fallback

    def _external_artifacts(
        self,
        *,
        context: TaskExecutionContext,
        request_path: Path,
        prediction_artifact: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        artifacts = [
            {
                'name': 'external_request',
                'path': self._relative_to_result_dir(context=context, path=request_path),
                'kind': 'metadata',
                'required': True,
            },
            {
                'name': 'external_stdout',
                'path': 'external/stdout.log',
                'kind': 'log',
                'required': False,
            },
            {
                'name': 'external_stderr',
                'path': 'external/stderr.log',
                'kind': 'log',
                'required': False,
            },
        ]
        if prediction_artifact is not None:
            artifacts.append(prediction_artifact)
        for name, default_path, required in (
            ('best_checkpoint', 'checkpoints/best.pt', True),
            ('last_checkpoint', 'checkpoints/last.pt', False),
        ):
            raw_path = context.record.params.get(f'external_{name}_file') or default_path
            path = self._result_relative_path(context=context, value=str(raw_path))
            if path.exists():
                artifacts.append(
                    {
                        'name': name,
                        'path': self._relative_to_result_dir(context=context, path=path),
                        'kind': 'checkpoint',
                        'required': required,
                    },
                )

        return artifacts

    def _result_relative_path(self, *, context: TaskExecutionContext, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path

        return (context.result_dir / path).resolve()

    def _relative_to_result_dir(self, *, context: TaskExecutionContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.result_dir.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def _required_input_root(self, *, context: TaskExecutionContext) -> str:
        if context.record.input_variant_path is None:
            raise ValueError(f'{self.adapter_key} requires input_variant_path')

        return context.record.input_variant_path

    def _expected_external_outputs(self, *, execution_mode: str) -> dict[str, object]:
        return {
            'request_json': 'runtime/external_detection_request.json',
            'metrics_csv': 'metrics.csv',
            'best_checkpoint': 'checkpoints/best.pt',
            'last_checkpoint': 'checkpoints/last.pt',
            'detection_predictions': 'predictions/detection_predictions.json'
            if execution_mode == 'inference_smoke'
            else 'optional for train mode',
        }


class DFineDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'd_fine_detection'
    default_model_id = 'd_fine'


class RfDetrDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'rf_detr_detection'
    default_model_id = 'rf_detr'


class RtDetrDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'rt_detr_detection'
    default_model_id = 'rt_detr'


class RtDetrV2DetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'rt_detr_v2_detection'
    default_model_id = 'rt_detr_v2'


class LwDetrDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'lw_detr_detection'
    default_model_id = 'lw_detr'


class MMDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'mmdet_detection'
    default_model_id = 'rtm_det'


class YoloWorldOpenVocabDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'yolo_world_open_vocab_detection'
    default_model_id = 'yolo_world'


class GroundingDinoOpenVocabDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'grounding_dino_open_vocab_detection'
    default_model_id = 'grounding_dino_1_5'


class Yolov10DetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'yolov10_detection'
    default_model_id = 'yolov10n'


class OwlV2OpenVocabDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'owlv2_open_vocab_detection'
    default_model_id = 'owlv2'


class DinoXOpenWorldDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'dino_x_open_world_detection'
    default_model_id = 'dino_x'


class SahiSlicingDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'sahi_slicing_detection'
    default_model_id = 'sahi_slicing'


class AsahiSlicingDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'asahi_slicing_detection'
    default_model_id = 'asahi_slicing'


class YoloEOpenVocabDetectionTaskAdapter(PlannedDetectionTaskAdapter):
    adapter_key = 'yoloe_open_vocab_detection'
    default_model_id = 'yoloe'


class PlannedSegmentationTaskAdapter(RealModelTaskAdapter):
    adapter_key = 'planned_segmentation'
    default_model_id = 'planned_segmentation'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        contract = SEGMENTATION_SKELETON_CONTRACTS.get(self.adapter_key)
        if contract is None:
            raise ValueError(f'unsupported planned segmentation adapter: {self.adapter_key}')
        model_id = context.record.model_id or str(contract['default_model_id'])
        supported_model_ids = tuple(str(value) for value in contract['supported_model_ids'])
        if model_id not in supported_model_ids:
            expected = ', '.join(supported_model_ids)
            raise ValueError(f'{self.adapter_key} expects one of model_id={expected}, got {model_id}')

        metadata: dict[str, object] = {
            'family': str(contract['family']),
            'input_contract': str(contract['input_contract']),
            'dependency_profile': str(contract['dependency_profile']),
            'required_artifacts': list(contract['required_artifacts']),
            'output_contract': str(contract['output_contract']),
            'supported_model_ids': list(supported_model_ids),
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_skeleton_not_implemented',
            'data_readiness': str(contract['missing_data_state']),
            **self._weight_policy_metadata_from_params(context=context),
        }
        if context.record.input_variant_path is not None:
            manifest = self._read_segmentation_manifest(root=Path(context.record.input_variant_path))
            metadata['dataset_id'] = manifest.dataset_id
            metadata['image_count'] = len(manifest.images)
            metadata['mask_record_count'] = self._mask_record_count(images=manifest.images)
            metadata['prompt_source_count'] = self._prompt_source_count(context=context, images=manifest.images)
            if self.adapter_key == 'ultralytics_yolo_segmentation' and int(metadata['mask_record_count']) > 0:
                metadata['data_readiness'] = 'mask_labels_available'
            if self.adapter_key == 'sam_promptable_segmentation' and int(metadata['prompt_source_count']) > 0:
                metadata['data_readiness'] = 'prompt_source_available'

        return metadata

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        command_template = self._external_command_template(
            context=context,
            command_param='external_inference_command',
        )
        if command_template is None:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} inference smoke requires params.external_inference_command',
                failure_type='real_adapter_not_ready',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        try:
            manifest = self._read_segmentation_manifest(root=self._segmentation_dataset_root(context=context))
            detection_predictions = self._segmentation_detection_predictions(context=context)
            request_path = self._write_external_segmentation_request(
                context=context,
                metadata=metadata,
                manifest=manifest,
                detection_predictions=detection_predictions,
            )
            command = self._format_external_request_command(
                template=command_template,
                context=context,
                execution_mode='inference_smoke',
                request_path=request_path,
                dataset_root=manifest.root_dir,
                input_manifest=manifest.root_dir / 'manifest.json',
                detection_predictions=detection_predictions,
            )
            completed = self._run_external_command(context=context, command=command)
            external_metadata = self._external_command_metadata(
                context=context,
                command=command,
                completed=completed,
                request_path=request_path,
            )
            if completed.returncode != 0:
                return self._failed_result(
                    context=context,
                    message=f'{self.adapter_key} external command failed with exit code {completed.returncode}',
                    failure_type='real_adapter_external_command_failed',
                    metadata={**metadata, **external_metadata, 'execution_mode': 'inference_smoke'},
                )

            predictions, prediction_artifact = self._external_segmentation_predictions(context=context)
            artifacts = self._external_foundation_artifacts(
                context=context,
                request_path=request_path,
                output_artifacts=[prediction_artifact],
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError) as error:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} external execution failed: {error}',
                failure_type='real_adapter_execution_failed',
                metadata={**metadata, 'execution_mode': 'inference_smoke', 'error': str(error)},
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'{self.adapter_key} external command completed',
            metrics=self._external_metric_rows(context=context),
            predictions=predictions,
            artifacts=artifacts,
            metadata={
                **metadata,
                **external_metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'inference_smoke',
                'prediction_count': len(predictions),
            },
        )

    def _segmentation_dataset_root(self, *, context: TaskExecutionContext) -> Path:
        if context.record.input_variant_path is not None:
            return Path(context.record.input_variant_path)
        manifest = context.first_dependency_artifact_path('detection_input_manifest')

        return manifest.parent

    def _segmentation_detection_predictions(self, *, context: TaskExecutionContext) -> Path | None:
        raw_path = context.record.params.get('detection_predictions_path')
        if raw_path is not None:
            return Path(str(raw_path)).expanduser()
        try:
            return context.first_dependency_artifact_path('detection_predictions')
        except KeyError:
            return None

    def _write_external_segmentation_request(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
        manifest: '_SegmentationManifest',
        detection_predictions: Path | None,
    ) -> Path:
        request_path = context.result_dir / 'runtime' / 'external_segmentation_request.json'
        request_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'adapter': self.adapter_key,
            'model_id': context.record.model_id or self.default_model_id,
            'execution_mode': 'inference_smoke',
            'result_dir': str(context.result_dir),
            'dataset_root': str(manifest.root_dir),
            'input_manifest': str(manifest.root_dir / 'manifest.json'),
            'detection_predictions': str(detection_predictions) if detection_predictions is not None else None,
            'params': self._external_request_params(context=context),
            'metadata': metadata,
        }
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

        return request_path

    def _external_segmentation_predictions(
        self,
        *,
        context: TaskExecutionContext,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        raw_path = context.record.params.get('external_prediction_file') or 'predictions/segmentation_predictions.json'
        prediction_path = self._result_relative_path(context=context, value=str(raw_path))
        if not prediction_path.exists():
            raise FileNotFoundError(f'external segmentation must write predictions: {prediction_path}')
        payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f'external segmentation prediction root must be an object: {prediction_path}')
        records = payload.get('records', [])
        if not isinstance(records, list):
            raise ValueError('external segmentation prediction records must be a list')

        return records, {
            'name': 'segmentation_predictions',
            'path': self._relative_to_result_dir(context=context, path=prediction_path),
            'kind': 'prediction',
            'required': True,
        }

    def _external_foundation_artifacts(
        self,
        *,
        context: TaskExecutionContext,
        request_path: Path,
        output_artifacts: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        artifacts = [
            {
                'name': 'external_request',
                'path': self._relative_to_result_dir(context=context, path=request_path),
                'kind': 'metadata',
                'required': True,
            },
            {
                'name': 'external_stdout',
                'path': 'external/stdout.log',
                'kind': 'log',
                'required': False,
            },
            {
                'name': 'external_stderr',
                'path': 'external/stderr.log',
                'kind': 'log',
                'required': False,
            },
        ]
        artifacts.extend(output_artifacts)

        return artifacts

    def _read_segmentation_manifest(self, *, root: Path) -> '_SegmentationManifest':
        manifest_path = root / 'manifest.json'
        data = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('manifest.json root must be an object')
        dataset_id = data.get('dataset_id')
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError('manifest.json.dataset_id must be a non-empty string')
        raw_images = data.get('images')
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError('manifest.json.images must be a non-empty list')
        images = [
            image
            for image in raw_images
            if isinstance(image, dict)
        ]
        if len(images) != len(raw_images):
            raise ValueError('manifest.json.images entries must be objects')

        return _SegmentationManifest(dataset_id=dataset_id, images=images, root_dir=root)

    def _mask_record_count(self, *, images: list[dict[str, object]]) -> int:
        count = 0
        for image in images:
            if self._has_mask_annotation(record=image):
                count += 1
                continue
            objects = image.get('objects')
            if isinstance(objects, list) and any(self._has_mask_annotation(record=object_record) for object_record in objects):
                count += 1

        return count

    def _has_mask_annotation(self, *, record: object) -> bool:
        if not isinstance(record, dict):
            return False

        return any(
            self._has_non_empty_value(record=record, key=key)
            for key in ('mask_path', 'source_mask_path', 'mask_polygon', 'segmentation')
        )

    def _prompt_source_count(self, *, context: TaskExecutionContext, images: list[dict[str, object]]) -> int:
        count = sum(
            1
            for image in images
            if self._has_non_empty_value(record=image, key='prompt_set_path')
            or self._has_non_empty_value(record=image, key='prompt')
            or self._has_non_empty_value(record=image, key='bbox_prompt')
            or self._has_non_empty_value(record=image, key='point_prompt')
        )
        if count > 0:
            return count
        if context.record.params.get('prompt_set_path') or context.record.params.get('detection_predictions_path'):
            return len(images)

        return 0

    def _has_non_empty_value(self, *, record: dict[str, object], key: str) -> bool:
        value = record.get(key)
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return bool(value)
        if isinstance(value, dict):
            return bool(value)

        return value is not None


class UltralyticsYoloSegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'ultralytics_yolo_segmentation'
    default_model_id = 'yolo11n_seg'


class SamPromptableSegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'sam_promptable_segmentation'
    default_model_id = 'sam2'


class FastSamSegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'fast_sam_segmentation'
    default_model_id = 'fast_sam'


class DetectionGuidedSegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'detection_guided_segmentation'
    default_model_id = 'rf_detr_seg'


class GroundedSamSegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'grounded_sam_segmentation'
    default_model_id = 'grounded_sam'


class YoloESegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'yoloe_segmentation'
    default_model_id = 'yoloe_seg'


class DinoXSegmentationTaskAdapter(PlannedSegmentationTaskAdapter):
    adapter_key = 'dino_x_segmentation'
    default_model_id = 'dino_x_seg'


class PlannedEmbeddingTaskAdapter(RealModelTaskAdapter):
    adapter_key = 'planned_embedding'
    default_model_id = 'planned_embedding'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        contract = EMBEDDING_SKELETON_CONTRACTS.get(self.adapter_key)
        if contract is None:
            raise ValueError(f'unsupported planned embedding adapter: {self.adapter_key}')
        model_id = context.record.model_id or str(contract['default_model_id'])
        supported_model_ids = tuple(str(value) for value in contract['supported_model_ids'])
        if model_id not in supported_model_ids:
            expected = ', '.join(supported_model_ids)
            raise ValueError(f'{self.adapter_key} expects one of model_id={expected}, got {model_id}')

        metadata: dict[str, object] = {
            'family': str(contract['family']),
            'input_contract': str(contract['input_contract']),
            'dependency_profile': self._dependency_profile(contract=contract, model_id=model_id),
            'required_artifacts': list(contract['required_artifacts']),
            'output_artifacts': list(contract['output_artifacts']),
            'embedding_predictions_schema': 'embedding_predictions.json@0.1',
            'embedding_vector_store': 'embeddings.npy',
            'embedding_metadata_table': 'embeddings_meta.csv',
            'supported_model_ids': list(supported_model_ids),
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_skeleton_not_implemented',
            **self._weight_policy_metadata_from_params(context=context),
        }
        source = self._embedding_input_source(context=context)
        if source is not None:
            metadata.update(
                {
                    'dataset_id': source.dataset_id,
                    'input_record_count': len(source.images),
                    'input_manifest_path': str(source.manifest_path),
                    'input_manifest_kind': source.manifest_kind,
                    'object_record_count': self._object_record_count(images=source.images),
                },
            )

        return metadata

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        command_template = self._external_command_template(
            context=context,
            command_param='external_inference_command',
        )
        if command_template is None:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} inference smoke requires params.external_inference_command',
                failure_type='real_adapter_not_ready',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        try:
            source = self._embedding_input_source(context=context)
            if source is None:
                raise ValueError(f'{self.adapter_key} requires input_variant_path or classification_input_manifest dependency')
            request_path = self._write_external_embedding_request(
                context=context,
                metadata=metadata,
                source=source,
            )
            command = self._format_external_request_command(
                template=command_template,
                context=context,
                execution_mode='inference_smoke',
                request_path=request_path,
                dataset_root=source.manifest_path.parent,
                input_manifest=source.manifest_path,
                classification_input_manifest=source.manifest_path
                if source.manifest_kind == 'classification_input_manifest'
                else None,
            )
            completed = self._run_external_command(context=context, command=command)
            external_metadata = self._external_command_metadata(
                context=context,
                command=command,
                completed=completed,
                request_path=request_path,
            )
            if completed.returncode != 0:
                return self._failed_result(
                    context=context,
                    message=f'{self.adapter_key} external command failed with exit code {completed.returncode}',
                    failure_type='real_adapter_external_command_failed',
                    metadata={**metadata, **external_metadata, 'execution_mode': 'inference_smoke'},
                )

            predictions, artifacts = self._external_embedding_outputs(
                context=context,
                request_path=request_path,
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError) as error:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} external execution failed: {error}',
                failure_type='real_adapter_execution_failed',
                metadata={**metadata, 'execution_mode': 'inference_smoke', 'error': str(error)},
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'{self.adapter_key} external command completed',
            metrics=self._external_metric_rows(context=context),
            predictions=predictions,
            artifacts=artifacts,
            metadata={
                **metadata,
                **external_metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'inference_smoke',
                'prediction_count': len(predictions),
            },
        )

    def _write_external_embedding_request(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
        source: '_EmbeddingInputSource',
    ) -> Path:
        request_path = context.result_dir / 'runtime' / 'external_embedding_request.json'
        request_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'adapter': self.adapter_key,
            'model_id': context.record.model_id or self.default_model_id,
            'execution_mode': 'inference_smoke',
            'result_dir': str(context.result_dir),
            'dataset_root': str(source.manifest_path.parent),
            'input_manifest': str(source.manifest_path),
            'input_manifest_kind': source.manifest_kind,
            'params': self._external_request_params(context=context),
            'metadata': metadata,
        }
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

        return request_path

    def _external_embedding_outputs(
        self,
        *,
        context: TaskExecutionContext,
        request_path: Path,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        raw_prediction_path = context.record.params.get('external_prediction_file') or 'predictions/embedding_predictions.json'
        prediction_path = self._result_relative_path(context=context, value=str(raw_prediction_path))
        if not prediction_path.exists():
            raise FileNotFoundError(f'external embedding must write predictions: {prediction_path}')
        payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f'external embedding prediction root must be an object: {prediction_path}')
        records = payload.get('records', [])
        if not isinstance(records, list):
            raise ValueError('external embedding prediction records must be a list')

        artifacts = [
            {
                'name': 'external_request',
                'path': self._relative_to_result_dir(context=context, path=request_path),
                'kind': 'metadata',
                'required': True,
            },
            {
                'name': 'external_stdout',
                'path': 'external/stdout.log',
                'kind': 'log',
                'required': False,
            },
            {
                'name': 'external_stderr',
                'path': 'external/stderr.log',
                'kind': 'log',
                'required': False,
            },
            {
                'name': 'embedding_predictions',
                'path': self._relative_to_result_dir(context=context, path=prediction_path),
                'kind': 'prediction',
                'required': True,
            },
        ]
        for name, default_path, kind in (
            ('embeddings', 'embeddings.npy', 'embedding_vector'),
            ('embeddings_meta', 'embeddings_meta.csv', 'metadata'),
        ):
            raw_path = context.record.params.get(f'external_{name}_file') or default_path
            path = self._result_relative_path(context=context, value=str(raw_path))
            if path.exists():
                artifacts.append(
                    {
                        'name': name,
                        'path': self._relative_to_result_dir(context=context, path=path),
                        'kind': kind,
                        'required': True,
                    },
                )

        return records, artifacts

    def _dependency_profile(self, *, contract: dict[str, object], model_id: str) -> str:
        by_model = contract.get('dependency_profile_by_model_id')
        if isinstance(by_model, dict):
            value = by_model.get(model_id)
            if isinstance(value, str) and value:
                return value

        return str(contract['dependency_profile'])

    def _embedding_input_source(self, *, context: TaskExecutionContext) -> '_EmbeddingInputSource | None':
        try:
            dependency_manifest = context.first_dependency_artifact_path('classification_input_manifest')
        except KeyError:
            dependency_manifest = None
        if dependency_manifest is not None:
            return self._read_embedding_manifest(path=dependency_manifest, manifest_kind='classification_input_manifest')

        if context.record.input_variant_path is None:
            return None
        manifest_path = Path(context.record.input_variant_path) / 'manifest.json'

        return self._read_embedding_manifest(path=manifest_path, manifest_kind='manifest')

    def _read_embedding_manifest(self, *, path: Path, manifest_kind: str) -> '_EmbeddingInputSource':
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')
        dataset_id = data.get('dataset_id')
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f'{path.name}.dataset_id must be a non-empty string')
        raw_images = data.get('images')
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f'{path.name}.images must be a non-empty list')
        images = [
            image
            for image in raw_images
            if isinstance(image, dict)
        ]
        if len(images) != len(raw_images):
            raise ValueError(f'{path.name}.images entries must be objects')

        return _EmbeddingInputSource(
            dataset_id=dataset_id,
            images=images,
            manifest_path=path,
            manifest_kind=manifest_kind,
        )

    def _object_record_count(self, *, images: list[dict[str, object]]) -> int:
        count = 0
        for image in images:
            if isinstance(image.get('object_id'), str) and image.get('object_id'):
                count += 1
            objects = image.get('objects')
            if isinstance(objects, list):
                count += len([item for item in objects if isinstance(item, dict)])

        return count


class Dinov2EmbeddingTaskAdapter(PlannedEmbeddingTaskAdapter):
    adapter_key = 'dinov2_embedding'
    default_model_id = 'dinov2_vits14'


class Dinov3EmbeddingTaskAdapter(PlannedEmbeddingTaskAdapter):
    adapter_key = 'dinov3_embedding'
    default_model_id = 'dinov3_vits16'


class ClipEmbeddingTaskAdapter(PlannedEmbeddingTaskAdapter):
    adapter_key = 'clip_embedding'
    default_model_id = 'clip_vit_b_32'


class SiglipEmbeddingTaskAdapter(PlannedEmbeddingTaskAdapter):
    adapter_key = 'siglip_embedding'
    default_model_id = 'siglip2_base_patch16'


class ClassifierPenultimateEmbeddingTaskAdapter(PlannedEmbeddingTaskAdapter):
    adapter_key = 'classifier_penultimate_embedding'
    default_model_id = 'classifier_penultimate_embedding'


class CocaEmbeddingTaskAdapter(PlannedEmbeddingTaskAdapter):
    adapter_key = 'coca_embedding'
    default_model_id = 'coca_vit_embedding'


class PlannedClassificationTaskAdapter(RealModelTaskAdapter):
    adapter_key = 'planned_classification'
    default_model_id = 'planned_classification'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        contract = CLASSIFICATION_SKELETON_CONTRACTS.get(self.adapter_key)
        if contract is None:
            raise ValueError(f'unsupported planned classification adapter: {self.adapter_key}')
        model_id = context.record.model_id or str(contract['default_model_id'])
        supported_model_ids = tuple(str(value) for value in contract['supported_model_ids'])
        if model_id not in supported_model_ids:
            expected = ', '.join(supported_model_ids)
            raise ValueError(f'{self.adapter_key} expects one of model_id={expected}, got {model_id}')

        return {
            'family': str(contract['family']),
            'input_contract': str(contract['input_contract']),
            'dependency_profile': str(contract['dependency_profile']),
            'required_artifacts': list(contract['required_artifacts']),
            'output_contract': str(contract['output_contract']),
            'supported_model_ids': list(supported_model_ids),
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_skeleton_not_implemented',
            **self._weight_policy_metadata_from_params(context=context),
        }


class VitClassifierTaskAdapter(PlannedClassificationTaskAdapter):
    adapter_key = 'vit_classifier'
    default_model_id = 'vit_tiny'


class CocaClassifierTaskAdapter(PlannedClassificationTaskAdapter):
    adapter_key = 'coca_classifier'
    default_model_id = 'coca_vit'


class TorchvisionClassifierTaskAdapter(RealModelTaskAdapter):
    adapter_key = 'torchvision_classifier'
    default_model_id = 'mobilenet_v3_small'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        model_config = self._model_config(context=context)
        adapter = TorchvisionClassificationAdapter(model_config=model_config)

        return {
            'builder_name': adapter._builder_name(),
            'weights_name': adapter._weights_name(),
            'pretrained': model_config.pretrained,
            'checkpoint': model_config.checkpoint,
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_ready',
            **self._weight_policy_metadata(context=context, model_config=model_config),
        }

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        model_config = self._model_config(context=context)
        if (
            model_config.checkpoint is None
            and model_config.pretrained
            and not bool(context.record.params.get('allow_pretrained_download', False))
        ):
            return self._failed_result(
                context=context,
                message=(
                    'torchvision_classifier inference smoke refuses implicit pretrained weight downloads; '
                    'set pretrained=false or provide an explicit prepared checkpoint path'
                ),
                failure_type='real_adapter_contract_invalid',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        try:
            runtime = self._load_runtime()
        except MissingTorchvisionDependencyError as error:
            return self._failed_result(
                context=context,
                message=str(error),
                failure_type='real_adapter_dependency_missing',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        try:
            manifest = self._load_classification_manifest(context=context)
            prediction_split = self._prediction_split(context=context)
            manifest = _ClassificationManifest(
                dataset_id=manifest.dataset_id,
                classes=manifest.classes,
                images=self._filtered_prediction_images(
                    images=manifest.images,
                    prediction_split=prediction_split,
                    manifest_name='classification manifest',
                ),
                root_dir=manifest.root_dir,
            )
            records = self._predict_records(
                context=context,
                model_config=model_config,
                manifest=manifest,
                runtime=runtime,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            return self._failed_result(
                context=context,
                message=f'torchvision_classifier inference smoke failed: {error}',
                failure_type='real_adapter_execution_failed',
                metadata={**metadata, 'execution_mode': 'inference_smoke', 'error': str(error)},
            )

        prediction_path = context.result_dir / 'predictions' / 'classification_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'task': 'classification',
            'model_id': model_config.model_id or self.default_model_id,
            'dataset_id': manifest.dataset_id,
            'success': True,
            'records': records,
        }
        prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='classification')
        if not validation.is_valid:
            return self._failed_result(
                context=context,
                message='torchvision_classifier prediction artifact validation failed: ' + '; '.join(validation.error_messages()),
                failure_type='real_adapter_output_invalid',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        metric_row = self._classification_prediction_metric_row(records=records, images=manifest.images)
        input_manifest_artifact = self._write_classification_input_manifest_artifact(
            context=context,
            dataset_id=manifest.dataset_id,
            classes=manifest.classes,
            images=manifest.images,
            root_dir=manifest.root_dir,
        )
        confusion_artifacts = self._write_classification_confusion_artifacts(
            context=context,
            records=records,
            images=manifest.images,
            classes=manifest.classes,
        )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'torchvision_classifier inference smoke completed: predictions={len(records)}',
            metrics=[
                {
                    'epoch': 1,
                    'train_loss': '',
                    'val_loss': '',
                    'accuracy': metric_row['accuracy'],
                    'precision': metric_row['precision'],
                    'recall': metric_row['recall'],
                    'macro_precision': metric_row['macro_precision'],
                    'macro_recall': metric_row['macro_recall'],
                    'macro_f1': metric_row['macro_f1'],
                    'class_recall': metric_row['class_recall'],
                    'map50': '',
                    'map50_95': '',
                    'num_predictions': metric_row['num_predictions'],
                    'num_gt': metric_row['num_gt'],
                    'label_error_rate': metric_row['label_error_rate'],
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
                input_manifest_artifact,
                *confusion_artifacts,
            ],
            metadata={
                **metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'inference_smoke',
                'dataset_id': manifest.dataset_id,
                'input_root': str(manifest.root_dir),
                'prediction_count': len(records),
                'checkpoint_loaded': model_config.checkpoint is not None,
            },
        )

    def _run_train(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        model_config = self._model_config(context=context)
        if (
            model_config.checkpoint is None
            and model_config.pretrained
            and not bool(context.record.params.get('allow_pretrained_download', False))
        ):
            return self._failed_result(
                context=context,
                message=(
                    f'{self.adapter_key} train mode refuses implicit pretrained weight downloads; '
                    'set pretrained=false, provide checkpoint, or set allow_pretrained_download=true'
                ),
                failure_type='real_adapter_contract_invalid',
                metadata={**metadata, 'execution_mode': 'train'},
            )

        try:
            runtime = self._load_runtime()
            resolution = DatasetInputResolver().resolve_classification(
                root=self._required_input_root(context=context),
                input_kind=context.record.input_variant_kind,
            )
            train_result = self._train_classifier(
                context=context,
                model_config=model_config,
                runtime=runtime,
                resolution=resolution,
            )
        except (OSError, ValueError, KeyError, RuntimeError, MissingTimmDependencyError, MissingTorchvisionDependencyError) as error:
            return self._failed_result(
                context=context,
                message=f'{self.adapter_key} train failed: {error}',
                failure_type='real_adapter_execution_failed'
                if not isinstance(error, (MissingTimmDependencyError, MissingTorchvisionDependencyError))
                else 'real_adapter_dependency_missing',
                metadata={**metadata, 'execution_mode': 'train', 'error': str(error)},
            )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'{self.adapter_key} train completed: model_id={model_config.model_id or self.default_model_id}',
            metrics=train_result['metrics'],
            predictions=train_result['predictions'],
            artifacts=train_result['artifacts'],
            metadata={
                **metadata,
                **train_result['resolution_metadata'],
                'adapter': self.adapter_key,
                'execution_mode': 'train',
                'class_count': len(train_result['classes']),
                'classes': train_result['classes'],
                'best_checkpoint': train_result['best_checkpoint'],
                'last_checkpoint': train_result['last_checkpoint'],
                'checkpoint_loaded': train_result['checkpoint_loaded'],
                'prediction_count': len(train_result['predictions']),
                'early_stopping_patience': train_result['early_stopping_patience'],
                'early_stopping_min_delta': train_result['early_stopping_min_delta'],
                'label_smoothing': train_result.get('label_smoothing', 0.0),
                'classification_loss': train_result.get('classification_loss', 'cross_entropy'),
                'class_weight_mode': train_result.get('class_weight_mode', 'none'),
                'class_weights': train_result.get('class_weights'),
                'early_stopping_metric': train_result['early_stopping_metric'],
                'best_metric_name': train_result['best_metric_name'],
                'best_metric_value': train_result['best_metric_value'],
                'early_stopped': train_result['early_stopped'],
                'stopped_epoch': train_result['stopped_epoch'],
                'best_epoch': train_result['best_epoch'],
            },
        )

    def _load_runtime(self) -> dict[str, object]:
        try:
            import torch
            from PIL import Image
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
        except ImportError as error:
            raise MissingTorchvisionDependencyError(
                'torch and torchvision are required for torchvision_classifier inference smoke',
            ) from error

        return {
            'torch': torch,
            'DataLoader': DataLoader,
            'datasets': datasets,
            'Image': Image,
            'transforms': transforms,
        }

    def _required_input_root(self, context: TaskExecutionContext) -> str:
        if context.record.input_variant_path is None:
            raise ValueError(f'{self.adapter_key} train mode requires input_variant_path')

        return context.record.input_variant_path

    def _train_classifier(
        self,
        *,
        context: TaskExecutionContext,
        model_config: ModelConfig,
        runtime: dict[str, object],
        resolution: object,
    ) -> dict[str, object]:
        torch = runtime['torch']
        datasets = runtime['datasets']
        dataloader_class = runtime['DataLoader']
        transforms = runtime['transforms']
        device = str(context.record.params.get('device') or 'cpu')
        cudnn_settings = self._classification_configure_cudnn(torch=torch, context=context, device=device)
        image_size = model_config.train.image_size or 224
        batch_size = model_config.train.batch_size or 4
        workers = self._optional_int(context.record.params.get('workers')) or 0
        epochs = model_config.train.epochs or 1
        learning_rate = model_config.train.learning_rate or 0.001
        early_stopping_patience = self._optional_int(
            context.record.params.get('early_stopping_patience')
            if context.record.params.get('early_stopping_patience') is not None
            else context.record.params.get('patience'),
        )
        early_stopping_min_delta = self._optional_float(context.record.params.get('early_stopping_min_delta')) or 0.0
        split_roots = getattr(resolution, 'split_roots')
        if 'train' not in split_roots:
            raise ValueError('classification train mode requires a train split')
        if 'val' not in split_roots:
            raise ValueError('classification train mode requires a val split')
        val_root = split_roots['val']
        normalize_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize_transform,
        ])
        crop_jitter_enabled = self._classification_bool_param(context.record.params.get('crop_jitter'))
        crop_jitter_scale_min = self._optional_float(context.record.params.get('crop_jitter_scale_min')) or 0.85
        crop_jitter_ratio_min = self._optional_float(context.record.params.get('crop_jitter_ratio_min')) or 0.9
        crop_jitter_ratio_max = self._optional_float(context.record.params.get('crop_jitter_ratio_max')) or 1.1
        if crop_jitter_enabled:
            if not 0.0 < crop_jitter_scale_min <= 1.0:
                raise ValueError('crop_jitter_scale_min must be in the range (0.0, 1.0]')
            if not 0.0 < crop_jitter_ratio_min <= crop_jitter_ratio_max:
                raise ValueError('crop_jitter_ratio_min/max must be positive and ordered')
            train_preprocess = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(crop_jitter_scale_min, 1.0),
                    ratio=(crop_jitter_ratio_min, crop_jitter_ratio_max),
                ),
                transforms.ToTensor(),
                normalize_transform,
            ])
        else:
            train_preprocess = preprocess
        train_dataset = datasets.ImageFolder(str(split_roots['train']), transform=train_preprocess)
        val_dataset = datasets.ImageFolder(str(val_root), transform=preprocess)
        if not train_dataset.classes:
            raise ValueError('classification train split has no classes')
        class_names = list(train_dataset.classes)
        model = self._build_model(model_config=model_config)
        self._replace_classifier_head(model=model, class_count=len(class_names), torch=torch)
        checkpoint_info = self._load_model_checkpoint(
            model=model,
            model_config=model_config,
            torch=torch,
            device=device,
        )
        model.to(device)
        train_loader = dataloader_class(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = dataloader_class(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        label_smoothing = self._optional_float(context.record.params.get('label_smoothing')) or 0.0
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError('classification label_smoothing must be in the range [0.0, 1.0)')
        loss_name = str(
            context.record.params.get('loss')
            or context.record.params.get('loss_function')
            or 'cross_entropy'
        ).strip().lower().replace('-', '_')
        class_weight_mode = str(
            context.record.params.get('class_weight_mode')
            or context.record.params.get('class_weights')
            or ''
        ).strip().lower().replace('-', '_')
        weighted_loss_names = {'class_weighted_ce', 'weighted_cross_entropy', 'weighted_ce'}
        focal_loss_names = {'focal_loss', 'focal_ce'}
        if loss_name in {'cross_entropy', 'ce'} and class_weight_mode in {'balanced', 'inverse_frequency', 'auto'}:
            loss_name = 'class_weighted_ce'
        if loss_name not in {'cross_entropy', 'ce', *weighted_loss_names, *focal_loss_names}:
            raise ValueError(f'unsupported classification loss: {loss_name}')

        criterion_weight = None
        class_weights_for_result = None
        if loss_name in weighted_loss_names:
            train_targets = [int(target) for target in getattr(train_dataset, 'targets', [])]
            if not train_targets:
                raise ValueError('class_weighted_ce requires train dataset targets')
            class_counts = [sum(1 for target in train_targets if target == index) for index in range(len(class_names))]
            if any(count <= 0 for count in class_counts):
                raise ValueError(f'class_weighted_ce requires every class to have train samples: {class_counts}')
            total_count = sum(class_counts)
            weights = [total_count / (len(class_counts) * count) for count in class_counts]
            criterion_weight = torch.tensor(weights, dtype=torch.float32, device=device)
            class_weights_for_result = {
                class_name: float(weights[index])
                for index, class_name in enumerate(class_names)
            }
            class_weight_mode = class_weight_mode or 'balanced'
        else:
            class_weight_mode = class_weight_mode or 'none'

        focal_gamma = self._optional_float(context.record.params.get('focal_gamma')) or 2.0
        if loss_name in focal_loss_names:
            criterion = self._classification_focal_loss(
                torch=torch,
                weight=criterion_weight,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight, label_smoothing=label_smoothing)
        optimizer_param_groups = self._classification_optimizer_param_groups(
            model=model,
            learning_rate=learning_rate,
            layerwise_lr_decay=self._optional_float(context.record.params.get('layerwise_lr_decay')),
        )
        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=learning_rate)
        optimizer_state_dict = checkpoint_info.get('optimizer_state_dict') if checkpoint_info is not None else None
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        scheduler_name = str(context.record.params.get('scheduler') or context.record.params.get('lr_scheduler') or '').strip().lower()
        min_learning_rate = self._optional_float(
            context.record.params.get('min_learning_rate')
            if context.record.params.get('min_learning_rate') is not None
            else context.record.params.get('eta_min'),
        )
        scheduler = self._classification_lr_scheduler(
            torch=torch,
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            epochs=epochs,
            min_learning_rate=min_learning_rate,
        )
        scheduler_state_dict = checkpoint_info.get('scheduler_state_dict') if checkpoint_info is not None else None
        if scheduler is not None and scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
        metrics: list[dict[str, object]] = []
        best_metric_name = self._classification_monitor_metric(context=context)
        best_metric_value = float('-inf')
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_epoch: int | None = None
        checkpoint_dir = context.result_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = checkpoint_dir / 'best.pt'
        last_checkpoint = checkpoint_dir / 'last.pt'

        for epoch in range(1, epochs + 1):
            epoch_learning_rate = self._optimizer_learning_rate(optimizer=optimizer, default=learning_rate)
            train_loss = self._classification_train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                torch=torch,
                device=device,
            )
            val_metrics = self._classification_eval(
                model=model,
                loader=val_loader,
                criterion=criterion,
                torch=torch,
                device=device,
            )
            val_metrics = self._normalized_classification_eval_metrics(val_metrics)
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss'],
                'accuracy': val_metrics['accuracy'],
                'macro_precision': val_metrics['macro_precision'],
                'macro_recall': val_metrics['macro_recall'],
                'macro_f1': val_metrics['macro_f1'],
                'class_recall': val_metrics['class_recall'],
                'num_predictions': val_metrics['num_predictions'],
                'num_gt': val_metrics['num_gt'],
                'map50': '',
                'map50_95': '',
                'lr': epoch_learning_rate,
            }
            metrics.append(row)
            if scheduler is not None:
                scheduler.step()
            checkpoint_payload = {
                'model_id': model_config.model_id or self.default_model_id,
                'classes': class_names,
                'epoch': epoch,
                'metrics': row,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric_name': best_metric_name,
                'label_smoothing': label_smoothing,
                'classification_loss': loss_name,
                'class_weight_mode': class_weight_mode,
                'class_weights': class_weights_for_result,
                'focal_gamma': focal_gamma if loss_name in focal_loss_names else None,
                'layerwise_lr_decay': self._optional_float(context.record.params.get('layerwise_lr_decay')),
                'crop_jitter': crop_jitter_enabled,
            }
            if scheduler is not None:
                checkpoint_payload['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_payload, last_checkpoint)
            current_best_metric = self._classification_metric_for_monitor(row=row, metric_name=best_metric_name)
            checkpoint_payload['best_metric_value'] = current_best_metric
            if current_best_metric > best_metric_value + early_stopping_min_delta:
                best_metric_value = current_best_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint_payload['best_metric_value'] = best_metric_value
                torch.save(checkpoint_payload, best_checkpoint)
            else:
                epochs_without_improvement += 1
                if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                    stopped_epoch = epoch
                    break

        if best_checkpoint.exists():
            self._load_checkpoint_path_into_model(
                model=model,
                path=best_checkpoint,
                torch=torch,
                device=device,
            )
        elif last_checkpoint.exists():
            shutil.copy2(last_checkpoint, best_checkpoint)
            if metrics and best_epoch == 0:
                best_epoch = int(metrics[-1].get('epoch') or 0)
            self._load_checkpoint_path_into_model(
                model=model,
                path=best_checkpoint,
                torch=torch,
                device=device,
            )
        prediction_split = self._prediction_split(context=context)
        if prediction_split is None:
            prediction_split = 'test' if 'test' in split_roots else 'val'
        prediction_root = split_roots.get(prediction_split)
        if prediction_root is None:
            prediction_split = 'test' if 'test' in split_roots else 'val'
            prediction_root = split_roots[prediction_split]
        prediction_dataset = datasets.ImageFolder(str(prediction_root), transform=preprocess)
        predictions = self._classification_prediction_records(
            context=context,
            model=model,
            dataset=prediction_dataset,
            classes=class_names,
            preprocess=preprocess,
            runtime=runtime,
            torch=torch,
            device=device,
        )
        prediction_path = context.result_dir / 'predictions' / 'classification_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_payload = {
            'schema_version': '0.1',
            'task': 'classification',
            'model_id': model_config.model_id or self.default_model_id,
            'dataset_id': getattr(resolution, 'dataset_root').name,
            'success': bool(predictions),
            'records': predictions,
        }
        prediction_path.write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        validation = PredictionArtifactValidator().validate_payload(prediction_payload, expected_task='classification')
        if not validation.is_valid:
            raise ValueError('classification prediction artifact validation failed: ' + '; '.join(validation.error_messages()))
        prediction_images = self._imagefolder_dataset_metric_images(
            dataset=prediction_dataset,
            classes=class_names,
            split=prediction_split,
        )
        input_manifest_artifact = self._write_classification_input_manifest_artifact(
            context=context,
            dataset_id=getattr(resolution, 'dataset_root').name,
            classes=class_names,
            images=prediction_images,
            root_dir=Path(getattr(prediction_dataset, 'root', getattr(resolution, 'dataset_root'))),
        )
        confusion_artifacts = self._write_classification_confusion_artifacts(
            context=context,
            records=predictions,
            images=prediction_images,
            classes=class_names,
        )

        return {
            'metrics': metrics,
            'predictions': predictions,
            'artifacts': [
                {
                    'name': 'best_checkpoint',
                    'path': 'checkpoints/best.pt',
                    'kind': 'checkpoint',
                    'required': True,
                },
                {
                    'name': 'last_checkpoint',
                    'path': 'checkpoints/last.pt',
                    'kind': 'checkpoint',
                    'required': False,
                },
                {
                    'name': 'classification_predictions',
                    'path': 'predictions/classification_predictions.json',
                    'kind': 'prediction',
                    'required': True,
                },
                input_manifest_artifact,
                *confusion_artifacts,
            ],
            'resolution_metadata': getattr(resolution, 'metadata'),
            'classes': class_names,
            'best_checkpoint': str(best_checkpoint),
            'last_checkpoint': str(last_checkpoint),
            'checkpoint_loaded': checkpoint_info is not None,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'label_smoothing': label_smoothing,
            'classification_loss': loss_name,
            'class_weight_mode': class_weight_mode,
            'class_weights': class_weights_for_result,
            'focal_gamma': focal_gamma if loss_name in focal_loss_names else None,
            'layerwise_lr_decay': self._optional_float(context.record.params.get('layerwise_lr_decay')),
            'crop_jitter': crop_jitter_enabled,
            'cudnn_settings': cudnn_settings,
            'early_stopping_metric': best_metric_name,
            'best_metric_name': best_metric_name,
            'best_metric_value': None if best_metric_value == float('-inf') else best_metric_value,
            'early_stopped': stopped_epoch is not None,
            'stopped_epoch': stopped_epoch,
            'best_epoch': best_epoch,
            'scheduler': scheduler_name or 'none',
            'min_learning_rate': min_learning_rate,
        }

    def _classification_monitor_metric(self, *, context: TaskExecutionContext) -> str:
        raw_metric = (
            context.record.params.get('best_metric')
            or context.record.params.get('early_stopping_metric')
            or context.record.params.get('monitor_metric')
            or context.record.params.get('primary_metric')
            or 'macro_f1'
        )
        metric_name = str(raw_metric).strip().lower().replace('-', '_')
        aliases = {
            'acc': 'accuracy',
            'top1': 'accuracy',
            'top_1': 'accuracy',
            'val_accuracy': 'accuracy',
            'f1': 'macro_f1',
            'macro_avg_f1': 'macro_f1',
            'val_macro_f1': 'macro_f1',
            'precision': 'macro_precision',
            'val_macro_precision': 'macro_precision',
            'recall': 'macro_recall',
            'val_macro_recall': 'macro_recall',
        }
        metric_name = aliases.get(metric_name, metric_name)
        allowed_metrics = {'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'class_recall'}
        if metric_name not in allowed_metrics:
            raise ValueError(
                'unsupported classification monitor metric: '
                f'{raw_metric!r}; expected one of {sorted(allowed_metrics)}',
            )

        return metric_name

    def _classification_metric_for_monitor(self, *, row: dict[str, object], metric_name: str) -> float:
        value = row.get(metric_name)
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)

        raise ValueError(f'classification monitor metric {metric_name!r} is missing or non-numeric')

    def _classification_lr_scheduler(
        self,
        *,
        torch: object,
        optimizer: object,
        scheduler_name: str,
        epochs: int,
        min_learning_rate: float | None,
    ) -> object | None:
        if scheduler_name in {'', 'none', 'constant'}:
            return None
        if scheduler_name not in {'cosine', 'cosine_annealing', 'cosineannealinglr'}:
            raise ValueError(f'unsupported classification scheduler: {scheduler_name}')

        lr_scheduler_module = getattr(getattr(torch, 'optim', None), 'lr_scheduler', None)
        scheduler_class = getattr(lr_scheduler_module, 'CosineAnnealingLR', None)
        if scheduler_class is None:
            raise ValueError('torch.optim.lr_scheduler.CosineAnnealingLR is required for scheduler=cosine')

        return scheduler_class(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=0.00001 if min_learning_rate is None else min_learning_rate,
        )

    def _optimizer_learning_rate(self, *, optimizer: object, default: float) -> float:
        param_groups = getattr(optimizer, 'param_groups', None)
        if isinstance(param_groups, list) and param_groups:
            value = param_groups[0].get('lr')
            if isinstance(value, (float, int)):
                return float(value)

        return float(default)

    def _replace_classifier_head(self, *, model: object, class_count: int, torch: object) -> None:
        linear = torch.nn.Linear
        classifier = getattr(model, 'classifier', None)
        if classifier is not None and hasattr(classifier, '__len__') and len(classifier) > 0:
            last_layer = classifier[-1]
            in_features = getattr(last_layer, 'in_features', None)
            if isinstance(in_features, int):
                classifier[-1] = linear(in_features, class_count)
                return
        fc = getattr(model, 'fc', None)
        in_features = getattr(fc, 'in_features', None)
        if isinstance(in_features, int):
            model.fc = linear(in_features, class_count)
            return
        head = getattr(model, 'head', None)
        in_features = getattr(head, 'in_features', None)
        if isinstance(in_features, int):
            model.head = linear(in_features, class_count)
            return

        raise ValueError('unsupported torchvision classifier head shape')

    def _normalized_classification_eval_metrics(self, metrics: object) -> dict[str, float]:
        if isinstance(metrics, dict):
            return {
                'val_loss': float(metrics.get('val_loss', 0.0) or 0.0),
                'accuracy': float(metrics.get('accuracy', 0.0) or 0.0),
                'macro_precision': float(metrics.get('macro_precision', 0.0) or 0.0),
                'macro_recall': float(metrics.get('macro_recall', 0.0) or 0.0),
                'macro_f1': float(metrics.get('macro_f1', 0.0) or 0.0),
                'class_recall': float(metrics.get('class_recall', 0.0) or 0.0),
                'num_predictions': float(metrics.get('num_predictions', 0.0) or 0.0),
                'num_gt': float(metrics.get('num_gt', 0.0) or 0.0),
            }
        if isinstance(metrics, (tuple, list)) and len(metrics) >= 2:
            return {
                'val_loss': float(metrics[0] or 0.0),
                'accuracy': float(metrics[1] or 0.0),
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'class_recall': 0.0,
                'num_predictions': 0.0,
                'num_gt': 0.0,
            }

        raise TypeError('classification eval metrics must be a dict or a (val_loss, accuracy) tuple')



    def _classification_configure_cudnn(
        self,
        *,
        torch: object,
        context: TaskExecutionContext,
        device: str,
    ) -> dict[str, object]:
        settings: dict[str, object] = {}
        if not str(device).startswith('cuda'):
            return settings
        cudnn = getattr(getattr(torch, 'backends', None), 'cudnn', None)
        if cudnn is None:
            return settings

        raw_enabled = context.record.params.get('cudnn_enabled')
        raw_benchmark = context.record.params.get('cudnn_benchmark')
        raw_deterministic = context.record.params.get('cudnn_deterministic')

        if raw_enabled is not None:
            cudnn.enabled = self._classification_bool_param(raw_enabled)
        if raw_benchmark is not None:
            cudnn.benchmark = self._classification_bool_param(raw_benchmark)
        if raw_deterministic is not None:
            cudnn.deterministic = self._classification_bool_param(raw_deterministic)

        settings['enabled'] = bool(getattr(cudnn, 'enabled', False))
        settings['benchmark'] = bool(getattr(cudnn, 'benchmark', False))
        settings['deterministic'] = bool(getattr(cudnn, 'deterministic', False))
        return settings

    def _classification_bool_param(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

    def _classification_focal_loss(
        self,
        *,
        torch: object,
        weight: object,
        gamma: float,
        label_smoothing: float,
    ) -> object:
        ce_loss = torch.nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction='none',
        )

        def criterion(outputs: object, labels: object) -> object:
            per_sample_loss = ce_loss(outputs, labels)
            pt = torch.exp(-per_sample_loss)
            return (((1.0 - pt) ** gamma) * per_sample_loss).mean()

        return criterion

    def _classification_optimizer_param_groups(
        self,
        *,
        model: object,
        learning_rate: float,
        layerwise_lr_decay: float | None,
    ) -> object:
        if layerwise_lr_decay is None or layerwise_lr_decay <= 0.0 or layerwise_lr_decay >= 1.0:
            return model.parameters()

        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if getattr(parameter, 'requires_grad', False)
        ]
        if not named_parameters:
            return model.parameters()

        max_depth = 8
        grouped: dict[int, list[object]] = {}
        for name, parameter in named_parameters:
            depth = self._classification_parameter_depth(name=name, max_depth=max_depth)
            grouped.setdefault(depth, []).append(parameter)

        return [
            {
                'params': parameters,
                'lr': learning_rate * (layerwise_lr_decay ** (max_depth - depth)),
            }
            for depth, parameters in sorted(grouped.items())
        ]

    def _classification_parameter_depth(self, *, name: str, max_depth: int) -> int:
        lowered = name.lower()
        if lowered.startswith(('head', 'classifier', 'fc')):
            return max_depth
        for pattern in (r'(?:layers|stages)\.(\d+)', r'(?:features)\.(\d+)', r'(?:blocks)\.(\d+)'):
            match = re.search(pattern, lowered)
            if match:
                return max(0, min(max_depth - 1, int(match.group(1)) + 2))
        if any(token in lowered for token in ('patch_embed', 'stem', 'downsample_layers.0')):
            return 0
        return max_depth // 2

    def _classification_train_epoch(
        self,
        *,
        model: object,
        loader: object,
        criterion: object,
        optimizer: object,
        torch: object,
        device: str,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_count = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(inputs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        return total_loss / max(total_count, 1)

    def _classification_eval(
        self,
        *,
        model: object,
        loader: object,
        criterion: object,
        torch: object,
        device: str,
    ) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_count = 0
        correct = 0
        all_labels: list[int] = []
        all_predictions: list[int] = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predicted = outputs.argmax(dim=1)
                batch_size = int(inputs.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size
                correct += int((predicted == labels).sum().item())
                all_labels.extend(int(value) for value in labels.detach().cpu().tolist())
                all_predictions.extend(int(value) for value in predicted.detach().cpu().tolist())

        precision, recall, f1, min_recall = self._classification_macro_metrics(
            labels=all_labels,
            predictions=all_predictions,
        )

        return {
            'val_loss': total_loss / max(total_count, 1),
            'accuracy': correct / max(total_count, 1),
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'class_recall': min_recall,
            'num_predictions': float(len(all_predictions)),
            'num_gt': float(len(all_labels)),
        }

    def _classification_macro_metrics(
        self,
        *,
        labels: list[object],
        predictions: list[object],
    ) -> tuple[float, float, float, float]:
        class_ids = sorted(set(labels) | set(predictions))
        if not class_ids:
            return 0.0, 0.0, 0.0, 0.0

        precisions: list[float] = []
        recalls: list[float] = []
        f1_scores: list[float] = []
        for class_id in class_ids:
            true_positive = sum(
                1
                for label, prediction in zip(labels, predictions)
                if label == class_id and prediction == class_id
            )
            false_positive = sum(
                1
                for label, prediction in zip(labels, predictions)
                if label != class_id and prediction == class_id
            )
            false_negative = sum(
                1
                for label, prediction in zip(labels, predictions)
                if label == class_id and prediction != class_id
            )
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return (
            sum(precisions) / len(precisions),
            sum(recalls) / len(recalls),
            sum(f1_scores) / len(f1_scores),
            min(recalls),
        )

    def _classification_prediction_records(
        self,
        *,
        context: TaskExecutionContext,
        model: object,
        dataset: object,
        classes: list[str],
        preprocess: object,
        runtime: dict[str, object],
        torch: object,
        device: str,
    ) -> list[dict[str, object]]:
        image_module = runtime['Image']
        model.eval()
        records: list[dict[str, object]] = []
        with torch.no_grad():
            for index, (image_path, _target) in enumerate(dataset.samples):
                with image_module.open(image_path) as image:
                    tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
                probabilities = torch.softmax(model(tensor)[0, :len(classes)], dim=0)
                scores = {
                    class_id: float(probabilities[class_index].item())
                    for class_index, class_id in enumerate(classes)
                }
                top1_class_id = max(scores, key=scores.get)
                records.append(
                    {
                        'image_id': f'img_{Path(image_path).stem}',
                        'sample_id': Path(image_path).stem,
                        'object_id': None,
                        'top1_class_id': top1_class_id,
                        'top1_score': scores[top1_class_id],
                        'scores': scores,
                    },
                )

        return records


    def _load_classification_manifest(self, context: TaskExecutionContext) -> '_ClassificationManifest':
        prediction_split = self._prediction_split(context=context)
        dependency_manifests: list['_ClassificationManifest'] = []
        for dependency in context.dependency_results.values():
            try:
                manifest_path = dependency.artifact_path('classification_input_manifest')
            except KeyError:
                continue
            manifest = self._read_manifest(path=manifest_path, root_dir=manifest_path.parent)
            dependency_manifests.append(manifest)
            if prediction_split is None or self._manifest_has_split(manifest=manifest, split=prediction_split):
                return manifest

        if context.record.input_variant_path is None:
            if dependency_manifests:
                return dependency_manifests[0]
            raise ValueError('torchvision_classifier inference smoke requires input_variant_path')

        resolution = DatasetInputResolver().resolve_classification(
            root=context.record.input_variant_path,
            input_kind=context.record.input_variant_kind,
        )
        if resolution.manifest_path is not None:
            manifest = self._read_manifest(path=resolution.manifest_path, root_dir=resolution.dataset_root)
            if prediction_split is None or self._manifest_has_split(manifest=manifest, split=prediction_split):
                return manifest

        return self._classification_manifest_from_imagefolder_resolution(resolution=resolution)

    def _manifest_has_split(self, *, manifest: '_ClassificationManifest', split: str) -> bool:
        return any(
            isinstance(image, dict) and str(image.get('split') or '').lower() == split
            for image in manifest.images
        )

    def _classification_manifest_from_imagefolder_resolution(self, *, resolution: object) -> '_ClassificationManifest':
        dataset_root = getattr(resolution, 'dataset_root')
        split_roots = getattr(resolution, 'split_roots')
        classes = [str(class_name) for class_name in getattr(resolution, 'classes') if str(class_name)]
        extensions = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
        images: list[dict[str, object]] = []
        for split, split_root in sorted(split_roots.items()):
            for class_dir in sorted(Path(split_root).iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name
                for image_path in sorted(class_dir.rglob('*')):
                    if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                        continue
                    try:
                        relative_path = image_path.relative_to(Path(dataset_root))
                    except ValueError:
                        try:
                            relative_path = image_path.resolve().relative_to(Path(dataset_root).resolve())
                        except ValueError:
                            relative_path = Path(split) / label / image_path.name
                    sample_id = f'{split}_{label}_{image_path.stem}'
                    images.append(
                        {
                            'image_id': sample_id,
                            'sample_id': sample_id,
                            'label': label,
                            'path': str(relative_path).replace('\\', '/'),
                            'split': str(split),
                        },
                    )
        if not images:
            raise ValueError(f'classification imagefolder has no images: {dataset_root}')

        return _ClassificationManifest(
            dataset_id=Path(dataset_root).name,
            classes=classes,
            images=images,
            root_dir=Path(dataset_root),
        )

    def _read_manifest(self, *, path: Path, root_dir: Path) -> '_ClassificationManifest':
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} root must be an object')
        manifest_root_dir = root_dir
        raw_root_dir = data.get('root_dir')
        if isinstance(raw_root_dir, str) and raw_root_dir:
            candidate_root = Path(raw_root_dir).expanduser()
            manifest_root_dir = candidate_root if candidate_root.is_absolute() else (path.parent / candidate_root)

        dataset_id = data.get('dataset_id')
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f'{path.name}.dataset_id must be a non-empty string')
        raw_classes = data.get('classes')
        classes = [
            str(class_id)
            for class_id in raw_classes
            if isinstance(class_id, str) and class_id
        ] if isinstance(raw_classes, list) else []
        raw_images = data.get('images')
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f'{path.name}.images must be a non-empty list')
        if not classes:
            classes = sorted({
                str(image.get('label'))
                for image in raw_images
                if isinstance(image, dict) and isinstance(image.get('label'), str) and image.get('label')
            })
        if not classes:
            raise ValueError(f'{path.name}.classes must contain at least one class')

        images = [
            image
            for image in raw_images
            if isinstance(image, dict)
        ]
        if len(images) != len(raw_images):
            raise ValueError(f'{path.name}.images entries must be objects')

        return _ClassificationManifest(
            dataset_id=dataset_id,
            classes=classes,
            images=images,
            root_dir=manifest_root_dir,
        )

    def _predict_records(
        self,
        *,
        context: TaskExecutionContext,
        model_config: ModelConfig,
        manifest: '_ClassificationManifest',
        runtime: dict[str, object],
    ) -> list[dict[str, object]]:
        torch = runtime['torch']
        image_module = runtime['Image']
        transforms = runtime['transforms']
        if not hasattr(torch, 'manual_seed'):
            raise RuntimeError('torch runtime is missing manual_seed')
        torch.manual_seed(int(context.record.params.get('seed', 42)))
        device = str(context.record.params.get('device') or 'cpu')
        model = self._build_model(model_config=model_config)
        prediction_classes = list(manifest.classes)
        if model_config.checkpoint is not None:
            checkpoint_classes = self._checkpoint_classes(model_config=model_config, torch=torch, device=device)
            if checkpoint_classes:
                prediction_classes = checkpoint_classes
            self._replace_classifier_head(model=model, class_count=len(prediction_classes), torch=torch)
            checkpoint_payload = self._load_model_checkpoint(
                model=model,
                model_config=model_config,
                torch=torch,
                device=device,
            )
            loaded_classes = self._classes_from_checkpoint_payload(payload=checkpoint_payload)
            if loaded_classes:
                prediction_classes = loaded_classes
        model.to(device)
        model.eval()
        image_size = model_config.train.image_size or 224
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        records: list[dict[str, object]] = []

        with torch.no_grad():
            for image_index, image_record in enumerate(manifest.images):
                image_path = self._image_path(root_dir=manifest.root_dir, image_record=image_record)
                with image_module.open(image_path) as image:
                    tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
                logits = model(tensor)
                records.append(
                    self._prediction_record(
                        image_record=image_record,
                        logits=logits,
                        classes=prediction_classes,
                        image_index=image_index,
                        torch=torch,
                    ),
                )

        return records

    def _load_model_checkpoint(
        self,
        *,
        model: object,
        model_config: ModelConfig,
        torch: object,
        device: str,
    ) -> dict[str, object] | None:
        checkpoint_path = self._prepared_checkpoint_path(model_config=model_config)
        if checkpoint_path is None:
            return None
        payload = self._torch_load_checkpoint(torch=torch, path=checkpoint_path, device=device)
        return self._load_checkpoint_payload_into_model(
            model=model,
            payload=payload,
        )

    def _load_checkpoint_path_into_model(
        self,
        *,
        model: object,
        path: Path,
        torch: object,
        device: str,
    ) -> dict[str, object]:
        payload = self._torch_load_checkpoint(torch=torch, path=path, device=device)
        return self._load_checkpoint_payload_into_model(
            model=model,
            payload=payload,
        )

    def _load_checkpoint_payload_into_model(
        self,
        *,
        model: object,
        payload: object,
    ) -> dict[str, object]:
        state_dict = self._state_dict_from_checkpoint_payload(payload=payload)
        state_dict, skipped_keys = self._compatible_checkpoint_state_dict(model=model, state_dict=state_dict)
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(load_result, 'missing_keys', []))
        unexpected_keys = list(getattr(load_result, 'unexpected_keys', []))
        if missing_keys or unexpected_keys or skipped_keys:
            # Keep loading permissive so classifier heads can evolve, but surface the mismatch for debugging.
            payload = {
                **payload,
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'skipped_mismatched_keys': skipped_keys,
            } if isinstance(payload, dict) else {
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'skipped_mismatched_keys': skipped_keys,
            }

        return payload if isinstance(payload, dict) else {'raw_checkpoint_type': type(payload).__name__}

    def _checkpoint_classes(
        self,
        *,
        model_config: ModelConfig,
        torch: object,
        device: str,
    ) -> list[str]:
        checkpoint_path = self._prepared_checkpoint_path(model_config=model_config)
        if checkpoint_path is None:
            return []
        payload = self._torch_load_checkpoint(torch=torch, path=checkpoint_path, device=device)

        return self._classes_from_checkpoint_payload(payload=payload)

    def _classes_from_checkpoint_payload(self, *, payload: object) -> list[str]:
        if not isinstance(payload, dict):
            return []
        raw_classes = payload.get('classes')
        if not isinstance(raw_classes, list):
            return []

        return [
            str(class_id)
            for class_id in raw_classes
            if isinstance(class_id, str) and class_id
        ]

    def _compatible_checkpoint_state_dict(self, *, model: object, state_dict: object) -> tuple[dict[str, object], list[str]]:
        if not isinstance(state_dict, dict):
            raise ValueError('checkpoint state_dict must be a dict')

        current_state = model.state_dict()
        compatible: dict[str, object] = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if current_value is not None and hasattr(current_value, 'shape') and hasattr(value, 'shape'):
                if tuple(current_value.shape) != tuple(value.shape):
                    skipped.append(str(key))
                    continue
            compatible[str(key)] = value

        return compatible, skipped

    def _torch_load_checkpoint(self, *, torch: object, path: Path, device: str) -> object:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    def _state_dict_from_checkpoint_payload(self, *, payload: object) -> object:
        if isinstance(payload, dict):
            for key in ('state_dict', 'model_state_dict'):
                state_dict = payload.get(key)
                if isinstance(state_dict, dict):
                    return state_dict
            if payload and all(hasattr(value, 'shape') for value in payload.values()):
                return payload
        if hasattr(payload, 'state_dict'):
            return payload.state_dict()

        raise ValueError('checkpoint does not contain a supported model state_dict')

    def _build_model(self, *, model_config: ModelConfig) -> object:
        adapter = TorchvisionClassificationAdapter(model_config=self._checkpoint_safe_build_config(model_config=model_config))
        models_module = adapter._load_torchvision_models()
        builder = getattr(models_module, adapter._builder_name())
        weights = adapter._weights(models_module=models_module)

        return builder(weights=weights)

    def _image_path(self, *, root_dir: Path, image_record: dict[str, object]) -> Path:
        raw_path = image_record.get('path')
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError('image record path must be a non-empty string')
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            image_path = relative_path.resolve()
            if not image_path.exists():
                raise FileNotFoundError(f'image file not found: {image_path}')
            return image_path
        if '..' in relative_path.parts:
            raise ValueError(f'image path must be a safe relative path: {raw_path}')
        image_path = (root_dir / relative_path).resolve()
        try:
            image_path.relative_to(root_dir.resolve())
        except ValueError as error:
            raise ValueError(f'image path escapes manifest root: {raw_path}') from error
        if not image_path.exists():
            raise FileNotFoundError(f'image file not found: {image_path}')

        return image_path

    def _prediction_record(
        self,
        *,
        image_record: dict[str, object],
        logits: object,
        classes: list[str],
        image_index: int,
        torch: object,
    ) -> dict[str, object]:
        class_logits = logits[0, :len(classes)]
        probabilities = torch.softmax(class_logits, dim=0)
        scores = {
            class_id: float(probabilities[index].item())
            for index, class_id in enumerate(classes)
        }
        top1_class_id = max(scores, key=scores.get)

        return {
            'image_id': self._required_str(image_record, 'image_id', f'image_{image_index:04d}'),
            'sample_id': self._required_str(image_record, 'sample_id', f'sample_{image_index:04d}'),
            'object_id': image_record.get('object_id') if isinstance(image_record.get('object_id'), str) else None,
            'top1_class_id': top1_class_id,
            'top1_score': scores[top1_class_id],
            'scores': scores,
        }

    def _required_str(self, record: dict[str, object], key: str, fallback: str) -> str:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value

        return fallback

    def _accuracy(self, *, records: list[dict[str, object]], images: list[dict[str, object]]) -> float | str:
        labels = [
            image.get('label')
            for image in images
        ]
        if not labels or not all(isinstance(label, str) and label for label in labels):
            return ''

        correct = 0
        for record, label in zip(records, labels, strict=True):
            if record['top1_class_id'] == label:
                correct += 1

        return correct / len(labels)

    def _classification_prediction_metric_row(
        self,
        *,
        records: list[dict[str, object]],
        images: list[dict[str, object]],
    ) -> dict[str, object]:
        row = self._empty_metric_row()
        labels = [
            str(image.get('label'))
            for image in images
            if isinstance(image.get('label'), str) and image.get('label')
        ]
        predictions = [
            str(record.get('top1_class_id'))
            for record in records
            if isinstance(record.get('top1_class_id'), str) and record.get('top1_class_id')
        ]
        row['num_predictions'] = len(predictions)
        row['num_gt'] = len(labels)
        if len(labels) != len(images) or len(predictions) != len(records) or len(labels) != len(predictions):
            return row

        correct = sum(1 for label, prediction in zip(labels, predictions, strict=True) if label == prediction)
        accuracy = correct / max(len(labels), 1)
        macro_precision, macro_recall, macro_f1, min_recall = self._classification_macro_metrics(
            labels=labels,
            predictions=predictions,
        )
        row.update(
            {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'class_recall': min_recall,
                'label_error_rate': 1.0 - accuracy,
            },
        )

        return row

    def _classification_label_prediction_pairs(
        self,
        *,
        records: list[dict[str, object]],
        images: list[dict[str, object]],
    ) -> list[tuple[str, str]]:
        labels = [
            str(image.get('label'))
            for image in images
            if isinstance(image.get('label'), str) and image.get('label')
        ]
        predictions = [
            str(record.get('top1_class_id'))
            for record in records
            if isinstance(record.get('top1_class_id'), str) and record.get('top1_class_id')
        ]
        if len(labels) != len(images) or len(predictions) != len(records) or len(labels) != len(predictions):
            return []

        return list(zip(labels, predictions, strict=True))

    def _write_classification_input_manifest_artifact(
        self,
        *,
        context: TaskExecutionContext,
        dataset_id: str,
        classes: list[str],
        images: list[dict[str, object]],
        root_dir: Path | None = None,
    ) -> dict[str, object]:
        manifest_path = context.result_dir / 'classification_input_manifest.json'
        resolved_root_dir = root_dir.resolve() if root_dir is not None else None
        manifest_images: list[dict[str, object]] = []
        for index, image in enumerate(images):
            if not isinstance(image, dict):
                continue
            label = image.get('label')
            if not isinstance(label, str) or not label:
                continue
            record: dict[str, object] = {
                'image_id': str(image.get('image_id') or f'image_{index:04d}'),
                'sample_id': str(image.get('sample_id') or f'sample_{index:04d}'),
                'label': label,
            }
            for key in ('object_id', 'path', 'split'):
                value = image.get(key)
                if isinstance(value, str) and value:
                    record[key] = self._manifest_artifact_value(
                        key=key,
                        value=value,
                        root_dir=resolved_root_dir,
                    )
            manifest_images.append(record)

        payload: dict[str, object] = {
            'schema_version': '0.1',
            'artifact_type': 'classification_input_manifest',
            'dataset_id': dataset_id,
            'classes': [
                class_name
                for class_name in classes
                if isinstance(class_name, str) and class_name
            ],
            'images': manifest_images,
        }
        if resolved_root_dir is not None:
            payload['root_dir'] = resolved_root_dir.as_posix()
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
            + '\n',
            encoding='utf-8',
        )
        return {
            'name': 'classification_input_manifest',
            'path': self._relative_to_result_dir(context=context, path=manifest_path),
            'kind': 'manifest',
            'required': False,
        }

    def _manifest_artifact_value(self, *, key: str, value: str, root_dir: Path | None) -> str:
        if key != 'path' or root_dir is None:
            return value
        image_path = Path(value)
        if not image_path.is_absolute():
            return image_path.as_posix()
        try:
            return image_path.resolve().relative_to(root_dir).as_posix()
        except ValueError:
            return image_path.as_posix()

    def _write_classification_confusion_artifacts(
        self,
        *,
        context: TaskExecutionContext,
        records: list[dict[str, object]],
        images: list[dict[str, object]],
        classes: list[str],
    ) -> list[dict[str, object]]:
        pairs = self._classification_label_prediction_pairs(records=records, images=images)
        if not pairs:
            return []

        class_names = [
            str(class_name)
            for class_name in classes
            if isinstance(class_name, str) and class_name
        ]
        for class_name in sorted(set(label for label, _prediction in pairs) | set(prediction for _label, prediction in pairs)):
            if class_name not in class_names:
                class_names.append(class_name)

        matrix = {
            actual: {predicted: 0 for predicted in class_names}
            for actual in class_names
        }
        for actual, predicted in pairs:
            matrix.setdefault(actual, {class_name: 0 for class_name in class_names})
            if predicted not in matrix[actual]:
                matrix[actual][predicted] = 0
            matrix[actual][predicted] += 1

        metrics_dir = context.result_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        count_path = metrics_dir / 'classification_confusion_matrix.csv'
        normalized_path = metrics_dir / 'classification_confusion_matrix_normalized.csv'
        pairs_path = metrics_dir / 'classification_confusion_pairs.csv'
        self._write_confusion_matrix_csv(path=count_path, class_names=class_names, matrix=matrix, normalized=False)
        self._write_confusion_matrix_csv(path=normalized_path, class_names=class_names, matrix=matrix, normalized=True)
        self._write_confusion_pairs_csv(path=pairs_path, class_names=class_names, matrix=matrix)

        return [
            {
                'name': 'classification_confusion_matrix',
                'path': self._relative_to_result_dir(context=context, path=count_path),
                'kind': 'metrics',
                'required': False,
            },
            {
                'name': 'classification_confusion_matrix_normalized',
                'path': self._relative_to_result_dir(context=context, path=normalized_path),
                'kind': 'metrics',
                'required': False,
            },
            {
                'name': 'classification_confusion_pairs',
                'path': self._relative_to_result_dir(context=context, path=pairs_path),
                'kind': 'metrics',
                'required': False,
            },
        ]

    def _write_confusion_matrix_csv(
        self,
        *,
        path: Path,
        class_names: list[str],
        matrix: dict[str, dict[str, int]],
        normalized: bool,
    ) -> None:
        with _windows_extended_path(path).open(mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['actual', *[f'pred_{class_name}' for class_name in class_names], 'support'])
            for actual in class_names:
                row_counts = matrix.get(actual, {})
                support = sum(int(row_counts.get(predicted, 0)) for predicted in class_names)
                values: list[float | int] = []
                for predicted in class_names:
                    count = int(row_counts.get(predicted, 0))
                    values.append(count / support if normalized and support else count)
                writer.writerow([actual, *values, support])

    def _write_confusion_pairs_csv(
        self,
        *,
        path: Path,
        class_names: list[str],
        matrix: dict[str, dict[str, int]],
    ) -> None:
        rows: list[tuple[str, str, int]] = []
        for actual in class_names:
            for predicted in class_names:
                if actual == predicted:
                    continue
                count = int(matrix.get(actual, {}).get(predicted, 0))
                if count:
                    rows.append((actual, predicted, count))
        rows.sort(key=lambda item: (-item[2], item[0], item[1]))
        with _windows_extended_path(path).open(mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['actual', 'predicted', 'count'])
            writer.writerows(rows)

    def _imagefolder_dataset_metric_images(
        self,
        *,
        dataset: object,
        classes: list[str],
        split: str | None = None,
    ) -> list[dict[str, object]]:
        samples = getattr(dataset, 'samples', None)
        if isinstance(samples, list):
            images: list[dict[str, object]] = []
            for index, (path, class_index) in enumerate(samples):
                try:
                    label = classes[int(class_index)]
                except (IndexError, TypeError, ValueError):
                    images.append({})
                    continue
                image_path = Path(path)
                record: dict[str, object] = {
                    'image_id': str(image_path.stem or f'image_{index:04d}'),
                    'sample_id': str(image_path.stem or f'sample_{index:04d}'),
                    'label': label,
                    'path': str(image_path),
                }
                if split:
                    record['split'] = split
                images.append(record)
            return images
        targets = getattr(dataset, 'targets', None)
        if isinstance(targets, list):
            images = []
            for index, class_index in enumerate(targets):
                try:
                    record = {
                        'image_id': f'image_{index:04d}',
                        'sample_id': f'sample_{index:04d}',
                        'label': classes[int(class_index)],
                    }
                    if split:
                        record['split'] = split
                    images.append(record)
                except (IndexError, TypeError, ValueError):
                    images.append({})
            return images

        return []


class UltralyticsYoloClassifierTaskAdapter(TorchvisionClassifierTaskAdapter):
    adapter_key = 'ultralytics_yolo_classifier'
    default_model_id = 'yolo26n-cls'
    _IMAGE_EXTENSIONS = frozenset({'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'})

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        model_config = self._model_config(context=context)
        adapter = UltralyticsYoloClassificationAdapter(model_config=model_config)
        reference = adapter._model_reference_contract()

        return {
            'model_reference': reference.reference,
            'pretrained': reference.pretrained,
            'checkpoint': reference.checkpoint,
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_ready',
            'ultralytics_task': 'classify',
            **self._weight_policy_metadata(context=context, model_config=model_config),
        }

    def _run_inference_smoke(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        model_config = self._model_config(context=context)
        if (
            model_config.checkpoint is None
            and model_config.pretrained
            and not bool(context.record.params.get('allow_pretrained_download', False))
        ):
            return self._failed_result(
                context=context,
                message=(
                    'ultralytics_yolo_classifier inference smoke refuses implicit pretrained weight downloads; '
                    'set pretrained=false or provide an explicit prepared checkpoint path'
                ),
                failure_type='real_adapter_contract_invalid',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        try:
            self._prepared_checkpoint_path(model_config=model_config)
            manifest = self._load_classification_manifest(context=context)
            prediction_split = self._prediction_split(context=context)
            manifest = _ClassificationManifest(
                dataset_id=manifest.dataset_id,
                classes=manifest.classes,
                images=self._filtered_prediction_images(
                    images=manifest.images,
                    prediction_split=prediction_split,
                    manifest_name='classification manifest',
                ),
                root_dir=manifest.root_dir,
            )
            yolo_class = self._load_yolo_runtime(context=context)
            model = self._build_yolo_model(model_config=model_config, yolo_class=yolo_class)
            records = self._predict_manifest_records(
                context=context,
                model=model,
                manifest=manifest,
                classes=manifest.classes,
            )
        except (OSError, ValueError, RuntimeError, MissingUltralyticsDependencyError) as error:
            return self._failed_result(
                context=context,
                message=f'ultralytics_yolo_classifier inference smoke failed: {error}',
                failure_type='real_adapter_execution_failed'
                if not isinstance(error, MissingUltralyticsDependencyError)
                else 'real_adapter_dependency_missing',
                metadata={**metadata, 'execution_mode': 'inference_smoke', 'error': str(error)},
            )

        prediction_path = context.result_dir / 'predictions' / 'classification_predictions.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '0.1',
            'task': 'classification',
            'model_id': model_config.model_id or self.default_model_id,
            'dataset_id': manifest.dataset_id,
            'success': True,
            'records': records,
        }
        prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        validation = PredictionArtifactValidator().validate_payload(payload, expected_task='classification')
        if not validation.is_valid:
            return self._failed_result(
                context=context,
                message='ultralytics_yolo_classifier prediction artifact validation failed: '
                + '; '.join(validation.error_messages()),
                failure_type='real_adapter_output_invalid',
                metadata={**metadata, 'execution_mode': 'inference_smoke'},
            )

        metric_row = self._classification_prediction_metric_row(records=records, images=manifest.images)
        input_manifest_artifact = self._write_classification_input_manifest_artifact(
            context=context,
            dataset_id=manifest.dataset_id,
            classes=manifest.classes,
            images=manifest.images,
            root_dir=manifest.root_dir,
        )
        confusion_artifacts = self._write_classification_confusion_artifacts(
            context=context,
            records=records,
            images=manifest.images,
            classes=manifest.classes,
        )

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'ultralytics_yolo_classifier inference smoke completed: predictions={len(records)}',
            metrics=[
                {
                    **self._empty_metric_row(),
                    'accuracy': metric_row['accuracy'],
                    'precision': metric_row['precision'],
                    'recall': metric_row['recall'],
                    'macro_precision': metric_row['macro_precision'],
                    'macro_recall': metric_row['macro_recall'],
                    'macro_f1': metric_row['macro_f1'],
                    'class_recall': metric_row['class_recall'],
                    'num_predictions': len(records),
                    'num_gt': metric_row['num_gt'],
                    'label_error_rate': metric_row['label_error_rate'],
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
                input_manifest_artifact,
                *confusion_artifacts,
            ],
            metadata={
                **metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'inference_smoke',
                'dataset_id': manifest.dataset_id,
                'input_root': str(manifest.root_dir),
                'prediction_split': prediction_split or 'all',
                'prediction_count': len(records),
                'checkpoint_loaded': model_config.checkpoint is not None,
            },
        )

    def _run_train(
        self,
        *,
        context: TaskExecutionContext,
        metadata: dict[str, object],
    ) -> TaskAdapterResult:
        model_config = self._model_config(context=context)
        if (
            model_config.checkpoint is None
            and model_config.pretrained
            and not bool(context.record.params.get('allow_pretrained_download', False))
        ):
            return self._failed_result(
                context=context,
                message=(
                    'ultralytics_yolo_classifier train mode refuses implicit pretrained weight downloads; '
                    'set pretrained=false, provide checkpoint, or set allow_pretrained_download=true'
                ),
                failure_type='real_adapter_contract_invalid',
                metadata={**metadata, 'execution_mode': 'train'},
            )

        try:
            self._prepared_checkpoint_path(model_config=model_config)
            resolution = DatasetInputResolver().resolve_classification(
                root=self._required_input_root(context=context),
                input_kind=context.record.input_variant_kind,
            )
            if resolution.imagefolder_root is None:
                raise ValueError('ultralytics_yolo_classifier train mode requires an image-folder classification dataset')
            if 'train' not in resolution.split_roots:
                raise ValueError('ultralytics_yolo_classifier train mode requires a train split')
            if 'val' not in resolution.split_roots:
                raise ValueError('ultralytics_yolo_classifier train mode requires a val split')
            yolo_class = self._load_yolo_runtime(context=context)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = self._build_yolo_model(model_config=model_config, yolo_class=yolo_class)
                train_result = model.train(
                    **self._train_kwargs(
                        context=context,
                        model_config=model_config,
                        data_root=resolution.imagefolder_root,
                    ),
                )
        except (OSError, ValueError, RuntimeError, MissingUltralyticsDependencyError) as error:
            return self._failed_result(
                context=context,
                message=f'ultralytics_yolo_classifier train failed: {error}',
                failure_type='real_adapter_execution_failed'
                if not isinstance(error, MissingUltralyticsDependencyError)
                else 'real_adapter_dependency_missing',
                metadata={**metadata, 'execution_mode': 'train', 'error': str(error)},
            )

        save_dir = self._train_save_dir(train_result=train_result)
        best_checkpoint = self._checkpoint_path(save_dir=save_dir, name='best.pt')
        last_checkpoint = self._checkpoint_path(save_dir=save_dir, name='last.pt')
        metrics = self._train_metric_rows(save_dir=save_dir)
        class_names = list(resolution.classes)
        prediction_split = self._prediction_split(context=context)
        if prediction_split is None:
            prediction_split = 'test' if 'test' in resolution.split_roots else 'val'
        prediction_split_root = resolution.split_roots.get(prediction_split)
        if prediction_split_root is None:
            raise ValueError(
                f'ultralytics_yolo_classifier train mode requires prediction_split={prediction_split} '
                f'under {resolution.imagefolder_root}'
            )

        try:
            prediction_model = self._build_yolo_model_from_path_or_config(
                path=best_checkpoint,
                model_config=model_config,
                yolo_class=yolo_class,
            )
            predictions = self._predict_imagefolder_records(
                context=context,
                model=prediction_model,
                split_root=prediction_split_root,
                classes=class_names,
            )
            prediction_path = context.result_dir / 'predictions' / 'classification_predictions.json'
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_payload = {
                'schema_version': '0.1',
                'task': 'classification',
                'model_id': model_config.model_id or self.default_model_id,
                'dataset_id': resolution.dataset_root.name,
                'success': True,
                'records': predictions,
            }
            prediction_path.write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2), encoding='utf-8')
            validation = PredictionArtifactValidator().validate_payload(prediction_payload, expected_task='classification')
            if not validation.is_valid:
                raise ValueError('classification prediction artifact validation failed: ' + '; '.join(validation.error_messages()))
        except (OSError, ValueError, RuntimeError) as error:
            return self._failed_result(
                context=context,
                message=f'ultralytics_yolo_classifier post-train prediction failed: {error}',
                failure_type='real_adapter_execution_failed',
                metadata={**metadata, 'execution_mode': 'train', 'error': str(error)},
            )

        prediction_images = self._imagefolder_metric_images(root=prediction_split_root, split=prediction_split)
        prediction_metric_row = self._classification_prediction_metric_row(records=predictions, images=prediction_images)
        input_manifest_artifact = self._write_classification_input_manifest_artifact(
            context=context,
            dataset_id=resolution.dataset_root.name,
            classes=class_names,
            images=prediction_images,
            root_dir=prediction_split_root,
        )
        confusion_artifacts = self._write_classification_confusion_artifacts(
            context=context,
            records=predictions,
            images=prediction_images,
            classes=class_names,
        )
        if metrics:
            metrics[-1] = {
                **metrics[-1],
                'accuracy': prediction_metric_row['accuracy'],
                'precision': prediction_metric_row['precision'],
                'recall': prediction_metric_row['recall'],
                'macro_precision': prediction_metric_row['macro_precision'],
                'macro_recall': prediction_metric_row['macro_recall'],
                'macro_f1': prediction_metric_row['macro_f1'],
                'class_recall': prediction_metric_row['class_recall'],
                'num_predictions': prediction_metric_row['num_predictions'],
                'num_gt': prediction_metric_row['num_gt'],
                'label_error_rate': prediction_metric_row['label_error_rate'],
            }

        artifacts = self._train_artifacts(
            context=context,
            save_dir=save_dir,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
        )
        artifacts.append(
            {
                'name': 'classification_predictions',
                'path': 'predictions/classification_predictions.json',
                'kind': 'prediction',
                'required': True,
            },
        )
        artifacts.append(input_manifest_artifact)
        artifacts.extend(confusion_artifacts)

        return TaskAdapterResult(
            success=True,
            status='finished',
            message=f'ultralytics_yolo_classifier train completed: model_id={model_config.model_id or self.default_model_id}',
            metrics=metrics,
            predictions=predictions,
            artifacts=artifacts,
            metadata={
                **metadata,
                **resolution.metadata,
                'adapter': self.adapter_key,
                'execution_mode': 'train',
                'ultralytics_task': 'classify',
                'train_save_dir': None if save_dir is None else str(save_dir),
                'best_checkpoint': None if best_checkpoint is None else str(best_checkpoint),
                'last_checkpoint': None if last_checkpoint is None else str(last_checkpoint),
                'class_count': len(class_names),
                'classes': class_names,
                'checkpoint_loaded': model_config.checkpoint is not None,
                'prediction_split': prediction_split,
                'prediction_count': len(predictions),
            },
        )

    def _load_yolo_runtime(self, context: TaskExecutionContext) -> type[Any]:
        config_dir = context.result_dir / 'runtime' / 'ultralytics_config'
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from ultralytics import YOLO
        except ImportError as error:
            raise MissingUltralyticsDependencyError(
                f'ultralytics is required for ultralytics_yolo_classifier: {error}',
            ) from error
        finally:
            Image.open = _ORIGINAL_PIL_IMAGE_OPEN

        return YOLO

    def _build_yolo_model(self, *, model_config: ModelConfig, yolo_class: type[Any]) -> Any:
        reference = UltralyticsYoloClassificationAdapter(model_config=model_config)._model_reference()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = yolo_class(reference)
        Image.open = _ORIGINAL_PIL_IMAGE_OPEN

        return model

    def _build_yolo_model_from_path_or_config(
        self,
        *,
        path: Path | None,
        model_config: ModelConfig,
        yolo_class: type[Any],
    ) -> Any:
        if path is not None and path.exists():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = yolo_class(str(path))
            Image.open = _ORIGINAL_PIL_IMAGE_OPEN

            return model

        return self._build_yolo_model(model_config=model_config, yolo_class=yolo_class)

    def _train_kwargs(
        self,
        *,
        context: TaskExecutionContext,
        model_config: ModelConfig,
        data_root: Path,
    ) -> dict[str, object]:
        params = context.record.params
        kwargs: dict[str, object] = {
            'data': str(data_root),
            'task': 'classify',
            'project': str(context.result_dir / 'checkpoints'),
            'name': str(model_config.model_id or self.default_model_id),
            'exist_ok': True,
            'verbose': bool(params.get('verbose', False)),
        }
        for param_name, yolo_name in [
            ('epochs', 'epochs'),
            ('image_size', 'imgsz'),
            ('batch_size', 'batch'),
            ('workers', 'workers'),
            ('patience', 'patience'),
            ('seed', 'seed'),
        ]:
            value = self._optional_int(params.get(param_name))
            if value is not None:
                kwargs[yolo_name] = value
        learning_rate = self._optional_float(params.get('learning_rate'))
        if learning_rate is not None:
            kwargs['lr0'] = learning_rate
        device = params.get('device')
        if device is not None and str(device).strip():
            kwargs['device'] = str(device)

        return kwargs

    def _train_save_dir(self, *, train_result: object) -> Path | None:
        save_dir = getattr(train_result, 'save_dir', None)
        if save_dir is None:
            return None

        return Path(save_dir)

    def _checkpoint_path(self, *, save_dir: Path | None, name: str) -> Path | None:
        if save_dir is None:
            return None
        path = save_dir / 'weights' / name

        return path if path.exists() else None

    def _train_metric_rows(self, *, save_dir: Path | None) -> list[dict[str, object]]:
        if save_dir is None:
            return [self._empty_metric_row()]
        results_csv = save_dir / 'results.csv'
        if not results_csv.exists():
            return [self._empty_metric_row()]
        with results_csv.open(mode='r', encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return [self._empty_metric_row()]

        return [
            self._metric_row_from_yolo_classify(row=row, fallback_epoch=index + 1)
            for index, row in enumerate(rows)
        ]

    def _metric_row_from_yolo_classify(self, *, row: dict[str, str], fallback_epoch: int) -> dict[str, object]:
        return {
            **self._empty_metric_row(),
            'epoch': self._metric_value(row=row, keys=('epoch',), fallback=fallback_epoch),
            'train_loss': self._metric_value(row=row, keys=('train/loss', 'train_loss', 'loss')),
            'val_loss': self._metric_value(row=row, keys=('val/loss', 'val_loss')),
            'accuracy': self._metric_value(row=row, keys=('metrics/accuracy_top1', 'accuracy', 'top1')),
            'macro_f1': '',
            'lr': self._metric_value(row=row, keys=('lr/pg0', 'lr')),
        }

    def _train_artifacts(
        self,
        *,
        context: TaskExecutionContext,
        save_dir: Path | None,
        best_checkpoint: Path | None,
        last_checkpoint: Path | None,
    ) -> list[dict[str, object]]:
        artifacts: list[dict[str, object]] = []
        if best_checkpoint is not None:
            artifacts.append(
                {
                    'name': 'best_checkpoint',
                    'path': self._relative_to_result_dir(context=context, path=best_checkpoint),
                    'kind': 'checkpoint',
                    'required': True,
                },
            )
        if last_checkpoint is not None:
            artifacts.append(
                {
                    'name': 'last_checkpoint',
                    'path': self._relative_to_result_dir(context=context, path=last_checkpoint),
                    'kind': 'checkpoint',
                    'required': False,
                },
            )
        if save_dir is not None and (save_dir / 'results.csv').exists():
            artifacts.append(
                {
                    'name': 'yolo_results_csv',
                    'path': self._relative_to_result_dir(context=context, path=save_dir / 'results.csv'),
                    'kind': 'metrics',
                    'required': False,
                },
            )

        return artifacts

    def _predict_manifest_records(
        self,
        *,
        context: TaskExecutionContext,
        model: Any,
        manifest: '_ClassificationManifest',
        classes: list[str],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        image_size = self._optional_int(context.record.params.get('image_size')) or 224
        device = str(context.record.params.get('device') or 'cpu')
        for image_index, image_record in enumerate(manifest.images):
            image_path = self._image_path(root_dir=manifest.root_dir, image_record=image_record)
            result = self._predict_one(model=model, image_path=image_path, image_size=image_size, device=device)
            records.append(
                self._record_from_yolo_result(
                    result=result,
                    image_id=self._required_str(image_record, 'image_id', f'image_{image_index:04d}'),
                    sample_id=self._required_str(image_record, 'sample_id', f'sample_{image_index:04d}'),
                    object_id=image_record.get('object_id') if isinstance(image_record.get('object_id'), str) else None,
                    fallback_classes=classes,
                ),
            )

        return records

    def _predict_imagefolder_records(
        self,
        *,
        context: TaskExecutionContext,
        model: Any,
        split_root: Path,
        classes: list[str],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        image_size = self._optional_int(context.record.params.get('image_size')) or 224
        device = str(context.record.params.get('device') or 'cpu')
        for image_path in self._iter_imagefolder_images(root=split_root):
            result = self._predict_one(model=model, image_path=image_path, image_size=image_size, device=device)
            records.append(
                self._record_from_yolo_result(
                    result=result,
                    image_id=f'img_{image_path.stem}',
                    sample_id=image_path.stem,
                    object_id=None,
                    fallback_classes=classes,
                ),
            )

        return records

    def _iter_imagefolder_images(self, *, root: Path) -> list[Path]:
        return sorted(
            path
            for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in self._IMAGE_EXTENSIONS
        )

    def _imagefolder_metric_images(self, *, root: Path, split: str | None = None) -> list[dict[str, object]]:
        images: list[dict[str, object]] = []
        for index, image_path in enumerate(self._iter_imagefolder_images(root=root)):
            record: dict[str, object] = {
                'image_id': str(image_path.stem or f'image_{index:04d}'),
                'sample_id': str(image_path.stem or f'sample_{index:04d}'),
                'label': image_path.parent.name,
                'path': str(image_path),
            }
            if split:
                record['split'] = split
            images.append(record)

        return images

    def _predict_one(self, *, model: Any, image_path: Path, image_size: int, device: str) -> Any:
        kwargs: dict[str, object] = {
            'source': str(image_path),
            'imgsz': image_size,
            'verbose': False,
        }
        if device:
            kwargs['device'] = device
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results = model.predict(**kwargs)
        Image.open = _ORIGINAL_PIL_IMAGE_OPEN
        if not results:
            raise ValueError(f'YOLO classifier produced no result for image: {image_path}')

        return results[0]

    def _record_from_yolo_result(
        self,
        *,
        result: Any,
        image_id: str,
        sample_id: str,
        object_id: str | None,
        fallback_classes: list[str],
    ) -> dict[str, object]:
        probs = getattr(result, 'probs', None)
        if probs is None:
            raise ValueError('YOLO classification result has no probs field')
        names = getattr(result, 'names', None)
        score_values = self._probability_values(probs=probs)
        top1_index = self._top1_index(probs=probs, score_values=score_values)
        class_names = self._class_names(names=names, fallback_classes=fallback_classes, score_count=len(score_values))
        top1_class_id = class_names[top1_index] if top1_index < len(class_names) else str(top1_index)
        top1_score = self._top1_score(probs=probs, score_values=score_values, top1_index=top1_index)
        scores = {
            class_names[index] if index < len(class_names) else str(index): float(score)
            for index, score in enumerate(score_values)
        }
        if not scores:
            scores = {top1_class_id: top1_score}

        return {
            'image_id': image_id,
            'sample_id': sample_id,
            'object_id': object_id,
            'top1_class_id': top1_class_id,
            'top1_score': top1_score,
            'scores': scores,
        }

    def _probability_values(self, *, probs: Any) -> list[float]:
        data = getattr(probs, 'data', None)
        converted = self._to_python(value=data)
        if isinstance(converted, list):
            return [float(value) for value in converted]

        return []

    def _top1_index(self, *, probs: Any, score_values: list[float]) -> int:
        top1 = getattr(probs, 'top1', None)
        if top1 is not None:
            return int(self._scalar(value=top1))
        if score_values:
            return max(range(len(score_values)), key=score_values.__getitem__)

        return 0

    def _top1_score(self, *, probs: Any, score_values: list[float], top1_index: int) -> float:
        top1_conf = getattr(probs, 'top1conf', None)
        if top1_conf is not None:
            return float(self._scalar(value=top1_conf))
        if top1_index < len(score_values):
            return float(score_values[top1_index])

        return 0.0

    def _class_names(self, *, names: object, fallback_classes: list[str], score_count: int) -> list[str]:
        if isinstance(names, dict) and names:
            return [
                str(names.get(index) or names.get(str(index)) or index)
                for index in range(max(score_count, len(names)))
            ]
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        if fallback_classes:
            return fallback_classes

        return [str(index) for index in range(score_count)]

    def _scalar(self, *, value: Any) -> float:
        converted = self._to_python(value=value)
        if isinstance(converted, list):
            if not converted:
                return 0.0
            return float(converted[0])

        return float(converted)

    def _to_python(self, *, value: Any) -> Any:
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'tolist'):
            return value.tolist()

        return value


class TimmClassifierTaskAdapter(TorchvisionClassifierTaskAdapter):
    adapter_key = 'timm_classifier'
    default_model_id = 'convnext_v2_tiny'

    def _contract_metadata(self, context: TaskExecutionContext) -> dict[str, object]:
        model_config = self._model_config(context=context)
        adapter = TimmClassificationAdapter(model_config=model_config)
        reference = adapter._model_reference_contract()

        return {
            'timm_model_name': reference.timm_model_name,
            'pretrained': reference.pretrained,
            'checkpoint': reference.checkpoint,
            'input_variant_kind': context.record.input_variant_kind,
            'execution_gate': 'adapter_ready',
            **self._weight_policy_metadata(context=context, model_config=model_config),
        }

    def _load_runtime(self) -> dict[str, object]:
        runtime = super()._load_runtime()
        try:
            import timm
        except ImportError as error:
            raise MissingTimmDependencyError(
                'timm is required for timm_classifier train/inference smoke',
            ) from error
        runtime['timm'] = timm

        return runtime

    def _build_model(self, *, model_config: ModelConfig) -> object:
        adapter = TimmClassificationAdapter(model_config=self._checkpoint_safe_build_config(model_config=model_config))
        timm = adapter._load_timm()
        reference = adapter._model_reference_contract()
        image_size = model_config.train.image_size or getattr(model_config.predict, 'image_size', None)
        create_kwargs: dict[str, object] = {'pretrained': reference.pretrained}
        if image_size is not None and self._supports_timm_img_size(reference.timm_model_name):
            create_kwargs['img_size'] = int(image_size)

        return timm.create_model(reference.timm_model_name, **create_kwargs)

    def _supports_timm_img_size(self, timm_model_name: str) -> bool:
        # Swin models assert against their configured img_size during forward.
        # Passing img_size keeps resolution ablation configs and the model graph aligned.
        return timm_model_name.startswith('swin_')

    def _replace_classifier_head(self, *, model: object, class_count: int, torch: object) -> None:
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=class_count)
            return

        super()._replace_classifier_head(model=model, class_count=class_count, torch=torch)


class _ClassificationManifest:
    def __init__(
        self,
        *,
        dataset_id: str,
        classes: list[str],
        images: list[dict[str, object]],
        root_dir: Path,
    ) -> None:
        self.dataset_id = dataset_id
        self.classes = classes
        self.images = images
        self.root_dir = root_dir


class _DetectionManifest:
    def __init__(
        self,
        *,
        dataset_id: str,
        images: list[dict[str, object]],
        root_dir: Path,
    ) -> None:
        self.dataset_id = dataset_id
        self.images = images
        self.root_dir = root_dir


class _SegmentationManifest:
    def __init__(
        self,
        *,
        dataset_id: str,
        images: list[dict[str, object]],
        root_dir: Path,
    ) -> None:
        self.dataset_id = dataset_id
        self.images = images
        self.root_dir = root_dir


class _EmbeddingInputSource:
    def __init__(
        self,
        *,
        dataset_id: str,
        images: list[dict[str, object]],
        manifest_path: Path,
        manifest_kind: str,
    ) -> None:
        self.dataset_id = dataset_id
        self.images = images
        self.manifest_path = manifest_path
        self.manifest_kind = manifest_kind
