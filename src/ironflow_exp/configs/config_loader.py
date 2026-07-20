import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# *************************
# External Library
#   - PyYAML : ver 6.0.3
# *************************
import yaml

from ironflow_exp.configs.dataset_config import DatasetConfig, SplitConfig
from ironflow_exp.configs.evaluation_config import EvaluationConfig, LatencyConfig, TaskMetricConfig
from ironflow_exp.configs.experiment_config import ExperimentConfig, ExperimentMetaConfig
from ironflow_exp.configs.export_config import ExportConfig
from ironflow_exp.configs.model_config import ModelConfig, ModelGroupConfig, PredictConfig, TrainConfig
from ironflow_exp.configs.preprocessing_config import (
    AugmentationConfig,
    CropConfig,
    PreprocessingConfig,
    SegmentationPreprocessConfig,
    TaskInputConfig,
)
from ironflow_exp.configs.runtime_config import RuntimeConfig


class ConfigLoader:
    def load_file(self, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        data = self.load_dict(path=config_path)

        config = self.from_dict(data=data)

        return self._resolve_file_relative_paths(config=config, base_dir=config_path.parent)

    def load_dict(self, path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        suffix = config_path.suffix.lower()

        with config_path.open(mode='r', encoding='utf-8') as file:
            if suffix == '.json':
                loaded = json.load(file)
            elif suffix in {'.yaml', '.yml'}:
                loaded = yaml.safe_load(file)
            else:
                raise ValueError(f'unsupported config extension: {suffix}')

        if not isinstance(loaded, dict):
            raise ValueError('config file must contain a mapping object')

        return loaded

    def from_dict(self, data: dict[str, Any]) -> ExperimentConfig:
        return ExperimentConfig(
            schema_version=str(data.get('schema_version', '0.1')),
            experiment=self._build_experiment_meta(data=data.get('experiment', {})),
            runtime=self._build_runtime(data=data.get('runtime', {})),
            dataset=self._build_dataset(data=data.get('dataset', {})),
            preprocessing=self._build_preprocessing(data=data.get('preprocessing', {})),
            models=self._build_models(data=data.get('models', {})),
            evaluation=self._build_evaluation(data=data.get('evaluation', {})),
            export=self._build_export(data=data.get('export', {})),
        )

    def to_dict(self, config: ExperimentConfig) -> dict[str, Any]:
        return asdict(config)

    def _build_experiment_meta(self, data: dict[str, Any]) -> ExperimentMetaConfig:
        return ExperimentMetaConfig(
            name=str(data.get('name', '')),
            description=str(data.get('description', '')),
            seed=int(data.get('seed', 42)),
            tags=list(data.get('tags', [])),
        )

    def _build_runtime(self, data: dict[str, Any]) -> RuntimeConfig:
        return RuntimeConfig(
            runner=str(data.get('runner', 'local')),
            device=str(data.get('device', 'cuda')),
            precision=str(data.get('precision', 'fp32')),
            num_workers=int(data.get('num_workers', 4)),
        )

    def _build_dataset(self, data: dict[str, Any]) -> DatasetConfig:
        return DatasetConfig(
            dataset_id=str(data.get('dataset_id', '')),
            source_root=str(data.get('source_root', '')),
            dataset_format=str(data.get('dataset_format', '')),
            label_format=str(data.get('label_format', '')),
            split=self._build_split(data=data.get('split', {})),
            class_map=self._build_class_map(data=data.get('class_map', {})),
        )

    def _build_split(self, data: dict[str, Any]) -> SplitConfig:
        return SplitConfig(
            strategy=str(data.get('strategy', 'existing_or_create')),
            group_key=self._optional_str(value=data.get('group_key')),
            train_ratio=float(data.get('train_ratio', 0.7)),
            val_ratio=float(data.get('val_ratio', 0.2)),
            test_ratio=float(data.get('test_ratio', 0.1)),
            seed=int(data.get('seed', 42)),
        )

    def _build_class_map(self, data: dict[Any, Any]) -> dict[int, str]:
        return {
            int(class_id): str(class_name)
            for class_id, class_name in data.items()
        }

    def _build_preprocessing(self, data: dict[str, Any]) -> PreprocessingConfig:
        return PreprocessingConfig(
            mode=str(data.get('mode', 'original')),
            task_inputs=self._build_task_inputs(data=data.get('task_inputs', {})),
            crop=self._build_crop(data=data.get('crop', {})),
            augmentation=self._build_augmentation(data=data.get('augmentation', {})),
            segmentation=self._build_segmentation_preprocess(data=data.get('segmentation', {})),
        )

    def _build_task_inputs(self, data: dict[str, Any]) -> TaskInputConfig:
        return TaskInputConfig(
            detection=str(data.get('detection', 'none')),
            classification=str(data.get('classification', 'none')),
            segmentation=str(data.get('segmentation', 'none')),
            embedding=str(data.get('embedding', 'none')),
        )

    def _build_crop(self, data: dict[str, Any]) -> CropConfig:
        return CropConfig(
            enabled=self._bool(value=data.get('enabled', False)),
            source=str(data.get('source', 'gt_bbox')),
            padding_ratio=float(data.get('padding_ratio', 0.08)),
        )

    def _build_augmentation(self, data: dict[str, Any]) -> AugmentationConfig:
        return AugmentationConfig(
            enabled=self._bool(value=data.get('enabled', False)),
            target_split=str(data.get('target_split', 'train')),
            policy_id=str(data.get('policy_id', 'none')),
        )

    def _build_segmentation_preprocess(self, data: dict[str, Any]) -> SegmentationPreprocessConfig:
        return SegmentationPreprocessConfig(
            enabled=self._bool(value=data.get('enabled', False)),
            source=str(data.get('source', 'none')),
            output_mode=str(data.get('output_mode', 'mask')),
        )

    def _build_models(self, data: dict[str, Any]) -> ModelGroupConfig:
        return ModelGroupConfig(
            detection=self._build_model(data=data.get('detection', {})),
            classification=self._build_model(data=data.get('classification', {})),
            segmentation=self._build_model(data=data.get('segmentation', {})),
            embedding=self._build_model(data=data.get('embedding', {})),
        )

    def _build_model(self, data: dict[str, Any]) -> ModelConfig:
        return ModelConfig(
            enabled=self._bool(value=data.get('enabled', False)),
            model_id=self._optional_str(value=data.get('model_id')),
            adapter=self._optional_str(value=data.get('adapter')),
            checkpoint=self._optional_str(value=data.get('checkpoint')),
            pretrained=self._bool(value=data.get('pretrained', True)),
            output_dim=self._optional_int(value=data.get('output_dim')),
            train=self._build_train(data=data.get('train', {})),
            predict=self._build_predict(data=data.get('predict', {})),
        )

    def _build_train(self, data: dict[str, Any]) -> TrainConfig:
        return TrainConfig(
            enabled=self._bool(value=data.get('enabled', False)),
            epochs=self._optional_int(value=data.get('epochs')),
            image_size=self._optional_int(value=data.get('image_size')),
            batch_size=self._optional_int(value=data.get('batch_size')),
            learning_rate=self._optional_float(value=data.get('learning_rate')),
        )

    def _build_predict(self, data: dict[str, Any]) -> PredictConfig:
        return PredictConfig(
            enabled=self._bool(value=data.get('enabled', True)),
            confidence_threshold=self._optional_float(value=data.get('confidence_threshold')),
            iou_threshold=self._optional_float(value=data.get('iou_threshold')),
            top_k=self._optional_int(value=data.get('top_k')),
        )

    def _build_evaluation(self, data: dict[str, Any]) -> EvaluationConfig:
        return EvaluationConfig(
            detection=self._build_task_metric(data=data.get('detection', {})),
            classification=self._build_task_metric(data=data.get('classification', {})),
            segmentation=self._build_task_metric(data=data.get('segmentation', {})),
            latency=self._build_latency(data=data.get('latency', {})),
        )

    def _build_task_metric(self, data: dict[str, Any]) -> TaskMetricConfig:
        return TaskMetricConfig(
            enabled=self._bool(value=data.get('enabled', False)),
            metrics=list(data.get('metrics', [])),
        )

    def _build_latency(self, data: dict[str, Any]) -> LatencyConfig:
        return LatencyConfig(
            enabled=self._bool(value=data.get('enabled', True)),
            warmup_runs=int(data.get('warmup_runs', 3)),
        )

    def _build_export(self, data: dict[str, Any]) -> ExportConfig:
        return ExportConfig(
            output_root=str(data.get('output_root', '')),
            save_json=self._bool(value=data.get('save_json', True)),
            save_csv=self._bool(value=data.get('save_csv', True)),
            save_report=self._bool(value=data.get('save_report', True)),
            save_bundle=self._bool(value=data.get('save_bundle', True)),
            save_visualizations=self._bool(value=data.get('save_visualizations', True)),
        )

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None

        return int(value)

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None

        return float(value)

    def _bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'true', '1', 'yes', 'y'}:
                return True
            if normalized in {'false', '0', 'no', 'n'}:
                return False

        return bool(value)

    def _resolve_file_relative_paths(
        self,
        config: ExperimentConfig,
        base_dir: Path,
    ) -> ExperimentConfig:
        source_root = self._resolve_path(value=config.dataset.source_root, base_dir=base_dir)
        if source_root == config.dataset.source_root:
            return config

        return replace(config, dataset=replace(config.dataset, source_root=source_root))

    def _resolve_path(
        self,
        value: str,
        base_dir: Path,
    ) -> str:
        if not value.strip():
            return value

        path = Path(value)
        if path.is_absolute():
            return str(path)

        return str((base_dir / path).resolve())
