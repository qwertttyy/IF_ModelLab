import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from ironflow_exp.engine.configs.experiment_config import (
    EngineAnalysisConfig,
    EngineCodeConfig,
    EngineDataVariantConfig,
    EngineExperimentConfig,
    EngineExperimentMetaConfig,
    EngineTaskConfig,
    EngineOutputConfig,
    EngineRepositoryConfig,
    EngineRuntimeConfig,
    EngineTrainConfig,
)


class EngineConfigLoader:
    def load_file(self, path: str | Path) -> EngineExperimentConfig:
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
                raise ValueError(f'unsupported engine config extension: {suffix}')

        if not isinstance(loaded, dict):
            raise ValueError('engine config file must contain a mapping object')

        return loaded

    def from_dict(self, data: dict[str, Any]) -> EngineExperimentConfig:
        return EngineExperimentConfig(
            schema_version=str(data.get('schema_version', '0.1')),
            experiment=self._build_experiment(data=data.get('experiment', {})),
            runtime=self._build_runtime(data=data.get('runtime', {})),
            code=self._build_code(data=data.get('code', {})),
            train=self._build_train(data=data.get('train', {})),
            data_variants=self._build_data_variants(data=data.get('data_variants')),
            tasks=self._build_tasks(data=data.get('tasks', [])),
            output=self._build_output(data=data.get('output', {})),
            analysis=self._build_analysis(data=data.get('analysis', {})),
            repository=self._build_repository(data=data.get('repository', {})),
        )

    def to_dict(self, config: EngineExperimentConfig) -> dict[str, Any]:
        return asdict(config)

    def _build_experiment(self, data: dict[str, Any]) -> EngineExperimentMetaConfig:
        return EngineExperimentMetaConfig(
            name=str(data.get('name', '')),
            description=str(data.get('description', '')),
            tags=[str(tag) for tag in data.get('tags', [])],
        )

    def _build_runtime(self, data: dict[str, Any]) -> EngineRuntimeConfig:
        return EngineRuntimeConfig(
            runner=str(data.get('runner', 'local_simulator')),
            workspace=str(data.get('workspace', 'runs/workspaces')),
            experiment_root=str(data.get('experiment_root', 'runs/experiments')),
            python_executable=str(data.get('python_executable', 'python')),
            remote_python_executable=str(data.get('remote_python_executable', 'python')),
            container_mode=str(data.get('container_mode', 'native_python')),
            container_image=str(data.get('container_image', '')),
        )

    def _build_code(self, data: dict[str, Any]) -> EngineCodeConfig:
        return EngineCodeConfig(
            entrypoint=str(data.get('entrypoint', '')),
            working_dir=self._optional_str(value=data.get('working_dir')),
            args=[str(arg) for arg in data.get('args', [])],
            package_include=[str(pattern) for pattern in data.get('package_include', [])],
            package_exclude=[str(pattern) for pattern in data.get('package_exclude', [])],
        )

    def _build_train(self, data: dict[str, Any]) -> EngineTrainConfig:
        env_data = data.get('env', {})
        return EngineTrainConfig(
            args=[str(arg) for arg in data.get('args', [])],
            env={
                str(key): str(value)
                for key, value in env_data.items()
            },
            max_seconds=self._optional_int(value=data.get('max_seconds')),
            fail_fast=self._bool(value=data.get('fail_fast', True)),
        )

    def _build_data_variants(self, data: Any) -> list[EngineDataVariantConfig]:
        if data is None:
            return [EngineDataVariantConfig(id='original', kind='original')]
        if not isinstance(data, list):
            raise ValueError('data_variants must be a list')

        variants: list[EngineDataVariantConfig] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError('data_variants items must be mapping objects')
            variants.append(
                EngineDataVariantConfig(
                    id=str(item.get('id', '')),
                    kind=str(item.get('kind', 'original')),
                    source=self._optional_str(value=item.get('source')),
                    path=self._optional_str(value=item.get('path')),
                    enabled=self._bool(value=item.get('enabled', True)),
                    params=self._object_dict(value=item.get('params', {})),
                ),
            )

        return variants

    def _build_tasks(self, data: Any) -> list[EngineTaskConfig]:
        if data is None:
            return []
        if not isinstance(data, list):
            raise ValueError('tasks must be a list')

        tasks: list[EngineTaskConfig] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError('tasks items must be mapping objects')
            tasks.append(
                EngineTaskConfig(
                    id=str(item.get('id', '')),
                    task_type=str(item.get('task_type', '')),
                    input_variant=str(item.get('input_variant', '')),
                    model_id=self._optional_str(value=item.get('model_id')),
                    adapter=self._optional_str(value=item.get('adapter')),
                    enabled=self._bool(value=item.get('enabled', True)),
                    depends_on=[str(task_id) for task_id in item.get('depends_on', [])],
                    params=self._object_dict(value=item.get('params', {})),
                ),
            )

        return tasks

    def _build_output(self, data: dict[str, Any]) -> EngineOutputConfig:
        default_output = EngineOutputConfig()
        return EngineOutputConfig(
            log_file=str(data.get('log_file', default_output.log_file)),
            metrics_file=str(data.get('metrics_file', default_output.metrics_file)),
            checkpoint_file=str(data.get('checkpoint_file', default_output.checkpoint_file)),
            status_file=str(data.get('status_file', default_output.status_file)),
            summary_file=str(data.get('summary_file', default_output.summary_file)),
            save_json=self._bool(value=data.get('save_json', default_output.save_json)),
            save_csv=self._bool(value=data.get('save_csv', default_output.save_csv)),
            save_summary=self._bool(value=data.get('save_summary', default_output.save_summary)),
            save_checkpoints=self._bool(value=data.get('save_checkpoints', default_output.save_checkpoints)),
            save_previews=self._bool(value=data.get('save_previews', default_output.save_previews)),
            artifact_collection_mode=str(
                data.get('artifact_collection_mode', default_output.artifact_collection_mode),
            ),
            collect_patterns=[str(pattern) for pattern in data.get('collect_patterns', default_output.collect_patterns)],
        )

    def _build_analysis(self, data: dict[str, Any]) -> EngineAnalysisConfig:
        return EngineAnalysisConfig(
            primary_metric=str(data.get('primary_metric', 'accuracy')),
            higher_is_better=self._bool(value=data.get('higher_is_better', True)),
            failure_rules_enabled=self._bool(value=data.get('failure_rules_enabled', True)),
        )

    def _build_repository(self, data: dict[str, Any]) -> EngineRepositoryConfig:
        return EngineRepositoryConfig(
            sqlite_path=str(data.get('sqlite_path', 'runs/ironflow_experiments.sqlite3')),
        )

    def _resolve_file_relative_paths(
        self,
        config: EngineExperimentConfig,
        base_dir: Path,
    ) -> EngineExperimentConfig:
        runtime = replace(
            config.runtime,
            workspace=self._resolve_path(value=config.runtime.workspace, base_dir=base_dir),
            experiment_root=self._resolve_path(value=config.runtime.experiment_root, base_dir=base_dir),
        )
        code = replace(
            config.code,
            entrypoint=self._resolve_path(value=config.code.entrypoint, base_dir=base_dir),
            working_dir=self._resolve_optional_path(value=config.code.working_dir, base_dir=base_dir),
        )
        repository = replace(
            config.repository,
            sqlite_path=self._resolve_path(value=config.repository.sqlite_path, base_dir=base_dir),
        )
        data_variants = [
            replace(
                variant,
                path=self._resolve_optional_path(value=variant.path, base_dir=base_dir),
            )
            for variant in config.data_variants
        ]

        return replace(
            config,
            runtime=runtime,
            code=code,
            repository=repository,
            data_variants=data_variants,
        )

    def _resolve_path(
        self,
        value: str,
        base_dir: Path,
    ) -> str:
        if not value.strip():
            return value
        if self._is_posix_absolute_path(value=value):
            return value

        path = Path(value)
        if path.is_absolute():
            return str(path)

        return str((base_dir / path).resolve())

    def _is_posix_absolute_path(self, *, value: str) -> bool:
        normalized = value.replace('\\', '/')
        return normalized.startswith('/') and not normalized.startswith('//')

    def _resolve_optional_path(
        self,
        value: str | None,
        base_dir: Path,
    ) -> str | None:
        if value is None:
            return None

        return self._resolve_path(value=value, base_dir=base_dir)

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None

        return int(value)

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

    def _object_dict(self, value: Any) -> dict[str, object]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('params must be a mapping object')

        return {
            str(key): item
            for key, item in value.items()
        }
