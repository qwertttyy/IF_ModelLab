from dataclasses import dataclass, field
from pathlib import Path

from ironflow_exp.engine.configs.experiment_config import EngineExperimentConfig


ALLOWED_ENGINE_SCHEMA_VERSIONS = {'0.1'}
ALLOWED_ENGINE_RUNNERS = {'local', 'local_simulator', 'ssh'}
ALLOWED_DATA_VARIANT_KINDS = {
    'original',
    'crop',
    'augmentation',
    'segmentation',
    'segmentation_crop',
    'detector_crop',
    'detection_sequence',
}
ALLOWED_TASK_TYPES = {
    'detection',
    'classification',
    'segmentation',
    'tracking',
    'embedding',
    'augmentation',
    'crop',
    'preprocessing',
}


@dataclass(frozen=True, slots=True)
class EngineConfigValidationIssue:
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class EngineConfigValidationResult:
    is_valid: bool
    errors: list[EngineConfigValidationIssue] = field(default_factory=list)
    warnings: list[EngineConfigValidationIssue] = field(default_factory=list)


class EngineConfigValidator:
    def validate(
        self,
        config: EngineExperimentConfig,
        check_paths: bool = False,
    ) -> EngineConfigValidationResult:
        errors: list[EngineConfigValidationIssue] = []
        warnings: list[EngineConfigValidationIssue] = []

        self._validate_root(config=config, errors=errors)
        self._validate_runtime(config=config, errors=errors)
        self._validate_code(config=config, errors=errors, check_paths=check_paths)
        self._validate_train(config=config, errors=errors)
        self._validate_data_variants(config=config, errors=errors, check_paths=check_paths)
        self._validate_tasks(config=config, errors=errors)
        self._validate_output(config=config, errors=errors, warnings=warnings)
        self._validate_analysis(config=config, errors=errors)
        self._validate_repository(config=config, errors=errors)

        return EngineConfigValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    def _validate_root(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
    ) -> None:
        if config.schema_version not in ALLOWED_ENGINE_SCHEMA_VERSIONS:
            self._add_error(errors=errors, field='schema_version', message='unsupported schema version')

        if not config.experiment.name.strip():
            self._add_error(errors=errors, field='experiment.name', message='experiment name is required')

    def _validate_runtime(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
    ) -> None:
        runtime = config.runtime

        if runtime.runner not in ALLOWED_ENGINE_RUNNERS:
            self._add_error(errors=errors, field='runtime.runner', message='unsupported runner')

        if not runtime.workspace.strip():
            self._add_error(errors=errors, field='runtime.workspace', message='workspace is required')

        if not runtime.experiment_root.strip():
            self._add_error(errors=errors, field='runtime.experiment_root', message='experiment_root is required')

        if not runtime.python_executable.strip():
            self._add_error(errors=errors, field='runtime.python_executable', message='python_executable is required')

        if runtime.runner == 'ssh' and not runtime.remote_python_executable.strip():
            self._add_error(
                errors=errors,
                field='runtime.remote_python_executable',
                message='remote_python_executable is required for ssh runner',
            )

    def _validate_code(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
        check_paths: bool,
    ) -> None:
        code = config.code

        if not code.entrypoint.strip():
            self._add_error(errors=errors, field='code.entrypoint', message='entrypoint is required')
        elif check_paths and not Path(code.entrypoint).exists():
            self._add_error(errors=errors, field='code.entrypoint', message='entrypoint does not exist')

        if code.working_dir is not None and not code.working_dir.strip():
            self._add_error(errors=errors, field='code.working_dir', message='working_dir cannot be blank')
        elif check_paths and code.working_dir is not None and not Path(code.working_dir).exists():
            self._add_error(errors=errors, field='code.working_dir', message='working_dir does not exist')

    def _validate_train(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
    ) -> None:
        train = config.train

        if train.max_seconds is not None and train.max_seconds <= 0:
            self._add_error(errors=errors, field='train.max_seconds', message='max_seconds must be greater than 0')

    def _validate_data_variants(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
        check_paths: bool,
    ) -> None:
        if not config.data_variants:
            self._add_error(errors=errors, field='data_variants', message='at least one data variant is required')
            return

        variant_ids = {variant.id for variant in config.data_variants if variant.id.strip()}
        seen_ids: set[str] = set()
        for index, variant in enumerate(config.data_variants):
            field_prefix = f'data_variants[{index}]'
            if not variant.id.strip():
                self._add_error(errors=errors, field=f'{field_prefix}.id', message='data variant id is required')
            elif variant.id in seen_ids:
                self._add_error(errors=errors, field=f'{field_prefix}.id', message='data variant id must be unique')
            else:
                seen_ids.add(variant.id)

            if variant.kind not in ALLOWED_DATA_VARIANT_KINDS:
                self._add_error(errors=errors, field=f'{field_prefix}.kind', message='unsupported data variant kind')

            if variant.kind != 'original' and not (variant.source or '').strip():
                self._add_error(
                    errors=errors,
                    field=f'{field_prefix}.source',
                    message='derived data variant source is required',
                )
            elif variant.source is not None and variant.source not in variant_ids:
                self._add_error(
                    errors=errors,
                    field=f'{field_prefix}.source',
                    message='data variant source must reference another data variant id',
                )

            if check_paths and variant.path is not None and not Path(variant.path).exists():
                self._add_error(errors=errors, field=f'{field_prefix}.path', message='data variant path does not exist')

    def _validate_tasks(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
    ) -> None:
        variant_ids = {variant.id for variant in config.data_variants}
        task_ids = {task.id for task in config.tasks if task.id.strip()}

        seen_ids: set[str] = set()
        for index, task in enumerate(config.tasks):
            field_prefix = f'tasks[{index}]'
            if not task.id.strip():
                self._add_error(errors=errors, field=f'{field_prefix}.id', message='task id is required')
            elif task.id in seen_ids:
                self._add_error(errors=errors, field=f'{field_prefix}.id', message='task id must be unique')
            else:
                seen_ids.add(task.id)

            if task.task_type not in ALLOWED_TASK_TYPES:
                self._add_error(errors=errors, field=f'{field_prefix}.task_type', message='unsupported task type')

            if task.input_variant not in variant_ids:
                self._add_error(
                    errors=errors,
                    field=f'{field_prefix}.input_variant',
                    message='input_variant must reference a data variant id',
                )

            if task.enabled and task.task_type in {'detection', 'classification', 'segmentation', 'tracking', 'embedding'}:
                if task.model_id is None or not task.model_id.strip():
                    self._add_error(errors=errors, field=f'{field_prefix}.model_id', message='model_id is required')
                if task.adapter is None or not task.adapter.strip():
                    self._add_error(errors=errors, field=f'{field_prefix}.adapter', message='adapter is required')

            for dependency_id in task.depends_on:
                if dependency_id not in task_ids:
                    self._add_error(
                        errors=errors,
                        field=f'{field_prefix}.depends_on',
                        message=f'unknown task dependency: {dependency_id}',
                    )

    def _validate_output(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
        warnings: list[EngineConfigValidationIssue],
    ) -> None:
        output = config.output
        required_file_fields = {
            'output.log_file': output.log_file,
            'output.metrics_file': output.metrics_file,
            'output.checkpoint_file': output.checkpoint_file,
            'output.status_file': output.status_file,
            'output.summary_file': output.summary_file,
        }

        for field, value in required_file_fields.items():
            if not value.strip():
                self._add_error(errors=errors, field=field, message='output file name is required')

        save_flags = {
            'output.save_json': output.save_json,
            'output.save_csv': output.save_csv,
            'output.save_summary': output.save_summary,
            'output.save_checkpoints': output.save_checkpoints,
            'output.save_previews': output.save_previews,
        }
        for field, value in save_flags.items():
            if not isinstance(value, bool):
                self._add_error(errors=errors, field=field, message='output save option must be boolean')

        if output.save_json is False:
            self._add_error(
                errors=errors,
                field='output.save_json',
                message='save_json cannot be disabled because JSON artifacts are the canonical engine output',
            )

        if output.artifact_collection_mode not in {'full', 'light', 'metrics_only'}:
            self._add_error(
                errors=errors,
                field='output.artifact_collection_mode',
                message='artifact_collection_mode must be one of: full, light, metrics_only',
            )

        if output.metrics_file not in output.collect_patterns:
            warnings.append(
                EngineConfigValidationIssue(
                    field='output.collect_patterns',
                    message='metrics_file is not included in collect_patterns',
                ),
            )

        if output.log_file not in output.collect_patterns:
            warnings.append(
                EngineConfigValidationIssue(
                    field='output.collect_patterns',
                    message='log_file is not included in collect_patterns',
                ),
            )

    def _validate_analysis(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
    ) -> None:
        if not config.analysis.primary_metric.strip():
            self._add_error(errors=errors, field='analysis.primary_metric', message='primary_metric is required')

    def _validate_repository(
        self,
        config: EngineExperimentConfig,
        errors: list[EngineConfigValidationIssue],
    ) -> None:
        if not config.repository.sqlite_path.strip():
            self._add_error(errors=errors, field='repository.sqlite_path', message='sqlite_path is required')

    def _add_error(
        self,
        errors: list[EngineConfigValidationIssue],
        field: str,
        message: str,
    ) -> None:
        errors.append(EngineConfigValidationIssue(field=field, message=message))
