import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ironflow_exp.configs import ConfigValidationResult, ConfigValidator, ExperimentConfig
from ironflow_exp.domain import CompatibilityResult, ExperimentContext
from ironflow_exp.pipelines import ExperimentPipeline, ExperimentPipelineResult
from ironflow_exp.services.compatibility_service import CompatibilityService
from ironflow_exp.services.export_service import ExportService, ExportServiceResult


@dataclass(frozen=True, slots=True)
class ExperimentServiceResult:
    context: ExperimentContext
    validation_result: ConfigValidationResult
    compatibility_result: CompatibilityResult
    pipeline_result: ExperimentPipelineResult
    export_result: ExportServiceResult


class ExperimentServiceError(RuntimeError):
    pass


class ExperimentConfigError(ExperimentServiceError):
    def __init__(self, validation_result: ConfigValidationResult) -> None:
        message = '; '.join(
            f'{issue.field}: {issue.message}'
            for issue in validation_result.errors
        )
        super().__init__(message)
        self.validation_result = validation_result


class ExperimentCompatibilityError(ExperimentServiceError):
    def __init__(self, compatibility_result: CompatibilityResult) -> None:
        super().__init__(compatibility_result.message)
        self.compatibility_result = compatibility_result


class ExperimentService:
    def __init__(
        self,
        config_validator: ConfigValidator | None = None,
        compatibility_service: CompatibilityService | None = None,
        pipeline: ExperimentPipeline | None = None,
        export_service: ExportService | None = None,
    ) -> None:
        self.config_validator = config_validator or ConfigValidator()
        self.compatibility_service = compatibility_service or CompatibilityService()
        self.pipeline = pipeline or ExperimentPipeline()
        self.export_service = export_service or ExportService()

    def run_preprocessing(
        self,
        config: ExperimentConfig,
        run_id: str | None = None,
        check_paths: bool = True,
    ) -> ExperimentServiceResult:
        validation_result = self.config_validator.validate(config=config, check_paths=check_paths)
        if not validation_result.is_valid:
            raise ExperimentConfigError(validation_result=validation_result)

        compatibility_result = self.compatibility_service.validate(
            config=config,
            artifact_root=config.export.output_root,
        )
        if not compatibility_result.is_allowed:
            raise ExperimentCompatibilityError(compatibility_result=compatibility_result)

        context = self.create_context(config=config, run_id=run_id)
        Path(context.output_dir).mkdir(parents=True, exist_ok=True)
        pipeline_result = self.pipeline.run_preprocessing(context=context)
        export_result = self.export_service.export_pipeline_result(
            pipeline_result=pipeline_result,
            output_root=context.output_dir,
            schema_version=config.schema_version,
            run_id=context.run_id,
            save_json=config.export.save_json,
            save_csv=config.export.save_csv,
        )
        self._attach_export_paths(context=context, export_result=export_result)

        return ExperimentServiceResult(
            context=context,
            validation_result=validation_result,
            compatibility_result=compatibility_result,
            pipeline_result=pipeline_result,
            export_result=export_result,
        )

    def create_context(
        self,
        config: ExperimentConfig,
        run_id: str | None = None,
    ) -> ExperimentContext:
        resolved_run_id = run_id or self._make_run_id(config=config)
        output_dir = Path(config.export.output_root) / resolved_run_id

        return ExperimentContext(
            run_id=resolved_run_id,
            config=config,
            output_dir=str(output_dir),
        )

    def _attach_export_paths(
        self,
        context: ExperimentContext,
        export_result: ExportServiceResult,
    ) -> None:
        exported_paths = export_result.exported_paths

        if 'dataset_manifest_json' in exported_paths:
            context.dataset_manifest_path = str(exported_paths['dataset_manifest_json'])
        if 'object_manifest_json' in exported_paths:
            context.object_manifest_path = str(exported_paths['object_manifest_json'])
        if 'artifact_manifest_json' in exported_paths:
            context.artifact_manifest_path = str(exported_paths['artifact_manifest_json'])

    def _make_run_id(self, config: ExperimentConfig) -> str:
        timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        name_slug = self._slug(value=config.experiment.name)

        return f'run_{timestamp}_{name_slug}'

    def _slug(self, value: str) -> str:
        normalized = re.sub(r'[^A-Za-z0-9]+', '_', value.strip().lower()).strip('_')

        if not normalized:
            return 'experiment'

        return normalized[:60]
