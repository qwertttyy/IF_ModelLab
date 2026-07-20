from pathlib import Path

from ironflow_exp.core.enums import JobStatus, TaskInputSource
from ironflow_exp.datasets import GtObjectManifestBuilder, ManifestBuilder, SplitManager
from ironflow_exp.domain import ArtifactRecord, ExperimentContext
from ironflow_exp.pipelines.result import ExperimentPipelineResult, PipelineExecutionError, PipelineStageResult
from ironflow_exp.preprocessing import GtCropGenerator


class ExperimentPipeline:
    def __init__(
        self,
        manifest_builder: ManifestBuilder | None = None,
        split_manager: SplitManager | None = None,
        object_manifest_builder: GtObjectManifestBuilder | None = None,
        crop_generator: GtCropGenerator | None = None,
    ) -> None:
        self.manifest_builder = manifest_builder or ManifestBuilder()
        self.split_manager = split_manager or SplitManager()
        self.object_manifest_builder = object_manifest_builder or GtObjectManifestBuilder()
        self.crop_generator = crop_generator or GtCropGenerator()

    def run_preprocessing(self, context: ExperimentContext) -> ExperimentPipelineResult:
        config = context.config
        stages: list[PipelineStageResult] = []
        artifacts: list[ArtifactRecord] = []

        dataset_manifest = self.manifest_builder.build(
            dataset_config=config.dataset,
            run_id=context.run_id,
            schema_version=config.schema_version,
        )
        stages.append(
            PipelineStageResult(
                name='dataset_manifest',
                status=JobStatus.SUCCESS.value,
                metadata=dataset_manifest.summary,
            ),
        )

        split_result = self.split_manager.validate_no_leakage(samples=dataset_manifest.samples)
        if not split_result.is_valid:
            stages.append(
                PipelineStageResult(
                    name='split_validation',
                    status=JobStatus.FAILED.value,
                    message='dataset split leakage detected',
                    metadata={
                        'issues': [
                            {
                                'code': issue.code,
                                'message': issue.message,
                                'sample_ids': issue.sample_ids,
                            }
                            for issue in split_result.issues
                        ],
                    },
                ),
            )
            raise PipelineExecutionError(message='dataset split leakage detected', stages=stages)

        stages.append(
            PipelineStageResult(
                name='split_validation',
                status=JobStatus.SUCCESS.value,
                metadata={'num_issues': len(split_result.issues)},
            ),
        )

        if not self._requires_gt_crop(context=context):
            stages.append(
                PipelineStageResult(
                    name='gt_crop',
                    status=JobStatus.SKIPPED.value,
                    message='gt crop is not required by config',
                ),
            )
            return ExperimentPipelineResult(
                dataset_manifest=dataset_manifest,
                stages=stages,
            )

        object_manifest = self.object_manifest_builder.build(
            dataset_manifest=dataset_manifest,
            source_root=config.dataset.source_root,
            schema_version=config.schema_version,
        )
        stages.append(
            PipelineStageResult(
                name='gt_object_manifest',
                status=JobStatus.SUCCESS.value,
                metadata=object_manifest.summary,
            ),
        )

        crop_result = self.crop_generator.generate(
            dataset_manifest=dataset_manifest,
            object_manifest=object_manifest,
            source_root=config.dataset.source_root,
            output_root=Path(context.output_dir),
            padding_ratio=config.preprocessing.crop.padding_ratio,
        )
        artifacts.extend(crop_result.artifacts)
        stages.append(
            PipelineStageResult(
                name='gt_crop',
                status=JobStatus.SUCCESS.value,
                metadata={
                    'num_crops': len(crop_result.artifacts),
                },
            ),
        )

        return ExperimentPipelineResult(
            dataset_manifest=dataset_manifest,
            object_manifest=crop_result.object_manifest,
            artifacts=artifacts,
            stages=stages,
        )

    def _requires_gt_crop(self, context: ExperimentContext) -> bool:
        preprocessing = context.config.preprocessing

        return (
            preprocessing.crop.enabled
            and preprocessing.crop.source == 'gt_bbox'
            and preprocessing.task_inputs.classification == TaskInputSource.GT_CROP.value
        )
