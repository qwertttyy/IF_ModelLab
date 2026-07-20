import json
import time
from dataclasses import replace
from pathlib import Path

from ironflow_exp.engine.core import TaskArtifactValidator, TaskOrchestrationPlan, TaskPlanCompatibilityValidator
from ironflow_exp.engine.domain import TaskExecutionRecord
from ironflow_exp.engine.tasks.artifact_writer import TaskArtifactWriter
from ironflow_exp.engine.tasks.base import (
    TaskDependencyContext,
    TaskAdapterResult,
    TaskExecutionContext,
    TaskExecutionPlanResult,
    TaskExecutionResult,
)
from ironflow_exp.engine.tasks.registry import TaskAdapterRegistry


class AdapterTaskExecutor:
    _PREDICTION_EXECUTION_MODES = frozenset({'predict', 'inference', 'inference_smoke'})
    _IMAGE_EXTENSIONS = frozenset({'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'})

    def __init__(
        self,
        registry: TaskAdapterRegistry,
        artifact_writer: TaskArtifactWriter | None = None,
        validator: TaskArtifactValidator | None = None,
        compatibility_validator: TaskPlanCompatibilityValidator | None = None,
    ) -> None:
        self.registry = registry
        self.artifact_writer = artifact_writer or TaskArtifactWriter()
        self.validator = validator or TaskArtifactValidator()
        self.compatibility_validator = compatibility_validator or TaskPlanCompatibilityValidator()

    def execute_plan(self, plan: TaskOrchestrationPlan) -> TaskExecutionPlanResult:
        compatibility = self.compatibility_validator.validate(plan)
        if not compatibility.is_valid:
            return TaskExecutionPlanResult(
                experiment_id=plan.experiment_id,
                success=False,
                task_results=[],
                message='task plan compatibility validation failed: ' + '; '.join(compatibility.error_messages()),
            )

        results: list[TaskExecutionResult] = []
        completed_by_id: dict[str, TaskExecutionResult] = {}

        for record in plan.records:
            dependency_contexts = self._dependency_contexts(record=record, completed_by_id=completed_by_id)
            result = self.execute_record(record=record, dependency_results=dependency_contexts)
            results.append(result)
            if not result.success:
                return TaskExecutionPlanResult(
                    experiment_id=plan.experiment_id,
                    success=False,
                    task_results=results,
                    message=f'task failed: {record.task_id}',
                )
            completed_by_id[record.task_id] = result

        return TaskExecutionPlanResult(
            experiment_id=plan.experiment_id,
            success=True,
            task_results=results,
            message='task execution completed',
        )

    def execute_record(
        self,
        record: TaskExecutionRecord,
        dependency_results: dict[str, TaskDependencyContext] | None = None,
    ) -> TaskExecutionResult:
        if record.result_dir is None:
            raise ValueError(f'task {record.task_id} has no result_dir')
        if record.adapter is None:
            raise ValueError(f'task {record.task_id} has no adapter')

        started_at = time.perf_counter()
        result_dir = Path(record.result_dir)
        dependency_contexts = dependency_results or {}
        missing_dependency_ids = sorted(set(record.depends_on) - set(dependency_contexts))
        if missing_dependency_ids:
            raise ValueError(
                f'task {record.task_id} is missing dependency results: {", ".join(missing_dependency_ids)}',
            )
        adapter = self.registry.create(task_type=record.task_type, adapter_key=record.adapter)
        context = TaskExecutionContext(
            record=record,
            result_dir=result_dir,
            dependency_results=dependency_contexts,
        )
        adapter_started_at = time.perf_counter()
        try:
            adapter_result = adapter.run(context=context)
        except Exception as error:
            adapter_result = TaskAdapterResult(
                success=False,
                status='failed',
                message=f'task adapter raised {type(error).__name__}: {error}',
                metadata={
                    'adapter': record.adapter,
                    'failure_type': 'task_adapter_exception',
                    'exception_type': type(error).__name__,
                    'exception_module': type(error).__module__,
                    'error': str(error),
                },
            )
        adapter_elapsed_seconds = max(time.perf_counter() - adapter_started_at, 0.0)
        adapter_result = self._with_latency_metrics(
            record=record,
            adapter_result=adapter_result,
            adapter_elapsed_seconds=adapter_elapsed_seconds,
        )
        completed_record = replace(record, status=adapter_result.status)
        elapsed_seconds = max(time.perf_counter() - started_at, 0.0)
        timing_rows = self._timing_rows(
            record=record,
            adapter_result=adapter_result,
            stage_elapsed_seconds=adapter_elapsed_seconds,
            task_total_seconds=elapsed_seconds,
        )
        artifacts = self.artifact_writer.write(
            result_dir=result_dir,
            record=completed_record,
            adapter_result=adapter_result,
            elapsed_seconds=elapsed_seconds,
            timing_rows=timing_rows,
        )
        validation = self.validator.validate_task_dir(task_dir=result_dir)
        success = adapter_result.success and validation.is_valid
        status = adapter_result.status if validation.is_valid else 'failed'

        return TaskExecutionResult(
            record=replace(completed_record, status=status),
            success=success,
            status=status,
            message=adapter_result.message if validation.is_valid else 'task artifact validation failed',
            metrics=adapter_result.metrics,
            artifacts=artifacts,
            elapsed_seconds=elapsed_seconds,
            timings=timing_rows,
            validation_errors=validation.error_messages(),
        )

    def _timing_rows(
        self,
        *,
        record: TaskExecutionRecord,
        adapter_result: TaskAdapterResult,
        stage_elapsed_seconds: float,
        task_total_seconds: float,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen_stages: set[str] = set()
        for row in adapter_result.timings:
            stage = str(row.get('stage') or '').strip()
            if not stage:
                continue
            rows.append(row)
            seen_stages.add(stage)

        representative_stage = self._representative_timing_stage(record=record)
        if representative_stage not in seen_stages:
            rows.append(
                {
                    'stage': representative_stage,
                    'elapsed_seconds': stage_elapsed_seconds,
                },
            )
        rows.append(
            {
                'stage': 'task_total',
                'elapsed_seconds': task_total_seconds,
            },
        )

        return rows

    def _representative_timing_stage(self, *, record: TaskExecutionRecord) -> str:
        execution_mode = str(record.params.get('execution_mode') or '').strip().lower()
        if execution_mode == 'train':
            return 'train'
        if execution_mode in {'predict', 'inference', 'inference_smoke'}:
            return 'predict'
        if record.task_type == 'preprocessing':
            return 'preprocessing'

        return record.task_type

    def _with_latency_metrics(
        self,
        *,
        record: TaskExecutionRecord,
        adapter_result: TaskAdapterResult,
        adapter_elapsed_seconds: float,
    ) -> TaskAdapterResult:
        if not adapter_result.success or not self._is_prediction_record(record=record):
            return adapter_result
        if not adapter_result.metrics:
            return adapter_result

        image_count = self._latency_image_count(record=record, adapter_result=adapter_result)
        if image_count <= 0:
            return adapter_result

        metrics = [dict(row) for row in adapter_result.metrics]
        target_row = metrics[-1]
        if self._is_blank_metric_value(target_row.get('latency_ms_per_image')):
            target_row['latency_ms_per_image'] = self._format_metric_float(
                adapter_elapsed_seconds * 1000.0 / image_count,
            )
        p95_latency_ms = self._p95_latency_ms(adapter_result=adapter_result)
        if p95_latency_ms is not None and self._is_blank_metric_value(target_row.get('p95_latency_ms')):
            target_row['p95_latency_ms'] = self._format_metric_float(p95_latency_ms)

        metadata = {
            **adapter_result.metadata,
            'latency_image_count': image_count,
            'latency_elapsed_seconds': adapter_elapsed_seconds,
            'latency_ms_per_image_source': 'task_elapsed_seconds/image_count',
        }
        if p95_latency_ms is not None:
            metadata['p95_latency_ms_source'] = 'per_image_latency_ms'

        return replace(adapter_result, metrics=metrics, metadata=metadata)

    def _is_prediction_record(self, *, record: TaskExecutionRecord) -> bool:
        execution_mode = str(record.params.get('execution_mode') or '').strip().lower()
        return execution_mode in self._PREDICTION_EXECUTION_MODES

    def _latency_image_count(self, *, record: TaskExecutionRecord, adapter_result: TaskAdapterResult) -> int:
        for key in ('image_count', 'prediction_image_count', 'inference_image_count', 'latency_image_count'):
            count = self._positive_int(adapter_result.metadata.get(key))
            if count is not None:
                return count

        count = self._input_split_image_count(record=record)
        if count is not None:
            return count

        image_ids: set[str] = set()
        for prediction in adapter_result.predictions:
            if not isinstance(prediction, dict):
                continue
            for key in ('image_id', 'sample_id'):
                record_value = prediction.get(key)
                if isinstance(record_value, str) and record_value:
                    image_ids.add(record_value)
        if image_ids:
            return len(image_ids)

        for row in reversed(adapter_result.metrics):
            count = self._positive_int(row.get('num_images') or row.get('image_count'))
            if count is not None:
                return count

        return 0

    def _input_split_image_count(self, *, record: TaskExecutionRecord) -> int | None:
        if record.input_variant_path is None:
            return None
        root = Path(record.input_variant_path).expanduser()
        split = str(record.params.get('prediction_split') or 'test').strip()
        candidates: list[Path] = []
        if split:
            candidates.extend([
                root / 'images' / split,
                root / split,
                root / 'crops' / split,
                root / 'classification' / split,
                root / 'detection' / 'images' / split,
            ])
        candidates.extend([root / 'images', root])

        for candidate in candidates:
            if not candidate.exists() or not candidate.is_dir():
                continue
            count = self._count_images(candidate)
            if count > 0:
                return count

        return None

    def _count_images(self, root: Path) -> int:
        return sum(
            1
            for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in self._IMAGE_EXTENSIONS
        )

    def _p95_latency_ms(self, *, adapter_result: TaskAdapterResult) -> float | None:
        raw_values = adapter_result.metadata.get('per_image_latency_ms')
        if not isinstance(raw_values, list):
            return None
        values = sorted(
            value
            for item in raw_values
            if (value := self._positive_float(item)) is not None
        )
        if not values:
            return None
        index = max(0, min(len(values) - 1, int((len(values) - 1) * 0.95)))
        return values[index]

    def _positive_int(self, value: object) -> int | None:
        try:
            parsed = int(float(str(value)))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _positive_float(self, value: object) -> float | None:
        try:
            parsed = float(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _is_blank_metric_value(self, value: object) -> bool:
        return value is None or str(value).strip() == ''

    def _format_metric_float(self, value: float) -> str:
        return f'{value:.6f}'

    def _dependency_contexts(
        self,
        *,
        record: TaskExecutionRecord,
        completed_by_id: dict[str, TaskExecutionResult],
    ) -> dict[str, TaskDependencyContext]:
        contexts: dict[str, TaskDependencyContext] = {}
        for dependency_id in record.depends_on:
            if dependency_id not in completed_by_id:
                raise ValueError(f'task {record.task_id} depends on unfinished task {dependency_id}')
            dependency_result = completed_by_id[dependency_id]
            if not dependency_result.success:
                raise ValueError(f'task {record.task_id} depends on failed task {dependency_id}')
            dependency_result_dir = dependency_result.record.result_dir
            if dependency_result_dir is None:
                raise ValueError(f'task {record.task_id} dependency {dependency_id} has no result_dir')
            contexts[dependency_id] = TaskDependencyContext(
                task_id=dependency_result.record.task_id,
                task_type=dependency_result.record.task_type,
                status=dependency_result.status,
                result_dir=Path(dependency_result_dir),
                artifacts=dependency_result.artifacts,
            )

        return contexts

    def write_plan_result(
        self,
        result: TaskExecutionPlanResult,
        result_dir: str | Path,
    ) -> Path:
        output_path = Path(result_dir) / 'task_results.json'
        output_path.write_text(
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        return output_path
