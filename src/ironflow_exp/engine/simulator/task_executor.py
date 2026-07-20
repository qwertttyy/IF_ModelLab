import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ironflow_exp.engine.core import TaskOrchestrationPlan
from ironflow_exp.engine.domain import TaskExecutionRecord
from ironflow_exp.engine.tasks import TaskAdapterResult, TaskArtifactWriter


@dataclass(frozen=True, slots=True)
class MockTaskExecutionResult:
    record: TaskExecutionRecord
    success: bool
    status: str
    message: str
    artifacts: list[dict[str, object]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data['record'] = self.record.to_dict()

        return data


@dataclass(frozen=True, slots=True)
class MockTaskExecutionPlanResult:
    experiment_id: str
    success: bool
    task_results: list[MockTaskExecutionResult] = field(default_factory=list)
    message: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'experiment_id': self.experiment_id,
            'success': self.success,
            'message': self.message,
            'task_results': [
                result.to_dict()
                for result in self.task_results
            ],
        }


class LocalMockTaskExecutor:
    def __init__(self, artifact_writer: TaskArtifactWriter | None = None) -> None:
        self.artifact_writer = artifact_writer or TaskArtifactWriter()

    def execute_plan(self, plan: TaskOrchestrationPlan) -> MockTaskExecutionPlanResult:
        results: list[MockTaskExecutionResult] = []

        for record in plan.records:
            result = self.execute_record(record=record)
            results.append(result)
            if not result.success:
                return MockTaskExecutionPlanResult(
                    experiment_id=plan.experiment_id,
                    success=False,
                    task_results=results,
                    message=f'task failed: {record.task_id}',
                )

        return MockTaskExecutionPlanResult(
            experiment_id=plan.experiment_id,
            success=True,
            task_results=results,
            message='mock task execution completed',
        )

    def execute_record(self, record: TaskExecutionRecord) -> MockTaskExecutionResult:
        if record.result_dir is None:
            raise ValueError(f'task {record.task_id} has no result_dir')

        started_at = time.perf_counter()
        result_dir = Path(record.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        fail_mode = str(record.params.get('fail_mode', 'none'))
        status = 'failed' if fail_mode != 'none' else 'finished'
        success = status == 'finished'
        message = self._message(record=record, status=status, fail_mode=fail_mode)
        completed_record = replace(record, status=status)
        elapsed_seconds = max(time.perf_counter() - started_at, 0.0)
        adapter_result = TaskAdapterResult(
            success=success,
            status=status,
            message=message,
            metrics=[self._metric_row(record=completed_record, success=success)],
            predictions=self._prediction_records(record=completed_record) if success else [],
            metadata={'executor': 'local_mock'},
        )
        artifacts = self.artifact_writer.write(
            result_dir=result_dir,
            record=completed_record,
            adapter_result=adapter_result,
            elapsed_seconds=elapsed_seconds,
        )

        return MockTaskExecutionResult(
            record=completed_record,
            success=success,
            status=status,
            message=message,
            artifacts=artifacts,
            elapsed_seconds=elapsed_seconds,
        )

    def write_plan_result(
        self,
        result: MockTaskExecutionPlanResult,
        result_dir: str | Path,
    ) -> Path:
        output_path = Path(result_dir) / 'task_results.json'
        output_path.write_text(
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        return output_path

    def _message(self, record: TaskExecutionRecord, status: str, fail_mode: str) -> str:
        if status == 'failed':
            return f'mock task failed with fail_mode={fail_mode}'

        return f'mock {record.task_type} task completed'

    def _metric_row(self, record: TaskExecutionRecord, success: bool) -> dict[str, object]:
        if not success:
            return {
                'epoch': 1,
                'train_loss': '',
                'val_loss': '',
                'accuracy': '',
                'map50': '',
                'map50_95': '',
                'lr': '',
            }

        if record.task_type == 'detection':
            return {
                'epoch': 1,
                'train_loss': 0.41,
                'val_loss': 0.36,
                'accuracy': '',
                'map50': 0.75,
                'map50_95': 0.55,
                'lr': record.params.get('learning_rate', ''),
            }
        if record.task_type == 'classification':
            return {
                'epoch': 1,
                'train_loss': 0.38,
                'val_loss': 0.31,
                'accuracy': 0.84,
                'map50': '',
                'map50_95': '',
                'lr': record.params.get('learning_rate', ''),
            }

        return {
            'epoch': 1,
            'train_loss': 0.25,
            'val_loss': 0.22,
            'accuracy': 0.8,
            'map50': '',
            'map50_95': '',
            'lr': record.params.get('learning_rate', ''),
        }
    def _prediction_records(self, record: TaskExecutionRecord) -> list[dict[str, object]]:
        if record.task_type == 'detection':
            return [
                {
                    'sample_id': 'mock_sample_001',
                    'class_name': 'class_a',
                    'confidence': 0.91,
                    'bbox_xyxy': [12, 18, 92, 118],
                },
            ]
        if record.task_type == 'classification':
            return [
                {
                    'sample_id': 'mock_sample_001',
                    'class_name': 'class_a',
                    'confidence': 0.88,
                    'top_k': record.params.get('top_k', 1),
                },
            ]
        if record.task_type == 'segmentation':
            return [
                {
                    'sample_id': 'mock_sample_001',
                    'mask_path': 'masks/mock_sample_001.png',
                    'confidence': 0.86,
                },
            ]

        return [{'sample_id': 'mock_sample_001', 'status': 'generated'}]
