from dataclasses import asdict, dataclass, field
from pathlib import Path

from ironflow_exp.engine.configs import EngineDataVariantConfig, EngineExperimentConfig, EngineTaskConfig
from ironflow_exp.engine.domain import TaskExecutionRecord


@dataclass(frozen=True, slots=True)
class TaskOrchestrationPlan:
    experiment_id: str
    records: list[TaskExecutionRecord] = field(default_factory=list)
    disabled_task_ids: list[str] = field(default_factory=list)
    disabled_variant_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TaskOrchestrationService:
    def build_plan(
        self,
        config: EngineExperimentConfig,
        experiment_id: str,
        result_dir: str | Path | None = None,
    ) -> TaskOrchestrationPlan:
        variants_by_id = {variant.id: variant for variant in config.data_variants}
        enabled_tasks = [task for task in config.tasks if task.enabled]
        disabled_task_ids = [task.id for task in config.tasks if not task.enabled]
        disabled_variant_ids = [variant.id for variant in config.data_variants if not variant.enabled]
        ordered_tasks = self._topological_sort(tasks=enabled_tasks)

        records = [
            self._build_record(
                experiment_id=experiment_id,
                task=task,
                variant=variants_by_id[task.input_variant],
                order_index=order_index,
                result_dir=result_dir,
            )
            for order_index, task in enumerate(ordered_tasks)
        ]

        return TaskOrchestrationPlan(
            experiment_id=experiment_id,
            records=records,
            disabled_task_ids=disabled_task_ids,
            disabled_variant_ids=disabled_variant_ids,
        )

    def _topological_sort(self, tasks: list[EngineTaskConfig]) -> list[EngineTaskConfig]:
        task_by_id = {task.id: task for task in tasks}
        original_order = {task.id: index for index, task in enumerate(tasks)}
        remaining_dependencies = {
            task.id: set(task.depends_on)
            for task in tasks
        }

        for task in tasks:
            for dependency_id in task.depends_on:
                if dependency_id not in task_by_id:
                    raise ValueError(
                        f'enabled task {task.id} depends on disabled or unknown task {dependency_id}',
                    )

        ordered: list[EngineTaskConfig] = []
        ready = sorted(
            [task_id for task_id, dependencies in remaining_dependencies.items() if not dependencies],
            key=lambda task_id: original_order[task_id],
        )

        while ready:
            task_id = ready.pop(0)
            ordered.append(task_by_id[task_id])

            for candidate_id, dependencies in remaining_dependencies.items():
                if task_id not in dependencies:
                    continue

                dependencies.remove(task_id)
                if not dependencies and candidate_id not in {task.id for task in ordered} and candidate_id not in ready:
                    ready.append(candidate_id)
                    ready.sort(key=lambda item: original_order[item])

        if len(ordered) != len(tasks):
            unresolved = sorted(set(task_by_id) - {task.id for task in ordered})
            raise ValueError(f'task dependency cycle detected: {", ".join(unresolved)}')

        return ordered

    def _build_record(
        self,
        experiment_id: str,
        task: EngineTaskConfig,
        variant: EngineDataVariantConfig,
        order_index: int,
        result_dir: str | Path | None,
    ) -> TaskExecutionRecord:
        if not variant.enabled:
            raise ValueError(f'enabled task {task.id} references disabled data variant {variant.id}')

        return TaskExecutionRecord(
            experiment_id=experiment_id,
            task_id=task.id,
            task_type=task.task_type,
            order_index=order_index,
            input_variant_id=variant.id,
            input_variant_kind=variant.kind,
            input_variant_path=variant.path,
            model_id=task.model_id,
            adapter=task.adapter,
            depends_on=task.depends_on,
            params=task.params,
            result_dir=self._task_result_dir(result_dir=result_dir, order_index=order_index, task_id=task.id),
        )

    def _task_result_dir(self, result_dir: str | Path | None, order_index: int, task_id: str) -> str | None:
        if result_dir is None:
            return None

        safe_task_id = ''.join(character if character.isalnum() or character in {'_', '-'} else '_' for character in task_id)
        return str(Path(result_dir) / 'tasks' / f'{order_index:02d}_{safe_task_id}')
