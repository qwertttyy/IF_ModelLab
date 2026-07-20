from dataclasses import dataclass

from ironflow_exp.models.registry import ModelRegistry
from ironflow_exp.models.spec import ModelSpec


@dataclass(frozen=True, slots=True)
class ModelCompatibilityRow:
    task: str
    model_id: str
    display_name: str
    family: str
    adapter_key: str
    readiness_status: str
    implementation_status: str
    runtime_targets: tuple[str, ...]
    input_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    notes: str

    @property
    def is_runnable(self) -> bool:
        return self.readiness_status in {'adapter_ready', 'validated'}


def build_model_compatibility_matrix(registry: ModelRegistry) -> tuple[ModelCompatibilityRow, ...]:
    return tuple(
        _row_from_spec(spec=spec)
        for spec in registry.list_specs()
    )


def readiness_counts(rows: tuple[ModelCompatibilityRow, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.readiness_status] = counts.get(row.readiness_status, 0) + 1

    return counts


def _row_from_spec(spec: ModelSpec) -> ModelCompatibilityRow:
    return ModelCompatibilityRow(
        task=spec.task,
        model_id=spec.model_id,
        display_name=spec.display_name,
        family=spec.family,
        adapter_key=spec.adapter_key,
        readiness_status=spec.readiness_status,
        implementation_status=spec.implementation_status,
        runtime_targets=spec.runtime_targets,
        input_artifacts=spec.input_artifacts,
        output_artifacts=spec.output_artifacts,
        notes=spec.notes,
    )
