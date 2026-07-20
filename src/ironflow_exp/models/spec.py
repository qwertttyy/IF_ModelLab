from dataclasses import dataclass, field

from ironflow_exp.models.base import BaseModelAdapter


@dataclass(frozen=True, slots=True)
class ModelSpec:
    task: str
    model_id: str
    adapter_key: str
    adapter_class: type[BaseModelAdapter]
    family: str
    display_name: str
    source: str
    supports_train: bool
    supports_predict: bool
    implementation_status: str = 'planned'
    readiness_status: str = 'prepared'
    runtime_targets: tuple[str, ...] = field(default_factory=tuple)
    input_formats: tuple[str, ...] = field(default_factory=tuple)
    input_artifacts: tuple[str, ...] = field(default_factory=tuple)
    output_artifacts: tuple[str, ...] = field(default_factory=tuple)
    required_dependencies: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ''

    @property
    def registry_key(self) -> tuple[str, str]:
        return self.task, self.model_id

    @property
    def is_implemented(self) -> bool:
        return self.implementation_status == 'implemented'

    @property
    def is_prepared(self) -> bool:
        return self.readiness_status in {'prepared', 'adapter_ready', 'validated'}

    @property
    def is_adapter_ready(self) -> bool:
        return self.readiness_status in {'adapter_ready', 'validated'}
