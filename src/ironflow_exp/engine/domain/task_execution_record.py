from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class TaskExecutionRecord:
    experiment_id: str
    task_id: str
    task_type: str
    order_index: int
    input_variant_id: str
    input_variant_kind: str
    input_variant_path: str | None = None
    model_id: str | None = None
    adapter: str | None = None
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, object] = field(default_factory=dict)
    result_dir: str | None = None
    status: str = 'pending'

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
