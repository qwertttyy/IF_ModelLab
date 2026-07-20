from dataclasses import dataclass, field

from ironflow_exp.engine.domain.status import ExperimentStatus


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    task_type: str = ''
    runner_type: str = 'local_simulator'
    server_name: str | None = None
    config_path: str | None = None
    result_path: str | None = None
    created_at: str = ''
    started_at: str | None = None
    finished_at: str | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    error_type: str | None = None
    memo: str = ''
    metadata: dict[str, object] = field(default_factory=dict)
