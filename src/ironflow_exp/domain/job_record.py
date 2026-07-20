from dataclasses import dataclass, field

from ironflow_exp.core.enums import JobStatus


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    run_id: str
    status: str = JobStatus.CREATED.value
    runner: str = 'local'
    started_at: str | None = None
    ended_at: str | None = None
    log_path: str | None = None
    failure_path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
