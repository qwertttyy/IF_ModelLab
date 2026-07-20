from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScheduleRecord:
    schedule_id: str
    experiment_config_path: str
    scheduled_at: str
    status: str = 'waiting'
    created_at: str = ''
    executed_at: str | None = None
    repeat_rule: str | None = None
