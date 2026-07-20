from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ironflow_exp.engine.configs import EngineConfigLoader
from ironflow_exp.engine.domain import ScheduleRecord
from ironflow_exp.engine.runners import LocalExperimentRunner
from ironflow_exp.engine.storage import SQLiteExperimentStorage


@dataclass(frozen=True, slots=True)
class SchedulerRunResult:
    schedule_id: str
    experiment_id: str | None
    status: str
    message: str = ''


class LocalSchedulerService:
    def __init__(
        self,
        storage: SQLiteExperimentStorage,
        config_loader: EngineConfigLoader | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self.config_loader = config_loader or EngineConfigLoader()
        self.clock = clock or (lambda: datetime.now(tz=timezone.utc))

    def add_schedule(
        self,
        config_path: str | Path,
        scheduled_at: str,
        schedule_id: str,
        repeat_rule: str | None = None,
    ) -> ScheduleRecord:
        resolved_config_path = Path(config_path).resolve()
        record = ScheduleRecord(
            schedule_id=schedule_id,
            experiment_config_path=str(resolved_config_path),
            scheduled_at=scheduled_at,
            status='waiting',
            created_at=self._timestamp(),
            repeat_rule=repeat_rule,
        )
        self.storage.save_schedule(record=record)

        return record

    def run_due_once(self, now: datetime | None = None, limit: int | None = None) -> list[SchedulerRunResult]:
        current_time = now or self.clock()
        due_schedules = self._due_schedules(now=current_time)
        if limit is not None:
            due_schedules = due_schedules[:limit]

        return [
            self._run_schedule(schedule=schedule)
            for schedule in due_schedules
        ]

    def _due_schedules(self, now: datetime) -> list[ScheduleRecord]:
        return [
            schedule
            for schedule in self.storage.list_schedules()
            if schedule.status == 'waiting' and self._parse_datetime(schedule.scheduled_at) <= now
        ]

    def _run_schedule(self, schedule: ScheduleRecord) -> SchedulerRunResult:
        executed_at = self._timestamp()
        self.storage.update_schedule_status(
            schedule_id=schedule.schedule_id,
            status='running',
            executed_at=executed_at,
        )
        experiment_id = f'exp_{schedule.schedule_id}'

        try:
            config = self.config_loader.load_file(path=schedule.experiment_config_path)
            runner = LocalExperimentRunner(
                storage=self.storage,
                config_loader=self.config_loader,
                clock=self.clock,
            )
            runner.prepare(config=config, experiment_id=experiment_id)
            run_result = runner.run(experiment_id=experiment_id)
            final_status = 'done' if run_result.is_success else 'failed'
            self.storage.update_schedule_status(
                schedule_id=schedule.schedule_id,
                status=final_status,
                executed_at=executed_at,
            )

            return SchedulerRunResult(
                schedule_id=schedule.schedule_id,
                experiment_id=experiment_id,
                status=final_status,
                message=run_result.message,
            )
        except Exception as error:
            self.storage.update_schedule_status(
                schedule_id=schedule.schedule_id,
                status='failed',
                executed_at=executed_at,
            )

            return SchedulerRunResult(
                schedule_id=schedule.schedule_id,
                experiment_id=experiment_id,
                status='failed',
                message=f'{type(error).__name__}: {error}',
            )

    def _parse_datetime(self, value: str) -> datetime:
        normalized = value.strip().replace('Z', '+00:00')
        if ' ' in normalized and 'T' not in normalized:
            normalized = normalized.replace(' ', 'T', 1)

        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)

        return parsed

    def _timestamp(self) -> str:
        return self.clock().isoformat()
