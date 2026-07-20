from enum import Enum


class ExperimentStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    COLLECTED = 'collected'

    @property
    def is_terminal(self) -> bool:
        return self in {
            ExperimentStatus.FINISHED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.COLLECTED,
        }
