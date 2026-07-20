from abc import ABC, abstractmethod
from dataclasses import dataclass

from ironflow_exp.configs import ExperimentConfig
from ironflow_exp.domain import JobRecord
from ironflow_exp.services import ExperimentServiceResult


@dataclass(frozen=True, slots=True)
class RunnerResult:
    job: JobRecord
    service_result: ExperimentServiceResult | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error_message is None


class BaseRunner(ABC):
    @abstractmethod
    def run_preprocessing(
        self,
        config: ExperimentConfig,
        run_id: str | None = None,
        check_paths: bool = True,
    ) -> RunnerResult:
        """
        Execute preprocessing flow and return a job-oriented result.
        """
