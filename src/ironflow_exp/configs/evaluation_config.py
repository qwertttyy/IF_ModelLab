from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TaskMetricConfig:
    enabled: bool = False
    metrics: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    enabled: bool = True
    warmup_runs: int = 3


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    detection: TaskMetricConfig = field(default_factory=TaskMetricConfig)
    classification: TaskMetricConfig = field(default_factory=TaskMetricConfig)
    segmentation: TaskMetricConfig = field(default_factory=TaskMetricConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
