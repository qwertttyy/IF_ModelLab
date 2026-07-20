from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricRecord:
    run_id: str
    task: str
    model_id: str
    metric_name: str
    metric_value: float | None
    split: str
    scope: str = 'overall'
