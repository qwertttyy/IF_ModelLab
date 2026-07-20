from dataclasses import dataclass, field

from ironflow_exp.configs.dataset_config import DatasetConfig
from ironflow_exp.configs.evaluation_config import EvaluationConfig
from ironflow_exp.configs.export_config import ExportConfig
from ironflow_exp.configs.model_config import ModelGroupConfig
from ironflow_exp.configs.preprocessing_config import PreprocessingConfig
from ironflow_exp.configs.runtime_config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class ExperimentMetaConfig:
    name: str
    description: str = ''
    seed: int = 42
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: str
    experiment: ExperimentMetaConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    models: ModelGroupConfig
    evaluation: EvaluationConfig
    export: ExportConfig
