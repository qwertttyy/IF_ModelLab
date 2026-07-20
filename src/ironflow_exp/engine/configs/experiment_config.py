from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EngineExperimentMetaConfig:
    name: str
    description: str = ''
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EngineRuntimeConfig:
    runner: str = 'local_simulator'
    workspace: str = 'runs/workspaces'
    experiment_root: str = 'runs/experiments'
    python_executable: str = 'python'
    remote_python_executable: str = 'python'
    container_mode: str = 'native_python'
    container_image: str = ''


@dataclass(frozen=True, slots=True)
class EngineCodeConfig:
    entrypoint: str
    working_dir: str | None = None
    args: list[str] = field(default_factory=list)
    package_include: list[str] = field(default_factory=list)
    package_exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EngineTrainConfig:
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    max_seconds: int | None = None
    fail_fast: bool = True


@dataclass(frozen=True, slots=True)
class EngineDataVariantConfig:
    id: str
    kind: str = 'original'
    source: str | None = None
    path: str | None = None
    enabled: bool = True
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineTaskConfig:
    id: str
    task_type: str
    input_variant: str
    model_id: str | None = None
    adapter: str | None = None
    enabled: bool = True
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineOutputConfig:
    log_file: str = 'train.log'
    metrics_file: str = 'metrics.csv'
    checkpoint_file: str = 'best.pt'
    status_file: str = 'status.marker'
    summary_file: str = 'summary.md'
    save_json: bool = True
    save_csv: bool = True
    save_summary: bool = True
    save_checkpoints: bool = True
    save_previews: bool = True
    artifact_collection_mode: str = 'full'
    collect_patterns: list[str] = field(default_factory=lambda: [
        'train.log',
        'metrics.csv',
        'metrics.json',
        'tasks/*/metrics/*.csv',
        'best.pt',
        'last.pt',
        'status.marker',
        'summary.md',
        'experiment.json',
        'artifacts.json',
        'timings.csv',
        'task_results.json',
        'sample_predictions/*.json',
    ])


@dataclass(frozen=True, slots=True)
class EngineAnalysisConfig:
    primary_metric: str = 'accuracy'
    higher_is_better: bool = True
    failure_rules_enabled: bool = True


@dataclass(frozen=True, slots=True)
class EngineRepositoryConfig:
    sqlite_path: str = 'runs/ironflow_experiments.sqlite3'


@dataclass(frozen=True, slots=True)
class EngineExperimentConfig:
    schema_version: str
    experiment: EngineExperimentMetaConfig
    runtime: EngineRuntimeConfig
    code: EngineCodeConfig
    train: EngineTrainConfig = field(default_factory=EngineTrainConfig)
    data_variants: list[EngineDataVariantConfig] = field(default_factory=lambda: [
        EngineDataVariantConfig(id='original', kind='original'),
    ])
    tasks: list[EngineTaskConfig] = field(default_factory=list)
    output: EngineOutputConfig = field(default_factory=EngineOutputConfig)
    analysis: EngineAnalysisConfig = field(default_factory=EngineAnalysisConfig)
    repository: EngineRepositoryConfig = field(default_factory=EngineRepositoryConfig)
