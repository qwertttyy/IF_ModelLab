"""Local-first experiment engine layer for IronFlow."""

__all__ = [
    'BaseExperimentRunner',
    'EngineConfigLoader',
    'EngineConfigValidator',
    'EngineExperimentConfig',
    'ExperimentRunnerResult',
    'ExperimentStatus',
    'LocalExperimentRunner',
    'SQLiteExperimentStorage',
]


def __getattr__(name: str) -> object:
    if name == 'ExperimentStatus':
        from ironflow_exp.engine.domain import ExperimentStatus

        return ExperimentStatus
    if name in {'EngineConfigLoader', 'EngineConfigValidator', 'EngineExperimentConfig'}:
        from ironflow_exp.engine.configs import EngineConfigLoader, EngineConfigValidator, EngineExperimentConfig

        return {
            'EngineConfigLoader': EngineConfigLoader,
            'EngineConfigValidator': EngineConfigValidator,
            'EngineExperimentConfig': EngineExperimentConfig,
        }[name]
    if name in {'BaseExperimentRunner', 'ExperimentRunnerResult', 'LocalExperimentRunner'}:
        from ironflow_exp.engine.runners import BaseExperimentRunner, ExperimentRunnerResult, LocalExperimentRunner

        return {
            'BaseExperimentRunner': BaseExperimentRunner,
            'ExperimentRunnerResult': ExperimentRunnerResult,
            'LocalExperimentRunner': LocalExperimentRunner,
        }[name]
    if name == 'SQLiteExperimentStorage':
        from ironflow_exp.engine.storage import SQLiteExperimentStorage

        return SQLiteExperimentStorage

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
