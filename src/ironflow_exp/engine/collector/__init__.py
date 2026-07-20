"""Result collection components for logs, metrics, checkpoints, and artifacts."""

from ironflow_exp.engine.collector.result_collector import (
    AugmentationArtifactCollection,
    CollectionResult,
    PredictionArtifactCollection,
    ResultCollector,
    TaskArtifactCollection,
)


__all__ = [
    'CollectionResult',
    'AugmentationArtifactCollection',
    'PredictionArtifactCollection',
    'ResultCollector',
    'TaskArtifactCollection',
]
