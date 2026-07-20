from typing import Any

from ironflow_exp.configs import ModelConfig
from ironflow_exp.domain import ArtifactRecord, ExperimentContext
from ironflow_exp.models.base import ClassificationModelAdapter, PredictionRecord


class MissingTorchvisionDependencyError(RuntimeError):
    pass


class UnsupportedTorchvisionClassifierError(ValueError):
    pass


TORCHVISION_CLASSIFIER_SPECS: dict[str, dict[str, str]] = {
    'mobilenet_v3_small': {
        'builder_name': 'mobilenet_v3_small',
        'weights_name': 'MobileNet_V3_Small_Weights',
    },
    'mobilenet_v3_large': {
        'builder_name': 'mobilenet_v3_large',
        'weights_name': 'MobileNet_V3_Large_Weights',
    },
    'efficientnet_b0': {
        'builder_name': 'efficientnet_b0',
        'weights_name': 'EfficientNet_B0_Weights',
    },
    'efficientnet_b3': {
        'builder_name': 'efficientnet_b3',
        'weights_name': 'EfficientNet_B3_Weights',
    },
    'efficientnet_v2_s': {
        'builder_name': 'efficientnet_v2_s',
        'weights_name': 'EfficientNet_V2_S_Weights',
    },
    'resnet50': {
        'builder_name': 'resnet50',
        'weights_name': 'ResNet50_Weights',
    },
    'resnext50_32x4d': {
        'builder_name': 'resnext50_32x4d',
        'weights_name': 'ResNeXt50_32X4D_Weights',
    },
    'resnet18': {
        'builder_name': 'resnet18',
        'weights_name': 'ResNet18_Weights',
    },
}
SUPPORTED_TORCHVISION_CLASSIFIER_MODEL_IDS = frozenset(TORCHVISION_CLASSIFIER_SPECS.keys())


class TorchvisionClassificationAdapter(ClassificationModelAdapter):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config=model_config)
        self.model: Any | None = None

    def load(self, context: ExperimentContext) -> None:
        models_module = self._load_torchvision_models()
        builder = getattr(models_module, self._builder_name())
        weights = self._weights(models_module=models_module)
        self.model = builder(weights=weights)

    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        raise NotImplementedError(
            'TorchvisionClassificationAdapter.train is an adapter skeleton and is not implemented yet',
        )

    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        raise NotImplementedError(
            'TorchvisionClassificationAdapter.predict is an adapter skeleton and is not implemented yet',
        )

    def _load_torchvision_models(self) -> Any:
        try:
            import torchvision.models as models
        except ImportError as error:
            raise MissingTorchvisionDependencyError(
                'torchvision is required for TorchvisionClassificationAdapter',
            ) from error

        return models

    def _builder_name(self) -> str:
        return self._classifier_spec()['builder_name']

    def _weights_name(self) -> str:
        return self._classifier_spec()['weights_name']

    def _weights(self, models_module: Any) -> Any | None:
        if not self.model_config.pretrained:
            return None

        weights_class = getattr(models_module, self._weights_name())

        return weights_class.DEFAULT

    def _classifier_spec(self) -> dict[str, str]:
        model_id = self.model_config.model_id
        if model_id is None or model_id not in TORCHVISION_CLASSIFIER_SPECS:
            supported = ', '.join(sorted(SUPPORTED_TORCHVISION_CLASSIFIER_MODEL_IDS))
            raise UnsupportedTorchvisionClassifierError(
                f'unsupported torchvision classifier: {model_id}; supported: {supported}',
            )

        return TORCHVISION_CLASSIFIER_SPECS[model_id]
