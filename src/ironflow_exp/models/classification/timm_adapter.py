from dataclasses import dataclass
from typing import Any

from ironflow_exp.configs import ModelConfig
from ironflow_exp.domain import ArtifactRecord, ExperimentContext
from ironflow_exp.models.base import ClassificationModelAdapter, PredictionRecord


class MissingTimmDependencyError(RuntimeError):
    pass


class UnsupportedTimmClassifierError(ValueError):
    pass


TIMM_CLASSIFIER_SPECS: dict[str, dict[str, str]] = {
    'convnext_v2_tiny': {
        'timm_model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    },
    'convnext_small': {
        'timm_model_name': 'convnext_small.fb_in22k_ft_in1k',
    },
    'convnext_tiny': {
        'timm_model_name': 'convnext_tiny',
    },
    'vit_tiny': {
        'timm_model_name': 'vit_tiny_patch16_224',
    },
    'vit_base': {
        'timm_model_name': 'vit_base_patch16_224',
    },
    'deit_tiny': {
        'timm_model_name': 'deit_tiny_patch16_224',
    },
    'swin_tiny': {
        'timm_model_name': 'swin_tiny_patch4_window7_224',
    },
}
SUPPORTED_TIMM_CLASSIFIER_MODEL_IDS = frozenset(TIMM_CLASSIFIER_SPECS.keys())


@dataclass(frozen=True, slots=True)
class TimmClassifierReference:
    model_id: str
    timm_model_name: str
    pretrained: bool
    checkpoint: str | None


class TimmClassificationAdapter(ClassificationModelAdapter):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config=model_config)
        self.model: Any | None = None

    def load(self, context: ExperimentContext) -> None:
        timm = self._load_timm()
        reference = self._model_reference_contract()
        self.model = timm.create_model(reference.timm_model_name, pretrained=reference.pretrained)

    def train(self, context: ExperimentContext) -> list[ArtifactRecord]:
        raise NotImplementedError(
            'TimmClassificationAdapter.train is an adapter skeleton and is not implemented yet',
        )

    def predict(self, context: ExperimentContext) -> list[PredictionRecord]:
        raise NotImplementedError(
            'TimmClassificationAdapter.predict is an adapter skeleton and is not implemented yet',
        )

    def _load_timm(self) -> Any:
        try:
            import timm
        except ImportError as error:
            raise MissingTimmDependencyError(
                'timm is required for TimmClassificationAdapter',
            ) from error

        return timm

    def _model_reference_contract(self) -> TimmClassifierReference:
        spec = self._classifier_spec()

        return TimmClassifierReference(
            model_id=self._validated_model_id(),
            timm_model_name=spec['timm_model_name'],
            pretrained=self.model_config.pretrained,
            checkpoint=self.model_config.checkpoint,
        )

    def _timm_model_name(self) -> str:
        return self._classifier_spec()['timm_model_name']

    def _validated_model_id(self) -> str:
        model_id = self.model_config.model_id
        if model_id is None or model_id not in TIMM_CLASSIFIER_SPECS:
            supported = ', '.join(sorted(SUPPORTED_TIMM_CLASSIFIER_MODEL_IDS))
            raise UnsupportedTimmClassifierError(
                f'unsupported timm classifier: {model_id}; supported: {supported}',
            )

        return model_id

    def _classifier_spec(self) -> dict[str, str]:
        return TIMM_CLASSIFIER_SPECS[self._validated_model_id()]
