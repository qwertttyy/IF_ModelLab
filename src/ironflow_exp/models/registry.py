from ironflow_exp.configs import ModelConfig
from ironflow_exp.models.base import BaseModelAdapter
from ironflow_exp.models.spec import ModelSpec


class ModelRegistryError(ValueError):
    pass


class DuplicateModelSpecError(ModelRegistryError):
    pass


class UnknownModelSpecError(ModelRegistryError):
    pass


class AdapterMismatchError(ModelRegistryError):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        key = spec.registry_key

        if key in self._specs:
            task, model_id = key
            raise DuplicateModelSpecError(f'model spec already registered: task={task}, model_id={model_id}')

        self._specs[key] = spec

    def get_spec(self, task: str, model_id: str) -> ModelSpec:
        key = task, model_id

        if key not in self._specs:
            raise UnknownModelSpecError(f'model spec not found: task={task}, model_id={model_id}')

        return self._specs[key]

    def has(self, task: str, model_id: str) -> bool:
        return (task, model_id) in self._specs

    def list_specs(self, task: str | None = None) -> list[ModelSpec]:
        specs = list(self._specs.values())

        if task is None:
            return sorted(specs, key=lambda spec: (spec.task, spec.model_id))

        return sorted(
            [spec for spec in specs if spec.task == task],
            key=lambda spec: spec.model_id,
        )

    def get_adapter_class(self, task: str, model_id: str) -> type[BaseModelAdapter]:
        return self.get_spec(task=task, model_id=model_id).adapter_class

    def create_adapter(
        self,
        task: str,
        model_id: str,
        model_config: ModelConfig,
    ) -> BaseModelAdapter:
        spec = self.get_spec(task=task, model_id=model_id)
        if model_config.adapter is not None and not self._adapter_matches(
            configured_adapter=model_config.adapter,
            spec_adapter=spec.adapter_key,
        ):
            raise AdapterMismatchError(
                f'adapter mismatch for task={task}, model_id={model_id}: '
                f'configured={model_config.adapter}, expected={spec.adapter_key}',
            )

        return spec.adapter_class(model_config=model_config)

    def _adapter_matches(self, configured_adapter: str, spec_adapter: str) -> bool:
        return self.adapter_matches(configured_adapter=configured_adapter, spec_adapter=spec_adapter)

    def adapter_matches(self, configured_adapter: str, spec_adapter: str) -> bool:
        configured_values = self._adapter_match_values(adapter_key=configured_adapter)
        spec_values = self._adapter_match_values(adapter_key=spec_adapter)

        return bool(configured_values & spec_values)

    def _adapter_match_values(self, adapter_key: str) -> set[str]:
        values = {adapter_key}
        for suffix in ('_detection', '_classification', '_segmentation', '_embedding', '_tracking', '_tracker', '_adapter'):
            if adapter_key.endswith(suffix):
                values.add(adapter_key.removesuffix(suffix))

        return values
