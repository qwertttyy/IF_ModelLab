from collections.abc import Callable

from ironflow_exp.engine.tasks.base import BaseTaskAdapter


TaskAdapterFactory = Callable[[], BaseTaskAdapter]


class TaskAdapterRegistryError(ValueError):
    pass


class DuplicateTaskAdapterError(TaskAdapterRegistryError):
    pass


class UnknownTaskAdapterError(TaskAdapterRegistryError):
    pass


class TaskAdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], TaskAdapterFactory] = {}

    def register(
        self,
        task_type: str,
        adapter_key: str,
        factory: TaskAdapterFactory,
    ) -> None:
        key = task_type, adapter_key
        if key in self._factories:
            raise DuplicateTaskAdapterError(f'task adapter already registered: task_type={task_type}, adapter={adapter_key}')

        self._factories[key] = factory

    def create(self, task_type: str, adapter_key: str) -> BaseTaskAdapter:
        key = task_type, adapter_key
        if key not in self._factories:
            raise UnknownTaskAdapterError(f'task adapter not found: task_type={task_type}, adapter={adapter_key}')

        return self._factories[key]()

    def has(self, task_type: str, adapter_key: str) -> bool:
        return (task_type, adapter_key) in self._factories

    def list_keys(self) -> list[tuple[str, str]]:
        return sorted(self._factories.keys())
