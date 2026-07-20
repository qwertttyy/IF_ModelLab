from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SplitConfig:
    strategy: str = 'existing_or_create'
    group_key: str | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_id: str
    source_root: str
    dataset_format: str
    label_format: str
    split: SplitConfig = field(default_factory=SplitConfig)
    class_map: dict[int, str] = field(default_factory=dict)
