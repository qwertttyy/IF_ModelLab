from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runner: str = 'local'
    device: str = 'cuda'
    precision: str = 'fp32'
    num_workers: int = 4
