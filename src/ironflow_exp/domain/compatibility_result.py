from dataclasses import dataclass, field

from ironflow_exp.core.enums import CompatibilityStatus


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: str = CompatibilityStatus.ALLOWED.value
    code: str = 'ALLOWED'
    message: str = 'configuration is compatible'
    required_artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        return self.status != CompatibilityStatus.BLOCKED.value
