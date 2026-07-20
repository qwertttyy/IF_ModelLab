from dataclasses import dataclass, field
from pathlib import Path

from ironflow_exp.engine.domain import ServerRecord


ALLOWED_SERVER_TYPES = {'local', 'local_ssh_simulator', 'ssh', 'vast_manual'}


@dataclass(frozen=True, slots=True)
class ServerProfileValidationIssue:
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class ServerProfileValidationResult:
    is_valid: bool
    errors: list[ServerProfileValidationIssue] = field(default_factory=list)
    warnings: list[ServerProfileValidationIssue] = field(default_factory=list)


class ServerProfileValidator:
    def validate(
        self,
        record: ServerRecord,
        check_paths: bool = False,
    ) -> ServerProfileValidationResult:
        errors: list[ServerProfileValidationIssue] = []
        warnings: list[ServerProfileValidationIssue] = []

        if not record.name.strip():
            self._add_error(errors=errors, field='name', message='server name is required')

        if record.server_type not in ALLOWED_SERVER_TYPES:
            self._add_error(errors=errors, field='type', message='unsupported server type')

        if record.server_type in {'ssh', 'vast_manual'}:
            self._validate_remote_profile(record=record, errors=errors, warnings=warnings, check_paths=check_paths)
        if record.server_type == 'local_ssh_simulator':
            self._validate_local_ssh_simulator(record=record, errors=errors)

        return ServerProfileValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    def _validate_remote_profile(
        self,
        record: ServerRecord,
        errors: list[ServerProfileValidationIssue],
        warnings: list[ServerProfileValidationIssue],
        check_paths: bool,
    ) -> None:
        if record.host is None or not record.host.strip():
            self._add_error(errors=errors, field='host', message='host is required for remote server profiles')

        if record.port is None:
            self._add_error(errors=errors, field='port', message='port is required for remote server profiles')
        elif record.port <= 0:
            self._add_error(errors=errors, field='port', message='port must be greater than 0')

        if record.username is None or not record.username.strip():
            self._add_error(errors=errors, field='username', message='username is required for remote server profiles')

        if record.remote_workspace is None or not record.remote_workspace.strip():
            self._add_error(
                errors=errors,
                field='remote_workspace',
                message='remote_workspace is required for remote server profiles',
            )

        if record.key_path is None or not record.key_path.strip():
            warnings.append(
                ServerProfileValidationIssue(
                    field='key_path',
                    message='key_path is not set; SSH execution may require password or another credential method',
                ),
            )
            return

        if check_paths and not Path(record.key_path).exists():
            self._add_error(errors=errors, field='key_path', message='key_path does not exist')

    def _validate_local_ssh_simulator(
        self,
        record: ServerRecord,
        errors: list[ServerProfileValidationIssue],
    ) -> None:
        if record.remote_workspace is None or not record.remote_workspace.strip():
            self._add_error(
                errors=errors,
                field='remote_workspace',
                message='remote_workspace is required for local SSH simulator profiles',
            )

    def _add_error(
        self,
        errors: list[ServerProfileValidationIssue],
        field: str,
        message: str,
    ) -> None:
        errors.append(ServerProfileValidationIssue(field=field, message=message))
