from dataclasses import asdict

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.ssh_client import SshCommandResult
from ironflow_exp.engine.time_display import add_kst_display_fields


REDACTED_KEY_PATH = '[redacted-key-path]'
DEFAULT_MAX_COMMAND_LENGTH = 500
STREAM_TAIL_LENGTH = 4000


def truncate_text(value: str, max_length: int = DEFAULT_MAX_COMMAND_LENGTH) -> str:
    if len(value) <= max_length:
        return value

    marker = ' ...[truncated]... '
    if max_length <= len(marker):
        return value[:max_length]

    head_length = max_length - len(marker)

    return f'{value[:head_length]}{marker}'


def redact_command_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            redacted.append(REDACTED_KEY_PATH)
            redact_next = False
            continue

        if arg == '-i':
            redacted.append(arg)
            redact_next = True
            continue

        if arg.startswith('IdentityFile='):
            redacted.append(f'IdentityFile={REDACTED_KEY_PATH}')
            continue

        redacted.append(truncate_text(str(arg)))

    return redacted


def command_args_were_redacted(original: list[str], redacted: list[str]) -> bool:
    return original != redacted


def command_result_summary(command_result: SshCommandResult) -> dict[str, object]:
    command = command_result.command
    redacted_command = truncate_text(command)

    return {
        'command': redacted_command,
        'command_length': len(command),
        'command_truncated': redacted_command != command,
        'exit_code': command_result.exit_code,
        'stdout_tail': command_result.stdout[-STREAM_TAIL_LENGTH:],
        'stderr_tail': command_result.stderr[-STREAM_TAIL_LENGTH:],
        'stdout_length': len(command_result.stdout),
        'stderr_length': len(command_result.stderr),
    }


def server_record_public_dict(record: ServerRecord) -> dict[str, object]:
    data = asdict(record)
    key_path = data.get('key_path')
    data['key_path_set'] = isinstance(key_path, str) and bool(key_path.strip())
    if data['key_path_set']:
        data['key_path'] = REDACTED_KEY_PATH

    return add_kst_display_fields(data=data, fields=('last_checked_at',))
