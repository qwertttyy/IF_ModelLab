from collections import deque
from pathlib import Path


def read_text_tail(path: Path, line_count: int | None = None) -> str:
    if not path.exists():
        return ''
    if line_count is not None and line_count <= 0:
        return ''
    if line_count is None:
        return path.read_text(encoding='utf-8', errors='replace')

    with path.open(mode='r', encoding='utf-8', errors='replace') as file:
        lines = deque(file, maxlen=line_count)

    return ''.join(lines).rstrip('\r\n')
