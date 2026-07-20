from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerRecord:
    name: str
    server_type: str = 'local'
    host: str | None = None
    port: int | None = None
    username: str | None = None
    key_path: str | None = None
    remote_workspace: str | None = None
    last_checked_at: str | None = None
    last_status: str | None = None
    gpu_name: str | None = None
    total_vram: str | None = None
