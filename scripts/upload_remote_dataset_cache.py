"""Upload a local dataset directory to a reusable remote cache path.

The script uses a tar archive plus scp/ssh so it works on plain Vast instances
without requiring rsync on Windows. Run with ``--dry-run`` first to inspect the
commands that will be executed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-path', required=True, help='Local dataset directory to upload.')
    parser.add_argument('--remote-path', required=True, help='Remote dataset cache directory.')
    parser.add_argument('--host', required=True, help='SSH host or IP address.')
    parser.add_argument('--user', default='root', help='SSH username.')
    parser.add_argument('--port', type=int, required=True, help='SSH port.')
    parser.add_argument('--identity-file', help='Optional SSH private key path.')
    parser.add_argument('--archive-dir', default='runs/asset_uploads', help='Local directory for temporary archives.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without creating or uploading archives.')
    args = parser.parse_args()

    local_path = Path(args.local_path).resolve()
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f'local dataset directory does not exist: {local_path}')

    remote_path = args.remote_path.rstrip('/')
    archive_dir = Path(args.archive_dir).resolve()
    archive_path = archive_dir / f'{local_path.name}.tar.gz'
    remote_archive_path = f'/tmp/ironflow_{local_path.name}.tar.gz'
    ssh_target = f'{args.user}@{args.host}'
    ssh_base = _ssh_base(port=args.port, identity_file=args.identity_file)
    scp_base = _scp_base(port=args.port, identity_file=args.identity_file)

    commands = [
        [*scp_base, str(archive_path), f'{ssh_target}:{remote_archive_path}'],
        [
            *ssh_base,
            ssh_target,
            (
                f'mkdir -p {shell_quote(remote_path)} && '
                f'tar -xzf {shell_quote(remote_archive_path)} -C {shell_quote(remote_path)} '
                f'--strip-components=1 && '
                f'rm -f {shell_quote(remote_archive_path)}'
            ),
        ],
    ]

    if args.dry_run:
        print(f'archive: {archive_path}')
        for command in commands:
            print(_format_command(command))
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    _create_archive(source_dir=local_path, archive_path=archive_path)
    try:
        for command in commands:
            subprocess.run(command, check=True)
    finally:
        if archive_path.exists():
            archive_path.unlink()

    print(f'uploaded {local_path} to {ssh_target}:{remote_path}')
    return 0


def _ssh_base(*, port: int, identity_file: str | None) -> list[str]:
    command = ['ssh', '-p', str(port)]
    if identity_file:
        command.extend(['-i', identity_file])
    return command


def _scp_base(*, port: int, identity_file: str | None) -> list[str]:
    command = ['scp', '-P', str(port)]
    if identity_file:
        command.extend(['-i', identity_file])
    return command


def _create_archive(*, source_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, mode='w:gz') as archive:
        archive.add(source_dir, arcname=source_dir.name, filter=_tar_filter)


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in {'__pycache__', '.pytest_cache'} for part in parts):
        return None
    return info


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _format_command(command: list[str]) -> str:
    executable = shutil.which(command[0]) or command[0]
    return ' '.join([executable, *[shell_quote(part) for part in command[1:]]])


if __name__ == '__main__':
    raise SystemExit(main())
