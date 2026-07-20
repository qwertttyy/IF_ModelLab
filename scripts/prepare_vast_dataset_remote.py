from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, timeout: int | None = None) -> int:
    print('[IronFlow] local:', subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            print(line.rstrip(), flush=True)
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print('[IronFlow][ERROR] command timed out', flush=True)
        return 124


def ssh_base(*, host: str, port: str, user: str, key: Path) -> list[str]:
    return [
        'ssh',
        '-i',
        str(key),
        '-p',
        str(port),
        '-o',
        'BatchMode=yes',
        '-o',
        'StrictHostKeyChecking=accept-new',
        '-o',
        'ServerAliveInterval=30',
        '-o',
        'ServerAliveCountMax=120',
        f'{user}@{host}',
    ]


def scp_base(*, port: str, key: Path) -> list[str]:
    return [
        'scp',
        '-i',
        str(key),
        '-P',
        str(port),
        '-o',
        'BatchMode=yes',
        '-o',
        'StrictHostKeyChecking=accept-new',
    ]


def remote_shell(command: str, *, host: str, port: str, user: str, key: Path, timeout: int | None = None) -> int:
    return run([*ssh_base(host=host, port=port, user=user, key=key), command], timeout=timeout)


def upload_script(local_script: Path, *, workspace: str, host: str, port: str, user: str, key: Path) -> int:
    remote_target = f'{user}@{host}:{workspace.rstrip("/")}/scripts/{local_script.name}'
    return run([*scp_base(port=port, key=key), str(local_script), remote_target])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Upload IronFlow dataset preparation scripts and run them on Vast over SSH.')
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--user', default='root')
    parser.add_argument('--key', required=True)
    parser.add_argument('--workspace', default='/workspace/ironflow')
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args(argv)

    key = Path(args.key).expanduser().resolve()
    if not key.exists():
        print(f'[IronFlow][ERROR] SSH private key not found: {key}', flush=True)
        return 2

    project_root = Path(args.project_root).resolve()
    scripts = [
        project_root / 'scripts' / 'restore_drive_dataset_release.py',
        project_root / 'scripts' / 'prepare_vast_tank_armor_dataset.sh',
    ]
    missing = [path for path in scripts if not path.exists()]
    if missing:
        for path in missing:
            print(f'[IronFlow][ERROR] local script not found: {path}', flush=True)
        return 2

    workspace = args.workspace.rstrip('/') or '/workspace/ironflow'
    mkdir_command = 'set -e; mkdir -p ' + ' '.join(
        shlex.quote(path)
        for path in [f'{workspace}/scripts', f'{workspace}/prestaged', f'{workspace}/drive_sources']
    )
    code = remote_shell(mkdir_command, host=args.host, port=args.port, user=args.user, key=key, timeout=60)
    if code != 0:
        return code

    for script in scripts:
        code = upload_script(script, workspace=workspace, host=args.host, port=args.port, user=args.user, key=key)
        if code != 0:
            return code

    chmod_command = 'set -e; chmod +x ' + ' '.join(
        shlex.quote(f'{workspace}/scripts/{script.name}') for script in scripts
    )
    code = remote_shell(chmod_command, host=args.host, port=args.port, user=args.user, key=key, timeout=60)
    if code != 0:
        return code

    run_command = (
        'set -e; '
        f'cd {shlex.quote(workspace)}; '
        'bash scripts/prepare_vast_tank_armor_dataset.sh'
    )
    return remote_shell(run_command, host=args.host, port=args.port, user=args.user, key=key, timeout=args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
