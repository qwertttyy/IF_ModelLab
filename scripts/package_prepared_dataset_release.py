"""Package a prepared IronFlow dataset as tar parts for Drive distribution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = 'tank_armor_prepared_v20260630'
DEFAULT_PART_SIZE_MIB = 4096


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset-root',
        default=f'runs/user_datasets/{DEFAULT_DATASET_ID}',
        help='Prepared dataset root to package, relative to the project root unless absolute.',
    )
    parser.add_argument(
        '--release-dir',
        default=f'runs/dataset_releases/{DEFAULT_DATASET_ID}_tar_parts',
        help='Output folder for tar parts and manifests, relative to the project root unless absolute.',
    )
    parser.add_argument('--dataset-id', default=DEFAULT_DATASET_ID)
    parser.add_argument('--part-size-mib', type=int, default=DEFAULT_PART_SIZE_MIB)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    dataset_root = _resolve_path(args.dataset_root)
    release_dir = _resolve_path(args.release_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'dataset root does not exist: {dataset_root}')
    if release_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'release dir already exists; pass --overwrite: {release_dir}')
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    tar_name = f'ironflow_dataset_{args.dataset_id}.tar'
    tar_path = release_dir / tar_name
    _write_tar(dataset_root=dataset_root, dataset_id=args.dataset_id, tar_path=tar_path)
    sha256 = _sha256(tar_path)
    parts = _split_file(path=tar_path, release_dir=release_dir, part_size_bytes=args.part_size_mib * 1024 * 1024)
    manifest = {
        'schema_version': '0.1',
        'dataset_id': args.dataset_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_dataset_root': str(dataset_root),
        'tar_name': tar_name,
        'tar_size_bytes': tar_path.stat().st_size,
        'tar_sha256': sha256,
        'part_size_mib': args.part_size_mib,
        'part_count': len(parts),
        'parts': [
            {
                'name': part.name,
                'size_bytes': part.stat().st_size,
                'sha256': _sha256(part),
            }
            for part in parts
        ],
        'extract_to': '/workspace/ironflow/prestaged',
        'expected_remote_root': f'/workspace/ironflow/prestaged/{args.dataset_id}',
        'gui_dataset_source': 'Remote pre-staged path',
        'gui_remote_root': f'/workspace/ironflow/prestaged/{args.dataset_id}',
        'restore_script_command': 'python scripts/restore_drive_dataset_release.py --release-folder-url RELEASE_FOLDER_URL',
        'commands': [
            'python3 -m pip install -q gdown',
            'mkdir -p /workspace/ironflow/drive_dataset_parts',
            '<gdown --folder RELEASE_FOLDER_URL -O /workspace/ironflow/drive_dataset_parts>',
            'cd /workspace/ironflow/drive_dataset_parts',
            f'cat part-*.bin > {tar_name}',
            f'echo "{sha256}  {tar_name}" | sha256sum -c -',
            'mkdir -p /workspace/ironflow/prestaged',
            f'tar -xf {tar_name} -C /workspace/ironflow/prestaged',
            f'ls /workspace/ironflow/prestaged/{args.dataset_id}',
        ],
    }
    manifest_path = release_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (release_dir / 'manifest.sha256').write_text(f'{sha256}  {tar_name}\n', encoding='utf-8')
    _write_readme(release_dir=release_dir, manifest=manifest)
    tar_path.unlink()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_tar(*, dataset_root: Path, dataset_id: str, tar_path: Path) -> None:
    with tarfile.open(tar_path, mode='w') as archive:
        archive.add(dataset_root, arcname=dataset_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _split_file(*, path: Path, release_dir: Path, part_size_bytes: int) -> list[Path]:
    parts: list[Path] = []
    with path.open('rb') as source:
        index = 0
        while True:
            chunk = source.read(part_size_bytes)
            if not chunk:
                break
            part_path = release_dir / f'part-{index:04d}.bin'
            part_path.write_bytes(chunk)
            parts.append(part_path)
            index += 1
    return parts


def _write_readme(*, release_dir: Path, manifest: dict[str, object]) -> None:
    lines = [
        f"# IronFlow Dataset Release: {manifest['dataset_id']}",
        '',
        'Upload all files in this folder to Google Drive.',
        '',
        '## Vast restore, preferred',
        '',
        'Run from an IronFlow checkout on the remote worker:',
        '',
        '```bash',
        manifest['restore_script_command'],
        '```',
        '',
        '## Vast restore, manual',
        '',
        '```bash',
        *manifest['commands'],
        '```',
        '',
        '## GUI',
        '',
        '```text',
        f"Dataset Source: {manifest['gui_dataset_source']}",
        f"Remote Root: {manifest['gui_remote_root']}",
        '```',
        '',
    ]
    (release_dir / 'README.md').write_text('\n'.join(str(line) for line in lines), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
