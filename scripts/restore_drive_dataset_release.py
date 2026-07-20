"""Restore a Drive tar-parts IronFlow dataset release on a remote worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any


DEFAULT_RELEASE_FOLDER_URL = 'https://drive.google.com/drive/folders/YOUR_RELEASE_FOLDER_ID'
DEFAULT_DOWNLOAD_DIR = '/workspace/ironflow/drive_dataset_parts/tank_armor_prepared_v20260630'
DEFAULT_EXTRACT_TO = '/workspace/ironflow/prestaged'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-folder-url', default=DEFAULT_RELEASE_FOLDER_URL)
    parser.add_argument('--download-dir', default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument('--extract-to', default=DEFAULT_EXTRACT_TO)
    parser.add_argument('--skip-gdown-install', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--overwrite-download-dir', action='store_true')
    parser.add_argument('--keep-tar', action='store_true')
    args = parser.parse_args()

    download_dir = Path(args.download_dir).expanduser()
    extract_to = Path(args.extract_to).expanduser()
    if download_dir.exists() and args.overwrite_download_dir:
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_to.mkdir(parents=True, exist_ok=True)

    if not args.skip_gdown_install:
        _run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
    if not args.skip_download:
        _run(['gdown', '--folder', args.release_folder_url, '-O', str(download_dir)])

    release_dir = _find_release_dir(download_dir)
    manifest = _read_manifest(release_dir / 'manifest.json')
    tar_name = str(manifest['tar_name'])
    tar_path = release_dir / tar_name
    _combine_parts(release_dir=release_dir, manifest=manifest, tar_path=tar_path)
    actual_sha256 = None
    expected_sha256 = manifest.get('tar_sha256')
    if expected_sha256:
        actual_sha256 = _sha256(tar_path)
        expected_sha256 = str(expected_sha256).upper()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f'tar sha256 mismatch: expected {expected_sha256}, got {actual_sha256}')
    else:
        print('[IronFlow][WARN] tar_sha256 missing in manifest; skipping tar sha256 verification', flush=True)

    _safe_extract_tar(tar_path=tar_path, extract_to=extract_to)
    if not args.keep_tar:
        tar_path.unlink()

    expected_remote_root = manifest.get('expected_remote_root') or manifest.get('gui_remote_root')
    print(json.dumps({
        'status': 'ok',
        'release_dir': str(release_dir),
        'tar_name': tar_name,
        'tar_sha256': actual_sha256,
        'extract_to': str(extract_to),
        'expected_remote_root': expected_remote_root,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


def _find_release_dir(download_dir: Path) -> Path:
    if (download_dir / 'manifest.json').is_file():
        return download_dir
    manifest_paths = sorted(download_dir.glob('*/manifest.json'))
    if len(manifest_paths) == 1:
        return manifest_paths[0].parent
    if not manifest_paths:
        raise FileNotFoundError(f'manifest.json was not found under {download_dir}')
    matches = ', '.join(str(path.parent) for path in manifest_paths)
    raise RuntimeError(f'multiple release manifests found under {download_dir}: {matches}')


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _combine_parts(*, release_dir: Path, manifest: dict[str, Any], tar_path: Path) -> None:
    parts = _ordered_part_paths(release_dir=release_dir, manifest=manifest)
    with tar_path.open('wb') as output:
        for part in parts:
            expected_part_sha256 = _part_sha256_from_manifest(manifest=manifest, part_name=part.name)
            if expected_part_sha256 is not None:
                actual_part_sha256 = _sha256(part)
                if actual_part_sha256 != expected_part_sha256:
                    raise RuntimeError(
                        f'part sha256 mismatch for {part.name}: expected {expected_part_sha256}, got {actual_part_sha256}',
                    )
            with part.open('rb') as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _ordered_part_paths(*, release_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    part_entries = manifest.get('parts')
    if not isinstance(part_entries, list) or not part_entries:
        fallback_parts = sorted(release_dir.glob('part-*.bin'))
        if not fallback_parts:
            raise ValueError(f'no release parts found under {release_dir}')
        return fallback_parts
    paths: list[Path] = []
    for entry in part_entries:
        if not isinstance(entry, dict) or not entry.get('name'):
            raise ValueError(f'invalid part manifest entry: {entry!r}')
        part_path = release_dir / str(entry['name'])
        if not part_path.is_file():
            raise FileNotFoundError(f'release part is missing: {part_path}')
        paths.append(part_path)
    return paths


def _part_sha256_from_manifest(*, manifest: dict[str, Any], part_name: str) -> str | None:
    for entry in manifest.get('parts') or []:
        if isinstance(entry, dict) and entry.get('name') == part_name and entry.get('sha256'):
            return str(entry['sha256']).upper()
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_extract_tar(*, tar_path: Path, extract_to: Path) -> None:
    extract_root = extract_to.resolve()
    with tarfile.open(tar_path, mode='r') as archive:
        for member in archive.getmembers():
            target_path = (extract_root / member.name).resolve()
            if not target_path.is_relative_to(extract_root):
                raise RuntimeError(f'refusing to extract tar member outside target: {member.name}')
        archive.extractall(path=extract_root)


def _run(command: list[str]) -> None:
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    raise SystemExit(main())
