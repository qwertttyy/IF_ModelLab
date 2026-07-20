"""Download a raw class-folder Drive dataset and materialize IronFlow datasets.

Use restore_drive_dataset_release.py for the packaged tar-parts release folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEAM_DRIVE_DATASET_URL = 'https://drive.google.com/drive/folders/YOUR_TEAM_DATASET_FOLDER_ID'
DEFAULT_DOWNLOAD_ROOT = '/workspace/ironflow/team_drive_source'
DEFAULT_DATASET_ROOT = '/workspace/ironflow/prestaged/tank9_prepared_v20260629'
DEFAULT_CLASSES = (
    'altay',
    'challenger_2',
    'k2',
    'leopard_2',
    'leclerc',
    'm1_abrams',
    'merkava_mk4',
    'type_10',
    't62',
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-url', default=DEFAULT_TEAM_DRIVE_DATASET_URL)
    parser.add_argument('--download-root', default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument('--dataset-root', default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--source-subdir', default='exports')
    parser.add_argument('--skip-gdown-install', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--no-overwrite', action='store_true')
    args = parser.parse_args()

    download_root = Path(args.download_root).expanduser()
    if not args.skip_gdown_install:
        _run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
    if not args.skip_download:
        download_root.mkdir(parents=True, exist_ok=True)
        _run(['gdown', '--folder', args.drive_url, '-O', str(download_root)])

    source_root = _resolve_source_root(download_root)
    import_command = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'import_class_folder_training_dataset.py'),
        '--source-root',
        str(source_root),
        '--dataset-root',
        str(Path(args.dataset_root).expanduser()),
        '--source-subdir',
        args.source_subdir,
        '--skip-empty-classes',
    ]
    if not args.no_overwrite:
        import_command.append('--overwrite')
    _run(import_command)
    return 0


def _resolve_source_root(download_root: Path) -> Path:
    if any((download_root / class_name).is_dir() for class_name in DEFAULT_CLASSES):
        return download_root
    child_dirs = [path for path in download_root.iterdir() if path.is_dir()]
    for child in child_dirs:
        if any((child / class_name).is_dir() for class_name in DEFAULT_CLASSES):
            return child
    return download_root


def _run(command: list[str]) -> None:
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    raise SystemExit(main())
