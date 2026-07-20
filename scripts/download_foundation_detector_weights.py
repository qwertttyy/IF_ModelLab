from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


WEIGHTS = {
    'rf_detr_base': {
        'url': 'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth',
        'path': 'rf_detr/rf_detr_base.pth',
        'md5': 'b4d3ce46099eaed50626ede388caf979',
        'min_bytes': 1_000_000,
    },
    'd_fine_n_coco': {
        'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth',
        'path': 'd_fine/dfine_hgnetv2_n_coco.pth',
        'md5': None,
        'min_bytes': 1_000_000,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Download foundation detector pretrained checkpoints.')
    parser.add_argument('--output-root', default='models/checkpoints/pretrained')
    parser.add_argument('--chunk-size', type=int, default=1024 * 1024)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, spec in WEIGHTS.items():
        target = output_root / str(spec['path'])
        if args.check_only:
            results.append(_check_weight(name=name, target=target, spec=spec))
        else:
            results.append(_download_weight(name=name, target=target, spec=spec, chunk_size=args.chunk_size))

    manifest_path = output_root / 'foundation_detector_weight_manifest.json'
    manifest_path.write_text(
        json.dumps({'schema_version': '0.1', 'weights': results}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({'manifest': str(manifest_path), 'weights': results}, ensure_ascii=False, indent=2))

    return 0 if all(result['ok'] for result in results) else 1


def _download_weight(*, name: str, target: Path, spec: dict[str, object], chunk_size: int) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + '.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    request = Request(str(spec['url']), headers={'User-Agent': 'IronFlow/1.0'})
    with urlopen(request, timeout=120) as response, tmp_path.open('wb') as file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file.write(chunk)
    os.replace(tmp_path, target)

    return _check_weight(name=name, target=target, spec=spec)


def _check_weight(*, name: str, target: Path, spec: dict[str, object]) -> dict[str, object]:
    exists = target.exists()
    size_bytes = target.stat().st_size if exists else 0
    md5 = _md5(target) if exists else ''
    expected_md5 = spec.get('md5')
    ok = exists and size_bytes >= int(spec['min_bytes'])
    if expected_md5:
        ok = ok and md5 == expected_md5

    return {
        'ok': ok,
        'name': name,
        'url': spec['url'],
        'checkpoint_path': str(target),
        'exists': exists,
        'size_bytes': size_bytes,
        'md5': md5,
        'expected_md5': expected_md5,
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)

    return digest.hexdigest()


if __name__ == '__main__':
    raise SystemExit(main())
