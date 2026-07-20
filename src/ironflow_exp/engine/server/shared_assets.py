import hashlib
import os
import posixpath
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path, PurePosixPath

from ironflow_exp.engine.configs import EngineExperimentConfig
from ironflow_exp.engine.server.ssh_plan import RemoteTransferItem


SHARED_ASSET_PACKAGE_PREFIXES = (
    'runs/user_datasets/',
    'models/checkpoints/pretrained/',
)


@dataclass(frozen=True, slots=True)
class SharedAsset:
    kind: str
    local_path: str
    relative_path: str
    remote_path: str
    transfer_kind: str
    cache_key: str
    file_count: int
    total_size_bytes: int
    source_pattern: str

    def to_dict(self) -> dict[str, object]:
        return {
            'kind': self.kind,
            'local_path': self.local_path,
            'relative_path': self.relative_path,
            'remote_path': self.remote_path,
            'transfer_kind': self.transfer_kind,
            'cache_key': self.cache_key,
            'file_count': self.file_count,
            'total_size_bytes': self.total_size_bytes,
            'source_pattern': self.source_pattern,
        }


@dataclass(frozen=True, slots=True)
class SharedAssetPlan:
    enabled: bool
    assets: list[SharedAsset] = field(default_factory=list)
    package_include: list[str] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(asset.total_size_bytes for asset in self.assets)

    def upload_items(self) -> list[RemoteTransferItem]:
        return [
            RemoteTransferItem(
                local_path=asset.local_path,
                remote_path=asset.remote_path,
                kind=asset.transfer_kind,
            )
            for asset in self.assets
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            'enabled': self.enabled,
            'assets': [asset.to_dict() for asset in self.assets],
            'package_include': list(self.package_include),
            'asset_count': len(self.assets),
            'total_size_bytes': self.total_size_bytes,
        }


class SharedAssetPlanner:
    def build(self, *, config: EngineExperimentConfig, remote_workspace: str | None) -> SharedAssetPlan:
        if remote_workspace is None or not remote_workspace.strip():
            return SharedAssetPlan(enabled=False, package_include=list(config.code.package_include))

        source_root = self._source_root(config=config)
        assets: list[SharedAsset] = []
        package_include: list[str] = []
        seen_roots: set[str] = set()

        for pattern in config.code.package_include:
            normalized_pattern = self._normalize_pattern(pattern)
            if not self._is_shared_asset_pattern(normalized_pattern):
                package_include.append(pattern)
                continue

            relative_root = self._concrete_prefix(normalized_pattern)
            if relative_root is None:
                package_include.append(pattern)
                continue
            local_path = (source_root / Path(*relative_root.parts)).resolve()
            if not local_path.exists():
                package_include.append(pattern)
                continue
            relative_key = relative_root.as_posix()
            if relative_key in seen_roots:
                continue
            seen_roots.add(relative_key)
            asset_kind = self._asset_kind(relative_key)
            transfer_kind = 'directory' if local_path.is_dir() else 'file'
            cache_key, file_count, total_size_bytes = self._asset_fingerprint(
                local_path=local_path,
                relative_path=relative_key,
            )
            remote_path = self._remote_asset_path(
                remote_workspace=remote_workspace,
                asset_kind=asset_kind,
                cache_key=cache_key,
                relative_path=relative_key,
            )
            assets.append(
                SharedAsset(
                    kind=asset_kind,
                    local_path=str(local_path),
                    relative_path=relative_key,
                    remote_path=remote_path,
                    transfer_kind=transfer_kind,
                    cache_key=cache_key,
                    file_count=file_count,
                    total_size_bytes=total_size_bytes,
                    source_pattern=pattern,
                ),
            )

        return SharedAssetPlan(enabled=bool(assets), assets=assets, package_include=package_include)

    def apply_to_config(self, *, config: EngineExperimentConfig, plan: SharedAssetPlan) -> EngineExperimentConfig:
        if not plan.assets:
            return config

        data_variants = []
        for variant in config.data_variants:
            rebased_path = self._remote_for_local_path(raw_path=variant.path, assets=plan.assets)
            data_variants.append(replace(variant, path=rebased_path or variant.path))

        tasks = []
        for task in config.tasks:
            params = dict(task.params)
            checkpoint = params.get('checkpoint')
            if isinstance(checkpoint, str) and checkpoint.strip():
                rebased_checkpoint = self._remote_for_relative_path(raw_path=checkpoint, assets=plan.assets)
                if rebased_checkpoint is not None:
                    params['checkpoint'] = rebased_checkpoint
            tasks.append(replace(task, params=params))

        code = replace(config.code, package_include=list(plan.package_include))
        return replace(config, code=code, data_variants=data_variants, tasks=tasks)

    def _source_root(self, *, config: EngineExperimentConfig) -> Path:
        if config.code.working_dir is not None:
            return Path(config.code.working_dir).resolve()
        return Path(config.code.entrypoint).resolve().parent

    def _normalize_pattern(self, pattern: str) -> str:
        return pattern.replace('\\', '/').lstrip('./')

    def _is_shared_asset_pattern(self, pattern: str) -> bool:
        return any(pattern.startswith(prefix) for prefix in SHARED_ASSET_PACKAGE_PREFIXES)

    def _concrete_prefix(self, pattern: str) -> PurePosixPath | None:
        parts: list[str] = []
        for part in PurePosixPath(pattern).parts:
            if any(char in part for char in '*?['):
                break
            parts.append(part)
        if not parts:
            return None

        return PurePosixPath(*parts)

    def _asset_kind(self, relative_path: str) -> str:
        if relative_path.startswith('runs/user_datasets/'):
            return 'dataset'
        if relative_path.startswith('models/checkpoints/pretrained/'):
            return 'weight'
        return 'asset'

    def _asset_fingerprint(self, *, local_path: Path, relative_path: str) -> tuple[str, int, int]:
        digest = hashlib.sha256()
        digest.update(relative_path.encode('utf-8', errors='surrogateescape'))
        digest.update(b'\n')
        file_count = 0
        total_size_bytes = 0
        if local_path.is_file():
            stat = local_path.stat()
            digest.update(local_path.name.encode('utf-8', errors='surrogateescape'))
            digest.update(b'\0')
            digest.update(str(stat.st_size).encode('ascii'))
            digest.update(b'\0')
            self._update_digest_with_file_content(digest=digest, file_path=local_path)
            return digest.hexdigest(), 1, stat.st_size

        for root, dirnames, filenames in os.walk(local_path):
            dirnames.sort()
            filenames.sort()
            root_path = Path(root)
            for filename in filenames:
                file_path = root_path / filename
                rel_file = file_path.relative_to(local_path).as_posix()
                stat = file_path.stat()
                digest.update(rel_file.encode('utf-8', errors='surrogateescape'))
                digest.update(b'\0')
                digest.update(str(stat.st_size).encode('ascii'))
                digest.update(b'\0')
                self._update_digest_with_file_content(digest=digest, file_path=file_path)
                digest.update(b'\n')
                file_count += 1
                total_size_bytes += stat.st_size

        return digest.hexdigest(), file_count, total_size_bytes

    def _update_digest_with_file_content(self, *, digest: 'hashlib._Hash', file_path: Path) -> None:
        with file_path.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)

    def _remote_asset_path(
        self,
        *,
        remote_workspace: str,
        asset_kind: str,
        cache_key: str,
        relative_path: str,
    ) -> str:
        parts = [remote_workspace.rstrip('/'), 'assets', asset_kind, cache_key, *PurePosixPath(relative_path).parts]
        return posixpath.normpath(posixpath.join(*parts))

    def _remote_for_local_path(self, *, raw_path: str | None, assets: list[SharedAsset]) -> str | None:
        if raw_path is None:
            return None
        raw = Path(raw_path).resolve()
        for asset in assets:
            local_root = Path(asset.local_path).resolve()
            if raw == local_root:
                return asset.remote_path
            try:
                relative = raw.relative_to(local_root)
            except ValueError:
                continue
            return posixpath.normpath(posixpath.join(asset.remote_path, *relative.parts))

        return None

    def _remote_for_relative_path(self, *, raw_path: str, assets: list[SharedAsset]) -> str | None:
        normalized = raw_path.replace('\\', '/').lstrip('./')
        for asset in assets:
            if normalized == asset.relative_path:
                return asset.remote_path

        return None
