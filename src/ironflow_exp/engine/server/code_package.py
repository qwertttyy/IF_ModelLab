import fnmatch
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ironflow_exp.engine.configs import EngineExperimentConfig


DEFAULT_CODE_PACKAGE_HARD_EXCLUDE_PATTERNS = (
    '.git',
    '.git/*',
    '.venv',
    '.venv/*',
    'venv',
    'venv/*',
    '__pycache__',
    '__pycache__/*',
    '.pytest_cache',
    '.pytest_cache/*',
    'local_remote_simulator/results',
    'local_remote_simulator/results/*',
    'local_remote_simulator/workspace',
    'local_remote_simulator/workspace/*',
    '.env',
    '*.pyc',
    '*.pt',
    '*.pth',
    '*.onnx',
    '*.sqlite',
    '*.sqlite3',
)

DEFAULT_CODE_PACKAGE_OVERRIDABLE_EXCLUDE_PATTERNS = (
    'data',
    'data/*',
    'datasets',
    'datasets/*',
    'local_remote_simulator/datasets',
    'local_remote_simulator/datasets/*',
    'runs',
    'runs/*',
)

ROOT_SCOPED_DEFAULT_EXCLUDE_NAMES = frozenset({'data', 'datasets', 'runs'})

DEFAULT_CODE_PACKAGE_EXCLUDE_PATTERNS = (
    *DEFAULT_CODE_PACKAGE_HARD_EXCLUDE_PATTERNS,
    *DEFAULT_CODE_PACKAGE_OVERRIDABLE_EXCLUDE_PATTERNS,
)

DEFAULT_CODE_PACKAGE_EXCLUDE_OVERRIDE_ALLOW_PATTERNS: tuple[str, ...] = ()

DEFAULT_CODE_PACKAGE_EXCLUDE_OVERRIDE_ROOT_PATTERNS = (
    'data',
    'datasets',
    'local_remote_simulator/datasets',
    'runs/user_datasets',
)

DEFAULT_CODE_PACKAGE_HARD_EXCLUDE_ROOT_DIR_NAMES = (
    'workspace',
    'results',
)


@dataclass(frozen=True, slots=True)
class CodePackageFile:
    path: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            'path': self.path,
            'size_bytes': self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CodePackageExcludedPath:
    path: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            'path': self.path,
            'reason': self.reason,
        }


@dataclass(frozen=True, slots=True)
class CodePackageResult:
    source_root: str
    package_dir: str
    manifest_path: str
    entrypoint_relative_path: str
    included_files: list[CodePackageFile] = field(default_factory=list)
    excluded_paths: list[CodePackageExcludedPath] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(file.size_bytes for file in self.included_files)

    def to_dict(self) -> dict[str, object]:
        return {
            'source_root': self.source_root,
            'package_dir': self.package_dir,
            'manifest_path': self.manifest_path,
            'entrypoint_relative_path': self.entrypoint_relative_path,
            'included_files': [file.to_dict() for file in self.included_files],
            'excluded_paths': [path.to_dict() for path in self.excluded_paths],
            'file_count': len(self.included_files),
            'total_size_bytes': self.total_size_bytes,
        }


class CodePackageService:
    def __init__(
        self,
        default_exclude_patterns: tuple[str, ...] = DEFAULT_CODE_PACKAGE_EXCLUDE_PATTERNS,
        hard_exclude_patterns: tuple[str, ...] = DEFAULT_CODE_PACKAGE_HARD_EXCLUDE_PATTERNS,
        override_allow_patterns: tuple[str, ...] = DEFAULT_CODE_PACKAGE_EXCLUDE_OVERRIDE_ALLOW_PATTERNS,
        override_root_patterns: tuple[str, ...] = DEFAULT_CODE_PACKAGE_EXCLUDE_OVERRIDE_ROOT_PATTERNS,
        hard_exclude_root_dir_names: tuple[str, ...] = DEFAULT_CODE_PACKAGE_HARD_EXCLUDE_ROOT_DIR_NAMES,
    ) -> None:
        self.default_exclude_patterns = default_exclude_patterns
        self.hard_exclude_patterns = hard_exclude_patterns
        self.override_allow_patterns = override_allow_patterns
        self.override_root_patterns = override_root_patterns
        self.hard_exclude_root_dir_names = hard_exclude_root_dir_names

    def package(
        self,
        config: EngineExperimentConfig,
        workspace_dir: str | Path,
    ) -> CodePackageResult:
        package_dir = self._prepare_package_dir(Path(workspace_dir) / 'code_package')

        source_root, entrypoint_relative_path = self._resolve_source(config=config)
        include_patterns = tuple(config.code.package_include)
        default_exclude_patterns = tuple(self.default_exclude_patterns)
        hard_exclude_patterns = tuple(self.hard_exclude_patterns)
        override_allow_patterns = tuple(self.override_allow_patterns)
        override_root_patterns = tuple(self.override_root_patterns)
        hard_exclude_root_dir_names = tuple(self.hard_exclude_root_dir_names)
        package_exclude_patterns = tuple(config.code.package_exclude)
        included_files: list[CodePackageFile] = []
        excluded_paths: list[CodePackageExcludedPath] = []

        if config.code.working_dir is None:
            source_file = source_root / entrypoint_relative_path
            self._copy_file(
                source_file=source_file,
                destination_file=package_dir / entrypoint_relative_path,
                relative_path=entrypoint_relative_path.as_posix(),
                included_files=included_files,
            )
        else:
            self._copy_tree(
                source_root=source_root,
                package_dir=package_dir,
                entrypoint_relative_path=entrypoint_relative_path,
                include_patterns=include_patterns,
                default_exclude_patterns=default_exclude_patterns,
                hard_exclude_patterns=hard_exclude_patterns,
                override_allow_patterns=override_allow_patterns,
                override_root_patterns=override_root_patterns,
                hard_exclude_root_dir_names=hard_exclude_root_dir_names,
                package_exclude_patterns=package_exclude_patterns,
                included_files=included_files,
                excluded_paths=excluded_paths,
            )

        manifest_path = package_dir / 'code_package_manifest.json'
        result = CodePackageResult(
            source_root=str(source_root),
            package_dir=str(package_dir),
            manifest_path=str(manifest_path),
            entrypoint_relative_path=entrypoint_relative_path.as_posix(),
            included_files=sorted(included_files, key=lambda item: item.path),
            excluded_paths=sorted(excluded_paths, key=lambda item: item.path),
        )
        if not any(file.path == result.entrypoint_relative_path for file in result.included_files):
            raise ValueError(f'entrypoint is not included in code package: {result.entrypoint_relative_path}')

        manifest_path.write_text(
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        return result

    def _prepare_package_dir(self, package_dir: Path) -> Path:
        if package_dir.exists():
            try:
                shutil.rmtree(self._filesystem_path(package_dir))
            except OSError:
                package_dir = self._next_fallback_package_dir(package_dir.parent)
        Path(self._filesystem_path(package_dir)).mkdir(parents=True, exist_ok=True)
        return package_dir

    def _next_fallback_package_dir(self, workspace_dir: Path) -> Path:
        for index in range(1, 1000):
            candidate = workspace_dir / f'code_package_fallback_{os.getpid()}_{index}'
            if not candidate.exists():
                return candidate
        raise RuntimeError(f'could not allocate fallback code package directory under {workspace_dir}')

    def _resolve_source(self, config: EngineExperimentConfig) -> tuple[Path, Path]:
        entrypoint = Path(config.code.entrypoint).resolve()
        if not entrypoint.exists() or not entrypoint.is_file():
            raise FileNotFoundError(f'code.entrypoint does not exist: {entrypoint}')

        if config.code.working_dir is None:
            return entrypoint.parent, Path(entrypoint.name)

        source_root = Path(config.code.working_dir).resolve()
        if not source_root.exists() or not source_root.is_dir():
            raise FileNotFoundError(f'code.working_dir does not exist: {source_root}')

        try:
            entrypoint_relative_path = entrypoint.relative_to(source_root)
        except ValueError as error:
            raise ValueError('code.entrypoint must be inside code.working_dir for SSH code packaging') from error

        return source_root, entrypoint_relative_path

    def _copy_tree(
        self,
        source_root: Path,
        package_dir: Path,
        entrypoint_relative_path: Path,
        include_patterns: tuple[str, ...],
        default_exclude_patterns: tuple[str, ...],
        hard_exclude_patterns: tuple[str, ...],
        override_allow_patterns: tuple[str, ...],
        override_root_patterns: tuple[str, ...],
        hard_exclude_root_dir_names: tuple[str, ...],
        package_exclude_patterns: tuple[str, ...],
        included_files: list[CodePackageFile],
        excluded_paths: list[CodePackageExcludedPath],
    ) -> None:
        self._copy_directory_contents(
            source_dir=source_root,
            source_root=source_root,
            package_dir=package_dir,
            entrypoint_relative_path=entrypoint_relative_path,
            include_patterns=include_patterns,
            default_exclude_patterns=default_exclude_patterns,
            hard_exclude_patterns=hard_exclude_patterns,
            override_allow_patterns=override_allow_patterns,
            override_root_patterns=override_root_patterns,
            hard_exclude_root_dir_names=hard_exclude_root_dir_names,
            package_exclude_patterns=package_exclude_patterns,
            included_files=included_files,
            excluded_paths=excluded_paths,
        )

    def _copy_directory_contents(
        self,
        source_dir: Path,
        source_root: Path,
        package_dir: Path,
        entrypoint_relative_path: Path,
        include_patterns: tuple[str, ...],
        default_exclude_patterns: tuple[str, ...],
        hard_exclude_patterns: tuple[str, ...],
        override_allow_patterns: tuple[str, ...],
        override_root_patterns: tuple[str, ...],
        hard_exclude_root_dir_names: tuple[str, ...],
        package_exclude_patterns: tuple[str, ...],
        included_files: list[CodePackageFile],
        excluded_paths: list[CodePackageExcludedPath],
    ) -> None:
        for source_path in sorted(source_dir.iterdir(), key=lambda path: path.as_posix()):
            relative_path = source_path.relative_to(source_root)
            relative_posix = relative_path.as_posix()
            if source_path.is_symlink():
                excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_symlink'))
                continue

            if source_path.is_dir():
                if self._should_prune_directory(
                    relative_path=relative_path,
                    relative_posix=relative_posix,
                    include_patterns=include_patterns,
                    default_exclude_patterns=default_exclude_patterns,
                    hard_exclude_patterns=hard_exclude_patterns,
                    override_allow_patterns=override_allow_patterns,
                    override_root_patterns=override_root_patterns,
                    hard_exclude_root_dir_names=hard_exclude_root_dir_names,
                    package_exclude_patterns=package_exclude_patterns,
                    excluded_paths=excluded_paths,
                ):
                    continue

                self._copy_directory_contents(
                    source_dir=source_path,
                    source_root=source_root,
                    package_dir=package_dir,
                    entrypoint_relative_path=entrypoint_relative_path,
                    include_patterns=include_patterns,
                    default_exclude_patterns=default_exclude_patterns,
                    hard_exclude_patterns=hard_exclude_patterns,
                    override_allow_patterns=override_allow_patterns,
                    override_root_patterns=override_root_patterns,
                    hard_exclude_root_dir_names=hard_exclude_root_dir_names,
                    package_exclude_patterns=package_exclude_patterns,
                    included_files=included_files,
                    excluded_paths=excluded_paths,
                )
                continue

            explicitly_included = bool(include_patterns and self._matches_any(relative_posix, include_patterns))
            concretely_included = self._is_concretely_included_path(
                relative_posix=relative_posix,
                include_patterns=include_patterns,
            )
            if self._matches_any(relative_posix, package_exclude_patterns):
                if relative_path == entrypoint_relative_path:
                    raise ValueError(f'code.entrypoint is excluded by package_exclude: {relative_posix}')
                excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_file'))
                continue

            if not concretely_included and self._matches_any(relative_posix, hard_exclude_patterns):
                if relative_path == entrypoint_relative_path:
                    raise ValueError(f'code.entrypoint is excluded by hard package rules: {relative_posix}')
                excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_file'))
                continue

            if not concretely_included and self._is_default_excluded_without_allowed_override(
                relative_posix=relative_posix,
                include_patterns=include_patterns,
                default_exclude_patterns=default_exclude_patterns,
                override_allow_patterns=override_allow_patterns,
                override_root_patterns=override_root_patterns,
            ):
                if relative_path == entrypoint_relative_path:
                    raise ValueError(f'code.entrypoint is excluded by default package rules: {relative_posix}')
                excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_file'))
                continue

            if include_patterns and not self._matches_any(relative_posix, include_patterns):
                if relative_path == entrypoint_relative_path:
                    raise ValueError(f'code.entrypoint is not matched by package_include: {relative_posix}')
                excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='not_included'))
                continue

            self._copy_file(
                source_file=source_path,
                destination_file=package_dir / relative_path,
                relative_path=relative_posix,
                included_files=included_files,
            )

    def _should_prune_directory(
        self,
        relative_path: Path,
        relative_posix: str,
        include_patterns: tuple[str, ...],
        default_exclude_patterns: tuple[str, ...],
        hard_exclude_patterns: tuple[str, ...],
        override_allow_patterns: tuple[str, ...],
        override_root_patterns: tuple[str, ...],
        hard_exclude_root_dir_names: tuple[str, ...],
        package_exclude_patterns: tuple[str, ...],
        excluded_paths: list[CodePackageExcludedPath],
    ) -> bool:
        if self._matches_any(relative_posix, package_exclude_patterns):
            excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_directory'))
            return True

        if self._is_hard_excluded_directory(
            relative_path=relative_path,
            relative_posix=relative_posix,
            hard_exclude_patterns=hard_exclude_patterns,
            hard_exclude_root_dir_names=hard_exclude_root_dir_names,
        ):
            excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_directory'))
            return True

        if self._is_default_excluded_without_allowed_override(
            relative_posix=relative_posix,
            include_patterns=include_patterns,
            default_exclude_patterns=default_exclude_patterns,
            override_allow_patterns=override_allow_patterns,
            override_root_patterns=override_root_patterns,
        ) and not self._directory_may_contain_allowed_override(
            relative_posix=relative_posix,
            include_patterns=include_patterns,
            override_allow_patterns=override_allow_patterns,
            override_root_patterns=override_root_patterns,
        ):
            excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='excluded_directory'))
            return True

        if include_patterns and not self._directory_may_contain_included_path(
            relative_posix=relative_posix,
            include_patterns=include_patterns,
        ):
            excluded_paths.append(CodePackageExcludedPath(path=relative_posix, reason='not_included'))
            return True

        return False

    def _is_hard_excluded_directory(
        self,
        relative_path: Path,
        relative_posix: str,
        hard_exclude_patterns: tuple[str, ...],
        hard_exclude_root_dir_names: tuple[str, ...],
    ) -> bool:
        if len(relative_path.parts) == 1 and relative_path.parts[0] in hard_exclude_root_dir_names:
            return True

        return self._matches_any(relative_posix, hard_exclude_patterns)

    def _directory_may_contain_allowed_override(
        self,
        relative_posix: str,
        include_patterns: tuple[str, ...],
        override_allow_patterns: tuple[str, ...],
        override_root_patterns: tuple[str, ...],
    ) -> bool:
        return (
            any(
                self._path_is_same_or_parent(parent=relative_posix, child=self._pattern_prefix(pattern))
                and any(
                    self._include_pattern_may_match_path(include_pattern=include_pattern, path=pattern)
                    for include_pattern in include_patterns
                )
                for pattern in override_allow_patterns
            )
            or any(
                self._include_pattern_allows_default_exclude_override(
                    include_pattern=include_pattern,
                    relative_posix=relative_posix,
                    override_root_patterns=override_root_patterns,
                    allow_parent_match=True,
                )
                for include_pattern in include_patterns
            )
        )

    def _directory_may_contain_included_path(
        self,
        relative_posix: str,
        include_patterns: tuple[str, ...],
    ) -> bool:
        return self._matches_any(relative_posix, include_patterns) or any(
            self._path_is_same_or_parent(parent=relative_posix, child=pattern)
            for pattern in include_patterns
        )

    def _path_is_same_or_parent(self, parent: str, child: str) -> bool:
        normalized_parent = parent.rstrip('/')
        normalized_child = child.rstrip('/')
        child_prefix = normalized_child.split('*', maxsplit=1)[0].rstrip('/')

        return (
            normalized_child == normalized_parent
            or normalized_child.startswith(f'{normalized_parent}/')
            or child_prefix == normalized_parent
            or child_prefix.startswith(f'{normalized_parent}/')
        )

    def _include_pattern_may_match_path(self, include_pattern: str, path: str) -> bool:
        path_prefix = self._pattern_prefix(path)

        return (
            fnmatch.fnmatch(path_prefix, include_pattern)
            or self._path_is_same_or_parent(parent=path_prefix, child=include_pattern)
        )

    def _pattern_prefix(self, pattern: str) -> str:
        return pattern.split('*', maxsplit=1)[0].rstrip('/')

    def _is_default_excluded_without_allowed_override(
        self,
        relative_posix: str,
        include_patterns: tuple[str, ...],
        default_exclude_patterns: tuple[str, ...],
        override_allow_patterns: tuple[str, ...],
        override_root_patterns: tuple[str, ...],
    ) -> bool:
        if not self._matches_any(relative_posix, default_exclude_patterns):
            return False

        explicitly_included = bool(include_patterns and self._matches_any(relative_posix, include_patterns))
        if not explicitly_included:
            return True

        if self._matches_any(relative_posix, override_allow_patterns):
            return False

        return not any(
            self._include_pattern_allows_default_exclude_override(
                include_pattern=include_pattern,
                relative_posix=relative_posix,
                override_root_patterns=override_root_patterns,
                allow_parent_match=False,
            )
            for include_pattern in include_patterns
        )

    def _include_pattern_allows_default_exclude_override(
        self,
        *,
        include_pattern: str,
        relative_posix: str,
        override_root_patterns: tuple[str, ...],
        allow_parent_match: bool,
    ) -> bool:
        normalized_pattern = include_pattern.strip().strip('/')
        root_pattern = self._matching_override_root_pattern(
            include_pattern=normalized_pattern,
            override_root_patterns=override_root_patterns,
        )
        if root_pattern is None:
            return False
        child_suffix = normalized_pattern[len(root_pattern):].strip('/')
        child_parts = child_suffix.split('/')
        if not child_parts:
            return False
        concrete_child = child_parts[0]
        if not concrete_child or concrete_child in {'.', '..'} or any(char in concrete_child for char in '*?['):
            return False
        if allow_parent_match:
            return self._path_is_same_or_parent(parent=relative_posix, child=self._pattern_prefix(normalized_pattern))

        return fnmatch.fnmatch(relative_posix, normalized_pattern)

    def _matching_override_root_pattern(
        self,
        *,
        include_pattern: str,
        override_root_patterns: tuple[str, ...],
    ) -> str | None:
        for root_pattern in sorted((pattern.strip().strip('/') for pattern in override_root_patterns), key=len, reverse=True):
            if include_pattern == root_pattern or include_pattern.startswith(f'{root_pattern}/'):
                return root_pattern

        return None

    def _is_concretely_included_path(self, *, relative_posix: str, include_patterns: tuple[str, ...]) -> bool:
        return any(
            pattern.strip().strip('/') == relative_posix
            for pattern in include_patterns
            if not any(char in pattern for char in '*?[')
        )

    def _copy_file(
        self,
        source_file: Path,
        destination_file: Path,
        relative_path: str,
        included_files: list[CodePackageFile],
    ) -> None:
        source_path = self._filesystem_path(source_file)
        destination_path = self._filesystem_path(destination_file)
        Path(self._filesystem_path(destination_file.parent)).mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_path, destination_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'failed to copy code package file: '
                f'source={source_file}; destination={destination_file}; relative={relative_path}',
            ) from error
        included_files.append(
            CodePackageFile(
                path=relative_path,
                size_bytes=os.stat(source_path).st_size,
            ),
        )

    def _filesystem_path(self, path: Path) -> str:
        text = str(path)
        if os.name != 'nt':
            return text
        if text.startswith('\\\\?\\'):
            return text
        if text.startswith('\\\\'):
            return '\\\\?\\UNC\\' + text[2:]
        if path.is_absolute():
            return '\\\\?\\' + text

        return text

    def _matches_any(self, relative_path: str, patterns: tuple[str, ...]) -> bool:
        parts = relative_path.split('/')
        return any(
            fnmatch.fnmatch(relative_path, pattern)
            or (
                pattern not in ROOT_SCOPED_DEFAULT_EXCLUDE_NAMES
                and any(fnmatch.fnmatch(part, pattern) for part in parts)
            )
            for pattern in patterns
        )
