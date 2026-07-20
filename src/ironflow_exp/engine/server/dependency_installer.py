import shlex
from dataclasses import dataclass, field

from ironflow_exp.engine.domain import ServerRecord
from ironflow_exp.engine.server.checker import (
    REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES,
    special_dependency_checks_for_profile,
)
from ironflow_exp.engine.server.ssh_client import BaseSshClient, OpenSshClient, SshCommandResult


MODULE_TO_PIP_PACKAGE = {
    'yaml': 'PyYAML',
    'PIL': 'Pillow',
    'pandas': 'pandas',
    'polars': 'polars',
    'cv2': 'opencv-python-headless',
    'albumentations': 'albumentations',
    'numpy': 'numpy',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'timm': 'timm',
    'ultralytics': 'ultralytics',
    'rfdetr': 'rfdetr[train,loggers]',
    'sam2': 'sam2',
    'mmcv': 'mmcv',
    'mmdet': 'mmdet',
    'transformers': 'transformers',
    'scipy': 'scipy',
    'tensorboard': 'tensorboard',
    'calflops': 'calflops',
    'loguru': 'loguru',
    'faster_coco_eval': 'faster-coco-eval',
    'open_clip_torch': 'open_clip_torch',
    'sahi': 'sahi',
}

FORCED_PIP_PACKAGES_BY_PROFILE = {
    # ``import rfdetr`` can succeed even when the training/logging extras are
    # missing. RF-DETR then fails only at model.train(), so install the extras
    # for native RF-DETR profiles regardless of the base import probe result.
    'foundation_native': ('rfdetr[train,loggers]',),
    'native_rfdetr_timm': ('rfdetr[train,loggers]',),
}

REMOTE_REPOSITORIES_BY_PROFILE = {
    'native_d_fine_torchvision': (
        {
            'name': 'D-FINE',
            'url': 'https://github.com/Peterande/D-FINE.git',
            'path': '/workspace/D-FINE',
            'env_path': 'D_FINE_REPO',
            'required_files': (
                'train.py',
                'configs/dfine/dfine_hgnetv2_n_coco.yml',
            ),
        },
    ),
}

MODULE_TO_APT_PACKAGES = {
    'cv2': (
        'libxcb1',
        'libgl1',
        'libglib2.0-0',
        'libxext6',
        'libsm6',
        'libgomp1',
    ),
    'ultralytics': (
        'libxcb1',
        'libgl1',
        'libglib2.0-0',
        'libxext6',
        'libsm6',
        'libgomp1',
    ),
}


@dataclass(frozen=True, slots=True)
class RemoteDependencyInstallPlan:
    dependency_profile: str
    packages: tuple[str, ...]
    command: str
    system_packages: tuple[str, ...] = ()
    probe_modules: tuple[tuple[str, str], ...] = ()
    special_checks: tuple[str, ...] = ()
    forced_packages: tuple[str, ...] = ()
    remote_repositories: tuple[dict[str, object], ...] = ()
    torch_index_url: str | None = None
    upgrade: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            'dependency_profile': self.dependency_profile,
            'packages': list(self.packages),
            'system_packages': list(self.system_packages),
            'probe_modules': [
                {'name': name, 'module': module_name}
                for name, module_name in self.probe_modules
            ],
            'special_checks': list(self.special_checks),
            'forced_packages': list(self.forced_packages),
            'remote_repositories': list(self.remote_repositories),
            'command': self.command,
            'torch_index_url': self.torch_index_url,
            'upgrade': self.upgrade,
        }


@dataclass(frozen=True, slots=True)
class RemoteDependencyInstallResult:
    success: bool
    status: str
    message: str
    plan: RemoteDependencyInstallPlan
    command_result: SshCommandResult | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            'success': self.success,
            'status': self.status,
            'message': self.message,
            'plan': self.plan.to_dict(),
            'command_result': None if self.command_result is None else {
                'command': self.command_result.command,
                'exit_code': self.command_result.exit_code,
                'stdout_tail': self.command_result.stdout[-2000:],
                'stderr_tail': self.command_result.stderr[-2000:],
            },
            'metadata': self.metadata,
        }


class RemoteDependencyInstaller:
    def __init__(self, ssh_client: BaseSshClient | None = None) -> None:
        self.ssh_client = ssh_client or OpenSshClient()

    def plan(
        self,
        *,
        dependency_profile: str,
        torch_index_url: str | None = None,
        upgrade: bool = True,
    ) -> RemoteDependencyInstallPlan:
        checks = REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES.get(dependency_profile)
        if checks is None:
            allowed = ', '.join(REMOTE_TASK_ADAPTER_DEPENDENCY_PROFILES)
            raise ValueError(f'unknown dependency profile: {dependency_profile}; allowed: {allowed}')

        packages = self._packages_for_checks(checks=checks)
        system_packages = self._system_packages_for_checks(checks=checks)
        special_checks = special_dependency_checks_for_profile(dependency_profile=dependency_profile)
        forced_packages = self._forced_packages_for_profile(dependency_profile=dependency_profile)
        remote_repositories = self._remote_repositories_for_profile(dependency_profile=dependency_profile)
        command = self._install_command(
            packages=packages,
            system_packages=system_packages,
            probe_modules=checks,
            special_checks=special_checks,
            forced_packages=forced_packages,
            remote_repositories=remote_repositories,
            torch_index_url=torch_index_url,
            upgrade=upgrade,
        )

        return RemoteDependencyInstallPlan(
            dependency_profile=dependency_profile,
            packages=packages,
            system_packages=system_packages,
            probe_modules=checks,
            special_checks=special_checks,
            forced_packages=forced_packages,
            remote_repositories=remote_repositories,
            command=command,
            torch_index_url=torch_index_url,
            upgrade=upgrade,
        )

    def install(
        self,
        *,
        server: ServerRecord,
        dependency_profile: str,
        torch_index_url: str | None = None,
        upgrade: bool = True,
        timeout_seconds: int | None = 900,
    ) -> RemoteDependencyInstallResult:
        plan = self.plan(
            dependency_profile=dependency_profile,
            torch_index_url=torch_index_url,
            upgrade=upgrade,
        )
        result = self.ssh_client.run_command(
            server=server,
            command=plan.command,
            timeout_seconds=timeout_seconds,
        )
        if result.is_success:
            return RemoteDependencyInstallResult(
                success=True,
                status='ok',
                message='remote dependency install completed',
                plan=plan,
                command_result=result,
                metadata={'server_type': server.server_type},
            )

        return RemoteDependencyInstallResult(
            success=False,
            status='failed',
            message='remote dependency install failed',
            plan=plan,
            command_result=result,
            metadata={'server_type': server.server_type},
        )

    def _packages_for_checks(self, *, checks: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
        packages: list[str] = []
        seen: set[str] = set()
        for name, _module_name in checks:
            package = MODULE_TO_PIP_PACKAGE.get(name)
            if package is None or package in seen:
                continue
            seen.add(package)
            packages.append(package)

        return tuple(packages)

    def _system_packages_for_checks(self, *, checks: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
        packages: list[str] = []
        seen: set[str] = set()
        for name, _module_name in checks:
            for package in MODULE_TO_APT_PACKAGES.get(name, ()):
                if package in seen:
                    continue
                seen.add(package)
                packages.append(package)

        return tuple(packages)

    def _forced_packages_for_profile(self, *, dependency_profile: str) -> tuple[str, ...]:
        return FORCED_PIP_PACKAGES_BY_PROFILE.get(dependency_profile, ())

    def _remote_repositories_for_profile(self, *, dependency_profile: str) -> tuple[dict[str, object], ...]:
        return REMOTE_REPOSITORIES_BY_PROFILE.get(dependency_profile, ())

    def _install_command(
        self,
        *,
        packages: tuple[str, ...],
        system_packages: tuple[str, ...],
        probe_modules: tuple[tuple[str, str], ...],
        special_checks: tuple[str, ...],
        forced_packages: tuple[str, ...],
        remote_repositories: tuple[dict[str, object], ...],
        torch_index_url: str | None,
        upgrade: bool,
    ) -> str:
        if not packages:
            raise ValueError('dependency install plan has no packages')

        python_selector = (
            'PYTHON_BIN="${IRONFLOW_REMOTE_PYTHON:-}"; '
            'if [ -z "$PYTHON_BIN" ] && [ -x /venv/main/bin/python ]; then '
            'PYTHON_BIN=/venv/main/bin/python; '
            'fi; '
            'if [ -z "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi; '
        )
        probe_and_install_command = self._probe_and_install_command(
            probe_modules=probe_modules,
            special_checks=special_checks,
            forced_packages=forced_packages,
            remote_repositories=remote_repositories,
            torch_index_url=torch_index_url,
            upgrade=upgrade,
        )

        return f'{python_selector}"$PYTHON_BIN" {self._quote(["-c", probe_and_install_command])}'

    def _probe_and_install_command(
        self,
        *,
        probe_modules: tuple[tuple[str, str], ...],
        special_checks: tuple[str, ...],
        forced_packages: tuple[str, ...],
        remote_repositories: tuple[dict[str, object], ...],
        torch_index_url: str | None,
        upgrade: bool,
    ) -> str:
        package_by_name = {
            name: MODULE_TO_PIP_PACKAGE[name]
            for name, _module_name in probe_modules
            if name in MODULE_TO_PIP_PACKAGE
        }
        apt_by_name = {
            name: list(MODULE_TO_APT_PACKAGES.get(name, ()))
            for name, _module_name in probe_modules
            if name in MODULE_TO_APT_PACKAGES
        }
        return (
            'import importlib, json, os, re, shutil, subprocess, sys\n'
            f'checks = {self._python_literal(probe_modules)}\n'
            f'special_checks = {self._python_literal(special_checks)}\n'
            f'forced_pip_packages = {self._python_literal(forced_packages)}\n'
            f'remote_repositories = {self._python_literal(remote_repositories)}\n'
            f'package_by_name = {self._python_literal(package_by_name)}\n'
            f'apt_by_name = {self._python_literal(apt_by_name)}\n'
            f'torch_index_url = {self._python_literal(torch_index_url.strip() if torch_index_url else None)}\n'
            f'upgrade = {self._python_literal(upgrade)}\n'
            '\n'
            'def emit(event, **payload):\n'
            '    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)\n'
            '\n'
            'def probe():\n'
            '    results = {}\n'
            '    missing = []\n'
            '    for name, module_name in checks:\n'
            '        try:\n'
            '            module = importlib.import_module(module_name)\n'
            '            results[name] = {"ok": True, "version": str(getattr(module, "__version__", ""))}\n'
            '        except Exception as error:\n'
            '            missing.append(name)\n'
            '            results[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}\n'
            '    if "ultralytics_yolo" in special_checks:\n'
            '        try:\n'
            '            from ultralytics import YOLO\n'
            '            results["ultralytics_yolo"] = {"ok": True, "object": str(YOLO)}\n'
            '        except Exception as error:\n'
            '            if "ultralytics" not in missing:\n'
            '                missing.append("ultralytics")\n'
            '            results["ultralytics_yolo"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}\n'
            '    return not missing, results, missing\n'
            '\n'
            'def dedupe(values):\n'
            '    seen = set()\n'
            '    ordered = []\n'
            '    for value in values:\n'
            '        if value in seen:\n'
            '            continue\n'
            '        seen.add(value)\n'
            '        ordered.append(value)\n'
            '    return ordered\n'
            '\n'
            'pip_repair_denylist = {"torch", "torchvision", "torchaudio"}\n'
            '\n'
            'def package_from_uninstall_no_record_error(text):\n'
            '    if "uninstall-no-record-file" not in text and "Cannot uninstall" not in text:\n'
            '        return None\n'
            '    match = re.search(r"Cannot uninstall ([A-Za-z0-9_.-]+)", text)\n'
            '    if not match:\n'
            '        return None\n'
            '    package = match.group(1).strip()\n'
            '    package_key = package.lower()\n'
            '    if package_key in pip_repair_denylist or package_key.startswith("nvidia-"):\n'
            '        return None\n'
            '    return package\n'
            '\n'
            'def run_and_echo(command):\n'
            '    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n'
            '    if result.stdout:\n'
            '        print(result.stdout, end="", flush=True)\n'
            '    if result.stderr:\n'
            '        print(result.stderr, end="", file=sys.stderr, flush=True)\n'
            '    return result\n'
            '\n'
            'def run_pip_install(command, *, packages):\n'
            '    repaired = set()\n'
            '    for attempt in range(1, 9):\n'
            '        emit("pip_install", status="running", packages=packages, upgrade=upgrade, attempt=attempt)\n'
            '        result = run_and_echo(command)\n'
            '        if result.returncode == 0:\n'
            '            return 0\n'
            '        failed_text = (result.stdout or "") + "\\n" + (result.stderr or "")\n'
            '        repair_package = package_from_uninstall_no_record_error(failed_text)\n'
            '        if not repair_package or repair_package.lower() in repaired:\n'
            '            return result.returncode\n'
            '        repaired.add(repair_package.lower())\n'
            '        repair_command = [sys.executable, "-m", "pip", "install", "--ignore-installed", repair_package]\n'
            '        emit("pip_repair", status="running", package=repair_package, reason="uninstall-no-record-file")\n'
            '        repair_result = run_and_echo(repair_command)\n'
            '        if repair_result.returncode != 0:\n'
            '            return result.returncode\n'
            '    emit("pip_repair", status="failed", reason="too_many_repairs", packages=sorted(repaired))\n'
            '    return 1\n'
            '\n'
            'def ensure_remote_repositories():\n'
            '    for setup in remote_repositories:\n'
            '        name = str(setup["name"])\n'
            '        url = str(setup["url"])\n'
            '        env_path = str(setup.get("env_path") or "")\n'
            '        default_path = str(setup["path"])\n'
            '        repo_dir = os.environ.get(env_path, default_path) if env_path else default_path\n'
            '        git_dir = os.path.join(repo_dir, ".git")\n'
            '        if not os.path.isdir(git_dir):\n'
            '            git_bin = shutil.which("git")\n'
            '            if git_bin is None:\n'
            '                emit("remote_repo", status="failed", name=name, reason="git_not_available", path=repo_dir)\n'
            '                raise SystemExit(127)\n'
            '            os.makedirs(os.path.dirname(os.path.abspath(repo_dir)) or ".", exist_ok=True)\n'
            '            emit("remote_repo", status="cloning", name=name, url=url, path=repo_dir)\n'
            '            clone = run_and_echo([git_bin, "clone", url, repo_dir])\n'
            '            if clone.returncode != 0:\n'
            '                emit("remote_repo", status="failed", name=name, reason="clone_failed", path=repo_dir)\n'
            '                raise SystemExit(clone.returncode)\n'
            '        else:\n'
            '            emit("remote_repo", status="already_present", name=name, path=repo_dir)\n'
            '        missing_files = []\n'
            '        for relative_path in setup.get("required_files", ()):\n'
            '            if not os.path.exists(os.path.join(repo_dir, str(relative_path))):\n'
            '                missing_files.append(str(relative_path))\n'
            '        if missing_files:\n'
            '            emit("remote_repo", status="failed", name=name, reason="missing_required_files", path=repo_dir, missing=missing_files)\n'
            '            raise SystemExit(2)\n'
            '        emit("remote_repo", status="ok", name=name, path=repo_dir)\n'
            '\n'
            'success, results, missing = probe()\n'
            'emit("dependency_probe", success=success, missing=missing, checks=results)\n'
            'if success and not forced_pip_packages and not remote_repositories:\n'
            '    emit("dependency_install", status="already_satisfied")\n'
            '    raise SystemExit(0)\n'
            '\n'
            'system_packages = dedupe(pkg for name in missing for pkg in apt_by_name.get(name, []))\n'
            'if system_packages:\n'
            '    if shutil.which("apt-get") is None:\n'
            '        emit("system_install", status="skipped", reason="apt-get not available", packages=system_packages)\n'
            '    else:\n'
            '        if os.geteuid() == 0:\n'
            '            prefix = []\n'
            '        elif shutil.which("sudo"):\n'
            '            prefix = ["sudo"]\n'
            '        else:\n'
            '            emit("system_install", status="failed", reason="root_or_sudo_required", packages=system_packages)\n'
            '            raise SystemExit(126)\n'
            '        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")\n'
            '        emit("system_install", status="running", packages=system_packages)\n'
            '        update = subprocess.run([*prefix, "apt-get", "update"], env=env)\n'
            '        if update.returncode != 0:\n'
            '            raise SystemExit(update.returncode)\n'
            '        install = subprocess.run([*prefix, "apt-get", "install", "-y", "--no-install-recommends", *system_packages], env=env)\n'
            '        if install.returncode != 0:\n'
            '            raise SystemExit(install.returncode)\n'
            '\n'
            'pip_packages = dedupe([package_by_name[name] for name in missing if name in package_by_name] + list(forced_pip_packages))\n'
            'if pip_packages:\n'
            '    command = [sys.executable, "-m", "pip", "install"]\n'
            '    if upgrade:\n'
            '        command.append("--upgrade")\n'
            '    if torch_index_url:\n'
            '        command.extend(["--extra-index-url", torch_index_url])\n'
            '    command.extend(pip_packages)\n'
            '    pip_exit_code = run_pip_install(command, packages=pip_packages)\n'
            '    if pip_exit_code != 0:\n'
            '        raise SystemExit(pip_exit_code)\n'
            'else:\n'
            '    emit("pip_install", status="skipped", reason="no mapped missing packages")\n'
            '\n'
            'success, results, missing = probe()\n'
            'emit("dependency_probe_after_install", success=success, missing=missing, checks=results)\n'
            'if success:\n'
            '    ensure_remote_repositories()\n'
            'raise SystemExit(0 if success else 2)\n'
        )

    def _quote(self, parts: list[str]) -> str:
        return ' '.join(shlex.quote(part) for part in parts)

    def _python_literal(self, value: object) -> str:
        return repr(value)
