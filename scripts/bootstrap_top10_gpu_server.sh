#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
D_FINE_REPO="${D_FINE_REPO:-$WORKSPACE_ROOT/D-FINE}"
DINOV3_REPO="${DINOV3_REPO:-$WORKSPACE_ROOT/dinov3}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
    return 0
  fi
  "$@"
}

run_shell() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
    return 0
  fi
  bash -lc "$*"
}

if [[ "$DRY_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

run "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# Keep the CUDA/PyTorch stack supplied by the Vast template. Installing torch
# here can silently replace a working CUDA build with a CPU or mismatched wheel.
run_shell "$PIP_BIN install --upgrade pandas numpy pillow opencv-python-headless pyyaml tqdm"
run_shell "$PIP_BIN install --upgrade ultralytics timm open_clip_torch supervision pycocotools faster-coco-eval"
run_shell "$PIP_BIN install --upgrade 'rfdetr[train,loggers]' sam2 termcolor calflops loguru"

if [[ ! -d "$D_FINE_REPO/.git" ]]; then
  run git clone https://github.com/Peterande/D-FINE.git "$D_FINE_REPO"
fi
if [[ -f "$D_FINE_REPO/requirements.txt" ]]; then
  run_shell "$PIP_BIN install -r '$D_FINE_REPO/requirements.txt'"
fi

if [[ ! -d "$DINOV3_REPO/.git" ]]; then
  run git clone https://github.com/facebookresearch/dinov3.git "$DINOV3_REPO"
fi
if [[ -f "$DINOV3_REPO/requirements.txt" ]]; then
  run_shell "$PIP_BIN install -r '$DINOV3_REPO/requirements.txt'"
fi

run "$PYTHON_BIN" - <<'PY'
import importlib

packages = [
    'torch',
    'ultralytics',
    'timm',
    'open_clip',
    'sam2',
    'rfdetr',
    'termcolor',
]
missing = [name for name in packages if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('missing packages: ' + ', '.join(missing))

import torch
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
PY

echo "Top10 GPU bootstrap completed."
