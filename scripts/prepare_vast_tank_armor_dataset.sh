#!/usr/bin/env bash
set -euo pipefail

DRIVE_URL="${DRIVE_URL:-https://drive.google.com/drive/folders/YOUR_PREPARED_DATASET_FOLDER_ID}"
DATASET_ID="${DATASET_ID:-tank_armor_prepared_v20260703_original_classifier_all_v1}"
WORK_ROOT="${WORK_ROOT:-/workspace/ironflow}"
PARTS_ROOT="${PARTS_ROOT:-$WORK_ROOT/drive_dataset_parts/$DATASET_ID}"
EXTRACT_ROOT="${EXTRACT_ROOT:-$WORK_ROOT/prestaged}"
TAR_NAME="${TAR_NAME:-ironflow_dataset_tank_armor_prepared_v20260703_original_classifier_all_v1.tar}"
DATASET_ROOT="$EXTRACT_ROOT/$DATASET_ID"

echo "[IronFlow] Preparing tank/armored-vehicle dataset on Vast"
echo "[IronFlow] Drive URL    : $DRIVE_URL"
echo "[IronFlow] Parts root   : $PARTS_ROOT"
echo "[IronFlow] Extract root : $EXTRACT_ROOT"
echo "[IronFlow] Dataset root : $DATASET_ROOT"
echo

python3 -m pip install -q gdown
mkdir -p "$PARTS_ROOT" "$EXTRACT_ROOT"

if ! compgen -G "$PARTS_ROOT/part-*.bin" > /dev/null; then
  gdown --folder "$DRIVE_URL" -O "$PARTS_ROOT"
else
  echo "[IronFlow] Existing part files found. Reusing: $PARTS_ROOT"
fi

if ! compgen -G "$PARTS_ROOT/part-*.bin" > /dev/null; then
  nested_parts_dir="$(find "$PARTS_ROOT" -type f -name 'part-*.bin' -printf '%h\n' | sort -u | head -n 1 || true)"
  if [[ -n "$nested_parts_dir" ]]; then
    PARTS_ROOT="$nested_parts_dir"
  fi
fi

if ! compgen -G "$PARTS_ROOT/part-*.bin" > /dev/null; then
  echo "[IronFlow] No part-*.bin files found under: $PARTS_ROOT" >&2
  exit 1
fi

cd "$PARTS_ROOT"
cat part-*.bin > "$TAR_NAME"
tar -xf "$TAR_NAME" -C "$EXTRACT_ROOT"

echo
echo "[IronFlow] Vast dataset is ready:"
echo "$DATASET_ROOT"
echo
echo "[IronFlow] GUI Remote Root:"
echo "$DATASET_ROOT"
