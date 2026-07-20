#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-}"
RELEASE_DIR="${2:-}"
DATASET_ID="${3:-tank_armor_prepared_v20260630}"
PART_SIZE="${PART_SIZE:-4G}"

if [[ -z "$DATASET_ROOT" || -z "$RELEASE_DIR" ]]; then
  echo "usage: $0 DATASET_ROOT RELEASE_DIR [DATASET_ID]" >&2
  exit 2
fi

case "$PWD" in
  /mnt/c/*)
    echo "[WARN] Current path is under /mnt/c. For faster packaging, copy or generate the dataset under /home and run there if possible." >&2
    ;;
esac

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] dataset root does not exist: $DATASET_ROOT" >&2
  exit 2
fi

mkdir -p "$RELEASE_DIR"
rm -f "$RELEASE_DIR"/part-*.bin "$RELEASE_DIR"/manifest.sha256 "$RELEASE_DIR"/restore_commands.txt

TAR_NAME="ironflow_dataset_${DATASET_ID}.tar"
PARENT_DIR="$(cd "$(dirname "$DATASET_ROOT")" && pwd)"
BASE_NAME="$(basename "$DATASET_ROOT")"

(
  cd "$PARENT_DIR"
  tar -cf - "$BASE_NAME"
) | split -b "$PART_SIZE" -d -a 4 - "$RELEASE_DIR/part-"

for f in "$RELEASE_DIR"/part-*; do
  mv "$f" "$f.bin"
done

PART_COUNT="$(find "$RELEASE_DIR" -maxdepth 1 -name 'part-*.bin' -type f | wc -l | tr -d ' ')"
TOTAL_BYTES="$(find "$RELEASE_DIR" -maxdepth 1 -name 'part-*.bin' -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"

cat > "$RELEASE_DIR/restore_commands.txt" <<EOF
cd /workspace/ironflow/drive_dataset_parts/${DATASET_ID}
cat part-*.bin > ${TAR_NAME}
mkdir -p /workspace/ironflow/prestaged
tar -xf ${TAR_NAME} -C /workspace/ironflow/prestaged
ls /workspace/ironflow/prestaged/${DATASET_ID}
EOF

echo "part_count=${PART_COUNT}"
echo "total_bytes=${TOTAL_BYTES}"
echo "restore_commands=${RELEASE_DIR}/restore_commands.txt"
