#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: scripts/prepare_dataset.sh DATASET RAW_ROOT OUTPUT_ROOT}"
RAW_ROOT="${2:?Missing raw dataset root}"
OUTPUT_ROOT="${3:?Missing output dataset root}"

for SPLIT in train val test; do
  EXTRA=()
  if [[ "$SPLIT" == "train" ]]; then EXTRA+=(--multimodal); fi
  python -m lcsif.preprocessing --input-root "$RAW_ROOT" --output-root "$OUTPUT_ROOT" \
    --dataset "$DATASET" --split "$SPLIT" --metric meter "${EXTRA[@]}"
done

