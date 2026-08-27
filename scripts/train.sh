#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:?Usage: scripts/train.sh social-last|social-full|text-last|text-full [dataset]}"
DATASET="${2:-eth}"

case "$VARIANT" in
  social-last|social-full|text-last|text-full) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

cd "$ROOT"
python -m accelerate.commands.launch -m lcsif.train --config "configs/$VARIANT.json" --dataset "$DATASET"

