#!/usr/bin/env bash
# Rename affected ablation conditions to *_PRE_FIX and restore original broken
# scores for top10 / 512 from experiments/Results/runs/ archive.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A8="$ROOT/experiments/Results/Test Runs and Ablations/Ablation 8 - Top-K"
A9="$ROOT/experiments/Results/Test Runs and Ablations/Ablation 9 - Chunk Size"
RUNS="$ROOT/experiments/Results/runs"

rename_if_exists() {
  local parent="$1"
  local name="$2"
  if [[ -d "$parent/$name" && ! -d "$parent/${name}_PRE_FIX" ]]; then
    mv "$parent/$name" "$parent/${name}_PRE_FIX"
    echo "Renamed $name -> ${name}_PRE_FIX"
  elif [[ -d "$parent/${name}_PRE_FIX" ]]; then
    echo "${name}_PRE_FIX already exists, skipping rename"
  else
    echo "WARNING: $parent/$name not found"
  fi
}

echo "=== Renaming affected conditions to *_PRE_FIX ==="
rename_if_exists "$A8" top7
rename_if_exists "$A8" top10
rename_if_exists "$A9" 512

echo ""
echo "=== Restoring original pre-fix scores for top10 and 512 ==="
bash "$ROOT/scripts/restore_prefix_archives.sh"

echo ""
echo "=== Re-scoring top7_PRE_FIX with fixed judge (for comparison) ==="
"$ROOT/.venv/bin/python" "$ROOT/scripts/rescore_run.py" \
  --ablation-folder "Ablation 8 - Top-K" \
  --condition top7_PRE_FIX \
  --progress

echo ""
echo "Done. PRE_FIX folders preserved; ready for fresh runs of top7, top10, 512."
