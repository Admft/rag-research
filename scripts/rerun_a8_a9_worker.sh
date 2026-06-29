#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SUPERVISOR="$ROOT/run_ablation_resilient.py"
A8="$ROOT/experiments/Results/Test Runs and Ablations/Ablation 8 - Top-K"

if [[ ! -d "$A8/top7_PRE_FIX" ]]; then
  echo "=== Preparing *_PRE_FIX archives ==="
  bash "$ROOT/scripts/prepare_prefix_reruns.sh"
else
  echo "=== *_PRE_FIX folders already exist, skipping prepare ==="
fi

echo "=== Waiting for in-flight rescore jobs to finish ==="
while pgrep -f "scripts/rescore_run.py" >/dev/null 2>&1; do
  echo "$(date -Iseconds) rescore still running..."
  sleep 30
done
echo "$(date -Iseconds) rescore complete (or none running)."

run_one() {
  local ablation="$1"
  local condition="$2"
  echo ""
  echo "========================================================================"
  echo "$(date -Iseconds) Starting $ablation / $condition (3 runs, --force)"
  echo "========================================================================"
  "$PYTHON" "$SUPERVISOR" --safe --repair-venv --until-complete \
    --ablation "$ablation" --condition "$condition" --runs 3 --force
}

run_one A8 top7
run_one A8 top10
run_one A9 512

echo ""
echo "=== top7 PRE_FIX vs fresh comparison ==="
"$PYTHON" "$ROOT/scripts/compare_top7_rescore.py"

echo ""
echo "$(date -Iseconds) All rerun jobs finished."
