#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SUPERVISOR="$ROOT/run_ablation_resilient.py"

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

run_one A8 top10
run_one A9 512

echo ""
echo "$(date -Iseconds) All rerun jobs finished."
