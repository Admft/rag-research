#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A8="$ROOT/experiments/Results/Test Runs and Ablations/Ablation 8 - Top-K"
A9="$ROOT/experiments/Results/Test Runs and Ablations/Ablation 9 - Chunk Size"
RUNS="$ROOT/experiments/Results/runs"

restore_run() {
  local archive="$1"
  local dest="$2"
  if [[ ! -d "$archive" ]]; then
    echo "WARNING: archive missing: $archive"
    return
  fi
  rm -rf "$dest"
  mkdir -p "$dest"
  for f in questions.jsonl data.json REPORT.txt .ablation_config.json; do
    if [[ -f "$archive/$f" ]]; then
      cp -a "$archive/$f" "$dest/$f"
    fi
  done
  if [[ -f "$archive/scores.json" ]]; then
    cp -a "$archive/scores.json" "$dest/scores.json"
  elif [[ -f "$dest/data.json" ]]; then
    DATA="$dest/data.json" OUT="$dest/scores.json" "$ROOT/.venv/bin/python" -c "
import json, os, sys
from pathlib import Path
sys.path.insert(0, '$ROOT/src')
sys.path.insert(0, '$ROOT')
from ablation.stats import extract_scores
data = json.loads(Path(os.environ['DATA']).read_text())
Path(os.environ['OUT']).write_text(
    json.dumps(extract_scores(data['summary']), indent=2) + '\n'
)
"
  fi
  score=$(python3 -c "import json; print(json.load(open('$dest/scores.json'))['final_score'])")
  echo "Restored $(basename "$dest") from $(basename "$archive") (final_score=$score)"
}

restore_run "$RUNS/061__ablation_ablation_a8_top10_r1__2026-06-28_01-09-25_CDT" "$A8/top10_PRE_FIX/run_1"
restore_run "$RUNS/062__ablation_ablation_a8_top10_r2__2026-06-28_01-56-04_CDT" "$A8/top10_PRE_FIX/run_2"
restore_run "$RUNS/063__ablation_ablation_a8_top10_r3__2026-06-28_02-43-25_CDT" "$A8/top10_PRE_FIX/run_3"
restore_run "$RUNS/070__ablation_ablation_a9_512_r1__2026-06-28_06-47-06_CDT" "$A9/512_PRE_FIX/run_1"
restore_run "$RUNS/071__ablation_ablation_a9_512_r2__2026-06-28_07-25-13_CDT" "$A9/512_PRE_FIX/run_2"
restore_run "$RUNS/072__ablation_ablation_a9_512_r3__2026-06-28_08-03-17_CDT" "$A9/512_PRE_FIX/run_3"
