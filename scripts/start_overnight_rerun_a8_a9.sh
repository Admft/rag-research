#!/usr/bin/env bash
# Re-run A8 top10 and A9 chunk512 with fixed judge (after optional rescore completes).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SESSION="rag-rerun-a8-a9"
LOG="$ROOT/experiments/Results/rerun_a8_a9_overnight.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv: $PYTHON"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install: sudo apt install tmux"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Rerun supervisor already running in tmux session: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec bash scripts/rerun_a8_a9_worker.sh 2>&1 | tee -a '$LOG'"

echo "Overnight A8 top10 + A9 512 rerun started."
echo "  tmux session: $SESSION"
echo "  attach:       tmux attach -t $SESSION"
echo "  log:          tail -f '$LOG'"
