#!/usr/bin/env bash
# Most conservative overnight run: tmux + supervisor safe preset.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SESSION="rag-ablations"
PYTHON="$ROOT/.venv/bin/python"
SUPERVISOR="$ROOT/run_ablation_resilient.py"
LOG="$ROOT/experiments/Results/ablation_watchdog.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv: $PYTHON"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install: sudo apt install tmux"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Supervisor already running in tmux session: $SESSION"
  echo "  attach:  tmux attach -t $SESSION"
  echo "  status:  $PYTHON $SUPERVISOR --status --all"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec $PYTHON '$SUPERVISOR' --safe --all"

echo "Safe overnight supervisor started."
echo "  tmux session: $SESSION"
echo "  attach:       tmux attach -t $SESSION"
echo "  status:       $PYTHON $SUPERVISOR --status --all"
echo "  log:          tail -f '$LOG'"
echo ""
echo "Safe mode:"
echo "  - tmux (survives terminal close)"
echo "  - auto-restart until 66/66 runs complete"
echo "  - resume from checkpoint.jsonl after every crash"
echo "  - CPU embeddings (avoids GPU fights with Ollama)"
echo "  - venv repair after crashes"
echo "  - 120s retry delay + 60s extra after segfaults"
echo "  - 90 min stall timeout (won't kill slow but healthy runs)"
echo "  - OLLAMA_NUM_PARALLEL=1"
