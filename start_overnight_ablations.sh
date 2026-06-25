#!/usr/bin/env bash
# Start the overnight ablation supervisor in a detached tmux session.
# Survives terminal close; auto-restarts on segfault; resumes checkpoints.
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
  echo "tmux is required for overnight runs."
  echo "Install: sudo apt install tmux"
  echo ""
  echo "Or run directly (stops if you close the terminal):"
  echo "  $PYTHON $SUPERVISOR --all --repair-venv --until-complete"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Supervisor already running in tmux session: $SESSION"
  echo "  attach:  tmux attach -t $SESSION"
  echo "  status:  $PYTHON $SUPERVISOR --status"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec $PYTHON '$SUPERVISOR' --all --repair-venv --until-complete"

echo "Overnight ablation supervisor started."
echo "  tmux session: $SESSION"
echo "  attach:       tmux attach -t $SESSION"
echo "  detach:       Ctrl+B then D"
echo "  status:       $PYTHON $SUPERVISOR --status"
echo "  log:          tail -f '$LOG'"
echo ""
echo "The supervisor will:"
echo "  - restart on segfaults (fresh Python each time)"
echo "  - resume from checkpoint.jsonl after crashes"
echo "  - kill and restart if stalled with no progress for 45 min"
echo "  - keep going until all 66 expected runs have scores.json"
