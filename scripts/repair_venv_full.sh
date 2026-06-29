#!/usr/bin/env bash
# Full venv repair after WSL2 segfault corrupts site-packages (~cipy, ~umpy junk).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
SITE="$VENV/lib/python3.12/site-packages"

echo "=== Stopping ablation worker processes (if any) ==="
pkill -f "run_ablation.py.*top10" 2>/dev/null || true
sleep 2

echo "=== Removing corrupted partial installs (~cipy, ~umpy, etc.) ==="
find "$SITE" -maxdepth 1 \( -name '~*' -o -name '_~*' \) -exec rm -rf {} + 2>/dev/null || true

echo "=== Clearing stale bytecode in crash-sensitive packages ==="
find "$SITE"/scipy "$SITE"/sklearn "$SITE"/numpy -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "=== Reinstalling crash-sensitive packages ==="
"$PY" -m pip install --force-reinstall --no-cache-dir \
  "sympy==1.13.1" \
  numpy scipy scikit-learn psutil pydantic qdrant-client

echo "=== Verifying imports ==="
"$PY" -c "
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import sympy, torch, pydantic
print('All imports OK')
"

echo ""
echo "Venv repaired. Resume top10 (checkpoint resumes automatically):"
echo "  cd $ROOT"
echo "  .venv/bin/python run_ablation_resilient.py --safe --repair-venv --until-complete \\"
echo "    --ablation A8 --condition top10 --runs 3"
