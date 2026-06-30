#!/usr/bin/env bash
# Build anonymized OpenReview supplementary zip (max 200MB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/paper/groundlm2026_supplement_anonymous.zip"
STAGE="$(mktemp -d)"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$STAGE/supplement"

# Core artifacts (no author paths in filenames)
cp "$ROOT/data/eval/questions.jsonl" "$STAGE/supplement/questions.jsonl"
cp "$ROOT/experiments/grid.json" "$STAGE/supplement/grid.json"
cp "$ROOT/experiments/Results/summary.csv" "$STAGE/supplement/summary.csv"
cp "$ROOT/src/scoring.py" "$STAGE/supplement/scoring.py"
cp "$ROOT/paper/supplementary_ablations.tex" "$STAGE/supplement/full_ablation_table.tex"

# Ablation summaries (preserve unique paths)
mkdir -p "$STAGE/supplement/ablation_summaries"
while IFS= read -r -d '' f; do
  rel="${f#"$ROOT/experiments/Results/Test Runs and Ablations/"}"
  dest_dir="$STAGE/supplement/ablation_summaries/$(dirname "$rel")"
  mkdir -p "$dest_dir"
  cp "$f" "$dest_dir/"
done < <(find "$ROOT/experiments/Results/Test Runs and Ablations" -name 'summary.json' -print0)

# Pre-fix judge traces (A8/A9)
mkdir -p "$STAGE/supplement/pre_fix_archives"
for dir in \
  "Ablation 8 - Top-K/top10_PRE_FIX" \
  "Ablation 9 - Chunk Size/512_PRE_FIX"; do
  src="$ROOT/experiments/Results/Test Runs and Ablations/$dir"
  if [[ -d "$src" ]]; then
    dest_name="$(basename "$(dirname "$src")")_$(basename "$src")"
    cp -R "$src" "$STAGE/supplement/pre_fix_archives/$dest_name"
  fi
done

cat > "$STAGE/supplement/README.txt" <<'EOF'
Anonymous supplementary material for GroundLM 2026 submission.
Contents: eval questions, grid config, scoring.py (judge cap),
summary.csv (82 runs), ablation summary.json files, pre-fix A8/A9 archives.
No author names or machine paths included.
EOF

rm -f "$OUT"
(cd "$STAGE" && zip -r "$OUT" supplement -x "*.DS_Store")
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
