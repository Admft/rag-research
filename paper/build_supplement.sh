#!/usr/bin/env bash
# Build anonymized OpenReview supplementary zip (max 200MB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$(cd "$(dirname "$0")" && pwd)"
OUT="$PAPER/groundlm2026_supplement_anonymous.zip"
STAGE="$(mktemp -d)"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$STAGE/supplement"

# 66-run subset (paper-reported ablations only)
python3 "$PAPER/build_paper_reported_runs.py"
cp "$PAPER/paper_reported_runs.csv" "$STAGE/supplement/paper_reported_runs.csv"

# Full ablation log in summary.csv (76 rows = 66 paper + 10 audit/pre-fix/superseded)
cp "$ROOT/data/eval/questions.jsonl" "$STAGE/supplement/questions.jsonl"
cp "$ROOT/experiments/grid.json" "$STAGE/supplement/grid.json"
cp "$ROOT/experiments/Results/summary.csv" "$STAGE/supplement/summary.csv"
cp "$ROOT/src/scoring.py" "$STAGE/supplement/scoring.py"
cp "$PAPER/supplementary_ablations.tex" "$STAGE/supplement/full_ablation_table.tex"
cp "$PAPER/requirements-audit.txt" "$STAGE/supplement/requirements-audit.txt"

# Ablation summaries (preserve unique paths)
mkdir -p "$STAGE/supplement/ablation_summaries"
while IFS= read -r -d '' f; do
  rel="${f#"$ROOT/experiments/Results/Test Runs and Ablations/"}"
  dest_dir="$STAGE/supplement/ablation_summaries/$(dirname "$rel")"
  mkdir -p "$dest_dir"
  cp "$f" "$dest_dir/"
done < <(find "$ROOT/experiments/Results/Test Runs and Ablations" -name 'summary.json' -print0)

# Pre-fix judge traces: scores + reports only (skip data.json — large PDF text)
mkdir -p "$STAGE/supplement/pre_fix_archives"
for dir in \
  "Ablation 8 - Top-K/top10_PRE_FIX" \
  "Ablation 9 - Chunk Size/512_PRE_FIX"; do
  src="$ROOT/experiments/Results/Test Runs and Ablations/$dir"
  if [[ -d "$src" ]]; then
    dest_name="$(basename "$(dirname "$src")")_$(basename "$src")"
    dest_root="$STAGE/supplement/pre_fix_archives/$dest_name"
    mkdir -p "$dest_root"
    for run_dir in "$src"/run_*; do
      [[ -d "$run_dir" ]] || continue
      run_name="$(basename "$run_dir")"
      mkdir -p "$dest_root/$run_name"
      cp "$run_dir/scores.json" "$run_dir/REPORT.txt" "$dest_root/$run_name/" 2>/dev/null || true
    done
  fi
done

cat > "$STAGE/supplement/README.txt" <<'EOF'
GroundLM 2026 — Anonymous Supplementary Material
================================================

PURPOSE
  Auditability for the short paper "When RAG Evaluation Fails Silently."
  This archive supports verifying reported numbers and failure modes.
  It is NOT a full end-to-end rerun package: Ollama model weights, Qdrant
  indexes, and the 12-paper PDF corpus are not bundled.

FILE GUIDE
  paper_reported_runs.csv   Exactly 66 runs cited in the paper (paper_reported=true)
  summary.csv               76 ablation rows total: 66 paper-reported + 10 audit rows
  full_ablation_table.tex   Aggregated means/SDs for all 21 conditions
  ablation_summaries/       Per-ablation summary.json files
  pre_fix_archives/         Pre-fix A8 top10 and A9 chunk512 judge-overflow traces
  scoring.py                Judge cap: JUDGE_MAX_CONTEXT_CHUNKS, JUDGE_MAX_CONTEXT_WORDS
  questions.jsonl           60 eval questions
  grid.json                 Locked baseline / ablation grid config
  requirements-audit.txt    Minimal Python deps for inspecting artifacts

66 vs 76 RUN COUNT
  The paper reports 66 ablation runs (9 ablations × up to 3 conditions × 3 reps,
  minus superseded audit rows). summary.csv has 76 rows because it also includes:
    - 058-060: A8 top-k=7 runs before judge-cap validation (superseded by 073-075)
    - 061-063: A8 top-k=10 pre-fix judge overflow (~10/100 artifact)
    - 070-072: A9 chunk-512 pre-fix judge overflow (~14/100 artifact)
    - 076:     A8 top-k=10 intermediate rescore (superseded by 077-079)
  Use paper_reported_runs.csv for the exact paper subset; filter summary.csv with
  column paper_reported in that file, or exclude run_folder prefixes listed above.

HOW TO VERIFY PAPER CLAIMS
  1. Table 1 / Figure 1 (judge overflow):
     - Pre-fix means ~9.9 (top-k=10): pre_fix_archives/Ablation 8 - Top-K_top10_PRE_FIX/
       and summary.csv rows 061-063 (final_score column)
     - Post-fix means 64.1: paper_reported_runs.csv rows with top_k=10
     - Pre-fix A9 chunk512 ~14.1: pre_fix_archives/Ablation 9 - Chunk Size_512_PRE_FIX/
       and summary.csv rows 070-072
     - Post-fix A9 chunk512 65.9: paper_reported_runs.csv rows with chunk_size=512

  2. Table 2 / Figure 2 (ablation deltas):
     - See full_ablation_table.tex and ablation_summaries/*/summary.json
     - Cross-check per-run final_score in paper_reported_runs.csv

  3. Context-filter regression (A4, -14.3 pts):
     - paper_reported_runs.csv: context_filter=top_sentences_5 (3 rows)
     - ablation_summaries/Ablation 4 - Context Filter/summary.json

  4. Judge cap implementation:
     - scoring.py constants JUDGE_MAX_CONTEXT_CHUNKS=5, JUDGE_MAX_CONTEXT_WORDS=1920

  5. adaptllm_q5 case study:
     - questions.jsonl id adaptllm_q5; inspect full-pipeline REPORT traces in repo
       (not all per-question logs are duplicated here to limit supplement size)

REQUIREMENTS
  See requirements-audit.txt. Full pipeline rerun additionally needs Ollama,
  Qdrant, GPU, and the indexed PDF corpus.

ANONYMITY
  No author names or machine paths are intentionally included. Retrieved PDF text
  in optional full run logs may contain third-party author names/emails from papers.
EOF

# Anonymity check (metadata files only; skip data.json corpus text)
echo "==> Anonymity grep (metadata files)"
GREP_PATTERN='Adam Moffat|Moffat|Admft|vusion|/Users/|/home/|C:\\\\|github\.com/'
if grep -RInE "$GREP_PATTERN" \
    "$STAGE/supplement/README.txt" \
    "$STAGE/supplement/scoring.py" \
    "$STAGE/supplement/grid.json" \
    "$STAGE/supplement/summary.csv" \
    "$STAGE/supplement/paper_reported_runs.csv" \
    "$STAGE/supplement/questions.jsonl" \
    "$STAGE/supplement/pre_fix_archives" \
    2>/dev/null; then
  echo "WARNING: possible deanonymizing strings found above — fix before upload."
  exit 1
fi
echo "    No submitter-identifying strings in metadata files."

if find "$STAGE/supplement" \( -name '.git' -o -name '__pycache__' -o -name '.DS_Store' \) | grep -q .; then
  echo "WARNING: unwanted files found"
  find "$STAGE/supplement" \( -name '.git' -o -name '__pycache__' -o -name '.DS_Store' \)
  exit 1
fi

rm -f "$OUT"
(cd "$STAGE" && zip -r "$OUT" supplement -x "*.DS_Store")
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
