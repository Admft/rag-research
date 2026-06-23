# RAG Research

Local RAG research lab over 12 RAG papers (`data/raw/*.pdf`) and 60 hand-written eval questions (`data/eval/questions.jsonl`).

Check progress anytime:

```bash
python src/milestone_status.py
```

## Results (start here)

All run output lives in **`experiments/Results/`**:

```
experiments/Results/
  MASTER_LOG.txt              ← compare all runs
  summary.csv                 ← spreadsheet table
  000-005_FULL_REPORT.txt     ← progression report (runs 000–005)
  runs/
    000__og_baseline__.../      ← original first-dev baseline
    001__baseline__.../        ← grid smoke test (retrieval-only)
    002–005__baseline__.../   ← experiment grid baseline iterations
      REPORT.txt
      data.json
      questions.jsonl
```

See `experiments/Results/README.txt` for the full guide.

## Original baseline (run 000)

Recreates the first-development setup before the experiment grid: chunk 120 / overlap 30, dense retrieval, no reranker, simple `og_strict` prompt (`"I do not know"` fallback). Uses a separate index at `data/qdrant_og`.

```bash
python src/run_og_baseline.py                  # full pipeline (60 questions)
python src/run_og_baseline.py --retrieval-only # Recall@k / MRR@k only
```

Saved as **`000__og_baseline__...`** so it sorts before grid runs.

## Experiment grid baseline (runs 001+)

Current grid baseline (runs 004/005): chunk 256, overlap 50, dense + BGE reranker, `strict_context_with_citations` prompt with `<scratchpad>` / `<answer>` tags.

```bash
python src/run_experiments.py --list
python src/run_experiments.py --run baseline --retrieval-only
python src/run_experiments.py --run baseline   # full pipeline, needs Ollama
python src/run_experiments.py --round chunk_size
```

Requires Ollama with `llama3.1:8b` (or change `generator` in `experiments/grid.json`).

## Ad-hoc scripts

Index, search, ask, and evaluate without the full experiment harness:

```bash
python src/build_index.py
python src/search.py "What is chunking?"
python src/ask.py "What is chunking?"
python src/evaluate_retrieval.py
```

These also save runs under `experiments/Results/runs/`.

## Scoring

Implementation: `src/scoring.py` and `src/pipeline.py`. Every full-pipeline run scores each of the 60 eval questions, then averages per-metric scores into the summary block in `REPORT.txt`.

### Retrieval metrics (all runs)

Computed from `expected_source` in `data/eval/questions.jsonl` vs the top-k retrieved chunks. No LLM involved.

| Metric | Definition |
|--------|------------|
| **Recall@k** | Fraction of questions where the expected PDF appears anywhere in the top-k results (binary hit per question, then averaged). |
| **MRR@k** | Mean Reciprocal Rank: for each question, `1 / rank` if the expected source is found, else `0`; then averaged. Rank 1 → 1.0, rank 3 → 0.33. |
| **Recall hit** | Per-question yes/no shown in `REPORT.txt` (expected source in top-k). |

`k` matches the run's `top_k` setting (5 for all baselines 000–005).

### Full-pipeline metrics (runs 000, 002–005)

Each question goes through **retrieve → generate → score**. Generation uses `llama3.1:8b`. Scoring uses the **same model as an LLM judge** (Ollama, `format: json`) plus a deterministic citation check.

**Step 1 — Parse the answer**

For grid runs (003–005), only the `<answer>...</answer>` block is scored (`extract_final_answer()` in `src/prompts.py`). Scratchpad text is ignored. OG run 000 uses the full raw answer (no XML blocks).

**Step 2 — LLM judge (4 sub-scores, 0–100 each)**

The judge receives the question, expected answer, expected source filename, retrieved context, and parsed generated answer. It must return JSON only:

```json
{
  "answer_correctness": 0,
  "faithfulness": 0,
  "context_recall": 0,
  "context_precision": 0
}
```

| Judge metric | What it measures |
|--------------|------------------|
| **answer_correctness** | How well the generated answer matches the hand-written expected answer. |
| **faithfulness** | Whether the answer is supported by the retrieved context only (no hallucination beyond chunks). |
| **context_recall** | Whether the retrieved chunks contain the information needed to produce the expected answer. |
| **context_precision** | Whether the retrieved chunks are relevant and low-noise for the question. |

If the judge returns invalid JSON, the pipeline retries once, then falls back to word-overlap heuristics (`judge_fallback` in `data.json`).

**Step 3 — Citation accuracy (deterministic, 0–100)**

Not judged by the LLM. Parsed from `[Doc X]` tags in the scored answer text:

- **0** if no `[Doc X]` citations are present (this is why run 000 scores 0% — `og_strict` does not require citations).
- Up to **40** points: fraction of cited doc IDs that map to valid positions in the retrieved list.
- **+50** if any cited doc matches `expected_source`.
- **+10** bonus if all citations point only to the expected source file.

**Step 4 — Final score (weighted average)**

```
final_score = 0.35 × answer_correctness
            + 0.25 × faithfulness
            + 0.20 × context_recall
            + 0.10 × context_precision
            + 0.10 × citation_accuracy
```

Reported as 0–100, rounded to two decimal places. Per-question `final_score` values appear under each question in `REPORT.txt`; the summary line is the mean across all 60 questions.

### Auxiliary metrics

| Metric | Definition |
|--------|------------|
| **answer_parse_rate** | Fraction of questions where `<answer>` was successfully parsed (grid runs 003–005 only). |
| **avg_latency** | Mean wall-clock seconds per question (retrieve + generate + judge). |

### What is not in final_score

Retrieval Recall@k and MRR@k are reported separately. They are **not** folded into `final_score` — you can have 98.3% Recall@5 and a low final score if generation or citations fail.

## Baseline progression (runs 000–005)

| Run | What changed | Final score |
|-----|--------------|-------------|
| 000 | OG baseline (120/30, no reranker, `og_strict`) | 57.48 |
| 001 | Grid smoke test, retrieval-only, 2 questions | — |
| 002 | First full grid baseline (256, BGE reranker, citation prompt) | 66.04 |
| 003 | `<scratchpad>` / `<answer>` output isolation | 70.53 |
| 004 | Token budget + scratchpad brevity (**best**) | **74.39** |
| 005 | Identical re-run of 004 config | 74.03 |

Full analysis: `experiments/Results/000-005_FULL_REPORT.txt`

## Failure case notes

When retrieval or generation fails, log it in `notes/failure_cases.md` (query, expected vs actual, likely cause, follow-up).

## Project layout

```
experiments/
  grid.json              experiment definitions
  Results/               all run output
data/raw/                source PDFs (and .txt)
data/eval/               hand-written eval questions
data/processed/          generated chunks
data/qdrant/             main experiment vector index
data/qdrant_og/          OG baseline vector index
notes/                   failure case notes
docs/                    milestone and research notes
src/                     scripts and shared config
```

## Configuration

Shared settings in `src/config.py`:

- chunk size / overlap (grid default: 256 / 50; OG: 120 / 30)
- embedding model (`BAAI/bge-small-en-v1.5`)
- Ollama URL / model / token limits
- dataset stage and Part 21 targets

Grid baseline overrides in `experiments/grid.json`.
