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

**Retrieval-only:** Recall@k, MRR@k (higher is better).

**Full pipeline:** weighted `final_score` (0–100):

| Metric | Weight |
|--------|--------|
| Answer correctness | 35% |
| Faithfulness | 25% |
| Context recall | 20% |
| Context precision | 10% |
| Citation accuracy | 10% |

Judge scoring uses Ollama JSON mode. Generation uses `llama3.1:8b`; answers with `[Doc X]` citations are parsed from the `<answer>` block only.

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
