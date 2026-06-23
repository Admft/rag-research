# Experiment grid

One-variable-at-a-time RAG experiments over 60 eval questions and 12 PDF papers.

## Quick start

```bash
# Original first-dev baseline (run 000)
python src/run_og_baseline.py

# Grid baseline (runs 001+)
python src/run_experiments.py --list
python src/run_experiments.py --run baseline --retrieval-only
python src/run_experiments.py --run baseline   # full pipeline, needs Ollama
python src/run_experiments.py --round chunk_size
```

## Run numbering

| Prefix | Meaning |
|--------|---------|
| `000__og_baseline__...` | Original baseline (`run_og_baseline.py`) — chunk 120/30, no reranker, `og_strict` prompt |
| `001+__baseline__...` | Experiment grid (`run_experiments.py`) — current baseline: chunk 256/50, BGE reranker, scratchpad/answer prompt |

## Where results go

**Everything is in `experiments/Results/`:**

| File / folder | What it is |
|---------------|------------|
| `MASTER_LOG.txt` | Compare all runs — **start here** |
| `summary.csv` | One row per run for spreadsheets |
| `000-005_FULL_REPORT.txt` | Research report on baseline progression |
| `runs/000__og_baseline__.../` | OG baseline run |
| `runs/001__baseline__.../` | Grid runs |
| `runs/.../REPORT.txt` | Human-readable report for that run |
| `runs/.../data.json` | Full raw data |
| `runs/.../questions.jsonl` | Per-question details |

See `experiments/Results/README.txt` for the full guide.

## Scoring

See the main **`README.md` → Scoring** section for the full methodology (`src/scoring.py`).

**Retrieval-only:** Recall@k, MRR@k — binary hit on `expected_source` in top-k, no LLM.

**Full pipeline:** LLM judge (4 metrics, 0–100) + deterministic citation check → weighted `final_score`:

| Component | Weight |
|-----------|--------|
| Answer correctness | 35% |
| Faithfulness | 25% |
| Context recall | 20% |
| Context precision | 10% |
| Citation accuracy | 10% |

Recall@k / MRR@k are reported separately and are **not** part of `final_score`.

## Current grid baseline (v2)

| Setting | Value |
|---------|-------|
| Chunk size / overlap | 256 / 50 |
| Retriever | **hybrid** (dense + BM25) |
| Reranker | bge |
| Top-k | 5 |
| Prompt | `strict_context_json` (Ollama JSON schema) |
| Context filter | none (006 used top_sentences_5 — disabled for 007) |
| Generator | `qwen2.5:14b` |
| Judge | `qwen2.5:14b` (separate from generator) |

Retrieval saturated at 98.3% Recall@5 by run 002; later gains came from prompt/output structure (see `000-005_FULL_REPORT.txt`).

## Config

Edit `experiments/grid.json` for run definitions. Shared paths and OG constants in `src/config.py`.

## Generation quality tips

If retrieval is strong but `final_score` is low:

- Ensure answers use `<answer>` blocks with `[Doc X]` citations (not leaked scratchpad text).
- Check `answer_parse_rate` in `REPORT.txt` — parse failures usually mean the model hit the generation token limit in the scratchpad.
- Try a larger Ollama model in `experiments/grid.json` → `generator` (e.g. `qwen2.5:14b`).

```bash
python src/run_experiments.py --run baseline
```
