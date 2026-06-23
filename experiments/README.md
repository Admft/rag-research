# Experiment grid

One-variable-at-a-time RAG experiments.

## Quick start

```bash
python src/run_experiments.py --list
python src/run_experiments.py --run baseline --retrieval-only
python src/run_experiments.py --round chunk_size --retrieval-only
python src/run_experiments.py --run baseline   # full pipeline, needs Ollama
```

## Where results go

**Everything is in `experiments/Results/`:**

| File / folder | What it is |
|---------------|------------|
| `MASTER_LOG.txt` | Compare all runs — **start here** |
| `summary.csv` | One row per run for spreadsheets |
| `runs/001__baseline__.../` | One folder per run |
| `runs/.../REPORT.txt` | Human-readable report for that run |
| `runs/.../data.json` | Full raw data |
| `runs/.../questions.jsonl` | Per-question details |

See `experiments/Results/README.txt` for the full guide.

## Scoring

Full runs use weighted `final_score` (0–100). Retrieval-only runs use Recall@k and MRR@k.

## Config

Edit `experiments/grid.json` for run definitions.

## Generation quality tips

If retrieval is strong but final_score is low:

- Baseline now uses **256-word chunks**, **bge reranker**, and **strict_context_with_citations** prompt with `[Doc X]` citations and chain-of-thought planning.
- Documents are injected as `<Document ID="1">...</Document>` blocks.
- Try a larger Ollama model in `experiments/grid.json` → `generator` (e.g. `qwen2.5:14b`).

```bash
python src/run_experiments.py --run baseline
```
