# Experiment grid

One-variable-at-a-time RAG experiments. Every run uses the same eval questions and outputs comparable scores.

## Quick start

```bash
# List all 25 configured runs
python src/run_experiments.py --list

# Fast test: retrieval metrics only (no Ollama generation)
python src/run_experiments.py --run baseline --retrieval-only --max-questions 2

# Run one round
python src/run_experiments.py --round chunk_size --retrieval-only

# Full pipeline on baseline (needs Ollama)
python src/run_experiments.py --run baseline

# Full recommended grid (slow — 25 runs × all questions × LLM calls)
python src/run_experiments.py --all
```

## Output

| Output | Location |
|--------|----------|
| Comparison CSV | `results/experiments/experiment_results.csv` |
| **Human-readable log** | **`experiments/Results/EXPERIMENT_LOG.txt`** |
| Per-run JSON + report | `results/experiments/{run_name}/` |

CSV columns include `final_score`, sub-metrics, recall/MRR, latency, and all config fields.

## Scoring

Full runs use this weighted final score (0–100):

```text
final_score =
  35% answer_correctness
+ 25% faithfulness
+ 20% context_recall
+ 10% context_precision
+ 10% citation_accuracy
```

Sub-scores are judged by Ollama. Latency and token estimates are tracked separately.

## Recommended order

1. `baseline`
2. `chunk_size` round → pick best chunk size
3. `overlap` round → pick best overlap
4. `top_k` round → pick best top-k
5. `retriever` round → pick best retriever
6. `reranker` round → pick best reranker
7. Update `best_full_pipeline` in `experiments/grid.json`, then run it

```bash
python src/run_experiments.py --round baseline
python src/run_experiments.py --round chunk_size
# ... etc
```

## Config

All runs inherit from `experiments/grid.json` → `baseline`, then apply `overrides` for one variable.

Edit `best_full_pipeline` overrides after earlier rounds complete.

Chunk sizes are in **words** (128, 256, 512, 1024, 2048).

## Eval questions

Update `data/eval/questions.jsonl` so `expected_source` matches your PDF filenames:

```json
{"id": "q1", "question": "...", "answer": "...", "expected_source": "2305.14283v3.pdf"}
```

Optional fields for later robustness testing:

```json
"question_type": "noise_robustness"
```

Target: 60 questions (12 PDFs × 5) before running the full grid.
