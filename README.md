# RAG Research

Local RAG baseline project. Start with the toy corpus, then scale to a real dataset and a hand-written eval set.

Check progress anytime:

```bash
python src/milestone_status.py
```

## Baseline retrieval

Index raw documents into a local Qdrant store with dense embeddings.

```bash
python src/build_index.py
python src/search.py "What is chunking?"
```

Results: `results/baseline/retrieval/`

## Baseline generation

Retrieve context, then generate an answer with Ollama.

```bash
python src/ask.py "What is chunking?"
```

Results: `results/baseline/generation/`

Requires Ollama running locally with `llama3.1:8b` (or change `OLLAMA_MODEL` in `src/config.py`).

## Baseline evaluation

Measure retrieval quality with Recall@k and MRR@k on hand-written questions.

```bash
python src/evaluate_retrieval.py
```

Results: `results/baseline/evaluation/`

Eval questions live in `data/eval/questions.jsonl`. The toy set has 5 questions; the Part 21 target is 50.

## Experiment grid

Run baseline + one-variable-at-a-time tests (chunk size, overlap, top-k, retriever, reranker).

```bash
python src/run_experiments.py --list
python src/run_experiments.py --run baseline --retrieval-only
python src/run_experiments.py --round chunk_size
python src/run_experiments.py --all
```

Results CSV: `results/experiments/experiment_results.csv`

See `experiments/README.md` for the full 25-run plan and scoring formula.

## Small experiments

Use `experiments/` to log parameter sweeps and comparisons (chunk size, overlap, embedding model, top-k, etc.).

Each experiment should note:

- hypothesis
- settings changed
- result file paths
- takeaway

See `experiments/README.md`.

## Failure case notes

When retrieval or generation fails, write it down in `notes/failure_cases.md`:

- query
- expected behavior
- actual behavior
- likely cause
- follow-up experiment

Failure notes are part of the research record, not bugs to ignore.

## Part 21: The exact next milestone

**Goal:** Build a local RAG baseline over 20-50 public documents and create 50 hand-written evaluation questions.

Current status: toy example (`data/raw/rag_intro.txt`, 5 eval questions).

After the toy example works:

1. Replace `data/raw/rag_intro.txt` with a real dataset (20-50 `.txt` files).
2. Expand `data/eval/questions.jsonl` to 50 hand-written questions with expected sources and answers.
3. Rebuild the index and rerun baseline evaluation.
4. Record failures and small experiments as you go.

Full checklist and dataset options: `docs/part21-next-milestone.md`

For publishable research later, use a standard benchmark like [BEIR](https://github.com/beir-cellar/beir) — a heterogeneous benchmark commonly used for retrieval evaluation.

## Project layout

```
data/raw/              source documents (.txt)
data/eval/             hand-written eval questions
data/processed/        generated chunks
results/baseline/      retrieval, evaluation, generation runs
results/experiments/   experiment grid CSV + per-run reports
experiments/           grid config and experiment notes
notes/                 failure case notes
docs/                  milestone and research notes
src/                   scripts and shared config
```

## Configuration

Shared settings are in `src/config.py`:

- chunk size / overlap
- embedding model
- eval top-k values
- dataset stage (`toy` or `real`)
- Part 21 targets (20-50 docs, 50 questions)
