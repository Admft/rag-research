# Part 21: The Exact Next Milestone

## Goal

Build a local RAG baseline over **20-50 public documents** and create **50 hand-written evaluation questions**.

This milestone turns the toy pipeline into a real research baseline you can iterate on.

## Current state

| Item | Toy baseline | Part 21 target |
|------|--------------|----------------|
| Documents | 1 (`rag_intro.txt`) | 20-50 public `.txt` files |
| Eval questions | 5 | 50 hand-written |
| Dataset stage | `toy` | `real` |

Run `python src/milestone_status.py` to see live progress.

## Step-by-step

### 1. Confirm the toy baseline works

```bash
python src/build_index.py
python src/evaluate_retrieval.py
python src/ask.py "What is chunking?"
```

You should get result files under:

- `results/baseline/retrieval/`
- `results/baseline/evaluation/`
- `results/baseline/generation/`

### 2. Replace the toy corpus

Remove or archive `data/raw/rag_intro.txt` and add 20-50 real documents as plain `.txt` files in `data/raw/`.

**Good beginner datasets:**

- **Public Cornell course catalog pages** copied into separate `.txt` files (one course or page per file).
- **A small collection of public AI papers** converted to text (one paper per file).

Tips:

- Keep filenames meaningful (`cs4780_syllabus.txt`, `attention_is_all_you_need.txt`).
- One logical document per file.
- Strip navigation boilerplate when copying web pages.

See `data/README.md` for more detail.

### 3. Write 50 eval questions by hand

Edit `data/eval/questions.jsonl`. One JSON object per line:

```json
{
  "id": "q6",
  "question": "Your question here?",
  "answer": "The correct answer based on the corpus.",
  "expected_source": "filename.txt"
}
```

Guidelines:

- Questions should be answerable from a single document or clearly retrievable chunk.
- Include the expected source filename for retrieval eval.
- Mix easy, medium, and hard questions.
- Cover different documents, not just one file.
- Avoid questions answerable from general knowledge alone.

### 4. Update config and rebuild

In `src/config.py`:

```python
DATASET_STAGE = "real"
```

Then:

```bash
python src/build_index.py
python src/evaluate_retrieval.py
```

Review the readable reports in `results/baseline/evaluation/`.

### 5. Record failures and experiments

- Failures → `notes/failure_cases.md`
- Parameter sweeps → `experiments/`

## What “done” looks like

- [ ] 20-50 documents in `data/raw/`
- [ ] 50 questions in `data/eval/questions.jsonl`
- [ ] `DATASET_STAGE = "real"` in config
- [ ] Baseline retrieval, evaluation, and generation all run cleanly
- [ ] At least a few failure cases documented
- [ ] At least one small experiment logged (e.g. chunk size 60 vs 120)

## After Part 21

For publishable retrieval research, move to a standard benchmark like **BEIR** (Benchmarking IR), which provides heterogeneous retrieval datasets and is widely used in the literature.

The local baseline you build here is practice. BEIR is for comparable, publishable evaluation.
