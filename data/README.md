# Dataset plan

## Toy stage (now)

- `rag_intro.txt` — single intro document for learning the pipeline
- 5 eval questions in `data/eval/questions.jsonl`
- `DATASET_STAGE = "toy"` in `src/config.py`

## Real stage (Part 21)

Replace the toy file with **20-50 public `.txt` documents** in this folder.

### Option A: Cornell course catalog pages

Good for beginners because pages are structured and factual.

1. Pick public Cornell course catalog or course description pages.
2. Copy the main content into plain text.
3. Save one course/page per file, e.g. `cs4780_overview.txt`.
4. Remove headers, footers, and navigation chrome.

### Option B: Small AI paper collection

Good if you care about technical retrieval.

1. Pick 20-50 public AI/ML papers with permissive access.
2. Convert PDFs to text (one file per paper).
3. Name files after the paper, e.g. `attention_is_all_you_need.txt`.

### Option C: BEIR (later, not Part 21)

[BEIR](https://github.com/beir-cellar/beir) is the standard choice for publishable retrieval research. It is heterogeneous and widely cited.

Use BEIR **after** your local baseline works. Part 21 is about building intuition with a corpus you control.

## Eval questions

Target: **50 hand-written questions** in `data/eval/questions.jsonl`.

Each line:

```json
{"id": "q1", "question": "...", "answer": "...", "expected_source": "filename.txt"}
```

Write questions **after** you have the real corpus so they match your documents.

## Rebuild after any data change

```bash
python src/build_index.py
python src/evaluate_retrieval.py
```
