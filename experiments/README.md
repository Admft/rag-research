# Small experiments

Log focused changes here — one experiment per file or section.

## Template

```markdown
# Experiment: chunk 60 vs 120

## Hypothesis
Larger chunks will improve Recall@1 on factual questions spanning multiple sentences.

## Settings
| Setting | Baseline | Experiment |
|---------|----------|------------|
| chunk_size_words | 60 | 120 |
| overlap_words | 10 | 30 |

## Commands
python src/build_index.py
python src/evaluate_retrieval.py

## Results
- Baseline: results/baseline/evaluation/eval__chunk60-overlap10__...
- Experiment: results/baseline/evaluation/eval__chunk120-overlap30__...

## Takeaway
(one sentence)
```

---

_Add experiments below as you run them._
