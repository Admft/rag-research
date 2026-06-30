# GroundLM 2026 Short Paper

**Venue:** [GroundLM @ EMNLP 2026](https://groundlm.github.io/grouplm_emnlp2026/) — *Grounding Language Models: Learning Faithfully and Efficiently*

**Track:** Direct Submission → Short paper (archival), anonymous review

## Files

| File | Purpose |
|------|---------|
| `groundlm2026.tex` | Main paper (≤4 pages content) |
| `custom.bib` | Bibliography |
| `acl.sty` | ACL 2026 style (review mode) |
| `acl_natbib.bst` | Bibliography style |

## Build

Requires a TeX distribution with `pdflatex` and `bibtex`:

```bash
cd paper
pdflatex groundlm2026
bibtex groundlm2026
pdflatex groundlm2026
pdflatex groundlm2026
```

Output: `groundlm2026.pdf`

## Submission checklist

- [x] ACL 2026 template (`\usepackage[review]{acl}`)
- [x] Anonymous (no author names or acknowledgments)
- [x] Limitations section before references
- [ ] Verify main body ≤4 pages (references unlimited after)
- [ ] Submit PDF to GroundLM OpenReview before deadline

## Framing

This is the **narrow** workshop version focused on three evaluation/grounding findings:

1. LLM-judge silent failure above ~2,500 tokens (A8/A9 fix story)
2. Post-reranking context filter regression (−14.3 pts, A4)
3. Document-level Recall@5 masking passage failure (`adaptllm_q5`)

Expected ablation effects (generator size, reranker, chunk size) are background only. Full 66-run grid lives in `../MASTER_EXPERIMENT_LOG.txt` and `../experiments/Results/`.

## Camera-ready

After acceptance, change `\usepackage[review]{acl}` to `\usepackage{acl}` and add author block.
