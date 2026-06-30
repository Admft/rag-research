# GroundLM 2026 Short Paper

**Venue:** [GroundLM @ EMNLP 2026](https://groundlm.github.io/grouplm_emnlp2026/) — *Grounding Language Models: Learning Faithfully and Efficiently*

**Track:** Direct Submission → Short paper (archival), anonymous review

## Files

| File | Purpose |
|------|---------|
| `groundlm2026.tex` | Main paper (≤4 pages content) |
| `custom.bib` | Bibliography |
| `fig1_judge_overflow.py` | Generates Figure 1 (judge overflow cliff) |
| `fig2_ablation_deltas.py` | Generates Figure 2 (ablation deltas bar chart) |
| `fig1_judge_overflow.pdf` | Figure 1 output (upload to Overleaf with .tex) |
| `fig2_ablation_deltas.pdf` | Figure 2 output (upload to Overleaf with .tex) |
| `build.sh` | Full pdflatex + bibtex build script |

## Build

**Citations show as `(??)` and References are empty if you only run `pdflatex` once.** You must run BibTeX in between.

Requires a TeX distribution with `pdflatex` and `bibtex` (install [MacTeX](https://www.tug.org/mactex/) on Mac).

```bash
cd paper
./build.sh
```

Or manually (must run from the `paper/` folder so `custom.bib` is found):

```bash
cd paper
pdflatex groundlm2026
bibtex groundlm2026      # <-- this step creates the References section
pdflatex groundlm2026
pdflatex groundlm2026
```

Output: `groundlm2026.pdf`

## Figures

Generate both PDFs before compiling (requires `matplotlib`):

```bash
cd paper
pip3 install matplotlib   # or use your project venv
python3 fig1_judge_overflow.py
python3 fig2_ablation_deltas.py
```

Upload the two `.pdf` files to Overleaf alongside the `.tex` file.

**If you're tight on space after compile:** keep Figure 1 (judge cliff); comment out the `fig2_ablation_deltas` figure block in `groundlm2026.tex` first.

**Overleaf:** Upload `groundlm2026.tex`, `custom.bib`, `acl.sty`, and `acl_natbib.bst` into one project; Overleaf runs BibTeX automatically.

**VS Code LaTeX Workshop:** Use a recipe that includes `bibtex`, or set `"latex-workshop.latex.recipes"` to include bibtex between pdflatex runs.

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
