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

**Overleaf:** Upload the 6 files listed below into one blank Overleaf project (drag into the file tree). Set `groundlm2026.tex` as the main document (Menu → Main document). Click **Recompile**. Overleaf runs BibTeX automatically.

| Upload to Overleaf | Required? |
|--------------------|-------------|
| `groundlm2026.tex` | Yes (main document) |
| `custom.bib` | Yes |
| `acl.sty` | Yes |
| `acl_natbib.bst` | Yes |
| `fig1_judge_overflow.pdf` | Yes |
| `fig2_ablation_deltas.pdf` | Yes (comment out Fig.~2 in `.tex` if over 4 pages) |

Do **not** upload: `README.md`, `build.sh`, `*.py`, `supplementary_ablations.tex` (optional extra only).

**Steps:**
1. Go to [overleaf.com](https://www.overleaf.com) → New Project → Blank Project
2. Delete the default `main.tex` (or replace it)
3. Upload all 6 files above into the project root (same folder)
4. Menu (top left) → **Main document** → select `groundlm2026.tex`
5. Click **Recompile** (green button)
6. Download PDF when citations and References look correct

**VS Code LaTeX Workshop:** Use a recipe that includes `bibtex`, or set `"latex-workshop.latex.recipes"` to include bibtex between pdflatex runs.

## Submission checklist

- [x] ACL 2026 template (`\usepackage[review]{acl}`)
- [x] Anonymous (no author names or acknowledgments)
- [x] Limitations section before references
- [ ] Verify main body ≤4 pages (references unlimited after)
- [ ] Build supplement: `./build_supplement.sh` → upload zip to OpenReview
- [ ] Submit PDF to GroundLM OpenReview before deadline

## OpenReview fields (copy-paste)

| Field | Value |
|-------|-------|
| **Submission Track** | Track 1: Direct Submission |
| **Submission Type** | Archival Short Paper (4 pages limitation for the main body) |
| **TL;DR** | We show that faithful RAG evaluation can fail silently through LLM-judge context overflow, harmful post-reranking context filtering, and document-level recall metrics that miss answer-bearing passage failures. |
| **Keywords** | retrieval-augmented generation, RAG evaluation, grounded generation, faithfulness, LLM-as-judge, hallucination, context filtering, passage-level recall |
| **Voluntary Reviewer** | Nominate at least one author (OpenReview profile ID) |
| **License** | Select CC BY 4.0 (or option offered by OpenReview) |
| **Supplementary Material** | `groundlm2026_supplement_anonymous.zip` from `./build_supplement.sh` |

**Deadline:** GroundLM site lists June 29, 2026 (AoE); OpenReview may show a later date—submit as soon as possible and treat the portal deadline as authoritative.

## Framing

This is the **narrow** workshop version focused on three evaluation/grounding findings:

1. LLM-judge silent failure above ~2,500 tokens (A8/A9 fix story)
2. Post-reranking context filter regression (−14.3 pts, A4)
3. Document-level Recall@5 masking passage failure (`adaptllm_q5`)

Expected ablation effects (generator size, reranker, chunk size) are background only. Full 66-run grid lives in `../MASTER_EXPERIMENT_LOG.txt` and `../experiments/Results/`.

## Camera-ready

After acceptance, change `\usepackage[review]{acl}` to `\usepackage{acl}` and add author block.
