#!/usr/bin/env bash
# Full ACL build: pdflatex -> bibtex -> pdflatex -> pdflatex
# Citations show as (??) if you only run pdflatex once.
set -euo pipefail
cd "$(dirname "$0")"

for cmd in pdflatex bibtex; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd not found. Install MacTeX or use Overleaf."
    exit 1
  fi
done

echo "==> pdflatex (pass 1)"
pdflatex -interaction=nonstopmode groundlm2026.tex >/dev/null

echo "==> bibtex"
bibtex groundlm2026

echo "==> pdflatex (pass 2)"
pdflatex -interaction=nonstopmode groundlm2026.tex >/dev/null

echo "==> pdflatex (pass 3)"
pdflatex -interaction=nonstopmode groundlm2026.tex >/dev/null

echo "Done: paper/groundlm2026.pdf"
