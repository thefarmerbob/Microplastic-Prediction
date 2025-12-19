#!/usr/bin/env bash
# Compile main.tex to PDF using latexmk (pdflatex). Run from repo root or this dir.
set -euo pipefail
cd "$(dirname "$0")"

# -pdf: generate PDF, -interaction=nonstopmode: don't stop on errors, -halt-on-error: fail fast
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

echo "PDF generated at main.pdf"
