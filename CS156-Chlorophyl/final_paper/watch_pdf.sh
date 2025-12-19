#!/usr/bin/env bash
# Auto-compile main.tex to PDF whenever you save (latexmk -pvc).
# Run this once and leave it open; it rebuilds on every file change.
set -euo pipefail
cd "$(dirname "$0")"

latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex
