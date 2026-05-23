#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[build] running pdflatex..."
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

echo "[build] running pandoc to produce docx..."
pandoc report.tex -o report.docx --resource-path=.:figures

echo "[build] done. Outputs: report.pdf, report.docx"
