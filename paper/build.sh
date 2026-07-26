#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex >/dev/null
if command -v bibtex >/dev/null 2>&1; then
  bibtex main >/dev/null
elif [[ -x /usr/bin/bibtex.original ]]; then
  /usr/bin/bibtex.original main >/dev/null
else
  echo "No BibTeX executable found" >&2
  exit 1
fi
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

echo "Built paper/main.pdf"
