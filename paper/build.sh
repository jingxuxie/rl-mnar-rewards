#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODE="${AAAI_MODE:-preprint}"

bibtex_command() {
  if command -v bibtex >/dev/null 2>&1; then
    command bibtex "$1"
  elif [[ -x /usr/bin/bibtex.original ]]; then
    /usr/bin/bibtex.original "$1"
  else
    echo "No BibTeX executable found" >&2
    exit 1
  fi
}

latex_once() {
  local name="$1"
  if [[ "$MODE" == "review" && -f aaai2027.sty ]]; then
    pdflatex -halt-on-error -interaction=nonstopmode -jobname="$name" \
      "\\def\\AAAIReview{1}\\input{${name}.tex}" >/dev/null
  else
    pdflatex -halt-on-error -interaction=nonstopmode "$name.tex" >/dev/null
  fi
}

build_bibliography_document() {
  local name="$1"
  latex_once "$name"
  bibtex_command "$name" >/dev/null
  latex_once "$name"
  latex_once "$name"
  if grep -Eq 'LaTeX Warning: (There were undefined references|Citation .* undefined)' "$name.log"; then
    echo "Undefined citation/reference in $name" >&2
    exit 1
  fi
}

build_bibliography_document main
build_bibliography_document supplement

if [[ -f aaai2027.sty ]]; then
  pdflatex -halt-on-error -interaction=nonstopmode ReproducibilityChecklist.tex >/dev/null
  pdflatex -halt-on-error -interaction=nonstopmode ReproducibilityChecklist.tex >/dev/null
else
  echo "Skipping standalone checklist locally: aaai2027.sty is not installed."
fi

echo "Built paper/main.pdf and paper/supplement.pdf${MODE:+ in $MODE mode}."
