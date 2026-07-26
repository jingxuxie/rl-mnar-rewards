#!/usr/bin/env python3
"""Check the generated AAAI submission artifacts for hard format failures."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def command_output(*args: str) -> str:
    executable = shutil.which(args[0])
    if executable is None:
        raise RuntimeError(f"Required executable is missing: {args[0]}")
    return subprocess.check_output((executable, *args[1:]), text=True)


def pdf_pages(path: Path) -> tuple[int, str]:
    output = command_output("pdfinfo", str(path))
    page_match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    size_match = re.search(r"^Page size:\s+(.+)$", output, re.MULTILINE)
    if page_match is None or size_match is None:
        raise RuntimeError(f"Could not parse pdfinfo for {path}")
    return int(page_match.group(1)), size_match.group(1).strip()


def content_end_page(aux_path: Path) -> int:
    text = aux_path.read_text()
    match = re.search(r"\\newlabel\{content:end\}\{\{[^}]*\}\{(\d+)\}", text)
    if match is None:
        raise RuntimeError("Could not find content:end label in main.aux")
    return int(match.group(1))


def assert_fonts_embedded(path: Path) -> None:
    output = command_output("pdffonts", str(path))
    for line in output.splitlines()[2:]:
        if not line.strip():
            continue
        flags = re.search(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
        if flags is None:
            raise RuntimeError(f"Could not parse pdffonts row: {line}")
        if flags.group(1) != "yes":
            raise RuntimeError(f"Unembedded font in {path.name}: {line}")


def main() -> None:
    required = [
        PAPER / "main.pdf",
        PAPER / "main.aux",
        PAPER / "supplement.pdf",
        PAPER / "ReproducibilityChecklist.pdf",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    main_pages, main_size = pdf_pages(PAPER / "main.pdf")
    supplement_pages, supplement_size = pdf_pages(PAPER / "supplement.pdf")
    checklist_pages, checklist_size = pdf_pages(PAPER / "ReproducibilityChecklist.pdf")
    content_page = content_end_page(PAPER / "main.aux")

    if main_pages > 9:
        raise RuntimeError(f"Main PDF has {main_pages} pages; AAAI maximum is 9")
    if content_page > 7:
        raise RuntimeError(
            f"Non-reference content ends on page {content_page}; maximum is page 7"
        )
    for name, size in [
        ("main", main_size),
        ("supplement", supplement_size),
        ("checklist", checklist_size),
    ]:
        if "612 x 792" not in size and "letter" not in size.lower():
            raise RuntimeError(f"{name} is not US letter: {size}")

    assert_fonts_embedded(PAPER / "main.pdf")
    assert_fonts_embedded(PAPER / "supplement.pdf")
    assert_fonts_embedded(PAPER / "ReproducibilityChecklist.pdf")

    print(
        "Submission checks passed: "
        f"main={main_pages} pages (content through {content_page}), "
        f"supplement={supplement_pages}, checklist={checklist_pages}."
    )


if __name__ == "__main__":
    main()
