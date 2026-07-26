# Paper build

`main.tex` is the seven-content-page manuscript, `supplement.tex` contains complete proofs and additional experiments, and `ReproducibilityChecklist.tex` is compiled separately.

Without `aaai2027.sty`, `bash build.sh` uses a local two-column fallback for development. The GitHub paper workflow downloads the unmodified official AAAI-27 author kit, builds with `AAAI_MODE=review`, and runs `scripts/check_submission.py`.

```bash
bash build.sh
```

Do not append the reproducibility checklist to `main.pdf`; the conference requests it as a separate upload.
