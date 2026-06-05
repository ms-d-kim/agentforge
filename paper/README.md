# AgentForge — paper source (Overleaf-ready)

Upload **`agentforge_overleaf.zip`** to Overleaf (*New Project → Upload Project*) and it
compiles as-is. `main.tex` is the entry point; `IEEEtran.cls`/`IEEEtran.bst` ship with
Overleaf, so nothing else needs uploading.

## Contents
- `main.tex` — the paper (IEEEtran conference, two-column).
- `references.bib` — bibliography. *Verify arXiv IDs / venues before camera-ready.*
- `figures/` — the real figures from `results/` (architecture, BIRD curves, cross-domain
  sweep, multi-step RL).

## Build
Overleaf: set the compiler to **pdfLaTeX**; the menu runs `pdfLaTeX → BibTeX → pdfLaTeX × 2`.
Locally:
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Notes
- Every number in the paper matches the committed results (`results/exp_summary.txt`,
  the figures). The contextual/RL and cross-domain numbers are flagged as synthetic
  stand-ins in the Limitations section — keep that framing.
- To switch to a single-column article look, change the first line to
  `\documentclass[11pt]{article}` and remove the `\IEEEauthorblock*` / `\IEEEkeywords`
  wrappers.

## Regenerate the zip after edits
```bash
cd paper && zip -r ../agentforge_overleaf.zip main.tex references.bib figures README.md
```
