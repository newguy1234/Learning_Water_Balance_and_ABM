# CyclicEnv — Overleaf LaTeX

## Upload

1. Zip the contents of this `overleaf/` folder.
2. Overleaf → New Project → Upload Project.
3. Main document: `main.tex`. Compiler: **pdfLaTeX**.

## Contents

| File | Role |
|------|------|
| `main.tex` | Full EMS manuscript (`elsarticle`, Harvard / author–year) |
| `figures/fig2_*.png`, `figures/fig3_*.png` | Result figures |
| `README.md` | This file |

Author names, DOIs and funding are placeholders. Fig. 1 is a placeholder box (ASCII architecture is in a `verbatim` block); redraw for submission.

Regenerate from markdown:

```bash
python outputs/paper/md_to_overleaf.py
```
