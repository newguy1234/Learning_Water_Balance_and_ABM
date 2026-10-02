# -*- coding: utf-8 -*-
"""Convert ems_full_manuscript.md -> Overleaf elsarticle project."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "ems_full_manuscript.md"
OUT = ROOT / "overleaf"
FIG = OUT / "figures"

SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc_tt(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in SPECIAL.items():
        s = s.replace(a, b)
    return s


def esc_text(s: str) -> str:
    """Escape plain text; no math/code markers left."""
    out = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(SPECIAL.get(ch, ch))
    return "".join(out)


def _plain_symbols(s: str) -> str:
    return (
        s.replace("—", "---")
        .replace("–", "--")
        .replace("×", r"$\times$")
        .replace("≈", r"$\approx$")
        .replace("≥", r"$\geq$")
        .replace("≤", r"$\leq$")
        .replace("⁻¹", r"$^{-1}$")
        .replace("³", r"$^{3}$")
        .replace("→", r"$\rightarrow$")
        .replace("…", "...")
        .replace("◄", "<--")
        .replace("│", "|")
        .replace("·", ".")
        .replace("“", "``")
        .replace("”", "''")
        .replace("‘", "'")
        .replace("’", "'")
    )


def _code_symbols(s: str) -> str:
    """ASCII-safe replacements inside ``code`` spans."""
    return (
        s.replace("—", "--")
        .replace("–", "-")
        .replace("→", "->")
        .replace("…", "...")
        .replace("×", "x")
    )


def md_inline(s: str) -> str:
    """Markdown inline -> LaTeX, preserving math and code."""
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s)

    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "$":
            j = s.find("$", i + 1)
            if j == -1:
                out.append(r"\$")
                i += 1
                continue
            out.append(s[i : j + 1])
            i = j + 1
            continue
        if s[i] == "`":
            j = s.find("`", i + 1)
            if j == -1:
                out.append(esc_text(s[i]))
                i += 1
                continue
            out.append(r"\texttt{" + esc_tt(_code_symbols(s[i + 1 : j])) + "}")
            i = j + 1
            continue
        if s.startswith("**", i):
            j = s.find("**", i + 2)
            if j != -1:
                out.append(r"\textbf{" + md_inline(s[i + 2 : j]) + "}")
                i = j + 2
                continue
        if s[i] == "*" and (i == 0 or s[i - 1].isspace() or s[i - 1] in "([{,"):
            j = s.find("*", i + 1)
            if j != -1 and not s.startswith("**", j):
                out.append(r"\textit{" + md_inline(s[i + 1 : j]) + "}")
                i = j + 1
                continue
        # curly / smart quotes as characters (before ASCII `` is created)
        if s[i] in "“":
            out.append("``")
            i += 1
            continue
        if s[i] in "”":
            out.append("''")
            i += 1
            continue
        j = i + 1
        while j < n and s[j] not in "`$*“”":
            j += 1
        chunk = _plain_symbols(s[i:j])
        # emit chunk; protect `` and '' quote markers and $math$
        k = 0
        while k < len(chunk):
            if chunk.startswith("``", k):
                out.append("``")
                k += 2
                continue
            if chunk.startswith("''", k):
                out.append("''")
                k += 2
                continue
            if chunk[k] == "$":
                e = chunk.find("$", k + 1)
                if e == -1:
                    out.append(esc_text(chunk[k]))
                    k += 1
                    continue
                out.append(chunk[k : e + 1])
                k = e + 1
                continue
            e = k + 1
            while e < len(chunk) and chunk[e] != "$" and not chunk.startswith("``", e) and not chunk.startswith("''", e):
                e += 1
            out.append(esc_text(chunk[k:e]))
            k = e
        i = j
    return "".join(out)


def strip_sec_num(title: str) -> str:
    return re.sub(r"^[\d.]+\s+", "", title).strip()


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    if len(rows) < 2:
        return "", start
    body = [rows[0]] + [
        r for r in rows[1:] if not all(re.match(r"^:?-+:?$", c) for c in r)
    ]
    ncol = len(body[0])
    # First column left; rest centred for numeric tables, left if wide text
    first_lens = [len(r[0]) for r in body]
    if max(first_lens) > 28 or ncol >= 5:
        colspec = "p{0.22\\linewidth}" + "c" * (ncol - 1)
    else:
        colspec = "l" + "c" * (ncol - 1)
    lines_out = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        " & ".join(md_inline(c) for c in body[0]) + r" \\",
        r"\midrule",
    ]
    for r in body[1:]:
        while len(r) < ncol:
            r.append("")
        lines_out.append(" & ".join(md_inline(c) for c in r[:ncol]) + r" \\")
    lines_out += [
        r"\bottomrule",
        r"\end{tabular}",
        "CAPTION_PLACEHOLDER",
        r"\end{table}",
    ]
    return "\n".join(lines_out), i


def finalize_table(tab: str, caption: str | None) -> str:
    if caption:
        return tab.replace(
            "CAPTION_PLACEHOLDER",
            r"\caption{" + md_inline(caption) + "}",
        )
    return tab.replace("\nCAPTION_PLACEHOLDER", "").replace("CAPTION_PLACEHOLDER", "")


def md_block_to_tex(md: str, *, is_appendix: bool = False) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    pending_caption = None
    in_list = False
    list_type = None
    appendix_started = False

    def close_list():
        nonlocal in_list, list_type
        if in_list:
            out.append(r"\end{" + list_type + "}")
            in_list = False
            list_type = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_list()
            out.append("")
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue

        if stripped.startswith("## "):
            close_list()
            title = stripped[3:].strip()
            if title.startswith("Appendix"):
                m = re.match(r"Appendix ([A-Z])\.\s*(.*)", title)
                if not appendix_started:
                    out.append(r"\appendix")
                    appendix_started = True
                if m:
                    out.append(r"\section{" + md_inline(m.group(2)) + "}")
                else:
                    out.append(r"\section{" + md_inline(title) + "}")
            elif title in {
                "Data availability",
                "CRediT authorship contribution statement",
                "Declaration of competing interest",
                "Declaration of generative AI and AI-assisted technologies in the writing process",
                "Acknowledgements",
            }:
                out.append(r"\section*{" + md_inline(title) + "}")
            else:
                out.append(r"\section{" + md_inline(strip_sec_num(title)) + "}")
            i += 1
            continue

        if stripped.startswith("### "):
            close_list()
            title = strip_sec_num(stripped[4:].strip()).replace("—", "---")
            out.append(r"\subsection{" + md_inline(title) + "}")
            i += 1
            continue

        if stripped.startswith("**Table "):
            close_list()
            cap = re.sub(r"^\*\*Table\s+\d+\.\*\*\s*", "", stripped)
            cap = cap.strip("*").strip()
            cap = re.sub(r"^Table\s+\d+\.\s*", "", cap)
            pending_caption = cap
            i += 1
            continue

        if stripped.startswith("|"):
            close_list()
            tab, i = parse_table(lines, i)
            out.append(finalize_table(tab, pending_caption))
            pending_caption = None
            continue

        if stripped.startswith("!["):
            close_list()
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            fname = Path(m.group(2)).name if m else "figure.png"
            cap = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("**Fig."):
                cap = re.sub(
                    r"^Fig\.\s*\d+\.\s*", "", lines[i + 1].strip().strip("*")
                )
                i += 2
            else:
                i += 1
            out.append(r"\begin{figure}[htbp]")
            out.append(r"\centering")
            out.append(
                r"\includegraphics[width=0.95\linewidth]{figures/" + fname + "}"
            )
            if cap:
                out.append(r"\caption{" + md_inline(cap) + "}")
            out.append(r"\label{fig:" + Path(fname).stem + "}")
            out.append(r"\end{figure}")
            continue

        if stripped.startswith("**Fig."):
            i += 1
            continue

        if stripped.startswith("```"):
            close_list()
            lang = stripped[3:].strip() or "text"
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            if lang in ("text", ""):
                out.append(r"\begin{verbatim}")
                # strip non-ascii box drawing for verbatim safety
                for c in code:
                    c2 = (
                        c.replace("│", "|")
                        .replace("┌", "+")
                        .replace("┐", "+")
                        .replace("└", "+")
                        .replace("┘", "+")
                        .replace("├", "+")
                        .replace("┼", "+")
                        .replace("─", "-")
                        .replace("▼", "v")
                        .replace("◄", "<")
                        .replace("►", ">")
                        .replace("·", "*")
                    )
                    out.append(c2)
                out.append(r"\end{verbatim}")
            else:
                lstlang = "Python" if lang == "python" else "Bash"
                out.append(r"\begin{lstlisting}[language=" + lstlang + "]")
                out.extend(code)
                out.append(r"\end{lstlisting}")
            continue

        if stripped == r"\[" or (stripped.startswith(r"\[") and not stripped.endswith(r"\]")):
            close_list()
            i += 1 if stripped == r"\[" else 0
            math = []
            if stripped.startswith(r"\[") and stripped != r"\[":
                math.append(stripped[2:])
                i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                math.append(lines[i])
                i += 1
            i += 1
            block = "\n".join(math)
            block = re.sub(r",?\s*\\tag\{[^}]+\}", "", block)
            out.append(r"\begin{equation}")
            out.append(block.strip())
            out.append(r"\end{equation}")
            continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            want = "enumerate" if ordered else "itemize"
            if not in_list or list_type != want:
                close_list()
                out.append(r"\begin{" + want + "}")
                in_list = True
                list_type = want
            item = re.sub(r"^([-*]|\d+\.)\s+", "", stripped)
            out.append(r"\item " + md_inline(item))
            i += 1
            continue

        close_list()
        # Fig. 1 caption after verbatim
        if stripped.startswith("**Fig. 1.**"):
            cap = re.sub(r"^Fig\.\s*1\.\s*", "", stripped.strip("*").strip())
            out.append(r"\begin{figure}[htbp]")
            out.append(r"\centering")
            out.append(
                r"\fbox{\parbox{0.92\linewidth}{\centering\small "
                r"Architecture diagram (see verbatim listing above; "
                r"redraw as vector art for submission.)}}"
            )
            out.append(r"\caption{" + md_inline(cap) + "}")
            out.append(r"\label{fig:architecture}")
            out.append(r"\end{figure}")
            i += 1
            continue

        out.append(md_inline(stripped))
        i += 1

    close_list()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def refs_to_thebibliography(refs_md: str) -> str:
    entries = []
    for para in re.split(r"\n\s*\n", refs_md.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        para = re.sub(r"\s*\*\*\[[^\]]+\]\*\*", "", para)
        m = re.match(r"^([^,]+),.*?(\d{4}[a-z]?)\.", para)
        if m:
            key = re.sub(r"[^A-Za-z]", "", m.group(1).split()[-1]) + m.group(2)
        else:
            key = f"ref{len(entries)+1}"
        # unique keys
        base = key
        k = 1
        existing = {e[0] for e in entries}
        while key in existing:
            k += 1
            key = f"{base}{k}"
        entries.append((key, para))

    lines = [r"\begin{thebibliography}{99}"]
    for key, para in entries:
        p = para.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
        p = p.replace("—", "---").replace("–", "--")
        # underscore only outside urls
        p = re.sub(
            r"https://doi\.org/(\S+)",
            lambda m: r"\doi{" + m.group(1).rstrip(".,;") + "}",
            p,
        )
        p = re.sub(
            r"(?<!\{)(https?://[^\s\}]+)",
            lambda m: r"\url{" + m.group(1).rstrip(".,;") + "}",
            p,
        )
        # remaining underscores
        parts = re.split(r"(\\doi\{[^}]+\}|\\url\{[^}]+\})", p)
        fixed = []
        for part in parts:
            if part.startswith(r"\doi{") or part.startswith(r"\url{"):
                fixed.append(part)
            else:
                fixed.append(part.replace("_", r"\_"))
        p = "".join(fixed)
        lines.append(r"\bibitem{" + key + "}")
        lines.append(p)
        lines.append("")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def extract_front(md: str) -> dict:
    abstract = re.search(
        r"## Abstract\n\n(.+?)\n\n\*\*Keywords:\*\*\s*(.+?)\n", md, re.S
    )
    highlights = re.findall(r"## Highlights\n\n((?:- .+\n)+)", md)
    hl = []
    if highlights:
        hl = [re.sub(r"^- ", "", x).strip() for x in highlights[0].strip().splitlines()]
    soft = re.search(r"## Software availability\n\n(.+?)\n\n---", md, re.S)
    return {
        "abstract": abstract.group(1).strip() if abstract else "",
        "keywords": [k.strip() for k in abstract.group(2).split(";")] if abstract else [],
        "highlights": hl,
        "software_table": soft.group(1).strip() if soft else "",
    }


def software_table_tex(md_table: str) -> str:
    lines = [l for l in md_table.splitlines() if l.strip().startswith("|")]
    rows = []
    for l in lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in cells):
            continue
        rows.append(cells)
    body = rows[1:]
    out = [
        r"\section*{Software availability}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{p{0.28\linewidth}p{0.62\linewidth}}",
        r"\toprule",
        r"Item & Detail \\",
        r"\midrule",
    ]
    for r in body:
        out.append(md_inline(r[0]) + " & " + md_inline(r[1]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def split_manuscript(md: str):
    # From ## 1. Introduction through before ## References
    m = re.search(r"\n## 1\. Introduction\n", md)
    if not m:
        raise SystemExit("missing Introduction")
    rest = md[m.start() + 1 :]  # keep ## 1. Introduction
    refs_m = re.search(r"\n## References\n", rest)
    body_all = rest[: refs_m.start()]
    after = rest[refs_m.end() :]
    app_m = re.search(r"\n## Appendix A\.", after)
    refs_md = after[: app_m.start()] if app_m else after
    apps_md = after[app_m.start() + 1 :] if app_m else ""

    # Split back matter from body
    back_keys = (
        r"## Data availability\n|"
        r"## CRediT authorship contribution statement\n"
    )
    bm = re.search(back_keys, body_all)
    if bm:
        body_md = body_all[: bm.start()]
        back_md = body_all[bm.start() :]
    else:
        body_md, back_md = body_all, ""
    return body_md, back_md, refs_md, apps_md


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    # drop pointer line if present
    if md.startswith(">"):
        md = md.split("\n\n", 1)[-1]
        if md.startswith(">"):
            md = re.sub(r"^>.*\n\n", "", md)

    front = extract_front(md)
    body_md, back_md, refs_md, apps_md = split_manuscript(md)

    body_tex = md_block_to_tex(body_md)
    back_tex = md_block_to_tex(back_md)
    apps_tex = md_block_to_tex(apps_md, is_appendix=True)
    bib_tex = refs_to_thebibliography(refs_md)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "ems").glob("fig*.png"):
        shutil.copy2(src, FIG / src.name)

    hl_items = "\n".join(r"\item " + md_inline(h) for h in front["highlights"])
    kws = " \\sep ".join(md_inline(k) for k in front["keywords"])

    preamble = r"""%% CyclicEnv manuscript for Environmental Modelling & Software (Elsevier)
%% Overleaf: Menu -> Main document = main.tex; Compiler = pdfLaTeX
\documentclass[preprint,12pt,authoryear,review]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{doi}
\usepackage{url}
\usepackage{tabularx}

\journal{Environmental Modelling \& Software}

\lstset{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  frame=single,
  columns=fullflexible,
  keepspaces=true
}

\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\begin{document}

\begin{frontmatter}

\title{CyclicEnv: an open Gymnasium environment and benchmark for learning unobserved water fate in a cyclic-irrigation paddy district}

\author[aff1]{Author One\corref{cor1}}
\ead{author.one@example.edu}
\author[aff1]{Author Two}
\address[aff1]{Affiliation to be completed}
\cortext[cor1]{Corresponding author}

\begin{abstract}
""" + md_inline(front["abstract"]) + r"""
\end{abstract}

\begin{highlights}
""" + hl_items + r"""
\end{highlights}

\begin{keyword}
""" + kws + r"""
\end{keyword}

\end{frontmatter}

""" + software_table_tex(front["software_table"]) + "\n\n"

    main_tex = (
        preamble
        + body_tex
        + "\n\n"
        + back_tex
        + "\n\n"
        + bib_tex
        + "\n\n"
        + apps_tex
        + "\n\n\\end{document}\n"
    )
    (OUT / "main.tex").write_text(main_tex, encoding="utf-8")

    (OUT / "README.md").write_text(
        """# CyclicEnv — Overleaf LaTeX

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
""",
        encoding="utf-8",
    )

    # Pointer on markdown
    raw = MD.read_text(encoding="utf-8")
    if "overleaf/main.tex" not in raw[:500]:
        MD.write_text(
            "> **Overleaf LaTeX:** [`overleaf/main.tex`](overleaf/main.tex) "
            "(zip `overleaf/` and upload to Overleaf).\n\n" + raw,
            encoding="utf-8",
        )

    print("Wrote", OUT / "main.tex", "lines", len(main_tex.splitlines()))
    print("figures:", sorted(p.name for p in FIG.glob("*")))


if __name__ == "__main__":
    main()
