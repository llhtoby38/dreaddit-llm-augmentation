"""Render report/report.md into NLP4Health_assignment3_LEE.docx.

Run AFTER you have hand-written the prose in report.md and after the experiment
artefacts (figures, tables, audit summary) have been produced. The student then
opens the .docx in Word, double-checks formatting, and exports to PDF.

JAMIA style conventions applied:
    - Title page: title, author, student ID, course, date.
    - Single column. Times New Roman 11pt body, 12pt headings (bold).
    - Page numbers in the footer.
    - Tables rendered with cell borders.
    - Markdown headings (#, ##, ###) become Word Heading 1/2/3.
    - Markdown <!-- comments --> are stripped (planning notes do not ship).
    - Markdown ``` code ``` blocks become Courier-New monospaced paragraphs.
    - Markdown tables become docx tables.
    - Markdown images (![alt](path)) are embedded (paths resolved relative to report.md).

Usage:
    python code/build_docx.py
"""

from __future__ import annotations
from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

A3 = Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\Assignment 3")
SRC = A3 / "report" / "report.md"
OUT = A3 / "report" / "NLP4Health_assignment3_LEE.docx"


def _resolve_out_path(path):
    """If the canonical path is locked (open in Word), pick the next suffix."""
    if not path.exists():
        return path
    try:
        with path.open("ab"):
            return path
    except PermissionError:
        for i in range(2, 50):
            candidate = path.with_stem(f"{path.stem}_v{i}")
            try:
                if candidate.exists():
                    with candidate.open("ab"):
                        return candidate
                else:
                    return candidate
            except PermissionError:
                continue
        raise


# --- helpers -----------------------------------------------------------------


def strip_html_comments(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


INLINE_BOLD = re.compile(r"\*\*([^*]+)\*\*")
INLINE_ITAL = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
INLINE_CODE = re.compile(r"`([^`]+)`")
INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def add_runs(paragraph, text, base_size=11):
    """Walk the text adding inline-formatted runs for bold/italic/code/links."""
    # Strip ![...](...) before plain links so images are handled by caller.
    pos = 0
    pattern = re.compile(r"(\*\*[^*]+\*\*|(?<!\*)\*[^*\n]+\*(?!\*)|`[^`]+`|\[[^\]]+\]\([^)]+\))")
    for m in pattern.finditer(text):
        if m.start() > pos:
            run = paragraph.add_run(text[pos:m.start()])
            run.font.size = Pt(base_size)
        tok = m.group()
        if tok.startswith("**") and tok.endswith("**"):
            run = paragraph.add_run(tok[2:-2])
            run.bold = True
            run.font.size = Pt(base_size)
        elif tok.startswith("`"):
            run = paragraph.add_run(tok[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(base_size - 1)
        elif tok.startswith("["):
            lm = INLINE_LINK.match(tok)
            run = paragraph.add_run(lm.group(1))
            run.font.size = Pt(base_size)
            run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        else:
            run = paragraph.add_run(tok[1:-1])
            run.italic = True
            run.font.size = Pt(base_size)
        pos = m.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.font.size = Pt(base_size)


def add_page_number(doc):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    p._p.append(fld)


def set_cell_borders(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "bottom", "left", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "888888")
        borders.append(b)
    tc_pr.append(borders)


def add_table(doc, header, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(header):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        add_runs(p, h.strip(), base_size=10)
        for r in p.runs:
            r.bold = True
        set_cell_borders(hdr_cells[i])
    for ri, row in enumerate(rows, start=1):
        for ci, cell_text in enumerate(row):
            cell = table.rows[ri].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            add_runs(p, cell_text.strip(), base_size=10)
            set_cell_borders(cell)
    doc.add_paragraph()


# --- markdown parsing --------------------------------------------------------


def is_table_separator(line):
    s = line.strip()
    if not s or not s.startswith("|"):
        return False
    inner = s.strip("|")
    return bool(re.fullmatch(r"\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", "|" + inner + "|")) or bool(
        re.fullmatch(r"\s*(:?-+:?\s*\|?\s*)+", inner)
    )


def parse_table_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(doc, md_text):
    md_text = strip_html_comments(md_text)
    lines = md_text.splitlines()

    # Title page
    doc.add_paragraph()
    title_line = None
    for ln in lines:
        if ln.startswith("# "):
            title_line = ln[2:].strip()
            break
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run(title_line or "Untitled")
    tr.bold = True
    tr.font.size = Pt(20)
    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Toby Lee — Student ID 1690649")
    sr.font.size = Pt(13)
    doc.add_paragraph()
    course = doc.add_paragraph()
    course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = course.add_run("COMP90090 Text Analytics for Health — Independent Project")
    cr.font.size = Pt(12)
    cr.italic = True
    doc.add_page_break()

    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # code fences
        if stripped.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_buf))
                run.font.name = "Consolas"
                run.font.size = Pt(9)
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        # skip title (we already rendered it)
        if line.startswith("# "):
            i += 1
            continue
        # headings
        if line.startswith("## "):
            p = doc.add_paragraph()
            p.style = doc.styles["Heading 1"]
            add_runs(p, line[3:].strip(), base_size=14)
            i += 1
            continue
        if line.startswith("### "):
            p = doc.add_paragraph()
            p.style = doc.styles["Heading 2"]
            add_runs(p, line[4:].strip(), base_size=12)
            i += 1
            continue
        if line.startswith("#### "):
            p = doc.add_paragraph()
            p.style = doc.styles["Heading 3"]
            add_runs(p, line[5:].strip(), base_size=11)
            i += 1
            continue
        # tables
        if line.lstrip().startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            header = parse_table_row(line)
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(parse_table_row(lines[j]))
                j += 1
            add_table(doc, header, rows)
            i = j
            continue
        # bullets
        if re.match(r"^\s*-\s+", line):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, re.sub(r"^\s*-\s+", "", line))
            i += 1
            continue
        # numbered
        if re.match(r"^\s*\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            add_runs(p, re.sub(r"^\s*\d+\.\s+", "", line))
            i += 1
            continue
        # horizontal rules
        if stripped in ("---", "***", "___"):
            i += 1
            continue
        # blank line
        if not stripped:
            i += 1
            continue
        # markdown images: ![alt](path) — embed at 6.5" width
        img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if img_match:
            alt, path = img_match.group(1), img_match.group(2)
            # resolve relative to report.md location
            from docx.shared import Inches
            from pathlib import Path
            img_path = (SRC.parent / path).resolve()
            if img_path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(str(img_path), width=Inches(6.5))
            else:
                p = doc.add_paragraph(f"[MISSING IMAGE: {path}]")
            i += 1
            continue
        # plain paragraph
        p = doc.add_paragraph()
        add_runs(p, line)
        i += 1


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    md = SRC.read_text(encoding="utf-8")
    doc = Document()

    # apply JAMIA-ish base style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    for hn, hs in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)):
        s = doc.styles[hn]
        s.font.name = "Calibri"
        s.font.size = Pt(hs)
        s.font.bold = True
        s.font.color.rgb = RGBColor(0, 0, 0)

    # page margins
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    add_page_number(doc)
    render(doc, md)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = _resolve_out_path(OUT)
    doc.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
