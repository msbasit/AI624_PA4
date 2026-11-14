"""Render the completed Markdown source to a readable PDF, including LaTeX."""

import html
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    Preformatted,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp").mkdir(exist_ok=True)
text = (ROOT / "results.md").read_text(encoding="utf-8")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="BodyCustom", fontName="Helvetica", fontSize=9, leading=13, spaceAfter=7
    )
)
styles.add(
    ParagraphStyle(name="CellCustom", fontName="Helvetica", fontSize=7, leading=9)
)
styles.add(
    ParagraphStyle(
        name="CodeCustom", fontName="Courier", fontSize=7, leading=10, spaceAfter=8
    )
)
styles["Heading1"].fontSize = 19
styles["Heading1"].leading = 24
styles["Heading2"].fontSize = 13
styles["Heading2"].leading = 17
styles["Heading2"].spaceBefore = 15
styles["Heading2"].spaceAfter = 8
styles["Heading2"].keepWithNext = True
styles.add(
    ParagraphStyle(name="PromptCustom", parent=styles["BodyCustom"], keepWithNext=True)
)


def inline(s):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2" color="#165b9b">\1</a>', s)
    # Avoid a control character from any accidental Python escape in source.
    assert all(ord(c) >= 32 or c in "\n\t" for c in s)
    return s


story = []
lines = text.splitlines()
index = 0
eq_count = 0
PAGE_WIDTH = 487


def add_image(path, max_height=190):
    width, height = PILImage.open(path).size
    scale = min(PAGE_WIDTH / width, max_height / height)
    figure = Image(str(path), width=width * scale, height=height * scale)
    figure.keepWithNext = True
    spacer = Spacer(1, 7)
    spacer.keepWithNext = True
    story.extend([figure, spacer])


while index < len(lines):
    line = lines[index].strip()
    if not line:
        index += 1
        continue
    if line.startswith("$$"):
        equation = line.strip("$").replace(r"\mathcal S", r"\mathcal{S}")
        if not line.endswith("$$") or line == "$$":
            parts = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                parts.append(lines[index])
                index += 1
            equation = " ".join(parts)
        fig = plt.figure(figsize=(10, 1))
        fig.text(0.01, 0.5, "$" + equation + "$", fontsize=15, va="center")
        path = ROOT / "tmp" / f"equation_{eq_count}.png"
        eq_count += 1
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.07)
        plt.close(fig)
        add_image(path, 45)
        index += 1
        continue
    if line.startswith("```"):
        code = []
        index += 1
        while index < len(lines) and not lines[index].startswith("```"):
            code.append(lines[index])
            index += 1
        story.append(Preformatted("\n".join(code), styles["CodeCustom"]))
        index += 1
        continue
    if line.startswith("|"):
        rows = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            if not all(re.fullmatch(r"[-: ]+", v) for v in row):
                rows.append(row)
            index += 1
        count = len(rows[0])
        weights = [
            max(8, min(35, max(len(row[j]) for row in rows))) for j in range(count)
        ]
        widths = [PAGE_WIDTH * w / sum(weights) for w in weights]
        formatted = [
            [Paragraph(inline(cell), styles["CellCustom"]) for cell in row]
            for row in rows
        ]
        tab = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tab.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf5")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c6d4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([tab, Spacer(1, 9)])
        continue
    image_match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
    if image_match:
        add_image(ROOT / image_match.group(2))
        index += 1
        continue
    if line.startswith("# "):
        story.append(Paragraph(inline(line[2:]), styles["Heading1"]))
        index += 1
        continue
    if line.startswith("## "):
        story.append(Paragraph(inline(line[3:]), styles["Heading2"]))
        index += 1
        continue
    if line.startswith("- "):
        story.append(Paragraph("- " + inline(line[2:]), styles["BodyCustom"]))
        index += 1
        continue
    if line.startswith("> "):
        story.append(Paragraph("<i>" + inline(line[2:]) + "</i>", styles["BodyCustom"]))
        index += 1
        continue
    paragraph = [line]
    index += 1
    while (
        index < len(lines)
        and lines[index].strip()
        and not lines[index].startswith(("#", "|", "!", "$$", "```", "- ", "> "))
    ):
        paragraph.append(lines[index].strip())
        index += 1
    style = (
        styles["PromptCustom"]
        if paragraph[0].startswith(("**Q1.**", "**Answer:**"))
        else styles["BodyCustom"]
    )
    story.append(Paragraph(inline(" ".join(paragraph)), style))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#506070"))
    canvas.drawString(54, 28, "AI624 PA4 | Muhammad Sami Ul Basit | Roll No: 25280025")
    canvas.drawRightString(541, 28, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(ROOT / "results.pdf"),
    pagesize=(595, 842),
    rightMargin=54,
    leftMargin=54,
    topMargin=44,
    bottomMargin=45,
    title="AI624 PA4 Results",
    author="Muhammad Sami Ul Basit",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Rendered results.pdf from results.md")
