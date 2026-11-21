#!/usr/bin/env python3
"""A4 printable page: full line-art coloring sheet + colored reference top-right."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
LINEART = ROOT / "docs/coloring-page/sonic-lineart.jpg"
COLORED = ROOT / "docs/coloring-page/sonic-colored-reference.jpg"
OUT_DIR = ROOT / "docs/coloring-page"
PDF_OUT = OUT_DIR / "sonic-coloring-page-a4.pdf"
JPG_OUT = OUT_DIR / "sonic-coloring-page-a4.jpg"

DPI = 300
REF_W_MM = 62  # top-right reference width
MARGIN_MM = 12
GAP_MM = 4


def px(mm: float) -> int:
    return int(mm / 25.4 * DPI)


def fit(w: int, h: int, box_w: float, box_h: float) -> tuple[float, float]:
    s = min(box_w / w, box_h / h)
    return w * s, h * s


def export_pdf() -> None:
    page_w, page_h = A4
    margin = MARGIN_MM * mm
    c = canvas.Canvas(str(PDF_OUT), pagesize=A4)
    c.setTitle("Sonic coloring page")

    ref_w = REF_W_MM * mm
    ref_reader = ImageReader(str(COLORED))
    rw, rh = ref_reader.getSize()
    rdw, rdh = fit(rw, rh, ref_w, ref_w)
    ref_x = page_w - margin - rdw
    ref_y = page_h - margin - rdh

    # Frame around reference
    c.setStrokeColorRGB(0.75, 0.75, 0.75)
    c.setLineWidth(0.6)
    c.roundRect(ref_x - 2 * mm, ref_y - 2 * mm, rdw + 4 * mm, rdh + 4 * mm, 2 * mm, stroke=1, fill=0)
    c.drawImage(ref_reader, ref_x, ref_y, width=rdw, height=rdh, preserveAspectRatio=True, anchor="sw")

    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(ref_x, ref_y - 3.5 * mm, "Color guide")

    line_reader = ImageReader(str(LINEART))
    lw, lh = line_reader.getSize()
    mdw, mdh = fit(lw, lh, page_w - 2 * margin, page_h - 2 * margin)
    mx = (page_w - mdw) / 2
    my = (page_h - mdh) / 2
    c.drawImage(line_reader, mx, my, width=mdw, height=mdh, preserveAspectRatio=True, anchor="sw")

    # Redraw reference on top so it sits in the top-right corner
    c.roundRect(ref_x - 2 * mm, ref_y - 2 * mm, rdw + 4 * mm, rdh + 4 * mm, 2 * mm, stroke=1, fill=0)
    c.drawImage(ref_reader, ref_x, ref_y, width=rdw, height=rdh, preserveAspectRatio=True, anchor="sw")
    c.drawString(ref_x, ref_y - 3.5 * mm, "Color guide")

    c.showPage()
    c.save()


def export_jpeg() -> None:
    pw, ph = px(210), px(297)
    img = Image.new("RGB", (pw, ph), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = px(MARGIN_MM)

    colored = Image.open(COLORED).convert("RGB")
    ref_w = px(REF_W_MM)
    cw, ch = colored.size
    rdw, rdh = fit(cw, ch, ref_w, ref_w)
    colored_r = colored.resize((int(rdw), int(rdh)), Image.Resampling.LANCZOS)
    ref_x = pw - margin - int(rdw)
    ref_y = margin
    pad = px(2)

    line = Image.open(LINEART).convert("RGB")
    main_w = pw - 2 * margin
    main_h = ph - 2 * margin
    lw, lh = line.size
    mdw, mdh = fit(lw, lh, main_w, main_h)
    line_r = line.resize((int(mdw), int(mdh)), Image.Resampling.LANCZOS)
    mx = (pw - int(mdw)) // 2
    my = (ph - int(mdh)) // 2
    img.paste(line_r, (mx, my))

    draw = ImageDraw.Draw(img)
    inset = Image.new("RGB", (int(rdw) + 2 * pad, int(rdh) + 2 * pad + px(5)), (255, 255, 255))
    inset.paste(colored_r, (pad, pad))
    img.paste(inset, (ref_x - pad, ref_y - pad))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", px(3))
    except OSError:
        font = ImageFont.load_default()
    draw.rounded_rectangle(
        (ref_x - pad, ref_y - pad, ref_x + int(rdw) + pad, ref_y + int(rdh) + pad + px(5)),
        radius=px(2),
        outline=(120, 120, 120),
        width=2,
    )
    draw.text((ref_x, ref_y + int(rdh) + px(1)), "Color guide", fill=(40, 40, 40), font=font)

    img.save(JPG_OUT, quality=95, dpi=(DPI, DPI), optimize=True)


def main() -> None:
    if not COLORED.exists():
        import shutil

        src = Path("/opt/cursor/artifacts/assets/sonic-colored-reference.jpg")
        if src.is_file():
            shutil.copy(src, COLORED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_pdf()
    export_jpeg()
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {JPG_OUT}")


if __name__ == "__main__":
    main()