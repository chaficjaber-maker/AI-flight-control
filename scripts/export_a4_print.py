#!/usr/bin/env python3
"""Export the SE Platform system context diagram for A4 printing (PDF + JPEG)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
# Prefer repo copy; fall back to fetched asset path.
DIAGRAM_CANDIDATES = [
    ROOT / "docs/diagrams/se-platform-system-context.png",
    Path("/tmp/se-platform.png"),
]

OUT_DIR = ROOT / "docs/print"
PDF_OUT = OUT_DIR / "se-platform-system-context-a4.pdf"
JPG_OUT = OUT_DIR / "se-platform-system-context-a4.jpg"

# 300 DPI A4 landscape (print quality)
DPI = 300
A4_L_W_MM, A4_L_H_MM = 297.0, 210.0
PX_W = int(A4_L_W_MM / 25.4 * DPI)
PX_H = int(A4_L_H_MM / 25.4 * DPI)
MARGIN_MM = 12


def find_diagram() -> Path:
    for p in DIAGRAM_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit("System context diagram PNG not found.")


def fit_size(img_w: int, img_h: int, box_w: float, box_h: float) -> tuple[float, float]:
    scale = min(box_w / img_w, box_h / img_h)
    return img_w * scale, img_h * scale


def export_pdf(diagram: Path) -> None:
    page_w, page_h = landscape(A4)
    margin = MARGIN_MM * mm
    c = canvas.Canvas(str(PDF_OUT), pagesize=landscape(A4))
    c.setTitle("SE Platform - System Context")

    reader = ImageReader(str(diagram))
    iw, ih = reader.getSize()
    dw, dh = fit_size(iw, ih, page_w - 2 * margin, page_h - 2 * margin - 8 * mm)
    x = (page_w - dw) / 2
    y = margin + 6 * mm
    c.drawImage(reader, x, y, width=dw, height=dh, preserveAspectRatio=True, anchor="sw")

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawCentredString(page_w / 2, margin, "SE Platform — System Context")
    c.showPage()
    c.save()


def export_jpeg(diagram: Path) -> None:
    margin_px = int(MARGIN_MM / 25.4 * DPI)
    footer_px = int(8 / 25.4 * DPI)
    canvas_img = Image.new("RGB", (PX_W, PX_H), (255, 255, 255))
    src = Image.open(diagram).convert("RGB")
    box_w = PX_W - 2 * margin_px
    box_h = PX_H - 2 * margin_px - footer_px
    dw, dh = fit_size(src.width, src.height, box_w, box_h)
    resized = src.resize((int(dw), int(dh)), Image.Resampling.LANCZOS)
    x = (PX_W - int(dw)) // 2
    y = margin_px + footer_px
    canvas_img.paste(resized, (x, y))
    canvas_img.save(JPG_OUT, quality=95, dpi=(DPI, DPI), optimize=True)


def main() -> None:
    diagram = find_diagram()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_pdf(diagram)
    export_jpeg(diagram)
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {JPG_OUT}")


if __name__ == "__main__":
    main()
