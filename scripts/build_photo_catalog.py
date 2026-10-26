#!/usr/bin/env python3
"""Lay out 14 photos on exactly two A4 pages (fixed grid)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/catalog/assets"
OUT = ROOT / "docs/catalog/photo-catalog.pdf"

CAPTIONS = [
    "Power pose — sword & bear",
    "Pool day with Dad",
    "Inflatable playground",
    "After swim — towel ponchos",
    "Lounger & plush",
    "Pool edge smile",
    "Mall scooters",
    "Lunch & bear",
    "Scooter ride",
    "Story time on the sofa",
    "Reading together",
    "Paw Patrol play",
    "Zoo window",
    "Safari explorer",
]

PAGE_W, PAGE_H = A4
MARGIN = 10 * mm
COLS = 2
ROWS = 4


def sorted_photos() -> list[Path]:
    return sorted(ASSETS.glob("*.jpg"))


def fit_box(img_w: int, img_h: int, box_w: float, box_h: float) -> tuple[float, float, float, float]:
    scale = min(box_w / img_w, box_h / img_h)
    w, h = img_w * scale, img_h * scale
    x_off = (box_w - w) / 2
    y_off = (box_h - h) / 2
    return x_off, y_off, w, h


def draw_page(
    c: canvas.Canvas,
    items: list[tuple[Path, str]],
    *,
    header: bool,
) -> None:
    content_w = PAGE_W - 2 * MARGIN
    header_h = 14 * mm if header else 0
    footer_h = 6 * mm
    grid_top = PAGE_H - MARGIN - header_h
    grid_h = PAGE_H - 2 * MARGIN - header_h - footer_h
    cell_w = content_w / COLS
    cell_h = grid_h / ROWS
    caption_h = 5 * mm
    photo_box_h = cell_h - caption_h - 2 * mm

    if header:
        c.setFillColor(colors.HexColor("#2c3e50"))
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN - 5 * mm, "Photo Catalog")
        c.setFillColor(colors.HexColor("#666666"))
        c.setFont("Helvetica", 9)
        c.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN - 11 * mm, "14 moments · 2 pages")

    for idx, (path, caption) in enumerate(items):
        row, col = divmod(idx, COLS)
        cell_x = MARGIN + col * cell_w
        cell_y = grid_top - (row + 1) * cell_h

        with Image.open(path) as im:
            iw, ih = im.size
        x_off, y_off, dw, dh = fit_box(iw, ih, cell_w - 4 * mm, photo_box_h)
        img_x = cell_x + 2 * mm + x_off
        img_y = cell_y + caption_h + 1 * mm + y_off
        c.drawImage(str(path), img_x, img_y, width=dw, height=dh, preserveAspectRatio=True, anchor="sw")

        c.setFillColor(colors.HexColor("#444444"))
        c.setFont("Helvetica", 7)
        c.drawCentredString(cell_x + cell_w / 2, cell_y + 1.5 * mm, caption)

    c.setFillColor(colors.HexColor("#888888"))
    c.setFont("Helvetica", 8)
    page_num = 1 if header else 2
    c.drawCentredString(PAGE_W / 2, MARGIN - 2 * mm, f"Page {page_num} of 2")


def build() -> None:
    photos = sorted_photos()
    if len(photos) != 14:
        raise SystemExit(f"Expected 14 photos, found {len(photos)}")

    pairs = list(zip(photos, CAPTIONS, strict=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Photo Catalog")
    draw_page(c, pairs[:7], header=True)
    c.showPage()
    draw_page(c, pairs[7:], header=False)
    c.save()
    print(f"Wrote {OUT} ({len(photos)} photos, 2 pages)")


if __name__ == "__main__":
    build()
