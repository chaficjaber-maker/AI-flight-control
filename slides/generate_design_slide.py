#!/usr/bin/env python3
"""Generate AI Flight Control design-analysis slide as .pptx."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path("/opt/cursor/artifacts/ai-flight-control-design-slide.pptx")

# 16:9 widescreen
W = Inches(13.333)
H = Inches(7.5)

BG = RGBColor(0x0F, 0x14, 0x19)
PANEL = RGBColor(0x16, 0x1B, 0x22)
BORDER = RGBColor(0x30, 0x36, 0x3D)
ACCENT = RGBColor(0x58, 0xA6, 0xFF)
GREEN = RGBColor(0x3F, 0xB9, 0x50)
TITLE = RGBColor(0xF0, 0xF3, 0xF6)
BODY = RGBColor(0xC9, 0xD1, 0xD9)
MUTED = RGBColor(0x8B, 0x94, 0x9E)
TRACK = RGBColor(0x21, 0x26, 0x2D)

WORKFLOW = [
    "Capture mission requirements, safety envelopes, and certification constraints",
    "Define and model candidate control designs (baseline, AI-augmented, hybrid)",
    "Run high-fidelity simulation and scenario-based benchmarking",
    "Score trade-offs (performance, safety, operability, maintainability)",
    "Consolidate evidence and select the recommended design",
]

OUTCOME = "Accurate analysis of the best design."
TIME_VALUE = "4–6 weeks"
TIME_NOTE = (
    "Assumes simulation assets, scenario suite, and stakeholder review "
    "slots available from Week 1"
)


def set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    box.text_frame.word_wrap = True
    return box


def set_run(run, *, size=18, bold=False, color=BODY, font_name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name


def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG

    margin_l = Inches(0.65)
    margin_r = Inches(0.65)
    content_w = W - margin_l - margin_r

    # Header
    title_box = add_textbox(slide, margin_l, Inches(0.45), content_w, Inches(0.9))
    tf = title_box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = "AI Flight Control — Design Selection"
    set_run(r, size=32, bold=True, color=TITLE)

    sub = tf.add_paragraph()
    sub.space_before = Pt(6)
    r2 = sub.add_run()
    r2.text = "Structured path from requirements to a defensible architecture choice"
    set_run(r2, size=14, color=MUTED)

    col_top = Inches(1.55)
    col_h = Inches(4.35)
    gap = Inches(0.35)
    col_w = (content_w - gap) / 2
    col_left = margin_l
    col_right = margin_l + col_w + gap

    # Section labels
    for x, label in ((col_left, "WORKFLOW"), (col_right, "EXPECTED OUTCOME")):
        lb = add_textbox(slide, x, col_top, col_w, Inches(0.35))
        p = lb.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = label
        set_run(run, size=10, bold=True, color=ACCENT)

    # Workflow list
    wf_top = col_top + Inches(0.42)
    wf_box = add_textbox(slide, col_left, wf_top, col_w, col_h - Inches(0.42))
    tf = wf_box.text_frame
    for i, step in enumerate(WORKFLOW, start=1):
        para = tf.paragraphs[0] if i == 1 else tf.add_paragraph()
        para.space_after = Pt(14)
        para.line_spacing = 1.15
        num = para.add_run()
        num.text = f"{i}.  "
        set_run(num, size=15, bold=True, color=ACCENT)
        body = para.add_run()
        body.text = step
        set_run(body, size=15, color=BODY)

    # Outcome panel
    panel_top = wf_top
    panel = slide.shapes.add_shape(
        1, col_right, panel_top, col_w, Inches(2.35)
    )  # MSO_SHAPE.RECTANGLE = 1
    set_fill(panel, PANEL)
    panel.line.color.rgb = BORDER

    accent = slide.shapes.add_shape(1, col_right, panel_top, Inches(0.06), Inches(2.35))
    set_fill(accent, GREEN)
    accent.line.fill.background()

    out_box = add_textbox(
        slide, col_right + Inches(0.25), panel_top + Inches(0.55), col_w - Inches(0.4), Inches(1.4)
    )
    tf = out_box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = OUTCOME
    set_run(run, size=22, bold=False, color=TITLE)

    # Footer bar
    foot_h = Inches(1.35)
    foot_top = H - foot_h
    foot = slide.shapes.add_shape(1, 0, foot_top, W, foot_h)
    set_fill(foot, PANEL)
    foot.line.color.rgb = BORDER

    time_lbl = add_textbox(slide, margin_l, foot_top + Inches(0.18), Inches(3), Inches(0.25))
    p = time_lbl.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "TIME TO OUTCOME"
    set_run(r, size=9, bold=True, color=MUTED)

    note = add_textbox(
        slide, Inches(4.2), foot_top + Inches(0.15), W - Inches(4.85), Inches(0.55)
    )
    tf = note.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = TIME_NOTE
    set_run(r, size=10, color=MUTED)

    bar_left = margin_l
    bar_w = content_w - Inches(2.35)
    bar_h = Inches(0.16)
    bar_top = foot_top + Inches(0.6)
    arrow_h = Inches(0.26)
    arrow_top = bar_top - (arrow_h - bar_h) / 2

    track = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bar_left, bar_top, bar_w, bar_h)
    set_fill(track, TRACK)
    track.line.color.rgb = BORDER

    progress = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        bar_left,
        arrow_top,
        bar_w - Inches(0.04),
        arrow_h,
    )
    set_fill(progress, ACCENT)
    progress.line.fill.background()
    progress.adjustments[0] = 0.18
    progress.adjustments[1] = 0.55

    start_lbl = add_textbox(slide, bar_left, bar_top + Inches(0.22), Inches(0.8), Inches(0.2))
    p = start_lbl.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "Start"
    set_run(r, size=8, color=MUTED)

    end_lbl = add_textbox(
        slide, bar_left + bar_w - Inches(0.95), bar_top + Inches(0.22), Inches(1), Inches(0.2)
    )
    p = end_lbl.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = "Outcome"
    set_run(r, size=8, color=MUTED)

    time_val = add_textbox(
        slide, bar_left + bar_w + Inches(0.15), arrow_top - Inches(0.06), Inches(2.2), Inches(0.5)
    )
    p = time_val.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = TIME_VALUE
    set_run(r, size=24, bold=True, color=ACCENT)

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
