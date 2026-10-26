#!/usr/bin/env python3
"""Blur background only; keep subjects, props, and labels sharp."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/catalog/baking-source"
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT = ROOT / "docs/catalog/processed"

SESSION = new_session("isnet-general-use")
SESSION_BIREFNET = new_session("birefnet-general")

# Light background blur only (foreground stays 100% original pixels).
BLUR_RADIUS = 10


@dataclass
class ImageConfig:
    crop: tuple[float, float, float, float] | None = None
    merge_left_crop: float | None = None


CONFIG: dict[str, ImageConfig] = {
    "01a17792-6139-7bfe-90f7-2feb70f463a1": ImageConfig(merge_left_crop=0.46),
}


def source_paths() -> list[Path]:
    if SRC.exists():
        files = sorted(SRC.glob("01a17792-*.jpg"))
        if files:
            return files
    return sorted(ASSETS.glob("01a17792-*.jpg"))


def crop_image(img: Image.Image, rect: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = img.size
    x0, y0, x1, y1 = rect
    box = (int(w * x0), int(h * y0), int(w * x1), int(h * y1))
    return img.crop(box), box


def drop_small_islands(alpha: np.ndarray, min_fraction: float = 0.003) -> np.ndarray:
    mask = alpha > 128
    labeled, count = ndimage.label(mask)
    if count <= 1:
        return alpha
    min_pixels = alpha.size * min_fraction
    keep = np.zeros_like(mask)
    for i in range(1, count + 1):
        region = labeled == i
        if region.sum() >= min_pixels:
            keep |= region
    return np.where(keep, np.maximum(alpha, 220), 0).astype(np.uint8)


def refine_alpha(alpha: np.ndarray) -> np.ndarray:
    mask = alpha > 110
    mask = ndimage.binary_dilation(mask, iterations=4)
    mask = ndimage.binary_closing(mask, iterations=2)
    out = (mask.astype(np.uint8) * 255)
    # Minimal edge feather so the subject/table stay sharp.
    return np.array(Image.fromarray(out, mode="L").filter(ImageFilter.GaussianBlur(radius=0.45)))


def harden_foreground_alpha(alpha: np.ndarray) -> np.ndarray:
    """Full-strength original on subjects; blur only where mask is clearly background."""
    a = alpha.astype(np.float32) / 255.0
    a = np.clip((a - 0.22) / 0.38, 0.0, 1.0)
    a = np.where(a > 0.72, 1.0, a)
    return (a * 255).astype(np.uint8)


def rembg_alpha(img: Image.Image, session=SESSION) -> np.ndarray:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    cut = remove(buf.getvalue(), session=session, post_process_mask=False)
    rgba = Image.open(io.BytesIO(cut)).convert("RGBA")
    return np.array(rgba.split()[3])


def merge_left_alpha(base: np.ndarray, left: np.ndarray) -> np.ndarray:
    h, w = base.shape
    left = np.array(Image.fromarray(left, mode="L").resize((w, h), Image.Resampling.LANCZOS))
    return np.where(left > 140, np.maximum(base, left), base)


def build_foreground_mask(original: Image.Image, cfg: ImageConfig) -> np.ndarray:
    alpha = rembg_alpha(original)
    if cfg.merge_left_crop:
        lw = int(original.size[0] * cfg.merge_left_crop)
        left_crop = original.crop((0, 0, lw, original.size[1]))
        left_alpha = rembg_alpha(left_crop, session=SESSION_BIREFNET)
        alpha[:, :lw] = merge_left_alpha(alpha[:, :lw], left_alpha)
    if cfg.crop:
        work, (x0, y0, x1, y1) = crop_image(original, cfg.crop)
        cropped_alpha = rembg_alpha(work)
        if cfg.merge_left_crop:
            lw = int(work.size[0] * cfg.merge_left_crop)
            left_crop = work.crop((0, 0, lw, work.size[1]))
            left_alpha = rembg_alpha(left_crop, session=SESSION_BIREFNET)
            cropped_alpha = merge_left_alpha(cropped_alpha, left_alpha)
        full_h, full_w = original.size[1], original.size[0]
        full_alpha = np.zeros((full_h, full_w), dtype=np.uint8)
        cropped_alpha = drop_small_islands(cropped_alpha)
        cropped_alpha = refine_alpha(cropped_alpha)
        full_alpha[y0:y1, x0:x1] = harden_foreground_alpha(cropped_alpha)
        return full_alpha
    alpha = drop_small_islands(alpha)
    alpha = refine_alpha(alpha)
    return harden_foreground_alpha(alpha)


def blur_background(original: Image.Image, alpha: np.ndarray, radius: float) -> Image.Image:
    blurred = original.filter(ImageFilter.GaussianBlur(radius=radius))
    fg = np.array(original, dtype=np.float32)
    bg = np.array(blurred, dtype=np.float32)
    a = harden_foreground_alpha(alpha).astype(np.float32) / 255.0
    a = a[..., np.newaxis]
    out = (fg * a + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def process(path: Path) -> None:
    cfg = CONFIG.get(path.stem, ImageConfig())
    original = Image.open(path).convert("RGB")
    alpha = build_foreground_mask(original, cfg)
    blur_radius = BLUR_RADIUS
    result = blur_background(original, alpha, radius=blur_radius)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    result.save(OUT / f"{stem}-blurred.jpg", quality=93, optimize=True)
    # Optional soft mask for edits
    Image.fromarray(alpha, mode="L").save(OUT / f"{stem}-mask.png")
    print("OK", path.name, f"blur={blur_radius:.0f}")


def main() -> None:
    paths = source_paths()
    if not paths:
        raise SystemExit("No source images found.")
    for p in paths:
        process(p)


if __name__ == "__main__":
    main()
