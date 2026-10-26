#!/usr/bin/env python3
"""Remove room backgrounds; preserve kid, bear, props, and label text."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove
from scipy import ndimage

SRC = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT = Path(__file__).resolve().parents[1] / "docs/catalog/processed"
BACKDROP = (248, 247, 245)
SESSION = new_session("isnet-general-use")
SESSION_BIREFNET = new_session("birefnet-general")


@dataclass
class ImageConfig:
    crop: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1 normalized
    trim_bottom: float | None = None  # fraction of fg bbox bottom to cut (countertop)
    merge_left_crop: float | None = None  # extra rembg pass on left strip (bear + flour)


CONFIG: dict[str, ImageConfig] = {
    "01a17792-6139-72c6-85ca-b8d98f4bec5b": ImageConfig(crop=(0.0, 0.22, 1.0, 1.0)),
    "01a17792-6139-719d-a9d9-ab82d528b780": ImageConfig(trim_bottom=0.24),
    "01a17792-6139-7bfe-90f7-2feb70f463a1": ImageConfig(
        crop=(0.0, 0.16, 1.0, 1.0),
        merge_left_crop=0.46,
    ),
}


def crop_image(img: Image.Image, rect: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = rect
    return img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))


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
    mask = alpha > 120
    mask = ndimage.binary_dilation(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    out = (mask.astype(np.uint8) * 255)
    return np.array(Image.fromarray(out, mode="L").filter(ImageFilter.GaussianBlur(radius=0.5)))


def trim_bottom_counter(alpha: np.ndarray, trim_fraction: float) -> np.ndarray:
    ys = np.where(alpha > 128)[0]
    if len(ys) == 0:
        return alpha
    y_max = ys.max()
    y_min = ys.min()
    cut = int(y_max - (y_max - y_min) * trim_fraction)
    out = alpha.copy()
    out[cut + 1 :] = 0
    return out


def remove_thin_artifacts(alpha: np.ndarray) -> np.ndarray:
    """Drop very thin islands (e.g. chair metal rods) while keeping main props."""
    mask = alpha > 128
    labeled, count = ndimage.label(mask)
    keep = np.zeros_like(mask)
    for i in range(1, count + 1):
        region = labeled == i
        area = region.sum()
        if area < alpha.size * 0.003:
            continue
        ys, xs = np.where(region)
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.min() + 1
        aspect = max(height, width) / max(1, min(height, width))
        if area < alpha.size * 0.01 and aspect > 6:
            continue
        keep |= region
    return np.where(keep, np.maximum(alpha, 220), 0).astype(np.uint8)


def add_soft_shadow(rgba: Image.Image) -> Image.Image:
    w, h = rgba.size
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    blob = rgba.split()[3].point(lambda p: 65 if p > 40 else 0).filter(ImageFilter.GaussianBlur(radius=12))
    shadow.putalpha(blob)
    canvas = Image.new("RGBA", (w, h), BACKDROP + (255,))
    canvas.alpha_composite(shadow, (0, 5))
    canvas.alpha_composite(rgba, (0, 0))
    return canvas


def rembg_rgba(img: Image.Image, session=SESSION) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    cut = remove(buf.getvalue(), session=session, post_process_mask=False)
    return Image.open(io.BytesIO(cut)).convert("RGBA")


def merge_left_rgba(base: Image.Image, left: Image.Image, left_width: int) -> Image.Image:
    base = base.copy()
    left = left.resize((left_width, base.height), Image.Resampling.LANCZOS)
    px_base = base.load()
    px_left = left.load()
    for y in range(base.height):
        for x in range(left_width):
            lr, lg, lb, la = px_left[x, y]
            _br, _bg, _bb, ba = px_base[x, y]
            if la > 140 and la >= ba:
                px_base[x, y] = (lr, lg, lb, la)
    return base


def process(path: Path) -> None:
    cfg = CONFIG.get(path.stem, ImageConfig())
    original = Image.open(path).convert("RGB")
    work = crop_image(original, cfg.crop) if cfg.crop else original

    rgba = rembg_rgba(work)
    if cfg.merge_left_crop:
        w, h = work.size
        lw = int(w * cfg.merge_left_crop)
        left_crop = work.crop((0, 0, lw, h))
        left_rgba = rembg_rgba(left_crop, session=SESSION_BIREFNET)
        rgba = merge_left_rgba(rgba, left_rgba, lw)

    alpha = np.array(rgba.split()[3])
    alpha = drop_small_islands(alpha)
    alpha = remove_thin_artifacts(alpha)
    alpha = refine_alpha(alpha)
    if cfg.trim_bottom:
        alpha = trim_bottom_counter(alpha, cfg.trim_bottom)
    rgba = Image.merge("RGBA", (*rgba.split()[:3], Image.fromarray(alpha)))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    rgba.save(OUT / f"{stem}-nobg.png")
    add_soft_shadow(rgba).convert("RGB").save(OUT / f"{stem}-nobg.jpg", quality=93, optimize=True)
    print("OK", path.name)


def main() -> None:
    for p in sorted(SRC.glob("01a17792-*.jpg")):
        process(p)


if __name__ == "__main__":
    main()
