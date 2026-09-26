"""Isolate the subject before image-to-3D.

A photo of a whole scene (a house, a character in a room) makes the model
latch onto a small salient piece. An existing alpha channel is trusted only
when it actually isolates something. Otherwise the background is flooded in
from the image border and bright edge glare is dropped.
"""
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


def is_isolated_cutout(alpha: np.ndarray) -> bool:
    """True when alpha already separates a subject from a transparent background."""
    if alpha is None:
        return False
    opaque = alpha > 250
    transparent = alpha < 8
    if transparent.mean() < 0.04:
        return False
    if opaque.mean() > 0.92:
        return False
    return True


def _flood_background(rgb: np.ndarray, barrier: np.ndarray, tol: int = 34) -> np.ndarray:
    """Background pixels connected to the border, stopping at `barrier` or a color edge."""
    height, width = rgb.shape[:2]
    img = rgb.astype(np.int16)
    seen = np.zeros((height, width), dtype=bool)
    bg = np.zeros((height, width), dtype=bool)
    stack = []
    for x in range(width):
        stack.append((0, x))
        stack.append((height - 1, x))
    for y in range(1, height - 1):
        stack.append((y, 0))
        stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or barrier[y, x]:
            continue
        seen[y, x] = True
        bg[y, x] = True
        col = img[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or nx < 0 or ny >= height or nx >= width or seen[ny, nx] or barrier[ny, nx]:
                continue
            if int(np.abs(img[ny, nx] - col).sum()) <= tol * 3:
                stack.append((ny, nx))
    return bg


def _downscale(rgb: np.ndarray, alpha: np.ndarray, max_side: int = 384):
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return rgb, alpha, 1.0
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    rgb_s = np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
    alpha_s = np.array(Image.fromarray(alpha).resize(size, Image.Resampling.BILINEAR))
    return rgb_s, alpha_s, scale


def _upscale_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """`size` is (width, height)."""
    image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    return np.array(image.resize(size, Image.Resampling.BILINEAR)).astype(np.float32) / 255.0


def _close_and_erode(mask: np.ndarray, erode: bool) -> np.ndarray:
    image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    image = image.filter(ImageFilter.MaxFilter(3))
    image = image.filter(ImageFilter.MinFilter(3))
    if erode:
        image = image.filter(ImageFilter.MinFilter(3))
    alpha = np.array(image)
    alpha[alpha < 48] = 0
    return alpha


def _strip_bright_fringe(rgb: np.ndarray, alpha: np.ndarray):
    """Drop bright pixels that sit on the silhouette. Those are glare, not the subject."""
    opaque = alpha > 200
    if int(opaque.sum()) < 16:
        return rgb, alpha
    transparent = alpha < 8
    # One-pixel ring around the transparent background.
    ring = np.zeros_like(transparent)
    ring[1:] |= transparent[:-1]
    ring[:-1] |= transparent[1:]
    ring[:, 1:] |= transparent[:, :-1]
    ring[:, :-1] |= transparent[:, 1:]
    core = opaque & ~ring
    if int(core.sum()) < 16:
        core = opaque
    fg_mean = float(rgb[core].mean())
    bright = ring & (rgb.astype(np.float32).mean(axis=-1) > fg_mean + 45)
    if not bright.any():
        return rgb, alpha
    alpha = alpha.copy()
    alpha[bright] = 0
    return rgb, alpha


def refine_foreground(rgb: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return uint8 RGB, uint8 alpha, and a one-line description.

    A tiny mask (ornaments on a house) is replaced by the region that is not
    the border background. A normal subject keeps its mask, with background
    glare removed.
    """
    rgb = np.asarray(rgb)
    alpha = np.asarray(alpha)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
    if alpha.shape[:2] != rgb.shape[:2]:
        raise ValueError("alpha shape does not match rgb")
    height, width = rgb.shape[:2]
    rgb_s, alpha_s, _ = _downscale(rgb, alpha)
    alpha_f = alpha_s.astype(np.float32) / 255.0
    subject = alpha_f > 0.45
    subject_frac = float(subject.mean())
    barrier = alpha_f > 0.65
    if subject_frac > 0.90:
        barrier = np.zeros_like(barrier)
    bg = _flood_background(rgb_s, barrier)
    fg_flood = ~bg
    flood_frac = float(fg_flood.mean())
    flood_ok = 0.05 < flood_frac < 0.92

    if subject_frac < 0.12 and flood_ok and flood_frac > max(0.08, subject_frac * 1.8):
        mask_s = fg_flood.astype(np.float32)
        erode = True
        note = (
            f"Subject mask was only {subject_frac:.0%} of the photo, so the cutout "
            f"kept the whole object ({flood_frac:.0%}) instead of the small piece."
        )
    elif flood_ok and subject_frac >= 0.12:
        kept = subject & fg_flood
        if float(kept.mean()) >= subject_frac * 0.45:
            mask_s = np.where(kept, np.maximum(alpha_f, kept.astype(np.float32)), 0.0)
            erode = False
            note = (
                f"Cut the subject out ({float((mask_s > 0.45).mean()):.0%} of the photo) "
                "and removed background glare."
            )
        else:
            mask_s = alpha_f
            erode = False
            note = f"Cut the subject out ({subject_frac:.0%} of the photo)."
    else:
        mask_s = alpha_f
        erode = False
        note = f"Cut the subject out ({subject_frac:.0%} of the photo)."

    if rgb_s.shape[:2] != (height, width):
        mask = _upscale_mask(mask_s, (width, height))
    else:
        mask = mask_s
    alpha_out = _close_and_erode(mask, erode=erode)
    rgb_out, alpha_out = _strip_bright_fringe(np.array(rgb), alpha_out)
    return rgb_out, alpha_out, note


def composite_original(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """
    Crop and premultiply the way the first 512 run did.

    The bbox is the pixels with alpha above 0.8. The foreground is left as it
    is: no erosion, no flood fill. That is the path that reconstructed T.png.
    """
    ys, xs = np.where(alpha > int(0.8 * 255))
    if len(xs) == 0:
        return composite_on_black(rgb, alpha)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    size = int(max(x1 - x0, y1 - y0))
    box = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2),
    )
    crop = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").crop(box)
    arr = np.array(crop).astype(np.float32) / 255.0
    rgb_out = arr[:, :, :3] * arr[:, :, 3:4]
    return Image.fromarray((rgb_out * 255).astype(np.uint8))


def subject_fraction(alpha: np.ndarray) -> float:
    return float((np.asarray(alpha) > int(0.45 * 255)).mean())


def composite_on_black(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Crop to the subject and premultiply onto black, matching the model input."""
    ys, xs = np.where(alpha > int(0.5 * 255))
    if len(xs) == 0:
        ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return Image.fromarray(rgb)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    size = int(max(x1 - x0, y1 - y0))
    x0 = int(center[0] - size // 2)
    y0 = int(center[1] - size // 2)
    x1 = int(center[0] + size // 2)
    y1 = int(center[1] + size // 2)
    crop = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").crop((x0, y0, x1, y1))
    arr = np.array(crop).astype(np.float32) / 255.0
    rgb_out = arr[:, :, :3] * arr[:, :, 3:4]
    return Image.fromarray((rgb_out * 255).astype(np.uint8))
