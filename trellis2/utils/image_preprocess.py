"""Foreground isolation helpers for TRELLIS.2 image preprocessing.

Game sprites (mage, dragon) are already on black. Running BiRefNet on them
keeps the painted glow and can punch out a black hood/face, so the 32³
occupancy becomes a blob with a hole. Photos still go through rembg.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def skip_rembg_requested() -> bool:
    env = os.environ.get("TRELLIS_SKIP_REMBG", "").strip().lower()
    return env in ("1", "true", "yes")


def force_rembg_requested() -> bool:
    env = os.environ.get("TRELLIS_FORCE_REMBG", "").strip().lower()
    return env in ("1", "true", "yes")


def border_background_fraction(rgb: np.ndarray, band: Optional[int] = None, tol: int = 28) -> Tuple[float, str]:
    """Return (fraction, kind) of a uniform dark/light border."""
    h, w = rgb.shape[:2]
    if band is None:
        band = max(2, min(h, w) // 40)
    band = min(band, h // 4, w // 4)
    strips = (
        rgb[:band],
        rgb[-band:],
        rgb[:, :band],
        rgb[:, -band:],
    )
    border = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    dark = (border.max(axis=1) <= tol).mean()
    light = (border.min(axis=1) >= 255 - tol).mean()
    if dark >= light:
        return float(dark), "dark"
    return float(light), "light"


def looks_like_isolated_sprite(image: Image.Image, min_fraction: float = 0.75) -> bool:
    """True when the frame is already a cutout on black/white (skip rembg)."""
    rgb = np.array(image.convert("RGB"))
    frac, _kind = border_background_fraction(rgb)
    return frac >= min_fraction


def flood_fill_foreground_mask(rgb: np.ndarray, tol: int = 28) -> np.ndarray:
    """Keep pixels not 4-connected to a near-black (or near-white) border.

    Enclosed dark regions (a mage's hood/face) stay foreground. Surrounding
    canvas black is dropped. The painted glow still counts as foreground if
    it is brighter than `tol`; crop+composite then sits on true black.
    """
    h, w = rgb.shape[:2]
    lum = rgb.max(axis=2)
    # Decide whether the canvas is black or white from the border.
    frac, kind = border_background_fraction(rgb, tol=tol)
    if kind == "light" and frac >= 0.5:
        bg = lum >= 255 - tol
    else:
        bg = lum <= tol

    vis = np.zeros((h, w), dtype=bool)
    vis[0] = bg[0]
    vis[-1] = bg[-1]
    vis[:, 0] = bg[:, 0]
    vis[:, -1] = bg[:, -1]
    vis &= bg

    changed = True
    while changed:
        dilated = vis.copy()
        dilated[1:] |= vis[:-1]
        dilated[:-1] |= vis[1:]
        dilated[:, 1:] |= vis[:, :-1]
        dilated[:, :-1] |= vis[:, 1:]
        new = dilated & bg
        changed = bool(new.sum() > vis.sum())
        vis = new

    return ~vis


def harden_rgba_alpha(image: Image.Image, threshold: float = 0.5) -> Image.Image:
    """Binarize alpha so painted glows do not become a 3D occupancy shell."""
    arr = np.array(image.convert("RGBA"))
    cut = int(np.clip(threshold, 0.0, 1.0) * 255)
    arr[:, :, 3] = np.where(arr[:, :, 3] > cut, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def strip_desaturated_halo(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Drop gray/white aura around a saturated sprite (mage glow, JPEG fringe)."""
    maxc = rgb.max(axis=2).astype(np.float32)
    minc = rgb.min(axis=2).astype(np.float32)
    sat = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1.0), 0.0)
    glow = (sat < 0.22) & (maxc > 48)
    return fg & ~glow


def rgb_to_rgba_cutout(image: Image.Image, tol: int = 28) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    fg = flood_fill_foreground_mask(rgb, tol=tol)
    fg = strip_desaturated_halo(rgb, fg)
    alpha = fg.astype(np.uint8) * 255
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def maybe_cutout_without_rembg(image: Image.Image) -> Optional[Image.Image]:
    """Return an RGBA cutout when rembg should be skipped, else None."""
    if force_rembg_requested():
        return None
    if skip_rembg_requested() or looks_like_isolated_sprite(image):
        return rgb_to_rgba_cutout(image)
    return None
