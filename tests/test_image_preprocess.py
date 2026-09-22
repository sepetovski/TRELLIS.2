import os
import unittest

import numpy as np
from PIL import Image

from trellis2.utils.image_preprocess import (
    border_background_fraction,
    flood_fill_foreground_mask,
    looks_like_isolated_sprite,
    maybe_cutout_without_rembg,
    rgb_to_rgba_cutout,
)


def _sprite_with_black_face(size=64):
    """Blue character on black with an enclosed black 'hood'."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[12:52, 12:52] = (40, 40, 180)
    rgb[20:36, 24:40] = (0, 0, 0)  # enclosed black face
    return rgb


class ImagePreprocessTests(unittest.TestCase):
    def test_sprite_border_is_dark(self):
        rgb = _sprite_with_black_face()
        frac, kind = border_background_fraction(rgb)
        self.assertEqual(kind, "dark")
        self.assertGreaterEqual(frac, 0.75)
        img = Image.fromarray(rgb)
        self.assertTrue(looks_like_isolated_sprite(img))

    def test_flood_fill_keeps_enclosed_black_face(self):
        rgb = _sprite_with_black_face()
        fg = flood_fill_foreground_mask(rgb)
        # Face pixel should remain foreground (not connected to the border).
        self.assertTrue(fg[28, 32])
        # Corner canvas should be background.
        self.assertFalse(fg[0, 0])
        # Cloak pixel should be foreground.
        self.assertTrue(fg[14, 14])

    def test_photo_like_image_does_not_look_like_sprite(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(rgb)
        self.assertFalse(looks_like_isolated_sprite(img))

    def test_maybe_cutout_skips_rembg_for_sprite(self):
        img = Image.fromarray(_sprite_with_black_face())
        cut = maybe_cutout_without_rembg(img)
        self.assertIsNotNone(cut)
        alpha = np.array(cut)[:, :, 3]
        self.assertGreater(alpha[28, 32], 0)
        self.assertEqual(alpha[0, 0], 0)

    def test_force_rembg_disables_cutout(self):
        old = os.environ.get("TRELLIS_FORCE_REMBG")
        os.environ["TRELLIS_FORCE_REMBG"] = "1"
        try:
            img = Image.fromarray(_sprite_with_black_face())
            self.assertIsNone(maybe_cutout_without_rembg(img))
        finally:
            if old is None:
                os.environ.pop("TRELLIS_FORCE_REMBG", None)
            else:
                os.environ["TRELLIS_FORCE_REMBG"] = old

    def test_harden_alpha_drops_soft_glow(self):
        rgb = _sprite_with_black_face()
        alpha = np.full(rgb.shape[:2], 40, dtype=np.uint8)  # glow
        alpha[12:52, 12:52] = 255
        rgba = np.dstack([rgb, alpha])
        img = Image.fromarray(rgba, mode="RGBA")
        from trellis2.utils.image_preprocess import harden_rgba_alpha
        hard = np.array(harden_rgba_alpha(img, threshold=0.5))[:, :, 3]
        self.assertEqual(hard[0, 0], 0)
        self.assertEqual(hard[20, 20], 255)

    def test_strip_halo_keeps_character_not_gray_ring(self):
        from trellis2.utils.image_preprocess import strip_desaturated_halo
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[4:28, 4:28] = 180  # gray glow
        rgb[10:22, 10:22] = (40, 40, 200)  # blue sprite
        fg = np.ones((32, 32), dtype=bool)
        kept = strip_desaturated_halo(rgb, fg)
        self.assertTrue(kept[16, 16])
        self.assertFalse(kept[6, 6])


if __name__ == "__main__":
    unittest.main()
