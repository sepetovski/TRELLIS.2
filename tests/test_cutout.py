import unittest

import numpy as np

from trellis2.utils.cutout import composite_on_black, is_isolated_cutout, refine_foreground


def _house_with_ornament():
    """Blue sky, red building, and a tiny mask that only marks a window."""
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:, :] = (70, 140, 220)
    rgb[25:70, 15:65] = (170, 40, 40)
    alpha = np.zeros((80, 80), dtype=np.uint8)
    alpha[40:48, 30:38] = 255
    return rgb, alpha


class CutoutTests(unittest.TestCase):
    def test_full_frame_alpha_is_not_a_cutout(self):
        self.assertFalse(is_isolated_cutout(np.full((16, 16), 255, dtype=np.uint8)))

    def test_transparent_border_is_a_cutout(self):
        alpha = np.zeros((32, 32), dtype=np.uint8)
        alpha[8:24, 8:24] = 255
        self.assertTrue(is_isolated_cutout(alpha))

    def test_tiny_mask_keeps_the_building(self):
        rgb, alpha = _house_with_ornament()
        _rgb, out_alpha, note = refine_foreground(rgb, alpha)
        building = out_alpha[25:70, 15:65]
        sky = out_alpha[:10, :]
        self.assertGreater(float((building > 128).mean()), 0.7)
        self.assertLess(float((sky > 128).mean()), 0.1)
        self.assertIn("whole object", note)

    def test_bright_fringe_is_removed(self):
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb[10:30, 10:30] = (40, 40, 160)
        rgb[9, 10:30] = (250, 250, 250)
        alpha = np.zeros((40, 40), dtype=np.uint8)
        alpha[10:30, 10:30] = 255
        alpha[9, 10:30] = 180
        _rgb, out_alpha, _note = refine_foreground(rgb, alpha)
        self.assertEqual(int(out_alpha[9, 20]), 0)
        self.assertGreater(int(out_alpha[20, 20]), 200)

    def test_composite_drops_the_background(self):
        rgb, alpha = _house_with_ornament()
        _rgb, out_alpha, _note = refine_foreground(rgb, alpha)
        image = composite_on_black(rgb, out_alpha)
        arr = np.array(image)
        self.assertEqual(arr.ndim, 3)
        self.assertGreater(arr.shape[0], 8)
        self.assertGreater(float(arr.mean()), 5)


if __name__ == "__main__":
    unittest.main()
