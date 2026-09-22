import unittest

import numpy as np

from trellis2.utils.hdri import load_latlong_rgb, preview_settings, studio_latlong


class HdriTests(unittest.TestCase):
    def test_studio_is_a_color_image(self):
        image = studio_latlong(32, 64)
        self.assertEqual(image.shape, (32, 64, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreater(float(image.max()), 1.0)

    def test_missing_exr_does_not_return_empty(self):
        image = load_latlong_rgb("assets/hdri/does-not-exist.exr")
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[-1], 3)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

    def test_preview_is_smaller_on_4gb(self):
        self.assertEqual(preview_settings(4.0), (512, 60, 1))
        self.assertEqual(preview_settings(24.0), (1024, 120, 2))


if __name__ == "__main__":
    unittest.main()
