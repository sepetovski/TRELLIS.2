import unittest


class PreviewLightTests(unittest.TestCase):
    def test_shade_without_nvdiffrec(self):
        try:
            import torch
            from trellis2.renderers.pbr_mesh_renderer import EnvMap
        except Exception as exc:
            self.skipTest(f"renderer deps missing: {exc}")
        image = torch.zeros(8, 16, 3)
        env = EnvMap(image)
        env._nvdiffrec_failed = True
        pos = torch.zeros(1, 4, 4, 3)
        normal = torch.zeros(1, 4, 4, 3)
        normal[..., 1] = 1
        kd = torch.ones(1, 4, 4, 3) * 0.5
        ks = torch.zeros(1, 4, 4, 3)
        ks[..., 1] = 0.4
        view = torch.zeros(4, 4, 3)
        view[..., 2] = 1
        shaded = env.shade(pos, normal, kd, ks, view, specular=True)[0]
        self.assertEqual(tuple(shaded.shape), (4, 4, 3))
        self.assertTrue(torch.isfinite(shaded).all())
        self.assertGreater(float(shaded.mean()), 0.05)


if __name__ == "__main__":
    unittest.main()
