import os
import unittest
from unittest import mock


class SubdivLimitTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from trellis2.utils import offload
        except Exception as exc:
            self.skipTest(f"torch missing: {exc}")
        self.torch = torch
        self.offload = offload

    def test_under_budget_is_unchanged(self):
        feats = self.torch.tensor([[0.2, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        out = self.offload.limit_subdiv_feats(feats, 4)
        self.assertTrue(self.torch.equal(out, feats))

    def test_keeps_best_child_then_next_logits(self):
        feats = self.torch.tensor(
            [
                [1.0, 0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [0.2, -3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [0.9, 0.8, 0.1, -1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        )
        out = self.offload.limit_subdiv_feats(feats, 4)
        self.assertEqual(int((out > 0).sum()), 4)
        self.assertGreater(out[0, 0].item(), 0)
        self.assertGreater(out[1, 0].item(), 0)
        self.assertGreater(out[2, 0].item(), 0)
        self.assertGreater(out[2, 1].item(), 0)
        self.assertEqual(out[0, 1].item(), 0)
        self.assertEqual(out[1, 1].item(), -3.0)

    def test_too_many_parents_keeps_strongest(self):
        feats = self.torch.tensor(
            [
                [0.1, 0.05, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [5.0, 4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [0.2, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [4.0, 3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        )
        out = self.offload.limit_subdiv_feats(feats, 2)
        self.assertEqual(int((out > 0).sum()), 2)
        self.assertEqual(int((out[0] > 0).sum()), 0)
        self.assertEqual(int((out[2] > 0).sum()), 0)
        self.assertEqual(int((out[1] > 0).sum()), 1)
        self.assertEqual(int((out[3] > 0).sum()), 1)
        self.assertGreater(out[1, 0].item(), 0)
        self.assertGreater(out[3, 0].item(), 0)

    def test_zero_budget_drops_every_child(self):
        feats = self.torch.tensor([[1.0, 2.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        out = self.offload.limit_subdiv_feats(feats, 0)
        self.assertEqual(int((out > 0).sum()), 0)
        self.assertEqual(out[0, 2].item(), -4.0)

    def test_recommend_tex_decode_voxels(self):
        old = os.environ.get("TRELLIS_TEX_VOXELS")
        try:
            os.environ["TRELLIS_TEX_VOXELS"] = "full"
            self.assertIsNone(self.offload.recommend_tex_decode_voxels())
            os.environ["TRELLIS_TEX_VOXELS"] = "12345"
            self.assertEqual(self.offload.recommend_tex_decode_voxels(), 12345)
            os.environ.pop("TRELLIS_TEX_VOXELS")
            with mock.patch.object(self.offload, "gpu_total_memory_gb", return_value=4.0):
                self.assertEqual(
                    self.offload.recommend_tex_decode_voxels(),
                    self.offload.TEX_DECODE_VOXEL_CAP,
                )
            with mock.patch.object(self.offload, "gpu_total_memory_gb", return_value=24.0):
                self.assertIsNone(self.offload.recommend_tex_decode_voxels())
        finally:
            if old is None:
                os.environ.pop("TRELLIS_TEX_VOXELS", None)
            else:
                os.environ["TRELLIS_TEX_VOXELS"] = old


if __name__ == "__main__":
    unittest.main()
