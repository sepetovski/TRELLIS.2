import unittest

import torch

from trellis2.utils.refine_ops import (
    faces_to_latent_mask,
    gather_feats_by_coord,
    row_index_after_cap,
    subdivision_guides,
    voxel_flags_to_latent_mask,
)


class RefineOpsTests(unittest.TestCase):
    def test_gather_matches_and_fills(self):
        src = torch.tensor([[0, 0, 0], [2, 0, 0], [2, 0, 0]])
        feats = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]])
        dst = torch.tensor([[2, 0, 0], [1, 1, 1], [0, 0, 0]])
        out, found = gather_feats_by_coord(src, feats, dst, fill=-1.0)
        self.assertTrue(bool(found[0]) and bool(found[2]) and not bool(found[1]))
        self.assertEqual(out[0].tolist(), [3.0, 4.0])  # first duplicate wins
        self.assertEqual(out[1].tolist(), [-1.0, -1.0])
        self.assertEqual(out[2].tolist(), [1.0, 2.0])

    def test_latent_mask_uses_16_cell(self):
        voxels = torch.tensor([[0, 0, 0], [15, 0, 0], [16, 0, 0]])
        flag = torch.tensor([False, True, False])
        latent = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        mask = voxel_flags_to_latent_mask(voxels, flag, latent, downsample=16)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_face_mask_hits_the_cell_under_the_triangle(self):
        # A triangle around the origin, which is voxel ~256 on a 512 grid, cell 16.
        vertices = torch.tensor([[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0]])
        faces = torch.tensor([[0, 1, 2]])
        latent = torch.tensor([[0, 16, 16, 16], [0, 0, 0, 0]])
        mask = faces_to_latent_mask(vertices, faces, [0], latent, resolution=512, downsample=16, dilate=0)
        self.assertEqual(mask.tolist(), [True, False])
        with self.assertRaises(ValueError):
            faces_to_latent_mask(vertices, faces, [3], latent, resolution=512)

    def test_cap_keeps_a_subset_of_real_rows(self):
        coords = torch.stack([
            torch.zeros(100, dtype=torch.long),
            torch.arange(100),
            torch.zeros(100, dtype=torch.long),
            torch.zeros(100, dtype=torch.long),
        ], dim=1)
        index = row_index_after_cap(coords, 10)
        self.assertLessEqual(int(index.shape[0]), 10)
        self.assertEqual(int(index.unique().shape[0]), int(index.shape[0]))
        self.assertTrue(bool((index >= 0).all() and (index < 100).all()))
        same = row_index_after_cap(coords[:4], 10)
        self.assertEqual(same.tolist(), [0, 1, 2, 3])

    def test_subdivision_octants_follow_spatial_to_channel(self):
        # Two adjacent voxels in one 2-wide parent, then their own children.
        voxels = torch.tensor([[0, 0, 0], [1, 0, 0]])
        latent = torch.tensor([[0, 0, 0, 0]])
        guides = subdivision_guides(latent, voxels, resolution=4, levels=2)
        self.assertEqual(len(guides), 2)
        # Level 0 parent size 4. Both voxels sit in parent 0, child size 2,
        # so both are octant 0.
        self.assertEqual(int(guides[0][1].sum().item()), 1.0)
        self.assertEqual(guides[0][1][0].tolist(), [1, 0, 0, 0, 0, 0, 0, 0])
        # Level 1 splits them into octants 0 and 1.
        self.assertEqual(guides[1][0].shape[0], 1)
        self.assertEqual(guides[1][1][0, 0].item(), 1.0)
        self.assertEqual(guides[1][1][0, 1].item(), 1.0)
        self.assertEqual(int(guides[1][1].sum().item()), 2.0)


if __name__ == "__main__":
    unittest.main()
