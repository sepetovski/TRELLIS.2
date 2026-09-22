import os
import unittest

import torch
import torch.nn as nn

from trellis2.utils import offload


class TinyBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


class TinyDiT(nn.Module):
    def __init__(self, dim=8, n_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(n_blocks)])
        self.out_layer = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.out_layer(h)


class OffloadTests(unittest.TestCase):
    def test_iter_flat_and_nested_blocks(self):
        flat = TinyDiT()
        self.assertEqual(len(list(offload.iter_offload_blocks(flat))), 3)

        nested = nn.Module()
        nested.blocks = nn.ModuleList([
            nn.ModuleList([TinyBlock(), TinyBlock()]),
            nn.ModuleList([TinyBlock()]),
        ])
        self.assertEqual(len(list(offload.iter_offload_blocks(nested))), 3)

    def test_module_nbytes_positive(self):
        model = TinyDiT()
        self.assertGreater(offload.module_nbytes(model), 0)

    def test_lazy_model_map_loads_on_access(self):
        loads = []

        def load_fn(name, spec):
            loads.append(name)
            return TinyDiT()

        store = offload.LazyModelMap(load_fn, {"a": "spec-a", "b": "spec-b"})
        self.assertIn("a", store)
        self.assertEqual(loads, [])
        model = store["a"]
        self.assertIsInstance(model, TinyDiT)
        self.assertEqual(loads, ["a"])
        self.assertIs(store["a"], model)
        self.assertEqual(loads, ["a"])
        store.pop("a")
        self.assertNotIn("a", store)

    def test_lazy_model_map_unload_keeps_spec(self):
        loads = []

        def load_fn(name, spec):
            loads.append(name)
            return TinyDiT()

        store = offload.LazyModelMap(load_fn, {"dec": "spec-dec"})
        first = store["dec"]
        self.assertEqual(loads, ["dec"])
        unloaded = offload.unload_models(store, ["dec"])
        self.assertEqual(unloaded, ["dec"])
        self.assertIn("dec", store)
        second = store["dec"]
        self.assertIsNot(first, second)
        self.assertEqual(loads, ["dec", "dec"])

    def test_drop_models_removes_keys(self):
        store = {"a": TinyDiT(), "b": TinyDiT()}
        dropped = offload.drop_models(store, ["a", "missing"])
        self.assertEqual(dropped, ["a"])
        self.assertNotIn("a", store)
        self.assertIn("b", store)

    def test_nested_blocks_are_treated_as_vae(self):
        nested = nn.Module()
        nested.blocks = nn.ModuleList([nn.ModuleList([TinyBlock()]), nn.ModuleList([TinyBlock()])])
        self.assertTrue(offload.is_nested_block_model(nested))
        self.assertFalse(offload.should_dit_block_offload(nested))
        self.assertFalse(offload.is_nested_block_model(TinyDiT()))

    def test_recommend_lr_tokens_env(self):
        old = os.environ.get("TRELLIS_LR_TOKENS")
        os.environ["TRELLIS_LR_TOKENS"] = "1234"
        try:
            self.assertEqual(offload.recommend_lr_tokens(), 1234)
        finally:
            if old is None:
                os.environ.pop("TRELLIS_LR_TOKENS", None)
            else:
                os.environ["TRELLIS_LR_TOKENS"] = old

    def test_recommend_sequential_cfg_default_off(self):
        old = os.environ.get("TRELLIS_SEQ_CFG")
        os.environ.pop("TRELLIS_SEQ_CFG", None)
        try:
            offload._SEQ_CFG_LOGGED = False
            self.assertFalse(offload.recommend_sequential_cfg())
            sparse = type("S", (), {})()
            sparse.coords = torch.zeros(5840, 4)
            self.assertFalse(offload.recommend_sequential_cfg(sparse, gpu_gb=0.0))
            self.assertTrue(offload.recommend_sequential_cfg(sparse, gpu_gb=4.0))
            self.assertFalse(offload.recommend_sequential_cfg(torch.zeros(2, 8, 16, 16, 16), gpu_gb=4.0))
        finally:
            offload._SEQ_CFG_LOGGED = False
            if old is None:
                os.environ.pop("TRELLIS_SEQ_CFG", None)
            else:
                os.environ["TRELLIS_SEQ_CFG"] = old

    def test_recommend_sequential_cfg_env_off(self):
        old = os.environ.get("TRELLIS_SEQ_CFG")
        os.environ["TRELLIS_SEQ_CFG"] = "0"
        try:
            offload._SEQ_CFG_LOGGED = False
            sparse = type("S", (), {})()
            sparse.coords = torch.zeros(5840, 4)
            self.assertFalse(offload.recommend_sequential_cfg(sparse, gpu_gb=4.0))
        finally:
            offload._SEQ_CFG_LOGGED = False
            if old is None:
                os.environ.pop("TRELLIS_SEQ_CFG", None)
            else:
                os.environ["TRELLIS_SEQ_CFG"] = old

    def test_recommend_pin_memory_default_off(self):
        self.assertFalse(offload.recommend_pin_memory())

    def test_cap_sparse_coords_noop_when_under_budget(self):
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])
        out = offload.cap_sparse_coords(coords, 10)
        self.assertTrue(torch.equal(out, coords))

    def test_cap_sparse_coords_reduces_dense_volume(self):
        xs = torch.arange(8)
        grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1).reshape(-1, 3)
        coords = torch.cat([torch.zeros(grid.shape[0], 1, dtype=torch.long), grid], dim=1)
        self.assertEqual(coords.shape[0], 512)
        out = offload.cap_sparse_coords(coords, 64)
        self.assertLessEqual(out.shape[0], 64)
        self.assertGreater(out.shape[0], 8)
        self.assertEqual(out.shape[1], 4)

    def test_cap_sparse_coords_does_not_2x2x2_a_character(self):
        """A mage-scale occupancy slightly over 4096 must not collapse to ~900."""
        coords = []
        for z in range(32):
            for y in range(32):
                for x in range(32):
                    nx, ny, nz = (x - 16) / 11.0, (y - 16) / 13.0, (z - 16) / 10.0
                    if nx * nx + ny * ny + nz * nz <= 1.0:
                        coords.append([0, x, y, z])
                    if 24 <= x <= 25 and 4 <= y <= 28 and 15 <= z <= 16:
                        coords.append([0, x, y, z])
        coords = torch.tensor(coords).unique(dim=0)
        self.assertGreater(coords.shape[0], 4096)
        self.assertLess(coords.shape[0], 9000)
        out = offload.cap_sparse_coords(coords, 4096)
        self.assertLessEqual(out.shape[0], 4096)
        # Old 2×2×2 binning kept ~1/4–1/8 of voxels. Interior-first must keep
        # a character-shaped cloud, not a potato.
        self.assertGreaterEqual(out.shape[0], 3500)
        staff_in = ((coords[:, 1] >= 24) & (coords[:, 1] <= 25) & (coords[:, 2] >= 4) & (coords[:, 2] <= 28)).sum().item()
        staff_out = ((out[:, 1] >= 24) & (out[:, 1] <= 25) & (out[:, 2] >= 4) & (out[:, 2] <= 28)).sum().item()
        self.assertGreater(staff_in, 0)
        self.assertGreaterEqual(staff_out, int(0.7 * staff_in))

    def test_cap_sparse_coords_noop_under_8192_character(self):
        coords = torch.randint(0, 32, (4548, 4))
        coords[:, 0] = 0
        out = offload.cap_sparse_coords(coords, 8192)
        self.assertTrue(torch.equal(out, coords))

    def test_dilate_occupancy_coords(self):
        coords = torch.tensor([[0, 1, 2, 3]])
        out = offload.dilate_occupancy_coords(coords, factor=2)
        self.assertEqual(out.shape[0], 8)
        self.assertTrue(torch.equal(out[0], torch.tensor([0, 2, 4, 6])))
        self.assertTrue(torch.equal(out[-1], torch.tensor([0, 3, 5, 7])))

    def test_scale_coords_upsample(self):
        coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])
        out = offload.scale_coords_upsample(coords, 4)
        self.assertTrue(torch.equal(out[:, 0], coords[:, 0]))
        self.assertTrue(torch.equal(out[:, 1:], coords[:, 1:] * 16))

    def test_recommend_upsample_voxels_env(self):
        old = os.environ.get("TRELLIS_UPSAMPLE_VOXELS")
        try:
            os.environ["TRELLIS_UPSAMPLE_VOXELS"] = "0"
            self.assertEqual(offload.recommend_upsample_voxels(), 0)
            os.environ["TRELLIS_UPSAMPLE_VOXELS"] = "full"
            self.assertIsNone(offload.recommend_upsample_voxels())
        finally:
            if old is None:
                os.environ.pop("TRELLIS_UPSAMPLE_VOXELS", None)
            else:
                os.environ["TRELLIS_UPSAMPLE_VOXELS"] = old

    def test_models_for_pipeline_type(self):
        names_512 = offload.models_for_pipeline_type("512")
        self.assertIn("shape_slat_flow_model_512", names_512)
        self.assertNotIn("shape_slat_flow_model_1024", names_512)
        names_cascade = offload.models_for_pipeline_type("1024_cascade")
        self.assertIn("shape_slat_flow_model_1024", names_cascade)
        self.assertIn("shape_slat_flow_model_512", names_cascade)
        names_1536 = offload.models_for_pipeline_type("1536_cascade")
        self.assertEqual(names_1536, names_cascade)
        self.assertNotIn("tex_slat_flow_model_512", names_1536)

    def test_block_offload_cpu_forward_matches(self):
        torch.manual_seed(0)
        model = TinyDiT()
        model.eval()
        x = torch.randn(4, 8)
        with torch.no_grad():
            baseline = model(x).clone()

        n = offload.enable_module_block_offload(model, "cpu", pin_memory=False)
        self.assertEqual(n, 3)
        offload.move_non_block_modules(model, "cpu")
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.allclose(baseline, out, atol=1e-5))
        offload.disable_module_block_offload(model)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_block_offload_moves_blocks_off_gpu_after_forward(self):
        model = TinyDiT().cuda()
        x = torch.randn(4, 8, device="cuda")
        offload.enable_module_block_offload(model, "cuda", pin_memory=False)
        offload.move_non_block_modules(model, "cuda")
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.device.type, "cuda")
        for block in model.blocks:
            devices = {p.device.type for p in block.parameters()}
            self.assertEqual(devices, {"cpu"})
        offload.disable_module_block_offload(model)


if __name__ == "__main__":
    unittest.main()
