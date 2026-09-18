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

    def test_recommend_pin_memory_default_off(self):
        self.assertFalse(offload.recommend_pin_memory())

    def test_models_for_pipeline_type(self):
        names_512 = offload.models_for_pipeline_type("512")
        self.assertIn("shape_slat_flow_model_512", names_512)
        self.assertNotIn("shape_slat_flow_model_1024", names_512)
        names_cascade = offload.models_for_pipeline_type("1024_cascade")
        self.assertIn("shape_slat_flow_model_1024", names_cascade)
        self.assertIn("shape_slat_flow_model_512", names_cascade)

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
