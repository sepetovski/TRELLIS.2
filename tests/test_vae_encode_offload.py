"""VAE stage offload must stream encoder levels onto the feature device.

The shape encoder used to ignore `low_vram`. Stage offload parked every
resolution ModuleList on CPU, then the first LayerNorm crashed because the
voxel features were already on CUDA.
"""
import unittest

import torch
import torch.nn as nn

from trellis2.models.sc_vaes import sparse_unet_vae as vae
from trellis2.modules import sparse as sp
from trellis2.utils import offload


class _StubRes(nn.Module):
    def __init__(self, channels, out_channels=None, use_checkpoint=False, **kwargs):
        super().__init__()
        self.lin = nn.Linear(channels, out_channels or channels)
        self.require_move = False
        self.ready = False
        self.ran_on = []

    def to(self, *args, **kwargs):
        self.ready = True
        return super().to(*args, **kwargs)

    def cpu(self):
        out = super().cpu()
        self.ready = False
        return out

    def forward(self, x):
        if self.require_move and not self.ready:
            raise RuntimeError("encoder block ran before its weights moved onto the feature device")
        if self.lin.weight.device != x.feats.device:
            raise RuntimeError(
                f"weight on {self.lin.weight.device}, features on {x.feats.device}"
            )
        self.ran_on.append(str(x.feats.device))
        return x.replace(self.lin(x.feats))


class _StubDown(_StubRes):
    """Channel change only. Real downsample is a sparse conv we cannot build on CPU."""


def _tiny_encoder():
    vae._StubRes = _StubRes
    vae._StubDown = _StubDown
    try:
        encoder = vae.SparseUnetVaeEncoder(
            in_channels=3,
            model_channels=[4, 8],
            latent_channels=2,
            num_blocks=[1, 1],
            block_type=["_StubRes", "_StubRes"],
            down_block_type=["_StubDown"],
            block_args=[{}, {}],
        )
    finally:
        del vae._StubRes
        del vae._StubDown
    return encoder


def _voxels(n=4, channels=3):
    feats = torch.arange(n * channels, dtype=torch.float32).reshape(n, channels)
    coords = torch.zeros((n, 4), dtype=torch.int32)
    coords[:, 1] = torch.arange(n)
    return sp.SparseTensor(feats, coords)


class VaeEncodeOffloadTests(unittest.TestCase):
    def test_encoder_declares_low_vram_so_stage_offload_can_set_it(self):
        encoder = _tiny_encoder()
        self.assertFalse(encoder.low_vram)
        offload.stage_model(encoder, "cpu", on_gpu=True, block_offload=True)
        self.assertTrue(encoder.low_vram)
        self.assertTrue(offload.is_nested_block_model(encoder))

    def test_low_vram_moves_each_block_before_it_runs(self):
        encoder = _tiny_encoder()
        offload.stage_model(encoder, "cpu", on_gpu=True, block_offload=True)
        moves = []

        def wrap(module, name):
            orig_to = module.to
            orig_cpu = module.cpu

            def to(*args, **kwargs):
                dest = args[0] if args else kwargs.get("device", "cpu")
                moves.append((name, "to", str(dest)))
                return orig_to(*args, **kwargs)

            def cpu(*args, **kwargs):
                moves.append((name, "cpu"))
                return orig_cpu(*args, **kwargs)

            module.to = to
            module.cpu = cpu

        for res in encoder.blocks:
            for block in res:
                block.require_move = True
                block.ready = False

        wrap(encoder.input_layer, "input")
        wrap(encoder.to_latent, "latent")
        block_names = []
        for i, res in enumerate(encoder.blocks):
            for j, block in enumerate(res):
                name = f"b{i}.{j}"
                block_names.append(name)
                wrap(block, name)

        latent = encoder(_voxels())
        self.assertEqual(tuple(latent.feats.shape), (4, 2))
        self.assertEqual(moves[0][0:2], ("input", "to"))
        self.assertEqual(moves[1], ("input", "cpu"))
        cursor = 2
        for name in block_names:
            self.assertEqual(moves[cursor], (name, "to", "cpu"))
            self.assertEqual(moves[cursor + 1], (name, "cpu"))
            cursor += 2
        self.assertEqual(moves[cursor], ("latent", "to", "cpu"))
        self.assertEqual(moves[cursor + 1], ("latent", "cpu"))
        self.assertEqual(cursor + 2, len(moves))
        for res in encoder.blocks:
            for block in res:
                self.assertEqual(block.ran_on, ["cpu"])
                self.assertFalse(block.ready)
                self.assertEqual(block.lin.weight.device.type, "cpu")

    def test_int64_coords_are_cast_before_the_blocks(self):
        encoder = _tiny_encoder()
        x = _voxels()
        long_coords = x.coords.to(dtype=torch.int64)
        x = sp.SparseTensor(feats=x.feats, coords=long_coords)
        seen = []
        for res in encoder.blocks:
            for block in res:
                orig = block.forward

                def wrapped(h, orig=orig):
                    seen.append(h.coords.dtype)
                    return orig(h)

                block.forward = wrapped
        latent = encoder(x)
        self.assertTrue(seen)
        self.assertTrue(all(dtype == torch.int32 for dtype in seen))
        self.assertEqual(latent.coords.dtype, torch.int32)
        self.assertEqual(latent.coords.tolist(), long_coords.to(dtype=torch.int32).tolist())

    def test_low_vram_matches_resident_encode(self):
        torch.manual_seed(0)
        resident = _tiny_encoder()
        streamed = _tiny_encoder()
        streamed.load_state_dict(resident.state_dict())
        streamed.low_vram = True
        x = _voxels()
        a = resident(x, sample_posterior=False)
        b = streamed(x, sample_posterior=False)
        self.assertTrue(torch.equal(a.feats, b.feats))
        self.assertTrue(torch.equal(a.coords, b.coords))


if __name__ == "__main__":
    unittest.main()
