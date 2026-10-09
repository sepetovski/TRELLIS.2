"""CPU checks for the 4 GB quality path: chunked MLP, conv halos, SDPA."""
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules.sparse.attention.full_attn import _sdpa_varlen
from trellis2.modules.sparse.basic import SparseTensor
from trellis2.modules.sparse.conv.tile import (
    build_coord_lookup,
    core_with_halo,
    kernel_offsets,
    lookup_indices,
    pack_coord_keys,
)
from trellis2.modules.sparse.transformer.blocks import SparseFeedForwardNet
from trellis2.utils import offload
from trellis2.utils.chunked import chunked_token_apply, mlp_chunk_size


def _neighbor_sum(coords, feats, kernel_size=(3, 3, 3), dilation=(1, 1, 1)):
    lookup = build_coord_lookup(coords)
    out = feats.clone()
    for ox, oy, oz in kernel_offsets(kernel_size, dilation):
        shifted = coords.clone()
        shifted[:, 1] = shifted[:, 1] + ox
        shifted[:, 2] = shifted[:, 2] + oy
        shifted[:, 3] = shifted[:, 3] + oz
        idx, hit = lookup_indices(pack_coord_keys(shifted), *lookup)
        extra = torch.zeros_like(feats)
        extra[hit] = feats[idx[hit]]
        out = out + extra
    return out


class ChunkedMlpTests(unittest.TestCase):
    def test_chunked_token_apply_matches_full(self):
        torch.manual_seed(0)
        layer = nn.Sequential(nn.Linear(8, 32), nn.GELU(approximate="tanh"), nn.Linear(32, 8))
        layer.eval()
        x = torch.randn(13, 8)
        with torch.no_grad():
            full = layer(x)
            chunked = chunked_token_apply(x, layer, 5)
        self.assertTrue(torch.allclose(full, chunked, atol=1e-5))

    def test_sparse_ffn_chunk_matches(self):
        torch.manual_seed(1)
        old = os.environ.get("TRELLIS_MLP_CHUNK")
        os.environ["TRELLIS_MLP_CHUNK"] = "4"
        try:
            net = SparseFeedForwardNet(channels=8, mlp_ratio=2.0)
            net.eval()
            coords = torch.stack([
                torch.zeros(11, dtype=torch.int32),
                torch.arange(11, dtype=torch.int32),
                torch.arange(11, dtype=torch.int32),
                torch.zeros(11, dtype=torch.int32),
            ], dim=1)
            feats = torch.randn(11, 8)
            x = SparseTensor(feats, coords)
            with torch.no_grad():
                chunked = net(x).feats.clone()
            os.environ["TRELLIS_MLP_CHUNK"] = "0"
            with torch.no_grad():
                full = net(x).feats.clone()
            self.assertTrue(torch.allclose(full, chunked, atol=1e-5))
            self.assertEqual(mlp_chunk_size(11), None)
        finally:
            if old is None:
                os.environ.pop("TRELLIS_MLP_CHUNK", None)
            else:
                os.environ["TRELLIS_MLP_CHUNK"] = old


class ConvTileTests(unittest.TestCase):
    def test_kernel_offsets_3x3x3(self):
        offsets = kernel_offsets((3, 3, 3), (1, 1, 1))
        self.assertEqual(len(offsets), 26)
        self.assertNotIn((0, 0, 0), offsets)

    def test_tiled_neighbor_sum_matches_full(self):
        torch.manual_seed(2)
        xs = torch.arange(6)
        grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1).reshape(-1, 3)
        keep = torch.rand(grid.shape[0]) > 0.45
        grid = grid[keep]
        coords = torch.cat([torch.zeros(grid.shape[0], 1, dtype=torch.long), grid.long()], dim=1)
        feats = torch.randn(coords.shape[0], 3)
        full = _neighbor_sum(coords, feats)
        lookup = build_coord_lookup(coords)
        parts = []
        chunk = 7
        for start in range(0, coords.shape[0], chunk):
            core = torch.arange(start, min(start + chunk, coords.shape[0]))
            selected = core_with_halo(coords, core, lookup=lookup)
            self.assertTrue(torch.equal(selected[: core.shape[0]], core))
            sub = _neighbor_sum(coords[selected], feats[selected])
            parts.append(sub[: core.shape[0]])
        tiled = torch.cat(parts, dim=0)
        self.assertTrue(torch.allclose(full, tiled, atol=1e-5))

    def test_negative_neighbor_does_not_alias_zero(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]])
        lookup = build_coord_lookup(coords)
        # The -1 neighbor of voxel 0 must not resolve to voxel 0.
        shifted = coords[:1].clone()
        shifted[:, 1] -= 1
        _idx, hit = lookup_indices(pack_coord_keys(shifted), *lookup)
        self.assertFalse(bool(hit.any()))


class SdpaTests(unittest.TestCase):
    def test_dense_sdpa_chunks_long_queries(self):
        from trellis2.modules.attention.full_attn import scaled_dot_product_attention
        torch.manual_seed(5)
        q = torch.randn(1, 600, 2, 8)
        k = torch.randn(1, 600, 2, 8)
        v = torch.randn(1, 600, 2, 8)
        out = scaled_dot_product_attention(q, k, v)
        ref = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
        ).permute(0, 2, 1, 3)
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_sdpa_varlen_matches_one_sequence(self):
        torch.manual_seed(3)
        q = torch.randn(6, 2, 8)
        k = torch.randn(6, 2, 8)
        v = torch.randn(6, 2, 8)
        out = _sdpa_varlen(q, k, v, [6], [6])
        ref = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
        ).squeeze(0).permute(1, 0, 2)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_sdpa_varlen_two_sequences_do_not_mix(self):
        torch.manual_seed(4)
        q = torch.randn(9, 2, 8)
        k = torch.randn(9, 2, 8)
        v = torch.randn(9, 2, 8)
        out = _sdpa_varlen(q, k, v, [4, 5], [4, 5])
        ref0 = F.scaled_dot_product_attention(
            q[:4].permute(1, 0, 2).unsqueeze(0),
            k[:4].permute(1, 0, 2).unsqueeze(0),
            v[:4].permute(1, 0, 2).unsqueeze(0),
        ).squeeze(0).permute(1, 0, 2)
        ref1 = F.scaled_dot_product_attention(
            q[4:].permute(1, 0, 2).unsqueeze(0),
            k[4:].permute(1, 0, 2).unsqueeze(0),
            v[4:].permute(1, 0, 2).unsqueeze(0),
        ).squeeze(0).permute(1, 0, 2)
        self.assertTrue(torch.allclose(out[:4], ref0, atol=1e-5))
        self.assertTrue(torch.allclose(out[4:], ref1, atol=1e-5))


class BudgetTests(unittest.TestCase):
    def test_hr_tokens_and_conv_chunk_env(self):
        old_hr = os.environ.get("TRELLIS_MAX_TOKENS")
        old_chunk = os.environ.get("TRELLIS_CONV_CHUNK")
        old_algo = os.environ.get("TRELLIS_FLEX_ALGO")
        try:
            os.environ["TRELLIS_MAX_TOKENS"] = "1000"
            os.environ["TRELLIS_CONV_CHUNK"] = "0"
            os.environ["TRELLIS_FLEX_ALGO"] = "implicit_gemm"
            self.assertEqual(offload.recommend_hr_tokens(), 1000)
            self.assertIsNone(offload.recommend_conv_chunk())
            self.assertEqual(offload.recommend_flex_algo(), "implicit_gemm")
        finally:
            for key, old in (
                ("TRELLIS_MAX_TOKENS", old_hr),
                ("TRELLIS_CONV_CHUNK", old_chunk),
                ("TRELLIS_FLEX_ALGO", old_algo),
            ):
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
