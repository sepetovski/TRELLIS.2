import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
from .tile import build_coord_lookup, core_with_halo
from ....utils import offload
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d

_TILE_LOGGED = False
_ALGO_LOGGED = False


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def _apply_flex_runtime():
    global _ALGO_LOGGED
    algo = offload.recommend_flex_algo() or config.FLEX_GEMM_ALGO
    flex_gemm.ops.spconv.set_algorithm(algo)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)
    if algo != config.FLEX_GEMM_ALGO and not _ALGO_LOGGED:
        print(
            f"[TRELLIS.2] Sparse conv algorithm {algo} "
            f"(library default is {config.FLEX_GEMM_ALGO}). "
            "Set TRELLIS_FLEX_ALGO to override."
        )
        _ALGO_LOGGED = True
    return algo


def _conv_once(module, feats, coords, shape):
    out, _neighbor_cache = sparse_submanifold_conv3d(
        feats,
        coords,
        shape,
        module.weight,
        module.bias,
        None,
        module.dilation,
    )
    return out


def _tiled_sparse_conv3d_forward(module, x: SparseTensor, chunk: int) -> SparseTensor:
    """Run one submanifold conv in haloed chunks. Output matches the full conv."""
    global _TILE_LOGGED
    feats = x.feats
    coords = x.coords
    n = feats.shape[0]
    shape = torch.Size([*x.shape, *x.spatial_shape])
    lookup = build_coord_lookup(coords)
    if not _TILE_LOGGED:
        print(
            f"[TRELLIS.2] Tiling sparse conv: {n} voxels in chunks of {chunk} "
            "(halo included, same result as one launch). "
            "Set TRELLIS_CONV_CHUNK=0 to disable."
        )
        _TILE_LOGGED = True
    parts = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        core = torch.arange(start, end, device=coords.device)
        selected = core_with_halo(
            coords,
            core,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            lookup=lookup,
        )
        out = _conv_once(module, feats[selected].contiguous(), coords[selected].contiguous(), shape)
        parts.append(out[: end - start])
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return x.replace(torch.cat(parts, dim=0))


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    _apply_flex_runtime()

    chunk = offload.recommend_conv_chunk()
    # Only the 3×3×3 VAE convs are tiled. The halo math matches that kernel,
    # which is the launch that TDRs 4 GB at the finest decoder level.
    if (
        chunk
        and x.feats.shape[0] > chunk
        and tuple(self.kernel_size) == (3, 3, 3)
        and not self.training
    ):
        return _tiled_sparse_conv3d_forward(self, x, chunk)

    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)
    
    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation
    )
    
    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
    
    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
