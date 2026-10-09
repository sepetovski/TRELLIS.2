"""Halo gathering for tiled submanifold sparse convolution.

A submanifold conv writes one output row per input voxel, and that row depends
only on the voxels inside the kernel window. Splitting the volume into chunks
and pulling in the occupied neighbors (the halo) is numerically the same as
running the conv on the full set, but each kernel launch stays small enough
that Windows TDR does not kill a 4 GB card.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


_SHIFT = 20
_BIAS = 1 << 19


def pack_coord_keys(coords: torch.Tensor) -> torch.Tensor:
    """Pack (batch, x, y, z) into one int64. Negative neighbor lookups stay unique."""
    c = coords.to(dtype=torch.int64)
    return (
        (c[:, 0] << (_SHIFT * 3))
        | ((c[:, 1] + _BIAS) << (_SHIFT * 2))
        | ((c[:, 2] + _BIAS) << _SHIFT)
        | (c[:, 3] + _BIAS)
    )


def kernel_offsets(kernel_size: Sequence[int], dilation: Sequence[int]):
    ks = tuple(int(k) for k in kernel_size)
    dil = tuple(int(d) for d in dilation)
    if len(dil) == 1:
        dil = dil * 3
    ranges = []
    for k, d in zip(ks, dil):
        radius = (k // 2) * d
        ranges.append(range(-radius, radius + 1, d if d else 1))
    offsets = []
    for oz in ranges[2]:
        for oy in ranges[1]:
            for ox in ranges[0]:
                if ox == 0 and oy == 0 and oz == 0:
                    continue
                offsets.append((ox, oy, oz))
    return offsets


def build_coord_lookup(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sorted unique keys and the first voxel index for each key."""
    keys = pack_coord_keys(coords)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    first = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=coords.device)
    if sorted_keys.shape[0] > 1:
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return sorted_keys[first], order[first]


def lookup_indices(query_keys: torch.Tensor, sorted_keys: torch.Tensor, sorted_idx: torch.Tensor):
    if sorted_keys.numel() == 0 or query_keys.numel() == 0:
        empty_idx = query_keys.new_empty(query_keys.shape[0], dtype=torch.long)
        hit = torch.zeros(query_keys.shape[0], dtype=torch.bool, device=query_keys.device)
        return empty_idx, hit
    pos = torch.searchsorted(sorted_keys, query_keys).clamp(max=sorted_keys.shape[0] - 1)
    hit = sorted_keys[pos] == query_keys
    return sorted_idx[pos], hit


def core_with_halo(
    coords: torch.Tensor,
    core_index: torch.Tensor,
    kernel_size: Sequence[int] = (3, 3, 3),
    dilation: Sequence[int] = (1, 1, 1),
    lookup: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Return `core_index` followed by occupied kernel-neighbors that are not in the core.

    Core rows stay in the original order so a submanifold conv's first
    `len(core)` outputs are the chunk the caller wants to keep.
    """
    if lookup is None:
        lookup = build_coord_lookup(coords)
    sorted_keys, sorted_idx = lookup
    core_index = core_index.to(device=coords.device, dtype=torch.long)
    if core_index.numel() == 0:
        return core_index
    core_coords = coords[core_index]
    halo_parts = []
    for ox, oy, oz in kernel_offsets(kernel_size, dilation):
        shifted = core_coords.clone()
        shifted[:, 1] = shifted[:, 1] + ox
        shifted[:, 2] = shifted[:, 2] + oy
        shifted[:, 3] = shifted[:, 3] + oz
        idx, hit = lookup_indices(pack_coord_keys(shifted), sorted_keys, sorted_idx)
        if bool(hit.any()):
            halo_parts.append(idx[hit])
    if not halo_parts:
        return core_index
    halo = torch.unique(torch.cat(halo_parts, dim=0))
    halo = halo[~torch.isin(halo, core_index)]
    if halo.numel() == 0:
        return core_index
    return torch.cat([core_index, halo], dim=0)
