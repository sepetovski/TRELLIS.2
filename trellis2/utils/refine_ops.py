"""Coord alignment and repaint masks for texture refinement.

Torch only. The CUDA voxelizers live in the pipeline; everything that decides
*which* latent tokens to keep or regenerate is tested here on CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def xyz_of(coords: torch.Tensor) -> torch.Tensor:
    coords = torch.as_tensor(coords).long()
    if coords.ndim != 2 or coords.shape[-1] not in (3, 4):
        raise ValueError(f"coords must be [N, 3] or [N, 4], got {tuple(coords.shape)}")
    if coords.shape[-1] == 4:
        coords = coords[:, 1:]
    return coords


def _keys(coords: torch.Tensor, origin: torch.Tensor, span: int) -> torch.Tensor:
    shifted = coords - origin
    return (shifted[:, 0] * span + shifted[:, 1]) * span + shifted[:, 2]


def match_coords(
    src_coords: torch.Tensor,
    dst_coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each dst row, the first src row with the same xyz.

    Returns (src_index [Ndst], found [Ndst]). Unmatched dst rows get index 0
    and found=False.
    """
    src = xyz_of(src_coords)
    dst = xyz_of(dst_coords)
    src_index = torch.zeros(dst.shape[0], dtype=torch.long)
    found = torch.zeros(dst.shape[0], dtype=torch.bool)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return src_index, found
    origin = torch.minimum(src.min(dim=0).values, dst.min(dim=0).values)
    span = int(torch.maximum(src.max(dim=0).values, dst.max(dim=0).values).sub(origin).max().item()) + 2
    if span > 100_000:
        raise ValueError(f"coord span {span} is too large to pack")
    src_key = _keys(src, origin, span)
    order = torch.argsort(src_key)
    src_key = src_key[order]
    unique = torch.ones(src_key.shape[0], dtype=torch.bool)
    unique[1:] = src_key[1:] != src_key[:-1]
    first = order[unique]
    unique_key = src_key[unique]
    dst_key = _keys(dst, origin, span)
    pos = torch.searchsorted(unique_key, dst_key)
    pos_clamp = pos.clamp(max=unique_key.shape[0] - 1)
    found = (pos < unique_key.shape[0]) & (unique_key[pos_clamp] == dst_key)
    src_index[found] = first[pos_clamp[found]]
    return src_index, found


def gather_feats_by_coord(
    src_coords: torch.Tensor,
    src_feats: torch.Tensor,
    dst_coords: torch.Tensor,
    fill: float | torch.Tensor = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy ``src_feats`` onto ``dst_coords``. Missing rows are ``fill``.

    Returns (feats [Ndst, C], found [Ndst]).
    """
    src_feats = torch.as_tensor(src_feats)
    if src_feats.ndim == 1:
        src_feats = src_feats.unsqueeze(-1)
    dst_n = xyz_of(dst_coords).shape[0]
    channels = int(src_feats.shape[-1]) if src_feats.ndim == 2 and src_feats.shape[0] > 0 else int(torch.as_tensor(fill).numel() or 1)
    if src_feats.ndim == 2 and src_feats.shape[0] > 0:
        channels = int(src_feats.shape[-1])
    out = torch.empty(dst_n, channels, dtype=src_feats.dtype if src_feats.numel() else torch.float32)
    fill_t = torch.as_tensor(fill, dtype=out.dtype)
    if fill_t.numel() == 1:
        out.fill_(fill_t.reshape(()))
    else:
        if int(fill_t.numel()) != channels:
            raise ValueError(f"fill has {fill_t.numel()} values, feats have {channels}")
        out[:] = fill_t.reshape(1, channels)
    if dst_n == 0 or src_feats.shape[0] == 0:
        return out, torch.zeros(dst_n, dtype=torch.bool)
    src_index, found = match_coords(src_coords, dst_coords)
    out[found] = src_feats[src_index[found]].to(dtype=out.dtype)
    return out, found


def membership(pool_coords: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
    """True for each pool row whose xyz appears in query."""
    if xyz_of(pool_coords).shape[0] == 0 or xyz_of(query_coords).shape[0] == 0:
        return torch.zeros(xyz_of(pool_coords).shape[0], dtype=torch.bool)
    _, found = match_coords(query_coords, pool_coords)
    return found


def row_index_after_cap(coords: torch.Tensor, max_tokens: Optional[int]) -> torch.Tensor:
    """Indices into ``coords`` after the 4 GB shell cap. Order follows the cap."""
    coords = torch.as_tensor(coords).long()
    n = int(coords.shape[0])
    if max_tokens is None or n <= int(max_tokens):
        return torch.arange(n)
    from trellis2.utils.offload import cap_sparse_coords

    if coords.shape[-1] == 3:
        coords4 = torch.cat([torch.zeros(n, 1, dtype=torch.long), coords], dim=1)
    elif coords.shape[-1] == 4:
        coords4 = coords
    else:
        raise ValueError(f"coords must be [N, 3] or [N, 4], got {tuple(coords.shape)}")
    selected = cap_sparse_coords(coords4, int(max_tokens)).long().cpu()
    src_index, found = match_coords(coords4.cpu(), selected)
    if not bool(found.all()):
        raise RuntimeError("capped coords could not be matched back to the input")
    return src_index


def voxel_flags_to_latent_mask(
    voxel_coords: torch.Tensor,
    voxel_flag: torch.Tensor,
    latent_coords: torch.Tensor,
    downsample: int = 16,
) -> torch.Tensor:
    """A latent token is flagged when any flagged voxel lands in its cell.

    The texture encoder downsamples with four stride-2 spatial-to-channel
    steps, so a full-res voxel (x, y, z) maps to latent (x//16, y//16, z//16).
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    voxel_xyz = xyz_of(voxel_coords)
    flag = torch.as_tensor(voxel_flag).bool().reshape(-1)
    if flag.shape[0] != voxel_xyz.shape[0]:
        raise ValueError(
            f"voxel_flag has {flag.shape[0]} rows, voxels have {voxel_xyz.shape[0]}"
        )
    latent_xyz = xyz_of(latent_coords)
    if not bool(flag.any()):
        return torch.zeros(latent_xyz.shape[0], dtype=torch.bool)
    cells = (voxel_xyz[flag] // int(downsample)).unique(dim=0)
    return membership(latent_xyz, cells)


def _expand_children(parent_xyz: torch.Tensor, subdiv: torch.Tensor) -> torch.Tensor:
    """Child coords in the same order as ``SparseChannel2Spatial``.

    ``subdiv`` is [N, 8] with a positive value where that octant is occupied.
    Octant ``c`` is ``(x%2) + 2*(y%2) + 4*(z%2)``, matching spatial-to-channel.
    """
    if subdiv.ndim != 2 or subdiv.shape[-1] != 8:
        raise ValueError(f"subdiv must be [N, 8], got {tuple(subdiv.shape)}")
    rows, cols = (subdiv > 0).nonzero(as_tuple=True)
    if rows.numel() == 0:
        return parent_xyz.new_zeros((0, 3))
    dx = cols % 2
    dy = (cols // 2) % 2
    dz = cols // 4
    base = parent_xyz[rows] * 2
    return torch.stack([base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz], dim=1)


def subdivision_guides(
    latent_coords: torch.Tensor,
    voxel_coords: torch.Tensor,
    resolution: int,
    levels: int = 4,
) -> list:
    """Build per-level subdivision masks from the mesh voxels.

    The texture decoder (``pred_subdiv=False``) needs a guide at every
    upsample or it refuses to expand. These guides are the input mesh's own
    occupancy, so the decoded texture lands on the same surface we encoded
    and we do not have to run the shape decoder.

    Each item is ``(coords [N, 4], subdiv [N, 8])`` in decoder order: index 0
    is the latent grid, and each later grid is the children of the previous.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if int(resolution) % (2 ** int(levels)) != 0:
        raise ValueError(f"resolution {resolution} is not divisible by {2 ** levels}")
    latent_coords = torch.as_tensor(latent_coords).long()
    if latent_coords.ndim != 2 or latent_coords.shape[-1] != 4:
        raise ValueError("latent_coords must be [N, 4]")
    voxel_xyz = xyz_of(voxel_coords)
    current = latent_coords
    parent_size = 2 ** int(levels)
    guides = []
    for _ in range(int(levels)):
        child_size = parent_size // 2
        parent_xyz = current[:, 1:]
        subdiv = torch.zeros(parent_xyz.shape[0], 8, dtype=torch.float32)
        if voxel_xyz.shape[0] > 0 and parent_xyz.shape[0] > 0:
            parent_of_voxel = voxel_xyz // parent_size
            local = (voxel_xyz // child_size) % 2
            octant = local[:, 0] + 2 * local[:, 1] + 4 * local[:, 2]
            src_index, found = match_coords(parent_xyz, parent_of_voxel)
            if bool(found.any()):
                subdiv[src_index[found], octant[found]] = 1.0
        guides.append((current, subdiv))
        children = _expand_children(parent_xyz, subdiv)
        batch = torch.zeros(children.shape[0], 1, dtype=torch.long)
        current = torch.cat([batch, children], dim=1) if children.shape[0] else current.new_zeros((0, 4))
        parent_size = child_size
    return guides


def faces_to_latent_mask(
    vertices,
    faces,
    face_index,
    latent_coords: torch.Tensor,
    resolution: int,
    downsample: int = 16,
    dilate: int = 1,
) -> torch.Tensor:
    """Mark latent tokens near the given faces.

    Samples each triangle at its vertices and centroid, quantizes those
    points onto the voxel grid, then dilates by ``dilate`` latent cells so a
    thin face does not collapse to a single token.
    """
    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    faces = torch.as_tensor(faces, dtype=torch.long)
    face_index = torch.as_tensor(face_index, dtype=torch.long).reshape(-1)
    latent_xyz = xyz_of(latent_coords)
    if face_index.numel() == 0:
        return torch.zeros(latent_xyz.shape[0], dtype=torch.bool)
    if int(face_index.min()) < 0 or int(face_index.max()) >= int(faces.shape[0]):
        raise ValueError("face_index is outside this mesh's face range")
    tris = vertices[faces[face_index]]
    samples = torch.cat([tris.mean(dim=1), tris[:, 0], tris[:, 1], tris[:, 2]], dim=0)
    grid = ((samples + 0.5) * int(resolution)).long().clamp(0, int(resolution) - 1)
    cells = grid // int(downsample)
    if dilate > 0:
        offs = torch.arange(-int(dilate), int(dilate) + 1)
        ox, oy, oz = torch.meshgrid(offs, offs, offs, indexing="ij")
        delta = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)
        cells = (cells.unsqueeze(1) + delta.unsqueeze(0)).reshape(-1, 3)
    limit = int(resolution) // int(downsample)
    keep = (cells >= 0).all(dim=1) & (cells < limit).all(dim=1)
    cells = cells[keep].unique(dim=0)
    return membership(latent_xyz, cells)
