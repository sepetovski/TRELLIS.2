"""Put mesh and camera tensors on the rasterizer device.

Low-VRAM decode keeps the mesh on CPU so the GPU can be cleared between
stages. Preview cameras are built on CUDA. nvdiffrast then dies in
``torch.bmm`` with ``cpu and cuda:0``.
"""
import torch


def _as_device(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _on_device(value, device: torch.device) -> bool:
    return not torch.is_tensor(value) or value.device == device


def mesh_on_device(mesh, device) -> bool:
    device = _as_device(device)
    if not _on_device(mesh.vertices, device) or not _on_device(mesh.faces, device):
        return False
    for name in ("vertex_attrs", "coords", "attrs", "origin", "uv_coords", "material_ids"):
        if not _on_device(getattr(mesh, name, None), device):
            return False
    for mat in getattr(mesh, "materials", None) or []:
        if not _on_device(getattr(mat, "base_color_factor", None), device):
            return False
        for tex_name in (
            "base_color_texture",
            "metallic_texture",
            "roughness_texture",
            "alpha_texture",
        ):
            tex = getattr(mat, tex_name, None)
            if tex is not None and not _on_device(getattr(tex, "image", None), device):
                return False
    return True


def prepare_raster_inputs(mesh, extrinsics, intrinsics, transformation, device):
    """Copy the mesh and cameras onto ``device`` when they are split across devices."""
    device = _as_device(device)
    if not mesh_on_device(mesh, device):
        mesh = mesh.to(device)
    extrinsics = extrinsics.to(device=device)
    intrinsics = intrinsics.to(device=device)
    if transformation is not None:
        transformation = transformation.to(device=device)
    return mesh, extrinsics, intrinsics, transformation
