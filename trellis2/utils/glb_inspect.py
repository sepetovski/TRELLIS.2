"""CPU inspection of a GLB/GLTF before texture refinement.

This does not load TRELLIS. No GPU, no checkpoints. It reports geometric
problems the refiner will not rebuild (holes, floaters, degenerate faces)
and texture problems the refiner can target (missing maps, flat regions).

Face indices written to the mask file match ``load_glb_meshes`` order, which
is also the order ``example_refine.py`` voxelizes.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from PIL import Image


MeshEntry = Tuple[str, trimesh.Trimesh]

# A tile whose RGB std (0-255) is below this is treated as unpainted.
_FLAT_STD = 4.0
# Fraction of faces sitting on flat tiles.
_INPAINT_FRACTION = 0.05
_RETEXTURE_FRACTION = 0.85

SUGGESTED_STRENGTH = {
    "enhance": 0.35,
    "inpaint": 0.9,
    "retexture": 1.0,
}


def load_glb_meshes(path: str) -> List[MeshEntry]:
    """Load a mesh file and bake node transforms into vertex positions."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        if len(loaded.faces) == 0:
            raise ValueError(f"{path} has no faces")
        return [("mesh", loaded)]
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)}")

    entries: List[MeshEntry] = []
    seen = set()
    nodes = list(getattr(loaded.graph, "nodes_geometry", []) or [])
    for node in nodes:
        transform, geom_name = loaded.graph[node]
        geom = loaded.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        name = str(node)
        if name in seen:
            name = f"{name}_{len(entries)}"
        seen.add(name)
        entries.append((name, mesh))
    if entries:
        return entries

    for name, geom in loaded.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
            entries.append((str(name), geom.copy()))
    if not entries:
        raise ValueError(f"{path} has no triangle meshes")
    return entries


def normalization_frame(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center/scale used by the texturing pipeline (unit cube, then Y/Z swap)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = float((vmax - vmin).max())
    if extent <= 0:
        raise ValueError("mesh has zero bounding-box extent")
    scale = 0.99999 / extent
    return center, scale


def apply_trellis_frame(vertices: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Match ``Trellis2TexturingPipeline.preprocess_mesh`` vertex transform."""
    out = (np.asarray(vertices, dtype=np.float64) - center) * scale
    swapped = out.copy()
    swapped[:, 1] = -out[:, 2]
    swapped[:, 2] = out[:, 1]
    return swapped


def retarget_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    """Normalize vertices. UVs and material stay attached."""
    vertices = apply_trellis_frame(mesh.vertices, center, scale)
    visual = _copy_visual(mesh)
    kwargs = {}
    if visual is not None:
        kwargs["visual"] = visual
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces).copy(),
        process=False,
        **kwargs,
    )


def _copy_visual(mesh: trimesh.Trimesh):
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    copy = getattr(visual, "copy", None)
    if copy is None:
        return visual
    try:
        return copy()
    except Exception:
        return visual


def concat_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    """One triangle soup for voxelization. Visuals are dropped on purpose."""
    if len(meshes) == 1:
        return meshes[0]
    verts = []
    faces = []
    offset = 0
    for mesh in meshes:
        verts.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(mesh.vertices)
    return trimesh.Trimesh(
        vertices=np.concatenate(verts, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )


def inspect_glb(path: str) -> dict:
    meshes = load_glb_meshes(path)
    reports = [inspect_mesh(name, mesh) for name, mesh in meshes]
    summary = summarize_reports(reports)
    summary["source"] = path
    return {"source": path, "summary": summary, "meshes": reports}


def inspect_mesh(name: str, mesh: trimesh.Trimesh) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    topology = _topology(vertices, faces)
    texture = _texture_report(mesh, faces)
    repaint = texture["repaint_faces"]
    recommendation = texture["recommendation"]
    return {
        "name": name,
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "degenerate_faces": topology["degenerate_faces"],
        "duplicate_face_groups": topology["duplicate_face_groups"],
        "open_boundary_edges": topology["open_boundary_edges"],
        "nonmanifold_edges": topology["nonmanifold_edges"],
        "components": topology["components"],
        "small_components": topology["small_components"],
        "nonfinite_vertices": topology["nonfinite_vertices"],
        "winding_consistent": topology["winding_consistent"],
        "texture": {k: v for k, v in texture.items() if k != "repaint_faces"},
        "repaint_face_count": int(len(repaint)),
        "repaint_face_preview": [int(i) for i in repaint[:64]],
        "repaint_faces": repaint,
        "recommendation": recommendation,
        "suggested_strength": SUGGESTED_STRENGTH[recommendation],
        "geometry_flagged": _geometry_flagged(topology),
    }


def summarize_reports(reports: Sequence[dict]) -> dict:
    if not reports:
        raise ValueError("no meshes to summarize")
    modes = [r["recommendation"] for r in reports]
    if any(m == "inpaint" for m in modes) or (
        "retexture" in modes and "enhance" in modes
    ):
        recommendation = "inpaint"
    elif all(m == "retexture" for m in modes):
        recommendation = "retexture"
    elif "retexture" in modes:
        recommendation = "inpaint"
    else:
        recommendation = "enhance"
    geometry = any(r["geometry_flagged"] for r in reports)
    reasons = []
    for report in reports:
        reasons.append(
            f"{report['name']}: {report['recommendation']} "
            f"({report['repaint_face_count']} faces to repaint)"
        )
        if report["geometry_flagged"]:
            reasons.append(
                f"{report['name']}: geometry issues are reported only; "
                "this pass does not rebuild the mesh"
            )
    return {
        "meshes": len(reports),
        "vertices": int(sum(r["vertices"] for r in reports)),
        "faces": int(sum(r["faces"] for r in reports)),
        "recommendation": recommendation,
        "suggested_strength": SUGGESTED_STRENGTH[recommendation],
        "geometry_flagged": geometry,
        "reasons": reasons,
    }


def _geometry_flagged(topology: dict) -> bool:
    return bool(
        topology["degenerate_faces"]
        or topology["duplicate_face_groups"]
        or topology["nonmanifold_edges"]
        or topology["small_components"]
        or topology["nonfinite_vertices"]
        or topology["winding_consistent"] is False
    )


def _topology(vertices: np.ndarray, faces: np.ndarray) -> dict:
    n_faces = int(faces.shape[0])
    nonfinite = int((~np.isfinite(vertices).all(axis=1)).sum()) if len(vertices) else 0
    if n_faces == 0:
        return {
            "degenerate_faces": 0,
            "duplicate_face_groups": 0,
            "open_boundary_edges": 0,
            "nonmanifold_edges": 0,
            "components": 0,
            "small_components": 0,
            "nonfinite_vertices": nonfinite,
            "winding_consistent": None,
        }
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) * 0.5
    degenerate = int(np.sum(~np.isfinite(area) | (area < 1e-12)))

    sorted_faces = np.sort(faces, axis=1)
    _, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    duplicate_groups = int(np.sum(counts > 1))

    boundary, nonmanifold, components, small = _edge_components(faces)
    winding = None
    if boundary == 0 and n_faces <= 100_000 and nonfinite == 0:
        try:
            winding = bool(trimesh.Trimesh(vertices, faces, process=False).is_winding_consistent)
        except Exception:
            winding = None
    return {
        "degenerate_faces": degenerate,
        "duplicate_face_groups": duplicate_groups,
        "open_boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "components": components,
        "small_components": small,
        "nonfinite_vertices": nonfinite,
        "winding_consistent": winding,
    }


def _edge_components(faces: np.ndarray) -> Tuple[int, int, int, int]:
    n = int(faces.shape[0])
    n_verts = int(faces.max()) + 1 if n else 1
    edges = np.sort(
        np.stack(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
            axis=1,
        ).reshape(-1, 2),
        axis=1,
    )
    keys = edges[:, 0] * np.int64(n_verts) + edges[:, 1]
    face_ids = np.repeat(np.arange(n, dtype=np.int64), 3)
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    face_ids = face_ids[order]
    breaks = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), breaks])
    ends = np.concatenate([breaks, np.array([keys.shape[0]], dtype=np.int64)])
    counts = ends - starts
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    paired = np.flatnonzero(counts == 2)
    for group in paired:
        s = int(starts[group])
        union(int(face_ids[s]), int(face_ids[s + 1]))

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, comp_counts = np.unique(roots, return_counts=True)
    n_comp = int(comp_counts.shape[0])
    if n_comp <= 1:
        return boundary, nonmanifold, n_comp, 0
    largest = int(comp_counts.max())
    small_limit = max(8, int(0.002 * n))
    small = int(np.sum((comp_counts < small_limit) & (comp_counts < largest)))
    return boundary, nonmanifold, n_comp, small


def _texture_report(mesh: trimesh.Trimesh, faces: np.ndarray) -> dict:
    n_faces = int(faces.shape[0])
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None) if visual is not None else None
    uv = getattr(visual, "uv", None) if visual is not None else None
    image = _base_color_image(material)
    has_mr = bool(material is not None and getattr(material, "metallicRoughnessTexture", None) is not None)
    kind = "texture" if image is not None else "vertex_color" if _has_vertex_colors(visual) else "factor" if material is not None else "none"

    all_faces = np.arange(n_faces, dtype=np.int64)
    if image is None or uv is None or len(np.asarray(uv)) != len(mesh.vertices):
        return {
            "kind": kind,
            "width": None if image is None else int(image.size[0]),
            "height": None if image is None else int(image.size[1]),
            "has_metallic_roughness_map": has_mr,
            "uvs": uv is not None,
            "flat_fraction": 1.0,
            "recommendation": "retexture",
            "repaint_faces": all_faces,
        }

    uv = np.asarray(uv, dtype=np.float64)
    block_std, flat_faces = _flat_faces(image, uv, faces)
    fraction = float(len(flat_faces) / n_faces) if n_faces else 1.0
    if fraction >= _RETEXTURE_FRACTION:
        recommendation = "retexture"
        repaint = all_faces
    elif fraction >= _INPAINT_FRACTION:
        recommendation = "inpaint"
        repaint = flat_faces
    else:
        recommendation = "enhance"
        repaint = np.zeros((0,), dtype=np.int64)
    uv_outside = int(np.sum((uv < -1e-3).any(axis=1) | (uv > 1.0 + 1e-3).any(axis=1)))
    return {
        "kind": kind,
        "width": int(image.size[0]),
        "height": int(image.size[1]),
        "has_metallic_roughness_map": has_mr,
        "uvs": True,
        "uv_vertices_outside_0_1": uv_outside,
        "flat_fraction": fraction,
        "median_tile_std": float(np.median(block_std)) if block_std.size else 0.0,
        "recommendation": recommendation,
        "repaint_faces": repaint.astype(np.int64, copy=False),
    }


def _base_color_image(material) -> Optional[Image.Image]:
    if material is None:
        return None
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.asarray(image))
        except Exception:
            return None
    return image.convert("RGB")


def _has_vertex_colors(visual) -> bool:
    if visual is None:
        return False
    colors = getattr(visual, "vertex_colors", None)
    return colors is not None and len(colors) > 0


def _flat_faces(image: Image.Image, uv: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    small = np.asarray(image.resize((64, 64), Image.Resampling.BOX), dtype=np.float32)
    blocks = small.reshape(8, 8, 8, 8, 3).std(axis=(1, 3)).mean(axis=-1)
    centroid = uv[faces].mean(axis=1)
    px = np.clip((centroid[:, 0] * 8).astype(np.int64), 0, 7)
    py = np.clip(((1.0 - centroid[:, 1]) * 8).astype(np.int64), 0, 7)
    std = blocks[py, px]
    flat = np.flatnonzero(std < _FLAT_STD).astype(np.int64)
    return blocks, flat


def public_mesh_report(report: dict) -> dict:
    """Drop the full face list so the JSON stays small."""
    out = dict(report)
    out.pop("repaint_faces", None)
    texture = dict(out.get("texture") or {})
    texture.pop("repaint_faces", None)
    out["texture"] = texture
    return out


def report_to_json(report: dict) -> str:
    payload = {
        "source": report["source"],
        "summary": report["summary"],
        "meshes": [public_mesh_report(mesh) for mesh in report["meshes"]],
    }
    return json.dumps(_jsonable(payload), indent=2)


def write_mask(path: str, report: dict) -> None:
    names = []
    arrays = {}
    for index, mesh in enumerate(report["meshes"]):
        names.append(mesh["name"])
        arrays[f"faces_{index}"] = np.asarray(mesh["repaint_faces"], dtype=np.int64)
    np.savez(path, names=np.array(names), **arrays)


def read_mask(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    names = [str(x) for x in data["names"].tolist()]
    return {name: np.asarray(data[f"faces_{index}"], dtype=np.int64) for index, name in enumerate(names)}


def format_report(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"{report['source']}",
        (
            f"  {summary['meshes']} mesh(es), {summary['vertices']} verts, "
            f"{summary['faces']} faces"
        ),
        (
            f"  recommendation: {summary['recommendation']} "
            f"(strength {summary['suggested_strength']})"
        ),
    ]
    if summary["geometry_flagged"]:
        lines.append("  geometry issues found. Texture refine will not rebuild the mesh.")
    for mesh in report["meshes"]:
        tex = mesh["texture"]
        size = (
            "no base-color texture"
            if not tex.get("width")
            else f"{tex['width']}x{tex['height']} {tex['kind']}"
        )
        lines.append(
            f"  [{mesh['name']}] {mesh['vertices']} verts, {mesh['faces']} faces, {size}"
        )
        lines.append(
            "    degenerate={degenerate_faces} duplicate_groups={duplicate_face_groups} "
            "open_edges={open_boundary_edges} nonmanifold_edges={nonmanifold_edges} "
            "components={components} small_components={small_components}".format(**mesh)
        )
        lines.append(
            f"    flat_fraction={tex.get('flat_fraction', 1):.2f} "
            f"metallic_roughness_map={tex.get('has_metallic_roughness_map')} "
            f"repaint_faces={mesh['repaint_face_count']} "
            f"-> {mesh['recommendation']}"
        )
    for reason in summary["reasons"]:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
