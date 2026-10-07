import os
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image

from trellis2.utils.glb_inspect import (
    apply_trellis_frame,
    format_report,
    inspect_glb,
    inspect_mesh,
    load_glb_meshes,
    normalization_frame,
    read_mask,
    retarget_mesh,
    write_mask,
)


def _textured_faces(image, n_faces_hint=None):
    """A box with unique vertices per face so UVs can target different tiles."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    faces = np.asarray(box.faces)
    vertices = np.asarray(box.vertices, dtype=np.float64)
    exploded_v = vertices[faces].reshape(-1, 3)
    exploded_f = np.arange(exploded_v.shape[0], dtype=np.int64).reshape(-1, 3)
    # 12 faces. Put the first 6 on the left (noisy) half and the rest on gray.
    uv = np.zeros((exploded_v.shape[0], 2), dtype=np.float64)
    for face_id in range(exploded_f.shape[0]):
        u0 = 0.05 if face_id < 6 else 0.70
        v0 = 0.05
        uv[face_id * 3 + 0] = (u0, v0)
        uv[face_id * 3 + 1] = (u0 + 0.2, v0)
        uv[face_id * 3 + 2] = (u0, v0 + 0.2)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(exploded_v, exploded_f, process=False, visual=visual)


class InspectTests(unittest.TestCase):
    def test_open_box_has_boundary_and_no_texture(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        box.faces = box.faces[1:]
        report = inspect_mesh("box", box)
        self.assertGreater(report["open_boundary_edges"], 0)
        self.assertEqual(report["recommendation"], "retexture")
        self.assertEqual(report["repaint_face_count"], len(box.faces))

    def test_small_component_is_flagged(self):
        big = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        small_v = np.array(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]],
            dtype=np.float64,
        )
        small_v += [3.0, 0.0, 0.0]
        small_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        vertices = np.concatenate([big.vertices, small_v], axis=0)
        faces = np.concatenate([big.faces, small_f + len(big.vertices)], axis=0)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        report = inspect_mesh("soup", mesh)
        self.assertGreaterEqual(report["components"], 2)
        self.assertGreaterEqual(report["small_components"], 1)
        self.assertTrue(report["geometry_flagged"])

    def test_noisy_texture_recommends_enhance(self):
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        mesh = _textured_faces(image)
        # Force every face onto the noisy image by using the left-half UVs only.
        mesh.visual.uv[:, 0] = np.clip(mesh.visual.uv[:, 0] * 0 + 0.1, 0, 1)
        mesh.visual.uv[:, 0] = 0.25
        report = inspect_mesh("noisy", mesh)
        self.assertEqual(report["recommendation"], "enhance")
        self.assertEqual(report["repaint_face_count"], 0)
        self.assertFalse(report["texture"]["has_metallic_roughness_map"])

    def test_flat_texture_recommends_retexture(self):
        image = Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8))
        mesh = _textured_faces(image)
        report = inspect_mesh("flat", mesh)
        self.assertEqual(report["recommendation"], "retexture")
        self.assertEqual(report["repaint_face_count"], len(mesh.faces))

    def test_half_flat_recommends_inpaint_and_roundtrips_mask(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        rng = np.random.default_rng(1)
        image[:, :32] = rng.integers(0, 255, (64, 32, 3), dtype=np.uint8)
        image[:, 32:] = 180
        mesh = _textured_faces(Image.fromarray(image))
        report = inspect_mesh("half", mesh)
        self.assertEqual(report["recommendation"], "inpaint")
        self.assertGreater(report["repaint_face_count"], 0)
        self.assertLess(report["repaint_face_count"], len(mesh.faces))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "half.glb")
            mesh.export(path)
            loaded = inspect_glb(path)
            self.assertIn("half.glb", loaded["source"])
            mask_path = os.path.join(tmp, "mask.npz")
            write_mask(mask_path, loaded)
            masks = read_mask(mask_path)
            self.assertEqual(len(masks), len(loaded["meshes"]))
            text = format_report(loaded)
            self.assertIn("inpaint", text)

    def test_scene_translation_is_baked(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        scene = trimesh.Scene()
        scene.add_geometry(box, geom_name="moved")
        scene.graph[scene.graph.nodes_geometry[0]] = trimesh.transformations.translation_matrix([4.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.glb")
            scene.export(path)
            meshes = load_glb_meshes(path)
            self.assertEqual(len(meshes), 1)
            self.assertGreater(float(np.asarray(meshes[0][1].vertices)[:, 0].mean()), 2.0)

    def test_retarget_keeps_uvs_and_unit_cube(self):
        image = Image.fromarray(np.full((8, 8, 3), 10, dtype=np.uint8))
        mesh = _textured_faces(image)
        mesh.apply_translation([10.0, -3.0, 2.0])
        center, scale = normalization_frame(mesh.vertices)
        retargeted = retarget_mesh(mesh, center, scale)
        self.assertTrue(np.all(retargeted.vertices >= -0.5 - 1e-6))
        self.assertTrue(np.all(retargeted.vertices <= 0.5 + 1e-6))
        self.assertEqual(len(retargeted.visual.uv), len(retargeted.vertices))
        framed = apply_trellis_frame(mesh.vertices, center, scale)
        np.testing.assert_allclose(retargeted.vertices, framed)


if __name__ == "__main__":
    unittest.main()
