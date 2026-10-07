"""Texture refine for an existing GLB, aimed at the 4 GB path.

Stages, one at a time (same offload rules as image-to-3D on a 4 GB card):

  1. voxelize the mesh on CPU/CUDA, capped like the texture decoder
  2. shape encoder  (already used by the texturing pipeline)
  3. texture encoder (same size as the shape encoder; not resident with the DiT)
  4. texture DiT for only the steps ``strength`` still needs
  5. texture decoder, guided by the mesh's own voxels
  6. bake onto the original UVs

The 1024 DiT and the shape decoder are not loaded. Geometry is not rebuilt.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image

import o_voxel

from ..modules.sparse import SparseTensor
from ..utils import offload
from ..utils.glb_inspect import (
    concat_meshes,
    load_glb_meshes,
    normalization_frame,
    retarget_mesh,
)
from ..utils.refine_ops import (
    faces_to_latent_mask,
    gather_feats_by_coord,
    row_index_after_cap,
    subdivision_guides,
    voxel_flags_to_latent_mask,
)
from .trellis2_texturing import Trellis2TexturingPipeline


_MODES = ("enhance", "inpaint", "retexture")
# base color 0.8, metallic 0, roughness 0.5, alpha 1 — same fallback the
# training voxelizer appends for faces with no material.
_DEFAULT_PBR_U8 = (204, 204, 204, 0, 128, 255)


class Trellis2RefinePipeline(Trellis2TexturingPipeline):
    """Encode an existing GLB and re-sample only its texture latent."""

    model_names_to_load = [
        "shape_slat_encoder",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    ]
    TEX_ENCODER_SPEC = "ckpts/tex_enc_next_dc_f16c32_fp16"
    RESOLUTION = 512
    DOWNSAMPLE = 16

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "texturing_pipeline.json") -> "Trellis2RefinePipeline":
        pipeline = super().from_pretrained(path, config_file)
        specs = getattr(pipeline.models, "_specs", None)
        if specs is not None:
            specs["tex_slat_encoder"] = cls.TEX_ENCODER_SPEC
        else:
            raise RuntimeError("refine pipeline expects the lazy model loader")
        pipeline._model_specs["tex_slat_encoder"] = cls.TEX_ENCODER_SPEC
        return pipeline

    @torch.no_grad()
    def run(
        self,
        mesh: Union[str, trimesh.Trimesh, trimesh.Scene],
        image: Image.Image,
        mode: str = "enhance",
        strength: float = 0.35,
        seed: int = 42,
        resolution: int = 512,
        texture_size: int = 2048,
        repaint_faces: Optional[Dict[str, np.ndarray]] = None,
        preprocess_image: bool = True,
        tex_slat_sampler_params: Optional[dict] = None,
    ) -> Union[trimesh.Trimesh, trimesh.Scene]:
        """
        Args:
            mesh: Path or a loaded mesh. A path is normalized the same way the
                texturing pipeline normalizes vertices, and UVs are kept.
            image: Reference photo. The original generation image works best.
            mode: ``enhance`` lightly restyles the whole texture.
                ``inpaint`` regenerates masked (or untextured) tokens and pins
                the rest to the encoded texture. ``retexture`` ignores the
                current texture and samples from pure noise.
            strength: 0 keeps the encoding, 1 is a full resample. Ignored by
                ``retexture`` (always 1).
            repaint_faces: Optional ``{mesh_name: face_index}`` from
                ``inspect_glb``. Used by ``inpaint``.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode}")
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        if resolution != self.RESOLUTION:
            raise ValueError(
                f"Texture refine is {self.RESOLUTION} only on this pipeline. "
                "The 1024 texture DiT is not loaded; it does not fit a 4 GB card."
            )
        if preprocess_image:
            image = self.preprocess_image(image)

        entries = self._prepare_entries(mesh)
        combined = concat_meshes([item[1] for item in entries])
        print(
            f"[refine] mode={mode} strength={strength} resolution={resolution} "
            f"meshes={len(entries)} faces={len(combined.faces)}"
        )

        torch.manual_seed(seed)
        cond = self.get_cond([image], 512)
        cond = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in cond.items()}
        offload.release_cuda_memory()

        max_voxels = offload.recommend_tex_decode_voxels()
        max_tokens = offload.recommend_lr_tokens()
        voxel_xyz, dual_vertices, intersected = self._voxelize_shape(combined, resolution, max_voxels)
        pbr, textured = self._voxelize_texture(entries, voxel_xyz, resolution)
        if not bool(textured.any()) and mode == "enhance":
            print("[refine] no texture samples on the mesh. Switching enhance -> retexture.")
            mode = "retexture"

        shape_slat = self._encode_shape(voxel_xyz, dual_vertices, intersected)
        self.unload_models("shape_slat_encoder")
        tex_slat = self._encode_texture(voxel_xyz, pbr)
        self.unload_models("tex_slat_encoder")
        offload.release_cuda_memory()

        shape_slat, tex_slat = self._cap_latents(shape_slat, tex_slat, max_tokens)
        tex_feats, found = gather_feats_by_coord(
            tex_slat.coords, tex_slat.feats, shape_slat.coords, fill=0.0,
        )
        missing = int((~found).sum().item()) if found.numel() else 0
        if missing:
            print(f"[refine] {missing} shape tokens had no texture latent; those get filled from neighbors.")
            if bool(found.any()):
                tex_feats[~found] = tex_feats[found].mean(dim=0)
        tex_on_shape = SparseTensor(
            feats=tex_feats.to(dtype=shape_slat.feats.dtype),
            coords=shape_slat.coords,
        )
        tex_on_shape._scale = getattr(shape_slat, "_scale", tex_on_shape._scale)

        guides = subdivision_guides(shape_slat.coords, voxel_xyz, resolution, levels=4)
        alive = guides[0][1].sum(dim=-1) > 0
        if not bool(alive.all()):
            dropped = int((~alive).sum().item())
            print(f"[refine] dropping {dropped} latent tokens that do not sit on the mesh")
            alive_idx = alive.nonzero(as_tuple=False).reshape(-1)
            shape_slat = self._take_rows(shape_slat, alive_idx)
            tex_on_shape = self._take_rows(tex_on_shape, alive_idx)
            guides = subdivision_guides(shape_slat.coords, voxel_xyz, resolution, levels=4)
        if int(shape_slat.coords.shape[0]) == 0:
            raise RuntimeError(
                "no latent tokens left. If the log says they were dropped because they "
                "do not sit on the mesh, the encoder grid and the voxel grid disagreed "
                "(send that log). Otherwise the mesh produced no voxels after the cap."
            )

        repaint_mask = None
        t_start = 1.0
        init = None
        if mode == "retexture":
            print("[refine] retexture: sampling texture from noise, mesh shape kept")
        elif mode == "enhance":
            t_start = strength
            init = tex_on_shape
            print(f"[refine] enhance: denoise the existing texture from t={t_start:.2f}")
        else:
            t_start = strength if strength > 0 else 1.0
            init = tex_on_shape
            repaint_mask = self._inpaint_mask(
                entries, combined, repaint_faces, voxel_xyz, textured, shape_slat.coords, resolution,
            )
            n_repaint = int(repaint_mask.sum().item())
            print(
                f"[refine] inpaint: {n_repaint}/{repaint_mask.shape[0]} tokens "
                f"resampled from t={t_start:.2f}; the rest stay on the encoded texture"
            )
            print("[refine] kept texels are a reconstruction of your texture, not a pixel copy")

        cond = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cond.items()}
        shape_slat = shape_slat.to(self.device)
        if init is not None:
            init = init.to(self.device)
        tex_model = self.models["tex_slat_flow_model_512"]
        tex_out = self.sample_tex_slat(
            cond,
            tex_model,
            shape_slat,
            sampler_params=tex_slat_sampler_params or {},
            init_tex_slat=init,
            t_start=t_start,
            repaint_mask=repaint_mask,
        )
        self.unload_models("tex_slat_flow_model_512")
        offload.release_cuda_memory()

        pbr_voxel = self.decode_tex_slat(tex_out, guide_subs=guides)
        self.unload_models("tex_slat_decoder")
        offload.release_cuda_memory()

        baked = []
        for name, item in entries:
            print(f"[refine] baking {name}")
            baked.append(self.postprocess_mesh(item, pbr_voxel, resolution, texture_size))
            offload.release_cuda_memory()
        if len(baked) == 1:
            return baked[0]
        scene = trimesh.Scene()
        for (name, _), mesh in zip(entries, baked):
            scene.add_geometry(mesh, geom_name=name)
        return scene

    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        init_tex_slat: Optional[SparseTensor] = None,
        t_start: float = 1.0,
        repaint_mask: Optional[torch.Tensor] = None,
    ) -> SparseTensor:
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(shape_slat.device)
        shape_cond = (shape_slat - mean) / std

        tex_std = torch.tensor(self.tex_slat_normalization["std"])[None].to(shape_slat.device)
        tex_mean = torch.tensor(self.tex_slat_normalization["mean"])[None].to(shape_slat.device)

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        tex_channels = in_channels - shape_slat.feats.shape[1]
        if tex_channels < 1:
            raise RuntimeError(
                f"texture flow in_channels={in_channels} is smaller than the shape latent "
                f"({shape_slat.feats.shape[1]})"
            )

        x0 = None
        mask = None
        start = 1.0
        if init_tex_slat is None:
            noise = shape_slat.replace(
                torch.randn(shape_slat.coords.shape[0], tex_channels, device=shape_slat.device)
            )
        else:
            if init_tex_slat.feats.shape[1] != tex_channels:
                raise RuntimeError(
                    f"encoded texture has {init_tex_slat.feats.shape[1]} channels, "
                    f"flow model expects {tex_channels}"
                )
            if init_tex_slat.coords.shape[0] != shape_slat.coords.shape[0]:
                raise RuntimeError("texture latent and shape latent have different token counts")
            x0 = shape_slat.replace(((init_tex_slat.feats - tex_mean) / tex_std).to(dtype=torch.float32))
            noise = x0.replace(torch.randn_like(x0.feats))
            start = float(t_start)
            if repaint_mask is not None:
                mask = repaint_mask.to(device=shape_slat.device)

        params = {**self.tex_slat_sampler_params, **sampler_params}
        with self._model_on_device(flow_model):
            slat = self.tex_slat_sampler.sample(
                flow_model,
                noise,
                concat_cond=shape_cond,
                x_0=x0,
                t_start=start,
                repaint_mask=mask,
                **cond,
                **params,
                verbose=True,
                tqdm_desc="Sampling texture SLat",
            ).samples
        return slat * tex_std + tex_mean

    def decode_tex_slat(self, slat: SparseTensor, guide_subs=None) -> SparseTensor:
        decoder = self.models["tex_slat_decoder"]
        small = offload.recommend_tex_decode_voxels() is not None
        decoder.low_vram = bool(small)
        guides = None
        if guide_subs is not None:
            guides = []
            for coords, feats in guide_subs:
                guide = SparseTensor(feats=feats.float().contiguous(), coords=coords.long().contiguous())
                if not small:
                    guide = guide.to(slat.device)
                guides.append(guide)
        with self._model_on_device(decoder):
            if guides is None:
                ret = decoder(slat)
            else:
                ret = decoder(slat, guide_subs=guides)
        return ret * 0.5 + 0.5

    def _prepare_entries(self, mesh):
        if isinstance(mesh, str):
            entries = load_glb_meshes(mesh)
        elif isinstance(mesh, trimesh.Scene):
            entries = [(str(name), geom) for name, geom in mesh.geometry.items() if len(geom.faces)]
        elif isinstance(mesh, trimesh.Trimesh):
            entries = [("mesh", mesh)]
        else:
            raise TypeError(f"unsupported mesh type {type(mesh)}")
        if not entries:
            raise ValueError("mesh has no faces")
        all_vertices = np.concatenate([np.asarray(item.vertices) for _, item in entries], axis=0)
        center, scale = normalization_frame(all_vertices)
        retargeted = [(name, retarget_mesh(item, center, scale)) for name, item in entries]
        for _, item in retargeted:
            if not (np.all(item.vertices >= -0.5) and np.all(item.vertices <= 0.5)):
                raise RuntimeError("normalized vertices fell outside [-0.5, 0.5]")
        return retargeted

    def _voxelize_shape(self, mesh: trimesh.Trimesh, resolution: int, max_voxels: Optional[int]):
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.faces)).long()
        print(f"[refine] voxelizing shape at {resolution}")
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
        voxel_indices = voxel_indices.detach().cpu().long()
        dual_vertices = dual_vertices.detach().cpu().float()
        intersected = intersected.detach().cpu()
        offload.release_cuda_memory()
        before = int(voxel_indices.shape[0])
        if max_voxels is not None and before > int(max_voxels):
            coords4 = torch.cat([torch.zeros(before, 1, dtype=torch.long), voxel_indices], dim=1)
            keep = row_index_after_cap(coords4, int(max_voxels))
            voxel_indices = voxel_indices[keep]
            dual_vertices = dual_vertices[keep]
            intersected = intersected[keep]
            print(f"[refine] shape voxels {before} -> {int(voxel_indices.shape[0])} (cap {max_voxels})")
        else:
            print(f"[refine] shape voxels {before}")
        return voxel_indices, dual_vertices, intersected

    def _voxelize_texture(self, entries, voxel_xyz: torch.Tensor, resolution: int):
        channels = len(_DEFAULT_PBR_U8)
        default = torch.tensor(_DEFAULT_PBR_U8, dtype=torch.float32)
        raw = default.view(1, channels).expand(voxel_xyz.shape[0], channels).clone()
        found = torch.zeros(voxel_xyz.shape[0], dtype=torch.bool)
        for name, mesh in entries:
            try:
                print(f"[refine] reading textures from {name}")
                coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
                    mesh,
                    grid_size=resolution,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    timing=False,
                )
            except Exception as exc:
                print(f"[refine] could not read textures on {name} ({exc}). Those voxels stay at the default material.")
                continue
            mat_xyz = coord.detach().cpu().long()
            pieces = []
            for key, cols in (("base_color", 3), ("metallic", 1), ("roughness", 1), ("alpha", 1)):
                part = torch.as_tensor(attr[key]).detach().cpu().float().reshape(-1, cols)
                pieces.append(part)
            mat = torch.cat(pieces, dim=-1)
            del coord, attr
            offload.release_cuda_memory()
            gathered, hit = gather_feats_by_coord(mat_xyz, mat, voxel_xyz, fill=default)
            raw[hit] = gathered[hit]
            found |= hit
        encoded = raw / 255.0 * 2.0 - 1.0
        print(f"[refine] texture samples on {int(found.sum().item())}/{voxel_xyz.shape[0]} voxels")
        return encoded, found

    def _encode_shape(self, voxel_xyz: torch.Tensor, dual_vertices: torch.Tensor, intersected: torch.Tensor) -> SparseTensor:
        coords = torch.cat(
            [torch.zeros(voxel_xyz.shape[0], 1, dtype=torch.long), voxel_xyz.long()],
            dim=1,
        )
        feats = dual_vertices.float() * self.RESOLUTION - voxel_xyz.float()
        incoming = SparseTensor(feats=feats.contiguous(), coords=coords.contiguous())
        flags = incoming.replace(intersected.to(dtype=torch.float32).contiguous())
        print(f"[refine] encoding shape ({incoming.coords.shape[0]} voxels)")
        encoder = self.models["shape_slat_encoder"]
        with self._model_on_device(encoder):
            latent = encoder(incoming.to(self.device), flags.to(self.device), sample_posterior=False)
        latent = latent.cpu()
        print(f"[refine] shape latent tokens {latent.coords.shape[0]}")
        return latent

    def _encode_texture(self, voxel_xyz: torch.Tensor, pbr: torch.Tensor) -> SparseTensor:
        coords = torch.cat([torch.zeros(voxel_xyz.shape[0], 1, dtype=torch.long), voxel_xyz.long()], dim=1)
        incoming = SparseTensor(feats=pbr.float().contiguous(), coords=coords.contiguous())
        print(f"[refine] encoding texture ({incoming.coords.shape[0]} voxels)")
        encoder = self.models["tex_slat_encoder"]
        with self._model_on_device(encoder):
            latent = encoder(incoming.to(self.device), sample_posterior=False)
        latent = latent.cpu()
        print(f"[refine] texture latent tokens {latent.coords.shape[0]}")
        return latent

    def _cap_latents(self, shape_slat: SparseTensor, tex_slat: SparseTensor, max_tokens: Optional[int]):
        n = int(shape_slat.coords.shape[0])
        if max_tokens is not None and n > int(max_tokens):
            keep = row_index_after_cap(shape_slat.coords.cpu(), int(max_tokens))
            print(f"[refine] latent tokens {n} -> {int(keep.shape[0])} (cap {max_tokens})")
            shape_slat = self._take_rows(shape_slat, keep)
        else:
            print(f"[refine] latent tokens {n}")
        return shape_slat, tex_slat

    def _inpaint_mask(self, entries, combined, repaint_faces, voxel_xyz, textured, latent_coords, resolution):
        mask = voxel_flags_to_latent_mask(voxel_xyz, ~textured, latent_coords, downsample=self.DOWNSAMPLE)
        if repaint_faces:
            by_name = {name: np.asarray(index, dtype=np.int64) for name, index in repaint_faces.items()}
            running = 0
            pieces = []
            for name, mesh in entries:
                selected = by_name.get(name)
                n = len(mesh.faces)
                if selected is not None and len(selected):
                    if int(selected.min()) < 0 or int(selected.max()) >= n:
                        raise ValueError(f"repaint mask for {name!r} is outside 0..{n - 1}")
                    pieces.append(selected + running)
                running += n
            if pieces:
                face_index = np.concatenate(pieces)
                mask = mask | faces_to_latent_mask(
                    combined.vertices,
                    combined.faces,
                    face_index,
                    latent_coords,
                    resolution,
                    downsample=self.DOWNSAMPLE,
                )
        if not bool(mask.any()):
            raise RuntimeError(
                "inpaint found nothing to change. The mesh looks fully textured. "
                "Pass a mask from inspect_glb.py, or use --mode enhance."
            )
        return mask

    @staticmethod
    def _take_rows(sparse: SparseTensor, index: torch.Tensor) -> SparseTensor:
        index = torch.as_tensor(index, dtype=torch.long, device=sparse.feats.device)
        taken = SparseTensor(
            feats=sparse.feats.index_select(0, index).contiguous(),
            coords=sparse.coords.index_select(0, index).contiguous(),
            shape=sparse.shape,
        )
        taken._scale = getattr(sparse, "_scale", taken._scale)
        return taken
