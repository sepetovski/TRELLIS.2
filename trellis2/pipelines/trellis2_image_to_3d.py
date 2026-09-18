from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel
from ..utils import offload


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode (one submodule at a time).
        block_offload (bool or 'auto'): If True, stream transformer blocks
            CPU↔GPU one layer at a time. 'auto' enables this on GPUs ≤12 GB.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        block_offload: Union[bool, str] = "auto",
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.block_offload = block_offload
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def models_for_pipeline_type(cls, pipeline_type: str) -> List[str]:
        return offload.models_for_pipeline_type(pipeline_type)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
        pipeline_type: Optional[str] = None,
    ) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            pipeline_type: If set, only the checkpoints required for that cascade
                are loaded (skips unused 1024/512 DiTs to save RAM).
        """
        models_to_load = None
        if pipeline_type is not None:
            models_to_load = cls.models_for_pipeline_type(pipeline_type)
        pipeline = super().from_pretrained(path, config_file, models_to_load=models_to_load)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        # Do not load DINOv3 / rembg until they are used — together with the
        # DiTs they blow past 12 GB WSL RAM.
        pipeline._image_cond_spec = args['image_cond_model']
        pipeline._rembg_spec = args['rembg_model']
        pipeline.image_cond_model = None
        pipeline.rembg_model = None
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.block_offload = args.get('block_offload', 'auto')
        pipeline.default_pipeline_type = pipeline_type or args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        self._resolve_block_offload()
        if not self.low_vram:
            super().to(device)
            self._ensure_image_cond()
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)
            self._ensure_rembg()
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def _ensure_rembg(self):
        if self.rembg_model is None:
            spec = getattr(self, "_rembg_spec", None)
            if spec is None:
                return
            print("[TRELLIS.2] Loading rembg into RAM...")
            self.rembg_model = getattr(rembg, spec["name"])(**spec["args"])
            offload.log_host_memory("loaded rembg")

    def _ensure_image_cond(self):
        if self.image_cond_model is None:
            spec = getattr(self, "_image_cond_spec", None)
            if spec is None:
                return
            print("[TRELLIS.2] Loading image encoder (DINOv3) into RAM...")
            self.image_cond_model = getattr(image_feature_extractor, spec["name"])(**spec["args"])
            offload.log_host_memory("loaded image cond")

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            self._ensure_rembg()
            # BiRefNet at 1024² does not fit a 4 GB card. Keep it on CPU.
            if self.low_vram:
                print("[TRELLIS.2] Removing background on CPU (slow, but safe on 4 GB).")
                self.rembg_model.cpu()
                output = self.rembg_model(input)
            else:
                with self._model_on_device(self.rembg_model):
                    output = self.rembg_model(input)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self._ensure_image_cond()
        self.image_cond_model.image_size = resolution
        with self._model_on_device(self.image_cond_model):
            cond = self.image_cond_model(image)
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        with self._model_on_device(flow_model):
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling sparse structure",
            ).samples
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        with self._model_on_device(decoder):
            decoded = decoder(z_s)>0
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        with self._model_on_device(flow_model):
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling shape SLat",
            ).samples

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        min_hr_resolution: int = 1024,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        with self._model_on_device(flow_model_lr):
            slat = self.shape_slat_sampler.sample(
                flow_model_lr,
                noise,
                **lr_cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling shape SLat",
            ).samples
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        decoder = self.models['shape_slat_decoder']
        with self._model_on_device(decoder):
            decoder.low_vram = True
            hr_coords = decoder.upsample(slat, upsample_times=4)
            decoder.low_vram = False
        del slat
        offload.release_cuda_memory()
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens <= max_num_tokens or hr_resolution <= min_hr_resolution:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        with self._model_on_device(flow_model):
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling shape SLat",
            ).samples

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        decoder = self.models['shape_slat_decoder']
        with self._model_on_device(decoder):
            decoder.low_vram = True
            ret = decoder(slat, return_subs=True)
            decoder.low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        with self._model_on_device(flow_model):
            slat = self.tex_slat_sampler.sample(
                flow_model,
                noise,
                concat_cond=shape_slat,
                **cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling texture SLat",
            ).samples

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        with self._model_on_device(self.models['tex_slat_decoder']):
            ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        parked_tex = None
        if self.low_vram and hasattr(tex_slat, "cpu"):
            parked_tex = tex_slat.cpu()
            tex_slat = parked_tex
            offload.release_cuda_memory()
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        if self.low_vram:
            self.drop_models("shape_slat_decoder")
            if parked_tex is not None:
                tex_slat = parked_tex.to(self.device)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        if self.low_vram:
            self.drop_models("tex_slat_decoder")
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        min_hr_resolution: Optional[int] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            min_hr_resolution (int): Floor resolution for cascade upsampling. Defaults
                to 512 on GPUs under 8 GB so token count can actually be capped.
        """
        requested_type = pipeline_type
        pipeline_type = pipeline_type or self.default_pipeline_type
        gpu_gb = offload.gpu_total_memory_gb()
        if requested_type is None:
            auto_type = offload.recommend_pipeline_type()
            if auto_type == '512' and pipeline_type != '512' and 'shape_slat_flow_model_512' in self.models:
                print(
                    f"[TRELLIS.2] GPU has {gpu_gb:.1f} GB VRAM. Using pipeline_type='512' "
                    "(the 1024 cascade is what OOM/TDR-kills 4 GB cards). "
                    "Pass pipeline_type='1024_cascade' to override."
                )
                pipeline_type = '512'
        if min_hr_resolution is None:
            min_hr_resolution = 512 if gpu_gb and gpu_gb < 8 else 1024
        if gpu_gb and gpu_gb < 8 and max_num_tokens > 8192:
            max_num_tokens = 8192
            print(f"[TRELLIS.2] Capping max_num_tokens to {max_num_tokens} for <8 GB VRAM.")
        # Check pipeline type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        if self.low_vram:
            offload.drop_module(self.rembg_model)
            self.rembg_model = None
            offload.log_host_memory("after rembg")
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        if self.low_vram:
            offload.drop_module(self.image_cond_model)
            self.image_cond_model = None
            offload.log_host_memory("after image cond")
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if self.low_vram:
            self.drop_models("sparse_structure_flow_model", "sparse_structure_decoder")
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            if self.low_vram:
                self.drop_models("shape_slat_flow_model_512")
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            if self.low_vram:
                self.drop_models("tex_slat_flow_model_512")
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            if self.low_vram:
                self.drop_models("shape_slat_flow_model_1024")
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            if self.low_vram:
                self.drop_models("tex_slat_flow_model_1024")
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                min_hr_resolution,
            )
            if self.low_vram:
                self.drop_models("shape_slat_flow_model_512", "shape_slat_flow_model_1024")
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            if self.low_vram:
                self.drop_models("tex_slat_flow_model_1024")
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                min_hr_resolution,
            )
            if self.low_vram:
                self.drop_models("shape_slat_flow_model_512", "shape_slat_flow_model_1024")
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            if self.low_vram:
                self.drop_models("tex_slat_flow_model_1024")
        if self.low_vram:
            del cond_512, cond_1024, coords
            print("[TRELLIS.2] Sampling finished; flow models freed before VAE decode.")
            offload.log_host_memory("before decode")
        offload.release_cuda_memory()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
