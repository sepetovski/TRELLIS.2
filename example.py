import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, offload, mesh_utils
from trellis2.utils.bake_limits import RAISED_DECIMATION_TARGET, default_texture_size
from trellis2.utils.hdri import load_latlong_rgb, preview_settings
from trellis2.renderers import EnvMap
import o_voxel


def gpu_gb() -> float:
    return offload.gpu_total_memory_gb()


# Decide the cascade *before* loading weights so unused 1024 DiTs are not
# pulled into RAM. A 4 GB card can run the 512 models (they already completed
# in isolation); the 1024 cascade is what pins VRAM at ~3640/4096 MiB and then
# faults with "CUDA driver error: device not ready".
total_gb = gpu_gb()
low_gpu = total_gb > 0 and total_gb < 8
pipeline_type = "512" if low_gpu else None
if low_gpu:
    print(
        f"Detected {total_gb:.1f} GB GPU. Loading the 512^3 pipeline only, "
        "with sequential transformer-block CPU offload. "
        "Use example_low_vram.py for the same path with extra comments."
    )

# 1. Load Pipeline (keep the HDRI off the GPU until generation finishes)
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS.2-4B",
    pipeline_type=pipeline_type,
)
pipeline.cuda()

# 2. Load Image & Run
image = Image.open("assets/example_image/T.png")
run_kwargs = {}
if pipeline_type is not None:
    run_kwargs["pipeline_type"] = pipeline_type
mesh = pipeline.run(image, **run_kwargs)[0]
mesh.simplify(16777216)  # nvdiffrast limit
offload.release_cuda_memory()

# 3. Export to GLB before the preview video, so the bake gets the free VRAM.
# On a small GPU the shape stays the 512 mesh (remesh off). Texture is 2048;
# 4096 attribute sampling TDRs a 4 GB card.
if low_gpu:
    decimation_target = int(os.environ.get("TRELLIS_DECIMATION_TARGET", str(RAISED_DECIMATION_TARGET)))
    texture_size = int(os.environ.get("TRELLIS_TEXTURE_SIZE", str(default_texture_size(total_gb))))
    mesh_utils.export_textured_glb(
        mesh,
        "sample.glb",
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=False,
    )
else:
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   1000000,
        texture_size        =   4096,
        remesh              =   True,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )
    glb.export("sample.glb", extension_webp=True)
offload.release_cuda_memory()

# 4. Preview video. The GLB above is the result; this only makes sample.mp4.
try:
    hdri = load_latlong_rgb("assets/hdri/forest.exr")
    envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))
    preview_res, preview_frames, preview_ssaa = preview_settings(total_gb)
    video = render_utils.make_pbr_vis_frames(
        render_utils.render_video(
            mesh, envmap=envmap,
            resolution=preview_res, num_frames=preview_frames, ssaa=preview_ssaa,
        ),
        resolution=preview_res,
    )
    imageio.mimsave("sample.mp4", video, fps=15)
except Exception as e:
    print(f"GLB is already saved. Preview video skipped ({e})")
