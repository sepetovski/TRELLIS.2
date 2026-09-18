import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, offload
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

# 3. Setup Environment Map (only needed for visualization)
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 4. Render Video
try:
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    imageio.mimsave("sample.mp4", video, fps=15)
except Exception as e:
    print(f"Video render skipped ({e})")

# 5. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   100000 if low_gpu else 1000000,
    texture_size        =   1024 if low_gpu else 4096,
    remesh              =   not low_gpu,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)
