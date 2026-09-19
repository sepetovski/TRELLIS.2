"""
1536³ cascade attempt — leave example_low_vram.py alone for the proven 512³ path.

Honest limit
------------
A 4 GB RTX 3050 cannot hold the full 1536³ token budget (up to ~49k sparse
tokens in the 1024 DiT). This script still *requests* pipeline_type
'1536_cascade' (512 shape → 1024 DiT aimed at 1536). If the upsampled
token count is too high, TRELLIS.2 lowers resolution 128 at a time until
it fits `max_num_tokens`, but not below `min_hr_resolution` (default 1024).

So you may get 1536, 1408, …, or 1024 — not a silent fall-back to 512.
If you see `device not ready` on "Sampling shape SLat (…)", the 1024 DiT
still overflowed. Lower tokens and retry after `wsl --shutdown`:

    TRELLIS_MAX_TOKENS=8192 python example_1536.py photo.png

Keep using example_low_vram.py when you want the proven 512³ path.

Usage:
    conda activate trellis2
    cd ~/TRELLIS.2
    git fetch fork cursor/low-vram-block-offload-3548
    git checkout cursor/low-vram-block-offload-3548
    git pull fork cursor/low-vram-block-offload-3548
    python example_1536.py yourphoto.png
"""
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, offload
from trellis2.renderers import EnvMap
import o_voxel


IMAGE_PATH = os.environ.get("TRELLIS_IMAGE", "assets/example_image/T.png")
PIPELINE_TYPE = "1536_cascade"
MAX_NUM_TOKENS = int(os.environ.get("TRELLIS_MAX_TOKENS", "12288"))
MIN_HR_RESOLUTION = int(os.environ.get("TRELLIS_MIN_HR", "1024"))
SAMPLER_STEPS = int(os.environ.get("TRELLIS_STEPS", "8"))


def main():
    total_gb = offload.gpu_total_memory_gb()
    avail_ram, total_ram = offload.host_memory_gb()
    small_gpu = bool(total_gb and total_gb < 8)
    texture_size = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "1024" if small_gpu else "2048"))
    print(f"GPU VRAM: {total_gb:.2f} GB  |  pipeline_type={PIPELINE_TYPE}")
    print(
        f"Token budget: {MAX_NUM_TOKENS}  |  min resolution: {MIN_HR_RESOLUTION}  |  "
        f"sampler steps: {SAMPLER_STEPS}  |  GLB texture: {texture_size}"
    )
    if total_ram:
        print(f"WSL RAM:  {avail_ram:.1f} GiB free / {total_ram:.1f} GiB total")
    if small_gpu:
        print(
            "4–8 GB GPU: experimental. example_low_vram.py is still the 512³ script. "
            "This cascade runs the 1024 DiT (the stage that used to crash). "
            "Expect 20–60+ minutes. After device-not-ready: wsl --shutdown first."
        )

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        pipeline_type=PIPELINE_TYPE,
    )
    pipeline.low_vram = True
    pipeline.block_offload = True
    pipeline.cuda()
    offload.ensure_cuda_ready()

    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "  python example_1536.py /home/damjan/TRELLIS.2/myphoto.png\n"
            "Windows files: /mnt/c/Users/<you>/Desktop/myphoto.png"
        )
    print(f"Using image: {os.path.abspath(image_path)}")
    image = Image.open(image_path)

    step = {"steps": SAMPLER_STEPS}
    mesh = pipeline.run(
        image,
        pipeline_type=PIPELINE_TYPE,
        max_num_tokens=MAX_NUM_TOKENS,
        min_hr_resolution=MIN_HR_RESOLUTION,
        sparse_structure_sampler_params=step,
        shape_slat_sampler_params=step,
        tex_slat_sampler_params=step,
    )[0]
    achieved = int(round(1 / mesh.voxel_size))
    print(f"Output voxel size {mesh.voxel_size} (~{achieved}³; requested 1536³)")
    mesh.simplify(16777216)
    offload.release_cuda_memory()

    try:
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device="cuda",
        ))
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave("sample_1536.mp4", video, fps=15)
        print("Wrote sample_1536.mp4")
    except Exception as e:
        print(f"Video render skipped ({e})")

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=150000,
        texture_size=texture_size,
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export("sample_1536.glb", extension_webp=True)
    print("Wrote sample_1536.glb")


if __name__ == "__main__":
    main()
