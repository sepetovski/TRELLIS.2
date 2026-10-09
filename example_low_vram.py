"""
4 GB GPU entry point for TRELLIS.2 (RTX 3050 4 GB laptop, Windows + WSL).

What this runs
--------------
512³ by default. That is the reliable setting on 4 GB and it is the right
amount of detail for a chunky low-poly game asset. 1024³ is opt-in and slow:

    TRELLIS_PIPELINE_TYPE=1024_cascade python example_low_vram.py chest.png

TRELLIS does not read a text prompt. Generate the picture first (transparent
PNG, one object, centered, no ground, no text), then pass that file here.
A real alpha channel is used as-is. A flat black or white backdrop is cut out
without the background-removal network.

Why a 4 GB card dies
--------------------
The weights of one 1.3B DiT are 2.6 GB. Loading the whole DiT, then running
classifier-free guidance twice, fills the card. Kernels stall, and Windows
resets the GPU (`CUDA driver error: device not ready`). After that, CUDA is
dead until PowerShell: wsl --shutdown.

This script keeps one transformer block on the GPU, runs guidance one pass at
a time, chunks the MLP, and tiles the big sparse convs in the VAE. Quality
knobs that used to be turned down to survive (2×2×2 occupancy binning, skipping
the learned upsampler, 4096² texture bakes) are not the default anymore.

Before the first run, on Windows
--------------------------------
1. `%UserProfile%\\.wslconfig` (then `wsl --shutdown`):

       [wsl2]
       memory=16GB
       swap=32GB

   Use less memory only if the PC has less than 16 GB of RAM. A bare `Killed`
   with no Python traceback is WSL running out of system RAM.

2. Elevated PowerShell, then reboot:

       powershell -ExecutionPolicy Bypass -File scripts/windows_tdr.ps1

   That sets TdrDelay=60 so a long kernel is not killed. It does not add VRAM.
   It matters once the card is no longer full: tiled convs and PCIe weight
   streaming are slow on purpose.

Usage (WSL, after `conda activate trellis2`):

    python example_low_vram.py assets/example_image/T.png
    python example_low_vram.py /mnt/c/Users/<you>/Pictures/chest.png

The GLB is named after the image. Useful overrides:

    TRELLIS_LR_TOKENS=8192      512 occupancy budget (training max; interior-first)
    TRELLIS_MAX_TOKENS=12288    1024 DiT budget
    TRELLIS_PIPELINE_TYPE=1024_cascade
    TRELLIS_TEXTURE_SIZE=1024   2048 is sharper and may TDR; try it after step 2
    TRELLIS_DECIMATION_TARGET=100000
    TRELLIS_STEPS=12            official step count; 24 is slower and a bit cleaner
    TRELLIS_VRAM_LOG=1
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
from trellis2.utils import render_utils, offload, mesh_utils
from trellis2.renderers import EnvMap


IMAGE_PATH = os.environ.get("TRELLIS_IMAGE", "assets/example_image/T.png")
PIPELINE_TYPE = os.environ.get("TRELLIS_PIPELINE_TYPE", "512")


def main():
    total_gb = offload.gpu_total_memory_gb()
    avail_ram, total_ram = offload.host_memory_gb()
    print(f"GPU VRAM: {total_gb:.2f} GB  |  pipeline_type={PIPELINE_TYPE}")
    if total_ram:
        print(f"WSL RAM:  {avail_ram:.1f} GiB free / {total_ram:.1f} GiB total")
        if total_ram < 10:
            print(
                "WSL RAM cap is still under 10 GiB. Set memory=12GB (or 16GB) "
                "in %UserProfile%\\.wslconfig, then PowerShell: wsl --shutdown"
            )
        elif total_ram < 13:
            print("WSL RAM cap ~12 GiB is OK for the 512 pipeline (one model at a time).")
    if total_gb and total_gb < 6:
        print(
            "Expect several minutes per stage: each 1.3B DiT streams 30 "
            "blocks over PCIe instead of sitting in VRAM."
        )
        print(
            "Shape sampling keeps up to 8192 voxels (the training budget) and "
            "parks one guidance pass on CPU. 1024³: "
            "TRELLIS_PIPELINE_TYPE=1024_cascade. After `device not ready`, "
            "PowerShell: wsl --shutdown — then retry."
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
            "Copy a PNG/JPG into WSL, then run:\n"
            "  python example_low_vram.py /mnt/c/Users/<you>/Pictures/chest.png\n"
            "Windows files are under /mnt/c/Users/<you>/..."
        )
    print(f"Using image: {os.path.abspath(image_path)}")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.environ.setdefault("TRELLIS_SAVE_PREPROCESS", f"{stem}.preprocessed.png")
    image = Image.open(image_path)
    steps = os.environ.get("TRELLIS_STEPS", "").strip()
    run_kwargs = {}
    if steps:
        n_steps = int(steps)
        run_kwargs = {
            "sparse_structure_sampler_params": {"steps": n_steps},
            "shape_slat_sampler_params": {"steps": n_steps},
            "tex_slat_sampler_params": {"steps": n_steps},
        }
    mesh = pipeline.run(image, pipeline_type=PIPELINE_TYPE, **run_kwargs)[0]
    mesh.simplify(16777216)
    offload.release_cuda_memory()

    mp4_name = f"{stem}.mp4"
    glb_name = f"{stem}.glb"

    # Bake while the card is empty. The preview video is optional and comes after.
    # 1024² texture is the safe bake on 4 GB. 2048 is sharper; set
    # TRELLIS_TEXTURE_SIZE=2048 after TdrDelay is raised if a bake resets the GPU.
    small_gpu = bool(total_gb and total_gb < 8)
    texture_size = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "1024" if small_gpu or not total_gb else "2048"))
    decimation_target = int(os.environ.get("TRELLIS_DECIMATION_TARGET", "100000"))
    print(f"GLB bake: {decimation_target} faces, {texture_size}² texture")
    mesh_utils.export_pbr_glb(
        mesh,
        glb_name,
        texture_size=texture_size,
        decimation_target=decimation_target,
        remesh=False,
    )
    offload.release_cuda_memory()

    try:
        hdri = cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED)
        if hdri is None:
            raise FileNotFoundError("assets/hdri/forest.exr")
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device="cuda",
        ))
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(mp4_name, video, fps=15)
        print(f"Wrote {mp4_name}")
    except Exception as e:
        print(f"GLB is already saved. Preview video skipped ({e})")


if __name__ == "__main__":
    main()
