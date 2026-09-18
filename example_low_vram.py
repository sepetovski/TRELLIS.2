"""
4–8 GB GPU entry point for TRELLIS.2 (e.g. RTX 3050 4 GB laptop in WSL).

Why the original example crashes
--------------------------------
TRELLIS.2 is *not* one 4B model. It is a cascade of independently trained
pieces, each ~1.3B:

  1. image cond (DINOv3) + background removal
  2. sparse-structure DiT + decoder
  3. shape SLat DiT @ 512
  4. (cascade) shape SLat DiT @ 1024   <-- crash site on 4 GB
  5. texture SLat DiT
  6. shape / texture VAEs, then mesh export

`low_vram=True` (the default) already keeps unused *modules* on CPU. That is
why sparse structure and the first shape-SLat pass finish. The second
"Sampling shape SLat" bar is the 1024 DiT: ~2.6 GB of bf16 weights plus MLP
activations (GELU at 8192-d) over tens of thousands of tokens. nvidia-smi
pins at ~3640/4096 MiB, kernels slow to ~15 s/it, then Windows/WDDM faults
the context (`CUDA driver error: device not ready`). Raising TdrDelay does
not fix that — it is a hard memory limit, not a timeout.

This script:
  * loads only the 512 checkpoints (skips 1024 DiTs)
  * streams transformer blocks CPU ↔ GPU one layer at a time
  * deletes each finished 1.3B DiT from RAM before the next stage
    (a bare `Killed` with no CUDA traceback is the WSL OOM killer)
  * does not keep the HDRI on the GPU during generation
  * uses a smaller GLB export so postprocess does not OOM

If WSL still `Killed`s the process, raise the WSL memory cap. In Windows
create/edit `%UserProfile%\\.wslconfig`:

    [wsl2]
    memory=16GB
    swap=8GB

then `wsl --shutdown` in PowerShell and reopen the terminal.

Usage (WSL, after `conda activate trellis2`):
    cd ~/TRELLIS.2
    git fetch fork cursor/low-vram-block-offload-3548
    git checkout cursor/low-vram-block-offload-3548
    python example_low_vram.py

Optional live VRAM log:
    TRELLIS_VRAM_LOG=1 python example_low_vram.py
"""
import os
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

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        pipeline_type=PIPELINE_TYPE,
    )
    pipeline.low_vram = True
    pipeline.block_offload = True
    pipeline.cuda()

    image = Image.open(IMAGE_PATH)
    mesh = pipeline.run(image, pipeline_type=PIPELINE_TYPE)[0]
    mesh.simplify(16777216)
    offload.release_cuda_memory()

    try:
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device="cuda",
        ))
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave("sample.mp4", video, fps=15)
        print("Wrote sample.mp4")
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
        decimation_target=100000,
        texture_size=1024,
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export("sample.glb", extension_webp=True)
    print("Wrote sample.glb")


if __name__ == "__main__":
    main()
