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
  * writes a `<name>_cutout.png` of the subject it actually sent to the model
  * renders the preview without nvdiffrec (a simple light if that package is missing)
    before the preview video, while VRAM is empty. 4096 texture sampling
    TDRs a 4 GB card (`device not ready`); that cannot retry in-process.

Override the bake without editing the file:

    TRELLIS_DECIMATION_TARGET=500000 TRELLIS_TEXTURE_SIZE=2048 python example_low_vram.py photo.png

If WSL still `Killed`s the process, raise the WSL memory cap. In Windows
create/edit `%UserProfile%\\.wslconfig`:

    [wsl2]
    memory=16GB
    swap=8GB

then `wsl --shutdown` in PowerShell and reopen the terminal.

Usage (WSL, after `conda activate trellis2`):
    cd ~/TRELLIS.2
    git fetch fork cursor/raise-glb-export-limits-6656
    git checkout cursor/raise-glb-export-limits-6656
    git pull fork cursor/raise-glb-export-limits-6656
    python example_low_vram.py

The GLB is named after the image (`house.png` → `house.glb`).

Optional live VRAM log:
    TRELLIS_VRAM_LOG=1 python example_low_vram.py
"""
import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, offload, mesh_utils
from trellis2.utils.bake_limits import RAISED_DECIMATION_TARGET, default_texture_size
from trellis2.utils.hdri import load_latlong_rgb, preview_settings
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
            "A photo (house, person, …) occupies more voxels than T.png. "
            "This build caps tokens on 4 GB so shape-SLat does not TDR."
        )
        print(
            "GLB bake on 4 GB is 1,000,000 faces / 2048 texture. "
            "4096 texture sampling TDRs this card."
        )
        print(
            "A dense shell (toilet, house) reaches ~2M voxels at the last "
            "texture upsample. That conv is capped at 1,500,000 so it does not TDR."
        )

    decimation_target = int(
        os.environ.get("TRELLIS_DECIMATION_TARGET", str(RAISED_DECIMATION_TARGET))
    )
    texture_size = int(
        os.environ.get("TRELLIS_TEXTURE_SIZE", str(default_texture_size(total_gb)))
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
            "  python example_low_vram.py /home/damjan/TRELLIS.2/myphoto.png\n"
            "Windows files are under /mnt/c/Users/<you>/..."
        )
    print(f"Using image: {os.path.abspath(image_path)}")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    pipeline.cutout_save_path = os.path.abspath(f"{stem}_cutout.png")
    image = Image.open(image_path)
    mesh = pipeline.run(image, pipeline_type=PIPELINE_TYPE)[0]
    mesh.simplify(16777216)
    offload.release_cuda_memory()

    mp4_name = f"{stem}.mp4"
    glb_name = f"{stem}.glb"

    # Bake while the card is empty. The preview video is optional and comes after.
    print(
        f"GLB bake request: decimation_target={decimation_target}, "
        f"texture_size={texture_size}, remesh=False"
    )
    mesh_utils.export_textured_glb(
        mesh,
        glb_name,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=False,
    )
    offload.release_cuda_memory()

    try:
        # forest.exr is DWAB. OpenCV 5 imread returns empty and cvtColor crashes.
        hdri = load_latlong_rgb("assets/hdri/forest.exr")
        envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))
        preview_res, preview_frames, preview_ssaa = preview_settings(total_gb)
        print(
            f"Rendering preview {mp4_name}: {preview_res}px, {preview_frames} frames. "
            "The GLB is already written."
        )
        video = render_utils.make_pbr_vis_frames(
            render_utils.render_video(
                mesh,
                envmap=envmap,
                resolution=preview_res,
                num_frames=preview_frames,
                ssaa=preview_ssaa,
            ),
            resolution=preview_res,
        )
        imageio.mimsave(mp4_name, video, fps=15)
        print(f"Wrote {mp4_name}")
    except Exception as e:
        print(f"GLB is already saved. Preview video skipped ({e})")


if __name__ == "__main__":
    main()
