"""
Refine the texture of an existing GLB on a 4 GB GPU.

This does not rebuild the mesh and it does not load the 1024 DiT or the shape
decoder. Peak VRAM is one stage at a time: a 709 MB encoder, then the same
texture DiT you already run, then the texture decoder.

  enhance     keep the current texture, denoise it part-way (default strength 0.35)
  inpaint     regenerate only flat / missing / masked faces; pin the rest
  retexture   ignore the current texture and sample a new one from the image

The original photo is the right --image. Outside an inpaint mask, texels are a
reconstruction of the current texture, not a bit-exact copy.

Usage (WSL, after `conda activate trellis2`):

    cd ~/TRELLIS.2
    git fetch fork cursor/glb-texture-refine-b2f6
    git checkout cursor/glb-texture-refine-b2f6
    git pull fork cursor/glb-texture-refine-b2f6

    # CPU only. Safe to run before the GPU pass. Writes a report and a face mask.
    python inspect_glb.py knight.glb

    # GPU. --mode auto follows the inspector.
    python example_refine.py knight.glb --image photo.png --mode auto

    # Or pick the pass yourself.
    python example_refine.py knight.glb --image photo.png --mode enhance --strength 0.35
    python example_refine.py knight.glb --image photo.png --mode inpaint --mask repaint_faces.npz
    python example_refine.py knight.glb --image photo.png --mode retexture

Strength 0 with --mode enhance copies the file and does not load the model.
4096 textures TDR a 4 GB card; the bake stays at 2048 unless you override it:

    TRELLIS_TEXTURE_SIZE=1024 python example_refine.py knight.glb --image photo.png

After `device not ready`, run `wsl --shutdown` in PowerShell before the next try.
"""
import argparse
import os
import shutil
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from PIL import Image

from trellis2.utils.bake_limits import default_texture_size
from trellis2.utils.glb_inspect import format_report, inspect_glb, read_mask


def _parse_args():
    parser = argparse.ArgumentParser(description="Refine a GLB's texture with TRELLIS.2 on a 4 GB GPU.")
    parser.add_argument("glb", help="Existing .glb/.gltf/.obj/.ply")
    parser.add_argument("--image", help="Reference image. Required unless --dry-run or strength 0.")
    parser.add_argument("--mode", default="auto", choices=["auto", "enhance", "inpaint", "retexture"])
    parser.add_argument("--strength", type=float, default=None, help="0 = no change, 1 = full resample")
    parser.add_argument("--mask", help="repaint_faces.npz from inspect_glb.py (inpaint)")
    parser.add_argument("--output", help="Output GLB path. Default: <name>_refined.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Full-schedule steps (default 12). Suffix depends on strength.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only. Does not load the model.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not os.path.isfile(args.glb):
        raise FileNotFoundError(args.glb)

    report = inspect_glb(args.glb)
    print(format_report(report))
    summary = report["summary"]
    mode = summary["recommendation"] if args.mode == "auto" else args.mode
    strength = summary["suggested_strength"] if args.strength is None else args.strength
    if args.mode == "auto":
        print(f"[refine] auto selected {mode} at strength {strength}")
    else:
        print(f"[refine] using {mode} at strength {strength}")
    if summary["geometry_flagged"]:
        print("[refine] geometry issues stay as they are. This pass only changes texture.")

    stem = os.path.splitext(os.path.basename(args.glb))[0]
    output = args.output or f"{stem}_refined.glb"
    if args.dry_run:
        print(f"[refine] dry run. Would write {output}")
        return
    if mode == "enhance" and strength <= 0:
        shutil.copyfile(args.glb, output)
        print(f"[refine] strength 0. Copied to {output} without loading the model.")
        return
    if not args.image:
        raise SystemExit("pass --image (the photo the mesh should match)")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    import torch
    from trellis2.pipelines import Trellis2RefinePipeline
    from trellis2.utils import offload

    total_gb = offload.gpu_total_memory_gb()
    avail_ram, total_ram = offload.host_memory_gb()
    print(f"GPU VRAM: {total_gb:.2f} GB")
    if total_ram:
        print(f"WSL RAM:  {avail_ram:.1f} GiB free / {total_ram:.1f} GiB total")
    if total_gb and total_gb < 6:
        print("4 GB path: 512 texture DiT, streamed blocks, one stage in RAM at a time.")
    texture_size = int(os.environ.get("TRELLIS_TEXTURE_SIZE", str(default_texture_size(total_gb or 4))))
    print(f"[refine] texture bake {texture_size}")

    pipeline = Trellis2RefinePipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = True
    pipeline.block_offload = True
    pipeline.cuda()
    offload.ensure_cuda_ready()

    image = Image.open(args.image)
    mask = read_mask(args.mask) if args.mask else None
    sampler_params = {}
    if args.steps is not None:
        sampler_params["steps"] = int(args.steps)
    result = pipeline.run(
        args.glb,
        image,
        mode=mode,
        strength=strength,
        seed=args.seed,
        resolution=512,
        texture_size=texture_size,
        repaint_faces=mask,
        tex_slat_sampler_params=sampler_params,
    )
    result.export(output, extension_webp=True)
    print(f"[refine] wrote {os.path.abspath(output)}")
    del result
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        text = str(exc).lower()
        if "device not ready" in text or "cuda driver error" in text:
            print("CUDA context died. In PowerShell run: wsl --shutdown")
            print("Then reopen WSL and run the command again.")
        raise
