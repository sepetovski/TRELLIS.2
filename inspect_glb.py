"""Inspect a GLB for texture-refine. CPU only, no checkpoint.

    python inspect_glb.py knight.glb
    python inspect_glb.py knight.glb --report knight_report.json --mask repaint_faces.npz
"""
import argparse
import os

from trellis2.utils.glb_inspect import format_report, inspect_glb, report_to_json, write_mask


def main():
    parser = argparse.ArgumentParser(description="Report geometry and texture issues in a GLB.")
    parser.add_argument("glb")
    parser.add_argument("--report", help="Write a JSON report (face lists stay in the mask file).")
    parser.add_argument("--mask", help="Write repaint_faces.npz for example_refine.py --mask.")
    args = parser.parse_args()
    if not os.path.isfile(args.glb):
        raise FileNotFoundError(args.glb)
    report = inspect_glb(args.glb)
    print(format_report(report))
    stem = os.path.splitext(os.path.basename(args.glb))[0]
    report_path = args.report or f"{stem}_report.json"
    mask_path = args.mask or "repaint_faces.npz"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_to_json(report))
        handle.write("\n")
    write_mask(mask_path, report)
    print(f"Wrote {report_path}")
    print(f"Wrote {mask_path}")


if __name__ == "__main__":
    main()
