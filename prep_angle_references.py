#!/usr/bin/env python3
"""
Prepare sharp framing references for the combined-angle turnaround pipeline.
=============================================================================
Given a base standing-pose image, this:
  1. 2x AI-upscales it on Comfy Cloud (4x-UltraSharp model, then 0.5x) so the
     source has real detail.
  2. Crops three framing tiers from the upscaled image (full / medium / close)
     and LANCZOS-*downscales* each to 1024x1024 — a downscale, so the crops stay
     crisp instead of picking up upscaling artifacts.

Why: Qwen Image Edit preserves the input's framing, so we control framing by the
crop and control the camera angle later with the 2511 multi-angle LoRA. Cropping
a tight region out of a 1-megapixel render and upscaling it looks pixelated; this
prep renders/upscales large first and downscales the crops instead.

Recommended base render (Qwen-native ~1.76 MP instead of the Flux ~1 MP cap):
  python batch_multi_angle.py --image ref.png --cloud --pipeline standing_relaxed \
      --megapixels 1.76 --output BASE_DIR

Then:
  python prep_angle_references.py --image BASE_DIR/standing_relaxed.png \
      --output-dir OUT/base

Produces in --output-dir:
  standing_relaxed.png         (full body, 1024)   -> full-body angle views
  standing_relaxed_medium.png  (mid-stomach, 1024) -> medium angle views
  standing_relaxed_close.png   (below-breast, 1024)-> closeup angle views
and standing_relaxed_2x.png    (the 2x upscale, kept as the crop source)

Downstream: render the turnaround with the 2511 LoRA fed the matching reference
per tier (crop = framing, LoRA horizontal/vertical angle = camera).
"""

import argparse
import asyncio
import os

import aiohttp
from PIL import Image
import numpy as np

# Reuse the cloud helpers from the batch renderer.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "batch_multi_angle",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_multi_angle.py"),
)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

UPSCALE_MODEL = "4x-UltraSharp.pth"   # confirmed available on Comfy Cloud

# Crop factors, as a fraction of detected figure height measured from the top of
# the head. Tuned for a stylized (large-head) character; adjust per style.
TIERS = {
    "medium": 0.517,  # bottom edge ~mid-stomach (10% wider framing than 0.47)
    "close":  0.36,   # bottom edge ~just below the bust (10% closer than 0.40)
}


async def ai_upscale_2x(src_path, out_path):
    """2x AI upscale on Comfy Cloud: 4x model, then downscale 0.5x -> net 2x."""
    api_key = os.environ.get("COMFY_CLOUD_API_KEY", "")
    if not api_key:
        raise SystemExit("Set COMFY_CLOUD_API_KEY")
    async with aiohttp.ClientSession(headers={"Accept-Encoding": "gzip, deflate"}) as s:
        img = await B.cloud_upload(s, api_key, src_path)
        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": img}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
            "3": {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            "4": {"class_type": "ImageScaleBy",
                  "inputs": {"image": ["3", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
            "5": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "up2x", "images": ["4", 0]}},
        }
        pid = await B.cloud_submit(s, api_key, wf)
        outs = await B._ws_collect_outputs(api_key, [pid], timeout=240)
        files = outs.get(pid, [])
        if not files:
            raise SystemExit("Upscale produced no output (model unavailable?)")
        await B.cloud_download(s, api_key, files[0]["filename"], out_path)


def crop_tiers(upscaled_path, out_dir):
    """Crop full/medium/close from the upscaled image, LANCZOS-down to 1024."""
    img = Image.open(upscaled_path).convert("RGB")
    a = np.asarray(img); H, W = a.shape[:2]
    ys, xs = np.where(a.min(axis=2) < 235)
    y_top, y_bot = ys.min(), ys.max()
    x_c = int((xs.min() + xs.max()) / 2)
    fig_h = y_bot - y_top

    # full body -> plain downscale to 1024
    img.resize((1024, 1024), Image.LANCZOS).save(
        os.path.join(out_dir, "standing_relaxed.png"))

    for name, frac in TIERS.items():
        top = max(0, int(y_top - 0.04 * fig_h))
        bot = int(y_top + frac * fig_h)
        half = (bot - top) // 2
        left, right = max(0, x_c - half), min(W, x_c + half)
        c = img.crop((left, top, right, bot))
        side = max(c.size)
        sq = Image.new("RGB", (side, side), (255, 255, 255))
        sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
        sq.resize((1024, 1024), Image.LANCZOS).save(
            os.path.join(out_dir, f"standing_relaxed_{name}.png"))
        print(f"  {name}: crop {side}px -> 1024 ({'downscale' if side >= 1024 else 'upscale'})")


def main():
    p = argparse.ArgumentParser(description="Prep sharp full/medium/close angle references")
    p.add_argument("--image", required=True, help="Base standing-pose image (ideally rendered at --megapixels 1.76)")
    p.add_argument("--output-dir", required=True, help="Where to write the references")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    up = os.path.join(args.output_dir, "standing_relaxed_2x.png")
    print("2x AI upscaling on Comfy Cloud...")
    asyncio.run(ai_upscale_2x(args.image, up))
    print(f"  upscaled -> {up} {Image.open(up).size}")
    print("Cropping tiers (LANCZOS to 1024)...")
    crop_tiers(up, args.output_dir)
    print("Done. References written to", args.output_dir)


if __name__ == "__main__":
    main()
