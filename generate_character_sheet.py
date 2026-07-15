#!/usr/bin/env python3
"""
Generate Character Sheet — End-to-End Pipeline
================================================
Fully automated, single-command character sheet:

  base normalize (Qwen-native res)  ->  2x AI upscale + tiered crops
    ->  combined-angle turnaround (crop = framing, 2511 LoRA = camera angle)
    ->  poses / expressions / outfits / lighting
    ->  PowerPoint

Designed to run unattended: the multi-angle views auto-retry (new seed) if the
LoRA drifts to a non-white/scene background, so you can batch several characters
and come back to finished sheets.

Usage:
  python generate_character_sheet.py --image photo.png --name "Nora" --desc "A curious adventurer"
  python generate_character_sheet.py --image photo.png --name "Nora" --skip expressions lighting
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# Combined-angle turnaround: (framing tier, azimuth, elevation, distance, output name).
# Framing comes from the pre-cropped reference for each tier; the 2511 LoRA sets
# the camera angle. Elevations: -30 low, 0 eye, +30 high.
COMBO_VIEWS = [
    ("full",     0,   0, 1.8, "1_full_front"),
    ("full",    45,   0, 1.8, "2_full_3q_front"),
    ("full",    90,   0, 1.8, "3_full_side"),
    ("full",   180,   0, 1.8, "4_full_back"),
    ("medium", 225,   0, 1.0, "5_medium_3q_back"),
    ("medium",  45, -30, 1.0, "6_medium_3q_front"),
    ("close",    0,   0, 0.6, "7_close_front"),
    ("close",   45,  30, 0.6, "8_close_3q_down"),
]
TIER_REF = {
    "full":   "standing_relaxed.png",
    "medium": "standing_relaxed_medium.png",
    "close":  "standing_relaxed_close.png",
}


def run_pass(image, pipeline, seed, concurrency, output_dir, get_pose=False, extra_args=None, quiet=False):
    """Run a single batch_multi_angle.py pass. Returns True on success."""
    cmd = [
        sys.executable, "batch_multi_angle.py",
        "--image", image, "--cloud",
        "--pipeline", pipeline,
        "--seed", str(seed),
        "--concurrency", str(concurrency),
        "--output", output_dir,
    ]
    if get_pose:
        cmd.append("--get-pose")
    if extra_args:
        cmd.extend(extra_args)
    if not quiet:
        print(f"\n{'='*60}\n  Pass: {pipeline}  ->  {output_dir}\n{'='*60}")
    out = subprocess.DEVNULL if quiet else None
    return subprocess.run(cmd, cwd=HERE, stdout=out, stderr=out).returncode == 0


def is_white_bg(path, thresh=238):
    """True if all four corners are (near-)white — used to reject scene drift."""
    try:
        a = np.asarray(Image.open(path).convert("RGB"))
        return all(min(a[y, x]) > thresh for y, x in [(0, 0), (0, -1), (-1, 0), (-1, -1)])
    except Exception:
        return False


def run_prep(base_image, out_dir):
    """2x AI upscale + tiered crops -> full/medium/close references in out_dir."""
    print(f"\n{'='*60}\n  Prep: 2x upscale + tiered crops\n{'='*60}")
    ok = subprocess.run(
        [sys.executable, "prep_angle_references.py", "--image", base_image, "--output-dir", out_dir],
        cwd=HERE,
    ).returncode == 0
    return ok and all(os.path.exists(os.path.join(out_dir, r)) for r in TIER_REF.values())


def run_combo_angles(prep_dir, out_dir, seed, concurrency, retries=3):
    """Render the 8-view turnaround, retrying non-white (scene-drift) views."""
    print(f"\n{'='*60}\n  Pass: combined-angle turnaround\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)
    all_ok = True
    for tier, az, el, dist, name in COMBO_VIEWS:
        ref = os.path.join(prep_dir, TIER_REF[tier])
        dest = os.path.join(out_dir, f"{name}.png")
        last = None
        for attempt in range(retries):
            s = seed + attempt * 1000
            tmp = os.path.join(out_dir, f"_tmp_{name}")
            shutil.rmtree(tmp, ignore_errors=True)
            run_pass(ref, "2511", s, concurrency, tmp,
                     extra_args=["--azimuths", str(az), f"--elevations={el}", "--distances", str(dist)],
                     quiet=True)
            outs = glob.glob(os.path.join(tmp, "*.png"))
            if outs:
                last = outs[0] + f".keep{attempt}"
                shutil.copy(outs[0], last)
                if is_white_bg(outs[0]):
                    shutil.copy(outs[0], dest)
                    shutil.rmtree(tmp, ignore_errors=True)
                    print(f"    ✓ {name}")
                    last = None
                    break
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"    … {name}: attempt {attempt+1} {'drifted (non-white)' if outs else 'failed'}, retrying")
        else:
            # Never got a clean white bg — keep the last render so the sheet is complete.
            if last and os.path.exists(last):
                shutil.copy(last, dest)
                print(f"    ⚠ {name}: kept best-effort render (bg not fully white)")
            else:
                print(f"    ✗ {name}: no render")
                all_ok = False
        for k in glob.glob(os.path.join(out_dir, "*.keep*")):
            os.remove(k)
    return all_ok


def run_presentation(image, name, desc, angles_dir, expressions_dir, outfits_dir, lighting_dir, poses_dir, out_pptx):
    cmd = [
        sys.executable, "make_presentation.py",
        "--image", image, "--name", name, "--desc", desc,
        "--output-dir", angles_dir, "--output", out_pptx,
    ]
    for flag, d in [("--poses-dir", poses_dir), ("--expressions-dir", expressions_dir),
                    ("--outfits-dir", outfits_dir), ("--lighting-dir", lighting_dir)]:
        if d and os.path.isdir(d):
            cmd.extend([flag, d])
    print(f"\n{'='*60}\n  Generating Presentation\n{'='*60}")
    return subprocess.run(cmd, cwd=HERE).returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete character sheet (combined-angle turnaround + poses/expressions/outfits/lighting + deck)")
    parser.add_argument("--image", required=True, help="Reference character image")
    parser.add_argument("--name", required=True, help="Character name")
    parser.add_argument("--desc", default="", help="Character description")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel cloud jobs")
    parser.add_argument("--output", default=None, help="Base output directory (default: ./charsheet_{name})")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["base", "angles", "poses", "expressions", "outfits", "lighting", "presentation"],
                        help="Skip specific passes")
    args = parser.parse_args()

    name_slug = args.name.lower().replace(" ", "_")
    base_dir = args.output or f"./charsheet_{name_slug}"
    os.makedirs(base_dir, exist_ok=True)

    base_pose_dir = os.path.join(base_dir, "base")
    angles_dir = os.path.join(base_dir, "angles_combo")
    poses_dir = os.path.join(base_dir, "poses_prompt")
    expressions_dir = os.path.join(base_dir, "expressions")
    outfits_dir = os.path.join(base_dir, "outfits")
    lighting_dir = os.path.join(base_dir, "lighting")

    start = time.time()
    results = {}

    # Pass 0: normalize to a front standing pose at Qwen-native res, then prep the
    # sharp full/medium/close framing references (2x upscale + tiered crops).
    source_image = args.image
    have_refs = False
    if "base" not in args.skip:
        ok = run_pass(args.image, "standing_relaxed", args.seed, args.concurrency,
                      base_pose_dir, extra_args=["--megapixels", "1.76", "--force"])
        base_candidate = os.path.join(base_pose_dir, "standing_relaxed.png")
        if ok and os.path.exists(base_candidate):
            have_refs = run_prep(base_candidate, base_pose_dir)
            source_image = base_candidate  # now the sharp 1024 full reference
            results["base"] = have_refs
        else:
            print("WARNING: base pose failed — using original input, skipping combined angles")
            results["base"] = False
    else:
        have_refs = all(os.path.exists(os.path.join(base_pose_dir, r)) for r in TIER_REF.values())
        print("Skipping: base" + ("" if have_refs else " (no prep refs — combined angles unavailable)"))

    # Pass 1: combined-angle turnaround (needs the prepped references).
    if "angles" not in args.skip:
        if have_refs:
            results["angles"] = run_combo_angles(base_pose_dir, angles_dir, args.seed, args.concurrency)
        else:
            print("Skipping: angles (prep references unavailable)")
    else:
        print("Skipping: angles")

    # Pass 2: Poses (16 base action poses) — from the normalized base.
    if "poses" not in args.skip:
        results["poses"] = run_pass(source_image, "poses_prompt", args.seed, args.concurrency,
                                    poses_dir, extra_args=["--poses", "base"])
    else:
        print("Skipping: poses")

    # Pass 3: Expressions — from the normalized base.
    if "expressions" not in args.skip:
        results["expressions"] = run_pass(source_image, "expressions", args.seed, args.concurrency,
                                          expressions_dir)
    else:
        print("Skipping: expressions")

    # Pass 4 & 5: Outfits and Lighting — from the ORIGINAL input (hero pose).
    if "outfits" not in args.skip:
        results["outfits"] = run_pass(args.image, "outfits", args.seed, args.concurrency, outfits_dir)
    else:
        print("Skipping: outfits")
    if "lighting" not in args.skip:
        results["lighting"] = run_pass(args.image, "lighting", args.seed, args.concurrency, lighting_dir)
    else:
        print("Skipping: lighting")

    # Final: assemble presentation (title slide shows the ORIGINAL input image).
    out_pptx = os.path.join(base_dir, f"{name_slug}_character_sheet.pptx")
    if "presentation" not in args.skip:
        results["presentation"] = run_presentation(
            args.image, args.name, args.desc,
            angles_dir, expressions_dir, outfits_dir, lighting_dir, poses_dir, out_pptx)

    elapsed = time.time() - start
    print(f"\n{'='*60}\n  Character Sheet Complete — {int(elapsed//60)}m {int(elapsed%60)}s\n{'='*60}")
    print(f"  Output: {base_dir}")
    for k, ok in results.items():
        print(f"    {'✓' if ok else '✗'} {k}")
    if os.path.exists(out_pptx):
        print(f"  Presentation: {out_pptx}")
    print()


if __name__ == "__main__":
    main()
