#!/usr/bin/env python3
"""
Generate Character Sheet — End-to-End Pipeline
================================================
Runs all passes (angles, expressions, lighting, outfits) with pose extraction,
then assembles everything into a PowerPoint presentation.

Usage:
  python generate_character_sheet.py --image photo.png --name "Nora" --desc "A curious adventurer"

  # Skip specific passes
  python generate_character_sheet.py --image photo.png --name "Nora" --skip expressions lighting

  # Custom seed and concurrency
  python generate_character_sheet.py --image photo.png --name "Nora" --seed 123 --concurrency 5
"""

import argparse
import os
import subprocess
import sys
import time


PIPELINES = ["2511", "expressions", "outfits", "lighting"]


def run_pass(image, pipeline, seed, concurrency, output_dir, get_pose=False, extra_args=None):
    """Run a single batch_multi_angle.py pass."""
    cmd = [
        sys.executable, "batch_multi_angle.py",
        "--image", image,
        "--cloud",
        "--pipeline", pipeline,
        "--seed", str(seed),
        "--concurrency", str(concurrency),
        "--output", output_dir,
    ]
    if get_pose:
        cmd.append("--get-pose")
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Pass: {pipeline}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def run_presentation(image, name, desc, angles_dir, expressions_dir, outfits_dir, lighting_dir, poses_dir):
    """Generate the PowerPoint presentation."""
    cmd = [
        sys.executable, "make_presentation.py",
        "--image", image,
        "--name", name,
        "--desc", desc,
        "--output-dir", angles_dir,
    ]
    if poses_dir and os.path.isdir(poses_dir):
        cmd.extend(["--poses-dir", poses_dir])
    if expressions_dir and os.path.isdir(expressions_dir):
        cmd.extend(["--expressions-dir", expressions_dir])
    if outfits_dir and os.path.isdir(outfits_dir):
        cmd.extend(["--outfits-dir", outfits_dir])
    if lighting_dir and os.path.isdir(lighting_dir):
        cmd.extend(["--lighting-dir", lighting_dir])

    print(f"\n{'='*60}")
    print(f"  Generating Presentation")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete character sheet: angles, expressions, outfits, lighting, and presentation")
    parser.add_argument("--image", required=True, help="Reference character image")
    parser.add_argument("--name", required=True, help="Character name")
    parser.add_argument("--desc", default="", help="Character description")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel cloud jobs")
    parser.add_argument("--output", default=None, help="Base output directory (default: ./charsheet_{name})")
    parser.add_argument("--angle-pipeline", default="2509", choices=["2509", "2511"],
                        help="Multi-angle pipeline (default: 2509)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["base", "angles", "poses", "expressions", "outfits", "lighting", "presentation"],
                        help="Skip specific passes")
    args = parser.parse_args()

    name_slug = args.name.lower().replace(" ", "_")
    base_dir = args.output or f"./charsheet_{name_slug}"
    os.makedirs(base_dir, exist_ok=True)

    base_pose_dir = os.path.join(base_dir, "base")
    angles_dir = os.path.join(base_dir, "angles")
    poses_dir = os.path.join(base_dir, "poses_prompt")
    expressions_dir = os.path.join(base_dir, "expressions")
    outfits_dir = os.path.join(base_dir, "outfits")
    lighting_dir = os.path.join(base_dir, "lighting")

    start = time.time()
    results = {}

    # Pass 0: Normalize the input into a clean, front-facing standing pose on a
    # solid white background. Every downstream pass derives from THIS image so
    # the whole sheet starts from one consistent reference.
    source_image = args.image
    if "base" not in args.skip:
        ok = run_pass(args.image, "standing_relaxed", args.seed, args.concurrency,
                      base_pose_dir)
        base_candidate = os.path.join(base_pose_dir, "standing_relaxed.png")
        if ok and os.path.exists(base_candidate):
            source_image = base_candidate
            results["base"] = True
        else:
            print("WARNING: base standing pose failed — falling back to the original input")
            results["base"] = False
    else:
        print("Skipping: base (using original input as source)")

    # Pass 1: Multi-angle. Uses the 2509 pipeline by default (no 2511).
    # 2509 only has 3 distinct elevation prompts (worm's-eye / level / bird's-eye),
    # so restrict it to -30/0/30 (8x3x3 = 72) to avoid redundant +60 duplicates.
    # No DWPose skeleton extraction (get_pose=False).
    if "angles" not in args.skip:
        # Use the "--opt=value" form: a bare "-30,..." would be misread by argparse
        # as a flag because it starts with a minus.
        angle_extra = ["--elevations=-30,0,30"] if args.angle_pipeline == "2509" else None
        ok = run_pass(source_image, args.angle_pipeline, args.seed, args.concurrency,
                      angles_dir, get_pose=False, extra_args=angle_extra)
        results["angles"] = ok
    else:
        print("Skipping: angles")

    # Pass 2: Poses (16 base action poses)
    if "poses" not in args.skip:
        ok = run_pass(source_image, "poses_prompt", args.seed, args.concurrency,
                      poses_dir, extra_args=["--poses", "base"])
        results["poses"] = ok
    else:
        print("Skipping: poses")

    # Pass 3: Expressions
    if "expressions" not in args.skip:
        ok = run_pass(source_image, "expressions", args.seed, args.concurrency,
                      expressions_dir)
        results["expressions"] = ok
    else:
        print("Skipping: expressions")

    # Pass 4: Outfits — derived from the ORIGINAL input pose (not the normalized
    # base), so outfit changes are shown on the character's own hero pose.
    if "outfits" not in args.skip:
        ok = run_pass(args.image, "outfits", args.seed, args.concurrency,
                      outfits_dir)
        results["outfits"] = ok
    else:
        print("Skipping: outfits")

    # Pass 5: Lighting — also derived from the ORIGINAL input pose.
    if "lighting" not in args.skip:
        ok = run_pass(args.image, "lighting", args.seed, args.concurrency,
                      lighting_dir)
        results["lighting"] = ok
    else:
        print("Skipping: lighting")

    # Final: Assemble presentation (title slide shows the ORIGINAL input image)
    if "presentation" not in args.skip:
        run_presentation(args.image, args.name, args.desc,
                         angles_dir, expressions_dir, outfits_dir, lighting_dir, poses_dir)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  Character Sheet Complete — {minutes}m {seconds}s")
    print(f"{'='*60}")
    print(f"  Output: {base_dir}")
    for pass_name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"    {status} {pass_name}")
    pptx = os.path.join(angles_dir, f"{name_slug}_character_sheet.pptx")
    if os.path.exists(pptx):
        print(f"  Presentation: {pptx}")
    print()


if __name__ == "__main__":
    main()
