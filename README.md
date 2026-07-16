<h1 align="center">Character Soup to Character Nuts</h1>
<h3 align="center">AI power-assist for character assets from design sheets to rigged geometry</h3>

A character sheet is the foundational document in any visual production pipeline. In animation, games, comics, and AI-driven content, it defines how a character looks from every angle, in every pose, and with every expression — the single source of truth that keeps a character consistent across hundreds of shots, scenes, or generated images.

This tool turns **one reference image into a complete, unattended character sheet** — an 8-view turnaround, 16 body poses, 16 facial expressions, plus outfit and lighting variations — assembled into a PowerPoint deck. The results still depend on artistic intent; careful prompting, thoughtful angle selection, and curation are what make the difference.

Runs on [cloud.comfy.org](https://cloud.comfy.org) (requires an API access token), combining [FLUX.1 Kontext](https://blackforestlabs.ai/) for clean source renders with [Qwen Image Edit](https://huggingface.co/collections/Qwen/qwen-image-edit-682e380fc18bf79d426663a2) for the variations.

### Original

![Original character reference](examples/chararcter_ref.png)

---

## The Default Character Sheet

One command takes a single photo/render to a finished deck:

```bash
python generate_character_sheet.py \
  --image photo.png --name "Nora" \
  --desc "A curious adventurer"
```

It runs unattended (~20 min) and writes everything to `charsheet_nora/` (override with `--output`):

```
charsheet_nora/
  base/                 # clean neutral standing reference (+ tiered crops)
  hero/                 # clean matte re-render of the original pose
  angles_combo/         # 8-view turnaround
  poses_prompt/         # 16 body poses
  expressions/          # 16 facial expressions
  outfits/              # 4 outfit variations
  lighting/             # 4 lighting variations
  nora_character_sheet.pptx
```

Skip any pass you don't need:

```bash
python generate_character_sheet.py --image photo.png --name "Nora" \
    --skip outfits lighting
```

### How it works

The pipeline's key move is to **launder the one input through FLUX.1 Kontext into two clean, matte source renders**, and derive every Qwen variation from those:

```
ORIGINAL (photo / render, possibly glossy)
   │
   ├─ Kontext ─► NEUTRAL base ──► Qwen ─► angles · poses · expressions
   │             (standing pose, tiered crops)
   │
   └─ Kontext ─► HERO pose ─────► Qwen ─► outfits · lighting
                 (original pose, keeps held props)
                                     │
   ORIGINAL ─────────────────────────┴─► title slide
                                     ▼
                          make_presentation.py ─► .pptx
```

Why the two-Kontext step exists: the Qwen edit models paint muddy grey-olive blotches over a glossy reference's **specular highlights** whenever they repose it. FLUX.1 Kontext (a different architecture) re-renders clean matte skin instead, so once the base/hero are clean, every downstream Qwen pass inherits clean skin. The **neutral base** feeds the turnaround/poses/expressions; the **hero** keeps the character's original pose and held props (a weapon, a wand, a toy) for the outfit/lighting slides. See [The Weeds](#the-weeds--models-tests-and-hard-won-lessons) for how we got here.

### The passes

| Pass | Source | Output |
|------|--------|--------|
| **base** (Kontext) | original | neutral front standing pose, upscaled + cropped into full/medium/close framing tiers |
| **hero** (Kontext) | original | matte re-render of the original pose on white; keeps held items |
| **angles** (Qwen 2511 + Multi-Angle LoRA) | base tiers | 8-view turnaround (4 full, 2 medium, 2 close) |
| **poses** (Qwen) | base | 16 prompt-driven body poses |
| **expressions** (Qwen) | base | 16 facial expressions |
| **outfits** (Qwen) | hero | 4 outfit variations |
| **lighting** (Qwen) | hero | 4 lighting variations |
| **presentation** | all | 16:9 PowerPoint deck |

The **8-view turnaround** uses the "crop for framing, LoRA for angle" trick: the multi-angle LoRA controls the *camera angle* reliably but is unreliable at *framing* (a "close-up" often renders full-body), while Qwen preserves the *input's* framing. So framing is set by feeding a pre-cropped reference (full / mid-stomach / head-and-chest) and the angle by the LoRA. Non-white drift auto-retries with a new seed.

## Setup

```bash
pip install -r requirements.txt
export COMFY_CLOUD_API_KEY="your-key-here"   # from https://cloud.comfy.org
```

The required models/LoRAs must be available in your Comfy Cloud workspace: `flux1-dev-kontext`, `qwen_image_edit_2511`, the 2511 Lightning + Multi-Angle LoRAs, `4x-UltraSharp` (upscale), and the Qwen VAE / CLIP.

## Customizing for your character

The `outfits` and `lighting` prompts ship with generic slots (`formal`, `casual`, `cold_weather`, `work`; `rim_light`, `side_light`, `golden_hour`, `moonlight`). **Customize them per character** — a soldier, a robot, and a fantasy elf each need different descriptions. Edit the `OUTFITS` and `LIGHTING` dicts in `batch_multi_angle.py`. Held items (a sword, a wand) are preserved automatically via the hero render.

---

## Individual pipelines

Every pass can also be run on its own with `batch_multi_angle.py`. This is the toolbox the default sheet is built from.

```bash
# Clean Kontext source renders (used by the default sheet)
python batch_multi_angle.py --image photo.png --cloud --pipeline kontext_base --steps 20
python batch_multi_angle.py --image photo.png --cloud --pipeline kontext_hero --steps 20

# Combined-angle turnaround references: upscale + tiered crops from a base render
python prep_angle_references.py --image base/standing_relaxed.png --output-dir base

# One turnaround view (crop = framing, 2511 LoRA = angle)
python batch_multi_angle.py --image base/standing_relaxed_close.png --cloud \
    --pipeline 2511 --azimuths 45 --elevations 30 --distances 0.6 --output angles_combo

# Raw multi-angle grids: 2511 (96 = 8az×4el×3dist) or 2509 (72 = 8az×3el×3dist)
python batch_multi_angle.py --image photo.png --cloud --pipeline 2511
python batch_multi_angle.py --image photo.png --cloud --pipeline 2509

# Prompt-driven variations
python batch_multi_angle.py --image photo.png --cloud --pipeline poses_prompt
python batch_multi_angle.py --image photo.png --cloud --pipeline expressions
python batch_multi_angle.py --image photo.png --cloud --pipeline outfits
python batch_multi_angle.py --image photo.png --cloud --pipeline lighting

# Pose transfer from reference skeletons/photos
python batch_multi_angle.py --image photo.png --cloud --pipeline anypose --pose-dir ./poses/F

# Inline DWPose skeleton extraction (body/face/hands) — combines with any pipeline
python batch_multi_angle.py --image photo.png --cloud --get-pose

# Preview every prompt without rendering
python batch_multi_angle.py --image photo.png --cloud --dry-run
```

![Multi-angle example output](examples/angles_4x4.jpg)
![Expressions example output](examples/expressions_4x4.jpg)

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `2511` | `kontext_base`, `kontext_hero`, `2511`, `2509`, `poses_prompt`, `expressions`, `lighting`, `outfits`, `anypose` |
| `--cloud` | off | Use Comfy Cloud (otherwise targets local ComfyUI) |
| `--seed` | `42` | Random seed (change to force a fresh render / dodge the cache) |
| `--steps` | `4` | Inference steps — Qwen Lightning is tuned for 4; Kontext uses 20 |
| `--concurrency` | `3` | Parallel cloud jobs |
| `--azimuths` / `--elevations` / `--distances` | all | Angle-grid subsets, e.g. `--azimuths 0,90,180,270` |
| `--poses` / `--expressions` / `--outfits` | all | Named subsets (`--poses base` = the 16 action poses) |
| `--lightning-lora` | — | Override the Lightning LoRA filename (e.g. the 8-step variant) |
| `--force` | off | Re-render even if the output exists |
| `--prompt-append` | `""` | Text appended to every prompt |
| `--get-pose` | off | DWPose skeleton + keypoint JSON per render |
| `--dry-run` | off | Print prompts without rendering |

### Presentation deck

`make_presentation.py` assembles a 16:9 PPTX (title → 8-view turnaround → poses → expressions → outfits & lighting). It auto-discovers sibling output dirs, or point it explicitly:

```bash
python make_presentation.py --image photo.png --name "Nora" --desc "A curious adventurer" \
  --output-dir angles_combo --poses-dir poses_prompt --expressions-dir expressions \
  --outfits-dir outfits --lighting-dir lighting --output nora_character_sheet.pptx
```

![Presentation template](examples/presentation_3x2.jpg)

### Included pose images

`poses/` contains OpenPose skeleton images from [Pose Depot](https://github.com/pose-depot/pose-depot) — `poses/F/` (61 female) and `poses/M/` (61 male) — for the `anypose` pipeline.

---

## The Weeds — models, tests, and hard-won lessons

Everything below is *why the defaults are the defaults*. Skip it unless you're extending the pipeline.

### The grey-patch artifact → why Kontext for the base

The original pipeline normalized the base pose with **Qwen Image Edit** directly. On glossy references (painted minis, 3D renders) it consistently painted **muddy grey-olive blotches** over the figure's specular highlights (cheekbones, forehead, knuckles, shoulders) during the repose. It was robust to everything we threw at it — unchanged by seed, resolution, step count, prompt wording, LoRA strength, or matte-preprocessing the input.

Models tried, on the hard cases:

| Model | Grey artifact | Fidelity | Notes |
|-------|---------------|----------|-------|
| Qwen 2511 (4-step) | ❌ present | ✅ faithful | the original default |
| Qwen 2511 (8-step LoRA) | ⚠️ reduced, **seed-dependent** | ✅ | not reliably clean; and non-deterministic per seed |
| Qwen 2509 | ✅ clean skin | ❌ drifts outfit/identity | reimagines the character |
| **FLUX.1 Kontext** | ✅ **clean** | ✅ **faithful** | **chosen** — different architecture, matte render, keeps the painted-mini look |
| FireRed-Image-Edit 1.1 | ✅ clean | ⚠️ drops back-items (capes) | good front repose, but can't keep a cloak generically |
| HiDream-E1.1 | — | — | couldn't get its node graph running headless |
| BRIA RMBG node | — | — | hosted-API node, **requires interactive login** — unusable headless |

Key finding — **propagation**: the artifact only appears when Qwen reposes a *glossy* input. Feed Qwen a clean matte source and it stays clean. So we only need Kontext at the two source renders; the ~90 downstream Qwen renders inherit clean skin for free.

### Held items and gear — conditional, never enumerated

Preserving a held sword / wand / prop across a repose is a prompt problem. **Enumerating** item types ("keep any cloak, bow, sword…") makes the model *hallucinate* that gear onto every character (a plain character sprouts a quiver). The fix is a **conditional, non-enumerated** clause: "if the character holds something, keep it; if a hand is empty, keep it empty." The **hero** render sidesteps this entirely for outfits/lighting by not reposing — it keeps whatever the original held.

### White backgrounds — a clamp, not a flood-fill (and not AI matting)

The Kontext/Qwen renders already come out on a near-white background (prompt-enforced, corners ~248–254). The old `flatten_to_white_bg()` was a connected-component **flood-fill** that classified any light, low-saturation region as background — which **bled into white garments** (blown-out sleeve patches) and hardened hair edges. We replaced it with a **per-pixel near-white clamp** (channels ≥ 250 → pure white): it can't touch a shaded white shirt (mid-200s) or soft hair blends, and needs no connectivity/scipy.

AI background removers (BRIA, BiRefNet/BEN2) were evaluated for perfect mattes but weren't worth it: BRIA needs a login (headless-hostile), and the render is already clean enough that a safe clamp wins on simplicity.

### Cache-hang recovery

Comfy Cloud returns `execution_cached` for a repeated (image, seed, prompt) with **no `executed` websocket message**, so the client would block until the timeout. Two mitigations: **change the seed** to force a fresh render, and an automatic recovery in the websocket loop — when the socket goes quiet and the cloud queue is empty but jobs are still pending, it fetches the finished renders from `/api/history_v2` instead of hanging.

### Framing tiers

The turnaround's medium/close crops are set in `prep_angle_references.py` as a fraction of figure height (`TIERS = {"medium": 0.517, "close": 0.36}`). Larger = wider shot, smaller = tighter. Tuned for stylized (large-head) characters; adjust per style.

---

# 3D Rig Generation

Generate a rigged 3D mesh from a single character image using [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) running on [Modal](https://modal.com). The pipeline extracts a full SMPL body mesh with 3D joint positions, skeleton hierarchy, and skin weights, then exports to standard interchange formats (FBX, glTF, OBJ) for import into any DCC tool.

![Pose to 3D rig](examples/pose_to_rig.png)

```bash
# 1. Extract 3D body from image (GPU A10G via Modal)
cd sam3d_pipeline
modal run run_sam3body.py --image-path /path/to/front_eyelevel.png

# 2. Build rig and export (sample Blender importer)
#    reads exports/character_full_data.json -> mesh + 127-joint skeleton + skin weights -> FBX/glTF
blender --python blender_import_rig.py
```

![3D rig](examples/3d_rig.png)

## License

MIT
