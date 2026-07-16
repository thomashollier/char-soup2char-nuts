#!/usr/bin/env python3
"""
Qwen Image Edit — Multi-Angle Batch Renderer (ComfyUI API)
================================================================
Renders all 96 camera poses (8 azimuths × 4 elevations × 3 distances)
from a single input image. Supports both 2509 and 2511 pipelines.

Supports both LOCAL ComfyUI and COMFY CLOUD (cloud.comfy.org).

Requirements:
  pip install aiohttp   (only needed for --cloud mode)

Usage:
  # ── 2511 pipeline (default) ────────────────────────────────
  python batch_multi_angle.py --image photo.png --cloud

  # ── 2509 pipeline ──────────────────────────────────────────
  python batch_multi_angle.py --image photo.png --cloud --pipeline 2509

  # ── SUBSET / DRY RUN ───────────────────────────────────────
  python batch_multi_angle.py --image photo.png --cloud --dry-run
  python batch_multi_angle.py --image photo.png --cloud --azimuths 0,90,180,270
"""

import argparse
import json
import os
import sys
import time
import tempfile
import urllib.request
import urllib.parse
import uuid
import asyncio
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# ANGLE DEFINITIONS (from the LoRA training data)
# ═══════════════════════════════════════════════════════════════════════════

AZIMUTH_MAP = {
    0:   "front view",
    45:  "front-right quarter view",
    90:  "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}

ELEVATION_MAP = {
    -30: "low-angle shot",
    0:   "eye-level shot",
    30:  "elevated shot",
    60:  "high-angle shot",
}

DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot",
}

ALL_AZIMUTHS   = sorted(AZIMUTH_MAP.keys())
ALL_ELEVATIONS = sorted(ELEVATION_MAP.keys())
ALL_DISTANCES  = sorted(DISTANCE_MAP.keys())


def build_prompt_2511(azimuth, elevation, distance):
    return f"<sks> {AZIMUTH_MAP[azimuth]} {ELEVATION_MAP[elevation]} {DISTANCE_MAP[distance]}"


def build_prompt_2509(azimuth, elevation, distance):
    """Build bilingual Chinese+English camera prompt for the 2509 angles LoRA."""
    parts = []
    # Rotation
    if azimuth != 0:
        if azimuth <= 180:
            parts.append(
                f"将镜头向右旋转{azimuth}度 "
                f"Rotate the camera {azimuth} degrees to the right."
            )
        else:
            deg = 360 - azimuth
            parts.append(
                f"将镜头向左旋转{deg}度 "
                f"Rotate the camera {deg} degrees to the left."
            )
    # Elevation / vertical tilt
    if elevation < 0:
        parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")
    elif elevation > 0:
        parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    # Distance / zoom
    if distance < 0.8:
        parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif distance > 1.5:
        parts.append("将镜头向后移动 Move the camera backward.")
    return " ".join(parts) if parts else "no camera movement"


def safe_filename(azimuth, elevation, distance):
    az_name   = AZIMUTH_MAP[azimuth].replace(" ", "_").replace("-", "")
    el_name   = ELEVATION_MAP[elevation].replace(" ", "_").replace("-", "")
    dist_name = DISTANCE_MAP[distance].replace("-", "").replace(" ", "_")
    return f"az{azimuth:03d}_el{elevation:+03d}_d{distance:.1f}_{az_name}_{el_name}_{dist_name}.png"


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSIONS
# ═══════════════════════════════════════════════════════════════════════════

EXPRESSIONS = {
    "neutral":       "Change to a calm, neutral expression with a relaxed face and loose, natural posture.",
    "happy":         "Change to a genuinely happy smile with bright eyes, shoulders lifted slightly and head tilted with warmth.",
    "laughing":      "Change to laughing out loud with an open mouth, squinted eyes, head thrown back slightly and shoulders shaking.",
    "smirk":         "Change to a subtle smirk with one corner of the mouth raised, chin tilted down slightly with a knowing look.",
    "sad":           "Change to a sad, sorrowful look with downturned mouth and drooping eyes, shoulders slumped and head bowed slightly.",
    "crying":        "Change to crying with tears streaming down the cheeks, face scrunched in pain, hands raised near the face.",
    "angry":         "Change to an angry scowl with furrowed brows and clenched jaw, shoulders squared and tensed, leaning forward slightly.",
    "disgusted":     "Change to a look of disgust with a wrinkled nose and curled upper lip, head pulled back and turned slightly away.",
    "surprised":     "Change to wide-eyed surprise with raised eyebrows and open mouth, shoulders pulled up and body leaning back slightly.",
    "fearful":       "Change to a fearful, scared look with wide eyes and tense mouth, shoulders hunched and body shrinking back.",
    "confused":      "Change to a confused look with one raised eyebrow and a slight frown, head tilted to the side.",
    "determined":    "Change to a fierce, determined look with a set jaw and focused eyes, chin raised and chest forward.",
    "flirty":        "Change to a playful, mischievous look with a warm smile and slightly raised eyebrow, head tilted gently.",
    "contempt":      "Change to a contemptuous facial expression with a slight sneer and raised chin, keep the same background.",
    "embarrassed":   "Change to an embarrassed blush with averted gaze and a sheepish smile, shoulders drawn in and head ducked.",
    "sleepy":        "Change to a drowsy, sleepy expression with half-lidded eyes. Lower the upper eyelids so each eye is open only about halfway, giving a relaxed droopy sleepy look, but keep the eyes clearly open with the irises and pupils plainly visible. Both eyes are lowered by the exact same amount, evenly and symmetrically — do not fully close the eyes and do not wink; the left and right eye look identical. Do NOT add any blindfold, mask, cloth, or band over the eyes; her eyes are bare and simply half-open. Keep the eyebrows relaxed. Add a slight, gentle closed-mouth smile with the lips together and the corners of the mouth softly turned up. Relaxed jaw, serene expression.",
}

LIGHTING = {
    "rim_light":     "Change the lighting so a bright light source is directly behind the figure, creating a glowing outline around the hair and shoulders. The front of the face is darker with only ambient fill light.",
    "side_light":    "Change the lighting to strong directional side lighting from the left, casting deep shadows on the right side of the face and body.",
    "golden_hour":   "Change the lighting to warm golden hour sunlight, with long soft shadows and a warm orange glow on the skin.",
    "moonlight":     "Change the lighting to cool blue moonlight at night, with soft shadows and a pale silvery glow on the face.",
}

LIGHTING_PREFIX = (
    "Keep the same pose, outfit, hairstyle, hair color, identity, and facial expression. "
    "Only change the lighting and shadows. "
)

# Generic, character-agnostic outfit slots. Customize these per character for
# best results (a soldier, robot, or elf each need different specifics), but as
# defaults they read reasonably for most humanoid characters.
OUTFITS = {
    "formal":        "Change the outfit to elegant formal attire — a refined, dressed-up version of the character's look, appropriate to their world and style. Keep the same pose and background.",
    "casual":        "Change the outfit to relaxed casual everyday clothes — a comfortable top in a soft but clearly visible color (such as dusty blue, teal, or warm rose, never white or cream) with casual pants or jeans. Keep the same pose and background.",
    "cold_weather":  "Change the outfit to a warm cold-weather ensemble: layered clothing with a heavy coat or jacket and a scarf, suited to the character. Keep the same pose and background.",
    "work":          "Change the outfit to practical work attire — functional, utilitarian clothing suited to physical activity and the character's role. Keep the same pose and background.",
}

OUTFITS_PREFIX = (
    "Keep the same pose, hairstyle, hair color, identity, and facial expression. "
    "The background must be solid pure white (#FFFFFF), completely flat, fully opaque, "
    "with no transparency, no checkerboard pattern, no gradient, and no scene elements. "
    "The clothing must be a distinct, clearly visible color that stands out against the "
    "white background — do NOT use pure white, cream, ivory, or very pale/near-white "
    "clothing that would blend into the background. "
    "Only change the clothing. "
)

POSES = {
    "running":        "Change the pose to running at full speed. Left leg extended forward, right leg pushing off behind. Arms bent at elbows pumping in opposite rhythm to legs. Torso leaning forward.",
    "jumping_joy":    "Change the pose to jumping in the air with joy. Both feet off the ground, knees bent. Both arms raised high above the head, fingers spread wide. Head tilted up, back slightly arched.",
    "sitting_cross":  "Change the pose to sitting cross-legged. The character sits upright with legs folded and ankles crossed in front, back straight, hands resting gently on the knees, head level and facing forward with a calm expression. Full body visible and centered in the frame, camera at eye level, even studio lighting.",
    "waving":         "Change the pose to waving hello. Right arm raised high above the shoulder, hand open with fingers together, wrist tilted side to side. Left arm relaxed at the side. Weight on both feet, slight lean toward the viewer.",
    "thinking":       "Change the pose to deep in thought. Right hand raised to chin, index finger touching the lower lip. Left arm crossed under the right elbow. Weight shifted to left leg, right foot slightly forward. Head tilted slightly to the right, eyes looking up.",
    "dancing":        "Change the pose to dancing energetically. Right arm extended out to the side at shoulder height, left arm bent with hand near the face. Right leg stepping to the side with pointed toe, left leg bent at the knee. Hips shifted to the left, torso twisting.",
    "walking":        "Change the pose to walking at a casual pace, viewed from a 3/4 angle. Left leg stepping forward, right leg behind mid-stride. Left arm swinging back, right arm swinging forward naturally. Torso upright, head facing the walking direction.",
    "crouching":      "Change the pose to a crouch seen from a three-quarter angle. The body is turned about 45 degrees to one side — a 3/4 view, not a full side profile and not straight-on. The character squats down with knees bent and feet flat on the ground, weight shifted onto one leg, one forearm resting on the forward knee and the other hand relaxed at her side, so the pose is not symmetrical. Her head and face are turned toward the camera, looking at the viewer so most of her face is visible. Full body visible and centered in the frame, camera at eye level, even studio lighting.",
    "pointing":       "Change the pose to pointing at something in the distance. Right arm fully extended forward, index finger pointing. Left hand on hip. Body turned slightly to the right, weight on the right foot, left foot angled out.",
    "hands_on_hips":  "Change the pose to standing confidently with both hands on hips. Elbows pushed out wide, fingers wrapped around the waist. Feet shoulder-width apart, chest out, chin raised slightly.",
    "kicking":        "Change the pose to kicking a ball. Right leg swung forward and up at hip height, toes pointed. Left leg planted firmly, slightly bent. Arms out to the sides for balance. Torso leaning back slightly.",
    "jump_rope":      "Change the pose to jumping rope, viewed from a slight 3/4 angle. At the peak of the jump with both feet off the ground, one knee higher than the other. The rope passing beneath the feet. Body slightly turned, not perfectly symmetrical, joyful expression.",
    "talking":        "Change the pose to standing at a 3/4 angle, in confident conversation. One hand raised at chest height with palm up, making a calm deliberate gesture. Other hand relaxed at her side. Weight on one leg, posture upright and poised. Calm confident expression, slight smile, speaking with composure.",
    "leaning_wall":   "Change the pose to leaning against a wall. Right shoulder pressed against the wall, arms crossed over the chest. Right leg crossed over the left at the ankle. Head tilted slightly, looking forward with a relaxed expression.",
    "reading":        "Change the pose to sitting on an institutional mid-century light colored wood chair, reading a book, viewed from a slight 3/4 right angle. Legs crossed at the ankles, leaning back comfortably. Both hands holding an open book at chest height, head tilted down looking at the pages. Relaxed, absorbed expression.",
    "shy_stance":     "Change the pose to a shy, bashful standing pose. The character stands upright with both hands clasped together in front at waist level, shoulders slightly raised, one foot turned slightly inward, head tilted down a little with a gentle upward gaze. Full body visible and centered in the frame, standing straight, camera at eye level, even studio lighting.",
    "tpose":          "Change the pose to a T-pose, facing directly toward the camera. Standing perfectly straight with feet together. Both arms extended straight out to the sides at exactly shoulder height, palms facing down, fingers together. Arms must be fully horizontal, forming a perfect T shape with the torso. Head facing forward, looking straight at the camera with a neutral expression. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_palms":    "Change the pose to a T-pose, facing directly toward the camera. Standing perfectly straight with feet together. Both arms extended straight out to the sides at exactly shoulder height. Wrists fully rotated so both open palms face completely toward the camera, with all five fingers on each hand spread wide apart and fully visible. The hands should be flat and open like a stop gesture, rotated 90 degrees from the arms so the full palm and all fingers are clearly seen. Arms must be fully horizontal, forming a perfect T shape with the torso. Head facing forward, looking straight at the camera with a neutral expression. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_34_palms": "Change the pose to a T-pose, viewed from a 3/4 angle (body turned about 30 degrees to the left). Full body visible from head to feet. Both arms extended straight out to the sides at exactly shoulder height. Wrists rotated forward with thumbs pointing up, palms open, fingers spread wide apart. Arms must be fully horizontal, forming a perfect T shape with the torso. Her face is aligned with her body facing the left edge of the frame, eyes looking left of the camera. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_palms_high": "Change the pose to a T-pose. The photo is taken by a tall person standing close, holding the camera at their eye level and pointing it downward at the shorter character. The character's head appears larger and closer to camera, her feet appear smaller and further away due to the downward perspective. Foreshortening is visible in the body. Her t-shirt and body are facing 30 degrees down from the camera, we see the top of her shoulders. Full body visible from head to feet. Both arms extended straight out to the sides at exactly shoulder height. Wrists rotated forward with thumbs pointing toward the top of the frame and pinky fingers closer to the floor than the thumbs, palms open and aligned with the body pointing 30 degrees below the camera, all five fingers on each hand spread wide apart and clearly visible. Arms fully horizontal forming a T shape. Her head is facing below the camera and her eyes are looking below camera. We see the top of her head and her hair part. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_left_30":    "Change the pose to a T-pose, shot with a wide angle lens up close showing full body from head to feet. The body is turned 30 degrees to the left. Both arms extended straight out to the sides at shoulder height. Wrists rotated forward with thumbs pointing up, palms facing forward, all five fingers spread apart and visible. Arms fully horizontal forming a T shape. Her face and eyes are aligned with her body direction, looking left. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_left_60":    "Change the pose to a T-pose, shot with a wide angle lens up close showing full body from head to feet. The body is turned 60 degrees to the left. Both arms extended straight out to the sides at shoulder height. Wrists rotated forward with thumbs pointing up, palms facing forward, all five fingers spread apart and visible. Arms fully horizontal forming a T shape. Her face and eyes are aligned with her body direction, looking left. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_left_90":    "Change the pose to a T-pose, shot with a wide angle lens up close showing full body from head to feet. The body is turned 90 degrees to the left, showing a perfect side profile. Both arms extended straight out — one arm pointing directly toward the camera, the other pointing directly away. Wrists rotated so palms face forward, all five fingers spread apart. Arms fully horizontal forming a T shape. Her face is in profile, looking left. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_right_30":   "Change the pose to a T-pose, shot with a wide angle lens up close showing full body from head to feet. The body is turned 30 degrees to the right. Both arms extended straight out to the sides at shoulder height. Wrists rotated forward with thumbs pointing up, palms facing forward, all five fingers spread apart and visible. Arms fully horizontal forming a T shape. Her face and eyes are aligned with her body direction, looking right. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_right_60":   "Change the pose to a T-pose, shot with a wide angle lens up close showing full body from head to feet. The body is turned 60 degrees to the right. Both arms extended straight out to the sides at shoulder height. Wrists rotated forward with thumbs pointing up, palms facing forward, all five fingers spread apart and visible. Arms fully horizontal forming a T shape. Her face and eyes are aligned with her body direction, looking right. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "tpose_right_90":   "Change the pose to a T-pose, shot with a wide angle lens up close showing full body from head to feet. The body is turned 90 degrees to the right, showing a perfect side profile. Both arms extended straight out — one arm pointing directly toward the camera, the other pointing directly away. Wrists rotated so palms face forward, all five fingers spread apart. Arms fully horizontal forming a T shape. Her face is in profile, looking right. Keep the character's full body proportions, do not make the body thinner or narrower.",
    "apose":            "Change the pose to a standard A-pose, facing directly toward the camera. Standing perfectly straight with feet shoulder-width apart. Both arms extended down and out to the sides at about 45 degrees from the body, forming an A shape. Wrists rotated forward with thumbs pointing up, palms open, fingers spread apart. Head facing forward, looking straight at the camera with a neutral expression. Keep the character's full body proportions, do not make the body thinner or narrower.",

    # ── A-pose 30° experiments (12 variations) ────────────────────────
    "apose_30_v01": "Change the pose to a standard A-pose. Her body is turned 30 degrees to the right — a slight angle, mostly frontal. Arms held away from body at 45 degree angle downward, hands open. Face looking in same direction as body. Full body head to feet.",
    "apose_30_v02": "A-pose with a gentle 30 degree rightward turn. Most of her front is visible but her right shoulder leads slightly. Arms angled down at 45 degrees from torso forming an A shape. Feet shoulder-width apart. Her face follows her body angle. Full body.",
    "apose_30_v03": "Change to A-pose, body rotated 30 degrees right like a subtle 3/4 view. The peace sign on her shirt is slightly angled but fully readable. Arms down and out at 45 degrees. Her right side is just a bit more prominent. Eyes looking slightly right. Full body head to feet.",
    "apose_30_v04": "A-pose from a gentle right angle. The camera is 30 degrees to the right of center. Her body is mostly frontal with a slight rightward bias. Both arms at 45 degrees from body making an A shape. We can see a hint of her right side. Face follows body. Full body.",
    "apose_30_v05": "Standing in an A-pose, turned 30 degrees right. Like a subtle character select pose — not quite facing the camera, slightly angled. Arms symmetrically at 45 degrees from body. The right hip is slightly closer to camera. Gaze follows body direction. Full body head to feet.",
    "apose_30_v06": "A-pose, slight 3/4 right view at 30 degrees. She is almost facing the camera but her chest is angled just enough that her right shoulder is closer. Arms hang down at 45 degrees from body, palms relaxed. Her face points the same way as her torso — slightly right. Full body.",
    "apose_30_v07": "Change to A-pose facing 30 degrees to the right. Think of a subtle fashion pose angle — mostly frontal with just a touch of dimension. Arms at 45 degrees from torso. The logo on her shirt is slightly perspective-shifted. Her right ear is more visible than her left. Full body head to feet.",
    "apose_30_v08": "A-pose, body aimed about 30 degrees right of the camera. A gentle angle — we still see both eyes clearly but her right cheekbone is more prominent. Both arms at 45 degrees from body. Feet apart. Her right arm is fractionally closer to camera. Full body.",
    "apose_30_v09": "Standing A-pose with 30 degree right rotation. Like the first click on a character turntable — just past front-facing. Arms at 45 degree angle from body. Her torso is slightly diagonal. The braid falls naturally. Full body head to feet.",
    "apose_30_v10": "A-pose at a mild 30 degree right angle. Her body is between frontal and 3/4 — closer to frontal. Arms extend down and outward at 45 degrees. We see the front of her shirt clearly, slightly angled. Her face is turned gently right. Full body.",
    "apose_30_v11": "Change to A-pose. Rotate the view 30 degrees right around her — a small rotation from the front. Her body is just barely angled. Arms at 45 degrees from torso making an A. The peace sign is still mostly front-facing. She looks in her body's direction. Full body head to feet.",
    "apose_30_v12": "A-pose, 30 degrees right. Not quite a 3/4 view, more of a gentle angle that adds dimension to the front view. Arms down at 45 degrees from body. Her right shoulder is slightly nearer to camera. Both braids visible. Face aimed slightly right. Full body.",

    # ── A-pose 45° experiments (12 variations) ────────────────────────
    "apose_45_v01": "Change the pose to a standard A-pose at a 3/4 angle. Her body is turned 45 degrees to the right. Arms held away from body at 45 degree angle, hands open. Face looking in same direction as body. Full body head to feet.",
    "apose_45_v02": "Keep the A-pose with arms angled down at 45 degrees from the torso. Rotate the entire character 45 degrees clockwise so we see her right side more than her left. The peace sign on her shirt should be visibly angled. Her right shoulder is closer to the camera. Full body.",
    "apose_45_v03": "Change to an A-pose viewed from a classic 3/4 right angle — like a character select screen in a video game. Both arms hang down and out at 45 degrees from the body. Feet apart. Body rotated so we see the front and right side equally. Eyes and face follow body direction. Full body head to feet.",
    "apose_45_v04": "Imagine the character is standing on a turntable that has been rotated 45 degrees to the right. She is in a standard A-pose: standing straight, feet apart, both arms angled down and away from her body at 45 degrees. We see her from the front-right. Her face points the same way as her chest. Full body.",
    "apose_45_v05": "A-pose, 3/4 view from the right. The character stands with arms making a letter A — arms straight, angled 45 degrees down from horizontal. Her torso is turned so the right side faces the camera more. The front of her shirt is partially visible. Profile of her nose visible. Full body head to feet.",
    "apose_45_v06": "Change the pose to standing in an A-pose. Move the camera 45 degrees to the right of center. Her right arm appears shorter due to perspective, her left arm extends toward the left of frame. Both arms at 45 degrees from torso. Her face is turned away from camera, showing the right side of her face. Full body.",
    "apose_45_v07": "A-pose from a 45 degree right angle. Like a fashion illustration turnaround at the 3/4 mark. Arms down and out at 45 degrees, palms facing her thighs. Body clearly rotated — we see her right hip and right shoulder prominently. Gaze follows body. Full body head to feet.",
    "apose_45_v08": "Change to A-pose. The viewpoint is halfway between the front view and the right side view. Her body is diagonal to the camera. Arms extend down and outward at 45 degrees. Her right arm is foreshortened, her left arm extends out. Face aligned with body. Full body.",
    "apose_45_v09": "A-pose, shot as if the character is on a lazy susan rotated 45 degrees right. Her chest faces the upper-left corner of the frame. Arms at 45 degrees from body, relaxed. We see the right side of her jeans and the side seam. Her braid falls over her right shoulder toward camera. Full body head to feet.",
    "apose_45_v10": "Standing A-pose with arms at 45 degrees from body. The camera orbits 45 degrees to the right. We see her front-right. The peace sign on her shirt is at a 45 degree angle to the camera. Her right ear is visible, her left ear is hidden. Full body.",
    "apose_45_v11": "A-pose 3/4 right. She stands with feet shoulder width apart, arms angled down at 45 degrees making an A shape. Her body faces the top-left corner of the image. We see the right side of her face, her right arm closer to us. The back pocket of her jeans is just barely visible. Full body head to feet.",
    "apose_45_v12": "Change to an A-pose. Rotate the view 45 degrees around her to the right. Like a character model sheet 3/4 view. Arms symmetrically at 45 degrees from torso, hanging down and out. Her body is clearly angled — not frontal, not profile. We see roughly equal amounts of her front and right side. Full body.",

    # ── A-pose 75° experiments (12 variations) ────────────────────────
    "apose_75_v01": "Change the pose to a standard A-pose. Her body is turned 75 degrees to the right — nearly a side view but her chest is still partially visible. Arms at 45 degrees from body. Face looking right in profile. Full body head to feet.",
    "apose_75_v02": "A-pose from a steep right angle, about 75 degrees. We see mostly her right side with just a sliver of her front visible. The peace sign on her shirt is barely readable at this angle. Arms at 45 degrees from torso. Her right arm is significantly foreshortened toward camera. Face nearly in profile. Full body.",
    "apose_75_v03": "Change to A-pose, body rotated 75 degrees right. Between a 3/4 view and a full side view. Her right hip and right shoulder dominate the frame. Her left arm extends behind her at 45 degrees. We see her right ear clearly. The front of her body is mostly hidden. Full body head to feet.",
    "apose_75_v04": "A-pose at 75 degrees right. Like a character turnaround 3 clicks past the 3/4 mark. Her body is almost perpendicular to camera but angled just enough to show a strip of her front. Arms held at 45 degrees from body — the right arm reaches toward us, the left extends away. Her face is in near-profile. Full body.",
    "apose_75_v05": "Standing A-pose, viewed from 75 degrees to her right. We see the right side of her body prominently — right shoulder, right hip, right leg in front. Her left side is mostly hidden behind her body. Arms at 45 degrees from torso. Her braid hangs behind her right shoulder. Face shows right profile with a hint of her right eye. Full body head to feet.",
    "apose_75_v06": "A-pose from a steep 75 degree right angle. Her body is close to a side view. The right arm at 45 degrees reaches toward the viewer and is foreshortened. The left arm at 45 degrees extends behind her away from camera. We see the side seam of her jeans. Her face is almost in profile. Full body.",
    "apose_75_v07": "Change to A-pose. The camera is positioned 75 degrees to her right — nearly beside her but still seeing a small portion of her front. Both arms at 45 degrees from body. Her right arm appears shorter due to perspective. The back pocket of her jeans is becoming visible. Full body head to feet.",
    "apose_75_v08": "A-pose at a steep 75 degree right angle. Her body is between 3/4 and full profile. We see her right side and just barely see the edge of her chest. Arms symmetrically at 45 degrees from body — one reaching toward camera, one away. Her right eye is visible but her left eye is hidden by her nose. Full body.",
    "apose_75_v09": "Standing in an A-pose, viewed from 75 degrees right. Nearly a side view. Her torso is almost perpendicular to camera. The right arm extends toward us at 45 degrees down. The left arm goes behind her at 45 degrees. We see the depth of her body — front to back. Profile view of her face. Full body head to feet.",
    "apose_75_v10": "A-pose, 75 degrees right. Deep angle, almost a profile but not quite. Her body shows mostly the right flank. Arms at 45 degrees from body, creating a subtle V shape when seen from this angle. Her right arm foreshortened, her left arm visible behind her body. The peace sign is at an extreme angle. Full body.",
    "apose_75_v11": "Change to A-pose rotated 75 degrees right. Like the last frame before a full side view on a turntable. Her right side fills most of the frame. A thin strip of the front of her t-shirt peeks past her right arm. Arms at 45 degrees from body. Her face is in near-profile, nose pointing right. Full body head to feet.",
    "apose_75_v12": "A-pose at 75 degrees. Steep angle — we see her right flank, right arm reaching toward camera at 45 degrees down, left arm behind her at 45 degrees. Her body has real depth from this angle. The side of her jeans is prominent. Her face is nearly in profile with just one eye visible. Full body.",

    # ── A-pose 90° experiments (12 variations) ────────────────────────
    "apose_90_v01": "Change the pose to a standard A-pose seen from a perfect right side profile. Arms extend down and outward at 45 degrees — the right arm points toward camera, the left arm points away behind her. We see the right side of her face only. Full body head to feet.",
    "apose_90_v02": "A-pose, perfect side view from the right. Like a medical posture diagram from the side. Both arms held away from body at 45 degrees. The near arm (right) reaches toward the viewer, the far arm (left) reaches away. Her body is a perfect silhouette from the right side. Full body.",
    "apose_90_v03": "She stands in an A-pose but we view her from directly to her right — a 90 degree side view. Her right arm extends toward us at a 45 degree downward angle. Her left arm extends away behind her at the same angle. We see her right ear, right braid, right side of her jeans. Her face is in full right profile, looking to the right. Full body head to feet.",
    "apose_90_v04": "Change to A-pose, side profile view. Imagine you are standing directly to her right side. Her arms form a V shape when viewed from this angle — right arm reaching toward you at 45 degrees down, left arm reaching away at 45 degrees down. We see her body as a side silhouette. Only the right side of her face is visible. Full body.",
    "apose_90_v05": "A-pose from 90 degrees right. Like a side-view character reference sheet. She faces to the right of the frame. Arms angle down and away from body at 45 degrees, forming a wide V when seen from the side. We see the side profile of her face, the side of her t-shirt, the side seam of her jeans. No front view at all. Full body head to feet.",
    "apose_90_v06": "Perfect right profile A-pose. Her nose points to the right edge of the frame. Both arms are at 45 degrees from her body — from this side view they overlap slightly, the right arm in front reaching toward camera, the left arm behind reaching away. Her braid hangs down her back. We see the right side of everything. Full body.",
    "apose_90_v07": "Side view A-pose. The camera is placed at exactly 90 degrees to her right. She looks like a paper doll viewed from the side. Arms at 45 degrees from torso. The arm closest to us (right) is foreshortened as it reaches toward the lens. Her body width appears narrow because we see her from the side. Full body head to feet.",
    "apose_90_v08": "A-pose seen from her right side. Like standing next to her and looking at her profile. Arms at 45 degrees from body. Her face shows a clean right profile — we see her right eye, nose in profile, chin. Her chest faces to the right. The peace sign is barely visible from this angle. Full body.",
    "apose_90_v09": "Change to A-pose, then rotate the camera 90 degrees to the right so we see her from directly beside her. Her body faces perpendicular to the camera. Arms at 45 degrees from torso. Right arm reaches toward camera, left arm extends behind her. She is in full right profile. Full body head to feet.",
    "apose_90_v10": "Standing A-pose, exact right side view. As if she is facing a wall to the right and we photograph her from the side. Arms at 45 degrees forming a V shape from this perspective. Her right arm is closer to camera. We see the thickness of her body, the side of her face, her ear. Full body.",
    "apose_90_v11": "A-pose at 90 degrees. She faces stage right. We see her as a side elevation — like an architectural side view of a person. Arms at 45 degrees down from body. The right arm reaches toward the viewer. Her hair falls behind her shoulder. Her face is a clean profile showing forehead, nose, lips, chin. Full body head to feet.",
    "apose_90_v12": "Perfect side view A-pose from the right. Think character turnaround sheet, side panel. Her body is perpendicular to the image plane. Arms angled 45 degrees from body. From this viewpoint we see the depth of her body rather than the width. Right profile of face. Right side of clothing. Full body.",
    "apose_90_v13": "Perfect side view A-pose from the right. Think character turnaround sheet, side panel. Her body is perpendicular to the image plane. Arms angled 45 degrees from body. From this viewpoint we see the depth of her body rather than the width. Right profile of face. Right side of clothing. Her feet are pointing to the right side of the frame, the same direction her face and body are pointing — we see the sides of her shoes, not the tops. Her toes point to the right edge of the image. Full body.",

    # ── A-pose 45° → +30° (to ~75°) variants ──────────────────────
    "apose_45p30_v01": "Rotate the character 30 degrees further to the right. Keep the same A-pose with arms at 45 degrees from the torso, arms in line with the torso. Keep the hands in the same orientation. Her face is aligned with her body direction, not turned toward camera. Full body head to feet.",
    "apose_45p30_v02": "Turn the body 30 degrees more to the right from its current angle. The arms stay at 45 degrees from the body, in line with the torso. Hands keep their same orientation — do not rotate the wrists. Her face follows her body, looking past the right edge of frame. Full body.",
    "apose_45p30_v03": "Rotate her body 30 degrees clockwise from this view. She was at 3/4 right, now she is at a steeper angle showing more of her right side. Arms remain at 45 degrees from torso, in line with the body. Hands unchanged. Face aligned with body direction. Full body head to feet.",
    "apose_45p30_v04": "Add 30 degrees of rightward rotation. Her right side becomes more prominent, her left side more hidden. Keep the arms at exactly 45 degrees from the torso, arms staying in line with her body. Hands stay in the same position relative to the arms. Her face points the same direction as her chest. Full body.",
    "apose_45p30_v05": "Turn the character another 30 degrees to the right. We now see mostly her right side with just a portion of her front. Arms at 45 degrees from body, in line with the torso — the right arm foreshortens toward camera, the left arm extends behind. Hands keep same orientation. Face aligned with body. Full body head to feet.",
    "apose_45p30_v06": "Rotate 30 degrees further right. From this steeper angle we see her right flank prominently. Her arms maintain 45 degrees from the torso, in line with her body. Do not change hand position or wrist rotation. Her face is turned with her body, nearly in profile. Full body.",
    "apose_45p30_v07": "Swing the camera 30 degrees further around to her right. Keep the A-pose identical — arms at 45 degrees from body, in line with torso. The peace sign on her shirt is now at a steep angle. Hands remain in same orientation. Her head faces the same direction as her body. Full body head to feet.",
    "apose_45p30_v08": "Rotate her 30 degrees more to the right from current position. Now almost a side view. Arms stay at 45 degrees from torso and in line with her body. The right arm reaches toward camera, left arm extends away. Keep hands in their current orientation. Face follows body direction. Full body.",
    "apose_45p30_v09": "Turn 30 degrees clockwise. Her body is now roughly 75 degrees from the camera. Maintain the A-pose with arms at 45 degrees from body, arms in line with the torso. Hands keep same orientation, no wrist rotation change. Her face aligns with her body, looking to the right. Full body head to feet.",
    "apose_45p30_v10": "Add 30 degrees rightward rotation to the current pose. We see significantly more of her right side now. Arms unchanged at 45 degrees from torso, staying in line with body. Same hand orientation. Her face is nearly in profile, aligned with her body. Full body.",
    "apose_45p30_v11": "Rotate the viewing angle 30 degrees further right around the character. From this steep angle, the right arm is foreshortened and the left arm extends behind her. Arms at 45 degrees from body, in line with torso. Hands in same orientation. Face aligned with body direction. Full body head to feet.",
    "apose_45p30_v12": "Turn her body 30 degrees more to the right. She is now between 3/4 and profile view. The A-pose arms remain at 45 degrees from the torso, in line with her body. No change to hand orientation. Her head and face follow her body angle, looking right. Full body.",

    # ── A-pose 45° → -30° (to ~15°) variants ──────────────────────
    "apose_45m30_v01": "Rotate the character 30 degrees back toward the camera. Keep the same A-pose with arms at 45 degrees from the torso, arms in line with the torso. Keep the hands in the same orientation. Her face is aligned with her body direction. Full body head to feet.",
    "apose_45m30_v02": "Turn the body 30 degrees to the left from its current angle, bringing her more toward frontal. The arms stay at 45 degrees from the body, in line with the torso. Hands keep their same orientation. Her face follows her body direction. Full body.",
    "apose_45m30_v03": "Rotate her body 30 degrees counter-clockwise from this view. She was at 3/4 right, now she is nearly frontal with a slight rightward angle. Arms remain at 45 degrees from torso, in line with the body. Hands unchanged. Face aligned with body direction. Full body head to feet.",
    "apose_45m30_v04": "Subtract 30 degrees of rotation, bringing her closer to facing the camera. Her front becomes more visible. Keep the arms at exactly 45 degrees from the torso, arms staying in line with her body. Hands stay in the same position relative to the arms. Her face points the same direction as her chest. Full body.",
    "apose_45m30_v05": "Turn the character 30 degrees back toward the camera. We now see mostly her front with just a slight angle. Arms at 45 degrees from body, in line with the torso. Hands keep same orientation. Face aligned with body. Full body head to feet.",
    "apose_45m30_v06": "Rotate 30 degrees toward frontal. From this gentler angle we see most of her front and just a hint of her right side. Her arms maintain 45 degrees from the torso, in line with her body. Do not change hand position or wrist rotation. Her face is turned with her body, mostly toward camera. Full body.",
    "apose_45m30_v07": "Swing the camera 30 degrees back toward front. Keep the A-pose identical — arms at 45 degrees from body, in line with torso. The peace sign on her shirt is now nearly front-facing. Hands remain in same orientation. Her head faces the same direction as her body. Full body head to feet.",
    "apose_45m30_v08": "Rotate her 30 degrees toward the camera from current position. Now a subtle angle, mostly frontal. Arms stay at 45 degrees from torso and in line with her body. Keep hands in their current orientation. Face follows body direction. Full body.",
    "apose_45m30_v09": "Turn 30 degrees counter-clockwise. Her body is now roughly 15 degrees from the camera — nearly straight on. Maintain the A-pose with arms at 45 degrees from body, arms in line with the torso. Hands keep same orientation. Her face aligns with her body. Full body head to feet.",
    "apose_45m30_v10": "Remove 30 degrees of rotation from the current pose. She faces almost directly at camera now. Arms unchanged at 45 degrees from torso, staying in line with body. Same hand orientation. Her face is nearly frontal, aligned with her body. Full body.",
    "apose_45m30_v11": "Rotate the viewing angle 30 degrees back toward front around the character. From this gentle angle, both arms are clearly visible at 45 degrees from body, in line with torso. Hands in same orientation. Face aligned with body direction. Full body head to feet.",
    "apose_45m30_v12": "Turn her body 30 degrees back toward the camera. She is now just slightly angled from frontal. The A-pose arms remain at 45 degrees from the torso, in line with her body. No change to hand orientation. Her head and face follow her body angle. Full body.",
}

ANGLES_PROMPT = {}
_GAZE_BRAID = "The character's eyes look straight ahead in the exact direction the face is pointing, with a calm neutral gaze — the eyes do not glance to the side, up, down, or toward the camera. Keep the exact same hairstyle as in the original image; do not loosen, let down, add, or otherwise change the hair."
# Explicit medium-shot crop: bottom edge at mid-stomach, top edge just above the head.
_MEDIUM_FRAME = "This is a medium shot: frame the character so the bottom edge of the image cuts across the middle of the stomach and the top edge sits just above the top of the head. Only the head, shoulders, chest, and upper stomach are visible; the lower body is out of frame."
_DIRECTIONS = [
    ("front",       "000", "front-facing",                                          ""),
    ("front_right", "045", "from a 3/4 front-right angle",                          "Rotate the camera 45 degrees to the right."),
    ("right",       "090", "in right side profile",                                 "Rotate the camera 90 degrees to the right."),
    ("back_right",  "135", "from behind and slightly to the right",                 "Rotate the camera 135 degrees to the right."),
    ("back",        "180", "from directly behind",                                  "Rotate the camera 180 degrees to show the back."),
    ("back_left",   "225", "from behind and slightly to the left",                  "Rotate the camera 135 degrees to the left."),
    ("left",        "270", "in left side profile",                                  "Rotate the camera 90 degrees to the left."),
    ("front_left",  "315", "from a 3/4 front-left angle",                           "Rotate the camera 45 degrees to the left."),
]
for _name, _deg, _view, _rotate in _DIRECTIONS:
    _r = f"{_rotate} " if _rotate else ""
    ANGLES_PROMPT[f"{_name}_close"]  = f"{_r}Close-up shot showing head and shoulders {_view}. {_GAZE_BRAID}"
    ANGLES_PROMPT[f"{_name}_medium"] = f"{_r}Medium shot {_view}. {_MEDIUM_FRAME} {_GAZE_BRAID}"
    ANGLES_PROMPT[f"{_name}_wide"]   = f"{_r}Full body wide shot {_view}, showing the entire figure from head to toe. {_GAZE_BRAID}"
    # Elevated 30° versions (camera above looking down)
    _elev_cam = "The camera is higher up, at head height plus 3 feet, angled slightly downward. The horizon line is low in the frame. Do not roll or rotate the camera, keep it perfectly level left to right."
    _neutral_gaze = "Keep the exact same neutral facial expression as the original image. The character's eyes look straight ahead in the exact direction the face is pointing, with a calm neutral gaze — the eyes do not glance to the side, up, down, or toward the camera. Keep the exact same hairstyle as in the original image; do not loosen, let down, add, or otherwise change the hair."
    ANGLES_PROMPT[f"{_name}_elev_close"]  = f"{_r}{_elev_cam} Close-up shot showing head and shoulders {_view}. {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_elev_medium"] = f"{_r}{_elev_cam} Medium shot {_view}. {_MEDIUM_FRAME} {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_elev_wide"]   = f"{_r}{_elev_cam} Full body wide shot {_view}, showing the entire figure from head to toe. {_neutral_gaze}"
    # Low angle 30° versions (camera below looking up)
    _low_cam = "The camera is lower, at knee height, angled slightly upward. The horizon line is high in the frame. Do not roll or rotate the camera, keep it perfectly level left to right."
    ANGLES_PROMPT[f"{_name}_low_close"]  = f"{_r}{_low_cam} Close-up shot showing head and shoulders {_view}. {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_low_medium"] = f"{_r}{_low_cam} Medium shot {_view}. {_MEDIUM_FRAME} {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_low_wide"]   = f"{_r}{_low_cam} Full body wide shot {_view}, showing the entire figure from head to toe. {_neutral_gaze}"

# ── Extra hand-tuned angle prompts ──────────────────────────────────────────
ANGLES_PROMPT["front_3q_right_medium"] = (
    "Rotate the camera 45 degrees to the right so the character is seen from a 3/4 "
    "front-right angle, the body turned about 45 degrees. " + _MEDIUM_FRAME + " " + _GAZE_BRAID
)
ANGLES_PROMPT["back_3q_left_medium"] = (
    "Rotate the camera 135 degrees to the left so the character is seen from a 3/4 "
    "back-left angle — mostly from behind, turned about 45 degrees toward the left, with "
    "the back of the head and one shoulder toward the camera. " + _MEDIUM_FRAME + " " + _GAZE_BRAID
)
ANGLES_PROMPT["front_slight_close"] = (
    "Close-up shot showing the head and chest. The whole character is rotated as one single "
    "rigid unit — shoulders, chest, neck, and head all turned together by the same tiny "
    "amount to one side, the gentlest possible rotation away from a dead-on front view. Her "
    "shoulders and chest face the exact same direction as her head; the neck stays straight "
    "and does NOT twist, so the head and body stay perfectly aligned, as if the entire "
    "figure were turned together on a turntable. It is only barely off from front-on: both "
    "ears remain visible, both eyes are clearly and almost equally seen, and the nose points "
    "nearly straight at the camera, offset by only a whisper. This is NOT a 3/4 view and NOT "
    "a profile. The eyes look straight ahead in the same direction the body and head face, "
    "in line with the body — do not turn the head or eyes back toward the camera. A neutral "
    "reference view. " + _GAZE_BRAID
)

# Shared clause forcing a clean, solid white background on every derived render.
WHITE_BG = (
    "The background must be solid pure white (#FFFFFF), completely flat and fully "
    "opaque, with no transparency, no checkerboard pattern, no gradient, no shadows "
    "on the background, and no scene elements or props. "
)

ANGLES_PROMPT_PREFIX = (
    "Keep the same outfit, hairstyle, hair color, and identity. "
    + WHITE_BG
)

POSES_PREFIX = (
    "Keep the same outfit, hairstyle, hair color, and identity. "
    + WHITE_BG
    + "Clothing should move naturally with the body. "
)

EXPRESSION_PREFIX = (
    "Keep the same outfit, hairstyle, hair color, lighting, "
    "and identity. Clothing should move naturally with the body. "
    + WHITE_BG
    + "Frame the image as a medium close-up shot showing the head and shoulders, "
    "with the face large and clearly visible and centered in the frame. "
    "The camera is positioned at a 3/4 angle to the character, so the head and body "
    "are turned roughly 45 degrees away from the camera rather than facing straight on, "
    "showing a three-quarter view of the face. "
)


# ═══════════════════════════════════════════════════════════════════════════
# ANYPOSE DEFAULT PROMPT
# ═══════════════════════════════════════════════════════════════════════════

STANDING_RELAXED_PROMPT = (
    "Keep the same outfit, hairstyle, hair color, identity, and facial features. "
    "Clothing should move naturally with the body. "
    "Change the pose to a neutral, relaxed standing pose facing directly toward the camera. "
    "Full body visible from head to feet, both feet planted flat on the ground, standing "
    "up straight. Arms hang naturally and relaxed at the sides, not crossed and not on the "
    "hips, shoulders down and loose. Head facing forward with a calm neutral expression, "
    "looking straight at the camera. Center the full figure in the frame. "
    "The background must be solid pure white (#FFFFFF), completely flat, fully even and "
    "opaque, with no transparency, no checkerboard pattern, no gradient, no shadows on the "
    "background, and no scene elements. "
    "Keep the character's full body proportions, do not make the body thinner or narrower."
)


STANDING_RELAXED_PROMPT_NANOBANANA = (
    "Change the character to a relaxed standing pose, full body visible from head to feet, "
    "both feet on the ground. The body is turned about 20 degrees to the left — not a full "
    "3/4 view, just a subtle angle. Arms hang naturally at the sides. Weight shifted onto "
    "one leg with the other knee slightly soft, hip cocked subtly. Head and gaze aligned "
    "with the body, looking slightly to the left, calm neutral expression. Replace the "
    "background with a plain solid medium grey, completely flat and even, no shadows, no "
    "gradient, no scene elements. Preserve the character's exact body proportions and "
    "silhouette from the original image; do not broaden, narrow, or otherwise reshape the "
    "body. Keep the same outfit, hairstyle, hair color, identity, facial features, and full "
    "body proportions."
)

# ═══════════════════════════════════════════════════════════════════════════
# FLUX KONTEXT — clean matte source renders
# ═══════════════════════════════════════════════════════════════════════════
# The Qwen edit models paint muddy grey-olive blotches over a glossy reference's
# specular highlights whenever they repose it. Flux Kontext (a different
# architecture) re-renders clean matte skin instead, so we use it up front to
# turn the one glossy original into two clean matte sources:
#   • kontext_base — a neutral standing pose (feeds angles / poses / expressions)
#   • kontext_hero — the original hero pose, matte re-render (feeds outfits /
#     lighting / title; keeps held items because it does not repose)
# Every downstream Qwen pass then starts from a clean input and stays artifact-free.

# Conditional, NON-enumerated: keep held objects only if the reference has them,
# keep empty hands empty. Enumerating item types makes the model hallucinate them
# onto every character, so we never name specific gear.
HELD_ITEMS_CLAUSE = (
    " If the character holds or grips any object in a hand in the reference image, they must "
    "continue to firmly hold that exact same object in the same hand, gripped and lowered at "
    "their side, and it must remain clearly visible. If a hand is empty in the reference, keep "
    "it empty; never place a new object into an empty hand."
)

KONTEXT_BASE_PROMPT = STANDING_RELAXED_PROMPT + HELD_ITEMS_CLAUSE

# Keep the ORIGINAL pose, just re-render it clean/matte on white — an artifact-free
# hero source for the outfit/lighting/title passes.
KONTEXT_HERO_PROMPT = (
    "Keep the character's exact pose, stance, body position, arm and hand positions, and "
    "camera framing identical to the reference image — do not repose or move anything, and "
    "keep the same outfit, identity, hair, and every detail and accessory. Render the "
    "character with clean, smooth, evenly painted matte surfaces without glossy sheen or "
    "bright specular hotspots. Place the character on a solid pure white background with no "
    "scene, floor, shadow, or props."
)

NANOBANANA_SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)


ANYPOSE_DEFAULT_PROMPT = (
    "Make the person in image 1 do the exact same pose of the person in image 2. "
    "Changing the style and background of the image of the person in image 1 is undesirable, so don't do it. "
    "The new pose should be pixel accurate to the pose we are trying to copy. "
    "The position of the arms and head and legs should be the same as the pose we are trying to copy. "
    "Change the field of view and angle to match exactly image 2. Head tilt and eye gaze pose should match the person in image 2. "
    "Remove the background of image 2, and replace it with the background of image 1. "
    "Don't change the identity of the person in image 1, keep their appearance the same, "
    "it is undesirable to change their facial features or hair style. don't do it."
)


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def inject_dwpose_nodes(workflow, image_node_output, pose_prefix):
    """Inject DWPose extraction nodes into an existing workflow.

    Adds a DWPreprocessor that reads from image_node_output (e.g. ["4:102", 0]),
    plus SaveImage and SavePoseKpsAsJsonFile nodes for the skeleton and JSON.
    Uses node IDs "dwpose:1", "dwpose:save_img", "dwpose:save_json" to avoid collisions.
    """
    workflow["dwpose:1"] = {
        "class_type": "DWPreprocessor",
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 1024,
            "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "scale_stick_for_xinsr_cn": "disable",
            "image": image_node_output,
        },
    }
    workflow["dwpose:save_img"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": f"{pose_prefix}_skeleton",
            "images": ["dwpose:1", 0],
        },
    }
    # No need for SavePoseKpsAsJsonFile — DWPreprocessor already outputs
    # JSON via its "ui" key as "openpose_json". We capture it from the
    # websocket "executed" message for the dwpose:1 node.
    return workflow


# Map pipeline -> the node output that produces the final rendered image
PIPELINE_IMAGE_OUTPUT = {
    "2511":           ["4:102", 0],
    "2509":           ["162:19", 0],
    "anypose":        ["4:102", 0],
    "expressions":    ["4:102", 0],
    "lighting":       ["4:102", 0],
    "outfits":        ["4:102", 0],
    "poses_prompt":   ["4:102", 0],
    "angles_prompt":  ["4:102", 0],
    "standing_relaxed": ["4:102", 0],
}

# Map pipeline -> the SaveImage node id used by its workflow builder.
# (build_workflow_2511 uses "2", _2509 uses "31", _expressions and _anypose use "10")
PIPELINE_SAVE_NODE = {
    "2511":             "2",
    "2509":             "31",
    "anypose":          "10",
    "expressions":      "10",
    "lighting":         "10",
    "outfits":          "10",
    "poses_prompt":     "10",
    "angles_prompt":    "10",
    "standing_relaxed": "10",
    "kontext_base":     "13",
    "kontext_hero":     "13",
}


def build_workflow(
    image_filename, azimuth, elevation, distance, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, lora_strength_angles=1.0,
    filename_prefix="multi_angle", pipeline="2511",
    pose_image_filename=None, get_pose=False,
    qwen_version="2511", engine="qwen", reinforce="", megapixels=None,
):
    """Dispatch to the appropriate pipeline workflow builder."""
    if engine == "nanobanana":
        return build_workflow_nanobanana(
            image_filename, prompt, seed=seed, filename_prefix=filename_prefix,
        )
    if pipeline == "2509":
        wf = build_workflow_2509(
            image_filename, prompt, seed, steps, guidance_scale,
            lora_strength_lightning, lora_strength_angles, filename_prefix,
        )
    elif pipeline == "anypose":
        wf = build_workflow_anypose(
            image_filename, pose_image_filename, prompt, seed, steps,
            guidance_scale, lora_strength_lightning, filename_prefix,
        )
    elif pipeline in ("kontext_base", "kontext_hero"):
        wf = build_workflow_kontext(image_filename, prompt, seed, steps, filename_prefix)
    elif pipeline in ("expressions", "lighting", "outfits", "poses_prompt", "angles_prompt", "standing_relaxed"):
        wf = build_workflow_expressions(
            image_filename, prompt, seed, steps, guidance_scale,
            lora_strength_lightning, filename_prefix,
            qwen_version=qwen_version, megapixels=megapixels,
        )
    else:
        wf = build_workflow_2511(
            image_filename, azimuth, elevation, distance,
            seed, steps, guidance_scale,
            lora_strength_lightning, lora_strength_angles, filename_prefix,
            reinforce=reinforce,
        )

    if get_pose:
        image_output = PIPELINE_IMAGE_OUTPUT.get(pipeline, ["4:102", 0])
        inject_dwpose_nodes(wf, image_output, filename_prefix)

    return wf


def build_workflow_2511(
    image_filename, azimuth, elevation, distance,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, lora_strength_angles=1.0,
    filename_prefix="multi_angle", reinforce="",
):
    """
    Build a ComfyUI API-format workflow for 2511 pipeline.
    Uses QwenMultiangleCameraNode for camera control.

    If `reinforce` is given, its text is encoded separately and concatenated onto
    the camera node's positive conditioning (ConditioningConcat), letting a plain
    text prompt reinforce framing/gaze on top of the LoRA's camera control.
    """
    wf = {
        # ── Load input image ──────────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["4:102", 0],
            },
        },
        # ── Camera angle node (generates prompt automatically) ────
        "3": {
            "class_type": "QwenMultiangleCameraNode",
            "inputs": {
                "horizontal_angle": azimuth,
                "vertical_angle": elevation,
                # QwenMultiangleCameraNode zoom scale: <2="wide", <6="medium", >=6="close-up"
                # Our distance values: 0.6=close-up, 1.0=medium, 1.8=wide
                "zoom": {0.6: 8.0, 1.0: 4.0, 1.8: 1.0}.get(distance, 4.0),
                "default_prompts": False,
                "camera_view": False,
                "image": ["1", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "4:95": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "4:100": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["4:93", 0],
                "vae": ["4:95", 0],
                "image1": ["4:106", 0],
            },
        },
        # ── Positive conditioning (prompt from camera node) ───────
        "4:103": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": ["3", 0],
                "clip": ["4:93", 0],
                "vae": ["4:95", 0],
                "image1": ["4:106", 0],
            },
        },
        # ── Reference latent methods ──────────────────────────────
        "4:97": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                # positive conditioning: camera-node prompt, optionally with
                # reinforcement text concatenated on (see reinforce nodes below)
                "conditioning": ["4:concat", 0] if reinforce else ["4:103", 0],
            },
        },
        "4:96": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["4:100", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "4:94": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.1,
                "model": ["4:110", 0],
            },
        },
        "4:98": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["4:94", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "4:104": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["4:106", 0],
                "vae": ["4:95", 0],
            },
        },
        "4:102": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["4:105", 0],
                "vae": ["4:95", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "4:105": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["4:98", 0],
                "positive": ["4:97", 0],
                "negative": ["4:96", 0],
                "latent_image": ["4:104", 0],
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "4:108": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "4:107": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["4:108", 0],
            },
        },
        "4:110": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
                "strength_model": lora_strength_angles,
                "model": ["4:107", 0],
            },
        },
        # ── CLIP ──────────────────────────────────────────────────
        "4:93": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Image scaling ─────────────────────────────────────────
        "4:106": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["1", 0],
            },
        },
    }

    # Optional prompt reinforcement: encode the reinforce text and concatenate it
    # onto the camera node's positive conditioning (node 4:97 reads 4:concat).
    if reinforce:
        wf["4:103r"] = {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": reinforce,
                "clip": ["4:93", 0],
                "vae": ["4:95", 0],
                "image1": ["4:106", 0],
            },
        }
        wf["4:concat"] = {
            "class_type": "ConditioningConcat",
            "inputs": {
                "conditioning_to": ["4:103", 0],
                "conditioning_from": ["4:103r", 0],
            },
        }

    return wf


def build_workflow_2509(
    image_filename, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, lora_strength_angles=1.0,
    filename_prefix="multi_angle",
):
    """
    Build a ComfyUI API-format workflow for 2509 pipeline.
    Based on the 1-click multiple character angles template.
    """
    return {
        # ── Load input image ──────────────────────────────────────
        "25": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "31": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["162:19", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "48:9": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── CLIP ──────────────────────────────────────────────────
        "48:10": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "48:12": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "weight_dtype": "default",
            },
        },
        # ── Lightning LoRA ────────────────────────────────────────
        "48:26": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["48:12", 0],
            },
        },
        # ── Angles LoRA ───────────────────────────────────────────
        "48:20": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Edit-2509-Multiple-angles.safetensors",
                "strength_model": lora_strength_angles,
                "model": ["48:26", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "162:11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.0,
                "model": ["48:20", 0],
            },
        },
        "162:8": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["162:11", 0],
            },
        },
        # ── Image scaling ─────────────────────────────────────────
        "162:28": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "upscale_method": "nearest-exact",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": ["25", 0],
            },
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "162:14": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["48:10", 0],
                "vae": ["48:9", 0],
                "image1": ["162:28", 0],
            },
        },
        # ── Positive conditioning ─────────────────────────────────
        "162:17": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": prompt,
                "clip": ["48:10", 0],
                "vae": ["48:9", 0],
                "image1": ["162:28", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "162:13": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["162:28", 0],
                "vae": ["48:9", 0],
            },
        },
        "162:19": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["162:21", 0],
                "vae": ["48:9", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "162:21": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["162:8", 0],
                "positive": ["162:17", 0],
                "negative": ["162:14", 0],
                "latent_image": ["162:13", 0],
            },
        },
    }


def build_workflow_nanobanana(
    image_filename, prompt,
    seed=42,
    filename_prefix="standing_relaxed",
    aspect_ratio="1:1",
    resolution="2K",
    model="gemini-3-pro-image-preview",
):
    """ComfyUI workflow that calls Nano Banana Pro via the GeminiImage2Node."""
    return {
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "3": {
            "class_type": "GeminiImage2Node",
            "inputs": {
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "response_modalities": "IMAGE",
                "system_prompt": NANOBANANA_SYSTEM_PROMPT,
                "images": ["2", 0],
            },
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["3", 0],
            },
        },
    }


QWEN_EDIT_MODELS = {
    "2511": {
        "unet":      "qwen_image_edit_2511_bf16.safetensors",
        "lightning": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "clip":      "qwen_2.5_vl_7b.safetensors",
    },
    "2509": {
        "unet":      "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "lightning": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "clip":      "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    },
}


def build_workflow_kontext(image_filename, prompt, seed=42, steps=20, filename_prefix="kontext"):
    """Flux.1 Kontext reference-edit workflow (different architecture from Qwen, so it
    renders clean matte skin instead of the grey-blotch artifact). Used for the two
    clean source renders (kontext_base neutral pose, kontext_hero original pose)."""
    return {
        "1":  {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "2":  {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev-kontext_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "3":  {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
        "4":  {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "5":  {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
        "6":  {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["4", 0]}},
        "7":  {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "8":  {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
        "9":  {"class_type": "FluxGuidance", "inputs": {"conditioning": ["8", 0], "guidance": 2.5}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["2", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["12", 0]}},
    }


def build_workflow_expressions(
    image_filename, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, filename_prefix="expression",
    qwen_version="2511", megapixels=None,
):
    """
    Build a ComfyUI API-format workflow for expression editing.
    Uses Qwen Image Edit base model + Lightning LoRA with a text prompt.

    If `megapixels` is given, the input is scaled to that total-pixel target
    (e.g. 1.76 for Qwen's native 1328x1328) instead of Flux Kontext's ~1 MP.
    """
    models = QWEN_EDIT_MODELS[qwen_version]
    wf = {
        # ── Load input image ──────────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["9", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── CLIP ──────────────────────────────────────────────────
        "21": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": models["clip"],
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "30": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": models["unet"],
                "weight_dtype": "default",
            },
        },
        # ── Lightning LoRA ────────────────────────────────────────
        "31": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": models["lightning"],
                "strength_model": lora_strength_lightning,
                "model": ["30", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "34": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.1,
                "model": ["31", 0],
            },
        },
        "35": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["34", 0],
            },
        },
        # ── Image scaling ─────────────────────────────────────────
        "40": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["1", 0],
            },
        },
        # ── Positive conditioning ─────────────────────────────────
        "50": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": prompt,
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
            },
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "51": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
            },
        },
        # ── Reference latent methods ──────────────────────────────
        "52": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["50", 0],
            },
        },
        "53": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["51", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "60": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["40", 0],
                "vae": ["20", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["70", 0],
                "vae": ["20", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "70": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["35", 0],
                "positive": ["52", 0],
                "negative": ["53", 0],
                "latent_image": ["60", 0],
            },
        },
    }

    # Optionally render at a higher total-pixel target (e.g. Qwen-native ~1.76 MP
    # = 1328x1328) by replacing the Flux ~1 MP scale on node "40".
    if megapixels:
        wf["40"] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "megapixels": megapixels,
                "resolution_steps": 64,
            },
        }

    return wf


def build_workflow_anypose(
    image_filename, pose_image_filename, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, filename_prefix="anypose",
    lora_strength_base=0.7, lora_strength_helper=0.7,
):
    """
    Build a ComfyUI API-format workflow for AnyPose pipeline.
    Takes a reference image and a pose image, transfers the pose.
    Uses 2511 base model + Lightning + AnyPose base + AnyPose helper LoRAs.
    """
    return {
        # ── Load reference image ──────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Load pose image ───────────────────────────────────────
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["9", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── CLIP ──────────────────────────────────────────────────
        "21": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "30": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        # ── Lightning LoRA ────────────────────────────────────────
        "31": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["30", 0],
            },
        },
        # ── AnyPose base LoRA ─────────────────────────────────────
        "32": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "2511-AnyPose-base-000006250.safetensors",
                "strength_model": lora_strength_base,
                "model": ["31", 0],
            },
        },
        # ── AnyPose helper LoRA ───────────────────────────────────
        "33": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "2511-AnyPose-helper-00006000.safetensors",
                "strength_model": lora_strength_helper,
                "model": ["32", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "34": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.1,
                "model": ["33", 0],
            },
        },
        "35": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["34", 0],
            },
        },
        # ── Image scaling (reference) ─────────────────────────────
        "40": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["1", 0],
            },
        },
        # ── Image scaling (pose) ──────────────────────────────────
        "41": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["2", 0],
            },
        },
        # ── Positive conditioning (reference + pose images) ───────
        "50": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": prompt,
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
                "image2": ["41", 0],
            },
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "51": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
            },
        },
        # ── Reference latent methods ──────────────────────────────
        "52": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["50", 0],
            },
        },
        "53": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["51", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "60": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["40", 0],
                "vae": ["20", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["70", 0],
                "vae": ["20", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "70": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["35", 0],
                "positive": ["52", 0],
                "negative": ["53", 0],
                "latent_image": ["60", 0],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL COMFYUI API  (synchronous, sequential)
# ═══════════════════════════════════════════════════════════════════════════

def local_upload_image(server_url, filepath):
    filepath = Path(filepath)
    boundary = uuid.uuid4().hex
    with open(filepath, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filepath.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{server_url}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["name"]


def local_queue(server_url, workflow, client_id):
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{server_url}/prompt", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["prompt_id"]


def local_wait(server_url, prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{server_url}/history/{prompt_id}") as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out after {timeout}s")


def local_download(server_url, filename, subfolder, output_path):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    urllib.request.urlretrieve(f"{server_url}/view?{params}", output_path)


def run_local(jobs, args):
    print(f"  Mode: LOCAL → {args.server}\n")
    print("Uploading image...")
    try:
        server_filename = local_upload_image(args.server, args.image)
    except Exception as e:
        sys.exit(f"ERROR: Could not reach ComfyUI at {args.server}\n  {e}")
    print(f"  Uploaded as: {server_filename}\n")

    client_id = str(uuid.uuid4())
    total, ok, fail = len(jobs), 0, 0

    for i, (az, el, dist, prompt, fname) in enumerate(jobs, 1):
        out = os.path.join(args.output, fname)
        if os.path.exists(out) and not args.force:
            print(f"  [{i:3d}/{total}] SKIP  {fname}")
            ok += 1; continue
        print(f"  [{i:3d}/{total}] {prompt}")
        print(f"           → {fname}", end="", flush=True)
        try:
            cloud_prefix = os.path.splitext(fname)[0]
            wf = build_workflow(server_filename, az, el, dist, prompt,
                                args.seed, args.steps,
                                args.guidance, args.lora_lightning, args.lora_angles,
                                filename_prefix=cloud_prefix,
                                pipeline=args.pipeline,
                                get_pose=getattr(args, 'get_pose', False),
                                qwen_version=args.qwen_version,
                                engine=args.engine)
            pid = local_queue(args.server, wf, client_id)
            result = local_wait(args.server, pid, args.timeout)
            outputs = result.get("outputs", {})
            save_node = "4" if args.engine == "nanobanana" else PIPELINE_SAVE_NODE.get(args.pipeline, "2")
            images = outputs.get(save_node, {}).get("images", [])
            if images:
                local_download(args.server, images[0]["filename"],
                               images[0].get("subfolder", ""), out)
                if args.label:
                    add_label_below(out, args.label)
                print("  ✓"); ok += 1
                # Download pose outputs if --get-pose
                if getattr(args, 'get_pose', False):
                    pose_dir = os.path.join(args.output, "poses")
                    os.makedirs(pose_dir, exist_ok=True)
                    pose_base = os.path.splitext(fname)[0]
                    # Skeleton image
                    skel_imgs = outputs.get("dwpose:save_img", {}).get("images", [])
                    if skel_imgs:
                        local_download(args.server, skel_imgs[0]["filename"],
                                       skel_imgs[0].get("subfolder", ""),
                                       os.path.join(pose_dir, f"{pose_base}_skeleton.png"))
                        print(f"           🦴 skeleton ✓")
                    # Pose JSON from DWPreprocessor ui output
                    dw_output = outputs.get("dwpose:1", {})
                    openpose_json = dw_output.get("openpose_json", [])
                    if openpose_json:
                        json_out = os.path.join(pose_dir, f"{pose_base}_pose.json")
                        json_str = openpose_json[0] if isinstance(openpose_json[0], str) else json.dumps(openpose_json[0])
                        with open(json_out, "w") as jf:
                            jf.write(json_str)
                        print(f"           🦴 pose JSON ✓")
            else:
                print("  ✗ no output"); fail += 1
        except Exception as e:
            print(f"  ✗ {e}"); fail += 1
    return ok, fail


# ═══════════════════════════════════════════════════════════════════════════
# COMFY CLOUD API  (async, parallel)
# ═══════════════════════════════════════════════════════════════════════════

CLOUD_BASE = "https://cloud.comfy.org"


def get_reference_bg_color(image_path):
    """Get the average edge color of the reference image."""
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(edges.mean(axis=0).astype(int))


def prep_pose_image(pose_path, bg_color, tmp_dir):
    """Replace black background with bg_color and pad to square."""
    from PIL import Image
    import numpy as np
    img = Image.open(pose_path).convert("RGB")
    arr = np.array(img)
    # Replace black pixels with bg color
    mask = (arr[:, :, 0] < 15) & (arr[:, :, 1] < 15) & (arr[:, :, 2] < 15)
    arr[mask] = bg_color
    img = Image.fromarray(arr)
    # Pad to square if needed
    w, h = img.size
    if w != h:
        size = max(w, h)
        square = Image.new("RGB", (size, size), bg_color)
        square.paste(img, ((size - w) // 2, (size - h) // 2))
        img = square
    out = os.path.join(tmp_dir, os.path.basename(pose_path))
    img.save(out)
    return out


async def cloud_upload(session, api_key, filepath):
    import aiohttp
    filepath = Path(filepath)
    data = aiohttp.FormData()
    data.add_field("image", open(filepath, "rb"),
                   filename=filepath.name, content_type="image/png")
    data.add_field("type", "input")
    data.add_field("overwrite", "true")
    async with session.post(f"{CLOUD_BASE}/api/upload/image",
                            headers={"X-API-Key": api_key}, data=data) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Upload failed ({resp.status}): {await resp.text()}")
        return (await resp.json())["name"]


async def cloud_submit(session, api_key, workflow):
    async with session.post(
        f"{CLOUD_BASE}/api/prompt",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"prompt": workflow},
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Submit failed ({resp.status}): {await resp.text()}")
        return (await resp.json())["prompt_id"]


LABEL_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/arialbd.ttf",
    "/mnt/c/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def add_label_below(output_path, label, band_pct=0.12, font_pct=0.50):
    """Extend the canvas below with a band sampled from the image edges and draw centered text."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        img = Image.open(output_path).convert("RGB")
        W, H = img.size
        arr = np.array(img)
        edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        bg = tuple(int(c) for c in edges.mean(axis=0))
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        text_color = (30, 30, 30) if lum > 128 else (235, 235, 235)
        band_h = max(1, int(H * band_pct))
        font_px = max(8, int(band_h * font_pct))
        font = None
        for fp in LABEL_FONT_CANDIDATES:
            if os.path.isfile(fp):
                font = ImageFont.truetype(fp, font_px)
                break
        if font is None:
            font = ImageFont.load_default()
        canvas = Image.new("RGB", (W, H + band_h), bg)
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (W - tw) // 2 - bbox[0]
        y = H + (band_h - th) // 2 - bbox[1]
        draw.text((x, y), label, fill=text_color, font=font)
        canvas.save(output_path)
    except Exception as e:
        print(f"  (label failed: {e})")


def cleanup_checkerboard(output_path, ref_image_path):
    """Replace checkerboard artifacts near the background color with solid bg."""
    try:
        from PIL import Image
        import numpy as np
        ref = Image.open(ref_image_path).convert("RGB")
        arr = np.array(ref)
        edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        bg = edges.mean(axis=0).astype(int)

        img = Image.open(output_path).convert("RGB")
        a = np.array(img)
        diff = np.abs(a.astype(int) - bg.astype(int)).sum(axis=2)
        a[diff < 60] = bg
        Image.fromarray(a).save(output_path)
    except Exception:
        pass  # non-critical, skip if it fails


def flatten_to_white_bg(output_path, white_thresh=250):
    """Snap the already-near-white backdrop to a clean pure #FFFFFF.

    The render already targets a white background via the prompt (and the base/hero
    are Flux Kontext renders that come out clean white — corners ~248-254). We only
    snap pixels that are *already* near-pure-white in every channel (>= white_thresh)
    up to 255, so the solid backdrop becomes perfectly white while the SUBJECT is left
    alone: a white shirt (mid-200s with shading) stays, and soft anti-aliased hair
    edges (blends below the threshold) stay soft.

    This deliberately replaces the previous connected-component flood-fill, which
    classified any light, low-saturation region as background and bled into light
    garments — producing bright blown-out patches on white sleeves — and hardened
    hair edges. A plain per-pixel near-white clamp cannot do either.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(output_path).convert("RGB")
        a = np.array(img)
        near_white = a.min(axis=2) >= white_thresh
        if not near_white.any():
            return
        a[near_white] = (255, 255, 255)
        Image.fromarray(a).save(output_path)
    except Exception:
        pass  # non-critical, skip if it fails


async def cloud_download(session, api_key, filename, output_path):
    """Download an output file via /api/view (follows redirect to storage)."""
    params = urllib.parse.urlencode({"filename": filename, "subfolder": "", "type": "output"})
    url = f"{CLOUD_BASE}/api/view?{params}"
    async with session.get(url, headers={"X-API-Key": api_key}, allow_redirects=False) as resp:
        if resp.status in (301, 302, 303, 307, 308):
            # Follow redirect without auth header (storage URL doesn't need it)
            redirect_url = resp.headers["Location"]
            async with session.get(redirect_url) as dl_resp:
                if dl_resp.status != 200:
                    raise RuntimeError(f"Download failed ({dl_resp.status}) from redirect")
                with open(output_path, "wb") as f:
                    f.write(await dl_resp.read())
        elif resp.status == 200:
            with open(output_path, "wb") as f:
                f.write(await resp.read())
        else:
            raise RuntimeError(f"Download failed ({resp.status}): {await resp.text()}")


async def cloud_queue_empty(session, api_key):
    """True if the cloud queue has nothing pending or running — i.e. our jobs have
    already left the cloud. Used to detect the cache-hang (a cached render returns
    `execution_cached` with no `executed` websocket message, so the client would
    otherwise block on the socket until the timeout)."""
    try:
        async with session.get(f"{CLOUD_BASE}/api/queue",
                               headers={"X-API-Key": api_key}) as r:
            if r.status != 200:
                return False
            q = await r.json()
        return not q.get("queue_pending") and not q.get("queue_running")
    except Exception:
        return False


async def cloud_history_outputs(session, api_key):
    """Return {prompt_id: outputs} for completed jobs from /api/history_v2, so a
    render whose websocket `executed` message was never delivered (cache-hang) can
    still be fetched by prompt_id."""
    try:
        async with session.get(f"{CLOUD_BASE}/api/history_v2",
                               headers={"X-API-Key": api_key}) as r:
            if r.status != 200:
                return {}
            data = await r.json()
    except Exception:
        return {}
    out = {}
    for entry in data.get("history", []):
        pid = entry.get("prompt_id")
        if pid and entry.get("status", {}).get("completed"):
            out[pid] = entry.get("outputs", {})
    return out


async def _ws_collect_outputs(api_key, prompt_ids, timeout=600):
    """
    Connect to Comfy Cloud websocket and collect output filenames for given prompt_ids.
    Returns dict: {prompt_id: [{"filename": ..., "subfolder": ..., "type": ...}, ...]}
    """
    import aiohttp
    results = {}
    pending = set(prompt_ids)
    ws_url = f"wss://cloud.comfy.org/ws?token={api_key}"

    async with aiohttp.ClientSession() as ws_session:
        async with ws_session.ws_connect(ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=10)) as ws:
            start = time.time()
            while pending and (time.time() - start) < timeout:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                if msg.type in (aiohttp.WSMsgType.TEXT,):
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    msg_type = data.get("type", "")
                    msg_data = data.get("data", {})
                    pid = msg_data.get("prompt_id", "")
                    if msg_type == "executed" and pid in pending:
                        # executed message contains output info
                        node_output = msg_data.get("output") or {}
                        images = node_output.get("images", [])
                        if images:
                            results[pid] = images
                            pending.discard(pid)
                    elif msg_type == "execution_error" and pid in pending:
                        results[pid] = []
                        pending.discard(pid)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
    return results


async def _process_batch_ws(jobs, args, api_key, session):
    """Submit all jobs, listen on websocket for outputs, download results."""
    import aiohttp
    total = len(jobs)
    sem = asyncio.Semaphore(args.concurrency)
    ok, fail = 0, 0

    # Upload image(s)
    print("Uploading image to Comfy Cloud...")
    try:
        img_name = await cloud_upload(session, api_key, args.image)
    except Exception as e:
        sys.exit(f"ERROR: Upload failed — {e}")
    print(f"  Uploaded as: {img_name}\n")

    # For AnyPose, pre-process and upload all pose images
    pose_img_names = {}  # local_path -> cloud_name
    if args.pipeline == "anypose":
        pose_paths = sorted(set(j[5] for j in jobs))  # jobs[5] = pose_path
        bg_color = get_reference_bg_color(args.image)
        print(f"  Reference bg color: RGB{bg_color}")
        tmp_dir = tempfile.mkdtemp(prefix="anypose_")
        print(f"Uploading {len(pose_paths)} pose images (bg-matched)...")
        for pp in pose_paths:
            try:
                prepped = prep_pose_image(pp, bg_color, tmp_dir)
                pname = await cloud_upload(session, api_key, prepped)
                pose_img_names[pp] = pname
                print(f"  ✓ {os.path.basename(pp)} → {pname}")
            except Exception as e:
                print(f"  ✗ {os.path.basename(pp)} — {e}")
        print()

    pid_to_job = {}  # prompt_id -> (idx, fname, out_path, prompt)
    pid_got_render = set()   # prompt_ids that have received the render image
    pid_got_skeleton = set() # prompt_ids that have received the skeleton image
    pid_got_json = set()     # prompt_ids that have received the pose JSON
    pending = set()
    ws_url = f"wss://cloud.comfy.org/ws?token={api_key}"

    def _check_job_complete(pid):
        """Mark a job as done once all expected outputs have been received."""
        if pid not in pending:
            return
        if pid not in pid_got_render:
            return
        if getattr(args, 'get_pose', False):
            if pid not in pid_got_skeleton or pid not in pid_got_json:
                return
        pending.discard(pid)

    # Connect websocket FIRST, then submit jobs so we don't miss early completions
    print("  Connecting websocket...")
    async with session.ws_connect(ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=10),
                                  heartbeat=30) as ws:
        # Wait for initial status message
        try:
            await asyncio.wait_for(ws.receive(), timeout=5)
        except asyncio.TimeoutError:
            pass
        print("  Websocket connected.\n")

        # Submit jobs with concurrency limit
        async def submit_one(idx, job):
            az, el, dist, prompt, fname = job[:5]
            out = os.path.join(args.output, fname)
            if os.path.exists(out) and not args.force:
                print(f"  [{idx:3d}/{total}] SKIP  {fname}")
                return None
            async with sem:
                print(f"  [{idx:3d}/{total}] → {prompt[:80]}")
                pose_cloud_name = None
                if args.pipeline == "anypose":
                    pose_path = job[5]
                    pose_cloud_name = pose_img_names.get(pose_path)
                    if not pose_cloud_name:
                        print(f"  [{idx:3d}/{total}] ✗ {fname}  (pose image not uploaded)")
                        return None
                # Use descriptive filename (without .png) as the cloud prefix
                cloud_prefix = os.path.splitext(fname)[0]
                wf = build_workflow(img_name, az, el, dist, prompt,
                                    args.seed, args.steps,
                                    args.guidance, args.lora_lightning, args.lora_angles,
                                    filename_prefix=cloud_prefix,
                                    pipeline=args.pipeline,
                                    pose_image_filename=pose_cloud_name,
                                    get_pose=getattr(args, 'get_pose', False),
                                    qwen_version=args.qwen_version,
                                    engine=args.engine,
                                    reinforce=args.prompt_append,
                                    megapixels=args.megapixels)
                pid = await cloud_submit(session, api_key, wf)
                pid_to_job[pid] = (idx, fname, out, prompt)
                pending.add(pid)
                return pid

        # Start submitting in background while we listen on websocket
        async def submit_all():
            submit_tasks = []
            for i, job in enumerate(jobs, 1):
                submit_tasks.append(submit_one(i, job))
            results = await asyncio.gather(*submit_tasks)
            active = [p for p in results if p is not None]
            print(f"\n  All {len(active)} jobs submitted.\n", flush=True)
            return active

        submit_task = asyncio.create_task(submit_all())

        # Cache-hang recovery: a cached render returns `execution_cached` with no
        # `executed` websocket message, so the pid would sit in `pending` until the
        # full timeout. When the websocket goes quiet, if the cloud queue is empty
        # yet jobs are still pending, fetch their finished renders from history.
        async def recover_stalled():
            nonlocal ok, fail
            if getattr(args, 'get_pose', False):
                return  # multi-output pose mode: leave to normal ws handling
            if not await cloud_queue_empty(session, api_key):
                return  # jobs are still genuinely running
            hist = await cloud_history_outputs(session, api_key)
            save_node = "4" if args.engine == "nanobanana" else PIPELINE_SAVE_NODE.get(args.pipeline, "2")
            for pid in list(pending):
                outputs = hist.get(pid)
                if not outputs:
                    continue
                images = (outputs.get(save_node) or {}).get("images", [])
                idx, fname, out, prompt = pid_to_job[pid]
                if images:
                    try:
                        await cloud_download(session, api_key, images[0]["filename"], out)
                        cleanup_checkerboard(out, args.image)
                        flatten_to_white_bg(out)
                        if args.label:
                            add_label_below(out, args.label)
                        print(f"  [{idx:3d}/{total}] ✓ {fname}  (recovered from cache)")
                        ok += 1
                    except Exception as e:
                        print(f"  [{idx:3d}/{total}] ✗ {fname}  (cache-recover download: {e})")
                        fail += 1
                    pending.discard(pid)

        # Listen on websocket concurrently with submission
        skipped = 0
        start = time.time()
        last_msg = time.time()
        CACHE_STALL_SECS = 15  # quiet-socket grace before probing for a cache-hang

        # Wait until at least one job is submitted or all are done
        while not pending and not submit_task.done():
            await asyncio.sleep(0.1)

        while (time.time() - start) < args.timeout:
            # Check if all done
            if submit_task.done() and not pending:
                break

            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
            except asyncio.TimeoutError:
                # No websocket traffic. If submission is done and jobs are still
                # pending after a quiet stretch, we're likely cache-hung — recover
                # the finished renders from history rather than blocking to timeout.
                if submit_task.done() and pending and (time.time() - last_msg) > CACHE_STALL_SECS:
                    await recover_stalled()
                    last_msg = time.time()  # reset so we probe at most once per interval
                continue
            last_msg = time.time()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                msg_type = data.get("type", "")
                msg_data = data.get("data", {})
                pid = msg_data.get("prompt_id", "")

                # ── Handle job outputs ──
                if pid in pending:
                    if msg_type == "executed":
                        node_id = str(msg_data.get("node", ""))
                        node_output = msg_data.get("output") or {}
                        images = node_output.get("images", [])

                        idx, fname, out, prompt = pid_to_job[pid]

                        # Main render image (from SaveImage node "2" or "31")
                        save_node = "4" if args.engine == "nanobanana" else PIPELINE_SAVE_NODE.get(args.pipeline, "2")
                        if node_id == save_node and images:
                            try:
                                await cloud_download(session, api_key,
                                                     images[0]["filename"], out)
                                cleanup_checkerboard(out, args.image)
                                flatten_to_white_bg(out)
                                if args.label:
                                    add_label_below(out, args.label)
                                print(f"  [{idx:3d}/{total}] ✓ {fname}")
                                print(f"           prompt: {prompt}")
                                ok += 1
                                pid_got_render.add(pid)
                            except Exception as e:
                                print(f"  [{idx:3d}/{total}] ✗ {fname}  (download: {e})")
                                fail += 1
                                pending.discard(pid)
                            _check_job_complete(pid)

                        # Pose skeleton image (from dwpose:save_img)
                        elif node_id == "dwpose:save_img" and images:
                            pose_dir = os.path.join(args.output, "poses")
                            os.makedirs(pose_dir, exist_ok=True)
                            pose_base = os.path.splitext(fname)[0]
                            skel_out = os.path.join(pose_dir, f"{pose_base}_skeleton.png")
                            try:
                                await cloud_download(session, api_key,
                                                     images[0]["filename"], skel_out)
                                print(f"  [{idx:3d}/{total}] 🦴 ✓ skeleton")
                            except Exception as e:
                                print(f"  [{idx:3d}/{total}] 🦴 ✗ skeleton: {e}")
                            pid_got_skeleton.add(pid)
                            _check_job_complete(pid)

                        # Pose JSON data (from dwpose:1 DWPreprocessor ui output)
                        elif node_id == "dwpose:1":
                            openpose_json = node_output.get("openpose_json", [])
                            if openpose_json:
                                pose_dir = os.path.join(args.output, "poses")
                                os.makedirs(pose_dir, exist_ok=True)
                                pose_base = os.path.splitext(fname)[0]
                                json_out = os.path.join(pose_dir, f"{pose_base}_pose.json")
                                try:
                                    json_str = openpose_json[0] if isinstance(openpose_json[0], str) else json.dumps(openpose_json[0])
                                    with open(json_out, "w") as jf:
                                        jf.write(json_str)
                                    print(f"  [{idx:3d}/{total}] 🦴 ✓ pose JSON")
                                except Exception as e:
                                    print(f"  [{idx:3d}/{total}] 🦴 ✗ pose JSON: {e}")
                            pid_got_json.add(pid)
                            _check_job_complete(pid)

                    elif msg_type == "execution_complete":
                        if pid in pending:
                            pending.discard(pid)

                    elif msg_type == "execution_error":
                        idx, fname, out, prompt = pid_to_job[pid]
                        err = msg_data.get("exception_message", "unknown error")
                        print(f"  [{idx:3d}/{total}] ✗ {fname}  ({err})")
                        fail += 1
                        pending.discard(pid)

            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                print("  WebSocket closed unexpectedly")
                break

        # Ensure submit task is done
        if not submit_task.done():
            await submit_task
        active_pids = submit_task.result()
        skipped = total - len(active_pids)

    # Anything still pending timed out
    for pid in pending:
        idx, fname, out, prompt = pid_to_job[pid]
        print(f"  [{idx:3d}/{total}] ✗ {fname}  (timeout)")
        fail += 1
    return ok + skipped, fail


async def _run_cloud(jobs, args):
    import aiohttp
    api_key = os.environ.get("COMFY_CLOUD_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: Set COMFY_CLOUD_API_KEY env var.\n  Get key at: https://platform.comfy.org/login")

    print(f"  Mode: CLOUD → {CLOUD_BASE}")
    print(f"  Concurrency: {args.concurrency} parallel jobs\n")

    # Comfy Cloud returns Brotli-encoded bodies that aiohttp's streaming
    # decompressor chokes on ("Can not decode content-encoding: br"), so ask
    # the server for gzip/deflate instead — which aiohttp decodes cleanly.
    async with aiohttp.ClientSession(
        headers={"Accept-Encoding": "gzip, deflate"}
    ) as session:
        return await _process_batch_ws(jobs, args, api_key, session)


def run_cloud(jobs, args):
    return asyncio.run(_run_cloud(jobs, args))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Batch-render multi-angle poses via ComfyUI")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output", default=None, help="Output directory (default: ./multi_angle_output_{pipeline}_seed{seed})")
    p.add_argument("--cloud", action="store_true",
                   help="Use Comfy Cloud (set COMFY_CLOUD_API_KEY env var)")
    p.add_argument("--server", default="http://127.0.0.1:8188",
                   help="Local ComfyUI URL (ignored with --cloud)")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Parallel cloud jobs (Free=1, Standard=1, Creator=3, Pro=5)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--megapixels", type=float, default=None,
                   help="Render target in megapixels for the qwen-edit pipelines "
                        "(e.g. 1.76 for Qwen-native 1328x1328); default uses Flux ~1 MP")
    p.add_argument("--lora-angles", type=float, default=1.0)
    p.add_argument("--lora-lightning", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--poses", default=None,
                   help="Subset for --pipeline poses_prompt: comma-separated pose names, "
                        "or 'base' for the 16 action poses (excludes tpose/apose rig variants)")
    p.add_argument("--expressions", default=None,
                   help="Subset for --pipeline expressions: comma-separated expression names")
    p.add_argument("--outfits", default=None,
                   help="Subset for --pipeline outfits: comma-separated outfit names")
    p.add_argument("--angles", default=None,
                   help="Subset for --pipeline angles_prompt: comma-separated angle names")
    p.add_argument("--azimuths", default=None, help="Subset, e.g. 0,90,180,270")
    p.add_argument("--elevations", default=None, help="Subset, e.g. -30,0,30,60")
    p.add_argument("--distances", default=None, help="Subset, e.g. 0.6,1.0,1.8")
    p.add_argument("--pipeline", default="2511",
                   choices=["2509", "2511", "anypose", "expressions", "lighting", "outfits", "poses_prompt", "angles_prompt", "standing_relaxed", "kontext_base", "kontext_hero"],
                   help="Model pipeline: 2509, 2511 (default), anypose, expressions, lighting, outfits, poses_prompt, angles_prompt, standing_relaxed, "
                        "kontext_base (clean matte neutral base via Flux Kontext), or kontext_hero (clean matte original-pose re-render)")
    p.add_argument("--pose-dir", default=None,
                   help="Directory of pose images (required for --pipeline anypose)")
    p.add_argument("--prompt-append", default="",
                   help="String to append to every generated prompt")
    p.add_argument("--label", default=None,
                   help="If set, composite this text as a label band below every rendered image")
    p.add_argument("--engine", default="qwen", choices=["qwen", "nanobanana"],
                   help="Image-edit backend. 'qwen' (default) uses Qwen Image Edit. "
                        "'nanobanana' routes the request through GeminiImage2Node "
                        "(Nano Banana Pro / gemini-3-pro-image-preview) — only supported "
                        "for --pipeline standing_relaxed at the moment.")
    p.add_argument("--qwen-version", default="2511", choices=["2511", "2509"],
                   help="Qwen Image Edit base model version for text-prompt pipelines "
                        "(expressions/lighting/outfits/poses_prompt/angles_prompt/standing_relaxed). "
                        "Does not affect the 2511/2509 multi-angle or anypose pipelines, which have fixed models.")
    p.add_argument("--get-pose", action="store_true",
                   help="Run DWPose extraction on each rendered image (saves skeleton + JSON to poses/ subdir)")
    p.add_argument("--force", action="store_true",
                   help="Re-render even if the output file already exists (default: skip existing)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output is None:
        args.output = f"./multi_angle_output_{args.pipeline}_seed{args.seed}"

    # ── Parse filters ─────────────────────────────────────────────────
    azimuths = ALL_AZIMUTHS
    if args.azimuths:
        azimuths = [int(x) for x in args.azimuths.split(",")]
        for a in azimuths:
            assert a in AZIMUTH_MAP, f"Bad azimuth {a}. Valid: {list(AZIMUTH_MAP)}"

    elevations = ALL_ELEVATIONS
    if args.elevations:
        elevations = [int(x) for x in args.elevations.split(",")]
        for e in elevations:
            assert e in ELEVATION_MAP, f"Bad elevation {e}. Valid: {list(ELEVATION_MAP)}"

    distances = ALL_DISTANCES
    if args.distances:
        distances = [float(x) for x in args.distances.split(",")]
        for d in distances:
            assert d in DISTANCE_MAP, f"Bad distance {d}. Valid: {list(DISTANCE_MAP)}"

    suffix = f" {args.prompt_append}" if args.prompt_append else ""

    if args.pipeline == "angles_prompt":
        angle_items = list(ANGLES_PROMPT.items())
        if args.angles:
            wanted = [x.strip() for x in args.angles.split(",")]
            for w in wanted:
                assert w in ANGLES_PROMPT, f"Bad angle name '{w}'. Valid: {list(ANGLES_PROMPT)}"
            angle_items = [(w, ANGLES_PROMPT[w]) for w in wanted]
        jobs = [
            (None, None, None,
             ANGLES_PROMPT_PREFIX + desc + suffix,
             f"angle_{name}.png")
            for name, desc in angle_items
        ]
    elif args.pipeline == "poses_prompt":
        pose_items = list(POSES.items())
        if args.poses:
            if args.poses.strip().lower() == "base":
                pose_items = [(n, d) for n, d in POSES.items()
                              if not (n.startswith("tpose") or n.startswith("apose"))]
            else:
                wanted = [x.strip() for x in args.poses.split(",")]
                for w in wanted:
                    assert w in POSES, f"Bad pose name '{w}'. Valid: {list(POSES)}"
                pose_items = [(w, POSES[w]) for w in wanted]
        jobs = [
            (None, None, None,
             POSES_PREFIX + desc + suffix,
             f"pose_{name}.png")
            for name, desc in pose_items
        ]
    elif args.pipeline == "standing_relaxed":
        srp = STANDING_RELAXED_PROMPT_NANOBANANA if args.engine == "nanobanana" else STANDING_RELAXED_PROMPT
        jobs = [(None, None, None,
                 srp + suffix,
                 "standing_relaxed.png")]
    elif args.pipeline == "kontext_base":
        # Clean matte neutral base (Flux Kontext); output name matches standing_relaxed
        # so the existing prep/derive steps work unchanged.
        jobs = [(None, None, None,
                 KONTEXT_BASE_PROMPT + suffix,
                 "standing_relaxed.png")]
    elif args.pipeline == "kontext_hero":
        # Clean matte re-render of the ORIGINAL hero pose (Flux Kontext).
        jobs = [(None, None, None,
                 KONTEXT_HERO_PROMPT + suffix,
                 "hero.png")]
    elif args.pipeline == "expressions":
        expr_items = list(EXPRESSIONS.items())
        if args.expressions:
            wanted = [x.strip() for x in args.expressions.split(",")]
            for w in wanted:
                assert w in EXPRESSIONS, f"Bad expression name '{w}'. Valid: {list(EXPRESSIONS)}"
            expr_items = [(w, EXPRESSIONS[w]) for w in wanted]
        jobs = [
            (None, None, None,
             EXPRESSION_PREFIX + desc + suffix,
             f"expr_{name}.png")
            for name, desc in expr_items
        ]
    elif args.pipeline == "lighting":
        jobs = [
            (None, None, None,
             LIGHTING_PREFIX + desc + suffix,
             f"light_{name}.png")
            for name, desc in LIGHTING.items()
        ]
    elif args.pipeline == "outfits":
        outfit_items = list(OUTFITS.items())
        if args.outfits:
            wanted = [x.strip() for x in args.outfits.split(",")]
            for w in wanted:
                assert w in OUTFITS, f"Bad outfit name '{w}'. Valid: {list(OUTFITS)}"
            outfit_items = [(w, OUTFITS[w]) for w in wanted]
        jobs = [
            (None, None, None,
             OUTFITS_PREFIX + desc + suffix,
             f"outfit_{name}.png")
            for name, desc in outfit_items
        ]
    elif args.pipeline == "anypose":
        # AnyPose: iterate over pose images from a directory
        if not args.pose_dir:
            sys.exit("ERROR: --pose-dir is required for --pipeline anypose")
        assert os.path.isdir(args.pose_dir), f"Pose directory not found: {args.pose_dir}"
        pose_files = sorted([
            f for f in os.listdir(args.pose_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        assert pose_files, f"No image files found in {args.pose_dir}"
        prompt = ANYPOSE_DEFAULT_PROMPT + suffix
        # jobs tuple: (az, el, dist, prompt, fname, pose_path)
        jobs = [
            (None, None, None, prompt,
             f"pose_{os.path.splitext(pf)[0]}.png",
             os.path.join(args.pose_dir, pf))
            for pf in pose_files
        ]
    else:
        prompt_fn = build_prompt_2509 if args.pipeline == "2509" else build_prompt_2511
        # For 2509, skip duplicate elevations (30° and 60° produce the same prompt)
        if args.pipeline == "2509" and not args.elevations:
            elevations = [e for e in elevations if e != 60]
        jobs = [(az, el, dist, prompt_fn(az, el, dist) + suffix, safe_filename(az, el, dist))
                for az in azimuths for el in elevations for dist in distances]

    total = len(jobs)
    mode = "CLOUD" if args.cloud else "LOCAL"

    print(f"\n{'='*64}")
    if args.pipeline == "anypose":
        print(f"  Qwen Image Edit — AnyPose Batch Renderer")
    elif args.pipeline == "angles_prompt":
        print(f"  Qwen Image Edit — Prompt Angle Batch Renderer")
    elif args.pipeline == "poses_prompt":
        print(f"  Qwen Image Edit — Prompt Pose Batch Renderer")
    elif args.pipeline == "expressions":
        print(f"  Qwen Image Edit — Expression Batch Renderer")
    elif args.pipeline == "lighting":
        print(f"  Qwen Image Edit — Lighting Variation Renderer")
    elif args.pipeline == "outfits":
        print(f"  Qwen Image Edit — Outfit Variation Renderer")
    elif args.pipeline == "standing_relaxed":
        engine_label = "Nano Banana Pro" if args.engine == "nanobanana" else "Qwen Image Edit"
        print(f"  {engine_label} — Standing Relaxed 3/4 Pose Renderer")
    else:
        print(f"  Qwen Image Edit {args.pipeline} — Multi-Angle Batch Renderer")
    print(f"{'='*64}")
    print(f"  Input   : {args.image}")
    if args.pipeline == "anypose":
        print(f"  Pose Dir: {args.pose_dir}  ({total} poses)")
    elif args.pipeline == "angles_prompt":
        print(f"  Angles: {total}")
    elif args.pipeline == "poses_prompt":
        print(f"  Poses: {total}")
    elif args.pipeline == "expressions":
        print(f"  Expressions: {total}")
    elif args.pipeline == "lighting":
        print(f"  Lighting: {total} variations")
    elif args.pipeline == "outfits":
        print(f"  Outfits: {total} variations")
    print(f"  Output  : {args.output}")
    if args.pipeline not in ("anypose", "expressions", "lighting", "outfits", "poses_prompt", "angles_prompt", "standing_relaxed"):
        print(f"  Poses   : {total}  ({len(azimuths)} az × {len(elevations)} el × {len(distances)} dist)")
    print(f"  Steps   : {args.steps}  |  CFG: {args.guidance}  |  Seed: {args.seed}")
    print(f"  Pipeline: {args.pipeline}")
    print(f"  Target  : {mode}" + (f"  (concurrency: {args.concurrency})" if args.cloud else ""))
    print(f"{'='*64}\n")

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            prompt = job[3]
            fname = job[4]
            print(f"  [{i:3d}/{total}] {prompt[:80]}")
            print(f"           → {fname}\n")
        print(f"Total: {total} images. Remove --dry-run to execute.")
        return

    assert os.path.isfile(args.image), f"Image not found: {args.image}"
    os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    ok, fail = (run_cloud if args.cloud else run_local)(jobs, args)
    elapsed = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  COMPLETE  —  {ok}/{total} rendered, {fail} failed, {elapsed:.0f}s total")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
