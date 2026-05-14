"""Cache-time augmentation.

Applied at *cache creation* (not at every training step) so the augmented
features go through InternVL3 once and the training loop only pays the
disk-read cost.

  1. Visual augmentation: photometric jitter (brightness/contrast/saturation/
     hue) + Gaussian blur + small crop-and-resize, applied identically to all
     frames in a chunk so temporal coherence is preserved. Each augmented
     variant is a different draw of the random transform parameters.

  2. Prompt sampling: each augmented variant draws a random paraphrase from a
     static `PARAPHRASE_BANK` so InternVL3 sees prompt diversity without any
     external API dependency.
"""
from __future__ import annotations

import random
from typing import List, Sequence

from PIL import Image, ImageFilter, ImageEnhance


# ════════════════════════════════════════════════════════════
#  Visual augmentation
# ════════════════════════════════════════════════════════════

def _sample_visual_params(rng: random.Random,
                          brightness=(0.85, 1.15),
                          contrast=(0.85, 1.15),
                          saturation=(0.85, 1.15),
                          hue=(-0.04, 0.04),
                          blur_sigma=(0.0, 0.6),
                          crop_keep=(0.92, 1.0)):
    """Sample one set of transform parameters (used identically across the
    frames of a single chunk → temporally consistent augmentation).
    """
    return {
        'brightness': rng.uniform(*brightness),
        'contrast':   rng.uniform(*contrast),
        'saturation': rng.uniform(*saturation),
        'hue':        rng.uniform(*hue),
        'blur':       rng.uniform(*blur_sigma),
        'crop_keep':  rng.uniform(*crop_keep),
    }


def _apply_visual_params(img: Image.Image, params: dict) -> Image.Image:
    """Deterministic transform given a sampled parameter set."""
    img = ImageEnhance.Brightness(img).enhance(params['brightness'])
    img = ImageEnhance.Contrast(img).enhance(params['contrast'])
    img = ImageEnhance.Color(img).enhance(params['saturation'])
    if params.get('hue', 0.0):
        hsv = img.convert('HSV').split()
        h = hsv[0].point(lambda v: int((v + params['hue'] * 255)) % 256)
        img = Image.merge('HSV', (h, hsv[1], hsv[2])).convert('RGB')
    if params['blur'] > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=params['blur']))
    keep = params['crop_keep']
    if keep < 0.999:
        w, h = img.size
        cw, ch = int(w * keep), int(h * keep)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)
    return img


def visual_augment_chunk(pil_frames: Sequence[Image.Image],
                         seed: int) -> List[Image.Image]:
    """Apply one consistent random transform to every frame of a chunk.
    Returns a fresh list of PIL images. The caller picks `seed` so that
    different (chunk, aug_idx) pairs sample different transforms but all
    frames within one variant get the same one.
    """
    rng = random.Random(seed)
    params = _sample_visual_params(rng)
    return [_apply_visual_params(f, params) for f in pil_frames]


# ════════════════════════════════════════════════════════════
#  Prompt sampling (static bank — no external API dependency)
# ════════════════════════════════════════════════════════════

PARAPHRASE_BANK: dict[str, list[str]] = {
    "pick and place": [
        "Pick up the object and place it at the target location.",
        "Grasp the item and move it to the designated spot.",
        "Use the robot arm to pick and place the object.",
        "Reach for the object, grasp it, and set it down at the goal position.",
        "Grab the object and put it in the target area.",
        "Pick up the item, then set it down at the goal.",
        "Lift the object and transfer it to the placement zone.",
        "Collect the object and deposit it at the destination.",
        "Seize the item and relocate it to the marked position.",
        "The robot arm picks up an object and places it at the target.",
        "Perform a pick-and-place maneuver with the robotic arm.",
        "Execute a grasping and placing task on the workspace.",
        "Move to the object, close the gripper, lift, transport, and release.",
        "Approach the item, grip it firmly, carry it over, and set it down.",
        "Extend the arm toward the object, pick it up, and place it.",
        "Pick and place the object.",
        "Grasp and move the item.",
        "Pick, move, place.",
        "Object manipulation: pick and place.",
        "Complete the pick-and-place task on the tabletop.",
        "Using the SO-101 arm, transfer the object to the target position.",
        "Manipulate the object from its current location to the goal.",
        "Perform the demonstrated grasping and placement action.",
        "Pick and Place SO101 Arm.",
        "The robot picks up an object and deposits it at the target.",
    ],
    "red cube to bowl": [
        "Grasp the red cube and place it in the gray bowl.",
        "Pick up the red block and put it into the gray bowl.",
        "Grab the red cube and drop it in the bowl.",
        "Take the red cube and move it to the gray bowl.",
        "Lift the red block, carry it to the bowl, and release it inside.",
        "Transfer the red cube into the gray container.",
        "Place the red cube inside the gray bowl on the table.",
        "Pick the red block up and set it in the bowl.",
        "Grasp the red object and deposit it in the gray dish.",
        "Move the red cube from the table into the gray bowl.",
        "The robot picks up the red cube and places it in the bowl.",
        "Perform pick-and-place: red cube to gray bowl.",
        "Red cube into gray bowl.",
        "Grab red block, place in bowl.",
        "Transfer the block to the bowl.",
    ],
}


def _bank_key_for(base_prompt: str) -> str:
    p = base_prompt.lower()
    if 'red' in p and ('bowl' in p or 'cube' in p or 'block' in p):
        return "red cube to bowl"
    return "pick and place"


def sample_paraphrases(base_prompt: str, n: int = 20,
                       seed: int = 0) -> list[str]:
    """Return `n` paraphrases of `base_prompt` from the static bank.
    Samples with replacement once the bank is exhausted.
    """
    bank = PARAPHRASE_BANK[_bank_key_for(base_prompt)]
    rng = random.Random(seed)
    out = list(bank)
    rng.shuffle(out)
    while len(out) < n:
        out.append(rng.choice(bank))
    return out[:n]


def build_paraphrase_pool(base_prompts: Sequence[str],
                          n: int = 20) -> dict[str, list[str]]:
    """{base_prompt: [paraphrase, ...]} pool, one entry per unique base prompt.
    The pool is what cache_vision samples from per chunk-variant.
    """
    pool: dict[str, list[str]] = {}
    for bp in base_prompts:
        if bp not in pool:
            pool[bp] = sample_paraphrases(bp, n=n)
    return pool
