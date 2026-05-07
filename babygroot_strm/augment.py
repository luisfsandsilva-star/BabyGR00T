"""Visual and prompt augmentation for vision-cache creation.

Two complementary forms of augmentation, applied at *cache time* (not at every
training step) so the augmented features go through InternVL3 once and the
training loop only pays the disk-read cost.

  1. Visual augmentation: photometric jitter (brightness/contrast/saturation/hue)
     + Gaussian blur + small crop-and-resize, applied identically to all frames
     in a chunk so the temporal coherence within the chunk is preserved. Each
     augmented variant is a different draw of the random transform parameters.

  2. Prompt augmentation: each augmented variant gets a paraphrased version of
     the task prompt sampled from a paraphrase bank. The bank can be
     pre-generated (`paraphrase_prompts_fallback`) or generated at cache time
     by calling the Anthropic API (`paraphrase_prompts_with_llm`) if
     ANTHROPIC_API_KEY is set. The LLM call is fire-once-per-prompt so total
     API cost is O(unique_prompts * n_paraphrases), not O(chunks).
"""
from __future__ import annotations

import os
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
    # Photometric (PIL ImageEnhance — fast, no extra deps)
    img = ImageEnhance.Brightness(img).enhance(params['brightness'])
    img = ImageEnhance.Contrast(img).enhance(params['contrast'])
    img = ImageEnhance.Color(img).enhance(params['saturation'])
    if params.get('hue', 0.0):
        # Convert hue shift to a per-channel rotation in HSV space.
        hsv = img.convert('HSV').split()
        h = hsv[0].point(lambda v: int((v + params['hue'] * 255)) % 256)
        img = Image.merge('HSV', (h, hsv[1], hsv[2])).convert('RGB')
    # Gaussian blur
    if params['blur'] > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=params['blur']))
    # Centered crop-and-resize (keeps the same final size, mimics small zoom)
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
#  Prompt augmentation
# ════════════════════════════════════════════════════════════

# Pre-generated paraphrase bank — used as the fallback when the LLM API is
# not available. Covers the common SO-101 / pick-and-place phrasings.
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


def paraphrase_prompts_fallback(base_prompt: str, n: int = 20,
                                seed: int = 0) -> list[str]:
    """Static paraphrase bank, no API required. Always returns a list of length
    n by sampling with replacement once the bank is exhausted.
    """
    bank = PARAPHRASE_BANK[_bank_key_for(base_prompt)]
    rng = random.Random(seed)
    out = list(bank)
    rng.shuffle(out)
    while len(out) < n:
        out.append(rng.choice(bank))
    return out[:n]


def paraphrase_prompts_with_llm(base_prompt: str, n: int = 20,
                                model: str = "claude-haiku-4-5") -> list[str]:
    """Call the Anthropic API to generate `n` paraphrases of `base_prompt`.

    Requires `anthropic` installed and `ANTHROPIC_API_KEY` set. Falls back to
    the static bank on any error so cache-creation never aborts because of a
    transient API issue.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [prompt-aug] ANTHROPIC_API_KEY not set — using static bank.")
        return paraphrase_prompts_fallback(base_prompt, n)

    try:
        import anthropic
    except ImportError:
        print("  [prompt-aug] `anthropic` not installed — using static bank.")
        return paraphrase_prompts_fallback(base_prompt, n)

    sys_prompt = (
        "You generate diverse natural-language paraphrases of robot task "
        "instructions. Vary tone, length, and viewpoint (imperative / "
        "third-person / terse). Keep the meaning identical. "
        "Output ONLY the paraphrases, one per line, no numbering, no quotes, "
        "no explanations.")
    user_msg = (f"Generate {n} distinct paraphrases of this robot task "
                f"instruction:\n\n{base_prompt}")
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=1024, system=sys_prompt,
            messages=[{"role": "user", "content": user_msg}])
        text = "".join(block.text for block in msg.content
                       if block.type == "text")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Drop quotes / leading bullets if the model added them anyway.
        cleaned = [ln.lstrip("0123456789.-* '\"").rstrip(" '\"") for ln in lines]
        cleaned = [c for c in cleaned if c]
        if len(cleaned) >= max(3, n // 2):
            # Ensure exactly n by padding from the static bank if the model
            # under-delivered (rare with Haiku at n=20).
            if len(cleaned) < n:
                cleaned += paraphrase_prompts_fallback(
                    base_prompt, n - len(cleaned))
            return cleaned[:n]
        print(f"  [prompt-aug] LLM returned only {len(cleaned)} usable lines "
              f"(< {n // 2}); falling back to static bank.")
    except Exception as e:
        print(f"  [prompt-aug] LLM call failed ({type(e).__name__}: {e}); "
              f"falling back to static bank.")
    return paraphrase_prompts_fallback(base_prompt, n)


def build_paraphrase_pool(base_prompts: Sequence[str], n: int = 20,
                          use_llm: bool = False,
                          model: str = "claude-haiku-4-5") -> dict[str, list[str]]:
    """Build a {base_prompt: [paraphrase, ...]} pool, one entry per unique
    base prompt. The pool is what cache_vision samples from per chunk-variant.
    """
    pool: dict[str, list[str]] = {}
    for bp in base_prompts:
        if bp in pool:
            continue
        if use_llm:
            pool[bp] = paraphrase_prompts_with_llm(bp, n=n, model=model)
        else:
            pool[bp] = paraphrase_prompts_fallback(bp, n=n)
    return pool
