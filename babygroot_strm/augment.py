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


# ════════════════════════════════════════════════════════════
#  Template-based paraphraser for arbitrary natural-language tasks
#  (no external API; works on OXE / Bridge V2's ~20k unique strings)
# ════════════════════════════════════════════════════════════

# "Always-safe" templates: work for ANY task phrasing — imperative,
# declarative ("the robot picks up X"), passive, telegraphic, fragments, etc.
# These just wrap the raw task with a label, no grammatical assumption.
SAFE_TASK_TEMPLATES: list[str] = [
    "{x}.",
    "{X}.",                              # capitalised-first variant
    "Goal: {x}.",
    "Task: {x}.",
    "Instruction: {x}.",
    "Action: {x}.",
    "Step: {x}.",
    "Begin: {x}.",
    "Carry out the following: {x}.",
    "Complete the task: {x}.",
    "Execute the following: {x}.",
    "Now {x}.",                          # benign adverb prefix
]

# Imperative-required templates: assume `{x}` is a verb-phrase imperative
# ("pick up the cup", "fold the napkin"). Inserting these around a
# declarative subject (e.g. "the robot picks up the cup") would yield
# ungrammatical text like "The robot will the robot picks up the cup."
IMPERATIVE_TASK_TEMPLATES: list[str] = [
    "Please {x}.",
    "Robot, {x}.",
    "The robot will {x}.",
    "Use the robot arm to {x}.",
    "Your task is to {x}.",
    "Demonstrate how to {x}.",
    "Show how to {x}.",
    "The objective is to {x}.",
]

# Back-compat alias (older code may import this name).
TASK_PARAPHRASE_TEMPLATES = SAFE_TASK_TEMPLATES + IMPERATIVE_TASK_TEMPLATES


# Words that, as the first token of a task, signal a declarative / passive /
# non-imperative phrasing where imperative-template insertion would produce
# broken grammar. Lightweight first-word check — no POS tagger needed.
_NON_IMPERATIVE_STARTERS: frozenset = frozenset({
    'the', 'a', 'an',
    'i', 'he', 'she', 'it', 'they', 'you', 'we', 'this', 'that',
    'these', 'those', 'my', 'your', 'his', 'her', 'their', 'our',
    'there',
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'has', 'have', 'had',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
})


def is_likely_imperative(task: str) -> bool:
    """Lightweight heuristic. Returns True if the task looks like a verb-
    phrase imperative (the typical Bridge V2 form: "fold the napkin",
    "open the drawer"). Returns False for declarative / passive / pronoun-
    led phrasings where imperative-required templates would break grammar.

    Heuristic: first word must NOT be in _NON_IMPERATIVE_STARTERS, AND must
    not end in '-ing' (gerund) or '-ed' for common past-tense conjugations.
    """
    t = task.strip().lower()
    if not t:
        return False
    first = t.split(maxsplit=1)[0]
    if first in _NON_IMPERATIVE_STARTERS:
        return False
    # Gerund/progressive: "picking up the cup" → not imperative
    if first.endswith('ing') and len(first) > 4:
        return False
    # Past tense -ed: "picked up the cup" → not imperative.
    # Exclude common base-form verbs that happen to end in -ed (rare in robot
    # tasks but safer to whitelist a few).
    if (first.endswith('ed') and len(first) > 3
        and first not in {'feed', 'speed', 'need', 'seed', 'weed', 'shed',
                          'shred', 'fled', 'led', 'red', 'sled', 'wed',
                          'bed', 'bled', 'tread', 'spread', 'embed'}):
        return False
    return True


def _normalize_task_str(task: str) -> str:
    """Lowercase first letter and strip trailing punctuation so {x}/{X}
    interpolation works cleanly inside templates."""
    t = task.strip().rstrip(' .!?,;')
    if not t:
        return t
    return t[0].lower() + t[1:]


def template_paraphrases(task: str, n: int = 4, seed: int = 0) -> list[str]:
    """Generate `n` template-based paraphrases of any natural-language task
    string. No external API; works on OXE / Bridge V2's full task vocabulary.

    Robust against non-imperative phrasings: detects whether the task looks
    like an imperative verb phrase ("fold the napkin") vs a declarative
    ("the robot folds the napkin") and uses only safe templates for the
    latter — avoiding grammatically broken outputs like
    "Use the robot arm to the robot folds the napkin."
    """
    t_low = _normalize_task_str(task)
    t_hi = (t_low[:1].upper() + t_low[1:]) if t_low else t_low
    pool = list(SAFE_TASK_TEMPLATES)
    if is_likely_imperative(task):
        pool += IMPERATIVE_TASK_TEMPLATES
    rng = random.Random(seed ^ (hash(task) & 0xFFFFFFFF))
    rng.shuffle(pool)
    while len(pool) < n:
        pool.append(rng.choice(pool))
    return [t.format(x=t_low, X=t_hi) for t in pool[:n]]


def build_task_paraphrase_pool(base_prompts: Sequence[str],
                                n: int = 4) -> dict[str, list[str]]:
    """Like build_paraphrase_pool but uses template_paraphrases (programmatic)
    instead of the hand-curated SO-101-specific PARAPHRASE_BANK.  Suitable for
    OXE datasets with thousands of unique task descriptions."""
    pool: dict[str, list[str]] = {}
    for i, bp in enumerate(base_prompts):
        if bp not in pool:
            pool[bp] = template_paraphrases(bp, n=n, seed=i)
    return pool
